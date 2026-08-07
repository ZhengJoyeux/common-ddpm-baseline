"""D0-D3.2 一维 SERS DDPM 的训练、验证、EMA 和 checkpoint 管理。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.checkpoint_manager import (
    CheckpointManager,
    load_checkpoint_file,
)
from src.training_logger import TrainingLogger


class ExponentialMovingAverage:
    """维护扩散模型参数和缓冲区的指数移动平均。"""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        update_every: int,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay必须在0和1之间。")

        if update_every < 1:
            raise ValueError("EMA update_every必须至少为1。")

        self.decay = float(decay)
        self.update_every = int(update_every)
        self.number_of_updates = 0

        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.number_of_updates += 1

        if self.number_of_updates % self.update_every != 0:
            return

        for ema_parameter, model_parameter in zip(
            self.ema_model.parameters(),
            model.parameters(),
            strict=True,
        ):
            ema_parameter.lerp_(
                model_parameter.detach(),
                1.0 - self.decay,
            )

        for ema_buffer, model_buffer in zip(
            self.ema_model.buffers(),
            model.buffers(),
            strict=True,
        ):
            ema_buffer.copy_(model_buffer)

    def state_dict(self) -> dict:
        return {
            "ema_model": self.ema_model.state_dict(),
            "number_of_updates": self.number_of_updates,
            "decay": self.decay,
            "update_every": self.update_every,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state["ema_model"])
        self.number_of_updates = int(
            state.get("number_of_updates", 0)
        )


def _create_gradient_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class DdpmTrainer:
    """
    执行按 step 计数的一维 DDPM 训练与验证。

    D3.2 性能改动：训练分项损失先以 detached GPU tensor 累计，只在
    真正写日志或完成验证时一次性转换为 Python float，避免每个 step
    对十几个分项逐一调用 ``.item()`` 导致 GPU/CPU 强制同步。
    """

    def __init__(
        self,
        *,
        diffusion: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        configuration: dict,
        metadata: dict,
        checkpoint_manager: CheckpointManager,
        logger: TrainingLogger,
    ) -> None:
        self.diffusion = diffusion.to(device)
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.device = device
        self.configuration = configuration
        self.metadata = metadata
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

        training = configuration["training"]
        self.total_training_steps = int(
            training["total_training_steps"]
        )
        self.gradient_accumulation_steps = int(
            training["gradient_accumulation_steps"]
        )
        self.maximum_gradient_norm = float(
            training["maximum_gradient_norm"]
        )
        self.log_every_steps = int(training["log_every_steps"])
        self.validate_every_steps = int(
            training["validate_every_steps"]
        )
        self.checkpoint_every_steps = int(
            training["checkpoint_every_steps"]
        )
        self.maximum_validation_batches = int(
            training.get("maximum_validation_batches", 0)
        )

        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        self.use_mixed_precision = (
            bool(training["use_mixed_precision"])
            and device.type == "cuda"
        )
        self.gradient_scaler = _create_gradient_scaler(
            self.use_mixed_precision
        )
        self.ema = ExponentialMovingAverage(
            model=self.diffusion,
            decay=float(training["ema_decay"]),
            update_every=int(training["ema_update_every"]),
        )

        self.starting_step = 0
        self.best_validation_loss = float("inf")
        self._training_iterator: Iterable | None = None
        self.latest_validation_components: dict[str, float] | None = None

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_checkpoint_file(
            checkpoint_path,
            map_location=self.device,
        )
        self.diffusion.load_state_dict(checkpoint["diffusion_state"])
        ema_state = checkpoint.get("ema_state")

        if ema_state:
            self.ema.load_state_dict(ema_state)
        else:
            self.ema.ema_model.load_state_dict(
                checkpoint["diffusion_state"]
            )

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        scaler_state = checkpoint.get("scaler_state")

        if scaler_state:
            self.gradient_scaler.load_state_dict(scaler_state)

        self.starting_step = int(checkpoint.get("step", 0))
        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        print(
            f"从step={self.starting_step}继续训练，"
            f"历史最佳验证损失={self.best_validation_loss:.6f}"
        )

    def _next_training_batch(self) -> torch.Tensor:
        if self._training_iterator is None:
            self._training_iterator = iter(self.training_loader)

        try:
            batch = next(self._training_iterator)
        except StopIteration:
            self._training_iterator = iter(self.training_loader)
            batch = next(self._training_iterator)

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        return batch.to(
            self.device,
            non_blocking=True,
        )

    def _read_loss_components(
        self,
        total_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        getter = getattr(
            self.diffusion,
            "get_latest_loss_components",
            None,
        )

        if not callable(getter):
            scalar = total_loss.detach()

            return {
                "total_loss": scalar,
                "ddpm_loss": scalar,
                "physics_loss": torch.zeros_like(scalar),
            }

        raw_components = getter()

        if not isinstance(raw_components, dict):
            raise RuntimeError(
                "get_latest_loss_components必须返回字典。"
            )

        converted: dict[str, torch.Tensor] = {}

        for name, value in raw_components.items():
            if torch.is_tensor(value):
                converted[name] = value.detach()
            else:
                converted[name] = torch.as_tensor(
                    value,
                    device=self.device,
                    dtype=total_loss.dtype,
                )

        return converted

    @staticmethod
    def _accumulate_components(
        destination: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        *,
        scale: float = 1.0,
    ) -> None:
        for name, value in source.items():
            scaled = value.detach() * float(scale)

            if name in destination:
                destination[name] = destination[name] + scaled
            else:
                destination[name] = scaled.clone()

    @staticmethod
    def _average_components_to_float(
        accumulated: dict[str, torch.Tensor],
        count: int,
    ) -> dict[str, float]:
        if count <= 0:
            return {}

        names = list(accumulated)

        if not names:
            return {}

        # 把所有分项拼接后只进行一次GPU到CPU同步。
        stacked = torch.stack(
            [accumulated[name] / count for name in names]
        ).detach().cpu()

        return {
            name: float(stacked[index])
            for index, name in enumerate(names)
        }

    @torch.no_grad()
    def validate(self) -> float:
        self.diffusion.eval()
        accumulated_loss: torch.Tensor | None = None
        accumulated_components: dict[str, torch.Tensor] = {}
        number_of_batches = 0

        for batch_index, batch in enumerate(self.validation_loader):
            if (
                self.maximum_validation_batches > 0
                and batch_index >= self.maximum_validation_batches
            ):
                break

            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch = batch.to(
                self.device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_mixed_precision,
            ):
                loss = self.diffusion(batch)

            detached_loss = loss.detach()
            accumulated_loss = (
                detached_loss.clone()
                if accumulated_loss is None
                else accumulated_loss + detached_loss
            )
            self._accumulate_components(
                accumulated_components,
                self._read_loss_components(loss),
            )
            number_of_batches += 1

        self.diffusion.train()

        if number_of_batches == 0 or accumulated_loss is None:
            raise RuntimeError("验证集没有产生任何批次。")

        validation_loss = float(
            (accumulated_loss / number_of_batches).detach().cpu()
        )
        self.latest_validation_components = (
            self._average_components_to_float(
                accumulated_components,
                number_of_batches,
            )
        )

        return validation_loss

    def _save_checkpoint(
        self,
        *,
        step: int,
        file_name: str | None = None,
        update_latest: bool = True,
    ) -> Path:
        return self.checkpoint_manager.save(
            step=step,
            diffusion_state=self.diffusion.state_dict(),
            ema_state=self.ema.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.gradient_scaler.state_dict(),
            configuration=self.configuration,
            metadata=self.metadata,
            best_validation_loss=self.best_validation_loss,
            file_name=file_name,
            update_latest=update_latest,
        )

    def train(self) -> None:
        if self.starting_step >= self.total_training_steps:
            print(
                "检查点训练步数已经达到或超过"
                "total_training_steps，无需继续训练。"
            )
            return

        self.diffusion.train()
        accumulated_log_loss: torch.Tensor | None = None
        accumulated_log_components: dict[str, torch.Tensor] = {}
        number_of_logged_steps = 0
        latest_validation_loss: float | None = None

        for step in range(
            self.starting_step + 1,
            self.total_training_steps + 1,
        ):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss: torch.Tensor | None = None
            step_components: dict[str, torch.Tensor] = {}

            for _ in range(self.gradient_accumulation_steps):
                batch = self._next_training_batch()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_mixed_precision,
                ):
                    loss = self.diffusion(batch)
                    backward_loss = (
                        loss / self.gradient_accumulation_steps
                    )

                self.gradient_scaler.scale(backward_loss).backward()
                detached_loss = (
                    loss.detach() / self.gradient_accumulation_steps
                )
                step_loss = (
                    detached_loss.clone()
                    if step_loss is None
                    else step_loss + detached_loss
                )
                self._accumulate_components(
                    step_components,
                    self._read_loss_components(loss),
                    scale=(
                        1.0 / self.gradient_accumulation_steps
                    ),
                )

            if step_loss is None:
                raise RuntimeError("训练step没有产生损失。")

            self.gradient_scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.diffusion.parameters(),
                self.maximum_gradient_norm,
            )
            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()
            self.ema.update(self.diffusion)

            accumulated_log_loss = (
                step_loss.clone()
                if accumulated_log_loss is None
                else accumulated_log_loss + step_loss
            )
            self._accumulate_components(
                accumulated_log_components,
                step_components,
            )
            number_of_logged_steps += 1

            should_validate = (
                step % self.validate_every_steps == 0
                or step == self.total_training_steps
            )

            if should_validate:
                latest_validation_loss = self.validate()

                if latest_validation_loss < self.best_validation_loss:
                    self.best_validation_loss = latest_validation_loss
                    self._save_checkpoint(
                        step=step,
                        file_name="best.pt",
                        update_latest=False,
                    )

            should_log = (
                step % self.log_every_steps == 0
                or step == self.total_training_steps
            )

            if should_log:
                if accumulated_log_loss is None:
                    raise RuntimeError("日志区间没有累计训练损失。")

                average_training_loss = float(
                    (
                        accumulated_log_loss
                        / number_of_logged_steps
                    )
                    .detach()
                    .cpu()
                )
                average_training_components = (
                    self._average_components_to_float(
                        accumulated_log_components,
                        number_of_logged_steps,
                    )
                )
                learning_rate = self.optimizer.param_groups[0]["lr"]

                self.logger.record(
                    step=step,
                    training_loss=average_training_loss,
                    validation_loss=latest_validation_loss,
                    learning_rate=learning_rate,
                    training_components=average_training_components,
                    validation_components=(
                        self.latest_validation_components
                    ),
                )

                accumulated_log_loss = None
                accumulated_log_components = {}
                number_of_logged_steps = 0

            should_checkpoint = (
                step % self.checkpoint_every_steps == 0
                or step == self.total_training_steps
            )

            if should_checkpoint:
                checkpoint_path = self._save_checkpoint(
                    step=step,
                    update_latest=True,
                )
                print(f"已保存检查点：{checkpoint_path}")
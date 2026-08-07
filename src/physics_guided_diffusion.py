"""
在第三方 GaussianDiffusion1D 基础上加入 D3.2 SERS 物理损失。

本文件只重写训练阶段的 ``p_losses``。采样、DDPM、DDIM、EMA 和
已有模型参数结构仍沿用 denoising-diffusion-pytorch==2.2.6。
"""

from __future__ import annotations

from random import random
from typing import Any, Callable, TypeVar

import torch
from torch.nn import functional as F

from src.one_dimensional_ddpm import GaussianDiffusion1D
from src.sers_physics_constraints import DifferentiableSersPhysicsLoss


T = TypeVar("T")


def _default(value: T | None, factory: Callable[[], T]) -> T:
    """兼容第三方库 default 辅助函数的本地实现。"""

    return value if value is not None else factory()


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """按 batch 时间步读取一维扩散缓冲区，并扩展到目标形状。"""

    gathered = values.gather(-1, timesteps)

    return gathered.reshape(
        timesteps.shape[0],
        *((1,) * (len(target_shape) - 1)),
    )


class SersPhysicsGuidedGaussianDiffusion1D(GaussianDiffusion1D):
    """D3.2 目标自适应局部病态峰约束的一维 DDPM。"""

    def __init__(
        self,
        model,
        *,
        physics_configuration: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if self.objective != "pred_x0":
            raise ValueError(
                "D3.2物理约束当前只支持"
                "diffusion.objective=pred_x0。"
            )

        if not isinstance(physics_configuration, dict):
            raise TypeError("physics_configuration必须是字典。")

        self.physics_configuration = dict(physics_configuration)
        self.physics_total_weight = float(
            self.physics_configuration["total_weight"]
        )

        if self.physics_total_weight <= 0.0:
            raise ValueError(
                "physics_constraints.total_weight必须大于0。"
            )

        # 训练入口完成训练集划分、归一化、D2.1和D3.2状态拟合后注入。
        self.physics_loss_module: (
            DifferentiableSersPhysicsLoss | None
        ) = None

        # 只用于日志，不写入state_dict。
        self._latest_loss_components: dict[str, torch.Tensor] = {}

    def configure_physics_constraints(
        self,
        *,
        physics_constraint_state: dict[str, Any],
        prior_residual_state: dict[str, Any],
    ) -> None:
        """注入只使用训练集拟合的 D2.1 和 D3.2 状态。"""

        self.physics_loss_module = DifferentiableSersPhysicsLoss(
            physics_constraint_state=physics_constraint_state,
            prior_residual_state=prior_residual_state,
            padded_length=self.seq_length,
        )

    def get_latest_loss_components(self) -> dict[str, torch.Tensor]:
        """返回最近一次前向传播的分项损失副本。"""

        return {
            name: value.detach()
            for name, value in self._latest_loss_components.items()
        }

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        model_forward_kwargs: dict[str, Any] | None = None,
        return_reduced_loss: bool = True,
    ):
        """计算原始 DDPM 损失与 D3.2 物理损失。"""

        if model_forward_kwargs is None:
            model_forward_kwargs = {}

        noise = _default(
            noise,
            lambda: torch.randn_like(x_start),
        )
        noisy_input = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        )

        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                self_condition = self.model_predictions(
                    noisy_input,
                    t,
                ).pred_x_start
                self_condition.detach_()

            model_forward_kwargs = {
                **model_forward_kwargs,
                "self_cond": self_condition,
            }

        model_out = self.model(
            noisy_input,
            t,
            **model_forward_kwargs,
        )

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"未知扩散预测目标：{self.objective}")

        pointwise_ddpm_loss = F.mse_loss(
            model_out,
            target,
            reduction="none",
        )
        pointwise_loss_weight = _extract(
            self.loss_weight,
            t,
            pointwise_ddpm_loss.shape,
        )

        if not return_reduced_loss:
            return pointwise_ddpm_loss * pointwise_loss_weight

        ddpm_per_sample = pointwise_ddpm_loss.flatten(
            start_dim=1
        ).mean(dim=1)
        ddpm_per_sample = ddpm_per_sample * _extract(
            self.loss_weight,
            t,
            ddpm_per_sample.shape,
        )
        ddpm_loss = ddpm_per_sample.mean()

        if self.physics_loss_module is None:
            raise RuntimeError(
                "D3.2扩散模型尚未配置physics_constraint_state。"
                "训练入口必须在模型构建后调用"
                "configure_physics_constraints。"
            )

        physics = self.physics_loss_module(
            predicted_scaled_residual=model_out,
            target_scaled_residual=x_start,
            timesteps=t,
            alphas_cumprod=self.alphas_cumprod,
        )
        weighted_physics_loss = (
            self.physics_total_weight
            * physics["physics_timestep_weighted_loss"]
        )
        total_loss = ddpm_loss + weighted_physics_loss

        self._latest_loss_components = {
            "total_loss": total_loss,
            "ddpm_loss": ddpm_loss,
            "physics_loss": weighted_physics_loss,
            "physics_raw_loss": physics["physics_raw_loss"],
            "position_loss": physics["position_loss"],
            "width_loss": physics["width_loss"],
            "sharpness_loss": physics["sharpness_loss"],
            "presence_loss": physics["presence_loss"],
            "local_shape_loss": physics["local_shape_loss"],
            "roughness_loss": physics["roughness_loss"],
            "extreme_loss": physics["extreme_loss"],
            "negative_valley_loss": physics["negative_valley_loss"],
            "scaled_residual_guard_loss": physics[
                "scaled_residual_guard_loss"
            ],
            "mean_timestep_weight": physics[
                "mean_timestep_weight"
            ],
            "mean_pathology_timestep_weight": physics[
                "mean_pathology_timestep_weight"
            ],
            "mean_detected_peaks": physics["mean_detected_peaks"],
        }

        return total_loss
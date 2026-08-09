"""
在第三方 GaussianDiffusion1D 基础上加入 D3 SERS 额外损失。

D3.4.2：
1. D3.2 physics 继续负责负谷、粗糙度、极端强度和残差病态保护；
2. D3.4.1 高频残差多样性继续保留；
3. D3.4.2 新增目标 batch 驱动的峰位置/整峰协同位移/峰高/峰宽多样性；
4. 峰级多样性复用 D3.2 已加载的 D2.1 可微反变换状态，
   不改变训练入口接口；
5. 所有额外损失可由 YAML 独立开启或关闭。
"""

from __future__ import annotations

from random import random
from typing import Any, Callable, TypeVar

import torch
from torch.nn import functional as F

from src.one_dimensional_ddpm import (
    GaussianDiffusion1D,
)
from src.sers_diversity_constraints import (
    DifferentiableSersDiversityLoss,
    normalize_diversity_configuration,
)
from src.sers_physics_constraints import (
    DifferentiableSersPhysicsLoss,
    normalize_physics_configuration,
)


T = TypeVar("T")


def _default(
    value: T | None,
    factory: Callable[
        [],
        T,
    ],
) -> T:
    return (
        value
        if value is not None
        else factory()
    )


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    gathered = values.gather(
        -1,
        timesteps,
    )

    return gathered.reshape(
        timesteps.shape[0],
        *(
            (1,)
            * (
                len(
                    target_shape
                )
                - 1
            )
        ),
    )


class SersPhysicsGuidedGaussianDiffusion1D(
    GaussianDiffusion1D
):
    """D3物理病态保护 + D3.4.x多样性约束的一维DDPM。"""

    # 由DdpmTrainer识别，避免把D2.2的内部约束先验错误传给第三方U-Net。
    supports_constraint_reference_prior = True

    def __init__(
        self,
        model,
        *,
        physics_configuration: dict[
            str,
            Any,
        ]
        | None = None,
        diversity_configuration: dict[
            str,
            Any,
        ]
        | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            **kwargs,
        )

        self.physics_configuration = (
            normalize_physics_configuration(
                physics_configuration
                or {
                    "enabled": False
                }
            )
        )

        self.diversity_configuration = (
            normalize_diversity_configuration(
                diversity_configuration
                or {
                    "enabled": False
                }
            )
        )

        self.physics_enabled = bool(
            self.physics_configuration.get(
                "enabled",
                False,
            )
        )

        self.diversity_enabled = bool(
            self.diversity_configuration.get(
                "enabled",
                False,
            )
        )

        if (
            not self.physics_enabled
            and not self.diversity_enabled
        ):
            raise ValueError(
                "SersPhysicsGuidedGaussianDiffusion1D"
                "至少需要启用physics_constraints"
                "或diversity_constraints之一。"
            )

        if self.objective != "pred_x0":
            raise ValueError(
                "D3物理/多样性约束当前只支持"
                "diffusion.objective=pred_x0。"
            )

        self.physics_total_weight = (
            float(
                self.physics_configuration[
                    "total_weight"
                ]
            )
            if self.physics_enabled
            else 0.0
        )

        self.diversity_total_weight = (
            float(
                self.diversity_configuration[
                    "total_weight"
                ]
            )
            if self.diversity_enabled
            else 0.0
        )

        if (
            self.physics_enabled
            and self.physics_total_weight
            <= 0.0
        ):
            raise ValueError(
                "physics_constraints.total_weight"
                "必须大于0。"
            )

        if (
            self.diversity_enabled
            and self.diversity_total_weight
            <= 0.0
        ):
            raise ValueError(
                "diversity_constraints.total_weight"
                "必须大于0。"
            )

        self.physics_loss_module: (
            DifferentiableSersPhysicsLoss
            | None
        ) = None

        self.diversity_loss_module: (
            DifferentiableSersDiversityLoss
            | None
        ) = None

        self._latest_loss_components: dict[
            str,
            torch.Tensor,
        ] = {}

    def _sync_diversity_inverse_reference(
        self,
    ) -> None:
        """同步D3.2中的D2.1反变换参数到D3.4.2。

        这样 start_ddpm_training.py、generate_spectra.py 和
        diagnose_timestep_recovery.py 都不需要新增参数。

        physics 与 diversity 无论哪个先 configure，
        第二个完成后都会再次尝试同步。
        """

        if (
            self.physics_loss_module
            is None
            or self.diversity_loss_module
            is None
        ):
            return

        if not bool(
            self.diversity_loss_module
            .configuration
            .get(
                "peak_morphology",
                {},
            )
            .get(
                "enabled",
                False,
            )
        ):
            return

        physics = (
            self.physics_loss_module
        )

        # 当前D3.2历史版本曾使用
        # numerical_safety_limit或soft_argument_limit。
        # 这里同时兼容两种属性名。
        numerical_limit = float(
            getattr(
                physics,
                "numerical_safety_limit",
                getattr(
                    physics,
                    "soft_argument_limit",
                    15.0,
                ),
            )
        )

        self.diversity_loss_module.configure_inverse_reference(
            prior_normalized_intensity=(
                physics.prior.detach()
            ),
            pointwise_scale=(
                physics.pointwise_scale.detach()
            ),
            raman_shift=(
                physics.raman_shift.detach()
            ),
            target_abs_max=float(
                physics.target_abs_max
            ),
            standardized_residual_scale=float(
                physics.standardized_residual_scale
            ),
            asinh_normalizer=float(
                physics.asinh_normalizer
            ),
            numerical_safety_limit=(
                numerical_limit
            ),
        )

    def configure_physics_constraints(
        self,
        *,
        physics_constraint_state: dict[
            str,
            Any,
        ],
        prior_residual_state: dict[
            str,
            Any,
        ],
    ) -> None:
        """注入训练集拟合的D2.1与D3.2状态。"""

        self.physics_loss_module = (
            DifferentiableSersPhysicsLoss(
                physics_constraint_state=(
                    physics_constraint_state
                ),
                prior_residual_state=(
                    prior_residual_state
                ),
                padded_length=(
                    self.seq_length
                ),
            )
        )

        self._sync_diversity_inverse_reference()

    def configure_diversity_constraints(
        self,
        *,
        diversity_constraint_state: dict[
            str,
            Any,
        ],
    ) -> None:
        """注入训练集拟合的D3.4.x多样性状态。"""

        self.diversity_loss_module = (
            DifferentiableSersDiversityLoss(
                diversity_constraint_state=(
                    diversity_constraint_state
                ),
                padded_length=(
                    self.seq_length
                ),
            )
        )

        self._sync_diversity_inverse_reference()

    def get_latest_loss_components(
        self,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        return {
            name: value.detach()
            for name, value
            in self._latest_loss_components.items()
        }

    def forward(
        self,
        img: torch.Tensor,
        *args,
        constraint_reference_prior: torch.Tensor | None = None,
        **kwargs,
    ):
        """保持第三方forward流程，仅把D2.2先验传给约束损失。

        ``constraint_reference_prior``不是U-Net条件输入；它只用于把预测和
        目标残差恢复到同一条PCA先验对应的完整光谱域，因而不会改变网络结构。
        """

        batch_size, channels, sequence_length = img.shape
        if channels != self.channels or sequence_length != self.seq_length:
            raise ValueError(
                "输入光谱形状与扩散模型不一致："
                f"得到{tuple(img.shape)}，期望通道={self.channels}、"
                f"长度={self.seq_length}。"
            )
        if constraint_reference_prior is not None:
            if constraint_reference_prior.shape != img.shape:
                raise ValueError(
                    "constraint_reference_prior形状必须与img一致。"
                )
            constraint_reference_prior = constraint_reference_prior.to(
                device=img.device,
                dtype=img.dtype,
            )
        timestep = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=img.device,
        ).long()
        img = self.normalize(img)
        return self.p_losses(
            img,
            timestep,
            *args,
            constraint_reference_prior=constraint_reference_prior,
            **kwargs,
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
        | None = None,
        model_forward_kwargs: dict[
            str,
            Any,
        ]
        | None = None,
        constraint_reference_prior: torch.Tensor | None = None,
        return_reduced_loss: bool = True,
    ):
        if model_forward_kwargs is None:
            model_forward_kwargs = {}

        noise = _default(
            noise,
            lambda: torch.randn_like(
                x_start
            ),
        )

        noisy_input = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        )

        if (
            self.self_condition
            and random() < 0.5
        ):
            with torch.no_grad():
                self_condition = (
                    self.model_predictions(
                        noisy_input,
                        t,
                    ).pred_x_start
                )

                self_condition.detach_()

            model_forward_kwargs = {
                **model_forward_kwargs,
                "self_cond": (
                    self_condition
                ),
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
            target = self.predict_v(
                x_start,
                t,
                noise,
            )

        else:
            raise ValueError(
                "未知扩散预测目标："
                f"{self.objective}"
            )

        pointwise_ddpm_loss = (
            F.mse_loss(
                model_out,
                target,
                reduction="none",
            )
        )

        pointwise_loss_weight = (
            _extract(
                self.loss_weight,
                t,
                pointwise_ddpm_loss.shape,
            )
        )

        if not return_reduced_loss:
            return (
                pointwise_ddpm_loss
                * pointwise_loss_weight
            )

        ddpm_per_sample = (
            pointwise_ddpm_loss
            .flatten(
                start_dim=1
            )
            .mean(
                dim=1
            )
        )

        ddpm_per_sample = (
            ddpm_per_sample
            * _extract(
                self.loss_weight,
                t,
                ddpm_per_sample.shape,
            )
        )

        ddpm_loss = (
            ddpm_per_sample.mean()
        )

        zero = torch.zeros_like(
            ddpm_loss
        )

        if self.physics_enabled:
            if (
                self.physics_loss_module
                is None
            ):
                raise RuntimeError(
                    "D3物理扩散模型尚未配置"
                    "physics_constraint_state。"
                )

            physics = (
                self.physics_loss_module(
                    predicted_scaled_residual=(
                        model_out
                    ),
                    target_scaled_residual=(
                        x_start
                    ),
                    timesteps=t,
                    alphas_cumprod=(
                        self.alphas_cumprod
                    ),
                    reference_prior=constraint_reference_prior,
                )
            )

            weighted_physics_loss = (
                self.physics_total_weight
                * physics[
                    "physics_timestep_weighted_loss"
                ]
            )

        else:
            physics = {}
            weighted_physics_loss = (
                zero
            )

        if self.diversity_enabled:
            if (
                self.diversity_loss_module
                is None
            ):
                raise RuntimeError(
                    "D3.4扩散模型尚未配置"
                    "diversity_constraint_state。"
                )

            diversity = (
                self.diversity_loss_module(
                    predicted_scaled_residual=(
                        model_out
                    ),
                    target_scaled_residual=(
                        x_start
                    ),
                    timesteps=t,
                    alphas_cumprod=(
                        self.alphas_cumprod
                    ),
                    reference_prior=constraint_reference_prior,
                )
            )

            weighted_diversity_loss = (
                self.diversity_total_weight
                * diversity[
                    "diversity_timestep_weighted_loss"
                ]
            )

        else:
            diversity = {}
            weighted_diversity_loss = (
                zero
            )

        total_loss = (
            ddpm_loss
            + weighted_physics_loss
            + weighted_diversity_loss
        )

        self._latest_loss_components = {
            "total_loss": (
                total_loss
            ),
            "ddpm_loss": (
                ddpm_loss
            ),
            "physics_loss": (
                weighted_physics_loss
            ),
            "physics_raw_loss": physics.get(
                "physics_raw_loss",
                zero,
            ),
            "position_loss": physics.get(
                "position_loss",
                zero,
            ),
            "width_loss": physics.get(
                "width_loss",
                zero,
            ),
            "sharpness_loss": physics.get(
                "sharpness_loss",
                zero,
            ),
            "presence_loss": physics.get(
                "presence_loss",
                zero,
            ),
            "local_shape_loss": physics.get(
                "local_shape_loss",
                zero,
            ),
            "roughness_loss": physics.get(
                "roughness_loss",
                zero,
            ),
            "extreme_loss": physics.get(
                "extreme_loss",
                zero,
            ),
            "negative_valley_loss": physics.get(
                "negative_valley_loss",
                zero,
            ),
            "scaled_residual_guard_loss": physics.get(
                "scaled_residual_guard_loss",
                zero,
            ),
            "mean_timestep_weight": physics.get(
                "mean_timestep_weight",
                zero,
            ),
            "mean_pathology_timestep_weight": physics.get(
                "mean_pathology_timestep_weight",
                zero,
            ),
            "mean_detected_peaks": physics.get(
                "mean_detected_peaks",
                zero,
            ),
            "diversity_loss": (
                weighted_diversity_loss
            ),
            "diversity_raw_loss": diversity.get(
                "diversity_raw_loss",
                zero,
            ),
            "pairwise_distance_loss": diversity.get(
                "pairwise_distance_loss",
                zero,
            ),
            "pairwise_correlation_loss": diversity.get(
                "pairwise_correlation_loss",
                zero,
            ),
            "pointwise_variance_floor_loss": diversity.get(
                "pointwise_variance_floor_loss",
                zero,
            ),
            "peak_morphology_loss": diversity.get(
                "peak_morphology_loss",
                zero,
            ),
            "peak_position_dispersion_loss": diversity.get(
                "peak_position_dispersion_loss",
                zero,
            ),
            "peak_shift_coherence_loss": diversity.get(
                "peak_shift_coherence_loss",
                zero,
            ),
            "peak_height_dispersion_loss": diversity.get(
                "peak_height_dispersion_loss",
                zero,
            ),
            "peak_width_dispersion_loss": diversity.get(
                "peak_width_dispersion_loss",
                zero,
            ),
            "mean_morphology_peaks": diversity.get(
                "mean_morphology_peaks",
                zero,
            ),
            "diversity_active_samples": diversity.get(
                "diversity_active_samples",
                zero,
            ),
            "mean_diversity_timestep_weight": diversity.get(
                "mean_diversity_timestep_weight",
                zero,
            ),
        }

        return total_loss
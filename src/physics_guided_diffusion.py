"""
在第三方 GaussianDiffusion1D 上叠加 D3 SERS 物理、多样性、
residual-aware 以及局部稳定峰分布损失。

本文件只改变训练阶段损失组合；采样、DDPM、DDIM、EMA 和 U-Net
参数结构仍沿用 denoising-diffusion-pytorch==2.2.6。
"""

from __future__ import annotations

from copy import deepcopy
from random import random
from typing import Any, Callable, TypeVar

import torch
from torch.nn import functional as F

from src.one_dimensional_ddpm import GaussianDiffusion1D
from src.sers_diversity_constraints import (
    DifferentiableSersDiversityLoss,
    normalize_diversity_configuration,
)
from src.sers_local_peak_distribution_constraints import (
    DifferentiableSersLocalPeakDistributionLoss,
    normalize_local_peak_distribution_configuration,
)
from src.sers_physics_constraints import (
    DifferentiableSersPhysicsLoss,
    normalize_physics_configuration,
)


T = TypeVar("T")


def _default(value: T | None, factory: Callable[[], T]) -> T:
    return value if value is not None else factory()


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    gathered = values.gather(-1, timesteps)
    return gathered.reshape(
        timesteps.shape[0],
        *((1,) * (len(target_shape) - 1)),
    )


_DEFAULT_RESIDUAL_AWARE_CONFIGURATION: dict[str, Any] = {
    "enabled": False,
    "reference_quantile": 0.90,
    "strength": 2.0,
    "power": 1.0,
    "maximum_relative_magnitude": 3.0,
    "minimum_reference_scale": 0.02,
    "normalize_per_sample": True,
}


def _normalize_residual_aware_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = configuration or {"enabled": False}
    if not isinstance(raw, dict):
        raise TypeError("diffusion.residual_aware_loss必须是字典。")

    config = deepcopy(_DEFAULT_RESIDUAL_AWARE_CONFIGURATION)
    config.update(raw)
    config["enabled"] = bool(config["enabled"])

    quantile = float(config["reference_quantile"])
    if not 0.0 < quantile < 1.0:
        raise ValueError(
            "residual_aware_loss.reference_quantile必须在(0,1)内。"
        )
    config["reference_quantile"] = quantile

    for key in (
        "strength",
        "power",
        "maximum_relative_magnitude",
        "minimum_reference_scale",
    ):
        value = float(config[key])
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise ValueError(f"residual_aware_loss.{key}必须为有限正数。")
        config[key] = value

    config["normalize_per_sample"] = bool(config["normalize_per_sample"])
    return config


def _build_residual_aware_weights(
    target: torch.Tensor,
    configuration: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = target.detach().abs().flatten(start_dim=1)
    reference = torch.quantile(
        absolute,
        q=float(configuration["reference_quantile"]),
        dim=1,
        keepdim=True,
    ).clamp_min(float(configuration["minimum_reference_scale"]))

    relative = absolute / reference
    relative = relative.clamp_max(
        float(configuration["maximum_relative_magnitude"])
    )

    weights = 1.0 + float(configuration["strength"]) * torch.pow(
        relative,
        float(configuration["power"]),
    )
    weights = weights.reshape_as(target)

    if bool(configuration["normalize_per_sample"]):
        mean_weight = (
            weights.flatten(start_dim=1)
            .mean(dim=1, keepdim=True)
            .reshape(target.shape[0], *((1,) * (target.ndim - 1)))
            .clamp_min(1.0e-8)
        )
        weights = weights / mean_weight

    return weights, reference.squeeze(1)


class SersPhysicsGuidedGaussianDiffusion1D(GaussianDiffusion1D):
    """D3 额外约束的一维 DDPM。"""

    supports_constraint_reference_prior = True

    def __init__(
        self,
        model,
        *,
        physics_configuration: dict[str, Any] | None = None,
        diversity_configuration: dict[str, Any] | None = None,
        local_peak_distribution_configuration: dict[str, Any] | None = None,
        residual_aware_configuration: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # 四个项目自定义配置必须在这里显式接住，不能进入第三方 **kwargs。
        super().__init__(model, **kwargs)

        self.physics_configuration = normalize_physics_configuration(
            physics_configuration or {"enabled": False}
        )
        self.diversity_configuration = normalize_diversity_configuration(
            diversity_configuration or {"enabled": False}
        )
        self.local_peak_distribution_configuration = (
            normalize_local_peak_distribution_configuration(
                local_peak_distribution_configuration or {"enabled": False}
            )
        )
        self.residual_aware_configuration = (
            _normalize_residual_aware_configuration(
                residual_aware_configuration or {"enabled": False}
            )
        )

        self.physics_enabled = bool(
            self.physics_configuration.get("enabled", False)
        )
        self.diversity_enabled = bool(
            self.diversity_configuration.get("enabled", False)
        )
        self.local_peak_distribution_enabled = bool(
            self.local_peak_distribution_configuration.get("enabled", False)
        )
        self.residual_aware_enabled = bool(
            self.residual_aware_configuration.get("enabled", False)
        )

        if not any(
            (
                self.physics_enabled,
                self.diversity_enabled,
                self.local_peak_distribution_enabled,
                self.residual_aware_enabled,
            )
        ):
            raise ValueError(
                "SersPhysicsGuidedGaussianDiffusion1D至少需要启用"
                "physics、diversity、local_peak_distribution或"
                "residual_aware之一。"
            )

        if self.objective != "pred_x0":
            raise ValueError(
                "D3/residual-aware额外损失当前只支持"
                "diffusion.objective=pred_x0。"
            )

        if self.local_peak_distribution_enabled and not self.physics_enabled:
            raise ValueError(
                "local_peak_distribution_constraints依赖训练集physics状态，"
                "因此physics_constraints.enabled必须为true。"
            )

        self.physics_total_weight = (
            float(self.physics_configuration["total_weight"])
            if self.physics_enabled
            else 0.0
        )
        self.diversity_total_weight = (
            float(self.diversity_configuration["total_weight"])
            if self.diversity_enabled
            else 0.0
        )

        if self.physics_enabled and self.physics_total_weight <= 0.0:
            raise ValueError("physics_constraints.total_weight必须大于0。")
        if self.diversity_enabled and self.diversity_total_weight <= 0.0:
            raise ValueError("diversity_constraints.total_weight必须大于0。")

        self.physics_loss_module: DifferentiableSersPhysicsLoss | None = None
        self.diversity_loss_module: DifferentiableSersDiversityLoss | None = None
        self.local_peak_distribution_loss_module: (
            DifferentiableSersLocalPeakDistributionLoss | None
        ) = None

        self._latest_loss_components: dict[str, torch.Tensor] = {}

    def _sync_diversity_inverse_reference(self) -> None:
        if self.physics_loss_module is None or self.diversity_loss_module is None:
            return
        if not bool(
            self.diversity_loss_module.configuration
            .get("peak_morphology", {})
            .get("enabled", False)
        ):
            return

        physics = self.physics_loss_module
        numerical_limit = float(
            getattr(
                physics,
                "numerical_safety_limit",
                getattr(physics, "soft_argument_limit", 15.0),
            )
        )
        self.diversity_loss_module.configure_inverse_reference(
            prior_normalized_intensity=physics.prior.detach(),
            pointwise_scale=physics.pointwise_scale.detach(),
            raman_shift=physics.raman_shift.detach(),
            target_abs_max=float(physics.target_abs_max),
            standardized_residual_scale=float(
                physics.standardized_residual_scale
            ),
            asinh_normalizer=float(physics.asinh_normalizer),
            numerical_safety_limit=numerical_limit,
        )

    def configure_physics_constraints(
        self,
        *,
        physics_constraint_state: dict[str, Any],
        prior_residual_state: dict[str, Any],
    ) -> None:
        if self.physics_enabled:
            self.physics_loss_module = DifferentiableSersPhysicsLoss(
                physics_constraint_state=physics_constraint_state,
                prior_residual_state=prior_residual_state,
                padded_length=self.seq_length,
            )

        if self.local_peak_distribution_enabled:
            self.local_peak_distribution_loss_module = (
                DifferentiableSersLocalPeakDistributionLoss(
                    configuration=self.local_peak_distribution_configuration,
                    physics_constraint_state=physics_constraint_state,
                    prior_residual_state=prior_residual_state,
                    padded_length=self.seq_length,
                )
            )

        self._sync_diversity_inverse_reference()

    def configure_diversity_constraints(
        self,
        *,
        diversity_constraint_state: dict[str, Any],
    ) -> None:
        if not self.diversity_enabled:
            return

        self.diversity_loss_module = DifferentiableSersDiversityLoss(
            diversity_constraint_state=diversity_constraint_state,
            padded_length=self.seq_length,
        )
        self._sync_diversity_inverse_reference()

    def get_latest_loss_components(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach()
            for name, value in self._latest_loss_components.items()
        }

    def forward(
        self,
        img: torch.Tensor,
        *args,
        constraint_reference_prior: torch.Tensor | None = None,
        **kwargs,
    ):
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
        noise: torch.Tensor | None = None,
        model_forward_kwargs: dict[str, Any] | None = None,
        constraint_reference_prior: torch.Tensor | None = None,
        return_reduced_loss: bool = True,
    ):
        if model_forward_kwargs is None:
            model_forward_kwargs = {}

        noise = _default(noise, lambda: torch.randn_like(x_start))
        noisy_input = self.q_sample(x_start=x_start, t=t, noise=noise)

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

        model_out = self.model(noisy_input, t, **model_forward_kwargs)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"未知扩散预测目标：{self.objective}")

        pointwise_uniform_loss = F.mse_loss(
            model_out,
            target,
            reduction="none",
        )

        if self.residual_aware_enabled:
            residual_weights, residual_reference_scale = (
                _build_residual_aware_weights(
                    target,
                    self.residual_aware_configuration,
                )
            )
            pointwise_ddpm_loss = pointwise_uniform_loss * residual_weights
        else:
            residual_weights = torch.ones_like(pointwise_uniform_loss)
            residual_reference_scale = target.new_zeros(target.shape[0])
            pointwise_ddpm_loss = pointwise_uniform_loss

        pointwise_loss_weight = _extract(
            self.loss_weight,
            t,
            pointwise_ddpm_loss.shape,
        )

        if not return_reduced_loss:
            return pointwise_ddpm_loss * pointwise_loss_weight

        timestep_sample_weight = _extract(
            self.loss_weight,
            t,
            torch.Size([target.shape[0]]),
        )

        uniform_per_sample = (
            pointwise_uniform_loss.flatten(start_dim=1).mean(dim=1)
            * timestep_sample_weight
        )
        residual_aware_per_sample = (
            pointwise_ddpm_loss.flatten(start_dim=1).mean(dim=1)
            * timestep_sample_weight
        )

        ddpm_uniform_loss = uniform_per_sample.mean()
        ddpm_residual_aware_loss = residual_aware_per_sample.mean()
        ddpm_loss = (
            ddpm_residual_aware_loss
            if self.residual_aware_enabled
            else ddpm_uniform_loss
        )

        zero = torch.zeros_like(ddpm_loss)

        if self.physics_enabled:
            if self.physics_loss_module is None:
                raise RuntimeError(
                    "D3物理扩散模型尚未配置physics_constraint_state。"
                )
            physics = self.physics_loss_module(
                predicted_scaled_residual=model_out,
                target_scaled_residual=x_start,
                timesteps=t,
                alphas_cumprod=self.alphas_cumprod,
                reference_prior=constraint_reference_prior,
            )
            weighted_physics_loss = (
                self.physics_total_weight
                * physics["physics_timestep_weighted_loss"]
            )
        else:
            physics = {}
            weighted_physics_loss = zero

        if self.diversity_enabled:
            if self.diversity_loss_module is None:
                raise RuntimeError(
                    "D3.4扩散模型尚未配置diversity_constraint_state。"
                )
            diversity = self.diversity_loss_module(
                predicted_scaled_residual=model_out,
                target_scaled_residual=x_start,
                timesteps=t,
                alphas_cumprod=self.alphas_cumprod,
                reference_prior=constraint_reference_prior,
            )
            weighted_diversity_loss = (
                self.diversity_total_weight
                * diversity["diversity_timestep_weighted_loss"]
            )
        else:
            diversity = {}
            weighted_diversity_loss = zero

        if self.local_peak_distribution_enabled:
            if self.local_peak_distribution_loss_module is None:
                raise RuntimeError(
                    "局部峰分布模块尚未配置训练集physics/prior状态。"
                )

            local_peak = self.local_peak_distribution_loss_module(
                predicted_scaled_residual=model_out,
                target_scaled_residual=x_start,
                timesteps=t,
                alphas_cumprod=self.alphas_cumprod,
                reference_prior=constraint_reference_prior,
            )

            raw_local_peak_loss = local_peak["local_peak_distribution_loss"]
            maximum_local_ratio = float(
                self.local_peak_distribution_configuration[
                    "maximum_total_ratio_to_ddpm"
                ]
            )
            local_peak_loss_cap = ddpm_loss.detach() * maximum_local_ratio

            if float(raw_local_peak_loss.detach().abs().item()) <= 1.0e-12:
                local_peak_loss_scale = raw_local_peak_loss.new_tensor(1.0)
            else:
                local_peak_loss_scale = torch.clamp(
                    local_peak_loss_cap
                    / raw_local_peak_loss.detach().abs().clamp_min(1.0e-12),
                    max=1.0,
                )

            weighted_local_peak_loss = (
                raw_local_peak_loss * local_peak_loss_scale
            )
        else:
            local_peak = {}
            raw_local_peak_loss = zero
            local_peak_loss_cap = zero
            local_peak_loss_scale = zero
            weighted_local_peak_loss = zero

        total_loss = (
            ddpm_loss
            + weighted_physics_loss
            + weighted_diversity_loss
            + weighted_local_peak_loss
        )

        self._latest_loss_components = {
            "total_loss": total_loss,
            "ddpm_loss": ddpm_loss,
            "ddpm_uniform_loss": ddpm_uniform_loss,
            "ddpm_residual_aware_loss": ddpm_residual_aware_loss,
            "residual_reference_scale": residual_reference_scale.mean(),
            "residual_weight_mean": residual_weights.mean(),
            "residual_weight_min": residual_weights.min(),
            "residual_weight_max": residual_weights.max(),

            "physics_loss": weighted_physics_loss,
            "physics_raw_loss": physics.get("physics_raw_loss", zero),
            "position_loss": physics.get("position_loss", zero),
            "width_loss": physics.get("width_loss", zero),
            "sharpness_loss": physics.get("sharpness_loss", zero),
            "presence_loss": physics.get("presence_loss", zero),
            "local_shape_loss": physics.get("local_shape_loss", zero),
            "stable_peak_shift_loss": physics.get(
                "stable_peak_shift_loss", zero
            ),
            "stable_peak_coherence_loss": physics.get(
                "stable_peak_coherence_loss", zero
            ),
            "stable_peak_height_loss": physics.get(
                "stable_peak_height_loss", zero
            ),
            "stable_peak_width_loss": physics.get(
                "stable_peak_width_loss", zero
            ),
            "stable_peak_mean_absolute_shift_cm1": physics.get(
                "stable_peak_mean_absolute_shift_cm1", zero
            ),
            "stable_peak_count": physics.get("stable_peak_count", zero),
            "roughness_loss": physics.get("roughness_loss", zero),
            "extreme_loss": physics.get("extreme_loss", zero),
            "negative_valley_loss": physics.get("negative_valley_loss", zero),
            "scaled_residual_guard_loss": physics.get(
                "scaled_residual_guard_loss", zero
            ),
            "mean_timestep_weight": physics.get("mean_timestep_weight", zero),
            "mean_pathology_timestep_weight": physics.get(
                "mean_pathology_timestep_weight", zero
            ),
            "mean_detected_peaks": physics.get("mean_detected_peaks", zero),

            "diversity_loss": weighted_diversity_loss,
            "diversity_raw_loss": diversity.get("diversity_raw_loss", zero),
            "pairwise_distance_loss": diversity.get(
                "pairwise_distance_loss", zero
            ),
            "pairwise_correlation_loss": diversity.get(
                "pairwise_correlation_loss", zero
            ),
            "pointwise_variance_floor_loss": diversity.get(
                "pointwise_variance_floor_loss", zero
            ),
            "peak_morphology_loss": diversity.get(
                "peak_morphology_loss", zero
            ),
            "peak_position_dispersion_loss": diversity.get(
                "peak_position_dispersion_loss", zero
            ),
            "peak_shift_coherence_loss": diversity.get(
                "peak_shift_coherence_loss", zero
            ),
            "peak_height_dispersion_loss": diversity.get(
                "peak_height_dispersion_loss", zero
            ),
            "peak_width_dispersion_loss": diversity.get(
                "peak_width_dispersion_loss", zero
            ),
            "mean_morphology_peaks": diversity.get(
                "mean_morphology_peaks", zero
            ),
            "diversity_active_samples": diversity.get(
                "diversity_active_samples", zero
            ),
            "mean_diversity_timestep_weight": diversity.get(
                "mean_diversity_timestep_weight", zero
            ),

            "local_peak_distribution_loss": weighted_local_peak_loss,
            "local_peak_distribution_raw_loss": raw_local_peak_loss,
            "local_peak_loss_cap": local_peak_loss_cap,
            "local_peak_loss_scale": local_peak_loss_scale,
            "negative_tail_guard_loss": local_peak.get(
                "negative_tail_guard_loss", zero
            ),
            "negative_tail_guard_contribution": local_peak.get(
                "negative_tail_guard_contribution", zero
            ),
            "wing_shape_tracking_loss": local_peak.get(
                "wing_shape_tracking_loss", zero
            ),
            "wing_shape_tracking_contribution": local_peak.get(
                "wing_shape_tracking_contribution", zero
            ),
            "wing_dispersion_loss": local_peak.get(
                "wing_dispersion_loss", zero
            ),
            "wing_dispersion_contribution": local_peak.get(
                "wing_dispersion_contribution", zero
            ),
            "shift_dispersion_loss": local_peak.get(
                "shift_dispersion_loss", zero
            ),
            "shift_dispersion_contribution": local_peak.get(
                "shift_dispersion_contribution", zero
            ),
            "width_dispersion_loss": local_peak.get(
                "width_dispersion_loss", zero
            ),
            "width_dispersion_contribution": local_peak.get(
                "width_dispersion_contribution", zero
            ),
            "height_cv_upper_loss": local_peak.get(
                "height_cv_upper_loss", zero
            ),
            "height_cv_upper_contribution": local_peak.get(
                "height_cv_upper_contribution", zero
            ),
            "local_peak_active_samples": local_peak.get(
                "local_peak_active_samples", zero
            ),
            "local_peak_count": local_peak.get("local_peak_count", zero),
            "mean_local_peak_timestep_weight": local_peak.get(
                "mean_local_peak_timestep_weight", zero
            ),
        }

        return total_loss
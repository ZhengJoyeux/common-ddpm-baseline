"""
D3 局部稳定峰分布约束。

目标：
1. 不再把每条预测光谱逐峰拉向某一条 target；
2. 只在低噪声时间步比较 batch 内稳定峰的真实分布；
3. 让峰位、峰宽和峰下半部/两翼具有训练样本级别的适度多样性；
4. 只对“过大的峰高 CV”施加上界保护，避免用峰高变化来凑多样性；
5. 使用训练集逐波数下包络抑制异常深负峰，只限制下界，不限制正峰。

稳定峰位置和逐波数下包络均来自 physics_constraint_state，
而 physics_constraint_state 只由训练集拟合，因此不会使用 validation/test 信息。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_LOCAL_PEAK_DISTRIBUTION_CONFIGURATION: dict[str, Any] = {
    "enabled": False,
    "epsilon": 1.0e-8,
    "numerical_safety_limit": 15.0,
    "minimum_batch_samples": 3,
    "maximum_total_ratio_to_ddpm": 0.20,
    "low_noise_gate": {
        "full_weight_max_timestep": 20,
        "zero_weight_min_timestep": 35,
    },
    "stable_peak_filter": {
        "maximum_position_std_cm1": 2.0,
        "minimum_height_median": 0.05,
    },
    "peak_window": {
        "edge_fraction": 0.12,
        "softplus_temperature_fraction": 0.01,
    },
    "wing_shape_tracking": {
        "enabled": True,
        "weight": 0.0008,
        "lower_profile_fraction": 0.15,
        "upper_profile_fraction": 0.70,
    },
    "negative_tail": {
        "enabled": True,
        "weight": 0.006,
        "transition_fraction": 0.03,
        "minimum_timestep_weight": 0.05,
        "mean_weight": 0.20,
        "topk_weight": 0.80,
        "topk_fraction": 0.01,
    },
    "wing_dispersion": {
        "enabled": True,
        "weight": 0.0015,
        "lower_profile_fraction": 0.15,
        "upper_profile_fraction": 0.65,
        "minimum_reference_std": 0.01,
        "lower_std_ratio": 0.90,
        "upper_std_ratio": 1.15,
    },
    "shift_dispersion": {
        "enabled": True,
        "weight": 0.0005,
        "minimum_reference_std_cm1": 0.25,
        "lower_std_ratio": 0.85,
        "upper_std_ratio": 1.20,
    },
    "width_dispersion": {
        "enabled": True,
        "weight": 0.0005,
        "minimum_reference_std_cm1": 0.25,
        "lower_std_ratio": 0.85,
        "upper_std_ratio": 1.20,
    },
    "height_cv_upper": {
        "enabled": True,
        "weight": 0.0010,
        "minimum_reference_cv": 0.02,
        "upper_ratio": 1.05,
    },
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _require_finite_nonnegative(value: Any, name: str) -> float:
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)) or parsed < 0.0:
        raise ValueError(f"{name}必须是有限的非负数。")
    return parsed


def _require_finite_positive(value: Any, name: str) -> float:
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)) or parsed <= 0.0:
        raise ValueError(f"{name}必须是有限的正数。")
    return parsed


def _validate_ratio_band(configuration: dict[str, Any], prefix: str) -> None:
    lower = _require_finite_positive(
        configuration["lower_std_ratio"],
        f"{prefix}.lower_std_ratio",
    )
    upper = _require_finite_positive(
        configuration["upper_std_ratio"],
        f"{prefix}.upper_std_ratio",
    )
    if lower >= upper:
        raise ValueError(
            f"{prefix}.lower_std_ratio必须小于upper_std_ratio。"
        )


def normalize_local_peak_distribution_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """补齐并校验 D3 局部峰分布配置。"""

    raw = configuration or {"enabled": False}
    if not isinstance(raw, dict):
        raise TypeError("local_peak_distribution_constraints必须是字典。")

    config = _merge_dict(
        DEFAULT_LOCAL_PEAK_DISTRIBUTION_CONFIGURATION,
        raw,
    )
    config["enabled"] = bool(config.get("enabled", False))

    epsilon = _require_finite_positive(config["epsilon"], "epsilon")
    config["epsilon"] = epsilon
    config["numerical_safety_limit"] = _require_finite_positive(
        config["numerical_safety_limit"],
        "numerical_safety_limit",
    )

    minimum_batch_samples = int(config["minimum_batch_samples"])
    if minimum_batch_samples < 2:
        raise ValueError("minimum_batch_samples至少为2。")
    config["minimum_batch_samples"] = minimum_batch_samples

    maximum_ratio = _require_finite_positive(
        config["maximum_total_ratio_to_ddpm"],
        "maximum_total_ratio_to_ddpm",
    )
    if maximum_ratio > 1.0:
        raise ValueError("maximum_total_ratio_to_ddpm不能大于1。")
    config["maximum_total_ratio_to_ddpm"] = maximum_ratio

    gate = config["low_noise_gate"]
    full_t = int(gate["full_weight_max_timestep"])
    zero_t = int(gate["zero_weight_min_timestep"])
    if full_t < 0:
        raise ValueError("full_weight_max_timestep不能小于0。")
    if zero_t <= full_t:
        raise ValueError(
            "zero_weight_min_timestep必须大于full_weight_max_timestep。"
        )
    gate["full_weight_max_timestep"] = full_t
    gate["zero_weight_min_timestep"] = zero_t

    stable_filter = config["stable_peak_filter"]
    stable_filter["maximum_position_std_cm1"] = _require_finite_positive(
        stable_filter["maximum_position_std_cm1"],
        "stable_peak_filter.maximum_position_std_cm1",
    )
    stable_filter["minimum_height_median"] = _require_finite_nonnegative(
        stable_filter["minimum_height_median"],
        "stable_peak_filter.minimum_height_median",
    )

    peak_window = config["peak_window"]
    edge_fraction = float(peak_window["edge_fraction"])
    if not 0.0 < edge_fraction < 0.5:
        raise ValueError("peak_window.edge_fraction必须在(0,0.5)内。")
    peak_window["edge_fraction"] = edge_fraction
    peak_window["softplus_temperature_fraction"] = _require_finite_positive(
        peak_window["softplus_temperature_fraction"],
        "peak_window.softplus_temperature_fraction",
    )

    tracking = config["wing_shape_tracking"]
    tracking["enabled"] = bool(tracking["enabled"])
    tracking["weight"] = _require_finite_nonnegative(
        tracking["weight"], "wing_shape_tracking.weight"
    )
    tracking_lower = float(tracking["lower_profile_fraction"])
    tracking_upper = float(tracking["upper_profile_fraction"])
    if not 0.0 <= tracking_lower < tracking_upper <= 1.0:
        raise ValueError(
            "wing_shape_tracking的profile_fraction必须满足"
            "0<=lower<upper<=1。"
        )
    tracking["lower_profile_fraction"] = tracking_lower
    tracking["upper_profile_fraction"] = tracking_upper

    negative = config["negative_tail"]
    negative["enabled"] = bool(negative["enabled"])
    negative["weight"] = _require_finite_nonnegative(
        negative["weight"], "negative_tail.weight"
    )
    negative["transition_fraction"] = _require_finite_positive(
        negative["transition_fraction"],
        "negative_tail.transition_fraction",
    )
    minimum_timestep_weight = float(negative["minimum_timestep_weight"])
    if not 0.0 <= minimum_timestep_weight <= 1.0:
        raise ValueError(
            "negative_tail.minimum_timestep_weight必须在[0,1]内。"
        )
    negative["minimum_timestep_weight"] = minimum_timestep_weight
    mean_weight = _require_finite_nonnegative(
        negative["mean_weight"], "negative_tail.mean_weight"
    )
    topk_weight = _require_finite_nonnegative(
        negative["topk_weight"], "negative_tail.topk_weight"
    )
    if mean_weight + topk_weight <= epsilon:
        raise ValueError("negative_tail的mean_weight和topk_weight不能同时为0。")
    negative["mean_weight"] = mean_weight
    negative["topk_weight"] = topk_weight
    topk_fraction = float(negative["topk_fraction"])
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("negative_tail.topk_fraction必须在(0,1]内。")
    negative["topk_fraction"] = topk_fraction

    wing = config["wing_dispersion"]
    wing["enabled"] = bool(wing["enabled"])
    wing["weight"] = _require_finite_nonnegative(
        wing["weight"], "wing_dispersion.weight"
    )
    lower_profile = float(wing["lower_profile_fraction"])
    upper_profile = float(wing["upper_profile_fraction"])
    if not 0.0 <= lower_profile < upper_profile <= 1.0:
        raise ValueError(
            "wing_dispersion的profile_fraction必须满足"
            "0<=lower<upper<=1。"
        )
    wing["lower_profile_fraction"] = lower_profile
    wing["upper_profile_fraction"] = upper_profile
    wing["minimum_reference_std"] = _require_finite_positive(
        wing["minimum_reference_std"],
        "wing_dispersion.minimum_reference_std",
    )
    _validate_ratio_band(wing, "wing_dispersion")

    shift = config["shift_dispersion"]
    shift["enabled"] = bool(shift["enabled"])
    shift["weight"] = _require_finite_nonnegative(
        shift["weight"], "shift_dispersion.weight"
    )
    shift["minimum_reference_std_cm1"] = _require_finite_positive(
        shift["minimum_reference_std_cm1"],
        "shift_dispersion.minimum_reference_std_cm1",
    )
    _validate_ratio_band(shift, "shift_dispersion")

    width = config["width_dispersion"]
    width["enabled"] = bool(width["enabled"])
    width["weight"] = _require_finite_nonnegative(
        width["weight"], "width_dispersion.weight"
    )
    width["minimum_reference_std_cm1"] = _require_finite_positive(
        width["minimum_reference_std_cm1"],
        "width_dispersion.minimum_reference_std_cm1",
    )
    _validate_ratio_band(width, "width_dispersion")

    height = config["height_cv_upper"]
    height["enabled"] = bool(height["enabled"])
    height["weight"] = _require_finite_nonnegative(
        height["weight"], "height_cv_upper.weight"
    )
    height["minimum_reference_cv"] = _require_finite_positive(
        height["minimum_reference_cv"],
        "height_cv_upper.minimum_reference_cv",
    )
    upper_ratio = _require_finite_positive(
        height["upper_ratio"], "height_cv_upper.upper_ratio"
    )
    if upper_ratio < 1.0:
        raise ValueError("height_cv_upper.upper_ratio不能小于1。")
    height["upper_ratio"] = upper_ratio

    if config["enabled"]:
        active_weight = (
            (negative["weight"] if negative["enabled"] else 0.0)
            + (tracking["weight"] if tracking["enabled"] else 0.0)
            + (wing["weight"] if wing["enabled"] else 0.0)
            + (shift["weight"] if shift["enabled"] else 0.0)
            + (width["weight"] if width["enabled"] else 0.0)
            + (height["weight"] if height["enabled"] else 0.0)
        )
        if active_weight <= epsilon:
            raise ValueError(
                "local_peak_distribution_constraints已启用，"
                "但所有有效分项权重都为0。"
            )

    return config


class DifferentiableSersLocalPeakDistributionLoss(nn.Module):
    """低噪声稳定峰分布匹配 + 单边深负峰保护。"""

    def __init__(
        self,
        *,
        configuration: dict[str, Any],
        physics_constraint_state: dict[str, Any],
        prior_residual_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        super().__init__()
        self.configuration = normalize_local_peak_distribution_configuration(
            configuration
        )
        if not bool(self.configuration["enabled"]):
            raise ValueError("局部峰分布模块配置没有启用。")

        self.epsilon = float(self.configuration["epsilon"])
        self.numerical_safety_limit = float(
            self.configuration["numerical_safety_limit"]
        )
        self.minimum_batch_samples = int(
            self.configuration["minimum_batch_samples"]
        )
        self.padded_length = int(padded_length)
        if self.padded_length <= 0:
            raise ValueError("padded_length必须大于0。")

        self._load_physics_state(physics_constraint_state)
        self._load_prior_state(prior_residual_state)
        self._load_configuration_values()

    def _load_physics_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("physics_constraint_state必须是字典。")
        if not bool(state.get("enabled", False)):
            raise ValueError("局部峰分布模块要求启用physics_constraint_state。")

        self.original_length = int(state["original_length"])
        if self.padded_length < self.original_length:
            raise ValueError("padded_length不能小于original_length。")

        axis = torch.as_tensor(
            state["raman_shift"], dtype=torch.float32
        ).reshape(-1)
        lower = torch.as_tensor(
            state["pointwise_lower_envelope"], dtype=torch.float32
        ).reshape(-1)
        if axis.numel() != self.original_length:
            raise ValueError("physics_constraint_state拉曼轴长度不一致。")
        if lower.numel() != self.original_length:
            raise ValueError("逐波数下包络长度不一致。")
        if not torch.isfinite(axis).all() or not torch.isfinite(lower).all():
            raise ValueError("physics_constraint_state包含NaN或无穷值。")
        if torch.any(axis[1:] <= axis[:-1]):
            raise ValueError("拉曼轴必须严格递增。")

        stable = state.get("training_peak_distribution")
        if not isinstance(stable, dict) or not bool(stable.get("enabled", False)):
            raise ValueError(
                "局部峰分布模块要求"
                "physics_constraints.training_peak_distribution.enabled=true。"
            )

        peak_indices = torch.as_tensor(
            stable["selected_peak_indices"], dtype=torch.long
        ).reshape(-1)
        position_std = torch.as_tensor(
            stable["training_position_center_std_cm1"], dtype=torch.float32
        ).reshape(-1)
        height_median = torch.as_tensor(
            stable["height_median"], dtype=torch.float32
        ).reshape(-1)
        height_std = torch.as_tensor(
            stable["height_std"], dtype=torch.float32
        ).reshape(-1)
        width_std = torch.as_tensor(
            stable["width_std_cm1"], dtype=torch.float32
        ).reshape(-1)
        peak_count = peak_indices.numel()
        if peak_count <= 0:
            raise ValueError("训练集稳定峰为空。")
        if not all(
            tensor.numel() == peak_count
            for tensor in (position_std, height_median, height_std, width_std)
        ):
            raise ValueError("稳定峰训练分布统计长度不一致。")

        stable_filter = self.configuration["stable_peak_filter"]
        keep = (
            position_std
            <= float(stable_filter["maximum_position_std_cm1"])
        ) & (
            height_median
            >= float(stable_filter["minimum_height_median"])
        )
        if not torch.any(keep):
            raise ValueError("稳定峰过滤后为空，请放宽stable_peak_filter。")
        peak_indices = peak_indices[keep]
        position_std = position_std[keep]
        height_median = height_median[keep]
        height_std = height_std[keep]
        width_std = width_std[keep]
        training_height_cv = height_std / height_median.clamp_min(self.epsilon)

        half_width = int(
            stable.get(
                "analysis_half_width_points",
                state.get("analysis_half_width_points", 0),
            )
        )
        if half_width < 2:
            raise ValueError("稳定峰分析窗口半宽至少需要2个点。")
        if torch.any(peak_indices - half_width < 0) or torch.any(
            peak_indices + half_width >= self.original_length
        ):
            raise ValueError("稳定峰窗口越过拉曼轴边界。")

        self.register_buffer("raman_shift", axis, persistent=False)
        self.register_buffer(
            "pointwise_lower_envelope",
            lower.view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "stable_peak_indices", peak_indices, persistent=False
        )
        self.register_buffer(
            "training_peak_position_std_cm1", position_std, persistent=False
        )
        self.register_buffer(
            "training_peak_width_std_cm1", width_std, persistent=False
        )
        self.register_buffer(
            "training_peak_height_cv", training_height_cv, persistent=False
        )
        self.peak_half_width_points = half_width
        offsets = torch.arange(
            -half_width,
            half_width + 1,
            dtype=torch.long,
        )
        window_indices = peak_indices[:, None] + offsets[None, :]
        self.register_buffer(
            "stable_peak_window_indices",
            window_indices,
            persistent=False,
        )
        self.register_buffer(
            "stable_peak_window_axis",
            axis[window_indices],
            persistent=False,
        )

        allowed_minimum = float(state.get("allowed_intensity_minimum", -1.0))
        allowed_maximum = float(state.get("allowed_intensity_maximum", 1.0))
        self.training_intensity_range = max(
            allowed_maximum - allowed_minimum,
            self.epsilon,
        )

    def _load_prior_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("prior_residual_state必须是字典。")
        if not bool(state.get("enabled", False)):
            raise ValueError("局部峰分布模块要求启用prior_residual。")
        if state.get("domain") != "spectrum_global_minmax_normalized":
            raise ValueError("prior_residual_state数据域不正确。")

        residual = state.get("residual_normalization")
        if not isinstance(residual, dict):
            raise ValueError("prior_residual_state缺少residual_normalization。")
        if residual.get("method") != "pointwise_mad_asinh":
            raise ValueError(
                "局部峰分布模块当前只支持pointwise_mad_asinh。"
            )

        prior = torch.as_tensor(
            state["prior_normalized_intensity"], dtype=torch.float32
        ).reshape(-1)
        scale = torch.as_tensor(
            residual["pointwise_scale"], dtype=torch.float32
        ).reshape(-1)
        if prior.numel() != self.original_length or scale.numel() != self.original_length:
            raise ValueError("D2状态长度与拉曼轴不一致。")
        if torch.any(scale <= 0.0):
            raise ValueError("D2 pointwise_scale必须全部大于0。")

        self.register_buffer("prior", prior.view(1, 1, -1), persistent=False)
        self.register_buffer(
            "pointwise_scale", scale.view(1, 1, -1), persistent=False
        )
        self.target_abs_max = float(residual["target_abs_max"])
        self.standardized_residual_scale = float(
            residual["standardized_residual_scale"]
        )
        self.asinh_normalizer = float(residual["asinh_normalizer"])
        if min(
            self.target_abs_max,
            self.standardized_residual_scale,
            self.asinh_normalizer,
        ) <= 0.0:
            raise ValueError("D2可微反变换标量参数必须大于0。")

    def _load_configuration_values(self) -> None:
        gate = self.configuration["low_noise_gate"]
        self.full_weight_max_timestep = int(
            gate["full_weight_max_timestep"]
        )
        self.zero_weight_min_timestep = int(
            gate["zero_weight_min_timestep"]
        )

        peak_window = self.configuration["peak_window"]
        self.edge_fraction = float(peak_window["edge_fraction"])
        self.softplus_temperature_fraction = float(
            peak_window["softplus_temperature_fraction"]
        )

        self.tracking_configuration = self.configuration["wing_shape_tracking"]
        self.negative_configuration = self.configuration["negative_tail"]
        self.wing_configuration = self.configuration["wing_dispersion"]
        self.shift_configuration = self.configuration["shift_dispersion"]
        self.width_configuration = self.configuration["width_dispersion"]
        self.height_configuration = self.configuration["height_cv_upper"]

    def _resolve_reference_prior(
        self,
        scaled: torch.Tensor,
        reference_prior: torch.Tensor | None,
    ) -> torch.Tensor:
        if reference_prior is None:
            return self.prior.to(scaled)
        if reference_prior.ndim != 3 or reference_prior.shape[1] != 1:
            raise ValueError("reference_prior必须为[B,1,L]。")
        if reference_prior.shape[0] != scaled.shape[0]:
            raise ValueError("reference_prior批量大小不一致。")
        if reference_prior.shape[-1] != self.padded_length:
            raise ValueError("reference_prior长度不一致。")
        return reference_prior[..., : self.original_length].to(
            device=scaled.device,
            dtype=scaled.dtype,
        )

    def inverse_scaled_residual(
        self,
        scaled: torch.Tensor,
        *,
        reference_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scaled.ndim != 3 or scaled.shape[1] != 1:
            raise ValueError("scaled必须为[B,1,L]。")
        if scaled.shape[-1] != self.padded_length:
            raise ValueError("scaled长度与padded_length不一致。")

        cropped = scaled[..., : self.original_length].float()
        raw_argument = (
            cropped / self.target_abs_max * self.asinh_normalizer
        )
        safe_argument = self.numerical_safety_limit * torch.tanh(
            raw_argument / self.numerical_safety_limit
        )
        standardized = self.standardized_residual_scale * torch.sinh(
            safe_argument
        )
        return (
            self._resolve_reference_prior(cropped, reference_prior)
            + self.pointwise_scale.to(cropped) * standardized
        )

    def _local_timestep_gate(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.to(dtype=torch.float32)
        full = float(self.full_weight_max_timestep)
        zero = float(self.zero_weight_min_timestep)
        return torch.where(
            t <= full,
            torch.ones_like(t),
            torch.where(
                t >= zero,
                torch.zeros_like(t),
                (zero - t) / (zero - full),
            ),
        )

    def _pathology_timestep_weight(
        self,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        alpha = alphas_cumprod.gather(0, timesteps).to(dtype=torch.float32)
        weight = torch.sqrt(alpha.clamp_min(0.0))
        minimum = float(self.negative_configuration["minimum_timestep_weight"])
        return weight.clamp_min(minimum)

    def _aggregate_negative_tail(self, violation: torch.Tensor) -> torch.Tensor:
        flat = violation.reshape(violation.shape[0], -1)
        mean_loss = flat.mean(dim=1)
        fraction = float(self.negative_configuration["topk_fraction"])
        k = max(1, int(round(flat.shape[1] * fraction)))
        topk_loss = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        mean_weight = float(self.negative_configuration["mean_weight"])
        topk_weight = float(self.negative_configuration["topk_weight"])
        return (
            mean_weight * mean_loss + topk_weight * topk_loss
        ) / max(mean_weight + topk_weight, self.epsilon)

    def _negative_tail_components(
        self,
        predicted: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.negative_configuration["enabled"]):
            zero = predicted.new_zeros(())
            return zero, zero

        transition = max(
            self.training_intensity_range
            * float(self.negative_configuration["transition_fraction"]),
            self.epsilon,
        )
        violation = torch.square(
            F.relu(self.pointwise_lower_envelope.to(predicted) - predicted)
            / transition
        )
        per_sample = self._aggregate_negative_tail(violation)
        raw = per_sample.mean()
        timestep_weight = self._pathology_timestep_weight(
            timesteps,
            alphas_cumprod,
        ).to(per_sample)
        weighted = (per_sample * timestep_weight).mean()
        contribution = (
            float(self.negative_configuration["weight"]) * weighted
        )
        return raw, contribution

    def _extract_windows(self, spectra: torch.Tensor) -> torch.Tensor:
        rows = spectra.squeeze(1)
        indices = self.stable_peak_window_indices.to(rows.device)
        return rows[:, indices]

    def _peak_features(
        self,
        spectra: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local = self._extract_windows(spectra)
        axis = self.stable_peak_window_axis.to(local)
        window_length = local.shape[-1]
        edge_points = max(
            2,
            min(
                window_length // 3,
                int(round(window_length * self.edge_fraction)),
            ),
        )
        left = local[..., :edge_points].mean(dim=-1, keepdim=True)
        right = local[..., -edge_points:].mean(dim=-1, keepdim=True)
        fraction = torch.linspace(
            0.0,
            1.0,
            window_length,
            device=local.device,
            dtype=local.dtype,
        ).view(1, 1, -1)
        baseline = left + (right - left) * fraction
        corrected = local - baseline

        temperature = max(
            self.training_intensity_range
            * self.softplus_temperature_fraction,
            self.epsilon,
        )
        positive = F.softplus(corrected / temperature) * temperature
        height = positive.amax(dim=-1).clamp_min(self.epsilon)
        profile = positive / height.unsqueeze(-1)

        mass = positive.clamp_min(self.epsilon)
        total_mass = mass.sum(dim=-1).clamp_min(self.epsilon)
        centroid = (mass * axis.unsqueeze(0)).sum(dim=-1) / total_mass
        centered = axis.unsqueeze(0) - centroid.unsqueeze(-1)
        width = torch.sqrt(
            (mass * centered.square()).sum(dim=-1) / total_mass
            + self.epsilon
        )
        return {
            "profile": profile,
            "height": height,
            "centroid": centroid,
            "width": width,
        }

    @staticmethod
    def _std(values: torch.Tensor) -> torch.Tensor:
        return torch.std(values, dim=0, unbiased=False)

    def _ratio_band_loss(
        self,
        predicted_std: torch.Tensor,
        reference_std: torch.Tensor,
        *,
        minimum_reference_std: float,
        lower_ratio: float,
        upper_ratio: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reference = reference_std.detach()
        valid = reference >= minimum_reference_std
        if mask is not None:
            valid = valid & mask
        if not torch.any(valid):
            return predicted_std.new_zeros(())
        ratio = predicted_std / reference.clamp_min(minimum_reference_std)
        violation = (
            F.relu(lower_ratio - ratio).square()
            + F.relu(ratio - upper_ratio).square()
        )
        return violation[valid].mean()

    def _distribution_components(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = predicted.new_zeros(())
        gate = self._local_timestep_gate(timesteps).to(predicted)
        active = gate > 0.0
        active_count = int(active.sum().item())
        result = {
            "wing_shape_tracking_loss": zero,
            "wing_dispersion_loss": zero,
            "shift_dispersion_loss": zero,
            "width_dispersion_loss": zero,
            "height_cv_upper_loss": zero,
            "wing_shape_tracking_contribution": zero,
            "wing_dispersion_contribution": zero,
            "shift_dispersion_contribution": zero,
            "width_dispersion_contribution": zero,
            "height_cv_upper_contribution": zero,
            "local_peak_active_samples": predicted.new_tensor(
                float(active_count)
            ),
            "local_peak_count": predicted.new_tensor(
                float(self.stable_peak_indices.numel())
            ),
            "mean_local_peak_timestep_weight": gate.mean(),
        }
        if active_count < self.minimum_batch_samples:
            return result

        pred_features = self._peak_features(predicted[active])
        target_features = self._peak_features(target[active])
        active_gate_mean = gate[active].mean()

        tracking = self.tracking_configuration
        if bool(tracking["enabled"]):
            target_profile = target_features["profile"].detach()
            target_mean = target_profile.mean(dim=0)
            tracking_mask = (
                (target_mean >= float(tracking["lower_profile_fraction"]))
                & (target_mean <= float(tracking["upper_profile_fraction"]))
            )
            pointwise_error = (
                pred_features["profile"] - target_profile
            ).square()
            if torch.any(tracking_mask):
                loss = pointwise_error[:, tracking_mask].mean()
            else:
                loss = zero
            result["wing_shape_tracking_loss"] = loss
            result["wing_shape_tracking_contribution"] = (
                float(tracking["weight"]) * active_gate_mean * loss
            )

        wing = self.wing_configuration
        if bool(wing["enabled"]):
            target_profile = target_features["profile"].detach()
            pred_std = self._std(pred_features["profile"])
            target_std = self._std(target_profile)
            target_mean = target_profile.mean(dim=0)
            wing_mask = (
                (target_mean >= float(wing["lower_profile_fraction"]))
                & (target_mean <= float(wing["upper_profile_fraction"]))
            )
            loss = self._ratio_band_loss(
                pred_std,
                target_std,
                minimum_reference_std=float(
                    wing["minimum_reference_std"]
                ),
                lower_ratio=float(wing["lower_std_ratio"]),
                upper_ratio=float(wing["upper_std_ratio"]),
                mask=wing_mask,
            )
            result["wing_dispersion_loss"] = loss
            result["wing_dispersion_contribution"] = (
                float(wing["weight"]) * active_gate_mean * loss
            )

        shift = self.shift_configuration
        if bool(shift["enabled"]):
            loss = self._ratio_band_loss(
                self._std(pred_features["centroid"]),
                self.training_peak_position_std_cm1.to(predicted),
                minimum_reference_std=float(
                    shift["minimum_reference_std_cm1"]
                ),
                lower_ratio=float(shift["lower_std_ratio"]),
                upper_ratio=float(shift["upper_std_ratio"]),
            )
            result["shift_dispersion_loss"] = loss
            result["shift_dispersion_contribution"] = (
                float(shift["weight"]) * active_gate_mean * loss
            )

        width = self.width_configuration
        if bool(width["enabled"]):
            loss = self._ratio_band_loss(
                self._std(pred_features["width"]),
                self.training_peak_width_std_cm1.to(predicted),
                minimum_reference_std=float(
                    width["minimum_reference_std_cm1"]
                ),
                lower_ratio=float(width["lower_std_ratio"]),
                upper_ratio=float(width["upper_std_ratio"]),
            )
            result["width_dispersion_loss"] = loss
            result["width_dispersion_contribution"] = (
                float(width["weight"]) * active_gate_mean * loss
            )

        height = self.height_configuration
        if bool(height["enabled"]):
            pred_height = pred_features["height"]
            pred_cv = self._std(pred_height) / pred_height.mean(dim=0).clamp_min(
                self.epsilon
            )
            reference_cv = self.training_peak_height_cv.to(predicted).clamp_min(
                float(height["minimum_reference_cv"])
            )
            ratio = pred_cv / reference_cv
            loss = F.relu(ratio - float(height["upper_ratio"])).square().mean()
            result["height_cv_upper_loss"] = loss
            result["height_cv_upper_contribution"] = (
                float(height["weight"]) * active_gate_mean * loss
            )

        return result

    def forward(
        self,
        *,
        predicted_scaled_residual: torch.Tensor,
        target_scaled_residual: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        reference_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        predicted = self.inverse_scaled_residual(
            predicted_scaled_residual,
            reference_prior=reference_prior,
        )
        target = self.inverse_scaled_residual(
            target_scaled_residual,
            reference_prior=reference_prior,
        )

        negative_raw, negative_contribution = self._negative_tail_components(
            predicted,
            timesteps,
            alphas_cumprod,
        )
        distribution = self._distribution_components(
            predicted,
            target,
            timesteps,
        )
        total = negative_contribution
        for name in (
            "wing_shape_tracking_contribution",
            "wing_dispersion_contribution",
            "shift_dispersion_contribution",
            "width_dispersion_contribution",
            "height_cv_upper_contribution",
        ):
            total = total + distribution[name]

        return {
            "local_peak_distribution_loss": total,
            "negative_tail_guard_loss": negative_raw,
            "negative_tail_guard_contribution": negative_contribution,
            **distribution,
        }


__all__ = [
    "DEFAULT_LOCAL_PEAK_DISTRIBUTION_CONFIGURATION",
    "DifferentiableSersLocalPeakDistributionLoss",
    "normalize_local_peak_distribution_configuration",
]
"""D3.2训练集自动峰区相对峰强软约束。

该模块是D2.6 broad-local residual扩散的训练阶段兼容层。DDPM仍然
预测``scaled local residual``；约束计算前先反变换local residual，
再加回逐样本``outer PCA prior + true broad residual``，恢复完整归一化
光谱。

峰区只从training split自动检测，不允许配置固定峰位。约束目标是主要
自动峰之间的相对峰强关系，而不是把绝对峰高压到固定值；它只作为
训练soft loss，不改变U-Net结构、DDIM采样或生成后光谱。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torch import nn
from torch.nn import functional as F


RELATIVE_PEAK_INTENSITY_STATE_SCHEMA_VERSION = 1
RELATIVE_PEAK_INTENSITY_METHOD_VERSION = (
    "d3_2_train_auto_relative_peak_intensity_v1"
)


_DEFAULT_CONFIGURATION: dict[str, Any] = {
    "enabled": False,
    "reference_baseline_sigma_cm1": 30.0,
    "reference_peak_smoothing_sigma_cm1": 2.0,
    "minimum_relative_prominence": 0.06,
    "minimum_peak_distance_cm1": 12.0,
    "maximum_peak_count": 10,
    "minimum_peak_count": 2,
    "edge_exclusion_cm1": 24.0,
    "peak_half_width_cm1": 12.0,
    "baseline_inner_half_width_cm1": 18.0,
    "baseline_outer_half_width_cm1": 30.0,
    "minimum_relative_height_fraction": 0.03,
    "log_ratio_margin_iqr": 1.5,
    "minimum_log_ratio_width": 0.08,
    "smooth_l1_beta": 0.25,
    "timestep_weight_power": 1.0,
    "minimum_timestep_weight": 0.0,
    "total_weight": 1.0,
    "maximum_total_ratio_to_ddpm": 0.05,
    "numerical_safety_limit": 12.0,
    "epsilon": 1.0e-8,
}


_FORBIDDEN_FIXED_PEAK_KEYS = {
    "peak_centers",
    "peak_centers_cm1",
    "peak_positions",
    "peak_windows",
    "fixed_peak_positions",
    "reference_position_cm1",
}


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name}必须是有限数值。")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name}必须大于0。")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name}不能小于0。")
    return result


def normalize_relative_peak_intensity_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """补齐并校验D3.2配置，禁止任何人工固定峰位。"""

    raw = configuration or {"enabled": False}
    if not isinstance(raw, dict):
        raise TypeError(
            "relative_peak_intensity_constraints必须是字典。"
        )

    forbidden = sorted(_FORBIDDEN_FIXED_PEAK_KEYS.intersection(raw))
    if forbidden:
        raise ValueError(
            "D3.2峰区必须仅从训练集自动拟合，禁止配置固定峰位："
            f"{forbidden}。"
        )

    config = deepcopy(_DEFAULT_CONFIGURATION)
    config.update(raw)
    config["enabled"] = bool(config["enabled"])

    positive_names = (
        "reference_baseline_sigma_cm1",
        "reference_peak_smoothing_sigma_cm1",
        "minimum_peak_distance_cm1",
        "peak_half_width_cm1",
        "baseline_inner_half_width_cm1",
        "baseline_outer_half_width_cm1",
        "minimum_relative_height_fraction",
        "log_ratio_margin_iqr",
        "minimum_log_ratio_width",
        "smooth_l1_beta",
        "timestep_weight_power",
        "total_weight",
        "numerical_safety_limit",
        "epsilon",
    )
    for name in positive_names:
        config[name] = _positive_float(
            config[name],
            f"relative_peak_intensity_constraints.{name}",
        )

    if not (
        config["peak_half_width_cm1"]
        < config["baseline_inner_half_width_cm1"]
        < config["baseline_outer_half_width_cm1"]
    ):
        raise ValueError(
            "D3.2要求peak_half_width_cm1 < "
            "baseline_inner_half_width_cm1 < "
            "baseline_outer_half_width_cm1。"
        )

    config["edge_exclusion_cm1"] = _nonnegative_float(
        config["edge_exclusion_cm1"],
        "relative_peak_intensity_constraints.edge_exclusion_cm1",
    )
    config["minimum_timestep_weight"] = _nonnegative_float(
        config["minimum_timestep_weight"],
        "relative_peak_intensity_constraints.minimum_timestep_weight",
    )
    if config["minimum_timestep_weight"] > 1.0:
        raise ValueError(
            "relative_peak_intensity_constraints.minimum_timestep_weight"
            "不能大于1。"
        )

    prominence = _finite_float(
        config["minimum_relative_prominence"],
        "relative_peak_intensity_constraints.minimum_relative_prominence",
    )
    if not 0.0 < prominence < 1.0:
        raise ValueError(
            "relative_peak_intensity_constraints."
            "minimum_relative_prominence必须在(0,1)内。"
        )
    config["minimum_relative_prominence"] = prominence

    maximum_ratio = _finite_float(
        config["maximum_total_ratio_to_ddpm"],
        "relative_peak_intensity_constraints.maximum_total_ratio_to_ddpm",
    )
    if not 0.0 < maximum_ratio <= 1.0:
        raise ValueError(
            "relative_peak_intensity_constraints.maximum_total_ratio_to_ddpm"
            "必须在(0,1]内。"
        )
    config["maximum_total_ratio_to_ddpm"] = maximum_ratio

    maximum_peak_count = int(config["maximum_peak_count"])
    minimum_peak_count = int(config["minimum_peak_count"])
    if minimum_peak_count < 2:
        raise ValueError(
            "relative_peak_intensity_constraints.minimum_peak_count"
            "至少为2。"
        )
    if maximum_peak_count < minimum_peak_count:
        raise ValueError(
            "relative_peak_intensity_constraints.maximum_peak_count"
            "不能小于minimum_peak_count。"
        )
    config["maximum_peak_count"] = maximum_peak_count
    config["minimum_peak_count"] = minimum_peak_count

    return config


def _validate_axis(
    raman_shift: np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
    if axis.size < 3:
        raise ValueError("D3.2 Raman shift轴至少需要3个点。")
    if expected_length is not None and axis.size != expected_length:
        raise ValueError(
            f"D3.2 Raman轴长度{axis.size}与光谱长度"
            f"{expected_length}不一致。"
        )
    if not np.isfinite(axis).all():
        raise ValueError("D3.2 Raman shift轴包含NaN或无穷值。")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("D3.2 Raman shift轴必须严格递增。")
    return axis


def _validate_spectra(
    spectra: np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("D3.2训练光谱必须是非空二维数组[N,L]。")
    if expected_length is not None and values.shape[1] != expected_length:
        raise ValueError("D3.2训练光谱长度与Raman轴不一致。")
    if not np.isfinite(values).all():
        raise ValueError("D3.2训练光谱包含NaN或无穷值。")
    return values


def _detect_training_peaks(
    *,
    spectra: np.ndarray,
    axis: np.ndarray,
    configuration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = float(np.median(np.diff(axis)))
    reference = np.median(spectra, axis=0)
    baseline = gaussian_filter1d(
        reference,
        sigma=float(configuration["reference_baseline_sigma_cm1"]) / spacing,
        mode="nearest",
        truncate=4.0,
    )
    peak_signal = gaussian_filter1d(
        reference - baseline,
        sigma=(
            float(configuration["reference_peak_smoothing_sigma_cm1"])
            / spacing
        ),
        mode="nearest",
        truncate=4.0,
    )

    maximum_positive_signal = float(np.max(peak_signal))
    if maximum_positive_signal <= float(configuration["epsilon"]):
        raise ValueError(
            "D3.2无法从训练集参考谱检测正特征峰："
            "去基线后的最大正信号不大于0。"
        )

    detected, properties = find_peaks(
        peak_signal,
        prominence=(
            float(configuration["minimum_relative_prominence"])
            * maximum_positive_signal
        ),
        distance=max(
            1,
            int(
                round(
                    float(configuration["minimum_peak_distance_cm1"])
                    / spacing
                )
            ),
        ),
    )

    edge = float(configuration["edge_exclusion_cm1"])
    inside = (
        (axis[detected] >= axis[0] + edge)
        & (axis[detected] <= axis[-1] - edge)
    )
    detected = detected[inside]
    prominences = np.asarray(
        properties.get("prominences", np.zeros(inside.size)),
        dtype=np.float64,
    )[inside]

    if detected.size < int(configuration["minimum_peak_count"]):
        raise ValueError(
            "D3.2在边缘排除后检测到的训练集特征峰数量不足；"
            "相对峰强约束至少需要2个自动峰。"
        )

    maximum_peak_count = int(configuration["maximum_peak_count"])
    if detected.size > maximum_peak_count:
        keep = np.argsort(prominences)[-maximum_peak_count:]
        detected = detected[keep]
        prominences = prominences[keep]

    order = np.argsort(detected)
    detected = detected[order]
    prominences = prominences[order]
    centers = axis[detected]
    return detected.astype(np.int64), centers, prominences


def _window_mask(
    axis: np.ndarray,
    center: float,
    *,
    lower_cm1: float,
    upper_cm1: float,
) -> np.ndarray:
    distance = np.abs(axis - float(center))
    return (
        (distance >= float(lower_cm1))
        & (distance <= float(upper_cm1))
    )


def _measure_numpy_peak_heights(
    *,
    spectra: np.ndarray,
    axis: np.ndarray,
    centers: np.ndarray,
    configuration: dict[str, Any],
) -> np.ndarray:
    heights: list[np.ndarray] = []
    for center in centers:
        peak_mask = _window_mask(
            axis,
            float(center),
            lower_cm1=0.0,
            upper_cm1=float(configuration["peak_half_width_cm1"]),
        )
        baseline_mask = _window_mask(
            axis,
            float(center),
            lower_cm1=float(configuration["baseline_inner_half_width_cm1"]),
            upper_cm1=float(configuration["baseline_outer_half_width_cm1"]),
        )
        if not np.any(peak_mask) or not np.any(baseline_mask):
            raise ValueError(
                "D3.2自动峰区或侧翼基线窗口为空，请检查窗口参数。"
            )
        peak_values = np.max(spectra[:, peak_mask], axis=1)
        baseline_values = np.median(spectra[:, baseline_mask], axis=1)
        heights.append(peak_values - baseline_values)

    return np.stack(heights, axis=1)


def _fit_log_relative_bounds(
    *,
    heights: np.ndarray,
    configuration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    median_heights = np.median(heights, axis=0)
    maximum_median_height = float(np.max(median_heights))
    floor = (
        float(configuration["minimum_relative_height_fraction"])
        * max(maximum_median_height, float(configuration["epsilon"]))
    )
    clipped_heights = np.maximum(heights, floor)
    relative = (
        clipped_heights
        / np.sum(clipped_heights, axis=1, keepdims=True)
    )
    log_relative = np.log(
        np.maximum(relative, float(configuration["epsilon"]))
    )

    q25 = np.percentile(log_relative, 25.0, axis=0)
    q50 = np.percentile(log_relative, 50.0, axis=0)
    q75 = np.percentile(log_relative, 75.0, axis=0)
    iqr = q75 - q25
    margin = float(configuration["log_ratio_margin_iqr"]) * iqr
    lower = q25 - margin
    upper = q75 + margin

    minimum_width = float(configuration["minimum_log_ratio_width"])
    too_narrow = (upper - lower) < minimum_width
    lower[too_narrow] = q50[too_narrow] - 0.5 * minimum_width
    upper[too_narrow] = q50[too_narrow] + 0.5 * minimum_width
    return q50, lower, upper, relative, float(floor)


def fit_relative_peak_intensity_constraint_state(
    *,
    training_normalized_spectra: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """仅使用training split拟合自动峰区和相对峰强分布边界。"""

    config = normalize_relative_peak_intensity_configuration(configuration)
    if not config["enabled"]:
        return {
            "schema_version": RELATIVE_PEAK_INTENSITY_STATE_SCHEMA_VERSION,
            "method_version": RELATIVE_PEAK_INTENSITY_METHOD_VERSION,
            "enabled": False,
            "contains_fixed_peak_positions": False,
            "configuration": config,
        }

    spectra = _validate_spectra(training_normalized_spectra)
    axis = _validate_axis(raman_shift, expected_length=spectra.shape[1])
    spacing = float(np.median(np.diff(axis)))

    detected, centers, prominences = _detect_training_peaks(
        spectra=spectra,
        axis=axis,
        configuration=config,
    )
    heights = _measure_numpy_peak_heights(
        spectra=spectra,
        axis=axis,
        centers=centers,
        configuration=config,
    )
    (
        median_log_relative,
        lower,
        upper,
        relative,
        minimum_peak_height_floor,
    ) = _fit_log_relative_bounds(
        heights=heights,
        configuration=config,
    )

    peak_masks = []
    baseline_masks = []
    for center in centers:
        peak_masks.append(
            _window_mask(
                axis,
                float(center),
                lower_cm1=0.0,
                upper_cm1=float(config["peak_half_width_cm1"]),
            ).astype(np.float32)
        )
        baseline_masks.append(
            _window_mask(
                axis,
                float(center),
                lower_cm1=float(config["baseline_inner_half_width_cm1"]),
                upper_cm1=float(config["baseline_outer_half_width_cm1"]),
            ).astype(np.float32)
        )

    return {
        "schema_version": RELATIVE_PEAK_INTENSITY_STATE_SCHEMA_VERSION,
        "method_version": RELATIVE_PEAK_INTENSITY_METHOD_VERSION,
        "enabled": True,
        "contains_fixed_peak_positions": False,
        "fit_on": "train_only",
        "data_domain": "full_normalized_spectrum",
        "prediction_domain": "d2_6_scaled_local_residual",
        "configuration": config,
        "original_length": int(axis.size),
        "raman_shift": axis.tolist(),
        "raman_spacing_cm1": spacing,
        "detected_peak_indices": detected.tolist(),
        "detected_peak_centers_cm1": centers.tolist(),
        "detected_peak_prominences": prominences.tolist(),
        "peak_masks": [mask.tolist() for mask in peak_masks],
        "baseline_masks": [mask.tolist() for mask in baseline_masks],
        "training_peak_heights": heights.tolist(),
        "minimum_peak_height_floor": minimum_peak_height_floor,
        "training_relative_peak_intensities": relative.tolist(),
        "median_log_relative_peak_intensity": median_log_relative.tolist(),
        "lower_log_relative_peak_intensity": lower.tolist(),
        "upper_log_relative_peak_intensity": upper.tolist(),
        "number_of_training_spectra": int(spectra.shape[0]),
    }


class DifferentiableRelativePeakIntensityLoss(nn.Module):
    """在D2.6恢复后的完整光谱域计算相对峰强软边界损失。"""

    def __init__(
        self,
        *,
        relative_peak_intensity_constraint_state: dict[str, Any],
        broad_local_residual_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        super().__init__()

        state = relative_peak_intensity_constraint_state
        if not isinstance(state, dict) or not bool(state.get("enabled", False)):
            raise ValueError("D3.2 relative peak intensity state没有启用。")
        if int(state.get("schema_version", 0)) != (
            RELATIVE_PEAK_INTENSITY_STATE_SCHEMA_VERSION
        ):
            raise ValueError("D3.2 relative peak intensity state版本不受支持。")
        if bool(state.get("contains_fixed_peak_positions", True)):
            raise ValueError("D3.2 state不能包含人工固定峰位。")

        self.configuration = normalize_relative_peak_intensity_configuration(
            state.get("configuration")
        )
        self.original_length = int(state["original_length"])
        self.padded_length = int(padded_length)
        if self.padded_length < self.original_length:
            raise ValueError("D3.2 padded_length不能小于原始光谱长度。")

        _validate_axis(
            np.asarray(state["raman_shift"], dtype=np.float64),
            expected_length=self.original_length,
        )
        peak_masks = np.asarray(state["peak_masks"], dtype=np.float32)
        baseline_masks = np.asarray(state["baseline_masks"], dtype=np.float32)
        if peak_masks.ndim != 2 or peak_masks.shape[1] != self.original_length:
            raise ValueError("D3.2 peak_masks形状不正确。")
        if baseline_masks.shape != peak_masks.shape:
            raise ValueError("D3.2 baseline_masks形状不正确。")
        if peak_masks.shape[0] < 2:
            raise ValueError("D3.2至少需要2个自动峰。")

        local_state = broad_local_residual_state.get("local_normalization")
        if not isinstance(local_state, dict):
            raise ValueError(
                "D3.2需要有效的D2.6 broad_local_residual_state。"
            )
        if str(local_state.get("method", "")).lower() != "robust_asinh":
            raise ValueError("D3.2当前只支持D2.6 local robust_asinh。")

        self.local_target_abs_max = _positive_float(
            local_state["target_abs_max"],
            "broad_local_residual_state.local target_abs_max",
        )
        self.local_residual_scale = _positive_float(
            local_state["scale"],
            "broad_local_residual_state.local scale",
        )
        self.local_asinh_normalizer = _positive_float(
            local_state["asinh_normalizer"],
            "broad_local_residual_state.local asinh_normalizer",
        )
        self.minimum_peak_height_floor = _positive_float(
            state["minimum_peak_height_floor"],
            "relative_peak_intensity_state.minimum_peak_height_floor",
        )

        lower = np.asarray(
            state["lower_log_relative_peak_intensity"],
            dtype=np.float32,
        ).reshape(-1)
        upper = np.asarray(
            state["upper_log_relative_peak_intensity"],
            dtype=np.float32,
        ).reshape(-1)
        if lower.shape[0] != peak_masks.shape[0] or upper.shape != lower.shape:
            raise ValueError("D3.2 log relative边界长度不正确。")

        self.register_buffer(
            "peak_masks",
            torch.from_numpy(peak_masks).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "baseline_masks",
            torch.from_numpy(baseline_masks).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "lower_log_relative",
            torch.from_numpy(lower).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "upper_log_relative",
            torch.from_numpy(upper).view(1, -1),
            persistent=False,
        )

    def _inverse_local_transform(self, values: torch.Tensor) -> torch.Tensor:
        argument = (
            values
            / self.local_target_abs_max
            * self.local_asinh_normalizer
        )
        limit = float(self.configuration["numerical_safety_limit"])
        safe_argument = torch.clamp(argument, min=-limit, max=limit)
        return self.local_residual_scale * torch.sinh(safe_argument)

    def _relative_peak_intensity(
        self,
        spectra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = spectra.squeeze(1)
        peak_masks = self.peak_masks.to(values)
        baseline_masks = self.baseline_masks.to(values)
        masked_peak = values.unsqueeze(1).masked_fill(
            peak_masks <= 0.0,
            torch.finfo(values.dtype).min,
        )
        peak_values = torch.amax(masked_peak, dim=-1)
        baseline_values = (
            values.unsqueeze(1) * baseline_masks
        ).sum(dim=-1) / baseline_masks.sum(dim=-1).clamp_min(
            float(self.configuration["epsilon"])
        )
        raw_heights = peak_values - baseline_values
        positive_heights = torch.clamp(
            raw_heights,
            min=float(self.minimum_peak_height_floor),
        )
        relative = positive_heights / positive_heights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(float(self.configuration["epsilon"]))
        log_relative = torch.log(
            relative.clamp_min(float(self.configuration["epsilon"]))
        )
        return raw_heights, log_relative

    def forward(
        self,
        *,
        predicted_scaled_local_residual: torch.Tensor,
        target_scaled_local_residual: torch.Tensor,
        reconstruction_base: torch.Tensor | None,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if reconstruction_base is None:
            raise ValueError(
                "D3.2缺少逐样本reconstruction_base："
                "outer PCA prior + true broad residual。"
            )
        if not (
            predicted_scaled_local_residual.shape
            == target_scaled_local_residual.shape
            == reconstruction_base.shape
        ):
            raise ValueError("D3.2预测、目标和reconstruction_base形状必须一致。")

        predicted_scaled = predicted_scaled_local_residual[
            ..., : self.original_length
        ]
        target_scaled = target_scaled_local_residual[
            ..., : self.original_length
        ]
        base = reconstruction_base[..., : self.original_length]

        predicted_full = base + self._inverse_local_transform(predicted_scaled)
        target_full = base + self._inverse_local_transform(target_scaled)

        predicted_heights, predicted_log_relative = (
            self._relative_peak_intensity(predicted_full)
        )
        target_heights, target_log_relative = (
            self._relative_peak_intensity(target_full)
        )

        lower = self.lower_log_relative.to(predicted_log_relative)
        upper = self.upper_log_relative.to(predicted_log_relative)
        below = F.relu(lower - predicted_log_relative)
        above = F.relu(predicted_log_relative - upper)
        boundary_error = below + above
        boundary_pointwise = F.smooth_l1_loss(
            boundary_error,
            torch.zeros_like(boundary_error),
            reduction="none",
            beta=float(self.configuration["smooth_l1_beta"]),
        )

        target_difference = predicted_log_relative - target_log_relative
        target_pointwise = F.smooth_l1_loss(
            target_difference,
            torch.zeros_like(target_difference),
            reduction="none",
            beta=float(self.configuration["smooth_l1_beta"]),
        )

        per_sample = boundary_pointwise.mean(dim=1)
        target_per_sample = target_pointwise.mean(dim=1)

        alpha_bar = alphas_cumprod.gather(-1, timesteps).to(per_sample)
        timestep_weight = torch.pow(
            alpha_bar.clamp(0.0, 1.0),
            float(self.configuration["timestep_weight_power"]),
        ).clamp_min(float(self.configuration["minimum_timestep_weight"]))
        weighted_per_sample = per_sample * timestep_weight

        return {
            "relative_peak_intensity_loss": weighted_per_sample.mean(),
            "relative_peak_intensity_boundary_loss": per_sample.mean(),
            "relative_peak_intensity_target_loss": target_per_sample.mean(),
            "mean_relative_peak_intensity_timestep_weight": (
                timestep_weight.mean()
            ),
            "mean_relative_peak_intensity_boundary_error": (
                boundary_error.mean()
            ),
            "mean_relative_peak_intensity_target_error": (
                torch.abs(target_difference).mean()
            ),
            "mean_relative_peak_height": predicted_heights.mean(),
        }

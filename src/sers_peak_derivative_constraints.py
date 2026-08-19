"""D3.1训练集自动峰区一阶导数软约束。

该模块专门适配D2.6 broad-local residual数据域：DDPM预测的是
``scaled local residual``，损失计算前先使用checkpoint中的D2.6
local robust-asinh状态反变换，再加回逐样本的
``outer PCA prior + true broad residual``，恢复完整归一化光谱。

峰区只根据training split自动拟合，不允许配置固定峰位。该约束仅用于
训练损失，不改变U-Net结构、DDPM/DDIM采样过程或生成后光谱。
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


PEAK_DERIVATIVE_STATE_SCHEMA_VERSION = 1
PEAK_DERIVATIVE_METHOD_VERSION = (
    "d3_1_train_auto_peak_first_derivative_v1"
)


_DEFAULT_CONFIGURATION: dict[str, Any] = {
    "enabled": False,
    "reference_baseline_sigma_cm1": 30.0,
    "reference_peak_smoothing_sigma_cm1": 2.0,
    "minimum_relative_prominence": 0.06,
    "minimum_peak_distance_cm1": 12.0,
    "maximum_peak_count": 10,
    "edge_exclusion_cm1": 24.0,
    "peak_core_half_width_cm1": 18.0,
    "transition_width_cm1": 6.0,
    "derivative_scale_quantile": 95.0,
    "minimum_derivative_scale": 1.0e-4,
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


def normalize_peak_derivative_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """补齐并校验D3.1配置，禁止任何人工固定峰位。"""

    raw = configuration or {"enabled": False}
    if not isinstance(raw, dict):
        raise TypeError("peak_derivative_constraints必须是字典。")

    forbidden = sorted(_FORBIDDEN_FIXED_PEAK_KEYS.intersection(raw))
    if forbidden:
        raise ValueError(
            "D3.1峰区必须仅从训练集自动拟合，禁止配置固定峰位："
            f"{forbidden}。"
        )

    config = deepcopy(_DEFAULT_CONFIGURATION)
    config.update(raw)
    config["enabled"] = bool(config["enabled"])

    positive_names = (
        "reference_baseline_sigma_cm1",
        "reference_peak_smoothing_sigma_cm1",
        "minimum_peak_distance_cm1",
        "peak_core_half_width_cm1",
        "transition_width_cm1",
        "minimum_derivative_scale",
        "smooth_l1_beta",
        "timestep_weight_power",
        "total_weight",
        "numerical_safety_limit",
        "epsilon",
    )
    for name in positive_names:
        config[name] = _positive_float(
            config[name],
            f"peak_derivative_constraints.{name}",
        )

    config["edge_exclusion_cm1"] = _nonnegative_float(
        config["edge_exclusion_cm1"],
        "peak_derivative_constraints.edge_exclusion_cm1",
    )
    config["minimum_timestep_weight"] = _nonnegative_float(
        config["minimum_timestep_weight"],
        "peak_derivative_constraints.minimum_timestep_weight",
    )
    if config["minimum_timestep_weight"] > 1.0:
        raise ValueError(
            "peak_derivative_constraints.minimum_timestep_weight"
            "不能大于1。"
        )

    prominence = _finite_float(
        config["minimum_relative_prominence"],
        "peak_derivative_constraints.minimum_relative_prominence",
    )
    if not 0.0 < prominence < 1.0:
        raise ValueError(
            "peak_derivative_constraints.minimum_relative_prominence"
            "必须在(0,1)内。"
        )
    config["minimum_relative_prominence"] = prominence

    derivative_quantile = _finite_float(
        config["derivative_scale_quantile"],
        "peak_derivative_constraints.derivative_scale_quantile",
    )
    if not 0.0 < derivative_quantile < 100.0:
        raise ValueError(
            "peak_derivative_constraints.derivative_scale_quantile"
            "必须在(0,100)内。"
        )
    config["derivative_scale_quantile"] = derivative_quantile

    maximum_ratio = _finite_float(
        config["maximum_total_ratio_to_ddpm"],
        "peak_derivative_constraints.maximum_total_ratio_to_ddpm",
    )
    if not 0.0 < maximum_ratio <= 1.0:
        raise ValueError(
            "peak_derivative_constraints.maximum_total_ratio_to_ddpm"
            "必须在(0,1]内。"
        )
    config["maximum_total_ratio_to_ddpm"] = maximum_ratio

    maximum_peak_count = int(config["maximum_peak_count"])
    if maximum_peak_count <= 0:
        raise ValueError(
            "peak_derivative_constraints.maximum_peak_count必须大于0。"
        )
    config["maximum_peak_count"] = maximum_peak_count

    return config


def _validate_axis(
    raman_shift: np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
    if axis.size < 3:
        raise ValueError("D3.1 Raman shift轴至少需要3个点。")
    if expected_length is not None and axis.size != expected_length:
        raise ValueError(
            f"D3.1 Raman轴长度{axis.size}与光谱长度"
            f"{expected_length}不一致。"
        )
    if not np.isfinite(axis).all():
        raise ValueError("D3.1 Raman shift轴包含NaN或无穷值。")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("D3.1 Raman shift轴必须严格递增。")
    return axis


def _validate_spectra(
    spectra: np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("D3.1训练光谱必须是非空二维数组[N,L]。")
    if expected_length is not None and values.shape[1] != expected_length:
        raise ValueError("D3.1训练光谱长度与Raman轴不一致。")
    if not np.isfinite(values).all():
        raise ValueError("D3.1训练光谱包含NaN或无穷值。")
    return values


def _build_soft_peak_mask(
    axis: np.ndarray,
    peak_centers: np.ndarray,
    *,
    core_half_width_cm1: float,
    transition_width_cm1: float,
) -> np.ndarray:
    mask = np.zeros(axis.size, dtype=np.float64)
    for center in peak_centers:
        distance = np.abs(axis - float(center))
        local = np.zeros_like(distance)
        local[distance <= core_half_width_cm1] = 1.0
        transition = (
            (distance > core_half_width_cm1)
            & (
                distance
                < core_half_width_cm1 + transition_width_cm1
            )
        )
        phase = (
            distance[transition] - core_half_width_cm1
        ) / transition_width_cm1
        local[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
        mask = np.maximum(mask, local)
    return mask


def fit_peak_derivative_constraint_state(
    *,
    training_normalized_spectra: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """仅使用training split拟合自动峰区和导数尺度。"""

    config = normalize_peak_derivative_configuration(configuration)
    if not config["enabled"]:
        return {
            "schema_version": PEAK_DERIVATIVE_STATE_SCHEMA_VERSION,
            "method_version": PEAK_DERIVATIVE_METHOD_VERSION,
            "enabled": False,
            "contains_fixed_peak_positions": False,
            "configuration": config,
        }

    spectra = _validate_spectra(training_normalized_spectra)
    axis = _validate_axis(raman_shift, expected_length=spectra.shape[1])
    spacing = float(np.median(np.diff(axis)))

    reference = np.median(spectra, axis=0)
    baseline = gaussian_filter1d(
        reference,
        sigma=float(config["reference_baseline_sigma_cm1"]) / spacing,
        mode="nearest",
        truncate=4.0,
    )
    peak_signal = gaussian_filter1d(
        reference - baseline,
        sigma=(
            float(config["reference_peak_smoothing_sigma_cm1"])
            / spacing
        ),
        mode="nearest",
        truncate=4.0,
    )

    maximum_positive_signal = float(np.max(peak_signal))
    if maximum_positive_signal <= float(config["epsilon"]):
        raise ValueError(
            "D3.1无法从训练集参考谱检测正特征峰："
            "去基线后的最大正信号不大于0。"
        )

    detected, properties = find_peaks(
        peak_signal,
        prominence=(
            float(config["minimum_relative_prominence"])
            * maximum_positive_signal
        ),
        distance=max(
            1,
            int(
                round(
                    float(config["minimum_peak_distance_cm1"])
                    / spacing
                )
            ),
        ),
    )

    edge = float(config["edge_exclusion_cm1"])
    inside = (
        (axis[detected] >= axis[0] + edge)
        & (axis[detected] <= axis[-1] - edge)
    )
    detected = detected[inside]
    prominences = np.asarray(
        properties.get("prominences", np.zeros(inside.size)),
        dtype=np.float64,
    )[inside]

    if detected.size == 0:
        raise ValueError(
            "D3.1在边缘排除后没有检测到训练集特征峰；"
            "请先检查训练数据和自动检测参数。"
        )

    maximum_peak_count = int(config["maximum_peak_count"])
    if detected.size > maximum_peak_count:
        keep = np.argsort(prominences)[-maximum_peak_count:]
        detected = detected[keep]
        prominences = prominences[keep]

    order = np.argsort(detected)
    detected = detected[order]
    prominences = prominences[order]
    centers = axis[detected]

    point_mask = _build_soft_peak_mask(
        axis,
        centers,
        core_half_width_cm1=float(config["peak_core_half_width_cm1"]),
        transition_width_cm1=float(config["transition_width_cm1"]),
    )
    derivative_mask = 0.5 * (point_mask[:-1] + point_mask[1:])
    if float(np.sum(derivative_mask)) <= float(config["epsilon"]):
        raise RuntimeError("D3.1自动峰区导数mask为空。")

    first_derivative = np.diff(spectra, axis=1) / np.diff(axis)[None, :]
    selected_derivatives = np.abs(first_derivative[:, derivative_mask > 0.0])
    derivative_scale = float(
        np.percentile(
            selected_derivatives,
            float(config["derivative_scale_quantile"]),
        )
    )
    derivative_scale = max(
        derivative_scale,
        float(config["minimum_derivative_scale"]),
    )

    return {
        "schema_version": PEAK_DERIVATIVE_STATE_SCHEMA_VERSION,
        "method_version": PEAK_DERIVATIVE_METHOD_VERSION,
        "enabled": True,
        "contains_fixed_peak_positions": False,
        "fit_on": "train_only",
        "data_domain": "full_normalized_spectrum",
        "prediction_domain": "d2_6_scaled_local_residual",
        "configuration": config,
        "original_length": int(axis.size),
        "raman_shift": axis.tolist(),
        "raman_spacing_cm1": spacing,
        "reference_median_spectrum": reference.tolist(),
        "detected_peak_indices": detected.astype(np.int64).tolist(),
        "detected_peak_centers_cm1": centers.tolist(),
        "detected_peak_prominences": prominences.tolist(),
        "soft_peak_point_mask": point_mask.tolist(),
        "soft_peak_derivative_mask": derivative_mask.tolist(),
        "derivative_scale": derivative_scale,
        "number_of_training_spectra": int(spectra.shape[0]),
        "peak_mask_fraction": float(np.mean(point_mask > 0.0)),
    }


class DifferentiablePeakDerivativeLoss(nn.Module):
    """在D2.6恢复后的完整光谱域计算峰区一阶导数损失。"""

    def __init__(
        self,
        *,
        peak_derivative_constraint_state: dict[str, Any],
        broad_local_residual_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        super().__init__()

        state = peak_derivative_constraint_state
        if not isinstance(state, dict) or not bool(state.get("enabled", False)):
            raise ValueError("D3.1 peak derivative state没有启用。")
        if int(state.get("schema_version", 0)) != (
            PEAK_DERIVATIVE_STATE_SCHEMA_VERSION
        ):
            raise ValueError("D3.1 peak derivative state版本不受支持。")
        if bool(state.get("contains_fixed_peak_positions", True)):
            raise ValueError("D3.1 state不能包含人工固定峰位。")

        self.configuration = normalize_peak_derivative_configuration(
            state.get("configuration")
        )
        self.original_length = int(state["original_length"])
        self.padded_length = int(padded_length)
        if self.padded_length < self.original_length:
            raise ValueError("D3.1 padded_length不能小于原始光谱长度。")

        axis = _validate_axis(
            np.asarray(state["raman_shift"], dtype=np.float64),
            expected_length=self.original_length,
        )
        derivative_mask = np.asarray(
            state["soft_peak_derivative_mask"],
            dtype=np.float32,
        ).reshape(-1)
        if derivative_mask.size != self.original_length - 1:
            raise ValueError("D3.1导数mask长度不正确。")

        local_state = broad_local_residual_state.get("local_normalization")
        if not isinstance(local_state, dict):
            raise ValueError(
                "D3.1需要有效的D2.6 broad_local_residual_state。"
            )
        if str(local_state.get("method", "")).lower() != "robust_asinh":
            raise ValueError("D3.1当前只支持D2.6 local robust_asinh。")

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
        self.derivative_scale = _positive_float(
            state["derivative_scale"],
            "peak_derivative_constraint_state.derivative_scale",
        )

        spacing = np.diff(axis).astype(np.float32)
        self.register_buffer(
            "derivative_spacing",
            torch.from_numpy(spacing).view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "derivative_mask",
            torch.from_numpy(derivative_mask).view(1, 1, -1),
            persistent=False,
        )

    def _inverse_local_transform(self, values: torch.Tensor) -> torch.Tensor:
        argument = (
            values
            / self.local_target_abs_max
            * self.local_asinh_normalizer
        )
        limit = float(self.configuration["numerical_safety_limit"])
        # 训练目标的argument远小于该数值安全边界，因此这里与D2.6
        # inverse_local_transform严格一致；只在异常模型输出可能导致
        # sinh溢出时截断。
        safe_argument = torch.clamp(argument, min=-limit, max=limit)
        return self.local_residual_scale * torch.sinh(safe_argument)

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
                "D3.1缺少逐样本reconstruction_base："
                "outer PCA prior + true broad residual。"
            )
        if not (
            predicted_scaled_local_residual.shape
            == target_scaled_local_residual.shape
            == reconstruction_base.shape
        ):
            raise ValueError("D3.1预测、目标和reconstruction_base形状必须一致。")

        predicted_scaled = predicted_scaled_local_residual[
            ..., : self.original_length
        ]
        target_scaled = target_scaled_local_residual[
            ..., : self.original_length
        ]
        base = reconstruction_base[..., : self.original_length]

        predicted_full = base + self._inverse_local_transform(predicted_scaled)
        target_full = base + self._inverse_local_transform(target_scaled)

        spacing = self.derivative_spacing.to(predicted_full)
        mask = self.derivative_mask.to(predicted_full)
        predicted_derivative = (
            predicted_full[..., 1:] - predicted_full[..., :-1]
        ) / spacing
        target_derivative = (
            target_full[..., 1:] - target_full[..., :-1]
        ) / spacing

        normalized_difference = (
            predicted_derivative - target_derivative
        ) / self.derivative_scale
        pointwise = F.smooth_l1_loss(
            normalized_difference,
            torch.zeros_like(normalized_difference),
            reduction="none",
            beta=float(self.configuration["smooth_l1_beta"]),
        )
        per_sample = (
            (pointwise * mask).flatten(start_dim=1).sum(dim=1)
            / mask.sum().clamp_min(float(self.configuration["epsilon"]))
        )

        alpha_bar = alphas_cumprod.gather(-1, timesteps).to(per_sample)
        timestep_weight = torch.pow(
            alpha_bar.clamp(0.0, 1.0),
            float(self.configuration["timestep_weight_power"]),
        ).clamp_min(float(self.configuration["minimum_timestep_weight"]))
        weighted_per_sample = per_sample * timestep_weight

        return {
            "peak_derivative_raw_loss": per_sample.mean(),
            "peak_derivative_timestep_weighted_loss": (
                weighted_per_sample.mean()
            ),
            "mean_peak_derivative_timestep_weight": timestep_weight.mean(),
            "mean_peak_derivative_absolute_error": (
                (
                    torch.abs(predicted_derivative - target_derivative)
                    * mask
                ).flatten(start_dim=1).sum(dim=1)
                / mask.sum().clamp_min(float(self.configuration["epsilon"]))
            ).mean(),
        }

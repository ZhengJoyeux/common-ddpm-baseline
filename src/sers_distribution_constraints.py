"""D3.3-A：训练集驱动的 SERS 峰分布与批量多样性约束。

本模块不写死任何农药峰位，也不要求预测峰逐点贴合当前 target。
它只使用训练集完成三件事：

1. 自动寻找在多条训练光谱中重复出现的稳定峰区；
2. 统计峰位、峰高、峰宽、峰面积、相对峰高、局部总变差和尖锐度
   的训练分布，并仅惩罚超出稳健范围的异常值；
3. 比较同一 batch 中预测光谱与 target 光谱的峰特征方差和逐波数
   方差，显式惩罚多样性坍缩与过度离散。

所有拟合统计仅来自训练集，并通过 state_dict 保存到 checkpoint
metadata。稳定峰中心是从训练数据自动学习的统计结果，不是人工指定的
DEL、CHL 或 TEB 特征峰。
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


DISTRIBUTION_STATE_SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "position_cm1",
    "height",
    "width_cm1",
    "area",
    "relative_height",
    "normalized_total_variation",
    "normalized_sharpness",
)

def _trapezoid_integral(
    values: np.ndarray,
    *,
    x: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Use NumPy 2.x trapezoid integration while keeping older NumPy support."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(values, x=x, axis=axis)

    trapz = getattr(np, "trapz")
    return trapz(values, x=x, axis=axis)


def _finite_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是数值。") from error

    if not np.isfinite(parsed):
        raise ValueError(f"{field_name}必须为有限数值。")

    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)

    if parsed <= 0.0:
        raise ValueError(f"{field_name}必须大于0。")

    return parsed


def _nonnegative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)

    if parsed < 0.0:
        raise ValueError(f"{field_name}不能小于0。")

    return parsed


def _fraction(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = True,
) -> float:
    parsed = _finite_float(value, field_name)
    lower_is_valid = parsed >= 0.0 if allow_zero else parsed > 0.0

    if not lower_is_valid or parsed > 1.0:
        interval = "[0,1]" if allow_zero else "(0,1]"
        raise ValueError(f"{field_name}必须在{interval}范围内。")

    return parsed


def _percentile(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)

    if not 0.0 <= parsed <= 100.0:
        raise ValueError(f"{field_name}必须在[0,100]范围内。")

    return parsed


def _positive_integer(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是整数。") from error

    if parsed <= 0:
        raise ValueError(f"{field_name}必须大于0。")

    return parsed


def _section(configuration: dict[str, Any], name: str) -> dict[str, Any]:
    value = configuration.setdefault(name, {})

    if not isinstance(value, dict):
        raise TypeError(
            f"physics_constraints.distribution_preservation.{name}"
            "必须是字典。"
        )

    return value


def normalize_distribution_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """校验并补全 D3.3-A 分布保持配置。"""

    if configuration is None:
        configuration = {"enabled": False}

    if not isinstance(configuration, dict):
        raise TypeError(
            "physics_constraints.distribution_preservation必须是字典。"
        )

    config = deepcopy(configuration)
    config["enabled"] = bool(config.get("enabled", False))

    if not config["enabled"]:
        return config

    for forbidden_name in (
        "peak_positions",
        "peak_centers",
        "peak_windows",
        "characteristic_peak_positions",
    ):
        if forbidden_name in config:
            raise ValueError(
                f"D3.3-A不允许配置{forbidden_name}；"
                "稳定峰必须只从训练集自动拟合。"
            )

    config["method_version"] = "d3_3_a_peak_distribution_diversity"
    config["total_weight"] = _positive_float(
        config.get("total_weight", 0.005),
        "distribution_preservation.total_weight",
    )
    config["epsilon"] = _positive_float(
        config.get("epsilon", 1.0e-8),
        "distribution_preservation.epsilon",
    )

    detection = _section(config, "stable_peak_detection")
    detection["smoothing_sigma_cm1"] = _positive_float(
        detection.get("smoothing_sigma_cm1", 2.0),
        "distribution_preservation.stable_peak_detection."
        "smoothing_sigma_cm1",
    )
    detection["prominence_noise_multiplier"] = _positive_float(
        detection.get("prominence_noise_multiplier", 3.0),
        "distribution_preservation.stable_peak_detection."
        "prominence_noise_multiplier",
    )
    detection["minimum_relative_prominence"] = _fraction(
        detection.get("minimum_relative_prominence", 0.04),
        "distribution_preservation.stable_peak_detection."
        "minimum_relative_prominence",
        allow_zero=False,
    )
    detection["consensus_tolerance_cm1"] = _positive_float(
        detection.get("consensus_tolerance_cm1", 5.0),
        "distribution_preservation.stable_peak_detection."
        "consensus_tolerance_cm1",
    )
    detection["minimum_prevalence"] = _fraction(
        detection.get("minimum_prevalence", 0.60),
        "distribution_preservation.stable_peak_detection."
        "minimum_prevalence",
        allow_zero=False,
    )
    detection["minimum_separation_cm1"] = _positive_float(
        detection.get("minimum_separation_cm1", 12.0),
        "distribution_preservation.stable_peak_detection."
        "minimum_separation_cm1",
    )
    detection["analysis_half_width_cm1"] = _positive_float(
        detection.get("analysis_half_width_cm1", 20.0),
        "distribution_preservation.stable_peak_detection."
        "analysis_half_width_cm1",
    )
    detection["maximum_stable_peaks"] = _positive_integer(
        detection.get("maximum_stable_peaks", 10),
        "distribution_preservation.stable_peak_detection."
        "maximum_stable_peaks",
    )
    detection["edge_exclusion_cm1"] = _nonnegative_float(
        detection.get("edge_exclusion_cm1", 24.0),
        "distribution_preservation.stable_peak_detection."
        "edge_exclusion_cm1",
    )

    if (
        detection["edge_exclusion_cm1"]
        < detection["analysis_half_width_cm1"]
    ):
        raise ValueError(
            "stable_peak_detection.edge_exclusion_cm1必须大于等于"
            "analysis_half_width_cm1。"
        )

    feature_range = _section(config, "feature_range")
    feature_range["lower_quantile"] = _percentile(
        feature_range.get("lower_quantile", 2.5),
        "distribution_preservation.feature_range.lower_quantile",
    )
    feature_range["upper_quantile"] = _percentile(
        feature_range.get("upper_quantile", 97.5),
        "distribution_preservation.feature_range.upper_quantile",
    )
    feature_range["iqr_margin_multiplier"] = _nonnegative_float(
        feature_range.get("iqr_margin_multiplier", 0.25),
        "distribution_preservation.feature_range."
        "iqr_margin_multiplier",
    )
    feature_range["minimum_position_margin_cm1"] = _nonnegative_float(
        feature_range.get("minimum_position_margin_cm1", 1.0),
        "distribution_preservation.feature_range."
        "minimum_position_margin_cm1",
    )
    feature_range["transition_fraction"] = _positive_float(
        feature_range.get("transition_fraction", 0.20),
        "distribution_preservation.feature_range.transition_fraction",
    )

    if feature_range["lower_quantile"] >= feature_range["upper_quantile"]:
        raise ValueError(
            "feature_range.lower_quantile必须小于upper_quantile。"
        )

    batch_moments = _section(config, "batch_moments")
    batch_moments["mean_tolerance_fraction"] = _nonnegative_float(
        batch_moments.get("mean_tolerance_fraction", 0.20),
        "distribution_preservation.batch_moments."
        "mean_tolerance_fraction",
    )
    batch_moments["minimum_std_ratio"] = _fraction(
        batch_moments.get("minimum_std_ratio", 0.70),
        "distribution_preservation.batch_moments.minimum_std_ratio",
    )
    batch_moments["maximum_std_ratio"] = _positive_float(
        batch_moments.get("maximum_std_ratio", 1.50),
        "distribution_preservation.batch_moments.maximum_std_ratio",
    )
    batch_moments["reference_std_floor_fraction"] = _fraction(
        batch_moments.get("reference_std_floor_fraction", 0.25),
        "distribution_preservation.batch_moments."
        "reference_std_floor_fraction",
    )

    if batch_moments["maximum_std_ratio"] < 1.0:
        raise ValueError("batch_moments.maximum_std_ratio不能小于1。")

    pointwise = _section(config, "pointwise_diversity")
    pointwise["enabled"] = bool(pointwise.get("enabled", True))
    pointwise["active_std_quantile"] = _percentile(
        pointwise.get("active_std_quantile", 25.0),
        "distribution_preservation.pointwise_diversity."
        "active_std_quantile",
    )
    pointwise["minimum_std_ratio"] = _fraction(
        pointwise.get("minimum_std_ratio", 0.65),
        "distribution_preservation.pointwise_diversity."
        "minimum_std_ratio",
    )
    pointwise["maximum_std_ratio"] = _positive_float(
        pointwise.get("maximum_std_ratio", 1.60),
        "distribution_preservation.pointwise_diversity."
        "maximum_std_ratio",
    )
    pointwise["reference_std_floor_fraction"] = _fraction(
        pointwise.get("reference_std_floor_fraction", 0.25),
        "distribution_preservation.pointwise_diversity."
        "reference_std_floor_fraction",
    )

    if pointwise["maximum_std_ratio"] < 1.0:
        raise ValueError(
            "pointwise_diversity.maximum_std_ratio不能小于1。"
        )

    timestep = _section(config, "timestep_weighting")
    timestep["mode"] = str(
        timestep.get("mode", "sqrt_alpha_cumprod")
    ).strip().lower()

    if timestep["mode"] not in {"none", "sqrt_alpha_cumprod"}:
        raise ValueError(
            "distribution_preservation.timestep_weighting.mode只能是"
            "none或sqrt_alpha_cumprod。"
        )

    timestep["minimum_weight"] = _fraction(
        timestep.get("minimum_weight", 0.05),
        "distribution_preservation.timestep_weighting.minimum_weight",
    )

    component_weights = _section(config, "component_weights")
    defaults = {
        "feature_range": 1.0,
        "feature_mean": 0.5,
        "feature_std": 2.0,
        "pointwise_std": 2.0,
    }

    for name, default in defaults.items():
        component_weights[name] = _nonnegative_float(
            component_weights.get(name, default),
            f"distribution_preservation.component_weights.{name}",
        )

    if sum(component_weights.values()) <= 0.0:
        raise ValueError(
            "distribution_preservation.component_weights至少有一个"
            "权重大于0。"
        )

    return config


def _validate_axis(raman_shift: np.ndarray) -> np.ndarray:
    axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)

    if axis.size < 9:
        raise ValueError("拉曼位移轴至少需要9个点。")
    if not np.isfinite(axis).all():
        raise ValueError("拉曼位移轴包含NaN或无穷值。")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("拉曼位移轴必须严格递增。")

    return axis


def _validate_spectra(
    spectra: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)

    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("训练光谱必须为至少3条的二维数组[N,L]。")
    if values.shape[1] != expected_length:
        raise ValueError("训练光谱长度与拉曼轴长度不一致。")
    if not np.isfinite(values).all():
        raise ValueError("训练光谱包含NaN或无穷值。")

    return values


def _cm1_to_points(value_cm1: float, spacing_cm1: float) -> int:
    return max(1, int(np.ceil(float(value_cm1) / spacing_cm1)))


def _detect_peak_indices_per_spectrum(
    spectra: np.ndarray,
    *,
    spacing_cm1: float,
    configuration: dict[str, Any],
) -> list[np.ndarray]:
    detection = configuration["stable_peak_detection"]
    sigma_points = float(detection["smoothing_sigma_cm1"]) / spacing_cm1
    smoothed = gaussian_filter1d(
        spectra,
        sigma=max(sigma_points, 1.0e-6),
        axis=1,
        mode="reflect",
    )
    distance = _cm1_to_points(
        detection["minimum_separation_cm1"],
        spacing_cm1,
    )
    edge = _cm1_to_points(
        detection["edge_exclusion_cm1"],
        spacing_cm1,
    )
    results: list[np.ndarray] = []

    for raw_row, smooth_row in zip(spectra, smoothed, strict=True):
        differences = np.diff(raw_row)
        noise = float(np.median(np.abs(differences))) / 0.6745
        robust_range = float(
            np.percentile(smooth_row, 99.0)
            - np.percentile(smooth_row, 10.0)
        )
        prominence = max(
            noise * float(detection["prominence_noise_multiplier"]),
            robust_range * float(detection["minimum_relative_prominence"]),
            float(configuration["epsilon"]),
        )
        indices, _ = find_peaks(
            smooth_row,
            prominence=prominence,
            distance=distance,
        )
        indices = indices[
            (indices >= edge) & (indices < smooth_row.size - edge)
        ]
        results.append(indices.astype(np.int64, copy=False))

    return results


def _select_stable_peak_indices(
    peak_indices_per_spectrum: list[np.ndarray],
    *,
    length: int,
    spacing_cm1: float,
    configuration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    detection = configuration["stable_peak_detection"]
    tolerance = _cm1_to_points(
        detection["consensus_tolerance_cm1"],
        spacing_cm1,
    )
    separation = _cm1_to_points(
        detection["minimum_separation_cm1"],
        spacing_cm1,
    )
    presence = np.zeros(
        (len(peak_indices_per_spectrum), length),
        dtype=np.float32,
    )

    for row_index, indices in enumerate(peak_indices_per_spectrum):
        for peak_index in indices.tolist():
            start = max(0, peak_index - tolerance)
            stop = min(length, peak_index + tolerance + 1)
            presence[row_index, start:stop] = 1.0

    prevalence_curve = presence.mean(axis=0)
    candidate_indices, _ = find_peaks(
        prevalence_curve,
        height=float(detection["minimum_prevalence"]),
        distance=separation,
        plateau_size=True,
    )

    if candidate_indices.size == 0:
        raise ValueError(
            "训练集中没有检测到满足minimum_prevalence的稳定峰。"
            "请先检查真实光谱和峰检测阈值，不能手工写死峰位绕过。"
        )

    candidate_prevalence = prevalence_curve[candidate_indices]
    ranking = np.argsort(-candidate_prevalence, kind="stable")
    maximum = int(detection["maximum_stable_peaks"])
    selected = candidate_indices[ranking[:maximum]]
    selected_prevalence = prevalence_curve[selected]
    order = np.argsort(selected)

    return (
        selected[order].astype(np.int64),
        selected_prevalence[order].astype(np.float32),
    )


def _extract_numpy_features(
    spectra: np.ndarray,
    axis: np.ndarray,
    peak_indices: np.ndarray,
    half_width_points: int,
    epsilon: float,
) -> np.ndarray:
    number_of_spectra = spectra.shape[0]
    number_of_peaks = peak_indices.size
    features = np.zeros(
        (number_of_spectra, number_of_peaks, len(FEATURE_NAMES)),
        dtype=np.float64,
    )
    all_heights = np.zeros(
        (number_of_spectra, number_of_peaks),
        dtype=np.float64,
    )

    for peak_number, center_index in enumerate(peak_indices.tolist()):
        start = center_index - half_width_points
        stop = center_index + half_width_points + 1
        axis_window = axis[start:stop]
        window = spectra[:, start:stop]
        fraction = np.linspace(0.0, 1.0, window.shape[1])[None, :]
        baseline = (
            window[:, :1] * (1.0 - fraction)
            + window[:, -1:] * fraction
        )
        profile = np.maximum(window - baseline, 0.0)
        mass = np.maximum(profile.sum(axis=1), epsilon)
        height = np.maximum(profile.max(axis=1), epsilon)
        position = (profile * axis_window[None, :]).sum(axis=1) / mass
        variance = (
            profile
            * np.square(axis_window[None, :] - position[:, None])
        ).sum(axis=1) / mass
        width = 2.354820045 * np.sqrt(np.maximum(variance, epsilon))
        area = _trapezoid_integral(profile, x=axis_window, axis=1)
        normalized = profile / height[:, None]
        total_variation = np.abs(np.diff(normalized, axis=1)).sum(axis=1)
        second = np.diff(normalized, n=2, axis=1)
        sharpness = np.max(np.abs(second), axis=1)

        features[:, peak_number, 0] = position
        features[:, peak_number, 1] = height
        features[:, peak_number, 2] = width
        features[:, peak_number, 3] = area
        features[:, peak_number, 5] = total_variation
        features[:, peak_number, 6] = sharpness
        all_heights[:, peak_number] = height

    height_sum = np.maximum(all_heights.sum(axis=1, keepdims=True), epsilon)
    features[:, :, 4] = all_heights / height_sum

    return features


def fit_sers_distribution_constraint_state(
    *,
    training_normalized_spectra: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """仅用训练集拟合 D3.3-A 稳定峰与多样性统计。"""

    config = normalize_distribution_configuration(configuration)

    if not bool(config.get("enabled", False)):
        return {
            "schema_version": DISTRIBUTION_STATE_SCHEMA_VERSION,
            "enabled": False,
            "configuration": config,
        }

    axis = _validate_axis(raman_shift)
    spectra = _validate_spectra(
        training_normalized_spectra,
        expected_length=axis.size,
    )
    spacing = float(np.median(np.diff(axis)))
    epsilon = float(config["epsilon"])
    peak_indices_per_spectrum = _detect_peak_indices_per_spectrum(
        spectra,
        spacing_cm1=spacing,
        configuration=config,
    )
    peak_indices, prevalence = _select_stable_peak_indices(
        peak_indices_per_spectrum,
        length=axis.size,
        spacing_cm1=spacing,
        configuration=config,
    )
    half_width_points = _cm1_to_points(
        config["stable_peak_detection"]["analysis_half_width_cm1"],
        spacing,
    )

    if np.any(peak_indices - half_width_points < 0) or np.any(
        peak_indices + half_width_points >= axis.size
    ):
        raise RuntimeError("稳定峰局部窗口越出拉曼轴范围。")

    features = _extract_numpy_features(
        spectra,
        axis,
        peak_indices,
        half_width_points,
        epsilon,
    )
    range_config = config["feature_range"]
    lower = np.percentile(
        features,
        float(range_config["lower_quantile"]),
        axis=0,
    )
    upper = np.percentile(
        features,
        float(range_config["upper_quantile"]),
        axis=0,
    )
    q25 = np.percentile(features, 25.0, axis=0)
    q75 = np.percentile(features, 75.0, axis=0)
    iqr = q75 - q25
    lower -= float(range_config["iqr_margin_multiplier"]) * iqr
    upper += float(range_config["iqr_margin_multiplier"]) * iqr
    position_margin = float(
        range_config["minimum_position_margin_cm1"]
    )
    centers = axis[peak_indices]
    lower[:, 0] = np.minimum(lower[:, 0], centers - position_margin)
    upper[:, 0] = np.maximum(upper[:, 0], centers + position_margin)
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0, ddof=1)
    feature_scale = np.maximum.reduce(
        [
            feature_std,
            iqr / 1.349,
            np.full_like(feature_std, epsilon),
        ]
    )
    invalid_range = lower >= upper
    lower[invalid_range] = feature_mean[invalid_range] - feature_scale[
        invalid_range
    ]
    upper[invalid_range] = feature_mean[invalid_range] + feature_scale[
        invalid_range
    ]
    pointwise_std = np.std(spectra, axis=0, ddof=1)
    active_threshold = float(
        np.percentile(
            pointwise_std,
            config["pointwise_diversity"]["active_std_quantile"],
        )
    )
    active_mask = pointwise_std >= max(active_threshold, epsilon)

    if not np.any(active_mask):
        raise RuntimeError("训练集逐波数方差全部过小，无法拟合多样性状态。")

    return {
        "schema_version": DISTRIBUTION_STATE_SCHEMA_VERSION,
        "enabled": True,
        "method_version": "d3_3_a_peak_distribution_diversity",
        "contains_manually_fixed_peak_positions": False,
        "configuration": config,
        "number_of_training_spectra": int(spectra.shape[0]),
        "original_length": int(axis.size),
        "raman_shift": axis.astype(np.float32).tolist(),
        "raman_spacing_cm1": spacing,
        "stable_peak_indices": peak_indices.tolist(),
        "stable_peak_positions_cm1": centers.astype(np.float32).tolist(),
        "stable_peak_prevalence": prevalence.tolist(),
        "analysis_half_width_points": int(half_width_points),
        "feature_names": list(FEATURE_NAMES),
        "feature_lower": lower.astype(np.float32).tolist(),
        "feature_upper": upper.astype(np.float32).tolist(),
        "feature_mean": feature_mean.astype(np.float32).tolist(),
        "feature_std": feature_std.astype(np.float32).tolist(),
        "feature_scale": feature_scale.astype(np.float32).tolist(),
        "pointwise_training_std": pointwise_std.astype(np.float32).tolist(),
        "pointwise_active_mask": active_mask.tolist(),
    }


class DifferentiableSersDistributionLoss(nn.Module):
    """D3.3-A 稳定峰分布与 batch 多样性可微损失。"""

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()

        if not isinstance(state, dict):
            raise TypeError("distribution_constraint_state必须是字典。")
        if int(state.get("schema_version", 0)) != (
            DISTRIBUTION_STATE_SCHEMA_VERSION
        ):
            raise ValueError("distribution_constraint_state版本不受支持。")
        if not bool(state.get("enabled", False)):
            raise ValueError("distribution_constraint_state没有启用。")
        if state.get("method_version") != (
            "d3_3_a_peak_distribution_diversity"
        ):
            raise ValueError("distribution_constraint_state不是D3.3-A状态。")
        if bool(
            state.get("contains_manually_fixed_peak_positions", True)
        ):
            raise ValueError("D3.3-A状态不能包含人工写死的峰位。")

        self.configuration = normalize_distribution_configuration(
            state["configuration"]
        )
        self.original_length = int(state["original_length"])
        axis = torch.as_tensor(state["raman_shift"], dtype=torch.float32)
        peak_indices = torch.as_tensor(
            state["stable_peak_indices"],
            dtype=torch.long,
        )
        self.half_width_points = int(state["analysis_half_width_points"])
        offsets = torch.arange(
            -self.half_width_points,
            self.half_width_points + 1,
            dtype=torch.long,
        )
        window_indices = peak_indices[:, None] + offsets[None, :]

        if axis.numel() != self.original_length:
            raise ValueError("D3.3-A拉曼轴长度不一致。")
        if torch.any(window_indices < 0) or torch.any(
            window_indices >= self.original_length
        ):
            raise ValueError("D3.3-A稳定峰窗口越界。")

        self.register_buffer("raman_shift", axis, persistent=False)
        self.register_buffer("window_indices", window_indices, persistent=False)
        self.register_buffer(
            "feature_lower",
            torch.as_tensor(state["feature_lower"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_upper",
            torch.as_tensor(state["feature_upper"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(state["feature_mean"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(state["feature_std"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(state["feature_scale"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "pointwise_training_std",
            torch.as_tensor(
                state["pointwise_training_std"],
                dtype=torch.float32,
            ).view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "pointwise_active_mask",
            torch.as_tensor(
                state["pointwise_active_mask"],
                dtype=torch.bool,
            ).view(1, 1, -1),
            persistent=False,
        )

        expected_shape = (
            peak_indices.numel(),
            len(FEATURE_NAMES),
        )
        for name in (
            "feature_lower",
            "feature_upper",
            "feature_mean",
            "feature_std",
            "feature_scale",
        ):
            if tuple(getattr(self, name).shape) != expected_shape:
                raise ValueError(f"D3.3-A {name}形状不正确。")

        self.epsilon = float(self.configuration["epsilon"])
        self.transition_fraction = float(
            self.configuration["feature_range"]["transition_fraction"]
        )
        self.total_weight = float(self.configuration["total_weight"])
        moments = self.configuration["batch_moments"]
        self.mean_tolerance_fraction = float(
            moments["mean_tolerance_fraction"]
        )
        self.minimum_std_ratio = float(moments["minimum_std_ratio"])
        self.maximum_std_ratio = float(moments["maximum_std_ratio"])
        self.reference_std_floor_fraction = float(
            moments["reference_std_floor_fraction"]
        )
        pointwise = self.configuration["pointwise_diversity"]
        self.pointwise_enabled = bool(pointwise["enabled"])
        self.pointwise_minimum_std_ratio = float(
            pointwise["minimum_std_ratio"]
        )
        self.pointwise_maximum_std_ratio = float(
            pointwise["maximum_std_ratio"]
        )
        self.pointwise_reference_std_floor_fraction = float(
            pointwise["reference_std_floor_fraction"]
        )
        timestep = self.configuration["timestep_weighting"]
        self.timestep_mode = str(timestep["mode"])
        self.minimum_timestep_weight = float(timestep["minimum_weight"])
        self.component_weights = {
            name: float(value)
            for name, value in self.configuration[
                "component_weights"
            ].items()
        }
        self.component_weight_sum = sum(self.component_weights.values())

    def _sample_weights(
        self,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        if self.timestep_mode == "none":
            weights = torch.ones_like(timesteps, dtype=alphas_cumprod.dtype)
        else:
            weights = torch.sqrt(
                alphas_cumprod.gather(0, timesteps).clamp_min(0.0)
            ).clamp_min(self.minimum_timestep_weight)

        return weights

    def _extract_features(self, spectra: torch.Tensor) -> torch.Tensor:
        values = spectra[..., : self.original_length]
        batch_size = values.shape[0]
        number_of_peaks = self.window_indices.shape[0]
        expanded = values.squeeze(1)[:, None, :].expand(
            batch_size,
            number_of_peaks,
            self.original_length,
        )
        indices = self.window_indices[None, :, :].expand(
            batch_size,
            -1,
            -1,
        )
        windows = torch.gather(expanded, 2, indices)
        axis_windows = self.raman_shift[self.window_indices]
        fraction = torch.linspace(
            0.0,
            1.0,
            windows.shape[-1],
            device=windows.device,
            dtype=windows.dtype,
        ).view(1, 1, -1)
        baseline = (
            windows[..., :1] * (1.0 - fraction)
            + windows[..., -1:] * fraction
        )
        # 与训练集拟合阶段的 np.maximum(..., 0) 保持同一定义。
        # ReLU 在正峰区域可微，同时不会给平坦基线凭空增加软正质量。
        profile = F.relu(windows - baseline)
        mass = profile.sum(dim=-1).clamp_min(self.epsilon)
        height = profile.amax(dim=-1).clamp_min(self.epsilon)
        position = (
            profile * axis_windows[None, :, :]
        ).sum(dim=-1) / mass
        variance = (
            profile
            * torch.square(
                axis_windows[None, :, :] - position[..., None]
            )
        ).sum(dim=-1) / mass
        width = 2.354820045 * torch.sqrt(
            variance.clamp_min(self.epsilon)
        )
        spacing = torch.diff(axis_windows, dim=-1)
        area = (
            0.5
            * (profile[..., 1:] + profile[..., :-1])
            * spacing[None, :, :]
        ).sum(dim=-1)
        relative_height = height / height.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(self.epsilon)
        normalized = profile / height[..., None]
        total_variation = torch.abs(
            torch.diff(normalized, dim=-1)
        ).sum(dim=-1)
        sharpness = torch.abs(
            torch.diff(normalized, n=2, dim=-1)
        ).amax(dim=-1)

        return torch.stack(
            (
                position,
                height,
                width,
                area,
                relative_height,
                total_variation,
                sharpness,
            ),
            dim=-1,
        )

    @staticmethod
    def _weighted_mean_and_std(
        values: torch.Tensor,
        sample_weights: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = sample_weights / sample_weights.sum().clamp_min(epsilon)
        shaped_weights = weights.view(
            values.shape[0],
            *((1,) * (values.ndim - 1)),
        )
        mean = (values * shaped_weights).sum(dim=0)
        variance = (
            torch.square(values - mean.unsqueeze(0)) * shaped_weights
        ).sum(dim=0)

        return mean, torch.sqrt(variance.clamp_min(epsilon))

    def _feature_range_loss(
        self,
        predicted_features: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        transition = (
            self.feature_scale * self.transition_fraction
        ).clamp_min(self.epsilon)
        below = F.relu(
            self.feature_lower[None, :, :] - predicted_features
        ) / transition[None, :, :]
        above = F.relu(
            predicted_features - self.feature_upper[None, :, :]
        ) / transition[None, :, :]
        per_sample = torch.square(below) + torch.square(above)
        per_sample = per_sample.mean(dim=(1, 2))

        return (
            per_sample * sample_weights
        ).sum() / sample_weights.sum().clamp_min(self.epsilon)

    def _feature_moment_losses(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_mean, predicted_std = self._weighted_mean_and_std(
            predicted_features,
            sample_weights,
            self.epsilon,
        )
        target_mean, target_std = self._weighted_mean_and_std(
            target_features,
            sample_weights,
            self.epsilon,
        )
        mean_error = torch.abs(predicted_mean - target_mean)
        allowed_mean_error = (
            self.feature_scale * self.mean_tolerance_fraction
        )
        mean_loss = torch.square(
            F.relu(mean_error - allowed_mean_error)
            / self.feature_scale.clamp_min(self.epsilon)
        ).mean()
        reference_std = torch.maximum(
            target_std,
            self.feature_std * self.reference_std_floor_fraction,
        )
        lower = reference_std * self.minimum_std_ratio
        upper_reference = torch.maximum(reference_std, self.feature_std)
        upper = upper_reference * self.maximum_std_ratio
        scale = torch.maximum(
            self.feature_scale,
            reference_std,
        ).clamp_min(self.epsilon)
        std_loss = (
            torch.square(F.relu(lower - predicted_std) / scale)
            + torch.square(F.relu(predicted_std - upper) / scale)
        ).mean()

        return mean_loss, std_loss

    def _pointwise_std_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.pointwise_enabled or predicted.shape[0] < 2:
            return predicted.new_zeros(())

        _, predicted_std = self._weighted_mean_and_std(
            predicted[..., : self.original_length],
            sample_weights,
            self.epsilon,
        )
        _, target_std = self._weighted_mean_and_std(
            target[..., : self.original_length],
            sample_weights,
            self.epsilon,
        )
        reference = torch.maximum(
            target_std,
            self.pointwise_training_std
            * self.pointwise_reference_std_floor_fraction,
        )
        lower = reference * self.pointwise_minimum_std_ratio
        upper = torch.maximum(
            reference,
            self.pointwise_training_std,
        ) * self.pointwise_maximum_std_ratio
        scale = torch.maximum(
            self.pointwise_training_std,
            reference,
        ).clamp_min(self.epsilon)
        violation = (
            torch.square(F.relu(lower - predicted_std) / scale)
            + torch.square(F.relu(predicted_std - upper) / scale)
        )
        selected = violation[self.pointwise_active_mask]

        if selected.numel() == 0:
            return predicted.new_zeros(())

        return selected.mean()

    def forward(
        self,
        *,
        predicted_spectra: torch.Tensor,
        target_spectra: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_spectra.shape != target_spectra.shape:
            raise ValueError("D3.3-A预测光谱和target光谱形状必须一致。")
        if predicted_spectra.ndim != 3 or predicted_spectra.shape[1] != 1:
            raise ValueError("D3.3-A光谱必须为[B,1,L]。")
        if timesteps.ndim != 1 or timesteps.shape[0] != (
            predicted_spectra.shape[0]
        ):
            raise ValueError("D3.3-A timesteps批量形状不正确。")

        sample_weights = self._sample_weights(timesteps, alphas_cumprod)
        predicted_features = self._extract_features(predicted_spectra)
        target_features = self._extract_features(target_spectra)
        feature_range_loss = self._feature_range_loss(
            predicted_features,
            sample_weights,
        )

        if predicted_spectra.shape[0] < 2:
            feature_mean_loss = predicted_spectra.new_zeros(())
            feature_std_loss = predicted_spectra.new_zeros(())
        else:
            feature_mean_loss, feature_std_loss = (
                self._feature_moment_losses(
                    predicted_features,
                    target_features,
                    sample_weights,
                )
            )

        pointwise_std_loss = self._pointwise_std_loss(
            predicted_spectra,
            target_spectra,
            sample_weights,
        )
        components = {
            "distribution_feature_range_loss": feature_range_loss,
            "distribution_feature_mean_loss": feature_mean_loss,
            "distribution_feature_std_loss": feature_std_loss,
            "distribution_pointwise_std_loss": pointwise_std_loss,
        }
        raw_loss = (
            self.component_weights["feature_range"] * feature_range_loss
            + self.component_weights["feature_mean"] * feature_mean_loss
            + self.component_weights["feature_std"] * feature_std_loss
            + self.component_weights["pointwise_std"]
            * pointwise_std_loss
        ) / max(self.component_weight_sum, self.epsilon)
        timestep_factor = sample_weights.mean()
        weighted_loss = raw_loss * timestep_factor

        return {
            **components,
            "distribution_raw_loss": raw_loss,
            "distribution_timestep_weighted_loss": weighted_loss,
            "mean_distribution_timestep_weight": timestep_factor,
            "predicted_feature_std_mean": torch.std(
                predicted_features,
                dim=0,
                correction=0,
            ).mean(),
            "target_feature_std_mean": torch.std(
                target_features,
                dim=0,
                correction=0,
            ).mean(),
        }
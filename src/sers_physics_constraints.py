"""
D3.2：目标自适应 SERS 局部病态峰约束。

本模块在 D3.1 的目标光谱自适应峰形态约束基础上，专门处理生成
光谱中观察到的三类失败模式：

1. 真实特征峰窗口内出现极深负谷；
2. 一个正常宽峰被拆成两三个极窄尖峰；
3. 少量灾难点被 1901 点全谱平均稀释。

主要改动：

* 使用批量 GPU 峰检测和批量局部窗口提取，避免逐光谱、逐候选峰
    的 Python 循环和大量 ``.item()`` 同步；
* 在目标峰窗口内增加局部负谷保护；
* 通过局部平滑轮廓、导数、总变差和软单峰性约束抑制峰分裂；
* 对粗糙度、强度越界和缩放残差使用“全局平均 + 最坏 top-k”聚合；
* 高噪声时间步仍降低精确峰形约束，但对病态保护保留更高下限。

所有统计量只允许使用训练集拟合，并保存到 checkpoint metadata。
本模块不保存任何固定农药特征峰位置。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PHYSICS_STATE_SCHEMA_VERSION = 4

LOSS_COMPONENT_NAMES = (
    "position_loss",
    "width_loss",
    "sharpness_loss",
    "presence_loss",
    "local_shape_loss",
    "roughness_loss",
    "extreme_loss",
    "scaled_residual_guard_loss",
)

MORPHOLOGY_COMPONENT_NAMES = (
    "position_loss",
    "width_loss",
    "sharpness_loss",
    "presence_loss",
    "local_shape_loss",
)

PATHOLOGY_COMPONENT_NAMES = (
    "roughness_loss",
    "extreme_loss",
    "scaled_residual_guard_loss",
)


# ---------------------------------------------------------------------------
# 配置读取与校验
# ---------------------------------------------------------------------------


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
    lower_ok = parsed >= 0.0 if allow_zero else parsed > 0.0

    if not lower_ok or parsed > 1.0:
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
        raise TypeError(f"physics_constraints.{name}必须是字典。")

    return value


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
    *,
    expected_length: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(f"{name}必须是二维数组[N,L]。")

    if values.shape[0] < 2:
        raise ValueError(f"{name}至少需要2条光谱。")

    if values.shape[1] != expected_length:
        raise ValueError(f"{name}长度与拉曼位移轴长度不一致。")

    if not np.isfinite(values).all():
        raise ValueError(f"{name}包含NaN或无穷值。")

    return values


def normalize_physics_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """校验并补全 D3.2 配置。"""

    if not isinstance(configuration, dict):
        raise TypeError("physics_constraints必须是字典。")

    config = deepcopy(configuration)
    config["enabled"] = bool(config.get("enabled", False))

    if not config["enabled"]:
        return config

    forbidden_fields = (
        "peak_windows",
        "peak_centers",
        "fixed_peak_positions",
        "characteristic_peak_positions",
    )

    for field_name in forbidden_fields:
        if field_name in config:
            raise ValueError(
                f"D3.2目标自适应版本不允许配置{field_name}。"
                "峰位置必须从每条真实目标光谱中动态检测。"
            )

    strategy = str(
        config.get("strategy", "target_adaptive_peak_morphology")
    ).strip().lower()

    if strategy != "target_adaptive_peak_morphology":
        raise ValueError(
            "physics_constraints.strategy必须为"
            "target_adaptive_peak_morphology。"
        )

    config["strategy"] = strategy
    config["method_version"] = "d3_2_local_pathology_guard"
    config["calculation_domain"] = (
        "restored_global_minmax_normalized_spectrum"
    )
    config["total_weight"] = _positive_float(
        config.get("total_weight", 0.02),
        "physics_constraints.total_weight",
    )

    inverse = _section(config, "inverse_transform")
    inverse["numerical_safety_limit"] = _positive_float(
        inverse.get("numerical_safety_limit", 15.0),
        "physics_constraints.inverse_transform.numerical_safety_limit",
    )

    timestep = _section(config, "timestep_weighting")
    timestep_mode = str(
        timestep.get("mode", "sqrt_alpha_cumprod")
    ).strip().lower()

    if timestep_mode not in {"none", "sqrt_alpha_cumprod"}:
        raise ValueError(
            "physics_constraints.timestep_weighting.mode只能是"
            "none或sqrt_alpha_cumprod。"
        )

    timestep["mode"] = timestep_mode
    timestep["minimum_weight"] = _fraction(
        timestep.get("minimum_weight", 0.05),
        "physics_constraints.timestep_weighting.minimum_weight",
    )
    timestep["pathology_minimum_weight"] = _fraction(
        timestep.get("pathology_minimum_weight", 0.50),
        "physics_constraints.timestep_weighting."
        "pathology_minimum_weight",
    )

    if (
        timestep["pathology_minimum_weight"]
        < timestep["minimum_weight"]
    ):
        raise ValueError(
            "timestep_weighting.pathology_minimum_weight不能小于"
            "minimum_weight。"
        )

    detection = _section(config, "peak_detection")
    detection["smoothing_sigma_cm1"] = _positive_float(
        detection.get("smoothing_sigma_cm1", 2.0),
        "physics_constraints.peak_detection.smoothing_sigma_cm1",
    )
    detection["local_maximum_radius_cm1"] = _positive_float(
        detection.get("local_maximum_radius_cm1", 4.0),
        "physics_constraints.peak_detection.local_maximum_radius_cm1",
    )
    detection["prominence_radius_cm1"] = _positive_float(
        detection.get("prominence_radius_cm1", 12.0),
        "physics_constraints.peak_detection.prominence_radius_cm1",
    )
    detection["minimum_peak_separation_cm1"] = _positive_float(
        detection.get("minimum_peak_separation_cm1", 10.0),
        "physics_constraints.peak_detection.minimum_peak_separation_cm1",
    )
    detection["analysis_half_width_cm1"] = _positive_float(
        detection.get("analysis_half_width_cm1", 20.0),
        "physics_constraints.peak_detection.analysis_half_width_cm1",
    )
    detection["maximum_peaks_per_spectrum"] = _positive_integer(
        detection.get("maximum_peaks_per_spectrum", 12),
        "physics_constraints.peak_detection.maximum_peaks_per_spectrum",
    )
    detection["minimum_relative_prominence"] = _fraction(
        detection.get("minimum_relative_prominence", 0.08),
        "physics_constraints.peak_detection.minimum_relative_prominence",
        allow_zero=False,
    )
    detection["minimum_prominence_noise_multiplier"] = _positive_float(
        detection.get("minimum_prominence_noise_multiplier", 4.0),
        "physics_constraints.peak_detection."
        "minimum_prominence_noise_multiplier",
    )
    detection["training_prominence_floor_quantile"] = _percentile(
        detection.get("training_prominence_floor_quantile", 25.0),
        "physics_constraints.peak_detection."
        "training_prominence_floor_quantile",
    )
    detection["minimum_absolute_prominence"] = _positive_float(
        detection.get("minimum_absolute_prominence", 1.0e-4),
        "physics_constraints.peak_detection.minimum_absolute_prominence",
    )
    detection["edge_exclusion_cm1"] = _nonnegative_float(
        detection.get("edge_exclusion_cm1", 24.0),
        "physics_constraints.peak_detection.edge_exclusion_cm1",
    )
    detection["positive_softplus_temperature"] = _positive_float(
        detection.get("positive_softplus_temperature", 0.01),
        "physics_constraints.peak_detection."
        "positive_softplus_temperature",
    )
    detection["epsilon"] = _positive_float(
        detection.get("epsilon", 1.0e-8),
        "physics_constraints.peak_detection.epsilon",
    )

    if (
        detection["prominence_radius_cm1"]
        <= detection["local_maximum_radius_cm1"]
    ):
        raise ValueError(
            "peak_detection.prominence_radius_cm1必须大于"
            "local_maximum_radius_cm1。"
        )

    if (
        detection["edge_exclusion_cm1"]
        < detection["analysis_half_width_cm1"]
    ):
        raise ValueError(
            "peak_detection.edge_exclusion_cm1必须大于等于"
            "analysis_half_width_cm1，避免局部窗口越界。"
        )

    position = _section(config, "peak_position")
    position["zero_penalty_tolerance_cm1"] = _positive_float(
        position.get("zero_penalty_tolerance_cm1", 5.0),
        "physics_constraints.peak_position."
        "zero_penalty_tolerance_cm1",
    )
    position["transition_width_cm1"] = _positive_float(
        position.get("transition_width_cm1", 5.0),
        "physics_constraints.peak_position.transition_width_cm1",
    )

    width = _section(config, "peak_width")
    width["narrow_relative_tolerance"] = _fraction(
        width.get("narrow_relative_tolerance", 0.30),
        "physics_constraints.peak_width.narrow_relative_tolerance",
    )
    width["broad_relative_tolerance"] = _fraction(
        width.get("broad_relative_tolerance", 0.50),
        "physics_constraints.peak_width.broad_relative_tolerance",
    )
    width["absolute_tolerance_cm1"] = _positive_float(
        width.get("absolute_tolerance_cm1", 1.5),
        "physics_constraints.peak_width.absolute_tolerance_cm1",
    )
    width["transition_fraction"] = _positive_float(
        width.get("transition_fraction", 0.25),
        "physics_constraints.peak_width.transition_fraction",
    )
    width["narrow_penalty_multiplier"] = _positive_float(
        width.get("narrow_penalty_multiplier", 2.5),
        "physics_constraints.peak_width.narrow_penalty_multiplier",
    )
    width["broad_penalty_multiplier"] = _nonnegative_float(
        width.get("broad_penalty_multiplier", 0.5),
        "physics_constraints.peak_width.broad_penalty_multiplier",
    )

    sharpness = _section(config, "peak_sharpness")
    sharpness["upper_relative_tolerance"] = _fraction(
        sharpness.get("upper_relative_tolerance", 0.25),
        "physics_constraints.peak_sharpness."
        "upper_relative_tolerance",
    )
    sharpness["transition_fraction"] = _positive_float(
        sharpness.get("transition_fraction", 0.20),
        "physics_constraints.peak_sharpness.transition_fraction",
    )

    presence = _section(config, "peak_presence")
    presence["minimum_area_ratio"] = _fraction(
        presence.get("minimum_area_ratio", 0.45),
        "physics_constraints.peak_presence.minimum_area_ratio",
        allow_zero=False,
    )
    presence["transition_fraction"] = _positive_float(
        presence.get("transition_fraction", 0.25),
        "physics_constraints.peak_presence.transition_fraction",
    )

    local_shape = _section(config, "local_shape")
    local_shape["profile_tolerance"] = _fraction(
        local_shape.get("profile_tolerance", 0.10),
        "physics_constraints.local_shape.profile_tolerance",
    )
    local_shape["first_derivative_tolerance"] = _fraction(
        local_shape.get("first_derivative_tolerance", 0.25),
        "physics_constraints.local_shape.first_derivative_tolerance",
    )
    local_shape["second_derivative_tolerance"] = _fraction(
        local_shape.get("second_derivative_tolerance", 0.30),
        "physics_constraints.local_shape.second_derivative_tolerance",
    )
    local_shape["unimodality_tolerance"] = _fraction(
        local_shape.get("unimodality_tolerance", 0.05),
        "physics_constraints.local_shape.unimodality_tolerance",
    )
    local_shape["total_variation_tolerance"] = _fraction(
        local_shape.get("total_variation_tolerance", 0.20),
        "physics_constraints.local_shape.total_variation_tolerance",
    )
    local_shape["target_support_fraction"] = _fraction(
        local_shape.get("target_support_fraction", 0.08),
        "physics_constraints.local_shape.target_support_fraction",
        allow_zero=False,
    )
    local_shape["transition_fraction"] = _positive_float(
        local_shape.get("transition_fraction", 0.20),
        "physics_constraints.local_shape.transition_fraction",
    )
    local_shape["profile_weight"] = _nonnegative_float(
        local_shape.get("profile_weight", 1.5),
        "physics_constraints.local_shape.profile_weight",
    )
    local_shape["first_derivative_weight"] = _nonnegative_float(
        local_shape.get("first_derivative_weight", 1.0),
        "physics_constraints.local_shape.first_derivative_weight",
    )
    local_shape["second_derivative_weight"] = _nonnegative_float(
        local_shape.get("second_derivative_weight", 1.0),
        "physics_constraints.local_shape.second_derivative_weight",
    )
    local_shape["unimodality_weight"] = _nonnegative_float(
        local_shape.get("unimodality_weight", 2.0),
        "physics_constraints.local_shape.unimodality_weight",
    )
    local_shape["total_variation_weight"] = _nonnegative_float(
        local_shape.get("total_variation_weight", 1.0),
        "physics_constraints.local_shape.total_variation_weight",
    )

    local_weight_sum = sum(
        local_shape[name]
        for name in (
            "profile_weight",
            "first_derivative_weight",
            "second_derivative_weight",
            "unimodality_weight",
            "total_variation_weight",
        )
    )

    if local_weight_sum <= 0.0:
        raise ValueError("local_shape至少有一个分项权重大于0。")

    negative_valley = _section(config, "negative_valley")
    negative_valley["enabled"] = bool(
        negative_valley.get("enabled", True)
    )
    negative_valley["noise_margin_multiplier"] = _positive_float(
        negative_valley.get("noise_margin_multiplier", 4.0),
        "physics_constraints.negative_valley."
        "noise_margin_multiplier",
    )
    negative_valley["minimum_margin_fraction"] = _fraction(
        negative_valley.get("minimum_margin_fraction", 0.02),
        "physics_constraints.negative_valley."
        "minimum_margin_fraction",
    )
    negative_valley["center_weight_sigma_fraction"] = _fraction(
        negative_valley.get("center_weight_sigma_fraction", 0.40),
        "physics_constraints.negative_valley."
        "center_weight_sigma_fraction",
        allow_zero=False,
    )
    negative_valley["transition_fraction"] = _positive_float(
        negative_valley.get("transition_fraction", 0.20),
        "physics_constraints.negative_valley.transition_fraction",
    )
    negative_valley["weight_within_extreme"] = _positive_float(
        negative_valley.get("weight_within_extreme", 2.0),
        "physics_constraints.negative_valley."
        "weight_within_extreme",
    )

    aggregation = _section(config, "pathology_aggregation")
    aggregation["mean_weight"] = _nonnegative_float(
        aggregation.get("mean_weight", 0.50),
        "physics_constraints.pathology_aggregation.mean_weight",
    )
    aggregation["topk_weight"] = _nonnegative_float(
        aggregation.get("topk_weight", 0.50),
        "physics_constraints.pathology_aggregation.topk_weight",
    )
    aggregation["topk_fraction"] = _fraction(
        aggregation.get("topk_fraction", 0.01),
        "physics_constraints.pathology_aggregation.topk_fraction",
        allow_zero=False,
    )

    if aggregation["mean_weight"] + aggregation["topk_weight"] <= 0.0:
        raise ValueError(
            "pathology_aggregation的mean_weight和topk_weight"
            "不能同时为0。"
        )

    roughness = _section(config, "roughness")
    roughness["first_derivative_quantile"] = _percentile(
        roughness.get("first_derivative_quantile", 99.5),
        "physics_constraints.roughness.first_derivative_quantile",
    )
    roughness["second_derivative_quantile"] = _percentile(
        roughness.get("second_derivative_quantile", 99.5),
        "physics_constraints.roughness.second_derivative_quantile",
    )
    roughness["limit_multiplier"] = _positive_float(
        roughness.get("limit_multiplier", 1.15),
        "physics_constraints.roughness.limit_multiplier",
    )
    roughness["first_transition_fraction"] = _positive_float(
        roughness.get("first_transition_fraction", 0.20),
        "physics_constraints.roughness.first_transition_fraction",
    )
    roughness["second_transition_fraction"] = _positive_float(
        roughness.get("second_transition_fraction", 0.20),
        "physics_constraints.roughness.second_transition_fraction",
    )
    roughness["first_derivative_weight"] = _nonnegative_float(
        roughness.get("first_derivative_weight", 1.0),
        "physics_constraints.roughness.first_derivative_weight",
    )
    roughness["second_derivative_weight"] = _nonnegative_float(
        roughness.get("second_derivative_weight", 1.5),
        "physics_constraints.roughness.second_derivative_weight",
    )

    if roughness["limit_multiplier"] < 1.0:
        raise ValueError("roughness.limit_multiplier不能小于1。")

    if (
        roughness["first_derivative_weight"]
        + roughness["second_derivative_weight"]
        <= 0.0
    ):
        raise ValueError("roughness至少有一个导数权重大于0。")

    envelope = _section(config, "pointwise_envelope")
    envelope["enabled"] = bool(envelope.get("enabled", True))
    envelope["lower_quantile"] = _percentile(
        envelope.get("lower_quantile", 1.0),
        "physics_constraints.pointwise_envelope.lower_quantile",
    )
    envelope["upper_quantile"] = _percentile(
        envelope.get("upper_quantile", 99.0),
        "physics_constraints.pointwise_envelope.upper_quantile",
    )
    envelope["mad_margin_multiplier"] = _nonnegative_float(
        envelope.get("mad_margin_multiplier", 3.0),
        "physics_constraints.pointwise_envelope."
        "mad_margin_multiplier",
    )
    envelope["transition_fraction"] = _positive_float(
        envelope.get("transition_fraction", 0.20),
        "physics_constraints.pointwise_envelope.transition_fraction",
    )

    if envelope["lower_quantile"] >= envelope["upper_quantile"]:
        raise ValueError(
            "pointwise_envelope.lower_quantile必须小于upper_quantile。"
        )

    extreme = _section(config, "extreme_intensity")
    extreme["lower_margin_fraction"] = _nonnegative_float(
        extreme.get("lower_margin_fraction", 0.03),
        "physics_constraints.extreme_intensity.lower_margin_fraction",
    )
    extreme["upper_margin_fraction"] = _nonnegative_float(
        extreme.get("upper_margin_fraction", 0.03),
        "physics_constraints.extreme_intensity.upper_margin_fraction",
    )
    extreme["transition_fraction"] = _positive_float(
        extreme.get("transition_fraction", 0.08),
        "physics_constraints.extreme_intensity.transition_fraction",
    )

    residual_guard = _section(config, "scaled_residual_guard")
    residual_guard["enabled"] = bool(
        residual_guard.get("enabled", True)
    )
    residual_guard["absolute_quantile"] = _percentile(
        residual_guard.get("absolute_quantile", 99.5),
        "physics_constraints.scaled_residual_guard.absolute_quantile",
    )
    residual_guard["limit_multiplier"] = _positive_float(
        residual_guard.get("limit_multiplier", 1.10),
        "physics_constraints.scaled_residual_guard.limit_multiplier",
    )
    residual_guard["transition_fraction"] = _positive_float(
        residual_guard.get("transition_fraction", 0.15),
        "physics_constraints.scaled_residual_guard.transition_fraction",
    )

    if residual_guard["limit_multiplier"] < 1.0:
        raise ValueError(
            "scaled_residual_guard.limit_multiplier不能小于1。"
        )

    component_weights = _section(config, "component_weights")
    defaults = {
        "position": 1.0,
        "width": 1.5,
        "sharpness": 1.5,
        "presence": 0.75,
        "local_shape": 2.0,
        "roughness": 2.0,
        "extreme": 3.0,
        "scaled_residual_guard": 1.5,
    }

    for name, default_value in defaults.items():
        component_weights[name] = _nonnegative_float(
            component_weights.get(name, default_value),
            f"physics_constraints.component_weights.{name}",
        )

    if sum(component_weights.values()) <= 0.0:
        raise ValueError("component_weights至少有一个权重大于0。")

    return config


# ---------------------------------------------------------------------------
# 训练集统计状态拟合
# ---------------------------------------------------------------------------


def _cm1_to_radius(value_cm1: float, spacing_cm1: float) -> int:
    return max(1, int(np.ceil(float(value_cm1) / spacing_cm1)))


def _gaussian_kernel_numpy(sigma_points: float) -> np.ndarray:
    sigma = max(float(sigma_points), 1.0e-6)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _smooth_tensor(
    values: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    radius = kernel.shape[-1] // 2
    padded = F.pad(values, (radius, radius), mode="reflect")
    return F.conv1d(padded, kernel)


def fit_sers_physics_constraint_state(
    *,
    training_normalized_spectra: np.ndarray,
    training_scaled_residuals: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """只用训练集拟合 D3.2 物理统计状态。"""

    config = normalize_physics_configuration(configuration)

    if not bool(config.get("enabled", False)):
        return {
            "schema_version": PHYSICS_STATE_SCHEMA_VERSION,
            "enabled": False,
            "configuration": config,
        }

    axis = _validate_axis(raman_shift)
    spectra = _validate_spectra(
        training_normalized_spectra,
        expected_length=axis.size,
        name="training_normalized_spectra",
    )
    scaled_residuals = _validate_spectra(
        training_scaled_residuals,
        expected_length=axis.size,
        name="training_scaled_residuals",
    )

    spacing = float(np.median(np.diff(axis)))

    if spacing <= 0.0:
        raise ValueError("拉曼位移轴间隔必须大于0。")

    detection = config["peak_detection"]
    smoothing_sigma_points = (
        float(detection["smoothing_sigma_cm1"]) / spacing
    )
    maximum_radius_points = _cm1_to_radius(
        detection["local_maximum_radius_cm1"], spacing
    )
    prominence_radius_points = _cm1_to_radius(
        detection["prominence_radius_cm1"], spacing
    )
    minimum_peak_separation_points = _cm1_to_radius(
        detection["minimum_peak_separation_cm1"], spacing
    )
    analysis_half_width_points = _cm1_to_radius(
        detection["analysis_half_width_cm1"], spacing
    )
    edge_exclusion_points = _cm1_to_radius(
        detection["edge_exclusion_cm1"], spacing
    )

    kernel_np = _gaussian_kernel_numpy(smoothing_sigma_points)
    kernel = torch.as_tensor(kernel_np).view(1, 1, -1)
    training_tensor = torch.as_tensor(
        spectra,
        dtype=torch.float32,
    ).unsqueeze(1)

    with torch.no_grad():
        smoothed = _smooth_tensor(training_tensor, kernel)
        local_maximum = F.max_pool1d(
            smoothed,
            kernel_size=2 * maximum_radius_points + 1,
            stride=1,
            padding=maximum_radius_points,
        )
        local_minimum = -F.max_pool1d(
            -smoothed,
            kernel_size=2 * prominence_radius_points + 1,
            stride=1,
            padding=prominence_radius_points,
        )
        prominence = smoothed - local_minimum
        candidates = smoothed >= local_maximum - float(
            detection["epsilon"]
        )
        candidates[..., :edge_exclusion_points] = False
        candidates[..., -edge_exclusion_points:] = False
        candidate_prominence = prominence[candidates]
        candidate_prominence = candidate_prominence[
            candidate_prominence > 0.0
        ]

    if candidate_prominence.numel() > 0:
        training_prominence_floor = float(
            torch.quantile(
                candidate_prominence,
                float(
                    detection["training_prominence_floor_quantile"]
                )
                / 100.0,
            ).item()
        )
    else:
        training_prominence_floor = float(
            detection["minimum_absolute_prominence"]
        )

    training_prominence_floor = max(
        training_prominence_floor,
        float(detection["minimum_absolute_prominence"]),
    )

    first_derivative = np.diff(spectra, axis=1) / spacing
    second_derivative = np.diff(
        spectra,
        n=2,
        axis=1,
    ) / (spacing**2)
    roughness = config["roughness"]

    first_derivative_limit = float(
        np.percentile(
            np.abs(first_derivative),
            roughness["first_derivative_quantile"],
        )
        * roughness["limit_multiplier"]
    )
    second_derivative_limit = float(
        np.percentile(
            np.abs(second_derivative),
            roughness["second_derivative_quantile"],
        )
        * roughness["limit_multiplier"]
    )

    epsilon = float(detection["epsilon"])
    first_derivative_limit = max(first_derivative_limit, epsilon)
    second_derivative_limit = max(second_derivative_limit, epsilon)

    envelope = config["pointwise_envelope"]
    pointwise_median = np.median(spectra, axis=0)
    pointwise_mad = 1.4826 * np.median(
        np.abs(spectra - pointwise_median[None, :]),
        axis=0,
    )
    pointwise_lower = np.percentile(
        spectra,
        envelope["lower_quantile"],
        axis=0,
    ) - envelope["mad_margin_multiplier"] * pointwise_mad
    pointwise_upper = np.percentile(
        spectra,
        envelope["upper_quantile"],
        axis=0,
    ) + envelope["mad_margin_multiplier"] * pointwise_mad

    invalid_envelope = pointwise_lower >= pointwise_upper
    pointwise_lower[invalid_envelope] = (
        pointwise_median[invalid_envelope] - epsilon
    )
    pointwise_upper[invalid_envelope] = (
        pointwise_median[invalid_envelope] + epsilon
    )

    training_minimum = float(np.min(spectra))
    training_maximum = float(np.max(spectra))
    training_range = max(training_maximum - training_minimum, epsilon)
    extreme = config["extreme_intensity"]
    allowed_minimum = (
        training_minimum
        - float(extreme["lower_margin_fraction"]) * training_range
    )
    allowed_maximum = (
        training_maximum
        + float(extreme["upper_margin_fraction"]) * training_range
    )

    residual_guard = config["scaled_residual_guard"]
    scaled_residual_abs_limit = float(
        np.percentile(
            np.abs(scaled_residuals),
            residual_guard["absolute_quantile"],
        )
        * residual_guard["limit_multiplier"]
    )
    scaled_residual_abs_limit = max(
        scaled_residual_abs_limit,
        epsilon,
    )

    return {
    "schema_version": PHYSICS_STATE_SCHEMA_VERSION,
    "enabled": True,
    "strategy": "target_adaptive_peak_morphology",
    "method_version": "d3_2_local_pathology_guard",
        "contains_fixed_peak_positions": False,
        "configuration": config,

        # 供训练入口的物理状态摘要和checkpoint元数据使用。
        # 该数量只来自训练集，不包含validation和test。
        "number_of_training_spectra": int(spectra.shape[0]),

        "original_length": int(axis.size),
        "raman_shift": axis.astype(np.float32).tolist(),
        "raman_spacing_cm1": spacing,
        "smoothing_sigma_points": float(smoothing_sigma_points),
        "local_maximum_radius_points": int(maximum_radius_points),
        "prominence_radius_points": int(prominence_radius_points),
        "minimum_peak_separation_points": int(
            minimum_peak_separation_points
        ),
        "analysis_half_width_points": int(analysis_half_width_points),
        "edge_exclusion_points": int(edge_exclusion_points),
        "training_prominence_floor": training_prominence_floor,
        "first_derivative_abs_limit": first_derivative_limit,
        "second_derivative_abs_limit": second_derivative_limit,
        "pointwise_lower_envelope": pointwise_lower.astype(
            np.float32
        ).tolist(),
        "pointwise_upper_envelope": pointwise_upper.astype(
            np.float32
        ).tolist(),
        "allowed_intensity_minimum": float(allowed_minimum),
        "allowed_intensity_maximum": float(allowed_maximum),
        "scaled_residual_abs_limit": scaled_residual_abs_limit,
    }


# ---------------------------------------------------------------------------
# 可微物理损失
# ---------------------------------------------------------------------------


class DifferentiableSersPhysicsLoss(nn.Module):
    """批量目标峰检测与 D3.2 局部病态峰约束。"""

    def __init__(
        self,
        *,
        physics_constraint_state: dict[str, Any],
        prior_residual_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        super().__init__()
        self._load_physics_state(
            physics_constraint_state=physics_constraint_state,
            padded_length=padded_length,
        )
        self._load_prior_residual_state(prior_residual_state)
        self._load_configuration_values()

    def _load_physics_state(
        self,
        *,
        physics_constraint_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        if not isinstance(physics_constraint_state, dict):
            raise TypeError("physics_constraint_state必须是字典。")

        if int(
            physics_constraint_state.get("schema_version", 0)
        ) != PHYSICS_STATE_SCHEMA_VERSION:
            raise ValueError(
                "physics_constraint_state版本不受支持。"
                "D3.2不能直接使用D3.1的物理状态断点续训。"
            )

        if not bool(physics_constraint_state.get("enabled", False)):
            raise ValueError("physics_constraint_state没有启用。")

        if physics_constraint_state.get("strategy") != (
            "target_adaptive_peak_morphology"
        ):
            raise ValueError("physics_constraint_state策略不正确。")

        if physics_constraint_state.get("method_version") != (
            "d3_2_local_pathology_guard"
        ):
            raise ValueError("physics_constraint_state不是D3.2状态。")

        if bool(
            physics_constraint_state.get(
                "contains_fixed_peak_positions",
                True,
            )
        ):
            raise ValueError("D3.2状态不能包含固定特征峰位置。")

        self.configuration = normalize_physics_configuration(
            physics_constraint_state["configuration"]
        )
        self.original_length = int(
            physics_constraint_state["original_length"]
        )
        self.padded_length = int(padded_length)

        if self.padded_length < self.original_length:
            raise ValueError("padded_length不能小于original_length。")

        axis = torch.as_tensor(
            physics_constraint_state["raman_shift"],
            dtype=torch.float32,
        ).reshape(-1)
        lower_envelope = torch.as_tensor(
            physics_constraint_state["pointwise_lower_envelope"],
            dtype=torch.float32,
        ).reshape(-1)
        upper_envelope = torch.as_tensor(
            physics_constraint_state["pointwise_upper_envelope"],
            dtype=torch.float32,
        ).reshape(-1)

        if axis.numel() != self.original_length:
            raise ValueError("physics_constraint_state拉曼轴长度不一致。")

        if (
            lower_envelope.numel() != self.original_length
            or upper_envelope.numel() != self.original_length
        ):
            raise ValueError("逐波数强度包络长度不一致。")

        if torch.any(lower_envelope >= upper_envelope):
            raise ValueError("逐波数强度包络上下界无效。")

        self.register_buffer("raman_shift", axis, persistent=False)
        self.register_buffer(
            "pointwise_lower_envelope",
            lower_envelope.view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "pointwise_upper_envelope",
            upper_envelope.view(1, 1, -1),
            persistent=False,
        )

        self.raman_spacing_cm1 = float(
            physics_constraint_state["raman_spacing_cm1"]
        )
        self.smoothing_sigma_points = float(
            physics_constraint_state["smoothing_sigma_points"]
        )
        self.local_maximum_radius_points = int(
            physics_constraint_state["local_maximum_radius_points"]
        )
        self.prominence_radius_points = int(
            physics_constraint_state["prominence_radius_points"]
        )
        self.minimum_peak_separation_points = int(
            physics_constraint_state["minimum_peak_separation_points"]
        )
        self.analysis_half_width_points = int(
            physics_constraint_state["analysis_half_width_points"]
        )
        self.edge_exclusion_points = int(
            physics_constraint_state["edge_exclusion_points"]
        )
        self.training_prominence_floor = float(
            physics_constraint_state["training_prominence_floor"]
        )
        self.first_derivative_limit = float(
            physics_constraint_state["first_derivative_abs_limit"]
        )
        self.second_derivative_limit = float(
            physics_constraint_state["second_derivative_abs_limit"]
        )
        self.allowed_intensity_minimum = float(
            physics_constraint_state["allowed_intensity_minimum"]
        )
        self.allowed_intensity_maximum = float(
            physics_constraint_state["allowed_intensity_maximum"]
        )
        self.scaled_residual_abs_limit = float(
            physics_constraint_state["scaled_residual_abs_limit"]
        )
        self.training_intensity_range = max(
            self.allowed_intensity_maximum
            - self.allowed_intensity_minimum,
            1.0e-8,
        )

        smoothing_kernel = torch.as_tensor(
            _gaussian_kernel_numpy(self.smoothing_sigma_points),
            dtype=torch.float32,
        ).view(1, 1, -1)
        self.register_buffer(
            "smoothing_kernel",
            smoothing_kernel,
            persistent=False,
        )

        offsets = torch.arange(
            -self.analysis_half_width_points,
            self.analysis_half_width_points + 1,
            dtype=torch.long,
        )
        self.register_buffer("window_offsets", offsets, persistent=False)

    def _load_prior_residual_state(
        self,
        prior_residual_state: dict[str, Any],
    ) -> None:
        if not isinstance(prior_residual_state, dict):
            raise TypeError("prior_residual_state必须是字典。")

        if not bool(prior_residual_state.get("enabled", False)):
            raise ValueError("D3.2要求启用D2.1先验残差。")

        if prior_residual_state.get("domain") != (
            "spectrum_global_minmax_normalized"
        ):
            raise ValueError("prior_residual_state数据域不正确。")

        residual_state = prior_residual_state.get(
            "residual_normalization"
        )

        if not isinstance(residual_state, dict):
            raise ValueError("prior_residual_state缺少残差归一化状态。")

        if residual_state.get("method") != "pointwise_mad_asinh":
            raise ValueError(
                "D3.2当前只支持D2.1 pointwise_mad_asinh。"
            )

        prior = torch.as_tensor(
            prior_residual_state["prior_normalized_intensity"],
            dtype=torch.float32,
        ).reshape(-1)
        pointwise_scale = torch.as_tensor(
            residual_state["pointwise_scale"],
            dtype=torch.float32,
        ).reshape(-1)

        if (
            prior.numel() != self.original_length
            or pointwise_scale.numel() != self.original_length
        ):
            raise ValueError("D2.1状态长度与D3.2统一训练轴不一致。")

        if torch.any(pointwise_scale <= 0.0):
            raise ValueError("D2.1逐波数尺度必须全部大于0。")

        self.register_buffer(
            "prior",
            prior.view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "pointwise_scale",
            pointwise_scale.view(1, 1, -1),
            persistent=False,
        )

        self.target_abs_max = float(residual_state["target_abs_max"])
        self.standardized_residual_scale = float(
            residual_state["standardized_residual_scale"]
        )
        self.asinh_normalizer = float(
            residual_state["asinh_normalizer"]
        )

        if min(
            self.target_abs_max,
            self.standardized_residual_scale,
            self.asinh_normalizer,
        ) <= 0.0:
            raise ValueError("D2.1可微反变换参数必须大于0。")

    def _load_configuration_values(self) -> None:
        detection = self.configuration["peak_detection"]
        self.maximum_peaks_per_spectrum = int(
            detection["maximum_peaks_per_spectrum"]
        )
        self.minimum_relative_prominence = float(
            detection["minimum_relative_prominence"]
        )
        self.minimum_prominence_noise_multiplier = float(
            detection["minimum_prominence_noise_multiplier"]
        )
        self.minimum_absolute_prominence = float(
            detection["minimum_absolute_prominence"]
        )
        self.positive_softplus_temperature = float(
            detection["positive_softplus_temperature"]
        )
        self.epsilon = float(detection["epsilon"])

        inverse = self.configuration["inverse_transform"]
        self.numerical_safety_limit = float(
            inverse["numerical_safety_limit"]
        )

        position = self.configuration["peak_position"]
        self.position_tolerance_cm1 = float(
            position["zero_penalty_tolerance_cm1"]
        )
        self.position_transition_cm1 = float(
            position["transition_width_cm1"]
        )

        width = self.configuration["peak_width"]
        self.narrow_relative_tolerance = float(
            width["narrow_relative_tolerance"]
        )
        self.broad_relative_tolerance = float(
            width["broad_relative_tolerance"]
        )
        self.width_absolute_tolerance_cm1 = float(
            width["absolute_tolerance_cm1"]
        )
        self.width_transition_fraction = float(
            width["transition_fraction"]
        )
        self.narrow_penalty_multiplier = float(
            width["narrow_penalty_multiplier"]
        )
        self.broad_penalty_multiplier = float(
            width["broad_penalty_multiplier"]
        )

        sharpness = self.configuration["peak_sharpness"]
        self.sharpness_relative_tolerance = float(
            sharpness["upper_relative_tolerance"]
        )
        self.sharpness_transition_fraction = float(
            sharpness["transition_fraction"]
        )

        presence = self.configuration["peak_presence"]
        self.minimum_area_ratio = float(
            presence["minimum_area_ratio"]
        )
        self.presence_transition_fraction = float(
            presence["transition_fraction"]
        )

        local_shape = self.configuration["local_shape"]
        self.profile_tolerance = float(
            local_shape["profile_tolerance"]
        )
        self.first_derivative_tolerance = float(
            local_shape["first_derivative_tolerance"]
        )
        self.second_derivative_tolerance = float(
            local_shape["second_derivative_tolerance"]
        )
        self.unimodality_tolerance = float(
            local_shape["unimodality_tolerance"]
        )
        self.total_variation_tolerance = float(
            local_shape["total_variation_tolerance"]
        )
        self.target_support_fraction = float(
            local_shape["target_support_fraction"]
        )
        self.local_shape_transition_fraction = float(
            local_shape["transition_fraction"]
        )
        self.local_shape_subweights = {
            "profile": float(local_shape["profile_weight"]),
            "first": float(local_shape["first_derivative_weight"]),
            "second": float(local_shape["second_derivative_weight"]),
            "unimodality": float(local_shape["unimodality_weight"]),
            "total_variation": float(
                local_shape["total_variation_weight"]
            ),
        }
        self.local_shape_subweight_sum = sum(
            self.local_shape_subweights.values()
        )

        negative_valley = self.configuration["negative_valley"]
        self.negative_valley_enabled = bool(
            negative_valley["enabled"]
        )
        self.negative_valley_noise_multiplier = float(
            negative_valley["noise_margin_multiplier"]
        )
        self.negative_valley_minimum_margin_fraction = float(
            negative_valley["minimum_margin_fraction"]
        )
        self.negative_valley_center_sigma_fraction = float(
            negative_valley["center_weight_sigma_fraction"]
        )
        self.negative_valley_transition_fraction = float(
            negative_valley["transition_fraction"]
        )
        self.negative_valley_weight = float(
            negative_valley["weight_within_extreme"]
        )

        aggregation = self.configuration["pathology_aggregation"]
        self.pathology_mean_weight = float(
            aggregation["mean_weight"]
        )
        self.pathology_topk_weight = float(
            aggregation["topk_weight"]
        )
        self.pathology_topk_fraction = float(
            aggregation["topk_fraction"]
        )

        roughness = self.configuration["roughness"]
        self.first_transition_fraction = float(
            roughness["first_transition_fraction"]
        )
        self.second_transition_fraction = float(
            roughness["second_transition_fraction"]
        )
        self.first_derivative_weight = float(
            roughness["first_derivative_weight"]
        )
        self.second_derivative_weight = float(
            roughness["second_derivative_weight"]
        )
        self.roughness_weight_sum = (
            self.first_derivative_weight
            + self.second_derivative_weight
        )

        envelope = self.configuration["pointwise_envelope"]
        self.pointwise_envelope_enabled = bool(envelope["enabled"])
        self.pointwise_transition_fraction = float(
            envelope["transition_fraction"]
        )

        extreme = self.configuration["extreme_intensity"]
        self.extreme_transition_fraction = float(
            extreme["transition_fraction"]
        )

        residual_guard = self.configuration["scaled_residual_guard"]
        self.scaled_residual_guard_enabled = bool(
            residual_guard["enabled"]
        )
        self.scaled_residual_transition_fraction = float(
            residual_guard["transition_fraction"]
        )

        timestep = self.configuration["timestep_weighting"]
        self.timestep_mode = str(timestep["mode"])
        self.minimum_timestep_weight = float(
            timestep["minimum_weight"]
        )
        self.pathology_minimum_timestep_weight = float(
            timestep["pathology_minimum_weight"]
        )

        weights = self.configuration["component_weights"]
        self.component_weights = {
            "position_loss": float(weights["position"]),
            "width_loss": float(weights["width"]),
            "sharpness_loss": float(weights["sharpness"]),
            "presence_loss": float(weights["presence"]),
            "local_shape_loss": float(weights["local_shape"]),
            "roughness_loss": float(weights["roughness"]),
            "extreme_loss": float(weights["extreme"]),
            "scaled_residual_guard_loss": float(
                weights["scaled_residual_guard"]
            ),
        }
        self.component_weight_sum = sum(
            self.component_weights.values()
        )

    def inverse_scaled_residual(self, scaled: torch.Tensor) -> torch.Tensor:
        """按 D2.1 公式可微恢复完整归一化光谱。"""

        if scaled.ndim != 3 or scaled.shape[1] != 1:
            raise ValueError("scaled必须为单通道[B,1,L]。")

        if scaled.shape[-1] != self.padded_length:
            raise ValueError("scaled长度与padded_length不一致。")

        scaled = scaled[..., : self.original_length]
        argument = (
            scaled / self.target_abs_max * self.asinh_normalizer
        )
        argument = torch.clamp(
            argument,
            min=-self.numerical_safety_limit,
            max=self.numerical_safety_limit,
        )
        standardized = (
            self.standardized_residual_scale * torch.sinh(argument)
        )

        return self.prior + self.pointwise_scale * standardized

    def _smooth(self, values: torch.Tensor) -> torch.Tensor:
        return _smooth_tensor(
            values,
            self.smoothing_kernel.to(
                device=values.device,
                dtype=values.dtype,
            ),
        )

    def _select_target_peaks(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量选择每条目标光谱的 K 个可靠峰。"""

        with torch.no_grad():
            smoothed = self._smooth(target)
            local_maximum = F.max_pool1d(
                smoothed,
                kernel_size=2 * self.local_maximum_radius_points + 1,
                stride=1,
                padding=self.local_maximum_radius_points,
            )
            local_minimum = -F.max_pool1d(
                -smoothed,
                kernel_size=2 * self.prominence_radius_points + 1,
                stride=1,
                padding=self.prominence_radius_points,
            )
            prominence = (smoothed - local_minimum).squeeze(1)
            candidates = (
                smoothed >= local_maximum - self.epsilon
            ).squeeze(1)
            candidates[:, : self.edge_exclusion_points] = False
            candidates[:, -self.edge_exclusion_points :] = False

            first_difference = torch.diff(
                target.squeeze(1),
                dim=-1,
            ).abs()
            noise_scale = torch.median(
                first_difference,
                dim=-1,
            ).values / 0.6745
            maximum_prominence = prominence.amax(
                dim=-1,
                keepdim=True,
            )
            threshold = torch.maximum(
                maximum_prominence * self.minimum_relative_prominence,
                noise_scale[:, None]
                * self.minimum_prominence_noise_multiplier,
            )
            threshold = torch.maximum(
                threshold,
                torch.full_like(
                    threshold,
                    self.training_prominence_floor,
                ),
            )
            threshold = torch.maximum(
                threshold,
                torch.full_like(
                    threshold,
                    self.minimum_absolute_prominence,
                ),
            )

            negative_infinity = torch.full_like(
                prominence,
                -torch.inf,
            )
            working_scores = torch.where(
                candidates & (prominence >= threshold),
                prominence,
                negative_infinity,
            )
            positions = torch.arange(
                target.shape[-1],
                device=target.device,
            ).view(1, -1)
            selected_indices = []
            selected_valid = []

            # 只循环固定 K 次；batch 和候选峰全部并行。
            for _ in range(self.maximum_peaks_per_spectrum):
                values, indices = torch.max(working_scores, dim=-1)
                valid = torch.isfinite(values)
                selected_indices.append(indices)
                selected_valid.append(valid)

                suppress = (
                    torch.abs(positions - indices[:, None])
                    <= self.minimum_peak_separation_points
                ) & valid[:, None]
                working_scores = working_scores.masked_fill(
                    suppress,
                    -torch.inf,
                )

            peak_indices = torch.stack(selected_indices, dim=1)
            peak_valid = torch.stack(selected_valid, dim=1)

        return peak_indices, peak_valid, smoothed

    @staticmethod
    def _gather_windows(
        values: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, length = values.shape
        window_indices = indices[:, :, None] + offsets.view(1, 1, -1)
        inside = (window_indices >= 0) & (window_indices < length)
        window_indices = window_indices.clamp(0, length - 1)
        expanded = values.squeeze(1)[:, None, :].expand(
            batch_size,
            indices.shape[1],
            length,
        )
        windows = torch.gather(expanded, 2, window_indices)

        return windows, inside

    @staticmethod
    def _linear_baseline(windows: torch.Tensor) -> torch.Tensor:
        number_of_points = windows.shape[-1]
        fraction = torch.linspace(
            0.0,
            1.0,
            number_of_points,
            device=windows.device,
            dtype=windows.dtype,
        ).view(1, 1, -1)

        return (
            windows[..., :1] * (1.0 - fraction)
            + windows[..., -1:] * fraction
        )

    def _positive_profile(
        self,
        windows: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.positive_softplus_temperature

        return temperature * F.softplus(
            (windows - baseline) / temperature
        )

    @staticmethod
    def _masked_peak_mean(
        values: torch.Tensor,
        peak_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = peak_valid.to(values.dtype)
        denominator = valid.sum(dim=1).clamp_min(1.0)

        return (values * valid).sum(dim=1) / denominator

    @staticmethod
    def _dead_zone_penalty(
        value: torch.Tensor,
        limit: torch.Tensor | float,
        transition: torch.Tensor | float,
        epsilon: float,
    ) -> torch.Tensor:
        limit_tensor = torch.as_tensor(
            limit,
            device=value.device,
            dtype=value.dtype,
        )
        transition_tensor = torch.as_tensor(
            transition,
            device=value.device,
            dtype=value.dtype,
        ).clamp_min(epsilon)

        return torch.square(
            F.relu(value - limit_tensor) / transition_tensor
        )

    def _aggregate_pathology(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """对每个样本执行平均 + 最坏 top-k 聚合。"""

        flattened = values.flatten(start_dim=1)
        mean_value = flattened.mean(dim=1)
        number_of_points = flattened.shape[1]
        topk_count = max(
            1,
            int(np.ceil(number_of_points * self.pathology_topk_fraction)),
        )
        topk_value = torch.topk(
            flattened,
            k=topk_count,
            dim=1,
            largest=True,
            sorted=False,
        ).values.mean(dim=1)
        weight_sum = (
            self.pathology_mean_weight + self.pathology_topk_weight
        )

        return (
            self.pathology_mean_weight * mean_value
            + self.pathology_topk_weight * topk_value
        ) / max(weight_sum, self.epsilon)

    def _peak_components(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        peak_indices, peak_valid, target_smoothed = (
            self._select_target_peaks(target)
        )
        predicted_smoothed = self._smooth(predicted)
        offsets = self.window_offsets.to(predicted.device)

        predicted_windows, inside = self._gather_windows(
            predicted,
            peak_indices,
            offsets,
        )
        target_windows, _ = self._gather_windows(
            target,
            peak_indices,
            offsets,
        )
        predicted_smooth_windows, _ = self._gather_windows(
            predicted_smoothed,
            peak_indices,
            offsets,
        )
        target_smooth_windows, _ = self._gather_windows(
            target_smoothed,
            peak_indices,
            offsets,
        )
        axis_values = self.raman_shift.view(1, 1, -1).expand(
            predicted.shape[0],
            peak_indices.shape[1],
            self.original_length,
        )
        axis_windows = torch.gather(
            axis_values,
            2,
            (
                peak_indices[:, :, None] + offsets.view(1, 1, -1)
            ).clamp(0, self.original_length - 1),
        )

        window_valid = inside & peak_valid[:, :, None]
        predicted_baseline = self._linear_baseline(predicted_windows)
        target_baseline = self._linear_baseline(target_windows)
        predicted_profile = self._positive_profile(
            predicted_smooth_windows,
            self._linear_baseline(predicted_smooth_windows),
        )
        target_profile = self._positive_profile(
            target_smooth_windows,
            self._linear_baseline(target_smooth_windows),
        )
        valid_float = window_valid.to(predicted.dtype)
        predicted_profile = predicted_profile * valid_float
        target_profile = target_profile * valid_float
        predicted_mass = predicted_profile.sum(dim=-1).clamp_min(
            self.epsilon
        )
        target_mass = target_profile.sum(dim=-1).clamp_min(self.epsilon)
        predicted_area = predicted_mass * self.raman_spacing_cm1
        target_area = target_mass * self.raman_spacing_cm1
        predicted_centroid = (
            predicted_profile * axis_windows
        ).sum(dim=-1) / predicted_mass
        target_centroid = (
            target_profile * axis_windows
        ).sum(dim=-1) / target_mass
        predicted_variance = (
            predicted_profile
            * torch.square(axis_windows - predicted_centroid[..., None])
        ).sum(dim=-1) / predicted_mass
        target_variance = (
            target_profile
            * torch.square(axis_windows - target_centroid[..., None])
        ).sum(dim=-1) / target_mass
        predicted_width = 2.354820045 * torch.sqrt(
            predicted_variance.clamp_min(self.epsilon)
        )
        target_width = 2.354820045 * torch.sqrt(
            target_variance.clamp_min(self.epsilon)
        )
        predicted_height = predicted_profile.amax(dim=-1).clamp_min(
            self.epsilon
        )
        target_height = target_profile.amax(dim=-1).clamp_min(
            self.epsilon
        )

        predicted_second = torch.diff(
            predicted_profile,
            n=2,
            dim=-1,
        ) / (self.raman_spacing_cm1**2)
        target_second = torch.diff(
            target_profile,
            n=2,
            dim=-1,
        ) / (self.raman_spacing_cm1**2)
        predicted_sharpness = predicted_second.abs().amax(
            dim=-1
        ) / predicted_height
        target_sharpness = target_second.abs().amax(
            dim=-1
        ) / target_height

        position_loss = self._dead_zone_penalty(
            torch.abs(predicted_centroid - target_centroid),
            self.position_tolerance_cm1,
            self.position_transition_cm1,
            self.epsilon,
        )

        lower_width = torch.maximum(
            target_width * (1.0 - self.narrow_relative_tolerance),
            target_width - self.width_absolute_tolerance_cm1,
        )
        upper_width = torch.minimum(
            target_width * (1.0 + self.broad_relative_tolerance),
            target_width + self.width_absolute_tolerance_cm1,
        )
        width_transition = (
            target_width * self.width_transition_fraction
        ).clamp_min(self.raman_spacing_cm1)
        narrow_loss = torch.square(
            F.relu(lower_width - predicted_width) / width_transition
        )
        broad_loss = torch.square(
            F.relu(predicted_width - upper_width) / width_transition
        )
        width_loss = (
            self.narrow_penalty_multiplier * narrow_loss
            + self.broad_penalty_multiplier * broad_loss
        )

        allowed_sharpness = target_sharpness * (
            1.0 + self.sharpness_relative_tolerance
        )
        sharpness_transition = (
            target_sharpness * self.sharpness_transition_fraction
        ).clamp_min(self.epsilon)
        sharpness_loss = torch.square(
            F.relu(predicted_sharpness - allowed_sharpness)
            / sharpness_transition
        )

        required_area = target_area * self.minimum_area_ratio
        area_transition = (
            target_area * self.presence_transition_fraction
        ).clamp_min(self.epsilon)
        presence_loss = torch.square(
            F.relu(required_area - predicted_area) / area_transition
        )

        predicted_normalized = predicted_profile / predicted_height[..., None]
        target_normalized = target_profile / target_height[..., None]
        profile_error = torch.square(
            predicted_normalized - target_normalized
        )
        profile_error = (
            profile_error * valid_float
        ).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)
        profile_loss = self._dead_zone_penalty(
            torch.sqrt(profile_error + self.epsilon),
            self.profile_tolerance,
            self.profile_tolerance
            * self.local_shape_transition_fraction
            + self.epsilon,
            self.epsilon,
        )

        predicted_first_normalized = torch.diff(
            predicted_normalized,
            dim=-1,
        )
        target_first_normalized = torch.diff(
            target_normalized,
            dim=-1,
        )
        first_error = torch.mean(
            torch.abs(
                predicted_first_normalized - target_first_normalized
            ),
            dim=-1,
        )
        first_loss = self._dead_zone_penalty(
            first_error,
            self.first_derivative_tolerance,
            self.first_derivative_tolerance
            * self.local_shape_transition_fraction
            + self.epsilon,
            self.epsilon,
        )

        predicted_second_normalized = torch.diff(
            predicted_normalized,
            n=2,
            dim=-1,
        )
        target_second_normalized = torch.diff(
            target_normalized,
            n=2,
            dim=-1,
        )
        second_error = torch.mean(
            torch.abs(
                predicted_second_normalized
                - target_second_normalized
            ),
            dim=-1,
        )
        second_loss = self._dead_zone_penalty(
            second_error,
            self.second_derivative_tolerance,
            self.second_derivative_tolerance
            * self.local_shape_transition_fraction
            + self.epsilon,
            self.epsilon,
        )

        midpoint_axis = 0.5 * (
            axis_windows[..., 1:] + axis_windows[..., :-1]
        )
        target_support = (
            torch.maximum(
                target_normalized[..., 1:],
                target_normalized[..., :-1],
            )
            >= self.target_support_fraction
        ).to(predicted.dtype)
        left_mask = (
            midpoint_axis < target_centroid[..., None]
        ).to(predicted.dtype) * target_support
        right_mask = (
            midpoint_axis > target_centroid[..., None]
        ).to(predicted.dtype) * target_support
        predicted_unimodal_violation = (
            F.relu(-predicted_first_normalized) * left_mask
            + F.relu(predicted_first_normalized) * right_mask
        )
        target_unimodal_violation = (
            F.relu(-target_first_normalized) * left_mask
            + F.relu(target_first_normalized) * right_mask
        )
        unimodal_excess = F.relu(
            predicted_unimodal_violation
            - target_unimodal_violation
            - self.unimodality_tolerance
        )
        unimodality_loss = torch.square(unimodal_excess).sum(
            dim=-1
        ) / (left_mask + right_mask).sum(dim=-1).clamp_min(1.0)

        predicted_total_variation = torch.abs(
            predicted_first_normalized
        ).sum(dim=-1)
        target_total_variation = torch.abs(
            target_first_normalized
        ).sum(dim=-1)
        allowed_total_variation = target_total_variation * (
            1.0 + self.total_variation_tolerance
        )
        total_variation_transition = (
            target_total_variation
            * self.local_shape_transition_fraction
        ).clamp_min(self.epsilon)
        total_variation_loss = torch.square(
            F.relu(
                predicted_total_variation - allowed_total_variation
            )
            / total_variation_transition
        )

        local_shape_loss = (
            self.local_shape_subweights["profile"] * profile_loss
            + self.local_shape_subweights["first"] * first_loss
            + self.local_shape_subweights["second"] * second_loss
            + self.local_shape_subweights["unimodality"]
            * unimodality_loss
            + self.local_shape_subweights["total_variation"]
            * total_variation_loss
        ) / max(self.local_shape_subweight_sum, self.epsilon)

        if self.negative_valley_enabled:
            target_noise = torch.median(
                torch.abs(torch.diff(target_windows, dim=-1)),
                dim=-1,
            ).values / 0.6745
            minimum_margin = (
                target_height * self.negative_valley_minimum_margin_fraction
            )
            margin = torch.maximum(
                target_noise * self.negative_valley_noise_multiplier,
                minimum_margin,
            )
            half_width = max(self.analysis_half_width_points, 1)
            normalized_offset = (
                offsets.to(predicted.dtype)
                / float(half_width)
            )
            sigma = self.negative_valley_center_sigma_fraction
            center_weight = torch.exp(
                -0.5 * torch.square(normalized_offset / sigma)
            ).view(1, 1, -1)
            valley_transition = (
                target_height
                * self.negative_valley_transition_fraction
            ).clamp_min(self.epsilon)
            valley_depth = F.relu(
                predicted_baseline
                - margin[..., None]
                - predicted_windows
            )
            valley_point_loss = torch.square(
                valley_depth / valley_transition[..., None]
            ) * center_weight * valid_float
            negative_valley_loss = valley_point_loss.sum(
                dim=-1
            ) / (center_weight * valid_float).sum(dim=-1).clamp_min(1.0)
        else:
            negative_valley_loss = torch.zeros_like(position_loss)

        components = {
            "position_loss": self._masked_peak_mean(
                position_loss,
                peak_valid,
            ),
            "width_loss": self._masked_peak_mean(
                width_loss,
                peak_valid,
            ),
            "sharpness_loss": self._masked_peak_mean(
                sharpness_loss,
                peak_valid,
            ),
            "presence_loss": self._masked_peak_mean(
                presence_loss,
                peak_valid,
            ),
            "local_shape_loss": self._masked_peak_mean(
                local_shape_loss,
                peak_valid,
            ),
        }
        negative_valley_per_sample = self._masked_peak_mean(
            negative_valley_loss,
            peak_valid,
        )
        detected_peaks = peak_valid.to(predicted.dtype).sum(dim=1)

        return components, negative_valley_per_sample, detected_peaks

    def _roughness_loss(self, predicted: torch.Tensor) -> torch.Tensor:
        first = torch.diff(predicted, dim=-1) / self.raman_spacing_cm1
        second = torch.diff(
            predicted,
            n=2,
            dim=-1,
        ) / (self.raman_spacing_cm1**2)
        first_transition = max(
            self.first_derivative_limit
            * self.first_transition_fraction,
            self.epsilon,
        )
        second_transition = max(
            self.second_derivative_limit
            * self.second_transition_fraction,
            self.epsilon,
        )
        first_violation = torch.square(
            F.relu(
                first.abs() - self.first_derivative_limit
            )
            / first_transition
        )
        second_violation = torch.square(
            F.relu(
                second.abs() - self.second_derivative_limit
            )
            / second_transition
        )

        return (
            self.first_derivative_weight
            * self._aggregate_pathology(first_violation)
            + self.second_derivative_weight
            * self._aggregate_pathology(second_violation)
        ) / max(self.roughness_weight_sum, self.epsilon)

    def _extreme_loss(
        self,
        predicted: torch.Tensor,
        negative_valley_loss: torch.Tensor,
    ) -> torch.Tensor:
        pointwise_width = (
            self.pointwise_upper_envelope
            - self.pointwise_lower_envelope
        ).clamp_min(self.epsilon)
        pointwise_transition = (
            pointwise_width * self.pointwise_transition_fraction
        ).clamp_min(self.epsilon)

        if self.pointwise_envelope_enabled:
            pointwise_violation = torch.square(
                F.relu(
                    self.pointwise_lower_envelope - predicted
                )
                / pointwise_transition
            ) + torch.square(
                F.relu(
                    predicted - self.pointwise_upper_envelope
                )
                / pointwise_transition
            )
        else:
            pointwise_violation = torch.zeros_like(predicted)

        global_transition = max(
            self.training_intensity_range
            * self.extreme_transition_fraction,
            self.epsilon,
        )
        global_violation = torch.square(
            F.relu(self.allowed_intensity_minimum - predicted)
            / global_transition
        ) + torch.square(
            F.relu(predicted - self.allowed_intensity_maximum)
            / global_transition
        )
        global_pathology = self._aggregate_pathology(
            pointwise_violation + global_violation
        )

        return (
            global_pathology
            + self.negative_valley_weight * negative_valley_loss
        ) / (1.0 + self.negative_valley_weight)

    def _scaled_residual_guard_loss(
        self,
        predicted_scaled_residual: torch.Tensor,
    ) -> torch.Tensor:
        if not self.scaled_residual_guard_enabled:
            return torch.zeros(
                predicted_scaled_residual.shape[0],
                device=predicted_scaled_residual.device,
                dtype=predicted_scaled_residual.dtype,
            )

        transition = max(
            self.scaled_residual_abs_limit
            * self.scaled_residual_transition_fraction,
            self.epsilon,
        )
        violation = torch.square(
            F.relu(
                predicted_scaled_residual[..., : self.original_length].abs()
                - self.scaled_residual_abs_limit
            )
            / transition
        )

        return self._aggregate_pathology(violation)

    def _timestep_weights(
        self,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.timestep_mode == "none":
            morphology = torch.ones(
                timesteps.shape[0],
                device=timesteps.device,
                dtype=alphas_cumprod.dtype,
            )
        else:
            morphology = torch.sqrt(
                alphas_cumprod.gather(0, timesteps).clamp_min(0.0)
            )
            morphology = morphology.clamp_min(
                self.minimum_timestep_weight
            )

        pathology = morphology.clamp_min(
            self.pathology_minimum_timestep_weight
        )

        return morphology, pathology

    def forward(
        self,
        *,
        predicted_scaled_residual: torch.Tensor,
        target_scaled_residual: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_scaled_residual.shape != target_scaled_residual.shape:
            raise ValueError("预测残差和目标残差形状必须一致。")

        if predicted_scaled_residual.ndim != 3:
            raise ValueError("残差张量必须为[B,1,L]。")

        if timesteps.ndim != 1:
            raise ValueError("timesteps必须是一维张量。")

        if timesteps.shape[0] != predicted_scaled_residual.shape[0]:
            raise ValueError("timesteps批量大小与残差批量不一致。")

        predicted = self.inverse_scaled_residual(
            predicted_scaled_residual
        )
        target = self.inverse_scaled_residual(target_scaled_residual)
        peak_components, negative_valley_loss, detected_peaks = (
            self._peak_components(predicted, target)
        )
        roughness_loss = self._roughness_loss(predicted)
        extreme_loss = self._extreme_loss(
            predicted,
            negative_valley_loss,
        )
        scaled_residual_guard_loss = (
            self._scaled_residual_guard_loss(
                predicted_scaled_residual
            )
        )

        components_per_sample = {
            **peak_components,
            "roughness_loss": roughness_loss,
            "extreme_loss": extreme_loss,
            "scaled_residual_guard_loss": (
                scaled_residual_guard_loss
            ),
        }
        morphology_weighted_sum = torch.zeros_like(roughness_loss)
        pathology_weighted_sum = torch.zeros_like(roughness_loss)

        for name in MORPHOLOGY_COMPONENT_NAMES:
            morphology_weighted_sum = (
                morphology_weighted_sum
                + self.component_weights[name]
                * components_per_sample[name]
            )

        for name in PATHOLOGY_COMPONENT_NAMES:
            pathology_weighted_sum = (
                pathology_weighted_sum
                + self.component_weights[name]
                * components_per_sample[name]
            )

        physics_raw_per_sample = (
            morphology_weighted_sum + pathology_weighted_sum
        ) / max(self.component_weight_sum, self.epsilon)
        morphology_timestep_weight, pathology_timestep_weight = (
            self._timestep_weights(timesteps, alphas_cumprod)
        )
        physics_timestep_weighted_per_sample = (
            morphology_weighted_sum * morphology_timestep_weight
            + pathology_weighted_sum * pathology_timestep_weight
        ) / max(self.component_weight_sum, self.epsilon)

        result = {
            name: values.mean()
            for name, values in components_per_sample.items()
        }
        result.update(
            {
                "physics_raw_loss": physics_raw_per_sample.mean(),
                "physics_timestep_weighted_loss": (
                    physics_timestep_weighted_per_sample.mean()
                ),
                "mean_timestep_weight": (
                    morphology_timestep_weight.mean()
                ),
                "mean_pathology_timestep_weight": (
                    pathology_timestep_weight.mean()
                ),
                "mean_detected_peaks": detected_peaks.mean(),
                "negative_valley_loss": negative_valley_loss.mean(),
            }
        )

        return result
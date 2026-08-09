"""
D3.5：先验特征峰残差软包络限制器。

目的：
1. 以训练集逐点中位数先验为参考；
2. 自动识别先验谱中的主要正特征峰窗口；
3. 只用训练集统计每个峰区允许的残差变化范围；
4. 生成时仅在峰区对超过范围的残差进行tanh软压缩；
5. 非峰区不修改，保留合理的基线、噪声和样本差异。

该模块的计算域为：
训练集全局归一化后的真实光谱域，而不是原始强度域，
也不是pointwise_mad_asinh缩放残差域。

设训练集逐点中位数先验为 P，生成谱为 X_gen，则残差为：

R_gen = X_gen - P

在特征峰窗口内，对每个波数点 j 使用：

B_j = k * Q_j

R_limited,j = B_j * tanh(R_gen,j / B_j)

其中：
- Q_j：训练集在该点相对先验残差绝对值的指定分位数；
- k：feature_peak_residual_limiter.limit_multiplier；
- B_j：该点允许的残差软上限。

因此：
- k 越小，生成峰高越接近训练集先验；
- k 越大，允许的峰强差异越大；
- 该限制器不改变非特征峰区域；
- 不使用验证集或测试集拟合任何统计量。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


FEATURE_PEAK_RESIDUAL_LIMITER_STATE_SCHEMA_VERSION = 1


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是数值。") from error
    if not np.isfinite(result):
        raise ValueError(f"{field_name}必须是有限数值。")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name}必须大于0。")
    return result


def _positive_integer(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是正整数。") from error
    if result <= 0:
        raise ValueError(f"{field_name}必须是正整数。")
    return result


def _percentile(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.0 < result < 100.0:
        raise ValueError(f"{field_name}必须在(0, 100)范围内。")
    return result


def _fraction(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{field_name}必须在(0, 1]范围内。")
    return result


def normalize_feature_peak_residual_limiter_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """补全并校验 D3.5 生成端峰区残差限制器配置。"""

    if configuration is None:
        configuration = {"enabled": False}
    if not isinstance(configuration, dict):
        raise TypeError("feature_peak_residual_limiter必须是字典。")

    config = deepcopy(configuration)
    config["enabled"] = bool(config.get("enabled", False))
    if not config["enabled"]:
        return config

    strategy = str(
        config.get("strategy", "training_prior_peak_residual_soft_limit")
    ).strip().lower()
    if strategy != "training_prior_peak_residual_soft_limit":
        raise ValueError(
            "feature_peak_residual_limiter.strategy必须为"
            "training_prior_peak_residual_soft_limit。"
        )

    config["strategy"] = strategy
    config["method_version"] = "d3_5_training_prior_peak_residual_soft_limit"
    config["limit_multiplier"] = _positive_float(
        config.get("limit_multiplier", 1.20),
        "feature_peak_residual_limiter.limit_multiplier",
    )
    config["absolute_residual_quantile"] = _percentile(
        config.get("absolute_residual_quantile", 99.0),
        "feature_peak_residual_limiter.absolute_residual_quantile",
    )
    config["minimum_limit_fraction"] = _fraction(
        config.get("minimum_limit_fraction", 0.10),
        "feature_peak_residual_limiter.minimum_limit_fraction",
    )
    config["soft_transition_fraction"] = _fraction(
        config.get("soft_transition_fraction", 0.15),
        "feature_peak_residual_limiter.soft_transition_fraction",
    )
    config["smoothing_sigma_cm1"] = _positive_float(
        config.get("smoothing_sigma_cm1", 2.0),
        "feature_peak_residual_limiter.smoothing_sigma_cm1",
    )
    config["minimum_peak_distance_cm1"] = _positive_float(
        config.get("minimum_peak_distance_cm1", 10.0),
        "feature_peak_residual_limiter.minimum_peak_distance_cm1",
    )
    config["peak_window_half_width_cm1"] = _positive_float(
        config.get("peak_window_half_width_cm1", 15.0),
        "feature_peak_residual_limiter.peak_window_half_width_cm1",
    )
    config["peak_prominence_quantile"] = _percentile(
        config.get("peak_prominence_quantile", 70.0),
        "feature_peak_residual_limiter.peak_prominence_quantile",
    )
    config["minimum_peak_prominence"] = _positive_float(
        config.get("minimum_peak_prominence", 1.0e-4),
        "feature_peak_residual_limiter.minimum_peak_prominence",
    )
    config["maximum_peak_count"] = _positive_integer(
        config.get("maximum_peak_count", 12),
        "feature_peak_residual_limiter.maximum_peak_count",
    )
    config["epsilon"] = _positive_float(
        config.get("epsilon", 1.0e-8),
        "feature_peak_residual_limiter.epsilon",
    )
    return config


def _validate_axis(raman_shift: np.ndarray) -> np.ndarray:
    axis = np.asarray(raman_shift, dtype=np.float64).reshape(-1)
    if axis.size < 8:
        raise ValueError("D3.5拉曼位移轴至少需要8个点。")
    if not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
        raise ValueError("D3.5拉曼位移轴必须为有限且严格递增的数组。")
    return axis


def _validate_spectra(
    spectra: np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != length:
        raise ValueError(f"{name}必须是至少含两条光谱的二维[N, {length}]数组。")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}包含NaN或无穷值。")
    return values


def _cm1_to_points(width_cm1: float, axis: np.ndarray) -> int:
    spacing = float(np.median(np.diff(axis)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("拉曼位移间隔必须为正有限数值。")
    return max(1, int(round(width_cm1 / spacing)))


def _select_peak_indices(
    *,
    prior: np.ndarray,
    axis: np.ndarray,
    configuration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """从训练集逐点中位数先验中自动选择显著正峰。"""

    sigma = _cm1_to_points(configuration["smoothing_sigma_cm1"], axis)
    distance = _cm1_to_points(configuration["minimum_peak_distance_cm1"], axis)
    smoothed = gaussian_filter1d(prior, sigma=float(sigma), mode="nearest")
    all_indices, properties = find_peaks(smoothed, distance=distance, prominence=0.0)
    prominences = np.asarray(properties.get("prominences", []), dtype=np.float64)
    if all_indices.size == 0 or prominences.size == 0:
        raise RuntimeError("D3.5无法从训练集先验谱中自动识别正特征峰。")

    threshold = max(
        float(np.quantile(prominences, configuration["peak_prominence_quantile"] / 100.0)),
        float(configuration["minimum_peak_prominence"]),
    )
    retained = np.flatnonzero(prominences >= threshold)
    if retained.size == 0:
        raise RuntimeError("D3.5没有先验峰达到显著度阈值。")

    ranked = retained[np.argsort(prominences[retained])[::-1]]
    ranked = ranked[: int(configuration["maximum_peak_count"])]
    selected = np.sort(all_indices[ranked]).astype(np.int64, copy=False)
    selected_prominences = prominences[ranked]
    return selected, selected_prominences, threshold


def fit_feature_peak_residual_limiter_state(
    *,
    training_normalized_spectra: np.ndarray,
    prior_normalized_intensity: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """仅用训练集拟合 D3.5 的峰区残差软上限状态。"""

    config = normalize_feature_peak_residual_limiter_configuration(configuration)
    if not bool(config.get("enabled", False)):
        return {
            "schema_version": FEATURE_PEAK_RESIDUAL_LIMITER_STATE_SCHEMA_VERSION,
            "enabled": False,
            "configuration": config,
        }

    axis = _validate_axis(raman_shift)
    spectra = _validate_spectra(
        training_normalized_spectra,
        length=axis.size,
        name="training_normalized_spectra",
    )
    prior = np.asarray(prior_normalized_intensity, dtype=np.float64).reshape(-1)
    if prior.size != axis.size or not np.isfinite(prior).all():
        raise ValueError("prior_normalized_intensity必须是与拉曼位移轴等长的有限数组。")

    peak_indices, peak_prominences, prominence_threshold = _select_peak_indices(
        prior=prior,
        axis=axis,
        configuration=config,
    )
    half_width = _cm1_to_points(config["peak_window_half_width_cm1"], axis)
    mask = np.zeros(axis.size, dtype=bool)
    for index in peak_indices:
        mask[max(0, index - half_width) : min(axis.size, index + half_width + 1)] = True
    if not mask.any():
        raise RuntimeError("D3.5特征峰窗口为空。")

    absolute_residuals = np.abs(spectra - prior[np.newaxis, :])
    quantile = float(config["absolute_residual_quantile"])
    pointwise_quantile = np.quantile(absolute_residuals, quantile / 100.0, axis=0)
    global_peak_quantile = float(np.quantile(absolute_residuals[:, mask], quantile / 100.0))
    limit_floor = max(
        global_peak_quantile * float(config["minimum_limit_fraction"]),
        float(config["epsilon"]),
    )
    pointwise_limit = np.maximum(pointwise_quantile, limit_floor)
    pointwise_limit *= float(config["limit_multiplier"])
    if not np.isfinite(pointwise_limit).all() or np.any(pointwise_limit[mask] <= 0.0):
        raise RuntimeError("D3.5拟合得到的峰区残差上限不合理。")

    return {
        "schema_version": FEATURE_PEAK_RESIDUAL_LIMITER_STATE_SCHEMA_VERSION,
        "enabled": True,
        "strategy": config["strategy"],
        "method_version": config["method_version"],
        "configuration": config,
        "number_of_training_spectra": int(spectra.shape[0]),
        "original_length": int(axis.size),
        "raman_shift": axis.tolist(),
        "prior_normalized_intensity": prior.tolist(),
        "selected_peak_indices": peak_indices.tolist(),
        "selected_peak_raman_shifts": axis[peak_indices].tolist(),
        "selected_peak_prominences": peak_prominences.tolist(),
        "peak_prominence_threshold": float(prominence_threshold),
        "feature_peak_mask": mask.astype(np.uint8).tolist(),
        "feature_peak_point_count": int(mask.sum()),
        "absolute_residual_quantile": quantile,
        "training_feature_peak_absolute_residual_quantile": global_peak_quantile,
        "minimum_limit_before_multiplier": limit_floor,
        "pointwise_soft_limit": pointwise_limit.tolist(),
    }


class FeaturePeakResidualLimiter:
    """从检查点恢复，并在采样终点约束峰区残差。"""

    def __init__(self, *, feature_peak_residual_limiter_state: dict[str, Any]) -> None:
        state = feature_peak_residual_limiter_state
        if not isinstance(state, dict):
            raise TypeError("feature_peak_residual_limiter_state必须是字典。")
        if int(state.get("schema_version", 0)) != FEATURE_PEAK_RESIDUAL_LIMITER_STATE_SCHEMA_VERSION:
            raise ValueError("feature_peak_residual_limiter_state版本不受支持。")
        if not bool(state.get("enabled", False)):
            raise ValueError("feature_peak_residual_limiter_state没有启用。")

        self.configuration = normalize_feature_peak_residual_limiter_configuration(state["configuration"])
        if not bool(self.configuration.get("enabled", False)):
            raise ValueError("D3.5状态与配置启用状态不一致。")
        self.original_length = int(state["original_length"])
        self.raman_shift = _validate_axis(np.asarray(state["raman_shift"], dtype=np.float64))
        self.prior = np.asarray(state["prior_normalized_intensity"], dtype=np.float64).reshape(-1)
        self.mask = np.asarray(state["feature_peak_mask"], dtype=np.uint8).reshape(-1).astype(bool)
        self.limits = np.asarray(state["pointwise_soft_limit"], dtype=np.float64).reshape(-1)
        self.selected_peak_indices = np.asarray(
            state["selected_peak_indices"],
            dtype=np.int64,
        ).reshape(-1)
        if any(values.size != self.original_length for values in (self.raman_shift, self.prior, self.mask, self.limits)):
            raise ValueError("D3.5状态数组长度与original_length不一致。")
        if not np.isfinite(self.prior).all() or not np.isfinite(self.limits).all():
            raise ValueError("D3.5状态包含NaN或无穷值。")
        if not self.mask.any() or np.any(self.limits[self.mask] <= 0.0):
            raise ValueError("D3.5状态中的特征峰残差上限不合理。")

    @classmethod
    def from_state_dict(cls, feature_peak_residual_limiter_state: dict[str, Any]) -> "FeaturePeakResidualLimiter":
        return cls(feature_peak_residual_limiter_state=feature_peak_residual_limiter_state)

    def apply_to_normalized_spectra(self, normalized_spectra: np.ndarray) -> np.ndarray:
        """仅对真正超过峰区残差上限的尾部执行多样性保持软压缩。"""

        spectra = np.asarray(normalized_spectra, dtype=np.float64)
        if spectra.ndim != 2 or spectra.shape[1] != self.original_length:
            raise ValueError("normalized_spectra必须是与D3.5状态等长的二维数组。")
        if not np.isfinite(spectra).all():
            raise ValueError("normalized_spectra包含NaN或无穷值。")

        residuals = spectra - self.prior[np.newaxis, :]
        result = residuals.copy()

        limits = self.limits[self.mask]

        soft_fraction = float(
        self.configuration["soft_transition_fraction"]
        )
        transition = limits * soft_fraction

        active = residuals[:, self.mask]
        magnitude = np.abs(active)

    # D3.5.1：
    # 在训练集拟合得到的逐点上限 B 内完全保持原始残差，
    # 只有 |R| > B 的异常尾部才参与软压缩。
        excess = np.maximum(
            magnitude - limits[np.newaxis, :],
            0.0,
        )

    # 保留非零尾部斜率，避免不同大小的超限残差全部
    # 饱和到相同的逐点上限，从而丢失峰强多样性。
    #
    # 当 excess -> 0 时，局部斜率为 1；
    # 当 excess 很大时，渐近斜率为 soft_fraction。
        compressed_excess = (
            soft_fraction * excess
            + (1.0 - soft_fraction)
            * transition[np.newaxis, :]
            * np.tanh(
                excess
                / transition[np.newaxis, :]
            )
        )

        compressed_magnitude = (
            limits[np.newaxis, :]
            + compressed_excess
        )

        result[:, self.mask] = np.sign(active) * np.where(
            magnitude <= limits[np.newaxis, :],
            magnitude,
            compressed_magnitude,
        )

        limited = self.prior[np.newaxis, :] + result

        if not np.isfinite(limited).all():
            raise RuntimeError("D3.5限制后的生成光谱包含NaN或无穷值。")

        return limited.astype(
            np.float32,
            copy=False,
        )

    def summary(self) -> dict[str, float | int]:
        active_limits = self.limits[self.mask]
        return {
            "number_of_feature_peaks": int(self.selected_peak_indices.size),
            "feature_peak_point_count": int(self.mask.sum()),
            "limit_multiplier": float(self.configuration["limit_multiplier"]),
            "soft_limit_minimum": float(active_limits.min()),
            "soft_limit_median": float(np.median(active_limits)),
            "soft_limit_maximum": float(active_limits.max()),
        }
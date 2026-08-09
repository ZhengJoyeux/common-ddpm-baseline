"""D3.4.2：SERS 高频残差 + 峰级形态多样性约束。

D3.4.1 已经证明，仅约束高频残差可以降低不同生成谱之间的噪声模板复制，
但无法阻止完整谱主体、特征峰位置、峰高和峰宽发生模式坍缩。

D3.4.2 在保留 D3.4.1 高频残差约束的基础上，新增训练 batch 驱动的峰级
形态多样性约束：

1. 高频残差 pairwise 距离；
2. 高频残差 pairwise 相关性；
3. 高频残差逐波数方差上下界；
4. 峰位置离散度：同时比较左翼质心、整峰质心、右翼质心的 batch 标准差；
5. 整峰协同位移：限制“只动峰顶、峰翼不动”的不一致位移；
6. 峰高离散度；
7. 峰宽离散度。

峰位置不写死。每次前向传播只根据低噪声子 batch 的真实 target 光谱中位谱，
自动寻找稳定显著峰，再在相同窗口中提取预测谱和 target 谱的可微峰特征。
因此峰级多样性由真实训练 batch 自身提供参考，不人为规定“必须偏移 ±N cm^-1”。

计算域：
- 高频约束：prior_residual_scaled；
- 峰级形态约束：通过 D2.1 pointwise_mad_asinh 状态可微恢复到
  global_minmax normalized spectrum 域。

兼容性：
- 公开函数/类名保持不变；
- strategy 继续使用 low_noise_pairwise_residual_geometry；
- 旧 D3.4 / D3.4.1 checkpoint 的 diversity state 仍可构建；
- D3.4.2 不改变 U-Net 参数结构和采样接口。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.nn import functional as F


DIVERSITY_STATE_SCHEMA_VERSION = 1
DIVERSITY_STRATEGY = "low_noise_pairwise_residual_geometry"

LEGACY_METHOD_VERSION = "d3_4_pairwise_residual_diversity"
D3_4_1_METHOD_VERSION = "d3_4_1_high_frequency_residual_diversity"
D3_4_2_METHOD_VERSION = "d3_4_2_peak_morphology_diversity"


def _finite_float(
    value: Any,
    field_name: str,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name}必须是数值。"
        ) from error

    if not np.isfinite(parsed):
        raise ValueError(
            f"{field_name}必须为有限数值。"
        )

    return parsed


def _positive_float(
    value: Any,
    field_name: str,
) -> float:
    parsed = _finite_float(
        value,
        field_name,
    )

    if parsed <= 0.0:
        raise ValueError(
            f"{field_name}必须大于0。"
        )

    return parsed


def _nonnegative_float(
    value: Any,
    field_name: str,
) -> float:
    parsed = _finite_float(
        value,
        field_name,
    )

    if parsed < 0.0:
        raise ValueError(
            f"{field_name}不能小于0。"
        )

    return parsed


def _fraction(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = True,
) -> float:
    parsed = _finite_float(
        value,
        field_name,
    )

    lower_ok = (
        parsed >= 0.0
        if allow_zero
        else parsed > 0.0
    )

    if (
        not lower_ok
        or parsed > 1.0
    ):
        interval = (
            "[0,1]"
            if allow_zero
            else "(0,1]"
        )

        raise ValueError(
            f"{field_name}必须在"
            f"{interval}范围内。"
        )

    return parsed


def _percentile(
    value: Any,
    field_name: str,
) -> float:
    parsed = _finite_float(
        value,
        field_name,
    )

    if not 0.0 <= parsed <= 100.0:
        raise ValueError(
            f"{field_name}必须在"
            "[0,100]范围内。"
        )

    return parsed


def _positive_integer(
    value: Any,
    field_name: str,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name}必须是整数。"
        ) from error

    if parsed <= 0:
        raise ValueError(
            f"{field_name}必须大于0。"
        )

    return parsed


def _section(
    configuration: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = configuration.setdefault(
        name,
        {},
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"diversity_constraints.{name}"
            "必须是字典。"
        )

    return value


def normalize_diversity_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """校验并补全 D3.4.1 / D3.4.2 多样性配置。"""

    if configuration is None:
        configuration = {
            "enabled": False
        }

    if not isinstance(
        configuration,
        dict,
    ):
        raise TypeError(
            "diversity_constraints必须是字典。"
        )

    config = deepcopy(
        configuration
    )

    config["enabled"] = bool(
        config.get(
            "enabled",
            False,
        )
    )

    if not config["enabled"]:
        return config

    strategy = str(
        config.get(
            "strategy",
            DIVERSITY_STRATEGY,
        )
    ).strip().lower()

    if strategy != DIVERSITY_STRATEGY:
        raise ValueError(
            "diversity_constraints.strategy必须为"
            f"{DIVERSITY_STRATEGY}。"
        )

    config["strategy"] = strategy

    config["total_weight"] = (
        _positive_float(
            config.get(
                "total_weight",
                0.01,
            ),
            "diversity_constraints."
            "total_weight",
        )
    )

    config["epsilon"] = (
        _positive_float(
            config.get(
                "epsilon",
                1.0e-8,
            ),
            "diversity_constraints."
            "epsilon",
        )
    )

    gate = _section(
        config,
        "low_noise_gate",
    )

    gate[
        "minimum_alpha_cumprod"
    ] = _fraction(
        gate.get(
            "minimum_alpha_cumprod",
            0.50,
        ),
        "diversity_constraints."
        "low_noise_gate."
        "minimum_alpha_cumprod",
        allow_zero=False,
    )

    gate[
        "minimum_samples"
    ] = _positive_integer(
        gate.get(
            "minimum_samples",
            4,
        ),
        "diversity_constraints."
        "low_noise_gate."
        "minimum_samples",
    )

    high_frequency = _section(
        config,
        "high_frequency_filter",
    )

    high_frequency[
        "smoothing_sigma_points"
    ] = _positive_float(
        high_frequency.get(
            "smoothing_sigma_points",
            1.0,
        ),
        "diversity_constraints."
        "high_frequency_filter."
        "smoothing_sigma_points",
    )

    high_frequency[
        "kernel_truncate"
    ] = _positive_float(
        high_frequency.get(
            "kernel_truncate",
            3.0,
        ),
        "diversity_constraints."
        "high_frequency_filter."
        "kernel_truncate",
    )

    high_frequency[
        "distance_reference_quantile"
    ] = _percentile(
        high_frequency.get(
            "distance_reference_quantile",
            50.0,
        ),
        "diversity_constraints."
        "high_frequency_filter."
        "distance_reference_quantile",
    )

    high_frequency[
        "correlation_reference_quantile"
    ] = _percentile(
        high_frequency.get(
            "correlation_reference_quantile",
            90.0,
        ),
        "diversity_constraints."
        "high_frequency_filter."
        "correlation_reference_quantile",
    )

    distance = _section(
        config,
        "pairwise_distance",
    )

    distance["enabled"] = bool(
        distance.get(
            "enabled",
            True,
        )
    )

    distance["metric"] = str(
        distance.get(
            "metric",
            "whitened_l2",
        )
    ).strip().lower()

    if (
        distance["metric"]
        != "whitened_l2"
    ):
        raise ValueError(
            "pairwise_distance.metric"
            "当前只支持whitened_l2。"
        )

    distance[
        "minimum_distance_ratio"
    ] = _fraction(
        distance.get(
            "minimum_distance_ratio",
            0.50,
        ),
        "diversity_constraints."
        "pairwise_distance."
        "minimum_distance_ratio",
    )

    distance[
        "maximum_distance_ratio"
    ] = _positive_float(
        distance.get(
            "maximum_distance_ratio",
            2.00,
        ),
        "diversity_constraints."
        "pairwise_distance."
        "maximum_distance_ratio",
    )

    distance[
        "transition_fraction"
    ] = _positive_float(
        distance.get(
            "transition_fraction",
            0.20,
        ),
        "diversity_constraints."
        "pairwise_distance."
        "transition_fraction",
    )

    distance[
        "weight"
    ] = _nonnegative_float(
        distance.get(
            "weight",
            1.0,
        ),
        "diversity_constraints."
        "pairwise_distance.weight",
    )

    if (
        distance[
            "maximum_distance_ratio"
        ]
        <= distance[
            "minimum_distance_ratio"
        ]
    ):
        raise ValueError(
            "pairwise_distance."
            "maximum_distance_ratio"
            "必须大于"
            "minimum_distance_ratio。"
        )

    correlation = _section(
        config,
        "pairwise_correlation",
    )

    correlation["enabled"] = bool(
        correlation.get(
            "enabled",
            True,
        )
    )

    correlation[
        "maximum_excess_correlation"
    ] = _fraction(
        correlation.get(
            "maximum_excess_correlation",
            0.005,
        ),
        "diversity_constraints."
        "pairwise_correlation."
        "maximum_excess_correlation",
    )

    correlation[
        "transition_fraction"
    ] = _positive_float(
        correlation.get(
            "transition_fraction",
            0.20,
        ),
        "diversity_constraints."
        "pairwise_correlation."
        "transition_fraction",
    )

    correlation[
        "weight"
    ] = _nonnegative_float(
        correlation.get(
            "weight",
            0.5,
        ),
        "diversity_constraints."
        "pairwise_correlation.weight",
    )

    variance = _section(
        config,
        "pointwise_variance_floor",
    )

    variance["enabled"] = bool(
        variance.get(
            "enabled",
            True,
        )
    )

    variance[
        "active_std_quantile"
    ] = _percentile(
        variance.get(
            "active_std_quantile",
            20.0,
        ),
        "diversity_constraints."
        "pointwise_variance_floor."
        "active_std_quantile",
    )

    variance[
        "minimum_std_ratio"
    ] = _fraction(
        variance.get(
            "minimum_std_ratio",
            0.30,
        ),
        "diversity_constraints."
        "pointwise_variance_floor."
        "minimum_std_ratio",
    )

    variance[
        "maximum_std_ratio"
    ] = _positive_float(
        variance.get(
            "maximum_std_ratio",
            1.70,
        ),
        "diversity_constraints."
        "pointwise_variance_floor."
        "maximum_std_ratio",
    )

    variance[
        "reference_std_floor_fraction"
    ] = _fraction(
        variance.get(
            "reference_std_floor_fraction",
            0.25,
        ),
        "diversity_constraints."
        "pointwise_variance_floor."
        "reference_std_floor_fraction",
    )

    variance[
        "weight"
    ] = _nonnegative_float(
        variance.get(
            "weight",
            0.2,
        ),
        "diversity_constraints."
        "pointwise_variance_floor."
        "weight",
    )

    if (
        variance[
            "maximum_std_ratio"
        ]
        < 1.0
    ):
        raise ValueError(
            "pointwise_variance_floor."
            "maximum_std_ratio"
            "不能小于1。"
        )

    morphology = _section(
        config,
        "peak_morphology",
    )

    # 默认关闭，防止旧 D3.4 / D3.4.1
    # checkpoint 配置被静默升级为 D3.4.2。
    morphology["enabled"] = bool(
        morphology.get(
            "enabled",
            False,
        )
    )

    morphology[
        "weight"
    ] = _nonnegative_float(
        morphology.get(
            "weight",
            1.0,
        ),
        "diversity_constraints."
        "peak_morphology.weight",
    )

    morphology[
        "smoothing_sigma_cm1"
    ] = _positive_float(
        morphology.get(
            "smoothing_sigma_cm1",
            2.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "smoothing_sigma_cm1",
    )

    morphology[
        "local_maximum_radius_cm1"
    ] = _positive_float(
        morphology.get(
            "local_maximum_radius_cm1",
            4.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "local_maximum_radius_cm1",
    )

    morphology[
        "prominence_radius_cm1"
    ] = _positive_float(
        morphology.get(
            "prominence_radius_cm1",
            12.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "prominence_radius_cm1",
    )

    morphology[
        "minimum_peak_separation_cm1"
    ] = _positive_float(
        morphology.get(
            "minimum_peak_separation_cm1",
            10.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "minimum_peak_separation_cm1",
    )

    morphology[
        "analysis_half_width_cm1"
    ] = _positive_float(
        morphology.get(
            "analysis_half_width_cm1",
            20.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "analysis_half_width_cm1",
    )

    morphology[
        "edge_exclusion_cm1"
    ] = _nonnegative_float(
        morphology.get(
            "edge_exclusion_cm1",
            24.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "edge_exclusion_cm1",
    )

    morphology[
        "maximum_peaks"
    ] = _positive_integer(
        morphology.get(
            "maximum_peaks",
            10,
        ),
        "diversity_constraints."
        "peak_morphology."
        "maximum_peaks",
    )

    morphology[
        "minimum_relative_prominence"
    ] = _fraction(
        morphology.get(
            "minimum_relative_prominence",
            0.05,
        ),
        "diversity_constraints."
        "peak_morphology."
        "minimum_relative_prominence",
        allow_zero=False,
    )

    morphology[
        "minimum_absolute_prominence"
    ] = _positive_float(
        morphology.get(
            "minimum_absolute_prominence",
            1.0e-4,
        ),
        "diversity_constraints."
        "peak_morphology."
        "minimum_absolute_prominence",
    )

    morphology[
        "positive_softplus_temperature"
    ] = _positive_float(
        morphology.get(
            "positive_softplus_temperature",
            0.01,
        ),
        "diversity_constraints."
        "peak_morphology."
        "positive_softplus_temperature",
    )

    position = _section(
        morphology,
        "position_dispersion",
    )

    position[
        "minimum_std_ratio"
    ] = _fraction(
        position.get(
            "minimum_std_ratio",
            0.60,
        ),
        "diversity_constraints."
        "peak_morphology."
        "position_dispersion."
        "minimum_std_ratio",
    )

    position[
        "maximum_std_ratio"
    ] = _positive_float(
        position.get(
            "maximum_std_ratio",
            1.60,
        ),
        "diversity_constraints."
        "peak_morphology."
        "position_dispersion."
        "maximum_std_ratio",
    )

    position[
        "scale_floor_cm1"
    ] = _positive_float(
        position.get(
            "scale_floor_cm1",
            0.10,
        ),
        "diversity_constraints."
        "peak_morphology."
        "position_dispersion."
        "scale_floor_cm1",
    )

    position[
        "weight"
    ] = _nonnegative_float(
        position.get(
            "weight",
            1.0,
        ),
        "diversity_constraints."
        "peak_morphology."
        "position_dispersion.weight",
    )

    coherence = _section(
        morphology,
        "coherent_shift",
    )

    coherence[
        "maximum_incoherence_ratio"
    ] = _positive_float(
        coherence.get(
            "maximum_incoherence_ratio",
            1.50,
        ),
        "diversity_constraints."
        "peak_morphology."
        "coherent_shift."
        "maximum_incoherence_ratio",
    )

    coherence[
        "scale_floor_cm1"
    ] = _positive_float(
        coherence.get(
            "scale_floor_cm1",
            0.10,
        ),
        "diversity_constraints."
        "peak_morphology."
        "coherent_shift."
        "scale_floor_cm1",
    )

    coherence[
        "weight"
    ] = _nonnegative_float(
        coherence.get(
            "weight",
            0.75,
        ),
        "diversity_constraints."
        "peak_morphology."
        "coherent_shift.weight",
    )

    height = _section(
        morphology,
        "height_dispersion",
    )

    height[
        "minimum_std_ratio"
    ] = _fraction(
        height.get(
            "minimum_std_ratio",
            0.60,
        ),
        "diversity_constraints."
        "peak_morphology."
        "height_dispersion."
        "minimum_std_ratio",
    )

    height[
        "maximum_std_ratio"
    ] = _positive_float(
        height.get(
            "maximum_std_ratio",
            1.60,
        ),
        "diversity_constraints."
        "peak_morphology."
        "height_dispersion."
        "maximum_std_ratio",
    )

    height[
        "scale_floor_fraction"
    ] = _positive_float(
        height.get(
            "scale_floor_fraction",
            0.02,
        ),
        "diversity_constraints."
        "peak_morphology."
        "height_dispersion."
        "scale_floor_fraction",
    )

    height[
        "weight"
    ] = _nonnegative_float(
        height.get(
            "weight",
            0.75,
        ),
        "diversity_constraints."
        "peak_morphology."
        "height_dispersion.weight",
    )

    width = _section(
        morphology,
        "width_dispersion",
    )

    width[
        "minimum_std_ratio"
    ] = _fraction(
        width.get(
            "minimum_std_ratio",
            0.60,
        ),
        "diversity_constraints."
        "peak_morphology."
        "width_dispersion."
        "minimum_std_ratio",
    )

    width[
        "maximum_std_ratio"
    ] = _positive_float(
        width.get(
            "maximum_std_ratio",
            1.60,
        ),
        "diversity_constraints."
        "peak_morphology."
        "width_dispersion."
        "maximum_std_ratio",
    )

    width[
        "scale_floor_fraction"
    ] = _positive_float(
        width.get(
            "scale_floor_fraction",
            0.02,
        ),
        "diversity_constraints."
        "peak_morphology."
        "width_dispersion."
        "scale_floor_fraction",
    )

    width[
        "weight"
    ] = _nonnegative_float(
        width.get(
            "weight",
            0.75,
        ),
        "diversity_constraints."
        "peak_morphology."
        "width_dispersion.weight",
    )

    for name, section in (
        (
            "position_dispersion",
            position,
        ),
        (
            "height_dispersion",
            height,
        ),
        (
            "width_dispersion",
            width,
        ),
    ):
        if (
            section[
                "maximum_std_ratio"
            ]
            < 1.0
        ):
            raise ValueError(
                f"peak_morphology.{name}."
                "maximum_std_ratio不能小于1。"
            )

        if (
            section[
                "maximum_std_ratio"
            ]
            <= section[
                "minimum_std_ratio"
            ]
        ):
            raise ValueError(
                f"peak_morphology.{name}."
                "maximum_std_ratio必须大于"
                "minimum_std_ratio。"
            )

    high_frequency_weight_sum = (
        (
            distance["weight"]
            if distance["enabled"]
            else 0.0
        )
        + (
            correlation["weight"]
            if correlation["enabled"]
            else 0.0
        )
        + (
            variance["weight"]
            if variance["enabled"]
            else 0.0
        )
    )

    if (
        high_frequency_weight_sum
        <= 0.0
        and not morphology[
            "enabled"
        ]
    ):
        raise ValueError(
            "diversity_constraints"
            "至少需要一个启用且"
            "权重大于0的分量。"
        )

    if morphology["enabled"]:
        morphology_internal_weight_sum = (
            position["weight"]
            + coherence["weight"]
            + height["weight"]
            + width["weight"]
        )

        if (
            morphology["weight"]
            <= 0.0
            or morphology_internal_weight_sum
            <= 0.0
        ):
            raise ValueError(
                "启用peak_morphology时"
                "其权重必须大于0。"
            )

        config[
            "method_version"
        ] = (
            D3_4_2_METHOD_VERSION
        )

    else:
        config[
            "method_version"
        ] = (
            D3_4_1_METHOD_VERSION
        )

    return config


def _validate_training_spectra(
    spectra: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        spectra,
        dtype=np.float64,
    )

    if values.ndim == 3:
        if values.shape[1] != 1:
            raise ValueError(
                "训练scaled residual"
                "三维输入必须为[N,1,L]。"
            )

        values = values[
            :,
            0,
            :,
        ]

    if values.ndim != 2:
        raise ValueError(
            "训练scaled residual必须为"
            "二维[N,L]数组。"
        )

    if values.shape[0] < 3:
        raise ValueError(
            "拟合D3.4至少需要3条训练光谱。"
        )

    if values.shape[1] < 9:
        raise ValueError(
            "训练scaled residual长度"
            "至少需要9个点。"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "训练scaled residual包含"
            "NaN或无穷值。"
        )

    return values


def _high_frequency_numpy(
    spectra: np.ndarray,
    *,
    sigma_points: float,
    truncate: float,
) -> np.ndarray:
    smoothed = gaussian_filter1d(
        spectra,
        sigma=float(
            sigma_points
        ),
        axis=1,
        mode="reflect",
        truncate=float(
            truncate
        ),
    )

    return (
        spectra
        - smoothed
    )


def _pairwise_distance_numpy(
    values: np.ndarray,
) -> np.ndarray:
    distances: list[
        float
    ] = []

    for first in range(
        values.shape[0] - 1
    ):
        for second in range(
            first + 1,
            values.shape[0],
        ):
            difference = (
                values[first]
                - values[second]
            )

            distances.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                difference
                            )
                        )
                    )
                )
            )

    return np.asarray(
        distances,
        dtype=np.float64,
    )


def _pairwise_correlation_numpy(
    values: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    centered = (
        values
        - values.mean(
            axis=1,
            keepdims=True,
        )
    )

    rms = np.sqrt(
        np.mean(
            np.square(
                centered
            ),
            axis=1,
        )
    )

    rms = np.maximum(
        rms,
        epsilon,
    )

    correlations: list[
        float
    ] = []

    for first in range(
        values.shape[0] - 1
    ):
        for second in range(
            first + 1,
            values.shape[0],
        ):
            correlation = (
                np.mean(
                    centered[first]
                    * centered[second]
                )
                / (
                    rms[first]
                    * rms[second]
                )
            )

            correlations.append(
                float(
                    np.clip(
                        correlation,
                        -1.0,
                        1.0,
                    )
                )
            )

    return np.asarray(
        correlations,
        dtype=np.float64,
    )


def fit_sers_diversity_constraint_state(
    *,
    training_scaled_residuals: np.ndarray,
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """只用训练 scaled residual 拟合高频参考统计。

    D3.4.2 的峰级形态参考不在此处写死到 checkpoint；
    训练时使用低噪声 target batch 自身的真实峰分布作为参照。
    """

    config = (
        normalize_diversity_configuration(
            configuration
        )
    )

    if not bool(
        config.get(
            "enabled",
            False,
        )
    ):
        return {
            "schema_version": (
                DIVERSITY_STATE_SCHEMA_VERSION
            ),
            "enabled": False,
            "configuration": config,
        }

    spectra = (
        _validate_training_spectra(
            training_scaled_residuals
        )
    )

    epsilon = float(
        config[
            "epsilon"
        ]
    )

    high_frequency_config = (
        config[
            "high_frequency_filter"
        ]
    )

    high_frequency = (
        _high_frequency_numpy(
            spectra,
            sigma_points=float(
                high_frequency_config[
                    "smoothing_sigma_points"
                ]
            ),
            truncate=float(
                high_frequency_config[
                    "kernel_truncate"
                ]
            ),
        )
    )

    pointwise_std = np.std(
        high_frequency,
        axis=0,
        ddof=1,
    )

    positive_std = (
        pointwise_std[
            pointwise_std
            > epsilon
        ]
    )

    if positive_std.size == 0:
        raise ValueError(
            "训练集高频残差逐点标准差"
            "全部接近0。"
        )

    whitening_floor = max(
        float(
            np.percentile(
                positive_std,
                10.0,
            )
        ),
        epsilon,
    )

    whitening_std = np.maximum(
        pointwise_std,
        whitening_floor,
    )

    variance_config = (
        config[
            "pointwise_variance_floor"
        ]
    )

    active_threshold = float(
        np.percentile(
            pointwise_std,
            variance_config[
                "active_std_quantile"
            ],
        )
    )

    active_mask = (
        pointwise_std
        >= max(
            active_threshold,
            epsilon,
        )
    )

    pairwise_distance = (
        _pairwise_distance_numpy(
            high_frequency
        )
    )

    pairwise_correlation = (
        _pairwise_correlation_numpy(
            high_frequency,
            epsilon=epsilon,
        )
    )

    if (
        pairwise_distance.size
        == 0
    ):
        raise RuntimeError(
            "训练集不足以拟合"
            "高频残差pairwise距离。"
        )

    return {
        "schema_version": (
            DIVERSITY_STATE_SCHEMA_VERSION
        ),
        "enabled": True,
        "strategy": (
            DIVERSITY_STRATEGY
        ),
        "method_version": (
            config[
                "method_version"
            ]
        ),
        "configuration": config,
        "number_of_training_spectra": int(
            spectra.shape[0]
        ),
        "original_length": int(
            spectra.shape[1]
        ),
        "pointwise_std": (
            pointwise_std
            .astype(
                np.float32
            )
            .tolist()
        ),
        "whitening_std": (
            whitening_std
            .astype(
                np.float32
            )
            .tolist()
        ),
        "active_mask": (
            active_mask.tolist()
        ),
        "high_frequency_smoothing_sigma_points": float(
            high_frequency_config[
                "smoothing_sigma_points"
            ]
        ),
        "high_frequency_kernel_truncate": float(
            high_frequency_config[
                "kernel_truncate"
            ]
        ),
        "high_frequency_distance_reference": max(
            float(
                np.percentile(
                    pairwise_distance,
                    high_frequency_config[
                        "distance_reference_quantile"
                    ],
                )
            ),
            epsilon,
        ),
        "high_frequency_correlation_reference": float(
            np.percentile(
                pairwise_correlation,
                high_frequency_config[
                    "correlation_reference_quantile"
                ],
            )
        ),
        "high_frequency_training_pairwise_distance_median": float(
            np.median(
                pairwise_distance
            )
        ),
        "high_frequency_training_pairwise_correlation_median": float(
            np.median(
                pairwise_correlation
            )
        ),
    }


def _gaussian_kernel_torch(
    sigma_points: float,
    truncate: float,
) -> torch.Tensor:
    radius = max(
        1,
        int(
            float(
                truncate
            )
            * float(
                sigma_points
            )
            + 0.5
        ),
    )

    positions = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
    )

    kernel = torch.exp(
        -0.5
        * torch.square(
            positions
            / float(
                sigma_points
            )
        )
    )

    kernel = (
        kernel
        / kernel.sum()
    )

    return kernel.view(
        1,
        1,
        -1,
    )


def _trapezoid_weights_torch(
    axis: torch.Tensor,
) -> torch.Tensor:
    if (
        axis.ndim != 1
        or axis.numel() < 2
    ):
        raise ValueError(
            "局部拉曼轴至少需要2个点。"
        )

    spacing = (
        axis[1:]
        - axis[:-1]
    )

    weights = torch.empty_like(
        axis
    )

    weights[0] = (
        spacing[0]
        * 0.5
    )

    weights[-1] = (
        spacing[-1]
        * 0.5
    )

    if axis.numel() > 2:
        weights[1:-1] = (
            0.5
            * (
                spacing[:-1]
                + spacing[1:]
            )
        )

    return weights


class DifferentiableSersDiversityLoss(
    nn.Module
):
    """D3.4.2 高频残差 + 峰级形态多样性损失。"""

    def __init__(
        self,
        *,
        diversity_constraint_state: dict[
            str,
            Any,
        ],
        padded_length: int,
    ) -> None:
        super().__init__()

        state = (
            diversity_constraint_state
        )

        if not isinstance(
            state,
            dict,
        ):
            raise TypeError(
                "diversity_constraint_state"
                "必须是字典。"
            )

        if int(
            state.get(
                "schema_version",
                0,
            )
        ) != (
            DIVERSITY_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "diversity_constraint_state"
                "版本不受支持。"
            )

        if not bool(
            state.get(
                "enabled",
                False,
            )
        ):
            raise ValueError(
                "diversity_constraint_state"
                "没有启用。"
            )

        method_version = str(
            state.get(
                "method_version",
                LEGACY_METHOD_VERSION,
            )
        )

        if method_version not in {
            LEGACY_METHOD_VERSION,
            D3_4_1_METHOD_VERSION,
            D3_4_2_METHOD_VERSION,
        }:
            raise ValueError(
                "diversity_constraint_state"
                "方法版本不受支持。"
            )

        self.method_version = (
            method_version
        )

        self.configuration = (
            normalize_diversity_configuration(
                state[
                    "configuration"
                ]
            )
        )

        self.original_length = int(
            state[
                "original_length"
            ]
        )

        self.padded_length = int(
            padded_length
        )

        if (
            self.padded_length
            < self.original_length
        ):
            raise ValueError(
                "padded_length不能小于"
                "original_length。"
            )

        self.epsilon = float(
            self.configuration[
                "epsilon"
            ]
        )

        gate = (
            self.configuration[
                "low_noise_gate"
            ]
        )

        self.minimum_alpha_cumprod = float(
            gate[
                "minimum_alpha_cumprod"
            ]
        )

        self.minimum_samples = int(
            gate[
                "minimum_samples"
            ]
        )

        pointwise_std = torch.as_tensor(
            state[
                "pointwise_std"
            ],
            dtype=torch.float32,
        ).reshape(
            -1
        )

        whitening_std = torch.as_tensor(
            state[
                "whitening_std"
            ],
            dtype=torch.float32,
        ).reshape(
            -1
        )

        active_mask = torch.as_tensor(
            state[
                "active_mask"
            ],
            dtype=torch.bool,
        ).reshape(
            -1
        )

        if not (
            pointwise_std.numel()
            == whitening_std.numel()
            == active_mask.numel()
            == self.original_length
        ):
            raise ValueError(
                "D3.4状态长度与"
                "original_length不一致。"
            )

        self.register_buffer(
            "pointwise_std",
            pointwise_std.view(
                1,
                1,
                -1,
            ),
            persistent=False,
        )

        self.register_buffer(
            "whitening_std",
            whitening_std.view(
                1,
                1,
                -1,
            ),
            persistent=False,
        )

        self.register_buffer(
            "active_mask",
            active_mask.view(
                1,
                1,
                -1,
            ),
            persistent=False,
        )

        distance = (
            self.configuration[
                "pairwise_distance"
            ]
        )

        correlation = (
            self.configuration[
                "pairwise_correlation"
            ]
        )

        variance = (
            self.configuration[
                "pointwise_variance_floor"
            ]
        )

        self.distance_enabled = bool(
            distance[
                "enabled"
            ]
        )

        self.distance_minimum_ratio = float(
            distance[
                "minimum_distance_ratio"
            ]
        )

        self.distance_maximum_ratio = float(
            distance[
                "maximum_distance_ratio"
            ]
        )

        self.distance_transition_fraction = float(
            distance[
                "transition_fraction"
            ]
        )

        self.distance_weight = float(
            distance[
                "weight"
            ]
        )

        self.correlation_enabled = bool(
            correlation[
                "enabled"
            ]
        )

        self.maximum_excess_correlation = float(
            correlation[
                "maximum_excess_correlation"
            ]
        )

        self.correlation_transition_fraction = float(
            correlation[
                "transition_fraction"
            ]
        )

        self.correlation_weight = float(
            correlation[
                "weight"
            ]
        )

        self.variance_enabled = bool(
            variance[
                "enabled"
            ]
        )

        self.variance_minimum_ratio = float(
            variance[
                "minimum_std_ratio"
            ]
        )

        self.variance_maximum_ratio = float(
            variance[
                "maximum_std_ratio"
            ]
        )

        self.variance_reference_floor_fraction = float(
            variance[
                "reference_std_floor_fraction"
            ]
        )

        self.variance_weight = float(
            variance[
                "weight"
            ]
        )

        if (
            self.method_version
            == LEGACY_METHOD_VERSION
        ):
            high_frequency_kernel = torch.ones(
                1,
                1,
                1,
                dtype=torch.float32,
            )

        else:
            sigma = float(
                state[
                    "high_frequency_smoothing_sigma_points"
                ]
            )

            truncate = float(
                state[
                    "high_frequency_kernel_truncate"
                ]
            )

            high_frequency_kernel = (
                _gaussian_kernel_torch(
                    sigma,
                    truncate,
                )
            )

        self.register_buffer(
            "high_frequency_kernel",
            high_frequency_kernel,
            persistent=False,
        )

        self.high_frequency_padding = (
            high_frequency_kernel.shape[-1]
            // 2
        )

        morphology = (
            self.configuration.get(
                "peak_morphology",
                {},
            )
        )

        self.morphology_enabled = bool(
            morphology.get(
                "enabled",
                False,
            )
        )

        self.morphology_weight = float(
            morphology.get(
                "weight",
                0.0,
            )
        )

        self.morphology_configuration = (
            morphology
        )

        # 以下 buffer 由已经拟合好的
        # D3.2 / D2.1 状态注入。
        self.register_buffer(
            "prior",
            torch.zeros(
                1,
                1,
                self.original_length,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "prior_pointwise_scale",
            torch.ones(
                1,
                1,
                self.original_length,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "raman_shift",
            torch.arange(
                self.original_length,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.inverse_reference_ready = (
            False
        )

        self.target_abs_max = 1.0
        self.standardized_residual_scale = 1.0
        self.asinh_normalizer = 1.0
        self.numerical_safety_limit = 15.0

        self._load_morphology_configuration()

    def _load_morphology_configuration(
        self,
    ) -> None:
        if not self.morphology_enabled:
            self.morphology_internal_weight_sum = (
                1.0
            )

            return

        config = (
            self.morphology_configuration
        )

        position = (
            config[
                "position_dispersion"
            ]
        )

        coherence = (
            config[
                "coherent_shift"
            ]
        )

        height = (
            config[
                "height_dispersion"
            ]
        )

        width = (
            config[
                "width_dispersion"
            ]
        )

        self.morphology_smoothing_sigma_cm1 = float(
            config[
                "smoothing_sigma_cm1"
            ]
        )

        self.morphology_local_maximum_radius_cm1 = float(
            config[
                "local_maximum_radius_cm1"
            ]
        )

        self.morphology_prominence_radius_cm1 = float(
            config[
                "prominence_radius_cm1"
            ]
        )

        self.morphology_minimum_peak_separation_cm1 = float(
            config[
                "minimum_peak_separation_cm1"
            ]
        )

        self.morphology_analysis_half_width_cm1 = float(
            config[
                "analysis_half_width_cm1"
            ]
        )

        self.morphology_edge_exclusion_cm1 = float(
            config[
                "edge_exclusion_cm1"
            ]
        )

        self.morphology_maximum_peaks = int(
            config[
                "maximum_peaks"
            ]
        )

        self.morphology_minimum_relative_prominence = float(
            config[
                "minimum_relative_prominence"
            ]
        )

        self.morphology_minimum_absolute_prominence = float(
            config[
                "minimum_absolute_prominence"
            ]
        )

        self.morphology_softplus_temperature = float(
            config[
                "positive_softplus_temperature"
            ]
        )

        self.position_minimum_std_ratio = float(
            position[
                "minimum_std_ratio"
            ]
        )

        self.position_maximum_std_ratio = float(
            position[
                "maximum_std_ratio"
            ]
        )

        self.position_scale_floor_cm1 = float(
            position[
                "scale_floor_cm1"
            ]
        )

        self.position_weight = float(
            position[
                "weight"
            ]
        )

        self.coherence_maximum_incoherence_ratio = float(
            coherence[
                "maximum_incoherence_ratio"
            ]
        )

        self.coherence_scale_floor_cm1 = float(
            coherence[
                "scale_floor_cm1"
            ]
        )

        self.coherence_weight = float(
            coherence[
                "weight"
            ]
        )

        self.height_minimum_std_ratio = float(
            height[
                "minimum_std_ratio"
            ]
        )

        self.height_maximum_std_ratio = float(
            height[
                "maximum_std_ratio"
            ]
        )

        self.height_scale_floor_fraction = float(
            height[
                "scale_floor_fraction"
            ]
        )

        self.height_weight = float(
            height[
                "weight"
            ]
        )

        self.width_minimum_std_ratio = float(
            width[
                "minimum_std_ratio"
            ]
        )

        self.width_maximum_std_ratio = float(
            width[
                "maximum_std_ratio"
            ]
        )

        self.width_scale_floor_fraction = float(
            width[
                "scale_floor_fraction"
            ]
        )

        self.width_weight = float(
            width[
                "weight"
            ]
        )

        self.morphology_internal_weight_sum = max(
            self.position_weight
            + self.coherence_weight
            + self.height_weight
            + self.width_weight,
            self.epsilon,
        )

    def configure_inverse_reference(
        self,
        *,
        prior_normalized_intensity: (
            torch.Tensor
            | np.ndarray
        ),
        pointwise_scale: (
            torch.Tensor
            | np.ndarray
        ),
        raman_shift: (
            torch.Tensor
            | np.ndarray
        ),
        target_abs_max: float,
        standardized_residual_scale: float,
        asinh_normalizer: float,
        numerical_safety_limit: float,
    ) -> None:
        """注入D2.1反变换和统一拉曼轴。

        这里不拟合新的统计量，只复用已经由训练集拟合好的
        D2.1 / D3.2 状态。
        """

        prior = torch.as_tensor(
            prior_normalized_intensity,
            dtype=torch.float32,
        ).reshape(
            -1
        )

        scale = torch.as_tensor(
            pointwise_scale,
            dtype=torch.float32,
        ).reshape(
            -1
        )

        axis = torch.as_tensor(
            raman_shift,
            dtype=torch.float32,
        ).reshape(
            -1
        )

        if not (
            prior.numel()
            == scale.numel()
            == axis.numel()
            == self.original_length
        ):
            raise ValueError(
                "D2.1反变换状态与"
                "D3.4.2长度不一致。"
            )

        if torch.any(
            scale <= 0.0
        ):
            raise ValueError(
                "D2.1 pointwise_scale"
                "必须全部大于0。"
            )

        if torch.any(
            axis[1:]
            <= axis[:-1]
        ):
            raise ValueError(
                "D3.4.2拉曼轴必须严格递增。"
            )

        self.prior.copy_(
            prior.view(
                1,
                1,
                -1,
            )
        )

        self.prior_pointwise_scale.copy_(
            scale.view(
                1,
                1,
                -1,
            )
        )

        self.raman_shift.copy_(
            axis
        )

        self.target_abs_max = float(
            target_abs_max
        )

        self.standardized_residual_scale = float(
            standardized_residual_scale
        )

        self.asinh_normalizer = float(
            asinh_normalizer
        )

        self.numerical_safety_limit = float(
            numerical_safety_limit
        )

        if min(
            self.target_abs_max,
            self.standardized_residual_scale,
            self.asinh_normalizer,
            self.numerical_safety_limit,
        ) <= 0.0:
            raise ValueError(
                "D3.4.2反变换标量参数"
                "必须大于0。"
            )

        self.inverse_reference_ready = (
            True
        )

    def _resolve_reference_prior(
        self,
        scaled: torch.Tensor,
        reference_prior: torch.Tensor | None,
    ) -> torch.Tensor:
        if reference_prior is None:
            return self.prior
        if reference_prior.ndim != 3 or reference_prior.shape[1] != 1:
            raise ValueError("D3.4 reference_prior必须为[B,1,L]。")
        if reference_prior.shape[0] != scaled.shape[0]:
            raise ValueError("D3.4 reference_prior批量大小不一致。")
        if reference_prior.shape[-1] != self.padded_length:
            raise ValueError("D3.4 reference_prior长度不一致。")
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
        if not self.inverse_reference_ready:
            raise RuntimeError(
                "D3.4.2 peak_morphology已启用，"
                "但尚未注入D2.1反变换状态。"
            )

        cropped = (
            scaled[
                ...,
                : self.original_length,
            ].float()
        )

        raw_argument = (
            cropped
            / self.target_abs_max
            * self.asinh_normalizer
        )

        safe_argument = (
            self.numerical_safety_limit
            * torch.tanh(
                raw_argument
                / self.numerical_safety_limit
            )
        )

        standardized = (
            self.standardized_residual_scale
            * torch.sinh(
                safe_argument
            )
        )

        return (
            self._resolve_reference_prior(cropped, reference_prior)
            + self.prior_pointwise_scale
            * standardized
        )

    def _select_low_noise(
        self,
        values: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        alpha = (
            alphas_cumprod.gather(
                0,
                timesteps,
            )
        )

        mask = (
            alpha
            >= self.minimum_alpha_cumprod
        )

        return (
            values[
                mask,
                ...,
                : self.original_length,
            ],
            alpha[
                mask
            ],
            mask,
        )

    def _high_frequency(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.method_version
            == LEGACY_METHOD_VERSION
        ):
            return values

        kernel = (
            self.high_frequency_kernel.to(
                values
            )
        )

        padded = F.pad(
            values,
            (
                self.high_frequency_padding,
                self.high_frequency_padding,
            ),
            mode="reflect",
        )

        return (
            values
            - F.conv1d(
                padded,
                kernel,
            )
        )

    @staticmethod
    def _pair_indices(
        number: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        indices = (
            torch.triu_indices(
                number,
                number,
                offset=1,
                device=device,
            )
        )

        return (
            indices[0],
            indices[1],
        )

    def _pairwise_distance(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        rows = values.squeeze(
            1
        )

        first, second = (
            self._pair_indices(
                rows.shape[0],
                rows.device,
            )
        )

        difference = (
            rows[first]
            - rows[second]
        )

        if (
            self.method_version
            == LEGACY_METHOD_VERSION
        ):
            difference = (
                difference
                / self.whitening_std
                .reshape(
                    -1
                )
                .clamp_min(
                    self.epsilon
                )
            )

        return torch.sqrt(
            torch.mean(
                torch.square(
                    difference
                ),
                dim=1,
            )
            + self.epsilon
        )

    def _pairwise_correlation(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        rows = values.squeeze(
            1
        )

        rows = (
            rows
            - rows.mean(
                dim=1,
                keepdim=True,
            )
        )

        rms = torch.sqrt(
            torch.mean(
                torch.square(
                    rows
                ),
                dim=1,
            )
            + self.epsilon
        )

        first, second = (
            self._pair_indices(
                rows.shape[0],
                rows.device,
            )
        )

        correlation = (
            torch.mean(
                rows[first]
                * rows[second],
                dim=1,
            )
            / (
                rms[first]
                * rms[second]
            ).clamp_min(
                self.epsilon
            )
        )

        return correlation.clamp(
            -1.0,
            1.0,
        )

    def _distance_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not self.distance_enabled:
            return predicted.new_zeros(
                ()
            )

        predicted_distance = (
            self._pairwise_distance(
                predicted
            )
        )

        target_distance = (
            self._pairwise_distance(
                target
            ).detach()
        )

        lower = (
            target_distance
            * self.distance_minimum_ratio
        )

        upper = (
            target_distance
            * self.distance_maximum_ratio
        )

        transition = (
            target_distance.abs()
            * self.distance_transition_fraction
        ).clamp_min(
            self.epsilon
        )

        return (
            torch.square(
                F.relu(
                    lower
                    - predicted_distance
                )
                / transition
            )
            + torch.square(
                F.relu(
                    predicted_distance
                    - upper
                )
                / transition
            )
        ).mean()

    def _correlation_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not self.correlation_enabled:
            return predicted.new_zeros(
                ()
            )

        predicted_correlation = (
            self._pairwise_correlation(
                predicted
            )
        )

        target_correlation = (
            self._pairwise_correlation(
                target
            ).detach()
        )

        maximum_allowed = (
            target_correlation
            + self.maximum_excess_correlation
        ).clamp(
            max=1.0
        )

        transition = max(
            self.correlation_transition_fraction,
            self.epsilon,
        )

        return torch.square(
            F.relu(
                predicted_correlation
                - maximum_allowed
            )
            / transition
        ).mean()

    def _variance_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.variance_enabled
            or predicted.shape[0]
            < 2
        ):
            return predicted.new_zeros(
                ()
            )

        predicted_std = torch.std(
            predicted,
            dim=0,
            correction=0,
        )

        target_std = torch.std(
            target,
            dim=0,
            correction=0,
        ).detach()

        training_reference = (
            self.pointwise_std
        )

        reference = torch.maximum(
            target_std,
            (
                training_reference
                * self.variance_reference_floor_fraction
            ),
        )

        lower = (
            reference
            * self.variance_minimum_ratio
        )

        upper = (
            torch.maximum(
                reference,
                training_reference,
            )
            * self.variance_maximum_ratio
        )

        scale = torch.maximum(
            reference,
            training_reference,
        ).clamp_min(
            self.epsilon
        )

        violation = (
            torch.square(
                F.relu(
                    lower
                    - predicted_std
                )
                / scale
            )
            + torch.square(
                F.relu(
                    predicted_std
                    - upper
                )
                / scale
            )
        )

        selected = (
            violation[
                self.active_mask
            ]
        )

        if selected.numel() == 0:
            return predicted.new_zeros(
                ()
            )

        return selected.mean()

    def _morphology_points(
        self,
        value_cm1: float,
    ) -> int:
        spacing = torch.median(
            self.raman_shift[1:]
            - self.raman_shift[:-1]
        )

        spacing_value = float(
            spacing.detach().cpu()
        )

        if (
            not np.isfinite(
                spacing_value
            )
            or spacing_value <= 0.0
        ):
            raise RuntimeError(
                "D3.4.2无法计算"
                "有效拉曼轴间隔。"
            )

        return max(
            1,
            int(
                round(
                    float(
                        value_cm1
                    )
                    / spacing_value
                )
            ),
        )

    def _morphology_smooth(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        spacing = float(
            torch.median(
                self.raman_shift[1:]
                - self.raman_shift[:-1]
            )
            .detach()
            .cpu()
        )

        sigma_points = max(
            self.morphology_smoothing_sigma_cm1
            / spacing,
            1.0e-6,
        )

        kernel = (
            _gaussian_kernel_torch(
                sigma_points,
                3.0,
            ).to(
                values
            )
        )

        radius = (
            kernel.shape[-1]
            // 2
        )

        return F.conv1d(
            F.pad(
                values,
                (
                    radius,
                    radius,
                ),
                mode="reflect",
            ),
            kernel,
        )

    def _detect_batch_peak_indices(
        self,
        target: torch.Tensor,
    ) -> list[int]:
        """从真实target子batch中位谱自动找峰。

        此过程只决定窗口位置，不参与梯度。
        """

        with torch.no_grad():
            smoothed_rows = (
                self._morphology_smooth(
                    target
                )
            )

            median_spectrum = (
                torch.median(
                    smoothed_rows,
                    dim=0,
                    keepdim=True,
                ).values
            )

            maximum_radius = (
                self._morphology_points(
                    self.morphology_local_maximum_radius_cm1
                )
            )

            prominence_radius = max(
                self._morphology_points(
                    self.morphology_prominence_radius_cm1
                ),
                maximum_radius + 1,
            )

            separation = (
                self._morphology_points(
                    self.morphology_minimum_peak_separation_cm1
                )
            )

            half_width = (
                self._morphology_points(
                    self.morphology_analysis_half_width_cm1
                )
            )

            edge = max(
                self._morphology_points(
                    self.morphology_edge_exclusion_cm1
                ),
                half_width + 1,
            )

            local_maximum = (
                F.max_pool1d(
                    median_spectrum,
                    kernel_size=(
                        2
                        * maximum_radius
                        + 1
                    ),
                    stride=1,
                    padding=maximum_radius,
                )
            )

            local_minimum = (
                -F.max_pool1d(
                    -median_spectrum,
                    kernel_size=(
                        2
                        * prominence_radius
                        + 1
                    ),
                    stride=1,
                    padding=prominence_radius,
                )
            )

            prominence = (
                median_spectrum
                - local_minimum
            ).squeeze()

            candidates = (
                median_spectrum
                >= local_maximum
                - self.epsilon
            ).squeeze()

            candidates[
                :edge
            ] = False

            candidates[
                -edge:
            ] = False

            robust_range = (
                torch.quantile(
                    median_spectrum.reshape(
                        -1
                    ),
                    0.99,
                )
                - torch.quantile(
                    median_spectrum.reshape(
                        -1
                    ),
                    0.10,
                )
            )

            threshold = max(
                float(
                    robust_range
                    .detach()
                    .cpu()
                    * self.morphology_minimum_relative_prominence
                ),
                self.morphology_minimum_absolute_prominence,
            )

            candidate_indices = (
                torch.nonzero(
                    candidates
                    & (
                        prominence
                        >= threshold
                    ),
                    as_tuple=False,
                ).flatten()
            )

            if (
                candidate_indices.numel()
                == 0
            ):
                return []

            candidate_prominence = (
                prominence[
                    candidate_indices
                ]
            )

            ranking = torch.argsort(
                candidate_prominence,
                descending=True,
            )

            selected: list[
                int
            ] = []

            for order_index in ranking.tolist():
                index = int(
                    candidate_indices[
                        order_index
                    ].item()
                )

                if all(
                    abs(
                        index
                        - previous
                    )
                    >= separation
                    for previous in selected
                ):
                    selected.append(
                        index
                    )

                if (
                    len(
                        selected
                    )
                    >= self.morphology_maximum_peaks
                ):
                    break

            selected.sort()

            return selected

    def _extract_peak_features(
        self,
        spectra: torch.Tensor,
        peak_indices: list[int],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        half_width = (
            self._morphology_points(
                self.morphology_analysis_half_width_cm1
            )
        )

        position_features: list[
            torch.Tensor
        ] = []

        heights: list[
            torch.Tensor
        ] = []

        widths: list[
            torch.Tensor
        ] = []

        temperature = torch.as_tensor(
            self.morphology_softplus_temperature,
            dtype=spectra.dtype,
            device=spectra.device,
        )

        for center in peak_indices:
            start = (
                center
                - half_width
            )

            stop = (
                center
                + half_width
                + 1
            )

            window = spectra[
                ...,
                start:stop,
            ]

            axis = (
                self.raman_shift[
                    start:stop
                ].to(
                    window
                )
            )

            fraction = (
                (
                    axis
                    - axis[0]
                )
                / (
                    axis[-1]
                    - axis[0]
                ).clamp_min(
                    self.epsilon
                )
            ).view(
                1,
                1,
                -1,
            )

            baseline = (
                window[..., :1]
                * (
                    1.0
                    - fraction
                )
                + window[..., -1:]
                * fraction
            )

            corrected = (
                window
                - baseline
            )

            positive = (
                F.softplus(
                    corrected
                    / temperature
                )
                * temperature
            )

            quadrature = (
                _trapezoid_weights_torch(
                    axis
                ).view(
                    1,
                    1,
                    -1,
                )
            )

            mass = (
                positive
                * quadrature
            )

            total_mass = (
                mass.sum(
                    dim=-1
                )
                .clamp_min(
                    self.epsilon
                )
            )

            centroid = (
                (
                    mass
                    * axis.view(
                        1,
                        1,
                        -1,
                    )
                )
                .sum(
                    dim=-1
                )
                / total_mass
            )

            centered_axis = (
                axis.view(
                    1,
                    1,
                    -1,
                )
                - centroid.unsqueeze(
                    -1
                )
            )

            variance = (
                (
                    mass
                    * centered_axis.square()
                )
                .sum(
                    dim=-1
                )
                / total_mass
            )

            width = (
                2.354820045
                * torch.sqrt(
                    variance.clamp_min(
                        self.epsilon
                    )
                )
            )

            height = (
                positive.amax(
                    dim=-1
                )
            )

            local_center = (
                center
                - start
            )

            left_mass = (
                mass[
                    ...,
                    : local_center + 1,
                ]
            )

            left_axis = (
                axis[
                    : local_center + 1
                ]
            )

            right_mass = (
                mass[
                    ...,
                    local_center:,
                ]
            )

            right_axis = (
                axis[
                    local_center:
                ]
            )

            left_centroid = (
                (
                    left_mass
                    * left_axis.view(
                        1,
                        1,
                        -1,
                    )
                )
                .sum(
                    dim=-1
                )
                / left_mass.sum(
                    dim=-1
                ).clamp_min(
                    self.epsilon
                )
            )

            right_centroid = (
                (
                    right_mass
                    * right_axis.view(
                        1,
                        1,
                        -1,
                    )
                )
                .sum(
                    dim=-1
                )
                / right_mass.sum(
                    dim=-1
                ).clamp_min(
                    self.epsilon
                )
            )

            position_features.append(
                torch.stack(
                    (
                        left_centroid.squeeze(
                            1
                        ),
                        centroid.squeeze(
                            1
                        ),
                        right_centroid.squeeze(
                            1
                        ),
                    ),
                    dim=-1,
                )
            )

            heights.append(
                height.squeeze(
                    1
                )
            )

            widths.append(
                width.squeeze(
                    1
                )
            )

        return {
            "position": torch.stack(
                position_features,
                dim=1,
            ),
            "height": torch.stack(
                heights,
                dim=1,
            ),
            "width": torch.stack(
                widths,
                dim=1,
            ),
        }

    def _dispersion_band_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        minimum_ratio: float,
        maximum_ratio: float,
        scale_floor: (
            torch.Tensor
            | float
        ),
    ) -> torch.Tensor:
        predicted_std = torch.std(
            predicted,
            dim=0,
            correction=0,
        )

        target_std = torch.std(
            target,
            dim=0,
            correction=0,
        ).detach()

        # 注意：
        # 下界永远是 target_std × ratio。
        # scale_floor只用于分母数值稳定，
        # 不会人为制造非零峰位/峰高/峰宽变化。
        lower = (
            target_std
            * minimum_ratio
        )

        upper = (
            target_std
            * maximum_ratio
        )

        if isinstance(
            scale_floor,
            torch.Tensor,
        ):
            floor = (
                scale_floor.to(
                    predicted
                )
            )

        else:
            floor = (
                predicted.new_tensor(
                    float(
                        scale_floor
                    )
                )
            )

        scale = torch.maximum(
            target_std,
            floor,
        ).clamp_min(
            self.epsilon
        )

        return (
            torch.square(
                F.relu(
                    lower
                    - predicted_std
                )
                / scale
            )
            + torch.square(
                F.relu(
                    predicted_std
                    - upper
                )
                / scale
            )
        ).mean()

    def _peak_morphology_loss(
        self,
        predicted_scaled: torch.Tensor,
        target_scaled: torch.Tensor,
        reference_prior: torch.Tensor | None,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        zero = (
            predicted_scaled
            .new_zeros(
                ()
            )
        )

        if not self.morphology_enabled:
            return {
                "peak_morphology_loss": zero,
                "peak_position_dispersion_loss": zero,
                "peak_shift_coherence_loss": zero,
                "peak_height_dispersion_loss": zero,
                "peak_width_dispersion_loss": zero,
                "mean_morphology_peaks": zero,
            }

        predicted = (
            self.inverse_scaled_residual(
                predicted_scaled,
                reference_prior=reference_prior,
            )
        )

        target = (
            self.inverse_scaled_residual(
                target_scaled,
                reference_prior=reference_prior,
            ).detach()
        )

        peak_indices = (
            self._detect_batch_peak_indices(
                target
            )
        )

        if not peak_indices:
            return {
                "peak_morphology_loss": zero,
                "peak_position_dispersion_loss": zero,
                "peak_shift_coherence_loss": zero,
                "peak_height_dispersion_loss": zero,
                "peak_width_dispersion_loss": zero,
                "mean_morphology_peaks": zero,
            }

        predicted_features = (
            self._extract_peak_features(
                predicted,
                peak_indices,
            )
        )

        target_features = (
            self._extract_peak_features(
                target,
                peak_indices,
            )
        )

        position_loss = (
            self._dispersion_band_loss(
                predicted_features[
                    "position"
                ],
                target_features[
                    "position"
                ],
                minimum_ratio=(
                    self.position_minimum_std_ratio
                ),
                maximum_ratio=(
                    self.position_maximum_std_ratio
                ),
                scale_floor=(
                    self.position_scale_floor_cm1
                ),
            )
        )

        predicted_position = (
            predicted_features[
                "position"
            ]
        )

        target_position = (
            target_features[
                "position"
            ]
        )

        # 对每个峰、每个样本看它相对于batch平均形态的位置变化。
        # 真正整峰移动时左翼/整峰/右翼三个量应一起移动。
        predicted_displacement = (
            predicted_position
            - predicted_position.mean(
                dim=0,
                keepdim=True,
            )
        )

        target_displacement = (
            target_position
            - target_position.mean(
                dim=0,
                keepdim=True,
            )
        )

        predicted_incoherence = torch.std(
            predicted_displacement,
            dim=-1,
            correction=0,
        )

        target_incoherence = torch.std(
            target_displacement,
            dim=-1,
            correction=0,
        ).detach()

        coherence_upper = (
            target_incoherence
            * self.coherence_maximum_incoherence_ratio
        )

        coherence_scale = torch.maximum(
            target_incoherence,
            predicted_incoherence.new_tensor(
                self.coherence_scale_floor_cm1
            ),
        ).clamp_min(
            self.epsilon
        )

        coherence_loss = torch.square(
            F.relu(
                predicted_incoherence
                - coherence_upper
            )
            / coherence_scale
        ).mean()

        target_height_mean = (
            target_features[
                "height"
            ]
            .abs()
            .mean(
                dim=0
            )
            .detach()
        )

        height_floor = (
            target_height_mean
            * self.height_scale_floor_fraction
        ).clamp_min(
            self.epsilon
        )

        height_loss = (
            self._dispersion_band_loss(
                predicted_features[
                    "height"
                ],
                target_features[
                    "height"
                ],
                minimum_ratio=(
                    self.height_minimum_std_ratio
                ),
                maximum_ratio=(
                    self.height_maximum_std_ratio
                ),
                scale_floor=(
                    height_floor
                ),
            )
        )

        target_width_mean = (
            target_features[
                "width"
            ]
            .abs()
            .mean(
                dim=0
            )
            .detach()
        )

        width_floor = (
            target_width_mean
            * self.width_scale_floor_fraction
        ).clamp_min(
            self.epsilon
        )

        width_loss = (
            self._dispersion_band_loss(
                predicted_features[
                    "width"
                ],
                target_features[
                    "width"
                ],
                minimum_ratio=(
                    self.width_minimum_std_ratio
                ),
                maximum_ratio=(
                    self.width_maximum_std_ratio
                ),
                scale_floor=(
                    width_floor
                ),
            )
        )

        morphology_loss = (
            self.position_weight
            * position_loss
            + self.coherence_weight
            * coherence_loss
            + self.height_weight
            * height_loss
            + self.width_weight
            * width_loss
        ) / (
            self.morphology_internal_weight_sum
        )

        return {
            "peak_morphology_loss": (
                morphology_loss
            ),
            "peak_position_dispersion_loss": (
                position_loss
            ),
            "peak_shift_coherence_loss": (
                coherence_loss
            ),
            "peak_height_dispersion_loss": (
                height_loss
            ),
            "peak_width_dispersion_loss": (
                width_loss
            ),
            "mean_morphology_peaks": (
                predicted.new_tensor(
                    float(
                        len(
                            peak_indices
                        )
                    )
                )
            ),
        }

    def _zero_result(
        self,
        reference: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        zero = reference.new_zeros(
            ()
        )

        return {
            "diversity_raw_loss": zero,
            "diversity_timestep_weighted_loss": zero,
            "pairwise_distance_loss": zero,
            "pairwise_correlation_loss": zero,
            "pointwise_variance_floor_loss": zero,
            "peak_morphology_loss": zero,
            "peak_position_dispersion_loss": zero,
            "peak_shift_coherence_loss": zero,
            "peak_height_dispersion_loss": zero,
            "peak_width_dispersion_loss": zero,
            "mean_morphology_peaks": zero,
            "diversity_active_samples": zero,
            "mean_diversity_timestep_weight": zero,
        }

    def forward(
        self,
        *,
        predicted_scaled_residual: torch.Tensor,
        target_scaled_residual: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        reference_prior: torch.Tensor | None = None,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        if (
            predicted_scaled_residual.shape
            != target_scaled_residual.shape
        ):
            raise ValueError(
                "D3.4预测scaled residual和"
                "target形状必须一致。"
            )

        if (
            predicted_scaled_residual.ndim
            != 3
            or predicted_scaled_residual.shape[1]
            != 1
        ):
            raise ValueError(
                "D3.4 scaled residual"
                "必须为[B,1,L]。"
            )

        if (
            timesteps.ndim
            != 1
            or timesteps.shape[0]
            != predicted_scaled_residual.shape[0]
        ):
            raise ValueError(
                "D3.4 timesteps批量"
                "形状不正确。"
            )

        predicted, selected_alpha, low_noise_mask = (
            self._select_low_noise(
                predicted_scaled_residual,
                timesteps,
                alphas_cumprod,
            )
        )

        if (
            predicted.shape[0]
            < self.minimum_samples
        ):
            return self._zero_result(
                predicted_scaled_residual
            )

        target, _, _ = (
            self._select_low_noise(
                target_scaled_residual,
                timesteps,
                alphas_cumprod,
            )
        )

        predicted_high_frequency = (
            self._high_frequency(
                predicted
            )
        )

        target_high_frequency = (
            self._high_frequency(
                target
            )
        )

        distance_loss = (
            self._distance_loss(
                predicted_high_frequency,
                target_high_frequency,
            )
        )

        correlation_loss = (
            self._correlation_loss(
                predicted_high_frequency,
                target_high_frequency,
            )
        )

        variance_loss = (
            self._variance_loss(
                predicted_high_frequency,
                target_high_frequency,
            )
        )

        morphology = (
            self._peak_morphology_loss(
                predicted,
                target,
                (
                    None
                    if reference_prior is None
                    else reference_prior[low_noise_mask]
                ),
            )
        )

        weighted_terms: list[
            torch.Tensor
        ] = []

        weight_sum = 0.0

        if (
            self.distance_enabled
            and self.distance_weight > 0.0
        ):
            weighted_terms.append(
                self.distance_weight
                * distance_loss
            )

            weight_sum += (
                self.distance_weight
            )

        if (
            self.correlation_enabled
            and self.correlation_weight > 0.0
        ):
            weighted_terms.append(
                self.correlation_weight
                * correlation_loss
            )

            weight_sum += (
                self.correlation_weight
            )

        if (
            self.variance_enabled
            and self.variance_weight > 0.0
        ):
            weighted_terms.append(
                self.variance_weight
                * variance_loss
            )

            weight_sum += (
                self.variance_weight
            )

        if (
            self.morphology_enabled
            and self.morphology_weight > 0.0
        ):
            weighted_terms.append(
                self.morphology_weight
                * morphology[
                    "peak_morphology_loss"
                ]
            )

            weight_sum += (
                self.morphology_weight
            )

        if (
            not weighted_terms
            or weight_sum <= 0.0
        ):
            raw_loss = (
                predicted.new_zeros(
                    ()
                )
            )

        else:
            raw_loss = (
                sum(
                    weighted_terms
                )
                / weight_sum
            )

        timestep_weight = torch.sqrt(
            selected_alpha.clamp_min(
                0.0
            )
        ).mean()

        weighted_loss = (
            raw_loss
            * timestep_weight
        )

        return {
            "diversity_raw_loss": (
                raw_loss
            ),
            "diversity_timestep_weighted_loss": (
                weighted_loss
            ),
            "pairwise_distance_loss": (
                distance_loss
            ),
            "pairwise_correlation_loss": (
                correlation_loss
            ),
            "pointwise_variance_floor_loss": (
                variance_loss
            ),
            **morphology,
            "diversity_active_samples": (
                predicted.new_tensor(
                    float(
                        predicted.shape[0]
                    )
                )
            ),
            "mean_diversity_timestep_weight": (
                timestep_weight
            ),
        }
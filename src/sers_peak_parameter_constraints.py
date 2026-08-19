"""D3.4：完整重建 SERS 光谱域的峰位/峰宽物理软边界约束。

适用于 D2.6 broad/local residual diffusion：

    full spectrum
        =
    outer PCA prior
        +
    true broad residual
        +
    inverse(predicted scaled local residual)

所有峰均从 training split 自动检测，不允许硬编码农药特征峰。

训练集首先统计各自动峰的：
1. 可微定义对应的峰中心；
2. Gaussian-equivalent effective FWHM；

然后根据 training-only quantile + MAD 建立允许范围。

预测峰参数位于允许范围内部时损失严格为0，
只有超出训练合理分布范围才产生惩罚。

本模块不改变：
- U-Net结构；
- DDPM/DDIM采样；
- EMA；
- D2.6 PCA/broad/local分解；
- 生成后D3.3校准。
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


PEAK_PARAMETER_STATE_SCHEMA_VERSION = 1
PEAK_PARAMETER_METHOD_VERSION = (
    "d3_4_train_full_spectrum_peak_parameter_bounds_v1"
)


_DEFAULT_CONFIGURATION: dict[str, Any] = {
    "enabled": False,

    # training-only自动峰检测
    "reference_baseline_sigma_cm1": 30.0,
    "reference_peak_smoothing_sigma_cm1": 2.0,
    "minimum_relative_prominence": 0.06,
    "minimum_peak_distance_cm1": 12.0,
    "maximum_peak_count": 10,
    "minimum_peak_count": 3,
    "edge_exclusion_cm1": 30.0,

    # 每个峰用于可微参数估计的最大窗口。
    # 若邻峰过近，会自动进一步缩小。
    "maximum_measurement_half_window_cm1": 18.0,
    "minimum_measurement_half_window_cm1": 4.0,
    "neighbor_window_fraction": 0.45,

    # 过滤训练集中非常弱、参数不可靠的自动峰。
    "minimum_median_peak_height_fraction": 0.05,

    # soft positive weight：
    # weight = temperature * softplus(signal / temperature)
    "softplus_temperature_fraction": 0.05,
    "minimum_softplus_temperature": 1.0e-4,

    # training-only软边界
    "lower_quantile": 10.0,
    "upper_quantile": 90.0,
    "position_margin_mad": 0.50,
    "width_margin_mad": 0.50,

    # 防止16条小样本得到过窄边界
    "minimum_position_half_range_cm1": 1.0,
    "minimum_width_half_range_cm1": 1.0,

    # loss归一化尺度
    "position_scale_cm1": 1.0,
    "width_scale_cm1": 2.0,

    # 两类峰参数损失相对权重
    "position_loss_weight": 1.0,
    "width_loss_weight": 0.5,

    "smooth_l1_beta": 0.5,

    # 只强调低至中噪声阶段。
    # 100 timestep时，0.50意味着t<=49参与该约束。
    "maximum_active_timestep_fraction": 0.50,
    "timestep_weight_power": 1.0,
    "minimum_timestep_weight": 0.0,

    "total_weight": 1.0,

    # D3.4第一轮最多占DDPM loss的2%。
    "maximum_total_ratio_to_ddpm": 0.02,

    # 防止DDPM异常输出导致sinh溢出
    "numerical_safety_limit": 12.0,

    "epsilon": 1.0e-8,
}


_FORBIDDEN_FIXED_PEAK_KEYS = {
    "peak_centers",
    "peak_centers_cm1",
    "peak_positions",
    "peak_windows",
    "fixed_peak_positions",
    "reference_peak_positions",
}


def _finite_float(
    value: Any,
    name: str,
) -> float:
    parsed = float(value)

    if not np.isfinite(parsed):
        raise ValueError(
            f"{name}必须是有限数值。"
        )

    return parsed


def _positive_float(
    value: Any,
    name: str,
) -> float:
    parsed = _finite_float(
        value,
        name,
    )

    if parsed <= 0.0:
        raise ValueError(
            f"{name}必须大于0。"
        )

    return parsed


def _nonnegative_float(
    value: Any,
    name: str,
) -> float:
    parsed = _finite_float(
        value,
        name,
    )

    if parsed < 0.0:
        raise ValueError(
            f"{name}不能小于0。"
        )

    return parsed


def normalize_peak_parameter_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """补齐并严格检查D3.4配置。"""

    raw = (
        configuration
        or {"enabled": False}
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "peak_parameter_constraints必须是字典。"
        )

    forbidden = sorted(
        _FORBIDDEN_FIXED_PEAK_KEYS.intersection(
            raw
        )
    )

    if forbidden:
        raise ValueError(
            "D3.4禁止人工固定峰位；"
            "所有峰必须仅从training split自动拟合："
            f"{forbidden}。"
        )

    config = deepcopy(
        _DEFAULT_CONFIGURATION
    )

    config.update(
        raw
    )

    config["enabled"] = bool(
        config["enabled"]
    )

    positive_names = (
        "reference_baseline_sigma_cm1",
        "reference_peak_smoothing_sigma_cm1",
        "minimum_peak_distance_cm1",
        "maximum_measurement_half_window_cm1",
        "minimum_measurement_half_window_cm1",
        "neighbor_window_fraction",
        "minimum_median_peak_height_fraction",
        "softplus_temperature_fraction",
        "minimum_softplus_temperature",
        "minimum_position_half_range_cm1",
        "minimum_width_half_range_cm1",
        "position_scale_cm1",
        "width_scale_cm1",
        "position_loss_weight",
        "width_loss_weight",
        "smooth_l1_beta",
        "timestep_weight_power",
        "total_weight",
        "numerical_safety_limit",
        "epsilon",
    )

    for name in positive_names:
        config[name] = _positive_float(
            config[name],
            f"peak_parameter_constraints.{name}",
        )

    config["edge_exclusion_cm1"] = (
        _nonnegative_float(
            config["edge_exclusion_cm1"],
            "peak_parameter_constraints."
            "edge_exclusion_cm1",
        )
    )

    config["position_margin_mad"] = (
        _nonnegative_float(
            config["position_margin_mad"],
            "peak_parameter_constraints."
            "position_margin_mad",
        )
    )

    config["width_margin_mad"] = (
        _nonnegative_float(
            config["width_margin_mad"],
            "peak_parameter_constraints."
            "width_margin_mad",
        )
    )

    config["minimum_timestep_weight"] = (
        _nonnegative_float(
            config["minimum_timestep_weight"],
            "peak_parameter_constraints."
            "minimum_timestep_weight",
        )
    )

    if (
        config["minimum_timestep_weight"]
        > 1.0
    ):
        raise ValueError(
            "minimum_timestep_weight不能大于1。"
        )

    prominence = _finite_float(
        config[
            "minimum_relative_prominence"
        ],
        "peak_parameter_constraints."
        "minimum_relative_prominence",
    )

    if not 0.0 < prominence < 1.0:
        raise ValueError(
            "minimum_relative_prominence"
            "必须位于(0,1)。"
        )

    config[
        "minimum_relative_prominence"
    ] = prominence

    lower_quantile = _finite_float(
        config["lower_quantile"],
        "peak_parameter_constraints."
        "lower_quantile",
    )

    upper_quantile = _finite_float(
        config["upper_quantile"],
        "peak_parameter_constraints."
        "upper_quantile",
    )

    if not (
        0.0
        <= lower_quantile
        < upper_quantile
        <= 100.0
    ):
        raise ValueError(
            "必须满足0 <= lower_quantile "
            "< upper_quantile <= 100。"
        )

    config["lower_quantile"] = (
        lower_quantile
    )

    config["upper_quantile"] = (
        upper_quantile
    )

    active_fraction = _finite_float(
        config[
            "maximum_active_timestep_fraction"
        ],
        "peak_parameter_constraints."
        "maximum_active_timestep_fraction",
    )

    if not (
        0.0
        < active_fraction
        <= 1.0
    ):
        raise ValueError(
            "maximum_active_timestep_fraction"
            "必须位于(0,1]。"
        )

    config[
        "maximum_active_timestep_fraction"
    ] = active_fraction

    maximum_ratio = _finite_float(
        config[
            "maximum_total_ratio_to_ddpm"
        ],
        "peak_parameter_constraints."
        "maximum_total_ratio_to_ddpm",
    )

    if not (
        0.0
        < maximum_ratio
        <= 1.0
    ):
        raise ValueError(
            "maximum_total_ratio_to_ddpm"
            "必须位于(0,1]。"
        )

    config[
        "maximum_total_ratio_to_ddpm"
    ] = maximum_ratio

    maximum_peak_count = int(
        config[
            "maximum_peak_count"
        ]
    )

    minimum_peak_count = int(
        config[
            "minimum_peak_count"
        ]
    )

    if minimum_peak_count <= 0:
        raise ValueError(
            "minimum_peak_count必须大于0。"
        )

    if (
        maximum_peak_count
        < minimum_peak_count
    ):
        raise ValueError(
            "maximum_peak_count不能小于"
            "minimum_peak_count。"
        )

    config[
        "maximum_peak_count"
    ] = maximum_peak_count

    config[
        "minimum_peak_count"
    ] = minimum_peak_count

    if (
        config[
            "minimum_measurement_half_window_cm1"
        ]
        >
        config[
            "maximum_measurement_half_window_cm1"
        ]
    ):
        raise ValueError(
            "minimum_measurement_half_window_cm1"
            "不能大于"
            "maximum_measurement_half_window_cm1。"
        )

    return config


def _validate_axis(
    raman_shift: np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    axis = np.asarray(
        raman_shift,
        dtype=np.float64,
    ).reshape(-1)

    if axis.size < 5:
        raise ValueError(
            "D3.4 Raman shift轴至少需要5个点。"
        )

    if (
        expected_length is not None
        and axis.size
        != expected_length
    ):
        raise ValueError(
            "D3.4 Raman轴长度与光谱长度不一致。"
        )

    if not np.isfinite(
        axis
    ).all():
        raise ValueError(
            "D3.4 Raman轴包含NaN或无穷值。"
        )

    if not np.all(
        np.diff(axis) > 0.0
    ):
        raise ValueError(
            "D3.4 Raman轴必须严格递增。"
        )

    return axis


def _validate_spectra(
    spectra: np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    values = np.asarray(
        spectra,
        dtype=np.float64,
    )

    if (
        values.ndim != 2
        or values.shape[0] == 0
    ):
        raise ValueError(
            "D3.4训练光谱必须为非空二维数组[N,L]。"
        )

    if (
        expected_length is not None
        and values.shape[1]
        != expected_length
    ):
        raise ValueError(
            "D3.4训练光谱长度与Raman轴不一致。"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "D3.4训练光谱包含NaN或无穷值。"
        )

    return values


def _adaptive_half_windows(
    centers: np.ndarray,
    *,
    minimum_half_width: float,
    maximum_half_width: float,
    neighbor_fraction: float,
) -> np.ndarray:
    centers = np.asarray(
        centers,
        dtype=np.float64,
    ).reshape(-1)

    result = np.full(
        centers.size,
        maximum_half_width,
        dtype=np.float64,
    )

    for index in range(
        centers.size
    ):
        candidate = (
            maximum_half_width
        )

        if index > 0:
            candidate = min(
                candidate,
                neighbor_fraction
                * (
                    centers[index]
                    - centers[index - 1]
                ),
            )

        if (
            index
            < centers.size - 1
        ):
            candidate = min(
                candidate,
                neighbor_fraction
                * (
                    centers[index + 1]
                    - centers[index]
                ),
            )

        result[index] = max(
            minimum_half_width,
            candidate,
        )

    return result


def _softplus_numpy(
    values: np.ndarray,
    temperature: float,
) -> np.ndarray:
    temperature = float(
        temperature
    )

    return (
        temperature
        * np.logaddexp(
            0.0,
            values
            / temperature,
        )
    )


def _measure_numpy_parameters(
    *,
    spectra: np.ndarray,
    axis: np.ndarray,
    centers: np.ndarray,
    half_windows: np.ndarray,
    temperatures: np.ndarray,
    baseline_sigma_points: float,
    epsilon: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    baseline = gaussian_filter1d(
        spectra,
        sigma=baseline_sigma_points,
        axis=1,
        mode="nearest",
        truncate=4.0,
    )

    corrected = (
        spectra
        - baseline
    )

    positions: list[
        np.ndarray
    ] = []

    widths: list[
        np.ndarray
    ] = []

    gaussian_fwhm_factor = (
        2.3548200450309493
    )

    for (
        center,
        half_width,
        temperature,
    ) in zip(
        centers,
        half_windows,
        temperatures,
        strict=True,
    ):
        mask = (
            np.abs(
                axis
                - float(center)
            )
            <= float(
                half_width
            )
        )

        if (
            np.count_nonzero(mask)
            < 3
        ):
            raise ValueError(
                "D3.4峰测量窗口点数不足。"
            )

        local_axis = (
            axis[mask]
        )

        local_signal = (
            corrected[
                :,
                mask,
            ]
        )

        weight = (
            _softplus_numpy(
                local_signal,
                float(
                    temperature
                ),
            )
        )

        denominator = (
            np.sum(
                weight,
                axis=1,
            )
            + epsilon
        )

        position = (
            np.sum(
                weight
                * local_axis[
                    np.newaxis,
                    :
                ],
                axis=1,
            )
            / denominator
        )

        variance = (
            np.sum(
                weight
                * np.square(
                    local_axis[
                        np.newaxis,
                        :
                    ]
                    - position[
                        :,
                        np.newaxis,
                    ]
                ),
                axis=1,
            )
            / denominator
        )

        width = (
            gaussian_fwhm_factor
            * np.sqrt(
                np.maximum(
                    variance,
                    epsilon,
                )
            )
        )

        positions.append(
            position
        )

        widths.append(
            width
        )

    return (
        np.stack(
            positions,
            axis=1,
        ),
        np.stack(
            widths,
            axis=1,
        ),
    )


def _fit_bounds(
    values: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_mad: float,
    minimum_half_range: float,
    lower_floor: float | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    median = np.median(
        values,
        axis=0,
    )

    mad = (
        1.4826
        * np.median(
            np.abs(
                values
                - median[
                    np.newaxis,
                    :
                ]
            ),
            axis=0,
        )
    )

    lower = np.percentile(
        values,
        lower_quantile,
        axis=0,
    )

    upper = np.percentile(
        values,
        upper_quantile,
        axis=0,
    )

    lower = (
        lower
        - margin_mad
        * mad
    )

    upper = (
        upper
        + margin_mad
        * mad
    )

    lower = np.minimum(
        lower,
        median
        - minimum_half_range,
    )

    upper = np.maximum(
        upper,
        median
        + minimum_half_range,
    )

    if lower_floor is not None:
        lower = np.maximum(
            lower,
            lower_floor,
        )

    return (
        median,
        mad,
        lower,
        upper,
    )


def fit_peak_parameter_constraint_state(
    *,
    training_normalized_spectra: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """严格仅使用training split拟合D3.4物理边界。"""

    config = (
        normalize_peak_parameter_configuration(
            configuration
        )
    )

    if not config[
        "enabled"
    ]:
        return {
            "schema_version": (
                PEAK_PARAMETER_STATE_SCHEMA_VERSION
            ),
            "method_version": (
                PEAK_PARAMETER_METHOD_VERSION
            ),
            "enabled": False,
            "contains_fixed_peak_positions": False,
            "configuration": config,
        }

    spectra = _validate_spectra(
        training_normalized_spectra
    )

    axis = _validate_axis(
        raman_shift,
        expected_length=(
            spectra.shape[1]
        ),
    )

    spacing = float(
        np.median(
            np.diff(axis)
        )
    )

    baseline_sigma_points = (
        float(
            config[
                "reference_baseline_sigma_cm1"
            ]
        )
        / spacing
    )

    reference = np.median(
        spectra,
        axis=0,
    )

    reference_baseline = (
        gaussian_filter1d(
            reference,
            sigma=(
                baseline_sigma_points
            ),
            mode="nearest",
            truncate=4.0,
        )
    )

    peak_signal = (
        reference
        - reference_baseline
    )

    peak_signal = (
        gaussian_filter1d(
            peak_signal,
            sigma=(
                float(
                    config[
                        "reference_peak_smoothing_sigma_cm1"
                    ]
                )
                / spacing
            ),
            mode="nearest",
            truncate=4.0,
        )
    )

    maximum_positive_signal = float(
        np.max(
            peak_signal
        )
    )

    if (
        maximum_positive_signal
        <= float(
            config["epsilon"]
        )
    ):
        raise ValueError(
            "D3.4无法从training split检测正特征峰。"
        )

    detected, properties = (
        find_peaks(
            peak_signal,
            prominence=(
                float(
                    config[
                        "minimum_relative_prominence"
                    ]
                )
                * maximum_positive_signal
            ),
            distance=max(
                1,
                int(
                    round(
                        float(
                            config[
                                "minimum_peak_distance_cm1"
                            ]
                        )
                        / spacing
                    )
                ),
            ),
        )
    )

    edge = float(
        config[
            "edge_exclusion_cm1"
        ]
    )

    inside = (
        (
            axis[detected]
            >= axis[0] + edge
        )
        & (
            axis[detected]
            <= axis[-1] - edge
        )
    )

    detected = (
        detected[inside]
    )

    prominences = np.asarray(
        properties.get(
            "prominences",
            np.zeros(
                inside.size
            ),
        ),
        dtype=np.float64,
    )[inside]

    if (
        detected.size
        < int(
            config[
                "minimum_peak_count"
            ]
        )
    ):
        raise ValueError(
            "D3.4自动检测峰数量不足。"
        )

    maximum_peak_count = int(
        config[
            "maximum_peak_count"
        ]
    )

    if (
        detected.size
        > maximum_peak_count
    ):
        keep = np.argsort(
            prominences
        )[
            -maximum_peak_count:
        ]

        detected = (
            detected[keep]
        )

        prominences = (
            prominences[keep]
        )

    order = np.argsort(
        detected
    )

    detected = (
        detected[order]
    )

    prominences = (
        prominences[order]
    )

    centers = (
        axis[detected]
    )

    half_windows = (
        _adaptive_half_windows(
            centers,
            minimum_half_width=float(
                config[
                    "minimum_measurement_half_window_cm1"
                ]
            ),
            maximum_half_width=float(
                config[
                    "maximum_measurement_half_window_cm1"
                ]
            ),
            neighbor_fraction=float(
                config[
                    "neighbor_window_fraction"
                ]
            ),
        )
    )

    training_baseline = (
        gaussian_filter1d(
            spectra,
            sigma=(
                baseline_sigma_points
            ),
            axis=1,
            mode="nearest",
            truncate=4.0,
        )
    )

    training_corrected = (
        spectra
        - training_baseline
    )

    median_heights: list[
        float
    ] = []

    for (
        center,
        half_width,
    ) in zip(
        centers,
        half_windows,
        strict=True,
    ):
        mask = (
            np.abs(
                axis
                - float(center)
            )
            <= float(
                half_width
            )
        )

        heights = np.max(
            training_corrected[
                :,
                mask,
            ],
            axis=1,
        )

        median_heights.append(
            float(
                np.median(
                    heights
                )
            )
        )

    median_heights_array = (
        np.asarray(
            median_heights,
            dtype=np.float64,
        )
    )

    maximum_median_height = float(
        np.max(
            median_heights_array
        )
    )

    minimum_height = (
        float(
            config[
                "minimum_median_peak_height_fraction"
            ]
        )
        * maximum_median_height
    )

    keep = (
        median_heights_array
        >= minimum_height
    )

    detected = (
        detected[keep]
    )

    prominences = (
        prominences[keep]
    )

    centers = (
        centers[keep]
    )

    median_heights_array = (
        median_heights_array[
            keep
        ]
    )

    if (
        centers.size
        < int(
            config[
                "minimum_peak_count"
            ]
        )
    ):
        raise ValueError(
            "D3.4弱峰过滤后稳定峰数量不足。"
        )

    # 过滤弱峰后重新计算邻峰自适应窗口。
    half_windows = (
        _adaptive_half_windows(
            centers,
            minimum_half_width=float(
                config[
                    "minimum_measurement_half_window_cm1"
                ]
            ),
            maximum_half_width=float(
                config[
                    "maximum_measurement_half_window_cm1"
                ]
            ),
            neighbor_fraction=float(
                config[
                    "neighbor_window_fraction"
                ]
            ),
        )
    )

    temperatures = np.maximum(
        float(
            config[
                "minimum_softplus_temperature"
            ]
        ),
        float(
            config[
                "softplus_temperature_fraction"
            ]
        )
        * median_heights_array,
    )

    (
        training_positions,
        training_widths,
    ) = (
        _measure_numpy_parameters(
            spectra=spectra,
            axis=axis,
            centers=centers,
            half_windows=half_windows,
            temperatures=temperatures,
            baseline_sigma_points=(
                baseline_sigma_points
            ),
            epsilon=float(
                config[
                    "epsilon"
                ]
            ),
        )
    )

    (
        position_median,
        position_mad,
        position_lower,
        position_upper,
    ) = _fit_bounds(
        training_positions,
        lower_quantile=float(
            config[
                "lower_quantile"
            ]
        ),
        upper_quantile=float(
            config[
                "upper_quantile"
            ]
        ),
        margin_mad=float(
            config[
                "position_margin_mad"
            ]
        ),
        minimum_half_range=float(
            config[
                "minimum_position_half_range_cm1"
            ]
        ),
        lower_floor=None,
    )

    (
        width_median,
        width_mad,
        width_lower,
        width_upper,
    ) = _fit_bounds(
        training_widths,
        lower_quantile=float(
            config[
                "lower_quantile"
            ]
        ),
        upper_quantile=float(
            config[
                "upper_quantile"
            ]
        ),
        margin_mad=float(
            config[
                "width_margin_mad"
            ]
        ),
        minimum_half_range=float(
            config[
                "minimum_width_half_range_cm1"
            ]
        ),
        lower_floor=(
            0.5
            * spacing
        ),
    )

    return {
        "schema_version": (
            PEAK_PARAMETER_STATE_SCHEMA_VERSION
        ),
        "method_version": (
            PEAK_PARAMETER_METHOD_VERSION
        ),
        "enabled": True,

        "contains_fixed_peak_positions": False,
        "fit_on": "train_only",

        "data_domain": (
            "full_global_minmax_normalized_spectrum"
        ),

        "prediction_domain": (
            "d2_6_scaled_local_residual"
        ),

        "configuration": config,

        "number_of_training_spectra": int(
            spectra.shape[0]
        ),

        "original_length": int(
            axis.size
        ),

        "raman_shift": (
            axis.tolist()
        ),

        "raman_spacing_cm1": (
            spacing
        ),

        "detected_peak_indices": (
            detected.astype(
                np.int64
            ).tolist()
        ),

        "detected_peak_centers_cm1": (
            centers.tolist()
        ),

        "detected_peak_prominences": (
            prominences.tolist()
        ),

        "measurement_half_windows_cm1": (
            half_windows.tolist()
        ),

        "softplus_temperatures": (
            temperatures.tolist()
        ),

        "training_median_peak_heights": (
            median_heights_array.tolist()
        ),

        "training_peak_positions_cm1": (
            training_positions.tolist()
        ),

        "training_effective_widths_cm1": (
            training_widths.tolist()
        ),

        "position_median_cm1": (
            position_median.tolist()
        ),

        "position_mad_cm1": (
            position_mad.tolist()
        ),

        "position_lower_bound_cm1": (
            position_lower.tolist()
        ),

        "position_upper_bound_cm1": (
            position_upper.tolist()
        ),

        "width_median_cm1": (
            width_median.tolist()
        ),

        "width_mad_cm1": (
            width_mad.tolist()
        ),

        "width_lower_bound_cm1": (
            width_lower.tolist()
        ),

        "width_upper_bound_cm1": (
            width_upper.tolist()
        ),
    }


class DifferentiablePeakParameterLoss(
    nn.Module
):
    """D3.4完整重建谱峰位/峰宽物理边界损失。"""

    def __init__(
        self,
        *,
        peak_parameter_constraint_state: dict[str, Any],
        broad_local_residual_state: dict[str, Any],
        padded_length: int,
    ) -> None:
        super().__init__()

        state = (
            peak_parameter_constraint_state
        )

        if (
            not isinstance(
                state,
                dict,
            )
            or not bool(
                state.get(
                    "enabled",
                    False,
                )
            )
        ):
            raise ValueError(
                "D3.4 peak parameter state没有启用。"
            )

        if int(
            state.get(
                "schema_version",
                0,
            )
        ) != (
            PEAK_PARAMETER_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "D3.4 state版本不受支持。"
            )

        if bool(
            state.get(
                "contains_fixed_peak_positions",
                True,
            )
        ):
            raise ValueError(
                "D3.4 state禁止包含人工固定峰位。"
            )

        self.configuration = (
            normalize_peak_parameter_configuration(
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
                "D3.4 padded_length不能小于original_length。"
            )

        axis = _validate_axis(
            np.asarray(
                state[
                    "raman_shift"
                ],
                dtype=np.float64,
            ),
            expected_length=(
                self.original_length
            ),
        )

        centers = np.asarray(
            state[
                "detected_peak_centers_cm1"
            ],
            dtype=np.float64,
        )

        half_windows = np.asarray(
            state[
                "measurement_half_windows_cm1"
            ],
            dtype=np.float64,
        )

        temperatures = np.asarray(
            state[
                "softplus_temperatures"
            ],
            dtype=np.float64,
        )

        position_lower = np.asarray(
            state[
                "position_lower_bound_cm1"
            ],
            dtype=np.float64,
        )

        position_upper = np.asarray(
            state[
                "position_upper_bound_cm1"
            ],
            dtype=np.float64,
        )

        width_lower = np.asarray(
            state[
                "width_lower_bound_cm1"
            ],
            dtype=np.float64,
        )

        width_upper = np.asarray(
            state[
                "width_upper_bound_cm1"
            ],
            dtype=np.float64,
        )

        peak_count = (
            centers.size
        )

        expected_arrays = (
            half_windows,
            temperatures,
            position_lower,
            position_upper,
            width_lower,
            width_upper,
        )

        if (
            peak_count
            < int(
                self.configuration[
                    "minimum_peak_count"
                ]
            )
            or any(
                values.size
                != peak_count
                for values
                in expected_arrays
            )
        ):
            raise ValueError(
                "D3.4 state中的峰参数数组长度不一致。"
            )

        local_state = (
            broad_local_residual_state.get(
                "local_normalization"
            )
        )

        if not isinstance(
            local_state,
            dict,
        ):
            raise ValueError(
                "D3.4缺少D2.6 local normalization state。"
            )

        if str(
            local_state.get(
                "method",
                "",
            )
        ).strip().lower() != (
            "robust_asinh"
        ):
            raise ValueError(
                "D3.4当前仅支持D2.6 robust_asinh local residual。"
            )

        self.local_target_abs_max = (
            _positive_float(
                local_state[
                    "target_abs_max"
                ],
                "D2.6 target_abs_max",
            )
        )

        self.local_residual_scale = (
            _positive_float(
                local_state[
                    "scale"
                ],
                "D2.6 local scale",
            )
        )

        self.local_asinh_normalizer = (
            _positive_float(
                local_state[
                    "asinh_normalizer"
                ],
                "D2.6 local asinh_normalizer",
            )
        )

        window_masks = []

        for (
            center,
            half_width,
        ) in zip(
            centers,
            half_windows,
            strict=True,
        ):
            mask = (
                np.abs(
                    axis
                    - float(
                        center
                    )
                )
                <= float(
                    half_width
                )
            ).astype(
                np.float32
            )

            if (
                np.count_nonzero(
                    mask
                )
                < 3
            ):
                raise ValueError(
                    "D3.4 state中存在过窄峰窗口。"
                )

            window_masks.append(
                mask
            )

        window_masks_array = (
            np.stack(
                window_masks,
                axis=0,
            )
        )

        self.register_buffer(
            "raman_shift",
            torch.from_numpy(
                axis.astype(
                    np.float32
                )
            ).view(
                1,
                1,
                -1,
            ),
            persistent=False,
        )

        self.register_buffer(
            "window_masks",
            torch.from_numpy(
                window_masks_array
            ).view(
                1,
                peak_count,
                self.original_length,
            ),
            persistent=False,
        )

        self.register_buffer(
            "temperatures",
            torch.from_numpy(
                temperatures.astype(
                    np.float32
                )
            ).view(
                1,
                peak_count,
                1,
            ),
            persistent=False,
        )

        self.register_buffer(
            "position_lower",
            torch.from_numpy(
                position_lower.astype(
                    np.float32
                )
            ).view(
                1,
                peak_count,
            ),
            persistent=False,
        )

        self.register_buffer(
            "position_upper",
            torch.from_numpy(
                position_upper.astype(
                    np.float32
                )
            ).view(
                1,
                peak_count,
            ),
            persistent=False,
        )

        self.register_buffer(
            "width_lower",
            torch.from_numpy(
                width_lower.astype(
                    np.float32
                )
            ).view(
                1,
                peak_count,
            ),
            persistent=False,
        )

        self.register_buffer(
            "width_upper",
            torch.from_numpy(
                width_upper.astype(
                    np.float32
                )
            ).view(
                1,
                peak_count,
            ),
            persistent=False,
        )

        spacing = float(
            state[
                "raman_spacing_cm1"
            ]
        )

        sigma_points = (
            float(
                self.configuration[
                    "reference_baseline_sigma_cm1"
                ]
            )
            / spacing
        )

        radius = max(
            1,
            int(
                np.ceil(
                    4.0
                    * sigma_points
                )
            ),
        )

        offsets = np.arange(
            -radius,
            radius + 1,
            dtype=np.float64,
        )

        kernel = np.exp(
            -0.5
            * np.square(
                offsets
                / sigma_points
            )
        )

        kernel = (
            kernel
            / np.sum(
                kernel
            )
        )

        self.baseline_padding = (
            radius
        )

        self.register_buffer(
            "baseline_kernel",
            torch.from_numpy(
                kernel.astype(
                    np.float32
                )
            ).view(
                1,
                1,
                -1,
            ),
            persistent=False,
        )

    def _inverse_local_transform(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        argument = (
            values
            / self.local_target_abs_max
            * self.local_asinh_normalizer
        )

        safety_limit = float(
            self.configuration[
                "numerical_safety_limit"
            ]
        )

        argument = torch.clamp(
            argument,
            min=-safety_limit,
            max=safety_limit,
        )

        return (
            self.local_residual_scale
            * torch.sinh(
                argument
            )
        )

    def _measure(
        self,
        full_spectrum: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        values = full_spectrum[
            ...,
            : self.original_length
        ]

        kernel = (
            self.baseline_kernel.to(
                values
            )
        )

        padded = F.pad(
            values,
            (
                self.baseline_padding,
                self.baseline_padding,
            ),
            mode="replicate",
        )

        baseline = F.conv1d(
            padded,
            kernel,
        )

        corrected = (
            values
            - baseline
        )

        signal = (
            corrected[
                :,
                0,
                :
            ][
                :,
                None,
                :
            ]
        )

        temperature = (
            self.temperatures.to(
                signal
            )
        )

        mask = (
            self.window_masks.to(
                signal
            )
        )

        weights = (
            temperature
            * F.softplus(
                signal
                / temperature
            )
            * mask
        )

        epsilon = float(
            self.configuration[
                "epsilon"
            ]
        )

        denominator = (
            weights.sum(
                dim=-1
            )
            .clamp_min(
                epsilon
            )
        )

        axis = (
            self.raman_shift.to(
                signal
            )
        )

        positions = (
            (
                weights
                * axis
            ).sum(
                dim=-1
            )
            / denominator
        )

        centered = (
            axis
            - positions[
                :,
                :,
                None,
            ]
        )

        variance = (
            (
                weights
                * centered.square()
            ).sum(
                dim=-1
            )
            / denominator
        )

        widths = (
            2.3548200450309493
            * torch.sqrt(
                variance.clamp_min(
                    epsilon
                )
            )
        )

        return (
            positions,
            widths,
        )

    def _boundary_loss(
        self,
        *,
        positions: torch.Tensor,
        widths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        position_lower = (
            self.position_lower.to(
                positions
            )
        )

        position_upper = (
            self.position_upper.to(
                positions
            )
        )

        width_lower = (
            self.width_lower.to(
                widths
            )
        )

        width_upper = (
            self.width_upper.to(
                widths
            )
        )

        position_violation = (
            F.relu(
                position_lower
                - positions
            )
            + F.relu(
                positions
                - position_upper
            )
        )

        width_violation = (
            F.relu(
                width_lower
                - widths
            )
            + F.relu(
                widths
                - width_upper
            )
        )

        normalized_position = (
            position_violation
            / float(
                self.configuration[
                    "position_scale_cm1"
                ]
            )
        )

        normalized_width = (
            width_violation
            / float(
                self.configuration[
                    "width_scale_cm1"
                ]
            )
        )

        position_element = (
            F.smooth_l1_loss(
                normalized_position,
                torch.zeros_like(
                    normalized_position
                ),
                reduction="none",
                beta=float(
                    self.configuration[
                        "smooth_l1_beta"
                    ]
                ),
            )
        )

        width_element = (
            F.smooth_l1_loss(
                normalized_width,
                torch.zeros_like(
                    normalized_width
                ),
                reduction="none",
                beta=float(
                    self.configuration[
                        "smooth_l1_beta"
                    ]
                ),
            )
        )

        return {
            "position_per_sample": (
                position_element.mean(
                    dim=1
                )
            ),
            "width_per_sample": (
                width_element.mean(
                    dim=1
                )
            ),
            "position_violation": (
                position_violation
            ),
            "width_violation": (
                width_violation
            ),
        }

    def forward(
        self,
        *,
        predicted_scaled_local_residual: torch.Tensor,
        target_scaled_local_residual: torch.Tensor,
        reconstruction_base: torch.Tensor | None,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            reconstruction_base
            is None
        ):
            raise ValueError(
                "D3.4缺少reconstruction_base："
                "outer PCA prior + true broad residual。"
            )

        if not (
            predicted_scaled_local_residual.shape
            == target_scaled_local_residual.shape
            == reconstruction_base.shape
        ):
            raise ValueError(
                "D3.4预测、目标和reconstruction_base"
                "形状必须一致。"
            )

        predicted_scaled = (
            predicted_scaled_local_residual[
                ...,
                : self.original_length
            ]
        )

        target_scaled = (
            target_scaled_local_residual[
                ...,
                : self.original_length
            ]
        )

        base = (
            reconstruction_base[
                ...,
                : self.original_length
            ]
        )

        predicted_full = (
            base
            + self._inverse_local_transform(
                predicted_scaled
            )
        )

        target_full = (
            base
            + self._inverse_local_transform(
                target_scaled
            )
        )

        (
            predicted_positions,
            predicted_widths,
        ) = self._measure(
            predicted_full
        )

        (
            target_positions,
            target_widths,
        ) = self._measure(
            target_full
        )

        predicted_boundary = (
            self._boundary_loss(
                positions=(
                    predicted_positions
                ),
                widths=(
                    predicted_widths
                ),
            )
        )

        target_boundary = (
            self._boundary_loss(
                positions=(
                    target_positions
                ),
                widths=(
                    target_widths
                ),
            )
        )

        position_per_sample = (
            predicted_boundary[
                "position_per_sample"
            ]
        )

        width_per_sample = (
            predicted_boundary[
                "width_per_sample"
            ]
        )

        raw_per_sample = (
            float(
                self.configuration[
                    "position_loss_weight"
                ]
            )
            * position_per_sample
            +
            float(
                self.configuration[
                    "width_loss_weight"
                ]
            )
            * width_per_sample
        )

        alpha_bar = (
            alphas_cumprod.gather(
                -1,
                timesteps,
            ).to(
                raw_per_sample
            )
        )

        timestep_weight = (
            torch.pow(
                alpha_bar.clamp(
                    0.0,
                    1.0,
                ),
                float(
                    self.configuration[
                        "timestep_weight_power"
                    ]
                ),
            )
        )

        minimum_weight = float(
            self.configuration[
                "minimum_timestep_weight"
            ]
        )

        if minimum_weight > 0.0:
            timestep_weight = (
                timestep_weight.clamp_min(
                    minimum_weight
                )
            )

        maximum_active_timestep = int(
            np.floor(
                (
                    int(
                        alphas_cumprod.numel()
                    )
                    - 1
                )
                * float(
                    self.configuration[
                        "maximum_active_timestep_fraction"
                    ]
                )
            )
        )

        active = (
            timesteps
            <= maximum_active_timestep
        ).to(
            raw_per_sample
        )

        timestep_weight = (
            timestep_weight
            * active
        )

        weighted_per_sample = (
            raw_per_sample
            * timestep_weight
        )

        position_violation = (
            predicted_boundary[
                "position_violation"
            ]
        )

        width_violation = (
            predicted_boundary[
                "width_violation"
            ]
        )

        target_position_violation = (
            target_boundary[
                "position_violation"
            ]
        )

        target_width_violation = (
            target_boundary[
                "width_violation"
            ]
        )

        return {
            "peak_parameter_raw_loss": (
                raw_per_sample.mean()
            ),

            "peak_parameter_timestep_weighted_loss": (
                weighted_per_sample.mean()
            ),

            "peak_parameter_position_loss": (
                position_per_sample.mean()
            ),

            "peak_parameter_width_loss": (
                width_per_sample.mean()
            ),

            "mean_peak_position_violation_cm1": (
                position_violation.mean()
            ),

            "mean_peak_width_violation_cm1": (
                width_violation.mean()
            ),

            "peak_position_violation_fraction": (
                (
                    position_violation
                    > 0.0
                )
                .to(
                    raw_per_sample
                )
                .mean()
            ),

            "peak_width_violation_fraction": (
                (
                    width_violation
                    > 0.0
                )
                .to(
                    raw_per_sample
                )
                .mean()
            ),

            "target_peak_position_violation_fraction": (
                (
                    target_position_violation
                    > 0.0
                )
                .to(
                    raw_per_sample
                )
                .mean()
            ),

            "target_peak_width_violation_fraction": (
                (
                    target_width_violation
                    > 0.0
                )
                .to(
                    raw_per_sample
                )
                .mean()
            ),

            "mean_peak_parameter_timestep_weight": (
                timestep_weight.mean()
            ),

            "mean_predicted_peak_position_cm1": (
                predicted_positions.mean()
            ),

            "mean_predicted_effective_width_cm1": (
                predicted_widths.mean()
            ),
        }

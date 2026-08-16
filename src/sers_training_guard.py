"""Training-set-driven guard for generated SERS spectra.

Experiment C:
- automatically detect stable peak regions from the current training set;
- fit permissive training-driven bounds for peak position, height and FWHM;
- fit a permissive lower bound for spectrum minima;
- never hard-code pesticide-specific Raman shifts;
- only accept/reject generated spectra; do not modify intensities.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _as_2d_finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name}必须是二维数组[N,L]，实际={array.shape}。")
    if array.shape[0] < 1 or array.shape[1] < 3:
        raise ValueError(f"{name}形状无效：{array.shape}。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}包含NaN或无穷值。")
    return array


def _as_axis(values: np.ndarray, expected_length: int | None = None) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 3:
        raise ValueError("Raman shift轴至少需要3个点。")
    if expected_length is not None and axis.size != expected_length:
        raise ValueError(
            "Raman shift轴长度与光谱长度不一致："
            f"{axis.size} != {expected_length}。"
        )
    if not np.isfinite(axis).all():
        raise ValueError("Raman shift轴包含NaN或无穷值。")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("Raman shift轴必须严格递增。")
    return axis


def _finite_positive(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name}必须是有限正数。")
    return parsed


def _unit_interval(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name}必须位于[0,1]。")
    return parsed


def _iqr(values: np.ndarray) -> float:
    return float(
        np.percentile(values, 75.0)
        - np.percentile(values, 25.0)
    )


def _expanded_bounds(
    values: np.ndarray,
    *,
    margin_iqr: float,
    lower_floor: float | None = None,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("无法从空指标拟合训练边界。")

    minimum = float(np.min(clean))
    maximum = float(np.max(clean))
    spread = _iqr(clean)

    fallback = max(
        float(np.std(clean, ddof=0)),
        abs(float(np.median(clean))) * 1.0e-3,
        1.0e-8,
    )
    spread = max(spread, fallback)

    lower = minimum - margin_iqr * spread
    upper = maximum + margin_iqr * spread

    if lower_floor is not None:
        lower = max(float(lower_floor), lower)

    return float(lower), float(upper)


def _crossing_position(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
) -> float:
    denominator = y1 - y0
    if abs(denominator) <= np.finfo(np.float64).eps:
        return 0.5 * (x0 + x1)
    fraction = (target - y0) / denominator
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(x0 + fraction * (x1 - x0))


def _measure_peak(
    *,
    axis: np.ndarray,
    peak_signal: np.ndarray,
    center_cm1: float,
    half_width_cm1: float,
) -> dict[str, float]:
    mask = (
        (axis >= center_cm1 - half_width_cm1)
        & (axis <= center_cm1 + half_width_cm1)
    )
    indices = np.flatnonzero(mask)

    if indices.size < 5:
        return {
            "position_cm1": np.nan,
            "height": np.nan,
            "fwhm_cm1": np.nan,
        }

    x = axis[indices]
    y = peak_signal[indices]
    local_index = int(np.argmax(y))
    height = float(y[local_index])
    position = float(x[local_index])

    if not np.isfinite(height) or height <= 0.0:
        return {
            "position_cm1": position,
            "height": height,
            "fwhm_cm1": np.nan,
        }

    half_height = 0.5 * height

    left = local_index
    while left > 0 and y[left] > half_height:
        left -= 1

    right = local_index
    while right < y.size - 1 and y[right] > half_height:
        right += 1

    if (
        left == 0 and y[left] > half_height
    ) or (
        right == y.size - 1 and y[right] > half_height
    ):
        width = np.nan
    else:
        left_cross = _crossing_position(
            float(x[left]),
            float(y[left]),
            float(x[left + 1]),
            float(y[left + 1]),
            half_height,
        )
        right_cross = _crossing_position(
            float(x[right - 1]),
            float(y[right - 1]),
            float(x[right]),
            float(y[right]),
            half_height,
        )
        width = max(0.0, right_cross - left_cross)

    return {
        "position_cm1": position,
        "height": height,
        "fwhm_cm1": float(width),
    }


def fit_sers_training_guard_state(
    *,
    training_spectra: np.ndarray,
    raman_shift: np.ndarray,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Fit guard bounds using only the actual training spectra."""

    if not isinstance(configuration, dict):
        raise TypeError("training_guard配置必须是字典。")

    config = deepcopy(configuration)
    spectra = _as_2d_finite(training_spectra, "training_spectra")
    axis = _as_axis(raman_shift, expected_length=spectra.shape[1])

    if spectra.shape[0] < 4:
        raise ValueError("training_guard至少需要4条训练光谱。")

    spacing = float(np.median(np.diff(axis)))
    reference_sigma_cm1 = _finite_positive(
        config.get("reference_smoothing_sigma_cm1", 30.0),
        "reference_smoothing_sigma_cm1",
    )
    minimum_relative_prominence = _unit_interval(
        config.get("minimum_relative_prominence", 0.06),
        "minimum_relative_prominence",
    )
    minimum_peak_distance_cm1 = _finite_positive(
        config.get("minimum_peak_distance_cm1", 12.0),
        "minimum_peak_distance_cm1",
    )
    metric_half_width_cm1 = _finite_positive(
        config.get("metric_half_width_cm1", 24.0),
        "metric_half_width_cm1",
    )
    position_margin_cm1 = _finite_positive(
        config.get("position_margin_cm1", 2.0),
        "position_margin_cm1",
    )
    bound_margin_iqr = _finite_positive(
        config.get("bound_margin_iqr", 1.5),
        "bound_margin_iqr",
    )
    maximum_peak_count = int(config.get("maximum_peak_count", 20))

    minimum_guard_peak_median_height_fraction = (
        _unit_interval(
            config.get(
                "minimum_guard_peak_median_height_fraction",
                0.05,
            ),
            "minimum_guard_peak_median_height_fraction",
        )
    )

    minimum_valid_training_count = int(
        config.get(
            "minimum_valid_training_count",
            max(4, spectra.shape[0] // 2),
        )
    )

    if maximum_peak_count <= 0:
        raise ValueError("maximum_peak_count必须大于0。")
    if not 1 <= minimum_valid_training_count <= spectra.shape[0]:
        raise ValueError(
            "minimum_valid_training_count必须位于"
            f"[1,{spectra.shape[0]}]。"
        )

    sigma_samples = max(reference_sigma_cm1 / spacing, 1.0e-6)

    reference = np.median(spectra, axis=0)
    reference_baseline = gaussian_filter1d(
        reference,
        sigma=sigma_samples,
        mode="nearest",
    )
    reference_peak_signal = reference - reference_baseline
    signal_span = float(np.ptp(reference_peak_signal))

    if signal_span <= np.finfo(np.float64).eps:
        raise RuntimeError("训练集中位数谱没有可识别的峰形变化。")

    detected_indices, properties = find_peaks(
        reference_peak_signal,
        prominence=minimum_relative_prominence * signal_span,
        distance=max(
            1,
            int(round(minimum_peak_distance_cm1 / spacing)),
        ),
    )

    if detected_indices.size == 0:
        raise RuntimeError(
            "training_guard未能从当前训练集自动识别稳定峰。"
        )

    valid_edge = (
        (axis[detected_indices] - metric_half_width_cm1 >= axis[0])
        & (axis[detected_indices] + metric_half_width_cm1 <= axis[-1])
    )
    detected_indices = detected_indices[valid_edge]
    prominences = np.asarray(
        properties["prominences"],
        dtype=np.float64,
    )[valid_edge]

    if detected_indices.size == 0:
        raise RuntimeError("自动峰全部位于指标窗口边缘，无法拟合。")

    if detected_indices.size > maximum_peak_count:
        selected = np.argsort(prominences)[-maximum_peak_count:]
        detected_indices = detected_indices[selected]
        prominences = prominences[selected]

    order = np.argsort(detected_indices)
    detected_indices = detected_indices[order]
    prominences = prominences[order]

    training_baseline = gaussian_filter1d(
        spectra,
        sigma=sigma_samples,
        axis=1,
        mode="nearest",
    )
    training_peak_signal = spectra - training_baseline

    peak_states: list[dict[str, Any]] = []

    for peak_number, peak_index in enumerate(detected_indices, start=1):
        center = float(axis[peak_index])
        measured = [
            _measure_peak(
                axis=axis,
                peak_signal=training_peak_signal[row_index],
                center_cm1=center,
                half_width_cm1=metric_half_width_cm1,
            )
            for row_index in range(spectra.shape[0])
        ]

        positions = np.asarray(
            [item["position_cm1"] for item in measured],
            dtype=np.float64,
        )
        heights = np.asarray(
            [item["height"] for item in measured],
            dtype=np.float64,
        )
        widths = np.asarray(
            [item["fwhm_cm1"] for item in measured],
            dtype=np.float64,
        )

        valid_position = positions[np.isfinite(positions)]
        valid_height = heights[np.isfinite(heights)]
        valid_width = widths[
            np.isfinite(widths)
            & (widths > 0.0)
        ]

        if min(
            valid_position.size,
            valid_height.size,
            valid_width.size,
        ) < minimum_valid_training_count:
            continue

        position_lower = (
            float(np.min(valid_position))
            - position_margin_cm1
        )
        position_upper = (
            float(np.max(valid_position))
            + position_margin_cm1
        )
        height_lower, height_upper = _expanded_bounds(
            valid_height,
            margin_iqr=bound_margin_iqr,
            lower_floor=0.0,
        )
        width_lower, width_upper = _expanded_bounds(
            valid_width,
            margin_iqr=bound_margin_iqr,
            lower_floor=spacing,
        )

        peak_states.append(
            {
                "peak_id": int(peak_number),
                "reference_position_cm1": center,
                "reference_prominence": float(
                    prominences[peak_number - 1]
                ),
                "valid_training_count_position": int(
                    valid_position.size
                ),
                "valid_training_count_height": int(
                    valid_height.size
                ),
                "valid_training_count_width": int(
                    valid_width.size
                ),
                "position_lower_cm1": float(position_lower),
                "position_upper_cm1": float(position_upper),
                "height_lower": float(height_lower),
                "height_upper": float(height_upper),
                "fwhm_lower_cm1": float(width_lower),
                "fwhm_upper_cm1": float(width_upper),
                "training_position_median_cm1": float(
                    np.median(valid_position)
                ),
                "training_height_median": float(
                    np.median(valid_height)
                ),
                "training_fwhm_median_cm1": float(
                    np.median(valid_width)
                ),
            }
        )

    if not peak_states:
        raise RuntimeError(
            "没有任何自动峰具有足够的训练样本支持，"
            "无法建立training_guard。"
        )

    # ----------------------------------------------------------
    # 只保留当前训练集中的主要稳定峰。
    #
    # 不写死任何 Raman shift。
    # 阈值完全由本训练集所有候选峰的峰高中位数决定。
    # ----------------------------------------------------------

    maximum_training_height_median = max(
        float(
            peak_state[
                "training_height_median"
            ]
        )
        for peak_state in peak_states
    )

    guard_height_threshold = (
        minimum_guard_peak_median_height_fraction
        * maximum_training_height_median
    )

    peak_states = [
        peak_state
        for peak_state in peak_states
        if float(
            peak_state[
                "training_height_median"
            ]
        )
        >= guard_height_threshold
    ]

    if not peak_states:
        raise RuntimeError(
            "主要峰高度筛选后没有剩余guard峰；"
            "请检查"
            "minimum_guard_peak_median_height_fraction。"
        )

    # 重新连续编号，避免原候选峰被删除后peak_id跳号。
    for new_peak_id, peak_state in enumerate(
        peak_states,
        start=1,
    ):
        peak_state["peak_id"] = int(new_peak_id)

    minima = np.min(spectra, axis=1)
    minima_iqr = _iqr(minima)
    minima_fallback = max(
        float(np.std(minima, ddof=0)),
        1.0e-8,
    )
    minima_spread = max(minima_iqr, minima_fallback)
    minimum_lower_bound = float(
        np.min(minima) - bound_margin_iqr * minima_spread
    )

    return {
        "schema_version": 1,
        "enabled": True,
        "domain": "raw_intensity_on_output_raman_axis",
        "number_of_training_spectra": int(spectra.shape[0]),
        "axis_start_cm1": float(axis[0]),
        "axis_end_cm1": float(axis[-1]),
        "axis_length": int(axis.size),
        "axis_spacing_median_cm1": spacing,
        "reference_smoothing_sigma_cm1": reference_sigma_cm1,
        "minimum_relative_prominence": minimum_relative_prominence,
        "minimum_peak_distance_cm1": minimum_peak_distance_cm1,
        "metric_half_width_cm1": metric_half_width_cm1,
        "position_margin_cm1": position_margin_cm1,
        "bound_margin_iqr": bound_margin_iqr,
        "minimum_valid_training_count": minimum_valid_training_count,
        "minimum_guard_peak_median_height_fraction": (
            minimum_guard_peak_median_height_fraction
        ),
        "guard_peak_height_threshold": float(
            guard_height_threshold
        ),
        "negative_minimum_lower_bound": minimum_lower_bound,
        "training_minimum_median": float(np.median(minima)),
        "training_minimum_p05": float(np.percentile(minima, 5.0)),
        "training_minimum_minimum": float(np.min(minima)),
        "peaks": peak_states,
    }


class SersTrainingGuard:
    """Evaluate generated spectra against a fitted training-set guard."""

    def __init__(
        self,
        *,
        raman_shift: np.ndarray,
        state: dict[str, Any],
        configuration: dict[str, Any],
    ) -> None:
        if not isinstance(state, dict):
            raise TypeError("training_guard state必须是字典。")
        if not isinstance(configuration, dict):
            raise TypeError("training_guard配置必须是字典。")

        self.state = deepcopy(state)
        self.configuration = deepcopy(configuration)
        self.raman_shift = _as_axis(raman_shift)

        if int(self.state.get("schema_version", 0)) != 1:
            raise ValueError("不支持的training_guard state版本。")
        if self.state.get("domain") != "raw_intensity_on_output_raman_axis":
            raise ValueError("training_guard state数据域无效。")
        if int(self.state.get("axis_length", -1)) != self.raman_shift.size:
            raise ValueError("training_guard state轴长度与当前输出轴不一致。")

        tolerance = max(
            float(self.state.get("axis_spacing_median_cm1", 1.0)),
            1.0e-6,
        )
        if (
            abs(float(self.state["axis_start_cm1"]) - float(self.raman_shift[0]))
            > tolerance
            or abs(float(self.state["axis_end_cm1"]) - float(self.raman_shift[-1]))
            > tolerance
        ):
            raise ValueError("training_guard state Raman范围与当前输出轴不一致。")

        self.check_peak_position = bool(
            self.configuration.get("check_peak_position", True)
        )
        self.check_peak_height = bool(
            self.configuration.get("check_peak_height", True)
        )
        self.check_peak_width = bool(
            self.configuration.get("check_peak_width", True)
        )
        self.check_negative_minimum = bool(
            self.configuration.get("check_negative_minimum", True)
        )
        self.maximum_peak_violation_fraction = _unit_interval(
            self.configuration.get(
                "maximum_peak_violation_fraction",
                0.20,
            ),
            "maximum_peak_violation_fraction",
        )

        self._sigma_samples = max(
            float(self.state["reference_smoothing_sigma_cm1"])
            / float(self.state["axis_spacing_median_cm1"]),
            1.0e-6,
        )

    def evaluate(
        self,
        spectra: np.ndarray,
        *,
        spectrum_names: list[str] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
        values = _as_2d_finite(spectra, "generated_spectra")
        if values.shape[1] != self.raman_shift.size:
            raise ValueError(
                "生成光谱长度与training_guard Raman轴不一致。"
            )

        if spectrum_names is None:
            names = [
                f"generated_{index + 1:04d}"
                for index in range(values.shape[0])
            ]
        else:
            if len(spectrum_names) != values.shape[0]:
                raise ValueError("spectrum_names数量与生成光谱数量不一致。")
            names = [str(value) for value in spectrum_names]

        baseline = gaussian_filter1d(
            values,
            sigma=self._sigma_samples,
            axis=1,
            mode="nearest",
        )
        peak_signal = values - baseline

        accepted = np.ones(values.shape[0], dtype=bool)
        spectrum_rows: list[dict[str, Any]] = []
        peak_rows: list[dict[str, Any]] = []

        peaks = list(self.state["peaks"])
        maximum_allowed_peak_violations = int(
            np.floor(
                self.maximum_peak_violation_fraction
                * len(peaks)
                + 1.0e-12
            )
        )

        for spectrum_index in range(values.shape[0]):
            minimum_value = float(np.min(values[spectrum_index]))
            negative_violation = bool(
                self.check_negative_minimum
                and minimum_value
                < float(self.state["negative_minimum_lower_bound"])
            )

            peak_violation_count = 0
            position_violation_count = 0
            height_violation_count = 0
            width_violation_count = 0
            invalid_metric_count = 0

            for peak_state in peaks:
                measured = _measure_peak(
                    axis=self.raman_shift,
                    peak_signal=peak_signal[spectrum_index],
                    center_cm1=float(
                        peak_state["reference_position_cm1"]
                    ),
                    half_width_cm1=float(
                        self.state["metric_half_width_cm1"]
                    ),
                )

                position = float(measured["position_cm1"])
                height = float(measured["height"])
                width = float(measured["fwhm_cm1"])

                invalid_position = not np.isfinite(position)

                invalid_height = not np.isfinite(height)

                invalid_width = (
                    not np.isfinite(width)
                    or width <= 0.0
                )

                invalid = bool(
                    invalid_position
                    or invalid_height
                    or invalid_width
                )

                position_violation = bool(
                    self.check_peak_position
                    and (
                        invalid_position
                        or position
                        < float(
                            peak_state[
                                "position_lower_cm1"
                            ]
                        )
                        or position
                        > float(
                            peak_state[
                                "position_upper_cm1"
                            ]
                        )
                    )
                )

                height_violation = bool(
                    self.check_peak_height
                    and (
                        invalid_height
                        or height
                        < float(
                            peak_state[
                                "height_lower"
                            ]
                        )
                        or height
                        > float(
                            peak_state[
                                "height_upper"
                            ]
                        )
                    )
                )

                width_violation = bool(
                    self.check_peak_width
                    and (
                        invalid_width
                        or width
                        < float(
                            peak_state[
                                "fwhm_lower_cm1"
                            ]
                        )
                        or width
                        > float(
                            peak_state[
                                "fwhm_upper_cm1"
                            ]
                        )
                    )
                )

                peak_violation = bool(
                    position_violation
                    or height_violation
                    or width_violation
                )

                if invalid:
                    invalid_metric_count += 1
                if position_violation:
                    position_violation_count += 1
                if height_violation:
                    height_violation_count += 1
                if width_violation:
                    width_violation_count += 1
                if peak_violation:
                    peak_violation_count += 1

                peak_rows.append(
                    {
                        "spectrum_index": int(spectrum_index),
                        "spectrum_name": names[spectrum_index],
                        "peak_id": int(peak_state["peak_id"]),
                        "reference_position_cm1": float(
                            peak_state["reference_position_cm1"]
                        ),
                        "position_cm1": position,
                        "height": height,
                        "fwhm_cm1": width,
                        "position_violation": position_violation,
                        "height_violation": height_violation,
                        "width_violation": width_violation,
                        "peak_violation": peak_violation,
                    }
                )

            rejected = bool(
                negative_violation
                or peak_violation_count
                > maximum_allowed_peak_violations
            )
            accepted[spectrum_index] = not rejected

            spectrum_rows.append(
                {
                    "spectrum_index": int(spectrum_index),
                    "spectrum_name": names[spectrum_index],
                    "minimum_intensity": minimum_value,
                    "negative_minimum_lower_bound": float(
                        self.state["negative_minimum_lower_bound"]
                    ),
                    "negative_violation": negative_violation,
                    "number_of_guard_peaks": int(len(peaks)),
                    "maximum_allowed_peak_violations": int(
                        maximum_allowed_peak_violations
                    ),
                    "peak_violation_count": int(peak_violation_count),
                    "position_violation_count": int(
                        position_violation_count
                    ),
                    "height_violation_count": int(
                        height_violation_count
                    ),
                    "width_violation_count": int(
                        width_violation_count
                    ),
                    "invalid_metric_count": int(invalid_metric_count),
                    "accepted": bool(not rejected),
                }
            )

        return accepted, spectrum_rows, peak_rows

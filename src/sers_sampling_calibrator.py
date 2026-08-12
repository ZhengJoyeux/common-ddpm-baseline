"""Training-reference-driven calibration for generated SERS spectra.

This module performs two sampling-only operations:

1. suppress overly broad non-peak fluctuations while protecting peak windows
   detected automatically from the checkpoint PCA mean spectrum;
2. apply a small whole-spectrum Raman-axis drift so all peaks in one spectrum
   move coherently and their relative spacing is preserved.

No pesticide-specific peak position is hard-coded.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class SersSamplingCalibrator:
    """Calibrate non-peak noise width and coherent Raman-shift variation."""

    def __init__(
        self,
        *,
        raman_shift: np.ndarray,
        reference_spectrum: np.ndarray,
        configuration: dict[str, Any],
        random_seed: int | None,
    ) -> None:
        if not isinstance(configuration, dict):
            raise TypeError("sampling_calibration配置必须是字典。")

        self.configuration = deepcopy(configuration)
        self.raman_shift = self._validate_axis(raman_shift)
        self.reference_spectrum = self._validate_reference(
            reference_spectrum,
            expected_length=self.raman_shift.size,
        )
        self.random_seed = None if random_seed is None else int(random_seed)
        self._random_generator = np.random.default_rng(
            None if self.random_seed is None else self.random_seed + 104729
        )

        self.non_peak_noise_configuration = self._read_section(
            "non_peak_noise"
        )
        self.raman_shift_jitter_configuration = self._read_section(
            "raman_shift_jitter"
        )
        self.non_peak_noise_enabled = bool(
            self.non_peak_noise_configuration.get("enabled", False)
        )
        self.raman_shift_jitter_enabled = bool(
            self.raman_shift_jitter_configuration.get("enabled", False)
        )

        if not self.non_peak_noise_enabled and not self.raman_shift_jitter_enabled:
            raise ValueError(
                "sampling_calibration已启用，但non_peak_noise和"
                "raman_shift_jitter均未启用。"
            )

        self._axis_spacing = float(np.median(np.diff(self.raman_shift)))
        self._peak_indices = np.empty(0, dtype=np.int64)
        self._peak_protection_weight = np.zeros(
            self.raman_shift.size,
            dtype=np.float64,
        )
        self._last_shift_offsets = np.empty(0, dtype=np.float64)
        self._last_noise_width_before = 0.0
        self._last_noise_width_after = 0.0

        if self.non_peak_noise_enabled:
            self._prepare_non_peak_noise_calibration()

        if self.raman_shift_jitter_enabled:
            self._validate_shift_configuration()

    @staticmethod
    def _validate_axis(values: np.ndarray) -> np.ndarray:
        axis = np.asarray(values, dtype=np.float64).reshape(-1)
        if axis.size < 3:
            raise ValueError("拉曼位移轴至少需要3个点。")
        if not np.isfinite(axis).all():
            raise ValueError("拉曼位移轴包含NaN或无穷值。")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError("拉曼位移轴必须严格递增。")
        return axis.copy()

    @staticmethod
    def _validate_reference(
        values: np.ndarray,
        *,
        expected_length: int,
    ) -> np.ndarray:
        reference = np.asarray(values, dtype=np.float64).reshape(-1)
        if reference.size != expected_length:
            raise ValueError(
                "参考光谱长度与拉曼位移轴不一致："
                f"{reference.size} != {expected_length}。"
            )
        if not np.isfinite(reference).all():
            raise ValueError("参考光谱包含NaN或无穷值。")
        return reference.copy()

    def _read_section(self, name: str) -> dict[str, Any]:
        section = self.configuration.get(name, {}) or {}
        if not isinstance(section, dict):
            raise TypeError(f"sampling_calibration.{name}必须是字典。")
        return deepcopy(section)

    @staticmethod
    def _finite_positive(value: Any, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name}必须是数值。") from error
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{field_name}必须是有限正数。")
        return parsed

    @staticmethod
    def _unit_interval(value: Any, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name}必须是数值。") from error
        if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"{field_name}必须位于[0,1]范围内。")
        return parsed

    def _sigma_in_samples(self, sigma_cm1: float) -> float:
        return max(float(sigma_cm1) / self._axis_spacing, 1.0e-6)

    def _prepare_non_peak_noise_calibration(self) -> None:
        config = self.non_peak_noise_configuration
        reference_sigma_cm1 = self._finite_positive(
            config.get("reference_smoothing_sigma_cm1", 30.0),
            "reference_smoothing_sigma_cm1",
        )
        fine_sigma_cm1 = self._finite_positive(
            config.get("fine_sigma_cm1", 1.5),
            "fine_sigma_cm1",
        )
        broad_sigma_cm1 = self._finite_positive(
            config.get("broad_sigma_cm1", 7.0),
            "broad_sigma_cm1",
        )
        if broad_sigma_cm1 <= fine_sigma_cm1:
            raise ValueError("broad_sigma_cm1必须大于fine_sigma_cm1。")

        self._fine_sigma_samples = self._sigma_in_samples(fine_sigma_cm1)
        self._broad_sigma_samples = self._sigma_in_samples(broad_sigma_cm1)
        self._middle_component_scale = self._unit_interval(
            config.get("middle_component_scale", 0.55),
            "middle_component_scale",
        )
        self._fine_component_scale = self._unit_interval(
            config.get("fine_component_scale", 0.85),
            "fine_component_scale",
        )

        minimum_relative_prominence = self._unit_interval(
            config.get("minimum_relative_prominence", 0.06),
            "minimum_relative_prominence",
        )
        minimum_peak_distance_cm1 = self._finite_positive(
            config.get("minimum_peak_distance_cm1", 12.0),
            "minimum_peak_distance_cm1",
        )
        protection_half_width_cm1 = self._finite_positive(
            config.get("peak_protection_half_width_cm1", 24.0),
            "peak_protection_half_width_cm1",
        )
        transition_width_cm1 = self._finite_positive(
            config.get("transition_width_cm1", 6.0),
            "transition_width_cm1",
        )
        maximum_peak_count = int(config.get("maximum_peak_count", 20))
        if maximum_peak_count <= 0:
            raise ValueError("maximum_peak_count必须大于0。")

        reference_baseline = gaussian_filter1d(
            self.reference_spectrum,
            sigma=self._sigma_in_samples(reference_sigma_cm1),
            mode="nearest",
        )
        peak_signal = self.reference_spectrum - reference_baseline
        signal_span = float(np.ptp(peak_signal))
        if signal_span <= np.finfo(np.float64).eps:
            raise RuntimeError("PCA均值参考谱没有可识别的峰形变化。")

        peak_indices, properties = find_peaks(
            peak_signal,
            prominence=minimum_relative_prominence * signal_span,
            distance=max(
                1,
                int(round(minimum_peak_distance_cm1 / self._axis_spacing)),
            ),
        )
        if peak_indices.size == 0:
            raise RuntimeError(
                "未能从checkpoint的PCA均值谱自动识别特征峰；"
                "请检查minimum_relative_prominence。"
            )

        if peak_indices.size > maximum_peak_count:
            prominences = np.asarray(properties["prominences"], dtype=np.float64)
            selected = np.argsort(prominences)[-maximum_peak_count:]
            peak_indices = np.sort(peak_indices[selected])

        self._peak_indices = peak_indices.astype(np.int64, copy=False)
        protection_weight = np.zeros(self.raman_shift.size, dtype=np.float64)
        outer_half_width = protection_half_width_cm1 + transition_width_cm1

        for peak_index in self._peak_indices:
            distance = np.abs(self.raman_shift - self.raman_shift[peak_index])
            current = np.zeros_like(distance)
            current[distance <= protection_half_width_cm1] = 1.0
            transition = (
                (distance > protection_half_width_cm1)
                & (distance < outer_half_width)
            )
            phase = (
                distance[transition] - protection_half_width_cm1
            ) / transition_width_cm1
            current[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
            protection_weight = np.maximum(protection_weight, current)

        self._peak_protection_weight = protection_weight

    def _validate_shift_configuration(self) -> None:
        config = self.raman_shift_jitter_configuration
        distribution = str(config.get("distribution", "truncated_normal")).strip().lower()
        if distribution != "truncated_normal":
            raise ValueError(
                "raman_shift_jitter.distribution当前只支持truncated_normal。"
            )
        self._shift_standard_deviation_cm1 = self._finite_positive(
            config.get("standard_deviation_cm1", 3.0),
            "standard_deviation_cm1",
        )
        self._maximum_absolute_shift_cm1 = self._finite_positive(
            config.get("maximum_absolute_shift_cm1", 5.0),
            "maximum_absolute_shift_cm1",
        )

    def _calibrate_non_peak_noise(self, spectra: np.ndarray) -> np.ndarray:
        fine_smooth = gaussian_filter1d(
            spectra,
            sigma=self._fine_sigma_samples,
            axis=1,
            mode="nearest",
        )
        broad = gaussian_filter1d(
            spectra,
            sigma=self._broad_sigma_samples,
            axis=1,
            mode="nearest",
        )
        middle_component = fine_smooth - broad
        fine_component = spectra - fine_smooth
        calibrated_non_peak = (
            broad
            + self._middle_component_scale * middle_component
            + self._fine_component_scale * fine_component
        )
        peak_weight = self._peak_protection_weight[np.newaxis, :]
        return (
            peak_weight * spectra
            + (1.0 - peak_weight) * calibrated_non_peak
        )

    def _measure_non_peak_noise_width(self, spectra: np.ndarray) -> float:
        """Measure median 95% vertical span around a local broad trend."""

        non_peak = self._peak_protection_weight <= 0.05
        if int(np.count_nonzero(non_peak)) < 10:
            raise RuntimeError("自动峰区保护后剩余非峰区点数不足。")
        broad = gaussian_filter1d(
            spectra,
            sigma=self._broad_sigma_samples,
            axis=1,
            mode="nearest",
        )
        local_fluctuation = spectra - broad
        widths = (
            np.percentile(local_fluctuation[:, non_peak], 97.5, axis=1)
            - np.percentile(local_fluctuation[:, non_peak], 2.5, axis=1)
        )
        return float(np.median(widths))

    def _sample_shift_offsets(self, number_of_spectra: int) -> np.ndarray:
        offsets = np.empty(number_of_spectra, dtype=np.float64)
        for index in range(number_of_spectra):
            for _ in range(100000):
                candidate = float(
                    self._random_generator.normal(
                        loc=0.0,
                        scale=self._shift_standard_deviation_cm1,
                    )
                )
                if abs(candidate) <= self._maximum_absolute_shift_cm1:
                    offsets[index] = candidate
                    break
            else:
                raise RuntimeError("截断正态拉曼位移采样未能收敛。")
        return offsets

    def _apply_axis_shift(
        self,
        spectra: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        shifted = np.empty_like(spectra, dtype=np.float64)
        for index, (spectrum, offset) in enumerate(
            zip(spectra, offsets, strict=True)
        ):
            shifted[index] = np.interp(
                self.raman_shift - offset,
                self.raman_shift,
                spectrum,
                left=float(spectrum[0]),
                right=float(spectrum[-1]),
            )
        return shifted

    def apply(self, spectra: np.ndarray) -> np.ndarray:
        """Apply configured calibration to spectra on the output Raman axis."""

        values = np.asarray(spectra, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(
                "sampling calibration要求二维数组[N,L]，"
                f"实际形状为{values.shape}。"
            )
        if values.shape[1] != self.raman_shift.size:
            raise ValueError(
                "生成光谱长度与sampling calibration拉曼轴不一致。"
            )
        if not np.isfinite(values).all():
            raise ValueError("生成光谱包含NaN或无穷值。")

        calibrated = values.copy()
        if self.non_peak_noise_enabled:
            self._last_noise_width_before = (
                self._measure_non_peak_noise_width(calibrated)
            )
            calibrated = self._calibrate_non_peak_noise(calibrated)
            self._last_noise_width_after = (
                self._measure_non_peak_noise_width(calibrated)
            )
        if self.raman_shift_jitter_enabled:
            self._last_shift_offsets = self._sample_shift_offsets(
                calibrated.shape[0]
            )
            calibrated = self._apply_axis_shift(
                calibrated,
                self._last_shift_offsets,
            )
        else:
            self._last_shift_offsets = np.zeros(
                calibrated.shape[0], dtype=np.float64
            )

        if not np.isfinite(calibrated).all():
            raise RuntimeError("sampling calibration结果包含NaN或无穷值。")
        return calibrated.astype(np.float32, copy=False)

    def summary(self) -> dict[str, Any]:
        """Return concise diagnostics for terminal logging and tests."""

        if self._last_shift_offsets.size:
            shift_min = float(np.min(self._last_shift_offsets))
            shift_max = float(np.max(self._last_shift_offsets))
            shift_std = float(np.std(self._last_shift_offsets))
        else:
            shift_min = 0.0
            shift_max = 0.0
            shift_std = 0.0

        return {
            "non_peak_noise_enabled": self.non_peak_noise_enabled,
            "raman_shift_jitter_enabled": self.raman_shift_jitter_enabled,
            "detected_peak_count": int(self._peak_indices.size),
            "middle_component_scale": getattr(
                self, "_middle_component_scale", 1.0
            ),
            "fine_component_scale": getattr(
                self, "_fine_component_scale", 1.0
            ),
            "noise_width_before": self._last_noise_width_before,
            "noise_width_after": self._last_noise_width_after,
            "maximum_absolute_shift_cm1": getattr(
                self, "_maximum_absolute_shift_cm1", 0.0
            ),
            "sampled_shift_min_cm1": shift_min,
            "sampled_shift_max_cm1": shift_max,
            "sampled_shift_std_cm1": shift_std,
        }

"""Compare real training SERS distribution with generated spectra.

Purpose
-------
1. Reconstruct the exact training subset used by the checkpoint.
2. Detect peak regions automatically from the real training mean spectrum.
3. Compare real and generated distributions in:
   - whole spectrum
   - peak regions
   - non-peak regions
   - broad/background component
   - local fluctuation component
4. Compare peak position, height and approximate FWHM.
5. Estimate how much broad inter-spectrum variation should be compressed.

This script is diagnostic only. It does not modify checkpoints,
configuration files or generated spectra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import load_configuration
from src.dataset_splitter import split_spectrum_collection
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "比较checkpoint真实训练SERS与生成SERS的"
            "峰区、非峰区、背景及噪声分布。"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="当前项目YAML配置。",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="D2.5 checkpoint路径。",
    )

    parser.add_argument(
        "--generated",
        required=True,
        help="待评价的生成xlsx或csv。",
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        help="诊断结果输出目录。",
    )

    return parser.parse_args()


def read_generated_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"生成光谱文件不存在：{path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            "生成文件只支持xlsx或csv。"
        )

    if frame.shape[1] < 2:
        raise ValueError(
            "生成文件至少需要第一列Raman shift和一列光谱。"
        )

    axis = pd.to_numeric(
        frame.iloc[:, 0],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    spectra = (
        frame.iloc[:, 1:]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .to_numpy(
            dtype=np.float64
        )
        .T
    )

    if spectra.ndim != 2:
        raise ValueError(
            f"生成光谱形状异常：{spectra.shape}"
        )

    if spectra.shape[1] != axis.size:
        raise ValueError(
            "生成光谱点数与Raman轴长度不一致。"
        )

    if not np.isfinite(axis).all():
        raise ValueError(
            "生成Raman轴含NaN或无穷值。"
        )

    if not np.all(
        np.diff(axis) > 0
    ):
        raise ValueError(
            "生成Raman轴必须严格递增。"
        )

    if not np.isfinite(
        spectra
    ).all():
        raise ValueError(
            "生成光谱含NaN或无穷值。"
        )

    return axis, spectra


def sigma_samples(
    sigma_cm1: float,
    axis: np.ndarray,
) -> float:
    spacing = float(
        np.median(
            np.diff(axis)
        )
    )

    if spacing <= 0.0:
        raise ValueError(
            "Raman轴间距必须大于0。"
        )

    return max(
        float(sigma_cm1) / spacing,
        1.0e-6,
    )


def detect_peak_regions(
    *,
    axis: np.ndarray,
    training_spectra: np.ndarray,
    reference_smoothing_sigma_cm1: float,
    minimum_relative_prominence: float,
    minimum_peak_distance_cm1: float,
    maximum_peak_count: int,
    peak_protection_half_width_cm1: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Detect current-data peaks; never hard-code Raman positions."""

    reference = np.mean(
        training_spectra,
        axis=0,
    )

    reference_baseline = gaussian_filter1d(
        reference,
        sigma=sigma_samples(
            reference_smoothing_sigma_cm1,
            axis,
        ),
        mode="nearest",
    )

    peak_signal = (
        reference
        - reference_baseline
    )

    signal_span = float(
        np.ptp(
            peak_signal
        )
    )

    if signal_span <= np.finfo(
        np.float64
    ).eps:
        raise RuntimeError(
            "训练集平均谱没有足够的峰形变化。"
        )

    spacing = float(
        np.median(
            np.diff(axis)
        )
    )

    peak_indices, properties = find_peaks(
        peak_signal,
        prominence=(
            float(
                minimum_relative_prominence
            )
            * signal_span
        ),
        distance=max(
            1,
            int(
                round(
                    minimum_peak_distance_cm1
                    / spacing
                )
            ),
        ),
    )

    if peak_indices.size == 0:
        raise RuntimeError(
            "没有从当前训练数据自动识别到峰。"
        )

    if (
        peak_indices.size
        > int(maximum_peak_count)
    ):
        prominences = np.asarray(
            properties["prominences"],
            dtype=np.float64,
        )

        selected = np.argsort(
            prominences
        )[
            -int(maximum_peak_count):
        ]

        peak_indices = np.sort(
            peak_indices[selected]
        )

    peak_mask = np.zeros(
        axis.size,
        dtype=bool,
    )

    for peak_index in peak_indices:
        center = float(
            axis[
                int(peak_index)
            ]
        )

        peak_mask |= (
            np.abs(
                axis - center
            )
            <= float(
                peak_protection_half_width_cm1
            )
        )

    return (
        peak_indices.astype(
            np.int64
        ),
        peak_mask,
        reference,
    )


def pointwise_statistics(
    spectra: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(
            spectra,
            axis=0,
        ),
        "std": np.std(
            spectra,
            axis=0,
            ddof=0,
        ),
        "p2_5": np.percentile(
            spectra,
            2.5,
            axis=0,
        ),
        "p25": np.percentile(
            spectra,
            25.0,
            axis=0,
        ),
        "median": np.percentile(
            spectra,
            50.0,
            axis=0,
        ),
        "p75": np.percentile(
            spectra,
            75.0,
            axis=0,
        ),
        "p97_5": np.percentile(
            spectra,
            97.5,
            axis=0,
        ),
    }


def median_band_width(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    widths = (
        statistics["p97_5"]
        - statistics["p2_5"]
    )

    return float(
        np.median(
            widths[mask]
        )
    )


def mean_pointwise_std(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    return float(
        np.mean(
            statistics["std"][
                mask
            ]
        )
    )


def median_pointwise_std(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    return float(
        np.median(
            statistics["std"][
                mask
            ]
        )
    )


def median_iqr_width(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Median pointwise interquartile width in a selected Raman region."""

    widths = (
        statistics["p75"]
        - statistics["p25"]
    )

    return float(
        np.median(
            widths[mask]
        )
    )


def matched_sample_bootstrap(
    *,
    real_spectra: np.ndarray,
    generated_spectra: np.ndarray,
    axis: np.ndarray,
    non_peak_mask: np.ndarray,
    broad_sigma_cm1: float,
    repetitions: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare real and generated distributions using the same sample count.

    The real training set is kept fixed.
    Each repetition randomly draws the same number of generated spectra
    without replacement.

    This avoids directly comparing a 16-spectrum real P2.5-P97.5 envelope
    with a 100-spectrum generated envelope.
    """

    real_values = np.asarray(
        real_spectra,
        dtype=np.float64,
    )

    generated_values = np.asarray(
        generated_spectra,
        dtype=np.float64,
    )

    if real_values.ndim != 2:
        raise ValueError(
            "real_spectra必须是二维数组。"
        )

    if generated_values.ndim != 2:
        raise ValueError(
            "generated_spectra必须是二维数组。"
        )

    if (
        real_values.shape[1]
        != generated_values.shape[1]
    ):
        raise ValueError(
            "真实谱与生成谱长度不一致。"
        )

    matched_number = int(
        real_values.shape[0]
    )

    if generated_values.shape[0] < matched_number:
        raise ValueError(
            "生成光谱数量少于真实训练光谱数量，"
            "不能执行matched-size bootstrap。"
        )

    repetitions = int(
        repetitions
    )

    if repetitions <= 0:
        raise ValueError(
            "repetitions必须大于0。"
        )

    real_statistics = pointwise_statistics(
        real_values
    )

    (
        real_broad,
        real_local,
    ) = build_frequency_components(
        real_values,
        axis,
        broad_sigma_cm1,
    )

    real_broad_statistics = (
        pointwise_statistics(
            real_broad
        )
    )

    real_local_statistics = (
        pointwise_statistics(
            real_local
        )
    )

    real_metrics = {
        "non_peak_95_band_median":
            median_band_width(
                real_statistics,
                non_peak_mask,
            ),

        "non_peak_iqr_band_median":
            median_iqr_width(
                real_statistics,
                non_peak_mask,
            ),

        "non_peak_pointwise_std_mean":
            mean_pointwise_std(
                real_statistics,
                non_peak_mask,
            ),

        "broad_95_band_median":
            median_band_width(
                real_broad_statistics,
                non_peak_mask,
            ),

        "broad_iqr_band_median":
            median_iqr_width(
                real_broad_statistics,
                non_peak_mask,
            ),

        "broad_pointwise_std_mean":
            mean_pointwise_std(
                real_broad_statistics,
                non_peak_mask,
            ),

        "local_95_band_median":
            median_band_width(
                real_local_statistics,
                non_peak_mask,
            ),

        "local_iqr_band_median":
            median_iqr_width(
                real_local_statistics,
                non_peak_mask,
            ),

        "local_pointwise_std_mean":
            mean_pointwise_std(
                real_local_statistics,
                non_peak_mask,
            ),

        "per_spectrum_local_noise_95_span_median":
            local_noise_span_per_spectrum(
                real_values,
                axis,
                non_peak_mask,
                broad_sigma_cm1,
            ),
    }

    random_generator = np.random.default_rng(
        int(random_seed)
    )

    bootstrap_rows: list[dict[str, float | int]] = []

    for repetition in range(
        repetitions
    ):
        selected_indices = (
            random_generator.choice(
                generated_values.shape[0],
                size=matched_number,
                replace=False,
            )
        )

        subset = generated_values[
            selected_indices
        ]

        subset_statistics = (
            pointwise_statistics(
                subset
            )
        )

        (
            subset_broad,
            subset_local,
        ) = build_frequency_components(
            subset,
            axis,
            broad_sigma_cm1,
        )

        subset_broad_statistics = (
            pointwise_statistics(
                subset_broad
            )
        )

        subset_local_statistics = (
            pointwise_statistics(
                subset_local
            )
        )

        bootstrap_rows.append(
            {
                "repetition": repetition + 1,

                "non_peak_95_band_median":
                    median_band_width(
                        subset_statistics,
                        non_peak_mask,
                    ),

                "non_peak_iqr_band_median":
                    median_iqr_width(
                        subset_statistics,
                        non_peak_mask,
                    ),

                "non_peak_pointwise_std_mean":
                    mean_pointwise_std(
                        subset_statistics,
                        non_peak_mask,
                    ),

                "broad_95_band_median":
                    median_band_width(
                        subset_broad_statistics,
                        non_peak_mask,
                    ),

                "broad_iqr_band_median":
                    median_iqr_width(
                        subset_broad_statistics,
                        non_peak_mask,
                    ),

                "broad_pointwise_std_mean":
                    mean_pointwise_std(
                        subset_broad_statistics,
                        non_peak_mask,
                    ),

                "local_95_band_median":
                    median_band_width(
                        subset_local_statistics,
                        non_peak_mask,
                    ),

                "local_iqr_band_median":
                    median_iqr_width(
                        subset_local_statistics,
                        non_peak_mask,
                    ),

                "local_pointwise_std_mean":
                    mean_pointwise_std(
                        subset_local_statistics,
                        non_peak_mask,
                    ),

                "per_spectrum_local_noise_95_span_median":
                    local_noise_span_per_spectrum(
                        subset,
                        axis,
                        non_peak_mask,
                        broad_sigma_cm1,
                    ),
            }
        )

    raw_table = pd.DataFrame(
        bootstrap_rows
    )

    summary_rows = []

    for metric_name, real_value in (
        real_metrics.items()
    ):
        generated_distribution = (
            raw_table[
                metric_name
            ].to_numpy(
                dtype=np.float64
            )
        )

        generated_median = float(
            np.median(
                generated_distribution
            )
        )

        generated_p05 = float(
            np.percentile(
                generated_distribution,
                5.0,
            )
        )

        generated_p95 = float(
            np.percentile(
                generated_distribution,
                95.0,
            )
        )

        if abs(
            real_value
        ) > 1.0e-12:
            ratio = (
                generated_median
                / real_value
            )
        else:
            ratio = float(
                "nan"
            )

        summary_rows.append(
            {
                "metric": metric_name,
                "real_training_n16": real_value,
                "generated_n16_bootstrap_median":
                    generated_median,
                "generated_n16_bootstrap_p05":
                    generated_p05,
                "generated_n16_bootstrap_p95":
                    generated_p95,
                "generated_to_real_ratio":
                    ratio,
            }
        )

    summary_table = pd.DataFrame(
        summary_rows
    )

    return (
        summary_table,
        raw_table,
    )


def median_iqr_width(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Median pointwise interquartile width in a selected Raman region."""

    widths = (
        statistics["p75"]
        - statistics["p25"]
    )

    return float(
        np.median(
            widths[mask]
        )
    )


def matched_sample_bootstrap(
    *,
    real_spectra: np.ndarray,
    generated_spectra: np.ndarray,
    axis: np.ndarray,
    non_peak_mask: np.ndarray,
    broad_sigma_cm1: float,
    repetitions: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare real and generated distributions using the same sample count.

    The real training set is kept fixed.
    Each repetition randomly draws the same number of generated spectra
    without replacement.

    This avoids directly comparing a 16-spectrum real P2.5-P97.5 envelope
    with a 100-spectrum generated envelope.
    """

    real_values = np.asarray(
        real_spectra,
        dtype=np.float64,
    )

    generated_values = np.asarray(
        generated_spectra,
        dtype=np.float64,
    )

    if real_values.ndim != 2:
        raise ValueError(
            "real_spectra必须是二维数组。"
        )

    if generated_values.ndim != 2:
        raise ValueError(
            "generated_spectra必须是二维数组。"
        )

    if (
        real_values.shape[1]
        != generated_values.shape[1]
    ):
        raise ValueError(
            "真实谱与生成谱长度不一致。"
        )

    matched_number = int(
        real_values.shape[0]
    )

    if generated_values.shape[0] < matched_number:
        raise ValueError(
            "生成光谱数量少于真实训练光谱数量，"
            "不能执行matched-size bootstrap。"
        )

    repetitions = int(
        repetitions
    )

    if repetitions <= 0:
        raise ValueError(
            "repetitions必须大于0。"
        )

    real_statistics = pointwise_statistics(
        real_values
    )

    (
        real_broad,
        real_local,
    ) = build_frequency_components(
        real_values,
        axis,
        broad_sigma_cm1,
    )

    real_broad_statistics = (
        pointwise_statistics(
            real_broad
        )
    )

    real_local_statistics = (
        pointwise_statistics(
            real_local
        )
    )

    real_metrics = {
        "non_peak_95_band_median":
            median_band_width(
                real_statistics,
                non_peak_mask,
            ),

        "non_peak_iqr_band_median":
            median_iqr_width(
                real_statistics,
                non_peak_mask,
            ),

        "non_peak_pointwise_std_mean":
            mean_pointwise_std(
                real_statistics,
                non_peak_mask,
            ),

        "broad_95_band_median":
            median_band_width(
                real_broad_statistics,
                non_peak_mask,
            ),

        "broad_iqr_band_median":
            median_iqr_width(
                real_broad_statistics,
                non_peak_mask,
            ),

        "broad_pointwise_std_mean":
            mean_pointwise_std(
                real_broad_statistics,
                non_peak_mask,
            ),

        "local_95_band_median":
            median_band_width(
                real_local_statistics,
                non_peak_mask,
            ),

        "local_iqr_band_median":
            median_iqr_width(
                real_local_statistics,
                non_peak_mask,
            ),

        "local_pointwise_std_mean":
            mean_pointwise_std(
                real_local_statistics,
                non_peak_mask,
            ),

        "per_spectrum_local_noise_95_span_median":
            local_noise_span_per_spectrum(
                real_values,
                axis,
                non_peak_mask,
                broad_sigma_cm1,
            ),
    }

    random_generator = np.random.default_rng(
        int(random_seed)
    )

    bootstrap_rows: list[dict[str, float | int]] = []

    for repetition in range(
        repetitions
    ):
        selected_indices = (
            random_generator.choice(
                generated_values.shape[0],
                size=matched_number,
                replace=False,
            )
        )

        subset = generated_values[
            selected_indices
        ]

        subset_statistics = (
            pointwise_statistics(
                subset
            )
        )

        (
            subset_broad,
            subset_local,
        ) = build_frequency_components(
            subset,
            axis,
            broad_sigma_cm1,
        )

        subset_broad_statistics = (
            pointwise_statistics(
                subset_broad
            )
        )

        subset_local_statistics = (
            pointwise_statistics(
                subset_local
            )
        )

        bootstrap_rows.append(
            {
                "repetition": repetition + 1,

                "non_peak_95_band_median":
                    median_band_width(
                        subset_statistics,
                        non_peak_mask,
                    ),

                "non_peak_iqr_band_median":
                    median_iqr_width(
                        subset_statistics,
                        non_peak_mask,
                    ),

                "non_peak_pointwise_std_mean":
                    mean_pointwise_std(
                        subset_statistics,
                        non_peak_mask,
                    ),

                "broad_95_band_median":
                    median_band_width(
                        subset_broad_statistics,
                        non_peak_mask,
                    ),

                "broad_iqr_band_median":
                    median_iqr_width(
                        subset_broad_statistics,
                        non_peak_mask,
                    ),

                "broad_pointwise_std_mean":
                    mean_pointwise_std(
                        subset_broad_statistics,
                        non_peak_mask,
                    ),

                "local_95_band_median":
                    median_band_width(
                        subset_local_statistics,
                        non_peak_mask,
                    ),

                "local_iqr_band_median":
                    median_iqr_width(
                        subset_local_statistics,
                        non_peak_mask,
                    ),

                "local_pointwise_std_mean":
                    mean_pointwise_std(
                        subset_local_statistics,
                        non_peak_mask,
                    ),

                "per_spectrum_local_noise_95_span_median":
                    local_noise_span_per_spectrum(
                        subset,
                        axis,
                        non_peak_mask,
                        broad_sigma_cm1,
                    ),
            }
        )

    raw_table = pd.DataFrame(
        bootstrap_rows
    )

    summary_rows = []

    for metric_name, real_value in (
        real_metrics.items()
    ):
        generated_distribution = (
            raw_table[
                metric_name
            ].to_numpy(
                dtype=np.float64
            )
        )

        generated_median = float(
            np.median(
                generated_distribution
            )
        )

        generated_p05 = float(
            np.percentile(
                generated_distribution,
                5.0,
            )
        )

        generated_p95 = float(
            np.percentile(
                generated_distribution,
                95.0,
            )
        )

        if abs(
            real_value
        ) > 1.0e-12:
            ratio = (
                generated_median
                / real_value
            )
        else:
            ratio = float(
                "nan"
            )

        summary_rows.append(
            {
                "metric": metric_name,
                "real_training_n16": real_value,
                "generated_n16_bootstrap_median":
                    generated_median,
                "generated_n16_bootstrap_p05":
                    generated_p05,
                "generated_n16_bootstrap_p95":
                    generated_p95,
                "generated_to_real_ratio":
                    ratio,
            }
        )

    summary_table = pd.DataFrame(
        summary_rows
    )

    return (
        summary_table,
        raw_table,
    )


def pairwise_pearson_mean(
    spectra: np.ndarray,
) -> float:
    if spectra.shape[0] < 2:
        return float("nan")

    centered = (
        spectra
        - np.mean(
            spectra,
            axis=1,
            keepdims=True,
        )
    )

    norms = np.linalg.norm(
        centered,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1.0e-12,
    )

    normalized = (
        centered / norms
    )

    correlation = (
        normalized
        @ normalized.T
    )

    upper = np.triu_indices(
        spectra.shape[0],
        k=1,
    )

    return float(
        np.mean(
            correlation[
                upper
            ]
        )
    )


def build_frequency_components(
    spectra: np.ndarray,
    axis: np.ndarray,
    broad_sigma_cm1: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    broad = gaussian_filter1d(
        spectra,
        sigma=sigma_samples(
            broad_sigma_cm1,
            axis,
        ),
        axis=1,
        mode="nearest",
    )

    local = (
        spectra
        - broad
    )

    return broad, local


def local_noise_span_per_spectrum(
    spectra: np.ndarray,
    axis: np.ndarray,
    non_peak_mask: np.ndarray,
    broad_sigma_cm1: float,
) -> float:
    broad = gaussian_filter1d(
        spectra,
        sigma=sigma_samples(
            broad_sigma_cm1,
            axis,
        ),
        axis=1,
        mode="nearest",
    )

    local = (
        spectra
        - broad
    )

    widths = (
        np.percentile(
            local[
                :,
                non_peak_mask
            ],
            97.5,
            axis=1,
        )
        - np.percentile(
            local[
                :,
                non_peak_mask
            ],
            2.5,
            axis=1,
        )
    )

    return float(
        np.median(
            widths
        )
    )


def analyze_broad_outliers(
    *,
    real_broad: np.ndarray,
    generated_broad: np.ndarray,
    non_peak_mask: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Compare per-spectrum broad/background deviations.

    The reference is the pointwise median broad component fitted only
    from the real training spectra.

    total_rms:
        total broad deviation from the real-training center.

    offset_mean:
        signed overall vertical background shift.

    shape_rms:
        broad-shape deviation after removing the spectrum-wide offset.

    abs_p95:
        95th percentile of absolute broad deviation.
    """

    real_values = np.asarray(
        real_broad,
        dtype=np.float64,
    )

    generated_values = np.asarray(
        generated_broad,
        dtype=np.float64,
    )

    mask = np.asarray(
        non_peak_mask,
        dtype=bool,
    ).reshape(-1)

    if real_values.ndim != 2:
        raise ValueError(
            "real_broad必须是二维数组。"
        )

    if generated_values.ndim != 2:
        raise ValueError(
            "generated_broad必须是二维数组。"
        )

    if (
        real_values.shape[1]
        != generated_values.shape[1]
    ):
        raise ValueError(
            "真实与生成broad长度不一致。"
        )

    if mask.size != real_values.shape[1]:
        raise ValueError(
            "non_peak_mask长度与光谱长度不一致。"
        )

    if int(np.count_nonzero(mask)) < 10:
        raise ValueError(
            "非峰区点数不足。"
        )

    # 只使用真实训练集拟合中心背景。
    reference_broad = np.median(
        real_values,
        axis=0,
    )

    def build_table(
        spectra: np.ndarray,
        prefix: str,
    ) -> pd.DataFrame:

        deviation = (
            spectra[:, mask]
            - reference_broad[
                np.newaxis,
                mask
            ]
        )

        offset_mean = np.mean(
            deviation,
            axis=1,
        )

        shape_deviation = (
            deviation
            - offset_mean[:, np.newaxis]
        )

        total_rms = np.sqrt(
            np.mean(
                deviation ** 2,
                axis=1,
            )
        )

        shape_rms = np.sqrt(
            np.mean(
                shape_deviation ** 2,
                axis=1,
            )
        )

        abs_p95 = np.percentile(
            np.abs(deviation),
            95.0,
            axis=1,
        )

        signed_p02_5 = np.percentile(
            deviation,
            2.5,
            axis=1,
        )

        signed_p97_5 = np.percentile(
            deviation,
            97.5,
            axis=1,
        )

        broad_span = (
            signed_p97_5
            - signed_p02_5
        )

        return pd.DataFrame(
            {
                "spectrum_index": np.arange(
                    1,
                    spectra.shape[0] + 1,
                    dtype=np.int64,
                ),
                "spectrum_name": [
                    f"{prefix}_{index:04d}"
                    for index in range(
                        1,
                        spectra.shape[0] + 1,
                    )
                ],
                "broad_total_rms":
                    total_rms,
                "broad_offset_mean":
                    offset_mean,
                "broad_abs_offset":
                    np.abs(offset_mean),
                "broad_shape_rms":
                    shape_rms,
                "broad_abs_p95":
                    abs_p95,
                "broad_signed_p02_5":
                    signed_p02_5,
                "broad_signed_p97_5":
                    signed_p97_5,
                "broad_within_spectrum_span":
                    broad_span,
            }
        )

    real_table = build_table(
        real_values,
        "real",
    )

    generated_table = build_table(
        generated_values,
        "generated",
    )

    metric_names = [
        "broad_total_rms",
        "broad_abs_offset",
        "broad_shape_rms",
        "broad_abs_p95",
        "broad_within_spectrum_span",
    ]

    summary_rows = []

    for metric in metric_names:
        real_metric = real_table[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        generated_metric = generated_table[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        real_median = float(
            np.median(
                real_metric
            )
        )

        real_p95 = float(
            np.percentile(
                real_metric,
                95.0,
            )
        )

        real_maximum = float(
            np.max(
                real_metric
            )
        )

        generated_median = float(
            np.median(
                generated_metric
            )
        )

        generated_p95 = float(
            np.percentile(
                generated_metric,
                95.0,
            )
        )

        above_p95 = int(
            np.count_nonzero(
                generated_metric
                > real_p95
            )
        )

        above_maximum = int(
            np.count_nonzero(
                generated_metric
                > real_maximum
            )
        )

        summary_rows.append(
            {
                "metric": metric,
                "real_median":
                    real_median,
                "real_p95":
                    real_p95,
                "real_maximum":
                    real_maximum,
                "generated_median":
                    generated_median,
                "generated_p95":
                    generated_p95,
                "generated_median_to_real_median":
                    (
                        generated_median
                        / real_median
                        if abs(real_median) > 1.0e-12
                        else float("nan")
                    ),
                "generated_p95_to_real_p95":
                    (
                        generated_p95
                        / real_p95
                        if abs(real_p95) > 1.0e-12
                        else float("nan")
                    ),
                "generated_count_above_real_p95":
                    above_p95,
                "generated_percent_above_real_p95":
                    (
                        100.0
                        * above_p95
                        / generated_metric.size
                    ),
                "generated_count_above_real_max":
                    above_maximum,
                "generated_percent_above_real_max":
                    (
                        100.0
                        * above_maximum
                        / generated_metric.size
                    ),
            }
        )

        generated_table[
            f"{metric}_above_real_p95"
        ] = (
            generated_metric
            > real_p95
        )

        generated_table[
            f"{metric}_above_real_max"
        ] = (
            generated_metric
            > real_maximum
        )

        generated_table[
            f"{metric}_ratio_to_real_p95"
        ] = (
            generated_metric
            / max(
                real_p95,
                1.0e-12,
            )
        )

    generated_table[
        "broad_tail_outlier_count"
    ] = np.zeros(
        generated_table.shape[0],
        dtype=np.int64,
    )

    for metric in metric_names:
        generated_table[
            "broad_tail_outlier_count"
        ] += (
            generated_table[
                f"{metric}_above_real_p95"
            ]
            .astype(
                np.int64
            )
        )

    generated_table = (
        generated_table.sort_values(
            [
                "broad_tail_outlier_count",
                "broad_total_rms",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary_table = pd.DataFrame(
        summary_rows
    )

    return (
        real_table,
        generated_table,
        summary_table,
    )


def mean_spectrum_similarity(
    real_spectra: np.ndarray,
    generated_spectra: np.ndarray,
) -> tuple[float, float]:
    real_mean = np.mean(
        real_spectra,
        axis=0,
    )

    generated_mean = np.mean(
        generated_spectra,
        axis=0,
    )

    correlation = float(
        np.corrcoef(
            real_mean,
            generated_mean,
        )[0, 1]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    real_mean
                    - generated_mean
                )
                ** 2
            )
        )
    )

    return correlation, rmse


def compute_peak_statistics(
    *,
    spectra: np.ndarray,
    axis: np.ndarray,
    reference_peak_indices: np.ndarray,
    baseline_sigma_cm1: float,
    search_half_width_cm1: float,
) -> pd.DataFrame:
    """
    For every automatically detected reference peak:
    - find the local maximum after broad-baseline removal;
    - record local peak position;
    - record baseline-corrected peak height;
    - estimate FWHM.
    """

    corrected = (
        spectra
        - gaussian_filter1d(
            spectra,
            sigma=sigma_samples(
                baseline_sigma_cm1,
                axis,
            ),
            axis=1,
            mode="nearest",
        )
    )

    spacing = float(
        np.median(
            np.diff(axis)
        )
    )

    rows = []

    for peak_number, reference_index in enumerate(
        reference_peak_indices,
        start=1,
    ):
        center = float(
            axis[
                int(reference_index)
            ]
        )

        search_mask = (
            np.abs(
                axis - center
            )
            <= float(
                search_half_width_cm1
            )
        )

        candidate_indices = np.flatnonzero(
            search_mask
        )

        if candidate_indices.size < 3:
            continue

        positions = []
        heights = []
        widths = []

        for spectrum in corrected:
            local_values = spectrum[
                candidate_indices
            ]

            local_relative_index = int(
                np.argmax(
                    local_values
                )
            )

            peak_index = int(
                candidate_indices[
                    local_relative_index
                ]
            )

            peak_height = float(
                spectrum[
                    peak_index
                ]
            )

            positions.append(
                float(
                    axis[
                        peak_index
                    ]
                )
            )

            heights.append(
                peak_height
            )

            width_cm1 = float(
                "nan"
            )

            if (
                0 < peak_index
                < spectrum.size - 1
                and peak_height > 0.0
                and spectrum[peak_index]
                >= spectrum[peak_index - 1]
                and spectrum[peak_index]
                >= spectrum[peak_index + 1]
            ):
                try:
                    width_samples = (
                        peak_widths(
                            spectrum,
                            [peak_index],
                            rel_height=0.5,
                        )[0][0]
                    )

                    width_cm1 = float(
                        width_samples
                        * spacing
                    )
                except Exception:
                    width_cm1 = float(
                        "nan"
                    )

            widths.append(
                width_cm1
            )

        positions_array = np.asarray(
            positions,
            dtype=np.float64,
        )

        heights_array = np.asarray(
            heights,
            dtype=np.float64,
        )

        widths_array = np.asarray(
            widths,
            dtype=np.float64,
        )

        finite_widths = widths_array[
            np.isfinite(
                widths_array
            )
        ]

        height_mean = float(
            np.mean(
                heights_array
            )
        )

        height_std = float(
            np.std(
                heights_array,
                ddof=0,
            )
        )

        if abs(height_mean) > 1.0e-12:
            height_cv = (
                100.0
                * height_std
                / abs(
                    height_mean
                )
            )
        else:
            height_cv = float(
                "nan"
            )

        rows.append(
            {
                "peak_number": peak_number,
                "reference_position_cm1": center,
                "position_mean_cm1": float(
                    np.mean(
                        positions_array
                    )
                ),
                "position_std_cm1": float(
                    np.std(
                        positions_array,
                        ddof=0,
                    )
                ),
                "position_min_cm1": float(
                    np.min(
                        positions_array
                    )
                ),
                "position_max_cm1": float(
                    np.max(
                        positions_array
                    )
                ),
                "height_mean": height_mean,
                "height_std": height_std,
                "height_cv_percent": height_cv,
                "fwhm_mean_cm1": (
                    float(
                        np.mean(
                            finite_widths
                        )
                    )
                    if finite_widths.size
                    else float("nan")
                ),
                "fwhm_std_cm1": (
                    float(
                        np.std(
                            finite_widths,
                            ddof=0,
                        )
                    )
                    if finite_widths.size
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def add_metric(
    rows: list[dict],
    *,
    name: str,
    real_value: float,
    generated_value: float,
) -> None:
    if (
        np.isfinite(real_value)
        and abs(real_value) > 1.0e-12
    ):
        ratio = (
            generated_value
            / real_value
        )
    else:
        ratio = float(
            "nan"
        )

    rows.append(
        {
            "metric": name,
            "real_training": real_value,
            "generated": generated_value,
            "generated_to_real_ratio": ratio,
        }
    )


def main() -> None:
    args = parse_arguments()

    project_root = Path(
        __file__
    ).resolve().parents[1]

    checkpoint_path = Path(
        args.checkpoint
    ).expanduser()

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            project_root
            / checkpoint_path
        )

    checkpoint_path = checkpoint_path.resolve()

    generated_path = Path(
        args.generated
    ).expanduser()

    if not generated_path.is_absolute():
        generated_path = (
            project_root
            / generated_path
        )

    generated_path = (
        generated_path.resolve()
    )

    output_directory = Path(
        args.output_directory
    ).expanduser()

    if not output_directory.is_absolute():
        output_directory = (
            project_root
            / output_directory
        )

    output_directory = (
        output_directory.resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_configuration = (
        load_configuration(
            args.config
        )
    )

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    checkpoint_configuration = checkpoint.get(
        "configuration"
    )

    metadata = checkpoint.get(
        "metadata"
    )

    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少configuration。"
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少metadata。"
        )

    data_config = checkpoint_configuration[
        "data"
    ]

    project_config = (
        checkpoint_configuration.get(
            "project",
            {},
        )
    )

    random_config = (
        checkpoint_configuration.get(
            "random",
            {},
        )
    )

    random_seed = int(
        random_config.get(
            "seed",
            project_config.get(
                "random_seed",
                42,
            ),
        )
    )

    input_directory = Path(
        data_config[
            "input_directory"
        ]
    ).expanduser()

    if not input_directory.is_absolute():
        input_directory = (
            project_root
            / input_directory
        )

    input_directory = (
        input_directory.resolve()
    )

    collection = (
        read_spectrum_collection(
            input_directory=input_directory,
            data_config=data_config,
        )
    )

    dataset_split = (
        split_spectrum_collection(
            collection=collection,
            data_config=data_config,
            random_seed=random_seed,
        )
    )

    training_indices = np.asarray(
        dataset_split.train.indices,
        dtype=np.int64,
    )

    length_adapter = (
        SpectrumLengthAdapter.from_metadata(
            metadata
        )
    )

    training_spectra_original = [
        collection.spectra[
            int(index)
        ]
        for index in training_indices
    ]

    training_axes_original = [
        collection.raman_shifts[
            int(index)
        ]
        for index in training_indices
    ]

    training_model_axis = (
        length_adapter.interpolate_to_model_axis(
            training_spectra_original,
            training_axes_original,
        )
    )

    (
        generated_axis,
        generated_spectra,
    ) = read_generated_file(
        generated_path
    )

    training_spectra = (
        length_adapter.interpolate_from_model_axis(
            training_model_axis,
            generated_axis,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    if (
        training_spectra.shape[1]
        != generated_spectra.shape[1]
    ):
        raise RuntimeError(
            "真实训练谱与生成谱最终轴长度不一致。"
        )

    generation_config = (
        current_configuration.get(
            "generation",
            {},
        )
    )

    sampling_calibration = (
        generation_config.get(
            "sampling_calibration",
            {},
        )
        or {}
    )

    non_peak_config = (
        sampling_calibration.get(
            "non_peak_noise",
            {},
        )
        or {}
    )

    reference_smoothing_sigma_cm1 = float(
        non_peak_config.get(
            "reference_smoothing_sigma_cm1",
            30.0,
        )
    )

    minimum_relative_prominence = float(
        non_peak_config.get(
            "minimum_relative_prominence",
            0.06,
        )
    )

    minimum_peak_distance_cm1 = float(
        non_peak_config.get(
            "minimum_peak_distance_cm1",
            12.0,
        )
    )

    maximum_peak_count = int(
        non_peak_config.get(
            "maximum_peak_count",
            20,
        )
    )

    peak_protection_half_width_cm1 = float(
        non_peak_config.get(
            "peak_protection_half_width_cm1",
            24.0,
        )
    )

    broad_sigma_cm1 = float(
        non_peak_config.get(
            "broad_sigma_cm1",
            7.0,
        )
    )

    (
        peak_indices,
        peak_mask,
        reference_spectrum,
    ) = detect_peak_regions(
        axis=generated_axis,
        training_spectra=training_spectra,
        reference_smoothing_sigma_cm1=(
            reference_smoothing_sigma_cm1
        ),
        minimum_relative_prominence=(
            minimum_relative_prominence
        ),
        minimum_peak_distance_cm1=(
            minimum_peak_distance_cm1
        ),
        maximum_peak_count=(
            maximum_peak_count
        ),
        peak_protection_half_width_cm1=(
            peak_protection_half_width_cm1
        ),
    )

    non_peak_mask = ~peak_mask

    if int(
        np.count_nonzero(
            non_peak_mask
        )
    ) < 10:
        raise RuntimeError(
            "自动峰区保护后非峰区点数不足。"
        )

    real_stats = pointwise_statistics(
        training_spectra
    )

    generated_stats = (
        pointwise_statistics(
            generated_spectra
        )
    )

    (
        real_broad,
        real_local,
    ) = build_frequency_components(
        training_spectra,
        generated_axis,
        broad_sigma_cm1,
    )

    (
        generated_broad,
        generated_local,
    ) = build_frequency_components(
        generated_spectra,
        generated_axis,
        broad_sigma_cm1,
    )

    real_broad_stats = (
        pointwise_statistics(
            real_broad
        )
    )

    generated_broad_stats = (
        pointwise_statistics(
            generated_broad
        )
    )

    real_local_stats = (
        pointwise_statistics(
            real_local
        )
    )

    generated_local_stats = (
        pointwise_statistics(
            generated_local
        )
    )

    (
        real_broad_outlier_table,
        generated_broad_outlier_table,
        broad_outlier_summary,
    ) = analyze_broad_outliers(
        real_broad=real_broad,
        generated_broad=generated_broad,
        non_peak_mask=non_peak_mask,
    )

    rows: list[dict] = []

    add_metric(
        rows,
        name="non_peak_95_band_median",
        real_value=median_band_width(
            real_stats,
            non_peak_mask,
        ),
        generated_value=median_band_width(
            generated_stats,
            non_peak_mask,
        ),
    )

    add_metric(
        rows,
        name="non_peak_pointwise_std_mean",
        real_value=mean_pointwise_std(
            real_stats,
            non_peak_mask,
        ),
        generated_value=mean_pointwise_std(
            generated_stats,
            non_peak_mask,
        ),
    )

    add_metric(
        rows,
        name="peak_95_band_median",
        real_value=median_band_width(
            real_stats,
            peak_mask,
        ),
        generated_value=median_band_width(
            generated_stats,
            peak_mask,
        ),
    )

    add_metric(
        rows,
        name="peak_pointwise_std_mean",
        real_value=mean_pointwise_std(
            real_stats,
            peak_mask,
        ),
        generated_value=mean_pointwise_std(
            generated_stats,
            peak_mask,
        ),
    )

    add_metric(
        rows,
        name="non_peak_broad_95_band_median",
        real_value=median_band_width(
            real_broad_stats,
            non_peak_mask,
        ),
        generated_value=median_band_width(
            generated_broad_stats,
            non_peak_mask,
        ),
    )

    add_metric(
        rows,
        name="non_peak_broad_pointwise_std_mean",
        real_value=mean_pointwise_std(
            real_broad_stats,
            non_peak_mask,
        ),
        generated_value=mean_pointwise_std(
            generated_broad_stats,
            non_peak_mask,
        ),
    )

    add_metric(
        rows,
        name="non_peak_local_95_band_median",
        real_value=median_band_width(
            real_local_stats,
            non_peak_mask,
        ),
        generated_value=median_band_width(
            generated_local_stats,
            non_peak_mask,
        ),
    )

    add_metric(
        rows,
        name="non_peak_local_pointwise_std_mean",
        real_value=mean_pointwise_std(
            real_local_stats,
            non_peak_mask,
        ),
        generated_value=mean_pointwise_std(
            generated_local_stats,
            non_peak_mask,
        ),
    )

    add_metric(
        rows,
        name="per_spectrum_local_noise_95_span_median",
        real_value=local_noise_span_per_spectrum(
            training_spectra,
            generated_axis,
            non_peak_mask,
            broad_sigma_cm1,
        ),
        generated_value=local_noise_span_per_spectrum(
            generated_spectra,
            generated_axis,
            non_peak_mask,
            broad_sigma_cm1,
        ),
    )

    add_metric(
        rows,
        name="pairwise_pearson_mean",
        real_value=pairwise_pearson_mean(
            training_spectra
        ),
        generated_value=pairwise_pearson_mean(
            generated_spectra
        ),
    )

    real_minima = np.min(
        training_spectra,
        axis=1,
    )

    generated_minima = np.min(
        generated_spectra,
        axis=1,
    )

    add_metric(
        rows,
        name="per_spectrum_minimum_median",
        real_value=float(
            np.median(
                real_minima
            )
        ),
        generated_value=float(
            np.median(
                generated_minima
            )
        ),
    )

    add_metric(
        rows,
        name="per_spectrum_minimum_p05",
        real_value=float(
            np.percentile(
                real_minima,
                5.0,
            )
        ),
        generated_value=float(
            np.percentile(
                generated_minima,
                5.0,
            )
        ),
    )

    mean_correlation, mean_rmse = (
        mean_spectrum_similarity(
            training_spectra,
            generated_spectra,
        )
    )

    summary_table = pd.DataFrame(
        rows
    )

    real_broad_std = (
        mean_pointwise_std(
            real_broad_stats,
            non_peak_mask,
        )
    )

    generated_broad_std = (
        mean_pointwise_std(
            generated_broad_stats,
            non_peak_mask,
        )
    )

    real_broad_band = (
        median_band_width(
            real_broad_stats,
            non_peak_mask,
        )
    )

    generated_broad_band = (
        median_band_width(
            generated_broad_stats,
            non_peak_mask,
        )
    )

    scale_candidates = []

    if generated_broad_std > 1.0e-12:
        scale_candidates.append(
            real_broad_std
            / generated_broad_std
        )

    if generated_broad_band > 1.0e-12:
        scale_candidates.append(
            real_broad_band
            / generated_broad_band
        )

    if scale_candidates:
        suggested_broad_scale = float(
            np.median(
                scale_candidates
            )
        )

        suggested_broad_scale = float(
            np.clip(
                suggested_broad_scale,
                0.05,
                1.0,
            )
        )
    else:
        suggested_broad_scale = float(
            "nan"
        )

    (
        matched_bootstrap_summary,
        matched_bootstrap_raw,
    ) = matched_sample_bootstrap(
        real_spectra=training_spectra,
        generated_spectra=generated_spectra,
        axis=generated_axis,
        non_peak_mask=non_peak_mask,
        broad_sigma_cm1=broad_sigma_cm1,
        repetitions=500,
        random_seed=random_seed + 7919,
    )

    (
        matched_bootstrap_summary,
        matched_bootstrap_raw,
    ) = matched_sample_bootstrap(
        real_spectra=training_spectra,
        generated_spectra=generated_spectra,
        axis=generated_axis,
        non_peak_mask=non_peak_mask,
        broad_sigma_cm1=broad_sigma_cm1,
        repetitions=500,
        random_seed=random_seed + 7919,
    )

    raman_jitter_config = (
        sampling_calibration.get(
            "raman_shift_jitter",
            {},
        )
        or {}
    )

    maximum_shift = float(
        raman_jitter_config.get(
            "maximum_absolute_shift_cm1",
            5.0,
        )
    )

    peak_search_half_width = max(
        8.0,
        maximum_shift + 3.0,
    )

    real_peak_table = (
        compute_peak_statistics(
            spectra=training_spectra,
            axis=generated_axis,
            reference_peak_indices=peak_indices,
            baseline_sigma_cm1=(
                reference_smoothing_sigma_cm1
            ),
            search_half_width_cm1=(
                peak_search_half_width
            ),
        )
    )

    generated_peak_table = (
        compute_peak_statistics(
            spectra=generated_spectra,
            axis=generated_axis,
            reference_peak_indices=peak_indices,
            baseline_sigma_cm1=(
                reference_smoothing_sigma_cm1
            ),
            search_half_width_cm1=(
                peak_search_half_width
            ),
        )
    )

    peak_comparison = (
        real_peak_table.merge(
            generated_peak_table,
            on=[
                "peak_number",
                "reference_position_cm1",
            ],
            suffixes=(
                "_real",
                "_generated",
            ),
        )
    )

    pointwise_table = pd.DataFrame(
        {
            "raman_shift_cm1": generated_axis,
            "peak_region": peak_mask.astype(
                np.int8
            ),
            "reference_mean": reference_spectrum,

            "real_mean": real_stats["mean"],
            "real_std": real_stats["std"],
            "real_p2_5": real_stats["p2_5"],
            "real_p97_5": real_stats["p97_5"],

            "generated_mean": generated_stats["mean"],
            "generated_std": generated_stats["std"],
            "generated_p2_5": generated_stats["p2_5"],
            "generated_p97_5": generated_stats["p97_5"],

            "real_broad_std": real_broad_stats["std"],
            "generated_broad_std": generated_broad_stats["std"],

            "real_local_std": real_local_stats["std"],
            "generated_local_std": generated_local_stats["std"],
        }
    )

    metadata_table = pd.DataFrame(
        {
            "item": [
                "checkpoint",
                "generated_file",
                "input_directory",
                "random_seed",
                "training_spectrum_count",
                "generated_spectrum_count",
                "detected_peak_count",
                "detected_peak_positions_cm1",
                "mean_spectrum_pearson",
                "mean_spectrum_rmse",
                "suggested_broad_variation_scale",
            ],
            "value": [
                str(checkpoint_path),
                str(generated_path),
                str(input_directory),
                str(random_seed),
                str(training_spectra.shape[0]),
                str(generated_spectra.shape[0]),
                str(peak_indices.size),
                ", ".join(
                    f"{generated_axis[index]:.1f}"
                    for index in peak_indices
                ),
                f"{mean_correlation:.8f}",
                f"{mean_rmse:.8f}",
                f"{suggested_broad_scale:.6f}",
            ],
        }
    )

    excel_path = (
        output_directory
        / "real_vs_generated_distribution.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        summary_table.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        pointwise_table.to_excel(
            writer,
            sheet_name="pointwise",
            index=False,
        )

        peak_comparison.to_excel(
            writer,
            sheet_name="peak_comparison",
            index=False,
        )

        metadata_table.to_excel(
            writer,
            sheet_name="metadata",
            index=False,
        )

        matched_bootstrap_summary.to_excel(
            writer,
            sheet_name="matched_n16_summary",
            index=False,
        )

        matched_bootstrap_raw.to_excel(
            writer,
            sheet_name="matched_n16_bootstrap",
            index=False,
        )

        matched_bootstrap_summary.to_excel(
            writer,
            sheet_name="matched_n16_summary",
            index=False,
        )

        matched_bootstrap_raw.to_excel(
            writer,
            sheet_name="matched_n16_bootstrap",
            index=False,
        )

    broad_outlier_path = (
        output_directory
        / "broad_outlier_analysis.xlsx"
    )

    with pd.ExcelWriter(
        broad_outlier_path,
        engine="openpyxl",
    ) as writer:
        broad_outlier_summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        real_broad_outlier_table.to_excel(
            writer,
            sheet_name="real_training_scores",
            index=False,
        )

        generated_broad_outlier_table.to_excel(
            writer,
            sheet_name="generated_scores",
            index=False,
        )

    # ------------------------------------------------------------
    # Figure 1: mean + 95% envelope
    # ------------------------------------------------------------
    plt.figure(
        figsize=(14, 6)
    )

    plt.fill_between(
        generated_axis,
        real_stats["p2_5"],
        real_stats["p97_5"],
        alpha=0.25,
        label="Real training 95% envelope",
    )

    plt.plot(
        generated_axis,
        real_stats["mean"],
        linewidth=1.4,
        label="Real training mean",
    )

    plt.fill_between(
        generated_axis,
        generated_stats["p2_5"],
        generated_stats["p97_5"],
        alpha=0.20,
        label="Generated 95% envelope",
    )

    plt.plot(
        generated_axis,
        generated_stats["mean"],
        linewidth=1.2,
        label="Generated mean",
    )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )
    plt.ylabel(
        "Intensity"
    )
    plt.title(
        "Real training vs generated SERS distribution"
    )
    plt.legend()
    plt.tight_layout()

    envelope_figure = (
        output_directory
        / "real_vs_generated_envelope.png"
    )

    plt.savefig(
        envelope_figure,
        dpi=200,
    )
    plt.close()

    # ------------------------------------------------------------
    # Figure 2: pointwise standard deviation
    # ------------------------------------------------------------
    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        generated_axis,
        real_stats["std"],
        linewidth=1.3,
        label="Real training pointwise STD",
    )

    plt.plot(
        generated_axis,
        generated_stats["std"],
        linewidth=1.2,
        label="Generated pointwise STD",
    )

    for peak_index in peak_indices:
        plt.axvline(
            generated_axis[
                int(peak_index)
            ],
            alpha=0.12,
            linewidth=0.8,
        )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )
    plt.ylabel(
        "Pointwise STD"
    )
    plt.title(
        "Pointwise inter-spectrum variation"
    )
    plt.legend()
    plt.tight_layout()

    std_figure = (
        output_directory
        / "pointwise_std_comparison.png"
    )

    plt.savefig(
        std_figure,
        dpi=200,
    )
    plt.close()

    # ------------------------------------------------------------
    # Figure 3: broad-component distribution width
    # ------------------------------------------------------------
    real_broad_width = (
        real_broad_stats["p97_5"]
        - real_broad_stats["p2_5"]
    )

    generated_broad_width = (
        generated_broad_stats["p97_5"]
        - generated_broad_stats["p2_5"]
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        generated_axis,
        real_broad_width,
        linewidth=1.3,
        label="Real broad 95% width",
    )

    plt.plot(
        generated_axis,
        generated_broad_width,
        linewidth=1.2,
        label="Generated broad 95% width",
    )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )
    plt.ylabel(
        "95% inter-spectrum width"
    )
    plt.title(
        "Broad/background inter-spectrum distribution width"
    )
    plt.legend()
    plt.tight_layout()

    broad_figure = (
        output_directory
        / "broad_distribution_width.png"
    )

    plt.savefig(
        broad_figure,
        dpi=200,
    )
    plt.close()

    print(
        "\n===== 真实训练谱 vs 当前生成谱 ====="
    )

    print(
        f"checkpoint：{checkpoint_path}"
    )

    print(
        f"训练输入：{input_directory}"
    )

    print(
        f"训练集光谱数：{training_spectra.shape[0]}"
    )

    print(
        f"生成光谱数：{generated_spectra.shape[0]}"
    )

    print(
        "自动检测峰位："
        + ", ".join(
            f"{generated_axis[index]:.1f}"
            for index in peak_indices
        )
        + " cm⁻¹"
    )

    print(
        "\n===== 核心分布指标 ====="
    )

    print(
        summary_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        "\n===== 平均谱 ====="
    )

    print(
        f"真实mean vs 生成mean Pearson："
        f"{mean_correlation:.6f}"
    )

    print(
        f"真实mean vs 生成mean RMSE："
        f"{mean_rmse:.6f}"
    )

    print(
        "\n===== Broad背景校准建议 ====="
    )

    print(
        f"真实非峰 broad STD："
        f"{real_broad_std:.6g}"
    )

    print(
        f"生成非峰 broad STD："
        f"{generated_broad_std:.6g}"
    )

    print(
        f"真实非峰 broad 95%带宽："
        f"{real_broad_band:.6g}"
    )

    print(
        f"生成非峰 broad 95%带宽："
        f"{generated_broad_band:.6g}"
    )

    print(
        "根据真实训练集估算的"
        " broad_variation_scale："
        f"{suggested_broad_scale:.6f}"
    )

    print(
        "\n注意：该scale只是诊断估计值，"
        "本脚本不会自动修改任何生成参数。"
    )

    print(
        "\n===== 公平样本量比较：真实16条 vs 生成随机16条×500 ====="
    )

    print(
        matched_bootstrap_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        "\n这里的95% band比直接比较真实16条和生成100条"
        "更适合判断分布宽度。"
    )

    print(
        "\n===== 公平样本量比较：真实16条 vs 生成随机16条×500 ====="
    )

    print(
        matched_bootstrap_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        "\n这里的95% band比直接比较真实16条和生成100条"
        "更适合判断分布宽度。"
    )

    print(
        "\n===== 输出 ====="
    )

    print(
        f"Excel：{excel_path}"
    )

    print(
        f"95%分布图：{envelope_figure}"
    )

    print(
        f"pointwise STD图：{std_figure}"
    )

    print(
        f"broad宽度图：{broad_figure}"
    )

    print(
        "\n===== Broad尾部异常分析 ====="
    )

    print(
        broad_outlier_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        "\n===== Broad偏离最大的前10条生成光谱 ====="
    )

    display_columns = [
        "spectrum_index",
        "spectrum_name",
        "broad_tail_outlier_count",
        "broad_total_rms",
        "broad_offset_mean",
        "broad_shape_rms",
        "broad_abs_p95",
        "broad_within_spectrum_span",
    ]

    print(
        generated_broad_outlier_table[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        f"Broad异常分析Excel："
        f"{broad_outlier_path}"
    )


if __name__ == "__main__":
    main()

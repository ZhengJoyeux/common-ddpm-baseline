"""Diagnose PCA reference-prior distribution in D2.5.

This script answers one question:

Does the PCA prior sampler itself already create an abnormal broad/background
distribution before DDPM residuals and post-generation calibration are added?

No model parameters, checkpoints or generated spectra are modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import load_configuration
from src.dataset_splitter import split_spectrum_collection
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.prior_residual import PriorResidualTransformer
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter

from scripts.diagnose_real_vs_generated_sers_distribution import (
    detect_peak_regions,
    sigma_samples,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诊断D2.5 PCA prior本身的分布。"
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="采样PCA prior数量。",
    )

    parser.add_argument(
        "--output-directory",
        required=True,
    )

    return parser.parse_args()


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


def median_width(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    lower: str,
    upper: str,
) -> float:
    width = (
        statistics[upper]
        - statistics[lower]
    )

    return float(
        np.median(
            width[mask]
        )
    )


def mean_std(
    statistics: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    return float(
        np.mean(
            statistics["std"][mask]
        )
    )


def broad_components(
    spectra: np.ndarray,
    axis: np.ndarray,
    sigma_cm1: float,
) -> np.ndarray:
    return gaussian_filter1d(
        spectra,
        sigma=sigma_samples(
            sigma_cm1,
            axis,
        ),
        axis=1,
        mode="nearest",
    )


def calculate_metrics(
    spectra: np.ndarray,
    broad: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    stats = pointwise_statistics(
        spectra
    )

    broad_stats = pointwise_statistics(
        broad
    )

    return {
        "non_peak_95_band":
            median_width(
                stats,
                mask,
                lower="p2_5",
                upper="p97_5",
            ),

        "non_peak_iqr":
            median_width(
                stats,
                mask,
                lower="p25",
                upper="p75",
            ),

        "non_peak_std":
            mean_std(
                stats,
                mask,
            ),

        "broad_95_band":
            median_width(
                broad_stats,
                mask,
                lower="p2_5",
                upper="p97_5",
            ),

        "broad_iqr":
            median_width(
                broad_stats,
                mask,
                lower="p25",
                upper="p75",
            ),

        "broad_std":
            mean_std(
                broad_stats,
                mask,
            ),
    }


def matched_bootstrap(
    *,
    real_spectra: np.ndarray,
    prior_spectra: np.ndarray,
    real_broad: np.ndarray,
    prior_broad: np.ndarray,
    mask: np.ndarray,
    repetitions: int,
    random_seed: int,
) -> pd.DataFrame:

    real_metrics = calculate_metrics(
        real_spectra,
        real_broad,
        mask,
    )

    generator = np.random.default_rng(
        random_seed
    )

    rows = []

    n_real = real_spectra.shape[0]

    for repetition in range(
        repetitions
    ):
        indices = generator.choice(
            prior_spectra.shape[0],
            size=n_real,
            replace=False,
        )

        metrics = calculate_metrics(
            prior_spectra[indices],
            prior_broad[indices],
            mask,
        )

        metrics["repetition"] = (
            repetition + 1
        )

        rows.append(
            metrics
        )

    raw = pd.DataFrame(
        rows
    )

    summary_rows = []

    for metric, real_value in (
        real_metrics.items()
    ):
        values = raw[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        prior_median = float(
            np.median(
                values
            )
        )

        summary_rows.append(
            {
                "metric": metric,
                "real_training_n16":
                    real_value,
                "prior_n16_bootstrap_median":
                    prior_median,
                "prior_n16_bootstrap_p05":
                    float(
                        np.percentile(
                            values,
                            5.0,
                        )
                    ),
                "prior_n16_bootstrap_p95":
                    float(
                        np.percentile(
                            values,
                            95.0,
                        )
                    ),
                "prior_to_real_ratio":
                    (
                        prior_median
                        / real_value
                        if abs(
                            real_value
                        ) > 1.0e-12
                        else float("nan")
                    ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def main() -> None:
    args = parse_arguments()

    if args.number < 100:
        raise ValueError(
            "--number建议至少100。"
        )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    checkpoint_path = Path(
        args.checkpoint
    ).expanduser()

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            project_root
            / checkpoint_path
        )

    checkpoint_path = (
        checkpoint_path.resolve()
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

    configuration = (
        load_configuration(
            args.config
        )
    )

    checkpoint = (
        load_checkpoint_file(
            checkpoint_path,
            map_location="cpu",
        )
    )

    metadata = checkpoint.get(
        "metadata"
    )

    checkpoint_configuration = (
        checkpoint.get(
            "configuration"
        )
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少metadata。"
        )

    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少configuration。"
        )

    prior_state = metadata.get(
        "prior_residual_state"
    )

    normalization_state = (
        metadata.get(
            "normalization_state"
        )
    )

    if not isinstance(
        prior_state,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少prior_residual_state。"
        )

    if not isinstance(
        normalization_state,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少normalization_state。"
        )

    transformer = (
        PriorResidualTransformer
        .from_state_dict(
            prior_state
        )
    )

    if (
        transformer.prior_method
        != "pca_reconstruction"
    ):
        raise RuntimeError(
            "当前checkpoint不是PCA prior。"
        )

    normalizer = (
        GlobalMinMaxNormalizer
        .from_state_dict(
            normalization_state
        )
    )

    data_config = (
        checkpoint_configuration[
            "data"
        ]
    )

    random_config = (
        checkpoint_configuration.get(
            "random",
            {},
        )
    )

    project_config = (
        checkpoint_configuration.get(
            "project",
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

    split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=random_seed,
    )

    training_indices = np.asarray(
        split.train.indices,
        dtype=np.int64,
    )

    adapter = (
        SpectrumLengthAdapter
        .from_metadata(
            metadata
        )
    )

    real_model_axis = (
        adapter
        .interpolate_to_model_axis(
            [
                collection.spectra[
                    int(index)
                ]
                for index
                in training_indices
            ],
            [
                collection.raman_shifts[
                    int(index)
                ]
                for index
                in training_indices
            ],
        )
    )

    normalized_training = (
        normalizer.transform(
            real_model_axis
        )
    )

    rng = np.random.default_rng(
        random_seed + 100003
    )

    sampled_prior_normalized = (
        transformer
        .sample_reference_priors(
            int(args.number),
            random_generator=rng,
        )
    )

    sampled_prior_normalized = (
        np.asarray(
            sampled_prior_normalized,
            dtype=np.float32,
        )
    )

    if (
        sampled_prior_normalized.shape[1]
        != real_model_axis.shape[1]
    ):
        raise RuntimeError(
            "PCA prior长度与训练轴不一致。"
        )

    sampled_prior = (
        normalizer.inverse_transform(
            sampled_prior_normalized
        )
    ).astype(
        np.float64,
        copy=False,
    )

    real_spectra = np.asarray(
        real_model_axis,
        dtype=np.float64,
    )

    axis = adapter.model_axis

    # ------------------------------------------------------------
    # Current automatic peak definition
    # ------------------------------------------------------------
    sampling_configuration = (
        configuration
        .get(
            "generation",
            {},
        )
        .get(
            "sampling_calibration",
            {},
        )
        or {}
    )

    non_peak_config = (
        sampling_configuration.get(
            "non_peak_noise",
            {},
        )
        or {}
    )

    reference_sigma = float(
        non_peak_config.get(
            "reference_smoothing_sigma_cm1",
            30.0,
        )
    )

    broad_sigma = float(
        non_peak_config.get(
            "broad_sigma_cm1",
            7.0,
        )
    )

    (
        peak_indices,
        peak_mask,
        _,
    ) = detect_peak_regions(
        axis=axis,
        training_spectra=real_spectra,
        reference_smoothing_sigma_cm1=(
            reference_sigma
        ),
        minimum_relative_prominence=float(
            non_peak_config.get(
                "minimum_relative_prominence",
                0.06,
            )
        ),
        minimum_peak_distance_cm1=float(
            non_peak_config.get(
                "minimum_peak_distance_cm1",
                12.0,
            )
        ),
        maximum_peak_count=int(
            non_peak_config.get(
                "maximum_peak_count",
                20,
            )
        ),
        peak_protection_half_width_cm1=float(
            non_peak_config.get(
                "peak_protection_half_width_cm1",
                24.0,
            )
        ),
    )

    non_peak_mask = (
        ~peak_mask
    )

    real_broad = broad_components(
        real_spectra,
        axis,
        broad_sigma,
    )

    prior_broad = broad_components(
        sampled_prior,
        axis,
        broad_sigma,
    )

    bootstrap_summary = (
        matched_bootstrap(
            real_spectra=real_spectra,
            prior_spectra=sampled_prior,
            real_broad=real_broad,
            prior_broad=prior_broad,
            mask=non_peak_mask,
            repetitions=500,
            random_seed=random_seed + 17011,
        )
    )

    # ------------------------------------------------------------
    # PCA score diagnostics
    # ------------------------------------------------------------
    if (
        transformer.pca_mean is None
        or transformer.pca_components is None
        or transformer.pca_score_standard_deviation
        is None
    ):
        raise RuntimeError(
            "checkpoint PCA状态不完整。"
        )

    pca_mean = np.asarray(
        transformer.pca_mean,
        dtype=np.float64,
    )

    components = np.asarray(
        transformer.pca_components,
        dtype=np.float64,
    )

    score_std = np.asarray(
        transformer.pca_score_standard_deviation,
        dtype=np.float64,
    )

    if np.any(
        score_std <= 1.0e-12
    ):
        raise RuntimeError(
            "checkpoint中PCA score标准差无效。"
        )

    training_scores = (
        (
            normalized_training.astype(
                np.float64
            )
            - pca_mean[
                np.newaxis,
                :
            ]
        )
        @ components.T
    )

    sampled_scores = (
        (
            sampled_prior_normalized.astype(
                np.float64
            )
            - pca_mean[
                np.newaxis,
                :
            ]
        )
        @ components.T
    )

    score_mean = np.asarray(
        transformer.pca_training_score_mean,
        dtype=np.float64,
    )

    training_z = (
        training_scores
        - score_mean[
            np.newaxis,
            :
        ]
    ) / score_std[
        np.newaxis,
        :
    ]

    sampled_z = (
        sampled_scores
        - score_mean[
            np.newaxis,
            :
        ]
    ) / score_std[
        np.newaxis,
        :
    ]

    training_radius = np.sqrt(
        np.sum(
            training_z ** 2,
            axis=1,
        )
    )

    sampled_radius = np.sqrt(
        np.sum(
            sampled_z ** 2,
            axis=1,
        )
    )

    training_corner_count = np.sum(
        np.abs(
            training_z
        ) > 1.0,
        axis=1,
    )

    sampled_corner_count = np.sum(
        np.abs(
            sampled_z
        ) > 1.0,
        axis=1,
    )

    score_summary = pd.DataFrame(
        {
            "metric": [
                "score_radius_median",
                "score_radius_p95",
                "score_radius_max",
                "fraction_with_2plus_components_abs_z_gt_1",
            ],
            "real_training": [
                float(
                    np.median(
                        training_radius
                    )
                ),
                float(
                    np.percentile(
                        training_radius,
                        95.0,
                    )
                ),
                float(
                    np.max(
                        training_radius
                    )
                ),
                float(
                    np.mean(
                        training_corner_count
                        >= 2
                    )
                ),
            ],
            "sampled_prior": [
                float(
                    np.median(
                        sampled_radius
                    )
                ),
                float(
                    np.percentile(
                        sampled_radius,
                        95.0,
                    )
                ),
                float(
                    np.max(
                        sampled_radius
                    )
                ),
                float(
                    np.mean(
                        sampled_corner_count
                        >= 2
                    )
                ),
            ],
        }
    )

    score_summary[
        "sampled_to_real_ratio"
    ] = (
        score_summary[
            "sampled_prior"
        ]
        / score_summary[
            "real_training"
        ].replace(
            0.0,
            np.nan,
        )
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    excel_path = (
        output_directory
        / "pca_prior_distribution.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        bootstrap_summary.to_excel(
            writer,
            sheet_name="broad_summary",
            index=False,
        )

        score_summary.to_excel(
            writer,
            sheet_name="score_summary",
            index=False,
        )

        pd.DataFrame(
            training_z,
            columns=[
                f"PC{index + 1}_z"
                for index
                in range(
                    training_z.shape[1]
                )
            ],
        ).to_excel(
            writer,
            sheet_name="training_scores",
            index=False,
        )

        pd.DataFrame(
            sampled_z,
            columns=[
                f"PC{index + 1}_z"
                for index
                in range(
                    sampled_z.shape[1]
                )
            ],
        ).to_excel(
            writer,
            sheet_name="sampled_scores",
            index=False,
        )

    envelope_path = (
        output_directory
        / "real_vs_pca_prior_envelope.png"
    )

    real_stats = pointwise_statistics(
        real_spectra
    )

    prior_stats = pointwise_statistics(
        sampled_prior
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.fill_between(
        axis,
        real_stats["p2_5"],
        real_stats["p97_5"],
        alpha=0.25,
        label="Real training 95%",
    )

    plt.plot(
        axis,
        real_stats["mean"],
        linewidth=1.3,
        label="Real mean",
    )

    plt.fill_between(
        axis,
        prior_stats["p2_5"],
        prior_stats["p97_5"],
        alpha=0.20,
        label="PCA prior 95%",
    )

    plt.plot(
        axis,
        prior_stats["mean"],
        linewidth=1.2,
        label="PCA prior mean",
    )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )

    plt.ylabel(
        "Intensity"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        envelope_path,
        dpi=200,
    )

    plt.close()

    scatter_path = None

    if training_z.shape[1] >= 2:
        scatter_path = (
            output_directory
            / "pca_score_pc1_pc2.png"
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.scatter(
            sampled_z[:, 0],
            sampled_z[:, 1],
            s=10,
            alpha=0.25,
            label="Sampled prior",
        )

        plt.scatter(
            training_z[:, 0],
            training_z[:, 1],
            s=45,
            label="Real training",
        )

        plt.xlabel(
            "PC1 standardized score"
        )

        plt.ylabel(
            "PC2 standardized score"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            scatter_path,
            dpi=200,
        )

        plt.close()

    print(
        "\n===== PCA prior分布诊断 ====="
    )

    print(
        f"训练谱数量："
        f"{real_spectra.shape[0]}"
    )

    print(
        f"PCA prior采样数量："
        f"{sampled_prior.shape[0]}"
    )

    print(
        f"PCA主成分数："
        f"{components.shape[0]}"
    )

    print(
        "PCA采样策略："
        f"{transformer.pca_sampling_strategy}"
    )

    print(
        "PCA score截断：±"
        f"{transformer.pca_score_clip_standard_deviations:.3f} SD"
    )

    print(
        "\n===== 真实16条 vs PCA prior随机16条×500 ====="
    )

    print(
        bootstrap_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        "\n===== PCA score联合分布 ====="
    )

    print(
        score_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6g}"
            ),
        )
    )

    print(
        "\n===== 输出 ====="
    )

    print(
        f"Excel：{excel_path}"
    )

    print(
        f"Envelope：{envelope_path}"
    )

    if scatter_path is not None:
        print(
            f"PC1-PC2：{scatter_path}"
        )


if __name__ == "__main__":
    main()

"""Experiment D: final quality evaluation for generated SERS spectra.

This script is evaluation-only:
- it does not train or resume a model;
- it does not generate new spectra;
- it does not calibrate, filter, jitter or modify generated spectra;
- it uses checkpoint training/validation/test indices exactly as stored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.checkpoint_manager import (
    load_checkpoint_file,
    resolve_label_axis,
)
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.sers_generation_evaluator import (
    broad_local_distribution_evaluation,
    build_nonpeak_mask,
    diversity_summary,
    mean_spectrum_comparison,
    measure_peak_table,
    nearest_metric_summary,
    nearest_reference_metrics,
    pca_distribution_evaluation,
    peak_distribution_comparison,
    replication_screening_summary,
    row_pearson,
    training_nearest_other_metrics,
)
from src.sers_training_guard import (
    SersTrainingGuard,
    fit_sers_training_guard_state,
)
from src.spectrum_file_reader import (
    read_spectrum_collection,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment D：对真实训练谱、held-out真实谱和"
            "当前生成SERS光谱执行统一最终评价。"
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="当前项目YAML。",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="待评价生成模型对应checkpoint。",
    )
    parser.add_argument(
        "--generated",
        required=True,
        help="待评价生成xlsx或csv，第一列必须为Raman shift。",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Experiment D输出目录。",
    )
    return parser.parse_args()


def _read_generated(
    path: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            "generated只支持xlsx/xls/csv。"
        )

    if frame.shape[1] < 2:
        raise ValueError(
            "生成文件至少需要1列Raman shift和1列光谱。"
        )

    axis = pd.to_numeric(
        frame.iloc[:, 0],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(axis).all():
        raise ValueError(
            "生成文件第一列Raman shift包含非数值。"
        )

    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(
            "生成文件Raman shift必须严格递增。"
        )

    intensity_frame = frame.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce",
    )
    spectra = intensity_frame.to_numpy(
        dtype=np.float64
    ).T

    if not np.isfinite(spectra).all():
        raise ValueError(
            "生成光谱包含NaN、无穷值或非数值。"
        )

    names = [
        str(value)
        for value in frame.columns[1:]
    ]

    return frame, axis, spectra, names


def _normalize_path_text(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "/")
        .lstrip("./")
    )


def _validate_indices(
    *,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    total: int,
) -> None:
    groups = {
        "training": training_indices,
        "validation": validation_indices,
        "test": test_indices,
    }

    for name, values in groups.items():
        if np.any(values < 0) or np.any(
            values >= total
        ):
            raise RuntimeError(
                f"checkpoint {name}_indices越界。"
            )

    sets = {
        name: set(values.tolist())
        for name, values in groups.items()
    }

    if (
        sets["training"] & sets["validation"]
        or sets["training"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        raise RuntimeError(
            "checkpoint train/validation/test索引发生重叠。"
        )


def _load_real_spectra(
    *,
    configuration: dict[str, Any],
    checkpoint: dict[str, Any],
    output_axis: np.ndarray,
) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    checkpoint_configuration = checkpoint.get(
        "configuration"
    )

    if not isinstance(metadata, dict):
        raise RuntimeError(
            "checkpoint缺少有效metadata。"
        )
    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少有效configuration。"
        )

    data_config = checkpoint_configuration.get(
        "data"
    )
    if not isinstance(data_config, dict):
        raise RuntimeError(
            "checkpoint缺少有效data配置。"
        )

    input_directory = resolve_project_path(
        configuration,
        data_config["input_directory"],
    )

    collection = read_spectrum_collection(
        input_directory=input_directory,
        data_config=data_config,
    )

    expected_count = int(
        metadata.get(
            "number_of_spectra",
            len(collection.spectrum_names),
        )
    )

    if len(collection.spectrum_names) != expected_count:
        raise RuntimeError(
            "当前真实数据数量与checkpoint不一致："
            f"{len(collection.spectrum_names)} != "
            f"{expected_count}。"
        )

    checkpoint_relative = metadata.get(
        "relative_source_files"
    )

    if isinstance(checkpoint_relative, list):
        expected_relative = [
            _normalize_path_text(value)
            for value in checkpoint_relative
        ]
        current_relative = [
            _normalize_path_text(value)
            for value in collection.relative_source_files
        ]

        if expected_relative != current_relative:
            raise RuntimeError(
                "当前真实输入文件顺序与checkpoint记录不一致；"
                "为避免错误使用split indices，停止评价。"
            )

    training_indices = np.asarray(
        metadata.get("training_indices", []),
        dtype=np.int64,
    ).reshape(-1)

    validation_indices = np.asarray(
        metadata.get("validation_indices", []),
        dtype=np.int64,
    ).reshape(-1)

    test_indices = np.asarray(
        metadata.get("test_indices", []),
        dtype=np.int64,
    ).reshape(-1)

    if training_indices.size == 0:
        raise RuntimeError(
            "checkpoint没有training_indices。"
        )

    _validate_indices(
        training_indices=training_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        total=expected_count,
    )

    tolerance = float(
        data_config.get(
            "raman_range_tolerance",
            1.0,
        )
    )

    spectra_on_output_axis: list[np.ndarray] = []

    for spectrum_index in range(expected_count):
        source_axis = np.asarray(
            collection.raman_shifts[
                spectrum_index
            ],
            dtype=np.float64,
        ).reshape(-1)

        source_spectrum = np.asarray(
            collection.spectra[
                spectrum_index
            ],
            dtype=np.float64,
        ).reshape(-1)

        if (
            source_axis.size
            != source_spectrum.size
        ):
            raise RuntimeError(
                "真实光谱与Raman shift长度不一致。"
            )

        if not np.all(
            np.diff(source_axis) > 0.0
        ):
            raise RuntimeError(
                "真实Raman shift轴不是严格递增。"
            )

        if (
            output_axis[0]
            < source_axis[0] - tolerance
            or output_axis[-1]
            > source_axis[-1] + tolerance
        ):
            raise RuntimeError(
                "生成输出轴超出真实光谱覆盖范围，"
                "停止盲目插值。"
            )

        spectra_on_output_axis.append(
            np.interp(
                output_axis,
                source_axis,
                source_spectrum,
            )
        )

    all_spectra = np.stack(
        spectra_on_output_axis,
        axis=0,
    )

    all_names = [
        str(value)
        for value in collection.spectrum_names
    ]
    all_sources = [
        str(value)
        for value in collection.relative_source_files
    ]

    heldout_indices = np.concatenate(
        [
            validation_indices,
            test_indices,
        ]
    ).astype(np.int64)

    def select(
        indices: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[str],
        list[str],
    ]:
        return (
            all_spectra[indices],
            [
                all_names[index]
                for index in indices
            ],
            [
                all_sources[index]
                for index in indices
            ],
        )

    (
        training_spectra,
        training_names,
        training_sources,
    ) = select(training_indices)

    if heldout_indices.size:
        (
            heldout_spectra,
            heldout_names,
            heldout_sources,
        ) = select(heldout_indices)
    else:
        heldout_spectra = None
        heldout_names = []
        heldout_sources = []

    split_records: list[dict[str, Any]] = []

    index_to_split: dict[int, str] = {}
    for index in training_indices:
        index_to_split[int(index)] = "training"
    for index in validation_indices:
        index_to_split[int(index)] = "validation"
    for index in test_indices:
        index_to_split[int(index)] = "test"

    for index in range(expected_count):
        split_records.append(
            {
                "global_index": int(index),
                "split": index_to_split.get(
                    index,
                    "unassigned",
                ),
                "spectrum_name": all_names[index],
                "relative_source_file": all_sources[index],
            }
        )

    return {
        "input_directory": input_directory,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "heldout_indices": heldout_indices,
        "training_spectra": training_spectra,
        "training_names": training_names,
        "training_sources": training_sources,
        "heldout_spectra": heldout_spectra,
        "heldout_names": heldout_names,
        "heldout_sources": heldout_sources,
        "split_frame": pd.DataFrame(
            split_records
        ),
    }


def _final_evaluation_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    generation = configuration.get(
        "generation",
        {},
    )

    if not isinstance(generation, dict):
        raise TypeError(
            "generation配置必须是字典。"
        )

    config = generation.get(
        "final_evaluation",
        {},
    ) or {}

    if not isinstance(config, dict):
        raise TypeError(
            "generation.final_evaluation必须是字典。"
        )

    return {
        "random_seed": int(
            config.get(
                "random_seed",
                2026,
            )
        ),
        "pca_components": int(
            config.get(
                "pca_components",
                6,
            )
        ),
        "bootstrap_repeats": int(
            config.get(
                "bootstrap_repeats",
                500,
            )
        ),
        "generated_pair_count": int(
            config.get(
                "generated_pair_count",
                50000,
            )
        ),
        "broad_sigma_cm1": float(
            config.get(
                "broad_sigma_cm1",
                7.0,
            )
        ),
    }


def _training_guard_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    generation = configuration.get(
        "generation",
        {},
    )
    guard = (
        generation.get(
            "training_guard",
            {},
        )
        or {}
    )

    if not isinstance(guard, dict):
        raise TypeError(
            "generation.training_guard必须是字典。"
        )

    # 即使用户临时把enabled改为false，
    # D仍可使用相同参数自动拟合峰；评价不做筛选删除。
    guard = dict(guard)
    guard["enabled"] = True

    return guard


def _compare_generated_axis_with_checkpoint(
    *,
    checkpoint: dict[str, Any],
    generated_axis: np.ndarray,
) -> tuple[str, str]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "checkpoint缺少metadata。"
        )

    resolved_label, checkpoint_axis, profile_id = (
        resolve_label_axis(
            metadata=metadata,
            label=None,
        )
    )

    checkpoint_axis = np.asarray(
        checkpoint_axis,
        dtype=np.float64,
    )

    if checkpoint_axis.shape != generated_axis.shape:
        raise RuntimeError(
            "生成文件Raman轴点数与checkpoint输出轴不一致："
            f"{generated_axis.size} != "
            f"{checkpoint_axis.size}。"
        )

    maximum_error = float(
        np.max(
            np.abs(
                checkpoint_axis
                - generated_axis
            )
        )
    )

    spacing = float(
        np.median(
            np.diff(checkpoint_axis)
        )
    )
    tolerance = max(
        spacing * 1.0e-6,
        1.0e-8,
    )

    if maximum_error > tolerance:
        raise RuntimeError(
            "生成文件Raman轴与checkpoint输出轴不一致，"
            f"最大差值={maximum_error:.8g} cm^-1。"
        )

    return resolved_label, profile_id


def _build_peak_evaluation(
    *,
    training_spectra: np.ndarray,
    training_names: list[str],
    heldout_spectra: np.ndarray | None,
    heldout_names: list[str],
    generated_spectra: np.ndarray,
    generated_names: list[str],
    axis: np.ndarray,
    guard_state: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    centers = [
        float(
            peak_state[
                "reference_position_cm1"
            ]
        )
        for peak_state in guard_state["peaks"]
    ]

    smoothing_sigma = float(
        guard_state[
            "reference_smoothing_sigma_cm1"
        ]
    )
    half_width = float(
        guard_state[
            "metric_half_width_cm1"
        ]
    )

    training_table = measure_peak_table(
        spectra=training_spectra,
        spectrum_names=training_names,
        dataset_name="training",
        raman_shift=axis,
        peak_centers_cm1=centers,
        smoothing_sigma_cm1=smoothing_sigma,
        half_width_cm1=half_width,
    )

    generated_table = measure_peak_table(
        spectra=generated_spectra,
        spectrum_names=generated_names,
        dataset_name="generated",
        raman_shift=axis,
        peak_centers_cm1=centers,
        smoothing_sigma_cm1=smoothing_sigma,
        half_width_cm1=half_width,
    )

    heldout_table: pd.DataFrame | None = None

    if (
        heldout_spectra is not None
        and heldout_spectra.shape[0] > 0
    ):
        heldout_table = measure_peak_table(
            spectra=heldout_spectra,
            spectrum_names=heldout_names,
            dataset_name="heldout",
            raman_shift=axis,
            peak_centers_cm1=centers,
            smoothing_sigma_cm1=smoothing_sigma,
            half_width_cm1=half_width,
        )

    comparison = peak_distribution_comparison(
        training_peak_table=training_table,
        generated_peak_table=generated_table,
        heldout_peak_table=heldout_table,
    )

    long_frames = [
        training_table,
        generated_table,
    ]

    if heldout_table is not None:
        long_frames.append(
            heldout_table
        )

    long_table = pd.concat(
        long_frames,
        ignore_index=True,
    )

    return (
        comparison,
        long_table,
        training_table,
        heldout_table,
    )


def _screen_guard(
    *,
    axis: np.ndarray,
    generated_spectra: np.ndarray,
    generated_names: list[str],
    guard_state: dict[str, Any],
    guard_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    guard = SersTrainingGuard(
        raman_shift=axis,
        state=guard_state,
        configuration=guard_config,
    )

    accepted, rows, peak_rows = (
        guard.evaluate(
            generated_spectra,
            spectrum_names=generated_names,
        )
    )

    acceptance_rate = float(
        100.0
        * np.mean(accepted)
    )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(peak_rows),
        acceptance_rate,
    )


def _plot_mean_envelope(
    *,
    axis: np.ndarray,
    training: np.ndarray,
    generated: np.ndarray,
    heldout: np.ndarray | None,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))

    train_mean = np.mean(training, axis=0)
    gen_mean = np.mean(generated, axis=0)

    train_low = np.percentile(
        training,
        2.5,
        axis=0,
    )
    train_high = np.percentile(
        training,
        97.5,
        axis=0,
    )
    gen_low = np.percentile(
        generated,
        2.5,
        axis=0,
    )
    gen_high = np.percentile(
        generated,
        97.5,
        axis=0,
    )

    plt.plot(
        axis,
        train_mean,
        label="Training mean",
        linewidth=1.3,
    )
    plt.fill_between(
        axis,
        train_low,
        train_high,
        alpha=0.18,
        label="Training 95% envelope",
    )
    plt.plot(
        axis,
        gen_mean,
        label="Generated mean",
        linewidth=1.2,
    )
    plt.fill_between(
        axis,
        gen_low,
        gen_high,
        alpha=0.12,
        label="Generated 95% envelope",
    )

    if (
        heldout is not None
        and heldout.shape[0] > 0
    ):
        plt.plot(
            axis,
            np.mean(
                heldout,
                axis=0,
            ),
            label="Held-out mean",
            linewidth=1.0,
        )

    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(
        "Experiment D: real and generated spectra"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=220,
    )
    plt.close()


def _plot_pca(
    *,
    pca_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    if (
        "PC1" not in pca_scores
        or "PC2" not in pca_scores
    ):
        return

    plt.figure(figsize=(7, 6))

    for dataset_name, marker in [
        ("training", "o"),
        ("heldout", "^"),
        ("generated", "."),
    ]:
        current = pca_scores[
            pca_scores["dataset"]
            == dataset_name
        ]
        if current.empty:
            continue

        plt.scatter(
            current["PC1"],
            current["PC2"],
            marker=marker,
            alpha=(
                0.35
                if dataset_name == "generated"
                else 0.9
            ),
            label=dataset_name,
        )

    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title(
        "Training-fitted PCA score distribution"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=220,
    )
    plt.close()


def _plot_nearest_distance(
    *,
    training_nearest: pd.DataFrame,
    generated_nearest: pd.DataFrame,
    heldout_nearest: pd.DataFrame | None,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.hist(
        training_nearest[
            "standardized_shape_rmse"
        ],
        bins=20,
        alpha=0.5,
        label="Training -> nearest training",
    )
    plt.hist(
        generated_nearest[
            "standardized_shape_rmse"
        ],
        bins=30,
        alpha=0.5,
        label="Generated -> nearest training",
    )

    if (
        heldout_nearest is not None
        and not heldout_nearest.empty
    ):
        plt.hist(
            heldout_nearest[
                "standardized_shape_rmse"
            ],
            bins=10,
            alpha=0.5,
            label="Held-out -> nearest training",
        )

    plt.xlabel(
        "Standardized shape RMSE"
    )
    plt.ylabel("Count")
    plt.title(
        "Nearest-training distance screening"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=220,
    )
    plt.close()


def _plot_peak_ratios(
    *,
    peak_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    x = peak_comparison[
        "reference_position_cm1"
    ].to_numpy(dtype=np.float64)

    plt.figure(figsize=(11, 5))

    fields = [
        (
            "height_std_ratio",
            "Peak height STD ratio",
        ),
        (
            "fwhm_cm1_std_ratio",
            "FWHM STD ratio",
        ),
        (
            "position_cm1_std_ratio",
            "Peak position STD ratio",
        ),
    ]

    for field, label in fields:
        if field not in peak_comparison:
            continue
        plt.plot(
            x,
            peak_comparison[field],
            marker="o",
            label=label,
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="Real = 1",
    )
    plt.xlabel("Reference Raman shift (cm$^{-1}$)")
    plt.ylabel("Generated / training STD")
    plt.title(
        "Peak-parameter diversity ratios"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=220,
    )
    plt.close()


def _write_excel(
    *,
    output_path: Path,
    frames: dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        for sheet_name, frame in frames.items():
            safe_name = str(
                sheet_name
            )[:31]
            frame.to_excel(
                writer,
                sheet_name=safe_name,
                index=False,
            )


def _write_csv_frames(
    *,
    output_directory: Path,
    frames: dict[str, pd.DataFrame],
) -> None:
    for name, frame in frames.items():
        frame.to_csv(
            output_directory
            / f"{name}.csv",
            index=False,
        )


def _scalar_from_frame(
    frame: pd.DataFrame,
    column: str,
    fallback: float = np.nan,
) -> float:
    if (
        column not in frame
        or frame.empty
    ):
        return float(fallback)
    return float(
        frame[column].iloc[0]
    )


def _build_final_summary(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    generated_path: Path,
    input_directory: Path,
    output_label: str,
    profile_id: str,
    axis: np.ndarray,
    training_count: int,
    validation_count: int,
    test_count: int,
    generated_count: int,
    guard_state: dict[str, Any],
    guard_acceptance_rate: float,
    mean_comparison: pd.DataFrame,
    nearest_summary: pd.DataFrame,
    diversity: pd.DataFrame,
    replication: pd.DataFrame,
    pca_summary: pd.DataFrame,
    pca_distribution: pd.DataFrame,
    broad_bootstrap: pd.DataFrame,
    peak_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    summary_rows: list[
        dict[str, Any]
    ] = []

    def add(
        section: str,
        metric: str,
        value: Any,
        note: str,
    ) -> None:
        summary_rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "note": note,
            }
        )

    add(
        "provenance",
        "checkpoint_step",
        int(checkpoint.get("step", -1)),
        "checkpoint实际step。",
    )
    add(
        "provenance",
        "training_count",
        training_count,
        "checkpoint记录的训练集光谱数。",
    )
    add(
        "provenance",
        "validation_count",
        validation_count,
        "仅作held-out参考，不参与拟合。",
    )
    add(
        "provenance",
        "test_count",
        test_count,
        "仅作held-out参考，不参与拟合或调参。",
    )
    add(
        "provenance",
        "generated_count",
        generated_count,
        "本次固定评价的生成光谱数。",
    )
    add(
        "provenance",
        "guard_peak_count",
        len(guard_state["peaks"]),
        "完全由当前训练集自动检测/筛选。",
    )
    add(
        "guard",
        "acceptance_rate_percent",
        guard_acceptance_rate,
        "Experiment C guard复核；D不删除任何光谱。",
    )

    mean_row = mean_comparison[
        mean_comparison["comparison"]
        == "training_mean_vs_generated_mean"
    ]

    if not mean_row.empty:
        for metric in (
            "pearson",
            "cosine",
            "spectral_angle_degree",
            "rmse",
            "mae",
            "first_derivative_pearson",
        ):
            add(
                "mean_spectrum",
                metric,
                float(
                    mean_row[
                        metric
                    ].iloc[0]
                ),
                "训练均值谱 vs 生成均值谱。",
            )

    gen_nearest = nearest_summary[
        nearest_summary["dataset"]
        == "generated_to_nearest_training"
    ]
    train_nearest = nearest_summary[
        nearest_summary["dataset"]
        == "training_to_nearest_other_training"
    ]

    for metric in (
        "standardized_shape_rmse",
        "pearson",
        "raw_rmse",
        "first_derivative_pearson",
    ):
        current_gen = gen_nearest[
            gen_nearest["metric"] == metric
        ]
        current_train = train_nearest[
            train_nearest["metric"] == metric
        ]

        if not current_gen.empty:
            add(
                "nearest_training",
                f"generated_{metric}_median",
                float(
                    current_gen[
                        "median"
                    ].iloc[0]
                ),
                "生成谱到最近训练谱。",
            )
        if not current_train.empty:
            add(
                "nearest_training",
                f"training_nearest_other_{metric}_median",
                float(
                    current_train[
                        "median"
                    ].iloc[0]
                ),
                "真实训练谱内部最近邻基线。",
            )

    for _, row in replication.iterrows():
        add(
            "replication_screen",
            str(row["metric"]),
            float(row["value"]),
            str(row["note"]),
        )

    diversity_shape = diversity[
        diversity["metric"]
        == "standardized_shape_rmse"
    ]
    if not diversity_shape.empty:
        add(
            "diversity",
            "generated_pairwise_shape_rmse_median_to_training",
            float(
                diversity_shape[
                    "generated_median_to_training_median"
                ].iloc[0]
            ),
            "1附近表示成对形状离散度接近训练集。",
        )

    add(
        "pca",
        "cumulative_explained_variance_ratio",
        float(
            pca_summary[
                "explained_variance_ratio"
            ].sum()
        ),
        "仅由真实训练集拟合PCA。",
    )

    for metric in (
        "mmd2_rbf_biased",
        "wasserstein_pc_mean",
        "wasserstein_pc_median",
    ):
        current = pca_distribution[
            pca_distribution["metric"]
            == metric
        ]
        if not current.empty:
            add(
                "pca_distribution",
                f"{metric}_bootstrap_median",
                float(
                    current[
                        "median"
                    ].iloc[0]
                ),
                "真实train N vs 随机等量generated，bootstrap。",
            )

    for domain in (
        "nonpeak",
        "broad",
        "local",
    ):
        for metric in (
            "band95_median",
            "pointwise_std_mean",
        ):
            current = broad_bootstrap[
                (broad_bootstrap["domain"] == domain)
                & (
                    broad_bootstrap["metric"]
                    == metric
                )
            ]
            if not current.empty:
                add(
                    "broad_local",
                    f"{domain}_{metric}_matched_ratio_median",
                    float(
                        current[
                            "ratio_median"
                        ].iloc[0]
                    ),
                    "真实train N vs 随机等量generated。",
                )

    peak_fields = {
        "position_cm1_median_difference": (
            "peak_position_median_difference_abs_median_cm1"
        ),
        "height_median_ratio": (
            "peak_height_median_ratio_median"
        ),
        "height_std_ratio": (
            "peak_height_std_ratio_median"
        ),
        "fwhm_cm1_median_ratio": (
            "peak_fwhm_median_ratio_median"
        ),
        "fwhm_cm1_std_ratio": (
            "peak_fwhm_std_ratio_median"
        ),
        "position_cm1_std_ratio": (
            "peak_position_std_ratio_median"
        ),
    }

    for field, output_name in peak_fields.items():
        if field not in peak_comparison:
            continue

        values = (
            peak_comparison[field]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .to_numpy(dtype=np.float64)
        )

        if not values.size:
            continue

        if field == "position_cm1_median_difference":
            values = np.abs(values)

        add(
            "peak_distribution",
            output_name,
            float(np.median(values)),
            "8个训练集自动主要峰的中位汇总。",
        )

    summary = pd.DataFrame(
        summary_rows
    )

    lines = [
        "===== Experiment D final SERS evaluation =====",
        f"checkpoint: {checkpoint_path}",
        f"checkpoint step: {int(checkpoint.get('step', -1))}",
        f"real input: {input_directory}",
        f"generated: {generated_path}",
        f"output label: {output_label}",
        f"axis profile: {profile_id}",
        (
            "Raman axis: "
            f"{axis[0]:g}-{axis[-1]:g} cm^-1, "
            f"{axis.size} points"
        ),
        (
            "real split counts: "
            f"train={training_count}, "
            f"validation={validation_count}, "
            f"test={test_count}"
        ),
        f"generated count: {generated_count}",
        f"automatic major peaks: {len(guard_state['peaks'])}",
        (
            "guard acceptance rate: "
            f"{guard_acceptance_rate:.2f}%"
        ),
        "",
        "Important:",
        "1. Experiment D never modifies generated spectra.",
        "2. PCA, peak regions and all reference statistics are fitted from training spectra only.",
        "3. Validation/test are reported only as held-out reference; they are not used to tune the model.",
        "4. No single metric automatically declares the model qualified.",
        "5. Final decision must jointly inspect peaks, distributions, diversity, nearest-training distance and downstream performance.",
    ]

    return summary, "\n".join(lines)


def main() -> None:
    args = parse_arguments()

    configuration = load_configuration(
        args.config
    )
    evaluation_config = (
        _final_evaluation_configuration(
            configuration
        )
    )
    guard_config = (
        _training_guard_configuration(
            configuration
        )
    )

    checkpoint_path = Path(
        args.checkpoint
    ).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(
            configuration,
            checkpoint_path,
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    generated_path = Path(
        args.generated
    ).expanduser()
    if not generated_path.is_absolute():
        generated_path = resolve_project_path(
            configuration,
            generated_path,
        )
    if not generated_path.is_file():
        raise FileNotFoundError(
            generated_path
        )

    output_directory = Path(
        args.output_directory
    ).expanduser()
    if not output_directory.is_absolute():
        output_directory = (
            resolve_project_path(
                configuration,
                output_directory,
            )
        )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    (
        _,
        generated_axis,
        generated_spectra,
        generated_names,
    ) = _read_generated(
        generated_path
    )

    output_label, profile_id = (
        _compare_generated_axis_with_checkpoint(
            checkpoint=checkpoint,
            generated_axis=generated_axis,
        )
    )

    real = _load_real_spectra(
        configuration=configuration,
        checkpoint=checkpoint,
        output_axis=generated_axis,
    )

    training_spectra = real[
        "training_spectra"
    ]
    training_names = real[
        "training_names"
    ]
    heldout_spectra = real[
        "heldout_spectra"
    ]
    heldout_names = real[
        "heldout_names"
    ]

    # ----------------------------------------------------------
    # Training-driven peak/guard state
    # ----------------------------------------------------------

    guard_state = (
        fit_sers_training_guard_state(
            training_spectra=training_spectra,
            raman_shift=generated_axis,
            configuration=guard_config,
        )
    )

    (
        guard_summary,
        guard_peak_metrics,
        guard_acceptance_rate,
    ) = _screen_guard(
        axis=generated_axis,
        generated_spectra=generated_spectra,
        generated_names=generated_names,
        guard_state=guard_state,
        guard_config=guard_config,
    )

    peak_centers = [
        float(
            item[
                "reference_position_cm1"
            ]
        )
        for item in guard_state[
            "peaks"
        ]
    ]

    # ----------------------------------------------------------
    # Peak distributions
    # ----------------------------------------------------------

    (
        peak_comparison,
        peak_metrics_long,
        _,
        _,
    ) = _build_peak_evaluation(
        training_spectra=training_spectra,
        training_names=training_names,
        heldout_spectra=heldout_spectra,
        heldout_names=heldout_names,
        generated_spectra=generated_spectra,
        generated_names=generated_names,
        axis=generated_axis,
        guard_state=guard_state,
    )

    # ----------------------------------------------------------
    # Mean and nearest-real spectral similarity
    # ----------------------------------------------------------

    mean_comparison = mean_spectrum_comparison(
        training_spectra=training_spectra,
        generated_spectra=generated_spectra,
        heldout_spectra=heldout_spectra,
    )

    training_nearest = (
        training_nearest_other_metrics(
            training_spectra,
            names=training_names,
        )
    )

    generated_nearest = (
        nearest_reference_metrics(
            query_spectra=generated_spectra,
            reference_spectra=training_spectra,
            query_names=generated_names,
            reference_names=training_names,
        )
    )

    heldout_nearest: pd.DataFrame | None = None

    if (
        heldout_spectra is not None
        and heldout_spectra.shape[0] > 0
    ):
        heldout_nearest = (
            nearest_reference_metrics(
                query_spectra=heldout_spectra,
                reference_spectra=training_spectra,
                query_names=heldout_names,
                reference_names=training_names,
            )
        )

    nearest_summary = (
        nearest_metric_summary(
            training_nearest=training_nearest,
            generated_nearest=generated_nearest,
            heldout_nearest=heldout_nearest,
        )
    )

    replication_summary = (
        replication_screening_summary(
            training_nearest=training_nearest,
            generated_nearest=generated_nearest,
        )
    )

    # ----------------------------------------------------------
    # Pairwise diversity
    # ----------------------------------------------------------

    (
        diversity,
        training_pairs,
        generated_pairs,
    ) = diversity_summary(
        training_spectra=training_spectra,
        generated_spectra=generated_spectra,
        generated_pair_count=(
            evaluation_config[
                "generated_pair_count"
            ]
        ),
        random_seed=(
            evaluation_config[
                "random_seed"
            ]
        ),
    )

    # ----------------------------------------------------------
    # PCA / MMD / Wasserstein
    # ----------------------------------------------------------

    (
        pca_model,
        pca_comparison,
        pca_scores,
        pca_distribution,
        pca_reconstruction,
    ) = pca_distribution_evaluation(
        training_spectra=training_spectra,
        generated_spectra=generated_spectra,
        heldout_spectra=heldout_spectra,
        pca_components=(
            evaluation_config[
                "pca_components"
            ]
        ),
        bootstrap_repeats=(
            evaluation_config[
                "bootstrap_repeats"
            ]
        ),
        random_seed=(
            evaluation_config[
                "random_seed"
            ]
        ),
    )

    # ----------------------------------------------------------
    # Broad/local non-peak distributions
    # ----------------------------------------------------------

    nonpeak_mask = build_nonpeak_mask(
        raman_shift=generated_axis,
        peak_centers_cm1=peak_centers,
        half_width_cm1=float(
            guard_state[
                "metric_half_width_cm1"
            ]
        ),
    )

    (
        broad_local_all,
        broad_local_bootstrap,
    ) = broad_local_distribution_evaluation(
        training_spectra=training_spectra,
        generated_spectra=generated_spectra,
        raman_shift=generated_axis,
        nonpeak_mask=nonpeak_mask,
        broad_sigma_cm1=(
            evaluation_config[
                "broad_sigma_cm1"
            ]
        ),
        bootstrap_repeats=(
            evaluation_config[
                "bootstrap_repeats"
            ]
        ),
        random_seed=(
            evaluation_config[
                "random_seed"
            ]
        ),
    )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    final_summary, text_summary = (
        _build_final_summary(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            generated_path=generated_path,
            input_directory=real[
                "input_directory"
            ],
            output_label=output_label,
            profile_id=profile_id,
            axis=generated_axis,
            training_count=int(
                training_spectra.shape[0]
            ),
            validation_count=int(
                real[
                    "validation_indices"
                ].size
            ),
            test_count=int(
                real[
                    "test_indices"
                ].size
            ),
            generated_count=int(
                generated_spectra.shape[0]
            ),
            guard_state=guard_state,
            guard_acceptance_rate=(
                guard_acceptance_rate
            ),
            mean_comparison=mean_comparison,
            nearest_summary=nearest_summary,
            diversity=diversity,
            replication=replication_summary,
            pca_summary=pca_comparison,
            pca_distribution=pca_distribution,
            broad_bootstrap=(
                broad_local_bootstrap
            ),
            peak_comparison=peak_comparison,
        )
    )

    # ----------------------------------------------------------
    # Save tabular outputs
    # ----------------------------------------------------------

    summary_frames = {
        "final_summary": final_summary,
        "peak_comparison": peak_comparison,
        "mean_comparison": mean_comparison,
        "nearest_summary": nearest_summary,
        "replication_summary": replication_summary,
        "diversity_summary": diversity,
        "pca_comparison": pca_comparison,
        "pca_distribution": pca_distribution,
        "pca_reconstruction": pca_reconstruction,
        "broad_local_all": broad_local_all,
        "broad_local_bootstrap": broad_local_bootstrap,
        "guard_summary": guard_summary,
        "split_records": real[
            "split_frame"
        ],
    }

    _write_excel(
        output_path=(
            output_directory
            / "experiment_d_final_evaluation.xlsx"
        ),
        frames=summary_frames,
    )

    _write_csv_frames(
        output_directory=output_directory,
        frames=summary_frames,
    )

    # Larger detailed tables are CSV only.
    detailed_frames = {
        "peak_metrics_long": peak_metrics_long,
        "guard_peak_metrics_long": guard_peak_metrics,
        "generated_nearest_training": generated_nearest,
        "training_nearest_other": training_nearest,
        "pca_scores": pca_scores,
        "training_pair_metrics": training_pairs,
        "generated_pair_metrics": generated_pairs,
    }

    if heldout_nearest is not None:
        detailed_frames[
            "heldout_nearest_training"
        ] = heldout_nearest

    _write_csv_frames(
        output_directory=output_directory,
        frames=detailed_frames,
    )

    # Peak bounds used by D.
    pd.DataFrame(
        guard_state["peaks"]
    ).to_csv(
        output_directory
        / "automatic_peak_reference.csv",
        index=False,
    )

    metadata = {
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_step": int(
            checkpoint.get(
                "step",
                -1,
            )
        ),
        "generated": str(
            generated_path
        ),
        "output_label": output_label,
        "profile_id": profile_id,
        "raman_start_cm1": float(
            generated_axis[0]
        ),
        "raman_end_cm1": float(
            generated_axis[-1]
        ),
        "raman_points": int(
            generated_axis.size
        ),
        "training_indices": real[
            "training_indices"
        ].tolist(),
        "validation_indices": real[
            "validation_indices"
        ].tolist(),
        "test_indices": real[
            "test_indices"
        ].tolist(),
        "evaluation_configuration": (
            evaluation_config
        ),
        "automatic_peak_centers_cm1": (
            peak_centers
        ),
        "pca_components_used": int(
            pca_model.components.shape[0]
        ),
        "pca_cumulative_explained_variance_ratio": float(
            np.sum(
                pca_model.explained_variance_ratio
            )
        ),
    }

    (
        output_directory
        / "evaluation_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_directory
        / "experiment_d_summary.txt"
    ).write_text(
        text_summary + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------

    _plot_mean_envelope(
        axis=generated_axis,
        training=training_spectra,
        generated=generated_spectra,
        heldout=heldout_spectra,
        output_path=(
            output_directory
            / "mean_and_95_envelope.png"
        ),
    )

    _plot_pca(
        pca_scores=pca_scores,
        output_path=(
            output_directory
            / "pca_pc1_pc2.png"
        ),
    )

    _plot_nearest_distance(
        training_nearest=training_nearest,
        generated_nearest=generated_nearest,
        heldout_nearest=heldout_nearest,
        output_path=(
            output_directory
            / "nearest_training_distance.png"
        ),
    )

    _plot_peak_ratios(
        peak_comparison=peak_comparison,
        output_path=(
            output_directory
            / "peak_parameter_std_ratios.png"
        ),
    )

    # ----------------------------------------------------------
    # Terminal summary
    # ----------------------------------------------------------

    print()
    print(text_summary)

    print()
    print("===== Key numerical summary =====")

    display_rows = final_summary[
        final_summary["section"].isin(
            [
                "mean_spectrum",
                "diversity",
                "pca_distribution",
                "broad_local",
                "peak_distribution",
                "replication_screen",
            ]
        )
    ]

    print(
        display_rows[
            [
                "section",
                "metric",
                "value",
            ]
        ].to_string(index=False)
    )

    print()
    print("===== Output files =====")

    key_files = [
        "experiment_d_final_evaluation.xlsx",
        "final_summary.csv",
        "peak_comparison.csv",
        "nearest_summary.csv",
        "replication_summary.csv",
        "diversity_summary.csv",
        "pca_comparison.csv",
        "pca_distribution.csv",
        "pca_reconstruction.csv",
        "broad_local_bootstrap.csv",
        "generated_nearest_training.csv",
        "peak_metrics_long.csv",
        "evaluation_metadata.json",
        "experiment_d_summary.txt",
        "mean_and_95_envelope.png",
        "pca_pc1_pc2.png",
        "nearest_training_distance.png",
        "peak_parameter_std_ratios.png",
    ]

    for file_name in key_files:
        print(
            "  -",
            output_directory
            / file_name,
        )

    print()
    print(
        "Experiment D只评价，不修改任何生成光谱。"
    )
    print(
        "请根据本次输出联合判断模型质量；"
        "不要仅根据单个Pearson、MMD或Guard结果宣布模型有效。"
    )


if __name__ == "__main__":
    main()

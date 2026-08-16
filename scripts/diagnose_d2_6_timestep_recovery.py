"""D2.6 broad-local residual 的多时间步一步 x0 去噪诊断。

诊断目标
--------
D2.6 的 DDPM 不再学习 D2.5 的完整 residual，而只学习：

    scaled_local_residual

因此本脚本严格在 D2.6 的真实模型数据域中进行前向加噪和一步 x0 恢复：

    normalized spectrum
        -> outer PCA reference prior
        -> raw residual
        -> true broad residual + true local residual
        -> robust-asinh scaled local residual
        -> q_sample(x_t)
        -> DDPM one-step x0 recovery
        -> inverse local transform
        -> true broad residual + recovered local residual
        -> outer PCA reference prior + residual
        -> full normalized spectrum
        -> original intensity spectrum

重要说明
--------
1. broad residual 使用当前真实光谱对应的真实 broad residual，不随机采样 broad PCA。
   这样可以隔离评价 DDPM 对 local residual 的去噪能力。
2. outer PCA reference prior 使用当前真实光谱对应的 reference prior，不随机采样。
3. 不修改模型、checkpoint、配置文件或生成光谱。
4. 默认同时诊断全部真实光谱，并按 train/validation/test 标记；也可用
   --spectrum-index 只诊断一条。
5. 输出 clipped 和 unclipped 两套 x0 结果。D2.6 模型域训练目标通常位于 [-1, 1]，
   但诊断时保留 unclipped 结果，用于判断裁剪是否掩盖模型误差。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.broad_local_residual import BroadLocalResidualDecomposer
from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.prior_residual import PriorResidualTransformer
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


def calculate_rmse(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_64 = np.asarray(reference, dtype=np.float64)
    prediction_64 = np.asarray(prediction, dtype=np.float64)
    error = prediction_64 - reference_64
    return float(np.sqrt(np.mean(error**2)))


def calculate_mae(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_64 = np.asarray(reference, dtype=np.float64)
    prediction_64 = np.asarray(prediction, dtype=np.float64)
    return float(np.mean(np.abs(prediction_64 - reference_64)))


def calculate_pearson(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_64 = np.asarray(reference, dtype=np.float64)
    prediction_64 = np.asarray(prediction, dtype=np.float64)

    if not (
        np.isfinite(reference_64).all()
        and np.isfinite(prediction_64).all()
    ):
        return float("nan")

    if (
        float(np.std(reference_64)) < 1.0e-12
        or float(np.std(prediction_64)) < 1.0e-12
    ):
        return float("nan")

    return float(np.corrcoef(reference_64, prediction_64)[0, 1])


def safe_ratio(
    value: float,
    baseline: float,
) -> float:
    if not (
        np.isfinite(value)
        and np.isfinite(baseline)
        and baseline > 1.0e-12
    ):
        return float("nan")

    return float(value / baseline)


def improvement_percent(
    value: float,
    baseline: float,
) -> float:
    ratio = safe_ratio(value, baseline)

    if not np.isfinite(ratio):
        return float("nan")

    return float(100.0 * (1.0 - ratio))


def build_diagnostic_timesteps(
    total_timesteps: int,
) -> list[int]:
    if total_timesteps <= 0:
        raise ValueError("扩散总步数必须大于0。")

    fractions = (
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.0,
    )

    return sorted(
        {
            min(
                int(round(fraction * total_timesteps)),
                total_timesteps - 1,
            )
            for fraction in fractions
        }
    )


def resolve_output_directory(
    runtime_configuration: dict[str, Any],
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = resolve_project_path(
            runtime_configuration,
            path,
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def load_d2_6_components(
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[
    PriorResidualTransformer,
    BroadLocalResidualDecomposer,
]:
    prior_state = metadata.get(
        "prior_residual_state"
    )

    if not isinstance(prior_state, dict):
        raise KeyError(
            "D2.6 checkpoint缺少有效prior_residual_state。"
        )

    if not bool(prior_state.get("enabled", False)):
        raise ValueError(
            "D2.6需要启用prior_residual_state。"
        )

    if str(
        prior_state.get("prior_method", "")
    ).strip().lower() != "pca_reconstruction":
        raise ValueError(
            "D2.6去噪诊断要求outer prior为pca_reconstruction。"
        )

    broad_local_state = metadata.get(
        "broad_local_residual_state"
    )

    if not isinstance(
        broad_local_state,
        dict,
    ):
        raise KeyError(
            "checkpoint缺少broad_local_residual_state；"
            "这不是可诊断的D2.6 checkpoint。"
        )

    if not bool(
        broad_local_state.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "broad_local_residual_state未启用。"
        )

    broad_local_configuration = (
        checkpoint_configuration.get(
            "broad_local_residual",
            {},
        )
        or {}
    )

    if not bool(
        broad_local_configuration.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "checkpoint configuration中的"
            "broad_local_residual未启用，"
            "与metadata状态不一致。"
        )

    prior_transformer = (
        PriorResidualTransformer.from_state_dict(
            prior_state
        )
    )

    decomposer = (
        BroadLocalResidualDecomposer.from_state_dict(
            broad_local_state
        )
    )

    return (
        prior_transformer,
        decomposer,
    )


def build_split_labels(
    number_of_spectra: int,
    metadata: dict[str, Any],
) -> list[str]:
    labels = [
        "unassigned"
        for _ in range(number_of_spectra)
    ]

    split_items = (
        (
            "train",
            metadata.get(
                "training_indices",
                [],
            ),
        ),
        (
            "validation",
            metadata.get(
                "validation_indices",
                [],
            ),
        ),
        (
            "test",
            metadata.get(
                "test_indices",
                [],
            ),
        ),
    )

    for split_name, indices in split_items:
        if indices is None:
            continue

        for value in indices:
            index = int(value)

            if not 0 <= index < number_of_spectra:
                raise IndexError(
                    f"checkpoint中的{split_name}索引{index}越界。"
                )

            if labels[index] != "unassigned":
                raise ValueError(
                    f"光谱索引{index}同时属于多个数据子集。"
                )

            labels[index] = split_name

    return labels


def select_indices(
    number_of_spectra: int,
    spectrum_index: int | None,
) -> list[int]:
    if spectrum_index is None:
        return list(range(number_of_spectra))

    index = int(spectrum_index)

    if not 0 <= index < number_of_spectra:
        raise IndexError(
            f"spectrum_index={index}越界；"
            f"当前共有{number_of_spectra}条光谱。"
        )

    return [index]


def make_summary(
    detailed: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    metric_columns = [
        "clipped_scaled_local_pearson",
        "clipped_scaled_local_rmse",
        "unclipped_scaled_local_pearson",
        "unclipped_scaled_local_rmse",
        "clipped_raw_local_pearson",
        "clipped_raw_local_rmse",
        "unclipped_raw_local_pearson",
        "unclipped_raw_local_rmse",
        "clipped_full_original_pearson",
        "clipped_full_original_rmse",
        "unclipped_full_original_pearson",
        "unclipped_full_original_rmse",
        "zero_local_baseline_original_rmse",
        "clipped_full_vs_zero_local_improvement_percent",
        "unclipped_full_vs_zero_local_improvement_percent",
        "model_noise_rmse",
        "zero_x0_noise_baseline_rmse",
        "noise_improvement_percent",
        "model_domain_clipping_fraction",
    ]

    rows: list[dict[str, Any]] = []

    group_key: str | list[str]
    if len(group_columns) == 1:
        group_key = group_columns[0]
    else:
        group_key = group_columns

    grouped = detailed.groupby(
        group_key,
        dropna=False,
        sort=True,
    )

    for group_value, frame in grouped:
        if len(group_columns) == 1:
            group_values = (
                group_value,
            )
        else:
            group_values = tuple(
                group_value
            )

        row: dict[str, Any] = {
            column: value
            for column, value in zip(
                group_columns,
                group_values,
                strict=True,
            )
        }

        row["number_of_spectra"] = int(
            len(frame)
        )

        for metric in metric_columns:
            values = pd.to_numeric(
                frame[metric],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            finite = values[
                np.isfinite(values)
            ]

            if finite.size == 0:
                median = float("nan")
                mean = float("nan")
                std = float("nan")
            else:
                median = float(
                    np.median(finite)
                )
                mean = float(
                    np.mean(finite)
                )
                std = float(
                    np.std(
                        finite,
                        ddof=(
                            1
                            if finite.size > 1
                            else 0
                        ),
                    )
                )

            row[f"{metric}_median"] = median
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "D2.6 broad-local residual在不同噪声时间步的"
            "一步x0/local residual恢复诊断。"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="当前项目配置文件。",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="D2.6 checkpoint文件。",
    )

    parser.add_argument(
        "--model-source",
        choices=(
            "raw",
            "ema",
        ),
        default="ema",
        help="使用raw或EMA模型权重。默认ema。",
    )

    parser.add_argument(
        "--spectrum-index",
        type=int,
        default=None,
        help=(
            "只诊断指定真实光谱索引。"
            "不指定时诊断全部真实光谱。"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="例如cuda、cuda:0或cpu。",
    )

    parser.add_argument(
        "--output-directory",
        default=(
            "outputs/diagnostics/"
            "d2_6_timestep_recovery"
        ),
        help="诊断输出目录。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="固定前向加噪随机种子。",
    )

    arguments = parser.parse_args()

    runtime_configuration = load_configuration(
        arguments.config
    )

    checkpoint_path = Path(
        arguments.checkpoint
    ).expanduser()

    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(
            runtime_configuration,
            checkpoint_path,
        )

    output_directory = resolve_output_directory(
        runtime_configuration,
        arguments.output_directory,
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
        raise TypeError(
            "checkpoint中的configuration必须是字典。"
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError(
            "checkpoint中的metadata必须是字典。"
        )

    prior_transformer, decomposer = (
        load_d2_6_components(
            checkpoint_configuration,
            metadata,
        )
    )

    length_adapter = (
        SpectrumLengthAdapter.from_metadata(
            metadata
        )
    )

    _, diffusion = build_diffusion_model(
        checkpoint_configuration,
        sequence_length=(
            length_adapter.padded_length
        ),
    )

    if arguments.model_source == "raw":
        model_state = checkpoint.get(
            "diffusion_state"
        )

        if not isinstance(model_state, dict):
            raise KeyError(
                "checkpoint缺少diffusion_state。"
            )
    else:
        ema_state = checkpoint.get(
            "ema_state"
        )

        if not isinstance(ema_state, dict):
            raise KeyError(
                "checkpoint缺少ema_state。"
            )

        model_state = ema_state.get(
            "ema_model"
        )

        if not isinstance(model_state, dict):
            raise KeyError(
                "checkpoint ema_state缺少ema_model。"
            )

    diffusion.load_state_dict(
        model_state,
        strict=True,
    )

    device = torch.device(
        arguments.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "PyTorch未检测到可用GPU。"
        )

    diffusion = diffusion.to(
        device
    )
    diffusion.eval()

    data_configuration = (
        checkpoint_configuration["data"]
    )

    input_directory = resolve_project_path(
        runtime_configuration,
        data_configuration[
            "input_directory"
        ],
    )

    collection = read_spectrum_collection(
        input_directory,
        data_configuration,
    )

    number_of_real_spectra = len(
        collection.spectra
    )

    split_labels = build_split_labels(
        number_of_real_spectra,
        metadata,
    )

    selected_indices = select_indices(
        number_of_real_spectra,
        arguments.spectrum_index,
    )

    original_2d = (
        length_adapter.interpolate_to_model_axis(
            [
                collection.spectra[index]
                for index in selected_indices
            ],
            [
                collection.raman_shifts[index]
                for index in selected_indices
            ],
        )
    )

    if "normalization_state" not in metadata:
        raise KeyError(
            "checkpoint metadata缺少normalization_state。"
        )

    normalizer = (
        GlobalMinMaxNormalizer.from_state_dict(
            metadata[
                "normalization_state"
            ]
        )
    )

    normalized_2d = normalizer.transform(
        original_2d
    )

    reference_priors_2d = (
        prior_transformer.reference_priors_for_spectra(
            normalized_2d
        )
    )

    raw_residual_2d = (
        np.asarray(
            normalized_2d,
            dtype=np.float32,
        )
        - np.asarray(
            reference_priors_2d,
            dtype=np.float32,
        )
    )

    true_broad_2d, true_local_2d = (
        decomposer.split_raw_residuals(
            raw_residual_2d
        )
    )

    true_scaled_local_2d = (
        decomposer.transform_raw_residuals(
            raw_residual_2d
        )
    )

    reconstructed_raw_residual_2d = (
        true_broad_2d
        + true_local_2d
    )

    decomposition_max_abs_error = float(
        np.max(
            np.abs(
                reconstructed_raw_residual_2d
                - raw_residual_2d
            )
        )
    )

    if decomposition_max_abs_error > 1.0e-5:
        raise RuntimeError(
            "D2.6 broad+local不能精确重建raw residual；"
            f"max_abs_error={decomposition_max_abs_error:.6g}。"
        )

    zero_local_normalized_2d = (
        np.asarray(
            reference_priors_2d,
            dtype=np.float32,
        )
        + np.asarray(
            true_broad_2d,
            dtype=np.float32,
        )
    )

    zero_local_original_2d = (
        normalizer.inverse_transform(
            zero_local_normalized_2d
        )
    )

    padded_scaled_local_2d = (
        length_adapter.adapt(
            true_scaled_local_2d
        )
    )

    x_start = (
        torch.as_tensor(
            padded_scaled_local_2d,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(1)
    )

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        int(arguments.seed)
    )

    fixed_noise = torch.randn(
        x_start.shape,
        generator=generator,
        device=device,
        dtype=x_start.dtype,
    )

    true_noise_2d = (
        length_adapter.restore(
            fixed_noise
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
    )

    total_timesteps = int(
        diffusion.num_timesteps
    )
    timesteps = build_diagnostic_timesteps(
        total_timesteps
    )

    detailed_rows: list[
        dict[str, Any]
    ] = []

    representative_index_in_batch = 0
    representative_global_index = (
        selected_indices[
            representative_index_in_batch
        ]
    )

    representative_local_recovery: dict[
        int,
        np.ndarray,
    ] = {}
    representative_full_recovery: dict[
        int,
        np.ndarray,
    ] = {}

    with torch.inference_mode():
        for timestep in timesteps:
            time_tensor = torch.full(
                (
                    x_start.shape[0],
                ),
                int(timestep),
                device=device,
                dtype=torch.long,
            )

            noisy = diffusion.q_sample(
                x_start,
                time_tensor,
                noise=fixed_noise,
            )

            model_prediction = (
                diffusion.model_predictions(
                    noisy,
                    time_tensor,
                    clip_x_start=False,
                )
            )

            predicted_noise_tensor = (
                model_prediction.pred_noise
            )
            predicted_unclipped_tensor = (
                model_prediction.pred_x_start
            )
            predicted_clipped_tensor = (
                predicted_unclipped_tensor.clamp(
                    -1.0,
                    1.0,
                )
            )

            predicted_unclipped_scaled_2d = (
                length_adapter.restore(
                    predicted_unclipped_tensor
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            )

            predicted_clipped_scaled_2d = (
                length_adapter.restore(
                    predicted_clipped_tensor
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            )

            predicted_noise_2d = (
                length_adapter.restore(
                    predicted_noise_tensor
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            )

            predicted_unclipped_local_2d = (
                decomposer.inverse_local_transform(
                    predicted_unclipped_scaled_2d
                )
            )

            predicted_clipped_local_2d = (
                decomposer.inverse_local_transform(
                    predicted_clipped_scaled_2d
                )
            )

            predicted_unclipped_normalized_2d = (
                np.asarray(
                    reference_priors_2d,
                    dtype=np.float32,
                )
                + np.asarray(
                    true_broad_2d,
                    dtype=np.float32,
                )
                + predicted_unclipped_local_2d
            )

            predicted_clipped_normalized_2d = (
                np.asarray(
                    reference_priors_2d,
                    dtype=np.float32,
                )
                + np.asarray(
                    true_broad_2d,
                    dtype=np.float32,
                )
                + predicted_clipped_local_2d
            )

            predicted_unclipped_original_2d = (
                normalizer.inverse_transform(
                    predicted_unclipped_normalized_2d
                )
            )

            predicted_clipped_original_2d = (
                normalizer.inverse_transform(
                    predicted_clipped_normalized_2d
                )
            )

            alpha_bar = float(
                diffusion.alphas_cumprod[
                    timestep
                ].item()
            )
            signal_coefficient = float(
                np.sqrt(alpha_bar)
            )
            noise_coefficient = float(
                np.sqrt(
                    max(
                        1.0 - alpha_bar,
                        0.0,
                    )
                )
            )

            if noise_coefficient > 1.0e-12:
                baseline_noise_tensor = (
                    noisy
                    / noise_coefficient
                )
                baseline_noise_2d = (
                    length_adapter.restore(
                        baseline_noise_tensor
                        .squeeze(1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            else:
                baseline_noise_2d = np.full_like(
                    true_noise_2d,
                    np.nan,
                    dtype=np.float32,
                )

            unclipped_values = np.asarray(
                predicted_unclipped_scaled_2d,
                dtype=np.float64,
            )
            clipping_fractions = np.mean(
                (
                    unclipped_values < -1.0
                )
                | (
                    unclipped_values > 1.0
                ),
                axis=1,
            )

            for batch_index, global_index in enumerate(
                selected_indices
            ):
                true_original = np.asarray(
                    original_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                true_normalized = np.asarray(
                    normalized_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                true_scaled_local = np.asarray(
                    true_scaled_local_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                true_local = np.asarray(
                    true_local_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )

                clipped_scaled_local = np.asarray(
                    predicted_clipped_scaled_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                unclipped_scaled_local = np.asarray(
                    predicted_unclipped_scaled_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                clipped_local = np.asarray(
                    predicted_clipped_local_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                unclipped_local = np.asarray(
                    predicted_unclipped_local_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )

                clipped_normalized = np.asarray(
                    predicted_clipped_normalized_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                unclipped_normalized = np.asarray(
                    predicted_unclipped_normalized_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                clipped_original = np.asarray(
                    predicted_clipped_original_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                unclipped_original = np.asarray(
                    predicted_unclipped_original_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )

                zero_local_original = np.asarray(
                    zero_local_original_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )

                true_noise = np.asarray(
                    true_noise_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                predicted_noise = np.asarray(
                    predicted_noise_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )
                baseline_noise = np.asarray(
                    baseline_noise_2d[
                        batch_index
                    ],
                    dtype=np.float64,
                )

                zero_local_original_rmse = (
                    calculate_rmse(
                        true_original,
                        zero_local_original,
                    )
                )

                clipped_original_rmse = calculate_rmse(
                    true_original,
                    clipped_original,
                )
                unclipped_original_rmse = calculate_rmse(
                    true_original,
                    unclipped_original,
                )

                model_noise_rmse = calculate_rmse(
                    true_noise,
                    predicted_noise,
                )
                zero_x0_noise_baseline_rmse = (
                    calculate_rmse(
                        true_noise,
                        baseline_noise,
                    )
                    if np.isfinite(
                        baseline_noise
                    ).all()
                    else float("nan")
                )

                detailed_rows.append(
                    {
                        "spectrum_index": int(
                            global_index
                        ),
                        "split": split_labels[
                            global_index
                        ],
                        "timestep": int(
                            timestep
                        ),
                        "signal_coefficient": (
                            signal_coefficient
                        ),
                        "noise_coefficient": (
                            noise_coefficient
                        ),
                        "clipped_scaled_local_pearson": (
                            calculate_pearson(
                                true_scaled_local,
                                clipped_scaled_local,
                            )
                        ),
                        "clipped_scaled_local_rmse": (
                            calculate_rmse(
                                true_scaled_local,
                                clipped_scaled_local,
                            )
                        ),
                        "unclipped_scaled_local_pearson": (
                            calculate_pearson(
                                true_scaled_local,
                                unclipped_scaled_local,
                            )
                        ),
                        "unclipped_scaled_local_rmse": (
                            calculate_rmse(
                                true_scaled_local,
                                unclipped_scaled_local,
                            )
                        ),
                        "clipped_raw_local_pearson": (
                            calculate_pearson(
                                true_local,
                                clipped_local,
                            )
                        ),
                        "clipped_raw_local_rmse": (
                            calculate_rmse(
                                true_local,
                                clipped_local,
                            )
                        ),
                        "unclipped_raw_local_pearson": (
                            calculate_pearson(
                                true_local,
                                unclipped_local,
                            )
                        ),
                        "unclipped_raw_local_rmse": (
                            calculate_rmse(
                                true_local,
                                unclipped_local,
                            )
                        ),
                        "clipped_full_normalized_rmse": (
                            calculate_rmse(
                                true_normalized,
                                clipped_normalized,
                            )
                        ),
                        "unclipped_full_normalized_rmse": (
                            calculate_rmse(
                                true_normalized,
                                unclipped_normalized,
                            )
                        ),
                        "clipped_full_original_pearson": (
                            calculate_pearson(
                                true_original,
                                clipped_original,
                            )
                        ),
                        "clipped_full_original_rmse": (
                            clipped_original_rmse
                        ),
                        "clipped_full_original_mae": (
                            calculate_mae(
                                true_original,
                                clipped_original,
                            )
                        ),
                        "unclipped_full_original_pearson": (
                            calculate_pearson(
                                true_original,
                                unclipped_original,
                            )
                        ),
                        "unclipped_full_original_rmse": (
                            unclipped_original_rmse
                        ),
                        "unclipped_full_original_mae": (
                            calculate_mae(
                                true_original,
                                unclipped_original,
                            )
                        ),
                        "zero_local_baseline_original_rmse": (
                            zero_local_original_rmse
                        ),
                        "zero_local_baseline_original_mae": (
                            calculate_mae(
                                true_original,
                                zero_local_original,
                            )
                        ),
                        "clipped_full_to_zero_local_rmse_ratio": (
                            safe_ratio(
                                clipped_original_rmse,
                                zero_local_original_rmse,
                            )
                        ),
                        "clipped_full_vs_zero_local_improvement_percent": (
                            improvement_percent(
                                clipped_original_rmse,
                                zero_local_original_rmse,
                            )
                        ),
                        "unclipped_full_to_zero_local_rmse_ratio": (
                            safe_ratio(
                                unclipped_original_rmse,
                                zero_local_original_rmse,
                            )
                        ),
                        "unclipped_full_vs_zero_local_improvement_percent": (
                            improvement_percent(
                                unclipped_original_rmse,
                                zero_local_original_rmse,
                            )
                        ),
                        "model_noise_rmse": (
                            model_noise_rmse
                        ),
                        "zero_x0_noise_baseline_rmse": (
                            zero_x0_noise_baseline_rmse
                        ),
                        "model_to_noise_baseline_rmse_ratio": (
                            safe_ratio(
                                model_noise_rmse,
                                zero_x0_noise_baseline_rmse,
                            )
                        ),
                        "noise_improvement_percent": (
                            improvement_percent(
                                model_noise_rmse,
                                zero_x0_noise_baseline_rmse,
                            )
                        ),
                        "model_domain_clipping_fraction": float(
                            clipping_fractions[
                                batch_index
                            ]
                        ),
                    }
                )

            representative_local_recovery[
                timestep
            ] = np.asarray(
                predicted_unclipped_local_2d[
                    representative_index_in_batch
                ],
                dtype=np.float64,
            )

            representative_full_recovery[
                timestep
            ] = np.asarray(
                predicted_unclipped_original_2d[
                    representative_index_in_batch
                ],
                dtype=np.float64,
            )

    detailed = pd.DataFrame(
        detailed_rows
    )

    summary = make_summary(
        detailed,
        ["timestep"],
    )

    summary_by_split = make_summary(
        detailed,
        [
            "split",
            "timestep",
        ],
    )

    detailed_path = (
        output_directory
        / "d2_6_timestep_recovery_detailed.csv"
    )
    summary_path = (
        output_directory
        / "d2_6_timestep_recovery_summary.csv"
    )
    split_summary_path = (
        output_directory
        / "d2_6_timestep_recovery_summary_by_split.csv"
    )

    detailed.to_csv(
        detailed_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )
    summary_by_split.to_csv(
        split_summary_path,
        index=False,
    )

    metadata_output = {
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_step": int(
            checkpoint.get(
                "step",
                -1,
            )
        ),
        "model_source": str(
            arguments.model_source
        ),
        "diagnostic_domain": (
            "d2_6_scaled_local_residual"
        ),
        "broad_component_during_recovery": (
            "true_broad_residual_of_each_real_spectrum"
        ),
        "outer_prior_during_recovery": (
            "reference_prior_of_each_real_spectrum"
        ),
        "number_of_real_spectra_in_file": int(
            number_of_real_spectra
        ),
        "selected_spectrum_indices": [
            int(value)
            for value in selected_indices
        ],
        "split_counts_selected": {
            label: int(
                sum(
                    split_labels[index]
                    == label
                    for index in selected_indices
                )
            )
            for label in (
                "train",
                "validation",
                "test",
                "unassigned",
            )
        },
        "seed": int(
            arguments.seed
        ),
        "total_timesteps": int(
            total_timesteps
        ),
        "diagnostic_timesteps": [
            int(value)
            for value in timesteps
        ],
        "decomposition_max_abs_error": (
            decomposition_max_abs_error
        ),
        "broad_sigma_cm1": float(
            decomposer.broad_sigma_cm1
        ),
        "broad_pca_components": int(
            decomposer.broad_pca_components.shape[0]
        ),
        "broad_pca_cumulative_explained_variance_ratio": float(
            np.sum(
                decomposer.broad_explained_variance_ratio_
            )
        ),
        "broad_score_clip_standard_deviations": float(
            decomposer.broad_score_clip_standard_deviations
        ),
        "local_residual_scale": float(
            decomposer.local_residual_scale
        ),
        "local_asinh_normalizer": float(
            decomposer.local_asinh_normalizer
        ),
    }

    metadata_path = (
        output_directory
        / "d2_6_timestep_recovery_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata_output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 代表性恢复图：默认取所选集合中的第一条。图只用于辅助观察，
    # 数值判断以全部光谱的summary为主。
    raman_axis = np.asarray(
        length_adapter.model_axis,
        dtype=np.float64,
    )

    representative_true_local = np.asarray(
        true_local_2d[
            representative_index_in_batch
        ],
        dtype=np.float64,
    )
    representative_true_original = np.asarray(
        original_2d[
            representative_index_in_batch
        ],
        dtype=np.float64,
    )

    selected_plot_timesteps = [
        timesteps[0],
        timesteps[
            len(timesteps) // 2
        ],
        timesteps[-1],
    ]

    figure = plt.figure(
        figsize=(12, 6)
    )
    plt.plot(
        raman_axis,
        representative_true_local,
        label="True local residual",
        linewidth=1.5,
    )

    for timestep in selected_plot_timesteps:
        plt.plot(
            raman_axis,
            representative_local_recovery[
                timestep
            ],
            label=(
                "Recovered local, "
                f"t={timestep}"
            ),
            linewidth=1.0,
            alpha=0.85,
        )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )
    plt.ylabel(
        "Local residual"
    )
    plt.title(
        "D2.6 one-step local-residual recovery "
        f"(spectrum {representative_global_index})"
    )
    plt.legend()
    plt.tight_layout()
    figure.savefig(
        output_directory
        / "representative_local_recovery.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(
        figsize=(12, 6)
    )
    plt.plot(
        raman_axis,
        representative_true_original,
        label="True spectrum",
        linewidth=1.5,
    )

    for timestep in selected_plot_timesteps:
        plt.plot(
            raman_axis,
            representative_full_recovery[
                timestep
            ],
            label=(
                "Recovered spectrum, "
                f"t={timestep}"
            ),
            linewidth=1.0,
            alpha=0.85,
        )

    plt.xlabel(
        "Raman shift (cm$^{-1}$)"
    )
    plt.ylabel(
        "Intensity"
    )
    plt.title(
        "D2.6 one-step full-spectrum recovery "
        f"(spectrum {representative_global_index})"
    )
    plt.legend()
    plt.tight_layout()
    figure.savefig(
        output_directory
        / "representative_full_recovery.png",
        dpi=180,
    )
    plt.close(figure)

    # 尝试同时写xlsx，若服务器未安装对应Excel writer，CSV仍已完整保存。
    excel_path = (
        output_directory
        / "d2_6_timestep_recovery.xlsx"
    )

    try:
        with pd.ExcelWriter(
            excel_path
        ) as writer:
            detailed.to_excel(
                writer,
                sheet_name="detailed",
                index=False,
            )
            summary.to_excel(
                writer,
                sheet_name="summary",
                index=False,
            )
            summary_by_split.to_excel(
                writer,
                sheet_name="summary_by_split",
                index=False,
            )
    except Exception as error:
        print(
            "警告：xlsx写入失败，但CSV结果已保存：",
            repr(error),
        )

    print()
    print(
        "===== D2.6 timestep recovery ====="
    )
    print(
        "checkpoint:",
        checkpoint_path,
    )
    print(
        "checkpoint step:",
        checkpoint.get("step"),
    )
    print(
        "model source:",
        arguments.model_source,
    )
    print(
        "diagnostic domain: scaled local residual"
    )
    print(
        "real spectra evaluated:",
        len(selected_indices),
    )
    print(
        "split counts:",
        metadata_output[
            "split_counts_selected"
        ],
    )
    print(
        "timesteps:",
        timesteps,
    )
    print(
        "broad during recovery: TRUE broad residual per real spectrum"
    )
    print(
        "outer prior during recovery: TRUE corresponding PCA reference prior"
    )
    print(
        "broad sigma cm^-1:",
        decomposer.broad_sigma_cm1,
    )
    print(
        "broad PCA components:",
        decomposer.broad_pca_components.shape[0],
    )
    print(
        "broad PCA cumulative EV:",
        float(
            np.sum(
                decomposer.broad_explained_variance_ratio_
            )
        ),
    )
    print(
        "local residual scale:",
        decomposer.local_residual_scale,
    )
    print(
        "decomposition max abs error:",
        decomposition_max_abs_error,
    )

    display_columns = [
        "timestep",
        "number_of_spectra",
        "unclipped_scaled_local_pearson_median",
        "unclipped_scaled_local_rmse_median",
        "unclipped_raw_local_pearson_median",
        "unclipped_raw_local_rmse_median",
        "unclipped_full_original_pearson_median",
        "unclipped_full_original_rmse_median",
        "zero_local_baseline_original_rmse_median",
        "unclipped_full_vs_zero_local_improvement_percent_median",
        "model_noise_rmse_median",
        "zero_x0_noise_baseline_rmse_median",
        "noise_improvement_percent_median",
        "model_domain_clipping_fraction_mean",
    ]

    print()
    print(
        "===== All-spectrum median summary ====="
    )
    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== Held-out summary (validation/test) ====="
    )

    heldout = summary_by_split[
        summary_by_split[
            "split"
        ].isin(
            [
                "validation",
                "test",
            ]
        )
    ]

    if heldout.empty:
        print(
            "没有可用的validation/test记录。"
        )
    else:
        heldout_columns = [
            "split",
            *display_columns,
        ]
        print(
            heldout[
                heldout_columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "===== Output files ====="
    )
    for path in (
        detailed_path,
        summary_path,
        split_summary_path,
        metadata_path,
        output_directory
        / "representative_local_recovery.png",
        output_directory
        / "representative_full_recovery.png",
    ):
        print(
            " -",
            path,
        )

    if excel_path.exists():
        print(
            " -",
            excel_path,
        )

    print()
    print(
        "本脚本只诊断D2.6 DDPM的local residual去噪能力；"
        "不修改checkpoint、配置或任何生成光谱。"
    )


if __name__ == "__main__":
    main()

"""Diagnose one-step x0 and noise recovery at diffusion timesteps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


def calculate_rmse(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate RMSE using float64 to reduce numerical overflow."""
    reference_64 = np.asarray(
        reference,
        dtype=np.float64,
    )
    prediction_64 = np.asarray(
        prediction,
        dtype=np.float64,
    )

    error = prediction_64 - reference_64

    return float(
        np.sqrt(
            np.mean(error**2)
        )
    )


def calculate_mae(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate mean absolute error."""
    reference_64 = np.asarray(
        reference,
        dtype=np.float64,
    )
    prediction_64 = np.asarray(
        prediction,
        dtype=np.float64,
    )

    return float(
        np.mean(
            np.abs(
                prediction_64
                - reference_64
            )
        )
    )


def calculate_pearson(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate Pearson correlation safely."""
    reference_64 = np.asarray(
        reference,
        dtype=np.float64,
    )
    prediction_64 = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if not (
        np.all(np.isfinite(reference_64))
        and np.all(np.isfinite(prediction_64))
    ):
        return float("nan")

    reference_std = float(
        np.std(reference_64)
    )
    prediction_std = float(
        np.std(prediction_64)
    )

    if (
        reference_std < 1.0e-12
        or prediction_std < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            reference_64,
            prediction_64,
        )[0, 1]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "检查不同噪声时间步的一步x0恢复能力、"
            "噪声预测误差和零信号基线。"
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
        help="待检查的checkpoint文件。",
    )

    parser.add_argument(
        "--model-source",
        choices=("raw", "ema"),
        default="raw",
        help="使用原始模型权重或EMA权重。",
    )

    parser.add_argument(
        "--spectrum-index",
        type=int,
        default=None,
        help=(
            "真实光谱索引。未指定时优先读取"
            "checkpoint配置中的diagnostic_overfit.spectrum_index。"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备，例如cuda、cuda:0或cpu。",
    )

    parser.add_argument(
        "--output-directory",
        default=(
            "outputs/diagnostics/"
            "timestep_recovery_detailed"
        ),
        help="诊断结果输出目录。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="固定诊断噪声的随机种子。",
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

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    if "configuration" not in checkpoint:
        raise KeyError(
            "checkpoint中缺少configuration。"
        )

    if "metadata" not in checkpoint:
        raise KeyError(
            "checkpoint中缺少metadata。"
        )

    checkpoint_configuration = checkpoint[
        "configuration"
    ]
    metadata = checkpoint["metadata"]

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
        if "diffusion_state" not in checkpoint:
            raise KeyError(
                "checkpoint中缺少diffusion_state。"
            )

        model_state = checkpoint[
            "diffusion_state"
        ]

    else:
        if "ema_state" not in checkpoint:
            raise KeyError(
                "checkpoint中缺少ema_state。"
            )

        ema_state = checkpoint[
            "ema_state"
        ]

        if "ema_model" not in ema_state:
            raise KeyError(
                "checkpoint['ema_state']中缺少ema_model。"
            )

        model_state = ema_state[
            "ema_model"
        ]

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

    diffusion = diffusion.to(device)
    diffusion.eval()

    # 使用checkpoint中的数据读取设置，避免当前YAML
    # 后续修改后与训练时的读取参数不一致。
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

    spectrum_index = arguments.spectrum_index

    if spectrum_index is None:
        diagnostic_configuration = (
            checkpoint_configuration.get(
                "diagnostic_overfit",
                {},
            )
        )

        spectrum_index = int(
            diagnostic_configuration.get(
                "spectrum_index",
                0,
            )
        )

    if not (
        0
        <= spectrum_index
        < len(collection.spectra)
    ):
        raise IndexError(
            f"spectrum_index={spectrum_index}越界；"
            f"当前共有{len(collection.spectra)}条光谱。"
        )

    # 将真实光谱插值到训练时使用的模型轴。
    true_original_2d = (
        length_adapter.interpolate_to_model_axis(
            [
                collection.spectra[
                    spectrum_index
                ]
            ],
            [
                collection.raman_shifts[
                    spectrum_index
                ]
            ],
        )
    )

    if "normalization_state" not in metadata:
        raise KeyError(
            "checkpoint metadata中缺少normalization_state。"
        )

    normalizer = (
        GlobalMinMaxNormalizer.from_state_dict(
            metadata[
                "normalization_state"
            ]
        )
    )

    # 使用checkpoint保存的训练集归一化参数。
    true_normalized_2d = normalizer.transform(
        true_original_2d
    )

    padded_2d = length_adapter.adapt(
        true_normalized_2d
    )

    # 模型输入形状：
    # [batch, channel, sequence_length]
    x_start = (
        torch.as_tensor(
            padded_2d,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(1)
    )

    # 所有时间步使用同一份固定随机噪声，
    # 方便公平比较不同时间步。
    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        arguments.seed
    )

    fixed_noise = torch.randn(
        x_start.shape,
        generator=generator,
        device=device,
        dtype=x_start.dtype,
    )

    total_timesteps = int(
        diffusion.num_timesteps
    )

    timesteps = sorted(
        {
            round(
                fraction
                * (total_timesteps - 1)
            )
            for fraction in (
                0.0,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                1.0,
            )
        }
    )

    true_original = np.asarray(
        true_original_2d[0],
        dtype=np.float64,
    )

    true_normalized = np.asarray(
        true_normalized_2d[0],
        dtype=np.float64,
    )

    raman_axis = np.asarray(
        length_adapter.model_axis
    )

    # 只评价去掉补零区域后的真实噪声。
    true_noise_2d = length_adapter.restore(
        fixed_noise
        .squeeze(1)
        .detach()
        .cpu()
        .numpy()
    )

    true_noise = np.asarray(
        true_noise_2d[0],
        dtype=np.float64,
    )

    metric_rows: list[dict[str, float | int]] = []

    clipped_recovered_spectra: dict[
        int,
        np.ndarray,
    ] = {}

    unclipped_recovered_spectra: dict[
        int,
        np.ndarray,
    ] = {}

    predicted_noises: dict[
        int,
        np.ndarray,
    ] = {}

    baseline_noises: dict[
        int,
        np.ndarray,
    ] = {}

    with torch.inference_mode():
        for timestep in timesteps:
            time_tensor = torch.full(
                (x_start.shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )

            # 前向扩散：
            # x0 + noise -> xt
            noisy_spectrum = diffusion.q_sample(
                x_start,
                time_tensor,
                noise=fixed_noise,
            )

            # 不裁剪x0，保留模型的原始预测。
            model_prediction = (
                diffusion.model_predictions(
                    noisy_spectrum,
                    time_tensor,
                    clip_x_start=False,
                )
            )

            predicted_noise_tensor = (
                model_prediction.pred_noise
            )

            predicted_x0_unclipped_tensor = (
                model_prediction.pred_x_start
            )

            # 与原诊断保持一致，将归一化x0裁剪到[-1, 1]。
            predicted_x0_clipped_tensor = (
                predicted_x0_unclipped_tensor.clamp(
                    -1.0,
                    1.0,
                )
            )

            # 去除为满足U-Net长度而增加的补零位置。
            predicted_normalized_unclipped_2d = (
                length_adapter.restore(
                    predicted_x0_unclipped_tensor
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            )

            predicted_normalized_clipped_2d = (
                length_adapter.restore(
                    predicted_x0_clipped_tensor
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

            predicted_normalized_unclipped = (
                np.asarray(
                    predicted_normalized_unclipped_2d[0],
                    dtype=np.float64,
                )
            )

            predicted_normalized_clipped = (
                np.asarray(
                    predicted_normalized_clipped_2d[0],
                    dtype=np.float64,
                )
            )

            predicted_noise = np.asarray(
                predicted_noise_2d[0],
                dtype=np.float64,
            )

            predicted_original_unclipped = (
                normalizer.inverse_transform(
                    predicted_normalized_unclipped_2d
                )[0]
            )

            predicted_original_clipped = (
                normalizer.inverse_transform(
                    predicted_normalized_clipped_2d
                )[0]
            )

            predicted_original_unclipped = np.asarray(
                predicted_original_unclipped,
                dtype=np.float64,
            )

            predicted_original_clipped = np.asarray(
                predicted_original_clipped,
                dtype=np.float64,
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

            # 零信号噪声基线：
            #
            # xt = sqrt(alpha_bar) * x0
            #      + sqrt(1-alpha_bar) * noise
            #
            # 假设x0=0，则：
            # noise_baseline = xt / sqrt(1-alpha_bar)
            #
            # 高噪声时间步中，这个基线天然会比较准确，
            # 因为xt中几乎全部是噪声。
            if noise_coefficient > 1.0e-12:
                baseline_noise_tensor = (
                    noisy_spectrum
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

                baseline_noise = np.asarray(
                    baseline_noise_2d[0],
                    dtype=np.float64,
                )

                baseline_noise_rmse = calculate_rmse(
                    true_noise,
                    baseline_noise,
                )

                baseline_noise_mae = calculate_mae(
                    true_noise,
                    baseline_noise,
                )
            else:
                baseline_noise = np.full_like(
                    true_noise,
                    np.nan,
                )

                baseline_noise_rmse = float("nan")
                baseline_noise_mae = float("nan")

            model_noise_rmse = calculate_rmse(
                true_noise,
                predicted_noise,
            )

            model_noise_mae = calculate_mae(
                true_noise,
                predicted_noise,
            )

            if (
                np.isfinite(baseline_noise_rmse)
                and baseline_noise_rmse > 1.0e-12
            ):
                noise_improvement_percent = float(
                    100.0
                    * (
                        baseline_noise_rmse
                        - model_noise_rmse
                    )
                    / baseline_noise_rmse
                )

                model_to_baseline_ratio = float(
                    model_noise_rmse
                    / baseline_noise_rmse
                )
            else:
                noise_improvement_percent = float("nan")
                model_to_baseline_ratio = float("nan")

            outside_range_mask = (
                (
                    predicted_normalized_unclipped
                    < -1.0
                )
                |
                (
                    predicted_normalized_unclipped
                    > 1.0
                )
            )

            clipping_fraction = float(
                np.mean(
                    outside_range_mask
                )
            )

            metric_rows.append(
                {
                    "timestep": timestep,
                    "signal_coefficient": (
                        signal_coefficient
                    ),
                    "noise_coefficient": (
                        noise_coefficient
                    ),
                    "clipped_x0_pearson": (
                        calculate_pearson(
                            true_original,
                            predicted_original_clipped,
                        )
                    ),
                    "clipped_x0_normalized_rmse": (
                        calculate_rmse(
                            true_normalized,
                            predicted_normalized_clipped,
                        )
                    ),
                    "clipped_x0_original_rmse": (
                        calculate_rmse(
                            true_original,
                            predicted_original_clipped,
                        )
                    ),
                    "unclipped_x0_pearson": (
                        calculate_pearson(
                            true_original,
                            predicted_original_unclipped,
                        )
                    ),
                    "unclipped_x0_normalized_rmse": (
                        calculate_rmse(
                            true_normalized,
                            predicted_normalized_unclipped,
                        )
                    ),
                    "unclipped_x0_original_rmse": (
                        calculate_rmse(
                            true_original,
                            predicted_original_unclipped,
                        )
                    ),
                    "unclipped_x0_original_mae": (
                        calculate_mae(
                            true_original,
                            predicted_original_unclipped,
                        )
                    ),
                    "clipping_fraction": (
                        clipping_fraction
                    ),
                    "model_noise_rmse": (
                        model_noise_rmse
                    ),
                    "model_noise_mae": (
                        model_noise_mae
                    ),
                    "zero_signal_baseline_noise_rmse": (
                        baseline_noise_rmse
                    ),
                    "zero_signal_baseline_noise_mae": (
                        baseline_noise_mae
                    ),
                    "model_to_baseline_rmse_ratio": (
                        model_to_baseline_ratio
                    ),
                    "noise_improvement_percent": (
                        noise_improvement_percent
                    ),
                }
            )

            clipped_recovered_spectra[
                timestep
            ] = predicted_original_clipped

            unclipped_recovered_spectra[
                timestep
            ] = predicted_original_unclipped

            predicted_noises[
                timestep
            ] = predicted_noise

            baseline_noises[
                timestep
            ] = baseline_noise

    output_directory = Path(
        arguments.output_directory
    ).expanduser()

    if not output_directory.is_absolute():
        output_directory = resolve_project_path(
            runtime_configuration,
            output_directory,
        )

    output_directory = (
        output_directory
        / arguments.model_source
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    clipped_spectra_frame = pd.DataFrame(
        {
            "raman_shift": raman_axis,
            "true": true_original,
        }
    )

    unclipped_spectra_frame = pd.DataFrame(
        {
            "raman_shift": raman_axis,
            "true": true_original,
        }
    )

    noise_frame = pd.DataFrame(
        {
            "raman_shift": raman_axis,
            "true_noise": true_noise,
        }
    )

    for timestep in timesteps:
        clipped_spectra_frame[
            f"clipped_recovered_t{timestep}"
        ] = clipped_recovered_spectra[
            timestep
        ]

        unclipped_spectra_frame[
            f"unclipped_recovered_t{timestep}"
        ] = unclipped_recovered_spectra[
            timestep
        ]

        noise_frame[
            f"model_noise_t{timestep}"
        ] = predicted_noises[
            timestep
        ]

        noise_frame[
            f"baseline_noise_t{timestep}"
        ] = baseline_noises[
            timestep
        ]

    workbook_path = (
        output_directory
        / "timestep_recovery_detailed.xlsx"
    )

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        metrics.to_excel(
            writer,
            sheet_name="metrics",
            index=False,
        )

        clipped_spectra_frame.to_excel(
            writer,
            sheet_name="clipped_x0",
            index=False,
        )

        unclipped_spectra_frame.to_excel(
            writer,
            sheet_name="unclipped_x0",
            index=False,
        )

        noise_frame.to_excel(
            writer,
            sheet_name="noise_predictions",
            index=False,
        )

    # 选择低、中、高三个时间步绘图。
    selected_timesteps = (
        timesteps[0],
        timesteps[
            len(timesteps) // 2
        ],
        timesteps[-1],
    )

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(16, 11),
        sharex=True,
    )

    for row_index, timestep in enumerate(
        selected_timesteps
    ):
        clipped_axis = axes[
            row_index,
            0,
        ]

        unclipped_axis = axes[
            row_index,
            1,
        ]

        clipped_axis.plot(
            raman_axis,
            true_original,
            label="True",
            linewidth=1.2,
        )

        clipped_axis.plot(
            raman_axis,
            clipped_recovered_spectra[
                timestep
            ],
            label=(
                f"Clipped recovery, "
                f"t={timestep}"
            ),
            linewidth=1.0,
            alpha=0.85,
        )

        clipped_axis.set_title(
            f"Clipped x0 recovery, t={timestep}"
        )

        clipped_axis.set_ylabel(
            "Intensity"
        )

        clipped_axis.legend(
            fontsize=8,
        )

        unclipped_axis.plot(
            raman_axis,
            true_original,
            label="True",
            linewidth=1.2,
        )

        unclipped_axis.plot(
            raman_axis,
            unclipped_recovered_spectra[
                timestep
            ],
            label=(
                f"Unclipped recovery, "
                f"t={timestep}"
            ),
            linewidth=1.0,
            alpha=0.85,
        )

        unclipped_axis.set_title(
            f"Unclipped x0 recovery, t={timestep}"
        )

        unclipped_axis.set_ylabel(
            "Intensity"
        )

        unclipped_axis.legend(
            fontsize=8,
        )

    axes[-1, 0].set_xlabel(
        "Raman shift (cm$^{-1}$)"
    )

    axes[-1, 1].set_xlabel(
        "Raman shift (cm$^{-1}$)"
    )

    figure.tight_layout()

    recovery_figure_path = (
        output_directory
        / "timestep_recovery_detailed.png"
    )

    figure.savefig(
        recovery_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    # 绘制噪声RMSE与基线的比较。
    noise_figure, noise_axis = plt.subplots(
        figsize=(10, 6)
    )

    noise_axis.plot(
        metrics["timestep"],
        metrics["model_noise_rmse"],
        marker="o",
        label="Model noise RMSE",
    )

    noise_axis.plot(
        metrics["timestep"],
        metrics[
            "zero_signal_baseline_noise_rmse"
        ],
        marker="s",
        label="Zero-signal baseline RMSE",
    )

    noise_axis.set_xlabel(
        "Timestep"
    )

    noise_axis.set_ylabel(
        "Noise RMSE"
    )

    noise_axis.set_title(
        "Model noise prediction versus zero-signal baseline"
    )

    noise_axis.grid(
        alpha=0.25
    )

    noise_axis.legend()

    noise_figure.tight_layout()

    noise_figure_path = (
        output_directory
        / "noise_prediction_comparison.png"
    )

    noise_figure.savefig(
        noise_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(noise_figure)

    print(
        "\n===== 不同时间步详细恢复诊断 ====="
    )

    print(
        f"objective：{diffusion.objective}"
    )

    print(
        f"扩散总步数：{total_timesteps}"
    )

    print(
        f"模型来源：{arguments.model_source}"
    )

    print(
        f"诊断随机种子：{arguments.seed}"
    )

    print(
        f"原始光谱索引：{spectrum_index}"
    )

    print(
        "源文件："
        f"{collection.relative_source_files[spectrum_index]}"
    )

    print(
        "光谱名称："
        f"{collection.spectrum_names[spectrum_index]}"
    )

    terminal_columns = [
        "timestep",
        "clipped_x0_pearson",
        "unclipped_x0_pearson",
        "unclipped_x0_normalized_rmse",
        "clipping_fraction",
        "model_noise_rmse",
        "zero_signal_baseline_noise_rmse",
        "model_to_baseline_rmse_ratio",
        "noise_improvement_percent",
    ]

    print(
        "\n"
        + metrics[
            terminal_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    print(
        "\n指标含义："
    )

    print(
        "1. clipping_fraction越大，说明越多预测点超出[-1,1]。"
    )

    print(
        "2. model_to_baseline_rmse_ratio < 1，"
        "说明模型优于零信号基线。"
    )

    print(
        "3. noise_improvement_percent > 0，"
        "说明模型相对基线有改善。"
    )

    print(
        "4. noise_improvement_percent <= 0，"
        "说明模型没有优于简单基线。"
    )

    print(
        f"\n结果表：{workbook_path}"
    )

    print(
        f"恢复图：{recovery_figure_path}"
    )

    print(
        f"噪声比较图：{noise_figure_path}"
    )


if __name__ == "__main__":
    main()
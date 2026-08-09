from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model_builder import build_diffusion_model
from src.spectrum_length_adapter import SpectrumLengthAdapter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诊断 DDPM 采样过程中 x_t 与 pred_x0 的多样性坍缩位置。"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-source",
        choices=("raw", "ema"),
        default="raw",
        help="使用 checkpoint 的原始模型或 EMA 模型。",
    )
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-directory",
        default="outputs/diagnostics/sampling_diversity_collapse",
    )
    return parser.parse_args()


def flatten_samples(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(tensor.shape[0], -1)


def diversity_metrics(tensor: torch.Tensor) -> dict[str, float]:
    values = flatten_samples(tensor).cpu()
    number, length = values.shape

    pointwise_std = values.std(dim=0, unbiased=False)
    centered = values - values.mean(dim=1, keepdim=True)
    row_norm = torch.linalg.vector_norm(centered, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = centered / row_norm
    correlation = normalized @ normalized.T

    upper_mask = torch.triu(
        torch.ones((number, number), dtype=torch.bool),
        diagonal=1,
    )
    pairwise_pearson = correlation[upper_mask]

    pairwise_squared_distance = (
        (values[:, None, :] - values[None, :, :]) ** 2
    ).mean(dim=2)
    pairwise_rmse = pairwise_squared_distance.sqrt()
    pairwise_rmse_upper = pairwise_rmse[upper_mask]

    correlation_without_self = correlation.clone()
    correlation_without_self.fill_diagonal_(-float("inf"))
    nearest_neighbor_pearson = correlation_without_self.max(dim=1).values.mean()

    distance_without_self = pairwise_rmse.clone()
    distance_without_self.fill_diagonal_(float("inf"))
    nearest_neighbor_rmse = distance_without_self.min(dim=1).values.mean()

    pca_input = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(pca_input)
    explained_variance = singular_values.square()
    total_variance = explained_variance.sum().clamp_min(1e-12)
    explained_ratio = explained_variance / total_variance

    clipping_fraction = (
        (values <= -0.999999) | (values >= 0.999999)
    ).float().mean()

    return {
        "mean_pairwise_pearson": float(pairwise_pearson.mean()),
        "minimum_pairwise_pearson": float(pairwise_pearson.min()),
        "maximum_pairwise_pearson": float(pairwise_pearson.max()),
        "mean_pairwise_rmse": float(pairwise_rmse_upper.mean()),
        "mean_nearest_neighbor_pearson": float(nearest_neighbor_pearson),
        "mean_nearest_neighbor_rmse": float(nearest_neighbor_rmse),
        "mean_pointwise_std": float(pointwise_std.mean()),
        "median_pointwise_std": float(pointwise_std.median()),
        "pca_pc1_explained_ratio": float(explained_ratio[0]),
        "pca_pc2_explained_ratio": float(
            explained_ratio[1] if len(explained_ratio) > 1 else 0.0
        ),
        "clipping_fraction": float(clipping_fraction),
    }


def main() -> None:
    arguments = parse_arguments()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    output_directory = Path(arguments.output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    if arguments.number < 2:
        raise ValueError("--number 必须至少为 2。")

    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 未检测到可用 GPU。")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "configuration" not in checkpoint or "metadata" not in checkpoint:
        raise KeyError("checkpoint 缺少 configuration 或 metadata，不能安全重建模型。")

    checkpoint_configuration = checkpoint["configuration"]
    metadata = checkpoint["metadata"]

    length_adapter = SpectrumLengthAdapter.from_metadata(metadata)
    built_model = build_diffusion_model(
        checkpoint_configuration,
        sequence_length=length_adapter.padded_length,
    )
    diffusion = built_model[1] if isinstance(built_model, tuple) else built_model

    if arguments.model_source == "raw":
        model_state = checkpoint["diffusion_state"]
    else:
        model_state = checkpoint["ema_state"]["ema_model"]

    diffusion.load_state_dict(model_state, strict=True)
    diffusion = diffusion.to(arguments.device)
    diffusion.eval()

    if diffusion.is_ddim_sampling:
        raise RuntimeError(
            "当前 checkpoint 仍会走 DDIM；本实验必须使用 sampling_timesteps "
            "等于 diffusion_steps 的 DDPM100 checkpoint。"
        )

    total_timesteps = int(diffusion.num_timesteps)
    tracked_timesteps = sorted(
        {
            total_timesteps - 1,
            75,
            50,
            25,
            10,
            0,
        },
        reverse=True,
    )
    tracked_timesteps = [
        timestep
        for timestep in tracked_timesteps
        if 0 <= timestep < total_timesteps
    ]

    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    sequence_length = int(diffusion.seq_length)
    channels = int(diffusion.channels)
    shape = (arguments.number, channels, sequence_length)

    device = torch.device(arguments.device)
    initial_noise = torch.randn(shape, device=device)
    image = initial_noise.clone()

    rows: list[dict[str, float | int | str]] = []
    snapshots: dict[str, np.ndarray] = {
        "initial_x_T": initial_noise.detach().cpu().numpy(),
    }

    with torch.no_grad():
        for timestep in reversed(range(total_timesteps)):
            current_x_t = image.detach().clone()

            next_image, predicted_x0 = diffusion.p_sample(
                image,
                timestep,
                x_self_cond=None,
                clip_denoised=True,
            )

            if timestep in tracked_timesteps:
                x_t_metrics = diversity_metrics(current_x_t)
                pred_x0_metrics = diversity_metrics(predicted_x0)

                rows.append(
                    {
                        "timestep": timestep,
                        "tensor": "x_t_before_update",
                        **x_t_metrics,
                    }
                )
                rows.append(
                    {
                        "timestep": timestep,
                        "tensor": "pred_x0",
                        **pred_x0_metrics,
                    }
                )

                snapshots[f"x_t_{timestep:03d}"] = (
                    current_x_t.detach().cpu().numpy()
                )
                snapshots[f"pred_x0_{timestep:03d}"] = (
                    predicted_x0.detach().cpu().numpy()
                )

            image = next_image

    final_sample = image.detach().cpu().numpy()
    snapshots["final_x_0"] = final_sample

    result_table = pd.DataFrame(rows).sort_values(
        ["timestep", "tensor"],
        ascending=[False, True],
    )
    csv_path = output_directory / "sampling_diversity_metrics.csv"
    result_table.to_csv(csv_path, index=False, float_format="%.10f")

    npz_path = output_directory / "sampling_diversity_snapshots.npz"
    np.savez_compressed(npz_path, **snapshots)

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_source": arguments.model_source,
        "seed": arguments.seed,
        "number_of_generated_spectra": arguments.number,
        "diffusion_steps": total_timesteps,
        "sampling_method": "DDPM_p_sample_loop",
        "objective": diffusion.objective,
        "tracked_timesteps": tracked_timesteps,
        "sequence_length": sequence_length,
        "channels": channels,
    }
    json_path = output_directory / "diagnostic_metadata.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 诊断完成 =====")
    print(f"checkpoint：{checkpoint_path}")
    print(f"模型来源：{arguments.model_source}")
    print(f"采样方法：DDPM，{total_timesteps} 步")
    print(f"固定初始噪声数量：{arguments.number}")
    print(f"记录时间步：{tracked_timesteps}")
    print(f"指标文件：{csv_path}")
    print(f"中间张量：{npz_path}")

    print("\n===== 关键结果：pred_x0 =====")
    print(
        result_table.loc[
            result_table["tensor"] == "pred_x0",
            [
                "timestep",
                "mean_pairwise_pearson",
                "mean_pairwise_rmse",
                "mean_pointwise_std",
                "pca_pc1_explained_ratio",
                "clipping_fraction",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
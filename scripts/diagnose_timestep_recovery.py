"""诊断不同扩散时间步下的一步x0与噪声恢复能力。

该脚本同时支持：
1. D0/D1：模型直接学习完整的全局归一化光谱；
2. D2：模型学习“完整归一化光谱-参考先验”的缩放残差。

D2诊断必须在缩放残差域中执行前向加噪和模型恢复，然后依次：
缩放残差 -> 加回对应参考先验 -> 完整归一化光谱 -> 原始强度光谱。

对于D2.2 PCA可变先验，诊断使用当前真实光谱对应的
``reference_priors``，不使用随机采样先验，也不退回固定中位数先验。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from src.prior_residual import PriorResidualTransformer
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


def calculate_rmse(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """使用float64计算RMSE，降低数值溢出风险。"""

    reference_64 = np.asarray(reference, dtype=np.float64)
    prediction_64 = np.asarray(prediction, dtype=np.float64)
    error = prediction_64 - reference_64

    return float(np.sqrt(np.mean(error**2)))


def calculate_mae(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """计算平均绝对误差。"""

    reference_64 = np.asarray(reference, dtype=np.float64)
    prediction_64 = np.asarray(prediction, dtype=np.float64)

    return float(np.mean(np.abs(prediction_64 - reference_64)))


def calculate_pearson(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """安全计算Pearson相关系数。"""

    reference_64 = np.asarray(reference, dtype=np.float64)
    prediction_64 = np.asarray(prediction, dtype=np.float64)

    if not (
        np.all(np.isfinite(reference_64))
        and np.all(np.isfinite(prediction_64))
    ):
        return float("nan")

    if (
        float(np.std(reference_64)) < 1.0e-12
        or float(np.std(prediction_64)) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(reference_64, prediction_64)[0, 1]
    )


def safe_ratio(
    value: float,
    baseline: float,
) -> float:
    """计算value/baseline；基线无效时返回NaN。"""

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
    """计算相对基线的误差改善百分比。"""

    ratio = safe_ratio(value, baseline)

    if not np.isfinite(ratio):
        return float("nan")

    return float(100.0 * (1.0 - ratio))


def build_diagnostic_timesteps(
    total_timesteps: int,
) -> list[int]:
    """构建边界、低噪声、中噪声和高噪声诊断时间步。"""

    if total_timesteps <= 0:
        raise ValueError("扩散总步数必须大于0。")

    # 当总步数为100时得到：0、10、25、50、75、90、99。
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


def load_prior_residual_transformer(
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
) -> PriorResidualTransformer | None:
    """自动识别checkpoint是否为D2，并恢复先验残差变换器。"""

    prior_configuration = checkpoint_configuration.get(
        "prior_residual",
        {},
    )

    configured_enabled = (
        isinstance(prior_configuration, dict)
        and bool(prior_configuration.get("enabled", False))
    )

    state = metadata.get("prior_residual_state")

    if state is None:
        if configured_enabled:
            raise KeyError(
                "checkpoint配置启用了prior_residual，"
                "但metadata中缺少prior_residual_state。"
            )

        return None

    if not isinstance(state, dict):
        raise TypeError(
            "checkpoint metadata中的prior_residual_state"
            "必须是字典。"
        )

    state_enabled = bool(state.get("enabled", False))

    if not state_enabled:
        if configured_enabled:
            raise ValueError(
                "checkpoint配置启用了prior_residual，"
                "但保存的prior_residual_state未启用。"
            )

        return None

    return PriorResidualTransformer.from_state_dict(state)


def restore_full_normalized_spectra(
    model_domain_spectra: np.ndarray,
    prior_transformer: PriorResidualTransformer | None,
    reference_priors: np.ndarray | None = None,
) -> np.ndarray:
    """把模型数据域输出恢复为完整的全局归一化光谱。"""

    values = np.asarray(model_domain_spectra, dtype=np.float32)

    if prior_transformer is None:
        return values

    return prior_transformer.inverse_transform(
        values,
        reference_priors=reference_priors,
    )


def configure_checkpoint_diversity_constraints(
    diffusion: torch.nn.Module,
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    """按checkpoint状态注册D3.4多样性约束模块。

    D3.4的多样性约束模块含有缓冲区，例如逐波数标准差和白化
    标准差。这些缓冲区会被保存进state_dict。因此，必须在
    ``load_state_dict(strict=True)``之前重建并注册该模块；否则D3.4
    checkpoint会被误判为包含“Unexpected key(s)”。

    对D0–D3.3 checkpoint，本函数不注册任何模块并返回False。
    """

    diversity_configuration = checkpoint_configuration.get(
        "diversity_constraints",
        {},
    )

    diversity_enabled_in_configuration = (
        isinstance(diversity_configuration, dict)
        and bool(diversity_configuration.get("enabled", False))
    )

    diversity_constraint_state = metadata.get(
        "diversity_constraint_state"
    )

    if diversity_constraint_state is None:
        if diversity_enabled_in_configuration:
            raise KeyError(
                "checkpoint配置启用了D3.4多样性约束，"
                "但metadata中缺少diversity_constraint_state。"
            )

        return False

    if not isinstance(diversity_constraint_state, dict):
        raise TypeError(
            "checkpoint metadata中的diversity_constraint_state"
            "必须是字典。"
        )

    diversity_enabled_in_state = bool(
        diversity_constraint_state.get("enabled", False)
    )

    if diversity_enabled_in_state != diversity_enabled_in_configuration:
        raise ValueError(
            "checkpoint配置与metadata中的D3.4多样性约束启用状态"
            "不一致，无法安全加载checkpoint。"
        )

    if not diversity_enabled_in_state:
        return False

    configure_diversity = getattr(
        diffusion,
        "configure_diversity_constraints",
        None,
    )

    if not callable(configure_diversity):
        raise RuntimeError(
            "当前扩散模型缺少configure_diversity_constraints，"
            "无法加载D3.4 checkpoint。"
        )

    configure_diversity(
        diversity_constraint_state=diversity_constraint_state,
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "检查不同噪声时间步的一步x0恢复能力、"
            "噪声预测误差和简单基线。"
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
            "真实光谱索引。未指定时优先读取checkpoint配置中的"
            "diagnostic_overfit.spectrum_index。"
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

    runtime_configuration = load_configuration(arguments.config)

    checkpoint_path = Path(arguments.checkpoint).expanduser()

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
        raise KeyError("checkpoint中缺少configuration。")

    if "metadata" not in checkpoint:
        raise KeyError("checkpoint中缺少metadata。")

    checkpoint_configuration = checkpoint["configuration"]
    metadata = checkpoint["metadata"]

    if not isinstance(checkpoint_configuration, dict):
        raise TypeError("checkpoint中的configuration必须是字典。")

    if not isinstance(metadata, dict):
        raise TypeError("checkpoint中的metadata必须是字典。")

    length_adapter = SpectrumLengthAdapter.from_metadata(metadata)

    _, diffusion = build_diffusion_model(
        checkpoint_configuration,
        sequence_length=length_adapter.padded_length,
    )

    diversity_constraints_enabled = (
        configure_checkpoint_diversity_constraints(
            diffusion=diffusion,
            checkpoint_configuration=checkpoint_configuration,
            metadata=metadata,
        )
    )

    if arguments.model_source == "raw":
        if "diffusion_state" not in checkpoint:
            raise KeyError("checkpoint中缺少diffusion_state。")

        model_state = checkpoint["diffusion_state"]
    else:
        if "ema_state" not in checkpoint:
            raise KeyError("checkpoint中缺少ema_state。")

        ema_state = checkpoint["ema_state"]

        if not isinstance(ema_state, dict):
            raise TypeError("checkpoint['ema_state']必须是字典。")

        if "ema_model" not in ema_state:
            raise KeyError(
                "checkpoint['ema_state']中缺少ema_model。"
            )

        model_state = ema_state["ema_model"]

    diffusion.load_state_dict(model_state, strict=True)

    device = torch.device(arguments.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch未检测到可用GPU。")

    diffusion = diffusion.to(device)
    diffusion.eval()

    # 使用checkpoint中的数据读取设置，避免当前YAML后续修改后
    # 与训练时的数据参数不一致。
    data_configuration = checkpoint_configuration["data"]

    input_directory = resolve_project_path(
        runtime_configuration,
        data_configuration["input_directory"],
    )

    collection = read_spectrum_collection(
        input_directory,
        data_configuration,
    )

    spectrum_index = arguments.spectrum_index

    if spectrum_index is None:
        diagnostic_configuration = checkpoint_configuration.get(
            "diagnostic_overfit",
            {},
        )

        spectrum_index = int(
            diagnostic_configuration.get("spectrum_index", 0)
        )

    if not 0 <= spectrum_index < len(collection.spectra):
        raise IndexError(
            f"spectrum_index={spectrum_index}越界；"
            f"当前共有{len(collection.spectra)}条光谱。"
        )

    true_original_2d = length_adapter.interpolate_to_model_axis(
        [collection.spectra[spectrum_index]],
        [collection.raman_shifts[spectrum_index]],
    )

    if "normalization_state" not in metadata:
        raise KeyError(
            "checkpoint metadata中缺少normalization_state。"
        )

    normalizer = GlobalMinMaxNormalizer.from_state_dict(
        metadata["normalization_state"]
    )

    # 完整光谱的全局归一化结果。
    true_normalized_2d = normalizer.transform(true_original_2d)

    # 自动识别D0/D1或D2。
    prior_transformer = load_prior_residual_transformer(
        checkpoint_configuration,
        metadata,
    )

    # D2.2 PCA可变先验：为当前诊断光谱解析其对应参考先验。
    # 后续transform、prior-only恢复和预测恢复必须使用同一个先验。
    true_reference_priors: np.ndarray | None

    if prior_transformer is None:
        diagnostic_mode = "full_normalized_spectrum"
        true_reference_priors = None
        true_model_domain_2d = np.asarray(
            true_normalized_2d,
            dtype=np.float32,
        )
        prior_normalized_2d = None
        prior_original_2d = None
    else:
        diagnostic_mode = "prior_residual_scaled"

        true_reference_priors = (
            prior_transformer.reference_priors_for_spectra(
                true_normalized_2d
            )
        )

        # D2的模型输入必须是缩放残差，不能把完整光谱直接送入模型。
        true_model_domain_2d = prior_transformer.transform(
            true_normalized_2d,
            reference_priors=true_reference_priors,
        )

        # 先验基线必须是当前真实光谱对应的先验。
        # 对D2.2而言不能使用prior_batch(1)替代reference_priors。
        prior_normalized_2d = np.asarray(
            true_reference_priors,
            dtype=np.float32,
        )
        prior_original_2d = normalizer.inverse_transform(
            prior_normalized_2d
        )

    padded_2d = length_adapter.adapt(true_model_domain_2d)

    x_start = (
        torch.as_tensor(
            padded_2d,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(1)
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(arguments.seed)

    fixed_noise = torch.randn(
        x_start.shape,
        generator=generator,
        device=device,
        dtype=x_start.dtype,
    )

    total_timesteps = int(diffusion.num_timesteps)
    timesteps = build_diagnostic_timesteps(total_timesteps)

    true_original = np.asarray(
        true_original_2d[0],
        dtype=np.float64,
    )

    true_normalized = np.asarray(
        true_normalized_2d[0],
        dtype=np.float64,
    )

    true_model_domain = np.asarray(
        true_model_domain_2d[0],
        dtype=np.float64,
    )

    raman_axis = np.asarray(
        length_adapter.model_axis,
        dtype=np.float64,
    )

    true_noise_2d = length_adapter.restore(
        fixed_noise.squeeze(1).detach().cpu().numpy()
    )

    true_noise = np.asarray(
        true_noise_2d[0],
        dtype=np.float64,
    )

    if prior_transformer is None:
        prior_normalized = None
        prior_original = None
        prior_only_pearson = float("nan")
        prior_only_normalized_rmse = float("nan")
        prior_only_original_rmse = float("nan")
        prior_only_original_mae = float("nan")
    else:
        prior_normalized = np.asarray(
            prior_normalized_2d[0],
            dtype=np.float64,
        )
        prior_original = np.asarray(
            prior_original_2d[0],
            dtype=np.float64,
        )
        prior_only_pearson = calculate_pearson(
            true_original,
            prior_original,
        )
        prior_only_normalized_rmse = calculate_rmse(
            true_normalized,
            prior_normalized,
        )
        prior_only_original_rmse = calculate_rmse(
            true_original,
            prior_original,
        )
        prior_only_original_mae = calculate_mae(
            true_original,
            prior_original,
        )

    metric_rows: list[dict[str, float | int]] = []

    clipped_recovered_spectra: dict[int, np.ndarray] = {}
    unclipped_recovered_spectra: dict[int, np.ndarray] = {}
    predicted_noises: dict[int, np.ndarray] = {}
    baseline_noises: dict[int, np.ndarray] = {}

    with torch.inference_mode():
        for timestep in timesteps:
            time_tensor = torch.full(
                (x_start.shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )

            # D0/D1：对完整归一化光谱加噪。
            # D2：对缩放残差加噪。
            noisy_spectrum = diffusion.q_sample(
                x_start,
                time_tensor,
                noise=fixed_noise,
            )

            model_prediction = diffusion.model_predictions(
                noisy_spectrum,
                time_tensor,
                clip_x_start=False,
            )

            predicted_noise_tensor = model_prediction.pred_noise
            predicted_x0_unclipped_tensor = (
                model_prediction.pred_x_start
            )

            # 与扩散模型的常规采样范围保持一致。
            # D2中这里裁剪的是缩放残差，不是最终原始强度光谱。
            predicted_x0_clipped_tensor = (
                predicted_x0_unclipped_tensor.clamp(-1.0, 1.0)
            )

            predicted_model_unclipped_2d = length_adapter.restore(
                predicted_x0_unclipped_tensor
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )

            predicted_model_clipped_2d = length_adapter.restore(
                predicted_x0_clipped_tensor
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )

            predicted_noise_2d = length_adapter.restore(
                predicted_noise_tensor
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )

            predicted_model_unclipped = np.asarray(
                predicted_model_unclipped_2d[0],
                dtype=np.float64,
            )

            predicted_model_clipped = np.asarray(
                predicted_model_clipped_2d[0],
                dtype=np.float64,
            )

            predicted_noise = np.asarray(
                predicted_noise_2d[0],
                dtype=np.float64,
            )

            # D2必须先加回当前真实光谱对应的参考先验，
            # 再进行全局反归一化。
            predicted_normalized_unclipped_2d = (
                restore_full_normalized_spectra(
                    predicted_model_unclipped_2d,
                    prior_transformer,
                    reference_priors=true_reference_priors,
                )
            )

            predicted_normalized_clipped_2d = (
                restore_full_normalized_spectra(
                    predicted_model_clipped_2d,
                    prior_transformer,
                    reference_priors=true_reference_priors,
                )
            )

            predicted_normalized_unclipped = np.asarray(
                predicted_normalized_unclipped_2d[0],
                dtype=np.float64,
            )

            predicted_normalized_clipped = np.asarray(
                predicted_normalized_clipped_2d[0],
                dtype=np.float64,
            )

            predicted_original_unclipped = np.asarray(
                normalizer.inverse_transform(
                    predicted_normalized_unclipped_2d
                )[0],
                dtype=np.float64,
            )

            predicted_original_clipped = np.asarray(
                normalizer.inverse_transform(
                    predicted_normalized_clipped_2d
                )[0],
                dtype=np.float64,
            )

            alpha_bar = float(
                diffusion.alphas_cumprod[timestep].item()
            )

            signal_coefficient = float(np.sqrt(alpha_bar))
            noise_coefficient = float(
                np.sqrt(max(1.0 - alpha_bar, 0.0))
            )

            # 简单基线假定模型数据域中的x0=0：
            # D0/D1中表示零归一化信号；
            # D2中表示零缩放残差，即“只使用对应参考先验”。
            if noise_coefficient > 1.0e-12:
                baseline_noise_tensor = (
                    noisy_spectrum / noise_coefficient
                )

                baseline_noise_2d = length_adapter.restore(
                    baseline_noise_tensor
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
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

            model_to_noise_baseline_ratio = safe_ratio(
                model_noise_rmse,
                baseline_noise_rmse,
            )
            noise_improvement = improvement_percent(
                model_noise_rmse,
                baseline_noise_rmse,
            )

            model_domain_clipping_fraction = float(
                np.mean(
                    (predicted_model_unclipped < -1.0)
                    | (predicted_model_unclipped > 1.0)
                )
            )

            unclipped_full_outside_fraction = float(
                np.mean(
                    (predicted_normalized_unclipped < -1.0)
                    | (predicted_normalized_unclipped > 1.0)
                )
            )

            clipped_full_outside_fraction = float(
                np.mean(
                    (predicted_normalized_clipped < -1.0)
                    | (predicted_normalized_clipped > 1.0)
                )
            )

            clipped_original_rmse = calculate_rmse(
                true_original,
                predicted_original_clipped,
            )
            unclipped_original_rmse = calculate_rmse(
                true_original,
                predicted_original_unclipped,
            )

            clipped_to_prior_ratio = safe_ratio(
                clipped_original_rmse,
                prior_only_original_rmse,
            )
            unclipped_to_prior_ratio = safe_ratio(
                unclipped_original_rmse,
                prior_only_original_rmse,
            )

            metric_rows.append(
                {
                    "timestep": timestep,
                    "signal_coefficient": signal_coefficient,
                    "noise_coefficient": noise_coefficient,
                    "clipped_x0_pearson": calculate_pearson(
                        true_original,
                        predicted_original_clipped,
                    ),
                    "clipped_x0_model_domain_rmse": calculate_rmse(
                        true_model_domain,
                        predicted_model_clipped,
                    ),
                    "clipped_x0_normalized_rmse": calculate_rmse(
                        true_normalized,
                        predicted_normalized_clipped,
                    ),
                    "clipped_x0_original_rmse": clipped_original_rmse,
                    "clipped_x0_original_mae": calculate_mae(
                        true_original,
                        predicted_original_clipped,
                    ),
                    "unclipped_x0_pearson": calculate_pearson(
                        true_original,
                        predicted_original_unclipped,
                    ),
                    "unclipped_x0_model_domain_rmse": calculate_rmse(
                        true_model_domain,
                        predicted_model_unclipped,
                    ),
                    "unclipped_x0_normalized_rmse": calculate_rmse(
                        true_normalized,
                        predicted_normalized_unclipped,
                    ),
                    "unclipped_x0_original_rmse": (
                        unclipped_original_rmse
                    ),
                    "unclipped_x0_original_mae": calculate_mae(
                        true_original,
                        predicted_original_unclipped,
                    ),
                    "model_domain_clipping_fraction": (
                        model_domain_clipping_fraction
                    ),
                    # 保留旧列名，兼容此前查看表格的习惯。
                    "clipping_fraction": (
                        model_domain_clipping_fraction
                    ),
                    "clipped_full_normalized_outside_fraction": (
                        clipped_full_outside_fraction
                    ),
                    "unclipped_full_normalized_outside_fraction": (
                        unclipped_full_outside_fraction
                    ),
                    "model_noise_rmse": model_noise_rmse,
                    "model_noise_mae": model_noise_mae,
                    "zero_signal_baseline_noise_rmse": (
                        baseline_noise_rmse
                    ),
                    "zero_signal_baseline_noise_mae": (
                        baseline_noise_mae
                    ),
                    "model_to_baseline_rmse_ratio": (
                        model_to_noise_baseline_ratio
                    ),
                    "noise_improvement_percent": noise_improvement,
                    "prior_only_pearson": prior_only_pearson,
                    "prior_only_normalized_rmse": (
                        prior_only_normalized_rmse
                    ),
                    "prior_only_original_rmse": (
                        prior_only_original_rmse
                    ),
                    "prior_only_original_mae": prior_only_original_mae,
                    "clipped_model_to_prior_original_rmse_ratio": (
                        clipped_to_prior_ratio
                    ),
                    "clipped_prior_improvement_percent": (
                        improvement_percent(
                            clipped_original_rmse,
                            prior_only_original_rmse,
                        )
                    ),
                    "unclipped_model_to_prior_original_rmse_ratio": (
                        unclipped_to_prior_ratio
                    ),
                    "unclipped_prior_improvement_percent": (
                        improvement_percent(
                            unclipped_original_rmse,
                            prior_only_original_rmse,
                        )
                    ),
                }
            )

            clipped_recovered_spectra[timestep] = (
                predicted_original_clipped
            )
            unclipped_recovered_spectra[timestep] = (
                predicted_original_unclipped
            )
            predicted_noises[timestep] = predicted_noise
            baseline_noises[timestep] = baseline_noise

    output_directory = Path(arguments.output_directory).expanduser()

    if not output_directory.is_absolute():
        output_directory = resolve_project_path(
            runtime_configuration,
            output_directory,
        )

    output_directory = output_directory / arguments.model_source
    output_directory.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(metric_rows)

    clipped_spectra_data: dict[str, np.ndarray] = {
        "raman_shift": raman_axis,
        "true": true_original,
    }

    unclipped_spectra_data: dict[str, np.ndarray] = {
        "raman_shift": raman_axis,
        "true": true_original,
    }

    if prior_original is not None:
        clipped_spectra_data["prior_only"] = prior_original
        unclipped_spectra_data["prior_only"] = prior_original

    noise_data: dict[str, np.ndarray] = {
        "raman_shift": raman_axis,
        "true_noise": true_noise,
    }

    for timestep in timesteps:
        clipped_spectra_data[
            f"clipped_recovered_t{timestep}"
        ] = clipped_recovered_spectra[timestep]

        unclipped_spectra_data[
            f"unclipped_recovered_t{timestep}"
        ] = unclipped_recovered_spectra[timestep]

        noise_data[
            f"model_noise_t{timestep}"
        ] = predicted_noises[timestep]

        noise_data[
            f"baseline_noise_t{timestep}"
        ] = baseline_noises[timestep]

    clipped_spectra_frame = pd.DataFrame(clipped_spectra_data)
    unclipped_spectra_frame = pd.DataFrame(
        unclipped_spectra_data
    )
    noise_frame = pd.DataFrame(noise_data)

    diagnostic_information = pd.DataFrame(
        {
            "item": [
                "checkpoint",
                "model_source",
                "objective",
                "diagnostic_mode",
                "d3_4_diversity_constraints",
                "total_timesteps",
                "diagnostic_timesteps",
                "seed",
                "spectrum_index",
                "source_file",
                "spectrum_name",
            ],
            "value": [
                str(checkpoint_path),
                arguments.model_source,
                str(diffusion.objective),
                diagnostic_mode,
                "enabled"
                if diversity_constraints_enabled
                else "not_enabled",
                total_timesteps,
                ",".join(str(value) for value in timesteps),
                arguments.seed,
                spectrum_index,
                str(collection.relative_source_files[spectrum_index]),
                str(collection.spectrum_names[spectrum_index]),
            ],
        }
    )

    workbook_path = (
        output_directory / "timestep_recovery_detailed.xlsx"
    )

    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
    ) as writer:
        diagnostic_information.to_excel(
            writer,
            sheet_name="diagnostic_info",
            index=False,
        )
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

    # 当扩散步数为100时，绘制t=10、50、90。
    selected_timesteps = [
        min(
            int(round(fraction * total_timesteps)),
            total_timesteps - 1,
        )
        for fraction in (0.10, 0.50, 0.90)
    ]

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(16, 11),
        sharex=True,
    )

    for row_index, timestep in enumerate(selected_timesteps):
        clipped_axis = axes[row_index, 0]
        unclipped_axis = axes[row_index, 1]

        clipped_axis.plot(
            raman_axis,
            true_original,
            label="True",
            linewidth=1.2,
        )

        unclipped_axis.plot(
            raman_axis,
            true_original,
            label="True",
            linewidth=1.2,
        )

        if prior_original is not None:
            clipped_axis.plot(
                raman_axis,
                prior_original,
                label="Prior only",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
            )
            unclipped_axis.plot(
                raman_axis,
                prior_original,
                label="Prior only",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
            )

        clipped_axis.plot(
            raman_axis,
            clipped_recovered_spectra[timestep],
            label=f"Clipped recovery, t={timestep}",
            linewidth=1.0,
            alpha=0.85,
        )

        unclipped_axis.plot(
            raman_axis,
            unclipped_recovered_spectra[timestep],
            label=f"Unclipped recovery, t={timestep}",
            linewidth=1.0,
            alpha=0.85,
        )

        clipped_axis.set_title(
            f"Clipped x0 recovery, t={timestep}"
        )
        unclipped_axis.set_title(
            f"Unclipped x0 recovery, t={timestep}"
        )

        clipped_axis.set_ylabel("Intensity")
        unclipped_axis.set_ylabel("Intensity")
        clipped_axis.legend(fontsize=8)
        unclipped_axis.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[-1, 1].set_xlabel("Raman shift (cm$^{-1}$)")
    figure.tight_layout()

    recovery_figure_path = (
        output_directory / "timestep_recovery_detailed.png"
    )

    figure.savefig(
        recovery_figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    noise_figure, noise_axis = plt.subplots(figsize=(10, 6))

    noise_axis.plot(
        metrics["timestep"],
        metrics["model_noise_rmse"],
        marker="o",
        label="Model noise RMSE",
    )

    baseline_label = (
        "Zero-residual baseline RMSE"
        if prior_transformer is not None
        else "Zero-signal baseline RMSE"
    )

    noise_axis.plot(
        metrics["timestep"],
        metrics["zero_signal_baseline_noise_rmse"],
        marker="s",
        label=baseline_label,
    )

    noise_axis.set_xlabel("Timestep")
    noise_axis.set_ylabel("Noise RMSE")
    noise_axis.set_title(
        "Model noise prediction versus simple baseline"
    )
    noise_axis.grid(alpha=0.25)
    noise_axis.legend()
    noise_figure.tight_layout()

    noise_figure_path = (
        output_directory / "noise_prediction_comparison.png"
    )

    noise_figure.savefig(
        noise_figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(noise_figure)

    print("\n===== 不同时间步详细恢复诊断 =====")
    print(f"objective：{diffusion.objective}")
    print(f"扩散总步数：{total_timesteps}")
    print(f"诊断时间步：{timesteps}")
    print(f"模型来源：{arguments.model_source}")
    print(f"诊断数据域：{diagnostic_mode}")
    print(
        "D3.4多样性约束："
        "已按checkpoint状态注册"
        if diversity_constraints_enabled
        else "未启用"
    )
    print(f"诊断随机种子：{arguments.seed}")
    print(f"原始光谱索引：{spectrum_index}")
    print(
        "源文件："
        f"{collection.relative_source_files[spectrum_index]}"
    )
    print(
        "光谱名称："
        f"{collection.spectrum_names[spectrum_index]}"
    )

    if prior_transformer is not None:
        print("D2先验残差支持：已启用")
        print(
            "残差归一化方法："
            f"{prior_transformer.normalization_method}"
        )

        if (
            prior_transformer.normalization_method
            == "pointwise_mad_asinh"
        ):
            print(
                "逐波数尺度下限："
                f"{prior_transformer.pointwise_scale_floor:.8f}"
            )
            print(
                "逐波数尺度中位数："
                f"{np.median(prior_transformer.pointwise_scale):.8f}"
            )
            print(
                "标准化残差尺度："
                f"{prior_transformer.residual_scale:.8f}"
            )
        else:
            print(
                "残差缩放系数："
                f"{prior_transformer.residual_scale:.8f}"
            )

        print(
            "仅先验Pearson："
            f"{prior_only_pearson:.6f}"
        )
        print(
            "仅先验原始强度RMSE："
            f"{prior_only_original_rmse:.6f}"
        )
    else:
        print("D2先验残差支持：未启用，按D0/D1诊断")

    terminal_columns = [
        "timestep",
        "clipped_x0_pearson",
        "unclipped_x0_pearson",
        "unclipped_x0_normalized_rmse",
        "unclipped_x0_original_rmse",
        "clipping_fraction",
        "model_noise_rmse",
        "zero_signal_baseline_noise_rmse",
        "model_to_baseline_rmse_ratio",
        "noise_improvement_percent",
    ]

    if prior_transformer is not None:
        terminal_columns.extend(
            [
                "prior_only_original_rmse",
                "unclipped_model_to_prior_original_rmse_ratio",
                "unclipped_prior_improvement_percent",
            ]
        )

    print(
        "\n"
        + metrics[terminal_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n指标含义：")
    print(
        "1. clipping_fraction表示模型数据域中预测x0超出"
        "[-1,1]的点所占比例。"
    )
    print(
        "2. model_to_baseline_rmse_ratio < 1，说明模型的"
        "噪声预测优于简单零模型信号基线。"
    )
    print(
        "3. noise_improvement_percent > 0，说明模型的"
        "噪声预测相对简单基线有改善。"
    )

    if prior_transformer is not None:
        print(
            "4. D2中的零模型信号表示零缩放残差，"
            "对应完整光谱域中的仅参考先验。"
        )
        print(
            "5. unclipped_model_to_prior_original_rmse_ratio < 1，"
            "说明模型预测残差比仅使用先验更接近真实光谱。"
        )
        print(
            "6. unclipped_prior_improvement_percent > 0，"
            "说明模型残差为仅先验结果带来了正向改善。"
        )

    print(f"\n结果表：{workbook_path}")
    print(f"恢复图：{recovery_figure_path}")
    print(f"噪声比较图：{noise_figure_path}")


if __name__ == "__main__":
    main()
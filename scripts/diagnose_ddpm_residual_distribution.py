"""
诊断 D2.5 PCA 可变先验模型中的 DDPM residual 分布。

当前脚本专门用于：

    PCA reconstruction prior
        +
    robust_asinh residual normalization
        +
    unconditional 1D DDPM

核心目的不是继续增加 sampling post-processing，而是定位：

1. DDPM 在实际学习的 scaled residual 域中是否已经生成重尾；
2. robust_asinh inverse（sinh）是否进一步放大生成 residual 尾部；
3. 完全固定 PCA mean 后，仅由 DDPM residual 引起的完整光谱
    broad / non-peak 分布是否仍然异常；
4. broad 95% band 偏宽是否由少量 residual broad-shape tail 导致。

重要原则：

- 训练集仍严格使用 checkpoint 对应的数据划分；
- normalization_state 直接从 checkpoint 恢复，不重新拟合；
- prior_residual_state 直接从 checkpoint 恢复，不重新拟合；
- PCA prior 不随机采样；
- 不使用 variation_scale；
- 不使用 sampling_calibration；
- 不使用 Raman jitter；
- 不使用 feature peak limiter；
- 不执行全局强度反归一化；
- 所有诊断均在训练时统一模型 Raman shift 轴上完成；
- 特征峰位置从 checkpoint PCA mean 自动检测，不写死任何峰位。

运行示例：

CUDA_VISIBLE_DEVICES=1 \
python -m scripts.diagnose_ddpm_residual_distribution \
    --config config/ddpm_training.yaml \
    --checkpoint \
    outputs/experiments/d2_5_pca3_robust_asinh_s001_1200/checkpoints/best.pt \
    --model-source ema \
    --number 1000 \
    --batch-size 10 \
    --bootstrap-repeats 500 \
    --seed 2026 \
    --output-directory \
    outputs/diagnostics/d2_5_ddpm_residual_distribution_ema
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.dataset_splitter import split_spectrum_collection
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.prior_residual import PriorResidualTransformer
from src.spectrum_file_reader import read_spectrum_collection
from src.spectrum_length_adapter import SpectrumLengthAdapter


EPSILON = 1.0e-12


# ============================================================
# 基础工具
# ============================================================


def validate_2d_array(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    """检查二维有限数值数组。"""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name}必须为二维数组[N,L]，"
            f"实际形状为{array.shape}。"
        )

    if array.shape[0] < 2:
        raise ValueError(
            f"{name}至少需要2条光谱。"
        )

    if array.shape[1] < 2:
        raise ValueError(
            f"{name}至少需要2个Raman shift点。"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}包含NaN或无穷值。"
        )

    return array


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """安全计算比例。"""

    numerator = float(numerator)
    denominator = float(denominator)

    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or abs(denominator) <= EPSILON
    ):
        return float("nan")

    return float(
        numerator / denominator
    )


def percentile_key(
    value: float,
) -> str:
    """把百分位转换为适合CSV列名的文本。"""

    text = (
        f"{float(value):g}"
        .replace(".", "_")
    )

    return f"abs_p{text}"


def validate_axis(
    raman_shift: np.ndarray,
) -> np.ndarray:
    """检查模型 Raman shift 轴。"""

    axis = np.asarray(
        raman_shift,
        dtype=np.float64,
    ).reshape(-1)

    if axis.size < 2:
        raise ValueError(
            "Raman shift轴至少需要2个点。"
        )

    if not np.isfinite(axis).all():
        raise ValueError(
            "Raman shift轴包含NaN或无穷值。"
        )

    if not np.all(
        np.diff(axis) > 0.0
    ):
        raise ValueError(
            "Raman shift轴必须严格递增。"
        )

    return axis


def estimate_axis_spacing(
    raman_shift: np.ndarray,
) -> float:
    """估计模型轴的中位点间隔。"""

    differences = np.diff(
        raman_shift
    )

    spacing = float(
        np.median(differences)
    )

    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError(
            "无法得到有效的Raman shift点间隔。"
        )

    return spacing


def cm1_to_sigma_points(
    sigma_cm1: float,
    spacing_cm1: float,
) -> float:
    """将 cm^-1 Gaussian sigma 转换为数组点数。"""

    sigma_cm1 = float(sigma_cm1)

    if sigma_cm1 <= 0.0:
        raise ValueError(
            "Gaussian sigma_cm1必须大于0。"
        )

    sigma_points = (
        sigma_cm1
        / spacing_cm1
    )

    if sigma_points <= 0.0:
        raise ValueError(
            "Gaussian sigma_points必须大于0。"
        )

    return float(sigma_points)


# ============================================================
# checkpoint / model
# ============================================================


def load_prior_residual_transformer(
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
) -> PriorResidualTransformer:
    """恢复并严格检查 D2.5 prior residual transformer。"""

    prior_configuration = (
        checkpoint_configuration.get(
            "prior_residual",
            {},
        )
        or {}
    )

    if not isinstance(
        prior_configuration,
        dict,
    ):
        raise TypeError(
            "checkpoint中的prior_residual配置必须为字典。"
        )

    if not bool(
        prior_configuration.get(
            "enabled",
            False,
        )
    ):
        raise RuntimeError(
            "当前checkpoint没有启用prior_residual，"
            "本脚本只用于D2.5。"
        )

    state = metadata.get(
        "prior_residual_state"
    )

    if not isinstance(
        state,
        dict,
    ):
        raise RuntimeError(
            "checkpoint metadata缺少有效的"
            "prior_residual_state。"
        )

    transformer = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    if (
        transformer.prior_method
        != "pca_reconstruction"
    ):
        raise RuntimeError(
            "本脚本要求prior_method="
            "pca_reconstruction，实际为"
            f"{transformer.prior_method!r}。"
        )

    if (
        transformer.normalization_method
        != "robust_asinh"
    ):
        raise RuntimeError(
            "本脚本要求residual normalization="
            "robust_asinh，实际为"
            f"{transformer.normalization_method!r}。"
        )

    if transformer.pca_mean is None:
        raise RuntimeError(
            "checkpoint PCA状态缺少pca_mean。"
        )

    if transformer.pca_components is None:
        raise RuntimeError(
            "checkpoint PCA状态缺少pca_components。"
        )

    return transformer


def load_diffusion_model(
    *,
    checkpoint: dict[str, Any],
    checkpoint_configuration: dict[str, Any],
    length_adapter: SpectrumLengthAdapter,
    model_source: str,
    device: torch.device,
) -> torch.nn.Module:
    """构建扩散模型并加载 raw / EMA 权重。"""

    _, diffusion = build_diffusion_model(
        model_configuration=(
            checkpoint_configuration
        ),
        sequence_length=(
            length_adapter.padded_length
        ),
    )

    if model_source == "raw":
        model_state = checkpoint.get(
            "diffusion_state"
        )

        if not isinstance(
            model_state,
            dict,
        ):
            raise RuntimeError(
                "checkpoint缺少有效diffusion_state。"
            )

    elif model_source == "ema":
        ema_state = checkpoint.get(
            "ema_state"
        )

        if not isinstance(
            ema_state,
            dict,
        ):
            raise RuntimeError(
                "checkpoint缺少有效ema_state。"
            )

        model_state = ema_state.get(
            "ema_model"
        )

        if not isinstance(
            model_state,
            dict,
        ):
            raise RuntimeError(
                "checkpoint中的"
                "ema_state['ema_model']无效。"
            )

    else:
        raise ValueError(
            "model_source必须为raw或ema。"
        )

    diffusion.load_state_dict(
        model_state,
        strict=True,
    )

    diffusion = diffusion.to(
        device
    )

    diffusion.eval()

    return diffusion


# ============================================================
# 重建训练集
# ============================================================


def load_training_spectra(
    *,
    runtime_configuration: dict[str, Any],
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
    length_adapter: SpectrumLengthAdapter,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    严格按照 checkpoint 训练配置重新得到训练16条光谱。

    返回：
        model_axis
        train_normalized
        train_indices
    """

    data_configuration = (
        checkpoint_configuration.get(
            "data"
        )
    )

    if not isinstance(
        data_configuration,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少有效data配置。"
        )

    input_directory = resolve_project_path(
        runtime_configuration,
        data_configuration[
            "input_directory"
        ],
    )

    collection = (
        read_spectrum_collection(
            input_directory,
            data_configuration,
        )
    )

    project_configuration = (
        checkpoint_configuration.get(
            "project",
            {},
        )
        or {}
    )

    random_seed = int(
        project_configuration.get(
            "random_seed",
            2026,
        )
    )

    dataset_split = (
        split_spectrum_collection(
            collection=collection,
            data_config=(
                data_configuration
            ),
            random_seed=random_seed,
        )
    )

    train_indices = np.asarray(
        dataset_split.train.indices,
        dtype=np.int64,
    ).reshape(-1)

    if train_indices.size < 2:
        raise RuntimeError(
            "训练集光谱数量异常。"
        )

    train_spectra = [
        collection.spectra[
            int(index)
        ]
        for index in train_indices
    ]

    train_axes = [
        collection.raman_shifts[
            int(index)
        ]
        for index in train_indices
    ]

    train_on_model_axis = (
        length_adapter
        .interpolate_to_model_axis(
            train_spectra,
            train_axes,
        )
    )

    normalization_state = (
        metadata.get(
            "normalization_state"
        )
    )

    if not isinstance(
        normalization_state,
        dict,
    ):
        raise RuntimeError(
            "checkpoint metadata缺少"
            "normalization_state。"
        )

    normalizer = (
        GlobalMinMaxNormalizer
        .from_state_dict(
            normalization_state
        )
    )

    train_normalized = (
        normalizer.transform(
            train_on_model_axis
        )
    )

    train_normalized = (
        validate_2d_array(
            train_normalized,
            "train_normalized",
        )
    )

    model_axis = validate_axis(
        np.asarray(
            length_adapter.model_raman_shift,
            dtype=np.float64,
        )
    )

    if (
        train_normalized.shape[1]
        != model_axis.size
    ):
        raise RuntimeError(
            "训练光谱长度与checkpoint模型轴长度不一致。"
        )

    return (
        model_axis,
        train_normalized,
        train_indices,
    )


# ============================================================
# DDPM residual 生成
# ============================================================


@torch.inference_mode()
def sample_scaled_residuals(
    *,
    diffusion: torch.nn.Module,
    length_adapter: SpectrumLengthAdapter,
    number_of_spectra: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    直接从 diffusion.sample() 获得模型输出。

    对 D2.5 来说：
        diffusion.sample()
            ↓
        padded scaled residual
            ↓
        length_adapter.restore()
            ↓
        scaled residual

    此处故意不调用 PriorResidualTransformer.inverse_transform，
    因此不会加回任何 PCA prior。
    """

    if number_of_spectra <= 0:
        raise ValueError(
            "number_of_spectra必须大于0。"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size必须大于0。"
        )

    diffusion = diffusion.to(
        device
    )

    diffusion.eval()

    batches: list[np.ndarray] = []

    generated_count = 0

    while (
        generated_count
        < number_of_spectra
    ):
        current_batch_size = min(
            batch_size,
            (
                number_of_spectra
                - generated_count
            ),
        )

        generated = diffusion.sample(
            batch_size=(
                current_batch_size
            )
        )

        if generated.ndim != 3:
            raise RuntimeError(
                "diffusion.sample()应返回[B,C,L]，"
                f"实际为{tuple(generated.shape)}。"
            )

        if generated.shape[1] != 1:
            raise RuntimeError(
                "当前项目要求单通道光谱。"
            )

        generated_numpy = (
            generated[
                :,
                0,
                :,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        restored = (
            length_adapter.restore(
                generated_numpy
            )
        )

        restored = np.asarray(
            restored,
            dtype=np.float32,
        )

        if restored.shape != (
            current_batch_size,
            length_adapter.original_length,
        ):
            raise RuntimeError(
                "恢复后的DDPM residual形状异常："
                f"{restored.shape}。"
            )

        if not np.isfinite(
            restored
        ).all():
            raise RuntimeError(
                "DDPM生成scaled residual"
                "包含NaN或无穷值。"
            )

        batches.append(
            restored
        )

        generated_count += (
            current_batch_size
        )

        print(
            "DDPM residual生成进度："
            f"{generated_count}/"
            f"{number_of_spectra}"
        )

    result = np.concatenate(
        batches,
        axis=0,
    )

    return validate_2d_array(
        result,
        "generated_scaled_residual",
    )


# ============================================================
# 自动峰区检测
# ============================================================


def read_diagnostic_parameters(
    runtime_configuration: dict[str, Any],
) -> dict[str, float | int]:
    """
    优先读取当前 generation.sampling_calibration 中已经使用的
    自动峰检测尺度。

    注意：
    这里只借用“诊断参数”，绝不会执行 sampling calibration。
    """

    generation = (
        runtime_configuration.get(
            "generation",
            {},
        )
        or {}
    )

    calibration = (
        generation.get(
            "sampling_calibration",
            {},
        )
        or {}
    )

    non_peak = (
        calibration.get(
            "non_peak_noise",
            {},
        )
        or {}
    )

    return {
        "reference_smoothing_sigma_cm1": float(
            non_peak.get(
                "reference_smoothing_sigma_cm1",
                30.0,
            )
        ),
        "minimum_relative_prominence": float(
            non_peak.get(
                "minimum_relative_prominence",
                0.06,
            )
        ),
        "minimum_peak_distance_cm1": float(
            non_peak.get(
                "minimum_peak_distance_cm1",
                12.0,
            )
        ),
        "maximum_peak_count": int(
            non_peak.get(
                "maximum_peak_count",
                20,
            )
        ),
        "peak_protection_half_width_cm1": float(
            non_peak.get(
                "peak_protection_half_width_cm1",
                24.0,
            )
        ),
        "broad_sigma_cm1": float(
            non_peak.get(
                "broad_sigma_cm1",
                7.0,
            )
        ),
    }


def detect_peak_mask(
    *,
    raman_shift: np.ndarray,
    reference_spectrum: np.ndarray,
    reference_smoothing_sigma_cm1: float,
    minimum_relative_prominence: float,
    minimum_peak_distance_cm1: float,
    maximum_peak_count: int,
    peak_protection_half_width_cm1: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    从 PCA mean 自动检测峰区。

    不使用任何手工峰位。
    """

    axis = validate_axis(
        raman_shift
    )

    reference = np.asarray(
        reference_spectrum,
        dtype=np.float64,
    ).reshape(-1)

    if reference.size != axis.size:
        raise ValueError(
            "reference_spectrum与Raman shift长度不一致。"
        )

    spacing = estimate_axis_spacing(
        axis
    )

    smoothing_sigma_points = (
        cm1_to_sigma_points(
            reference_smoothing_sigma_cm1,
            spacing,
        )
    )

    baseline = gaussian_filter1d(
        reference,
        sigma=smoothing_sigma_points,
        mode="reflect",
    )

    peak_signal = (
        reference
        - baseline
    )

    positive_maximum = float(
        np.max(peak_signal)
    )

    if positive_maximum <= EPSILON:
        raise RuntimeError(
            "PCA mean中没有检测到有效正峰结构。"
        )

    minimum_prominence = (
        float(
            minimum_relative_prominence
        )
        * positive_maximum
    )

    minimum_distance_points = max(
        1,
        int(
            round(
                minimum_peak_distance_cm1
                / spacing
            )
        ),
    )

    peak_indices, properties = (
        find_peaks(
            peak_signal,
            prominence=(
                minimum_prominence
            ),
            distance=(
                minimum_distance_points
            ),
        )
    )

    if peak_indices.size == 0:
        raise RuntimeError(
            "自动峰检测没有找到任何特征峰。"
        )

    prominences = np.asarray(
        properties[
            "prominences"
        ],
        dtype=np.float64,
    )

    if (
        maximum_peak_count > 0
        and peak_indices.size
        > maximum_peak_count
    ):
        strongest_order = np.argsort(
            prominences
        )[::-1][
            :maximum_peak_count
        ]

        peak_indices = (
            peak_indices[
                strongest_order
            ]
        )

        prominences = (
            prominences[
                strongest_order
            ]
        )

    ascending_order = np.argsort(
        peak_indices
    )

    peak_indices = (
        peak_indices[
            ascending_order
        ]
    )

    prominences = (
        prominences[
            ascending_order
        ]
    )

    peak_mask = np.zeros(
        axis.size,
        dtype=bool,
    )

    records: list[
        dict[str, float | int]
    ] = []

    for (
        peak_number,
        (
            peak_index,
            prominence,
        ),
    ) in enumerate(
        zip(
            peak_indices,
            prominences,
            strict=True,
        ),
        start=1,
    ):
        center = float(
            axis[
                int(peak_index)
            ]
        )

        left = (
            center
            - peak_protection_half_width_cm1
        )

        right = (
            center
            + peak_protection_half_width_cm1
        )

        current_mask = (
            (axis >= left)
            & (axis <= right)
        )

        peak_mask |= (
            current_mask
        )

        records.append(
            {
                "peak_number": (
                    peak_number
                ),
                "peak_index": int(
                    peak_index
                ),
                "raman_shift_cm1": (
                    center
                ),
                "prominence_normalized": float(
                    prominence
                ),
                "window_left_cm1": float(
                    left
                ),
                "window_right_cm1": float(
                    right
                ),
            }
        )

    non_peak_mask = (
        ~peak_mask
    )

    if np.sum(
        non_peak_mask
    ) < 10:
        raise RuntimeError(
            "自动峰区覆盖过大，"
            "剩余非峰区点数不足。"
        )

    peak_table = pd.DataFrame(
        records
    )

    return (
        peak_mask,
        non_peak_mask,
        peak_table,
    )


# ============================================================
# residual 尾部统计
# ============================================================


def calculate_absolute_tail_statistics(
    values: np.ndarray,
) -> dict[str, float]:
    """计算 |residual| 的尾部分位数。"""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    flattened = np.abs(
        array.reshape(-1)
    )

    statistics: dict[
        str,
        float,
    ] = {}

    for percentile in (
        95.0,
        99.0,
        99.5,
        99.9,
    ):
        statistics[
            percentile_key(
                percentile
            )
        ] = float(
            np.percentile(
                flattened,
                percentile,
            )
        )

    statistics[
        "abs_max"
    ] = float(
        np.max(flattened)
    )

    statistics[
        "rms"
    ] = float(
        np.sqrt(
            np.mean(
                np.square(
                    array
                )
            )
        )
    )

    statistics[
        "mean_abs"
    ] = float(
        np.mean(
            np.abs(array)
        )
    )

    return statistics


def build_tail_comparison(
    *,
    domain: str,
    real_values: np.ndarray,
    generated_values: np.ndarray,
) -> list[dict[str, Any]]:
    """真实 vs 生成的尾部统计表。"""

    real_statistics = (
        calculate_absolute_tail_statistics(
            real_values
        )
    )

    generated_statistics = (
        calculate_absolute_tail_statistics(
            generated_values
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for metric in (
        "abs_p95",
        "abs_p99",
        "abs_p99_5",
        "abs_p99_9",
        "abs_max",
        "rms",
        "mean_abs",
    ):
        real_value = float(
            real_statistics[
                metric
            ]
        )

        generated_value = float(
            generated_statistics[
                metric
            ]
        )

        rows.append(
            {
                "domain": domain,
                "metric": metric,
                "real": real_value,
                "generated": (
                    generated_value
                ),
                "generated_to_real": (
                    safe_ratio(
                        generated_value,
                        real_value,
                    )
                ),
            }
        )

    return rows


# ============================================================
# broad / local decomposition
# ============================================================


def calculate_broad_component(
    *,
    values: np.ndarray,
    sigma_points: float,
) -> np.ndarray:
    """沿 Raman shift 轴提取 broad component。"""

    array = validate_2d_array(
        values,
        "broad_component_input",
    )

    broad = gaussian_filter1d(
        array,
        sigma=float(
            sigma_points
        ),
        axis=1,
        mode="reflect",
    )

    return np.asarray(
        broad,
        dtype=np.float64,
    )


def calculate_pointwise_dispersion(
    *,
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    计算点对点样本分布宽度。

    最终取指定 Raman 区域内各点指标的中位数。
    这样不会让少数单点直接主导整体统计。
    """

    array = validate_2d_array(
        values,
        "pointwise_dispersion_input",
    )

    boolean_mask = np.asarray(
        mask,
        dtype=bool,
    ).reshape(-1)

    if (
        boolean_mask.size
        != array.shape[1]
    ):
        raise ValueError(
            "mask长度与光谱长度不一致。"
        )

    if not np.any(
        boolean_mask
    ):
        raise ValueError(
            "mask中没有有效点。"
        )

    q025 = np.percentile(
        array,
        2.5,
        axis=0,
    )

    q25 = np.percentile(
        array,
        25.0,
        axis=0,
    )

    q75 = np.percentile(
        array,
        75.0,
        axis=0,
    )

    q975 = np.percentile(
        array,
        97.5,
        axis=0,
    )

    pointwise_std = np.std(
        array,
        axis=0,
        ddof=1,
    )

    band95 = (
        q975
        - q025
    )

    iqr = (
        q75
        - q25
    )

    return {
        "band95": float(
            np.median(
                band95[
                    boolean_mask
                ]
            )
        ),
        "iqr": float(
            np.median(
                iqr[
                    boolean_mask
                ]
            )
        ),
        "std": float(
            np.median(
                pointwise_std[
                    boolean_mask
                ]
            )
        ),
        "band95_mean": float(
            np.mean(
                band95[
                    boolean_mask
                ]
            )
        ),
        "std_mean": float(
            np.mean(
                pointwise_std[
                    boolean_mask
                ]
            )
        ),
    }


def calculate_component_metrics(
    *,
    values: np.ndarray,
    non_peak_mask: np.ndarray,
    broad_sigma_points: float,
) -> dict[str, float]:
    """
    把完整数组拆成：

        full
        broad
        local = full - broad

    然后只在 non-peak 区域统计。
    """

    array = validate_2d_array(
        values,
        "component_metric_input",
    )

    broad = (
        calculate_broad_component(
            values=array,
            sigma_points=(
                broad_sigma_points
            ),
        )
    )

    local = (
        array
        - broad
    )

    full_metrics = (
        calculate_pointwise_dispersion(
            values=array,
            mask=non_peak_mask,
        )
    )

    broad_metrics = (
        calculate_pointwise_dispersion(
            values=broad,
            mask=non_peak_mask,
        )
    )

    local_metrics = (
        calculate_pointwise_dispersion(
            values=local,
            mask=non_peak_mask,
        )
    )

    result: dict[
        str,
        float,
    ] = {}

    for (
        prefix,
        metrics,
    ) in (
        (
            "nonpeak",
            full_metrics,
        ),
        (
            "broad",
            broad_metrics,
        ),
        (
            "local",
            local_metrics,
        ),
    ):
        for (
            metric_name,
            metric_value,
        ) in metrics.items():
            result[
                f"{prefix}_{metric_name}"
            ] = float(
                metric_value
            )

    return result


def build_component_metric_table(
    *,
    datasets: dict[str, np.ndarray],
    non_peak_mask: np.ndarray,
    broad_sigma_points: float,
) -> pd.DataFrame:
    """为多个数据集统一计算 broad/nonpeak/local 指标。"""

    rows: list[
        dict[str, Any]
    ] = []

    for (
        dataset_name,
        values,
    ) in datasets.items():
        metrics = (
            calculate_component_metrics(
                values=values,
                non_peak_mask=(
                    non_peak_mask
                ),
                broad_sigma_points=(
                    broad_sigma_points
                ),
            )
        )

        row: dict[
            str,
            Any,
        ] = {
            "dataset": (
                dataset_name
            ),
            "number_of_spectra": int(
                values.shape[0]
            ),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# matched-size bootstrap
# ============================================================


BOOTSTRAP_METRICS = (
    "nonpeak_band95",
    "nonpeak_iqr",
    "nonpeak_std",
    "broad_band95",
    "broad_iqr",
    "broad_std",
    "local_band95",
    "local_iqr",
    "local_std",
)


def matched_size_bootstrap(
    *,
    comparison_name: str,
    reference_values: np.ndarray,
    generated_values: np.ndarray,
    non_peak_mask: np.ndarray,
    broad_sigma_points: float,
    number_of_repeats: int,
    random_generator: np.random.Generator,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    真实N条 vs 生成中随机N条。

    每次重新从 generated 中抽取与 reference 相同数量的光谱，
    避免 16 real vs 1000 generated 直接比较造成样本量偏差。
    """

    reference = validate_2d_array(
        reference_values,
        "bootstrap_reference",
    )

    generated = validate_2d_array(
        generated_values,
        "bootstrap_generated",
    )

    if (
        reference.shape[1]
        != generated.shape[1]
    ):
        raise ValueError(
            "bootstrap真实与生成光谱长度不一致。"
        )

    if number_of_repeats <= 0:
        raise ValueError(
            "bootstrap repeats必须大于0。"
        )

    reference_count = int(
        reference.shape[0]
    )

    generated_count = int(
        generated.shape[0]
    )

    reference_metrics = (
        calculate_component_metrics(
            values=reference,
            non_peak_mask=(
                non_peak_mask
            ),
            broad_sigma_points=(
                broad_sigma_points
            ),
        )
    )

    replace = (
        generated_count
        < reference_count
    )

    rows: list[
        dict[str, Any]
    ] = []

    for iteration in range(
        number_of_repeats
    ):
        selected_indices = (
            random_generator.choice(
                generated_count,
                size=reference_count,
                replace=replace,
            )
        )

        selected_generated = (
            generated[
                selected_indices
            ]
        )

        generated_metrics = (
            calculate_component_metrics(
                values=(
                    selected_generated
                ),
                non_peak_mask=(
                    non_peak_mask
                ),
                broad_sigma_points=(
                    broad_sigma_points
                ),
            )
        )

        row: dict[
            str,
            Any,
        ] = {
            "comparison": (
                comparison_name
            ),
            "iteration": (
                iteration + 1
            ),
        }

        for metric in (
            BOOTSTRAP_METRICS
        ):
            reference_value = float(
                reference_metrics[
                    metric
                ]
            )

            generated_value = float(
                generated_metrics[
                    metric
                ]
            )

            row[
                f"{metric}_reference"
            ] = (
                reference_value
            )

            row[
                f"{metric}_generated"
            ] = (
                generated_value
            )

            row[
                f"{metric}_ratio"
            ] = safe_ratio(
                generated_value,
                reference_value,
            )

        rows.append(
            row
        )

    bootstrap_table = pd.DataFrame(
        rows
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    for metric in (
        BOOTSTRAP_METRICS
    ):
        ratio_values = np.asarray(
            bootstrap_table[
                f"{metric}_ratio"
            ],
            dtype=np.float64,
        )

        finite = ratio_values[
            np.isfinite(
                ratio_values
            )
        ]

        if finite.size == 0:
            summary = {
                "comparison": (
                    comparison_name
                ),
                "metric": metric,
                "mean_ratio": float(
                    "nan"
                ),
                "median_ratio": float(
                    "nan"
                ),
                "ratio_p2_5": float(
                    "nan"
                ),
                "ratio_p97_5": float(
                    "nan"
                ),
            }
        else:
            summary = {
                "comparison": (
                    comparison_name
                ),
                "metric": metric,
                "mean_ratio": float(
                    np.mean(finite)
                ),
                "median_ratio": float(
                    np.median(finite)
                ),
                "ratio_p2_5": float(
                    np.percentile(
                        finite,
                        2.5,
                    )
                ),
                "ratio_p97_5": float(
                    np.percentile(
                        finite,
                        97.5,
                    )
                ),
            }

        summary_rows.append(
            summary
        )

    summary_table = pd.DataFrame(
        summary_rows
    )

    return (
        bootstrap_table,
        summary_table,
    )


# ============================================================
# 导出
# ============================================================


def export_matrix_csv(
    *,
    path: Path,
    raman_shift: np.ndarray,
    values: np.ndarray,
    prefix: str,
    maximum_spectra: int | None = None,
) -> None:
    """按 Raman shift + 多条光谱列的形式导出预览CSV。"""

    array = validate_2d_array(
        values,
        prefix,
    )

    if maximum_spectra is None:
        selected = array
    else:
        selected = array[
            :maximum_spectra
        ]

    data: dict[
        str,
        np.ndarray,
    ] = {
        "raman_shift_cm1": (
            np.asarray(
                raman_shift,
                dtype=np.float64,
            )
        )
    }

    for index in range(
        selected.shape[0]
    ):
        data[
            f"{prefix}_{index + 1:04d}"
        ] = selected[
            index
        ]

    pd.DataFrame(
        data
    ).to_csv(
        path,
        index=False,
    )


# ============================================================
# main
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "诊断D2.5 EMA DDPM residual重尾、"
            "robust_asinh inverse amplification以及"
            "固定PCA mean后的broad/nonpeak分布。"
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
        "--model-source",
        choices=(
            "raw",
            "ema",
        ),
        default="ema",
        help="使用raw或EMA模型，默认EMA。",
    )

    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="生成DDPM residual数量，默认1000。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="DDPM residual生成batch size。",
    )

    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=500,
        help="matched-size bootstrap次数。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="诊断随机种子。",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备，例如cuda、cuda:0或cpu。",
    )

    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help=(
            "仅在当前诊断进程中覆盖采样步数；"
            "不会修改YAML或checkpoint。"
            "小于diffusion_steps时走DDIM；"
            "等于diffusion_steps时走完整DDPM。"
        ),
    )

    parser.add_argument(
        "--output-directory",
        default=(
            "outputs/diagnostics/"
            "d2_5_ddpm_residual_distribution_ema"
        ),
        help="诊断输出目录。",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.number < 16:
        raise ValueError(
            "--number至少建议为16。"
        )

    if arguments.batch_size <= 0:
        raise ValueError(
            "--batch-size必须大于0。"
        )

    if (
        arguments.bootstrap_repeats
        <= 0
    ):
        raise ValueError(
            "--bootstrap-repeats必须大于0。"
        )

    # --------------------------------------------------------
    # 固定随机种子
    # --------------------------------------------------------

    np.random.seed(
        arguments.seed
    )

    torch.manual_seed(
        arguments.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            arguments.seed
        )

    bootstrap_generator = (
        np.random.default_rng(
            arguments.seed + 10001
        )
    )

    # --------------------------------------------------------
    # 配置和checkpoint
    # --------------------------------------------------------

    runtime_configuration = (
        load_configuration(
            arguments.config
        )
    )

    checkpoint_path = Path(
        arguments.checkpoint
    ).expanduser()

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            resolve_project_path(
                runtime_configuration,
                checkpoint_path,
            )
        )

    checkpoint = (
        load_checkpoint_file(
            checkpoint_path,
            map_location="cpu",
        )
    )

    checkpoint_configuration = (
        checkpoint.get(
            "configuration"
        )
    )

    metadata = checkpoint.get(
        "metadata"
    )

    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少有效configuration。"
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "checkpoint缺少有效metadata。"
        )

    diffusion_configuration = (
        checkpoint_configuration.get(
            "diffusion",
            {},
        )
        or {}
    )

    objective = str(
        diffusion_configuration.get(
            "objective",
            checkpoint_configuration.get(
                "model",
                {},
            ).get(
                "objective",
                "",
            ),
        )
    ).strip().lower()

    if objective != "pred_x0":
        raise RuntimeError(
            "本轮D2.5诊断要求objective=pred_x0，"
            f"checkpoint实际为{objective!r}。"
        )

    # --------------------------------------------------------
    # 仅用于本次诊断的 sampling override。
    #
    # 重要：
    # 1. 不修改 checkpoint；
    # 2. 不修改 config/ddpm_training.yaml；
    # 3. 不改变训练 timesteps；
    # 4. 只改变生成端 sampling_timesteps。
    #
    # 对当前 D2.5：
    #
    # diffusion_steps = 100
    #
    # sampling_steps = 50
    #     -> DDIM50
    #
    # sampling_steps = 100
    #     -> sampling_timesteps == timesteps
    #     -> 完整 DDPM p_sample_loop
    # --------------------------------------------------------

    training_diffusion_steps = int(
        diffusion_configuration.get(
            "diffusion_steps",
            diffusion_configuration.get(
                "diffusion_timesteps",
                0,
            ),
        )
    )

    if training_diffusion_steps <= 0:
        raise RuntimeError(
            "checkpoint中的diffusion_steps无效。"
        )

    checkpoint_sampling_steps = int(
        diffusion_configuration.get(
            "sampling_steps",
            diffusion_configuration.get(
                "sampling_timesteps",
                training_diffusion_steps,
            ),
        )
    )

    if arguments.sampling_steps is None:
        requested_sampling_steps = (
            checkpoint_sampling_steps
        )
    else:
        requested_sampling_steps = int(
            arguments.sampling_steps
        )

    if requested_sampling_steps <= 0:
        raise ValueError(
            "--sampling-steps必须大于0。"
        )

    if (
        requested_sampling_steps
        > training_diffusion_steps
    ):
        raise ValueError(
            "--sampling-steps不能大于训练扩散步数："
            f"{requested_sampling_steps} > "
            f"{training_diffusion_steps}。"
        )

    diagnostic_model_configuration = (
        copy.deepcopy(
            checkpoint_configuration
        )
    )

    diagnostic_diffusion_configuration = (
        diagnostic_model_configuration.setdefault(
            "diffusion",
            {},
        )
    )

    # 同时写两个兼容字段，避免历史checkpoint中别名优先级不同。
    diagnostic_diffusion_configuration[
        "sampling_steps"
    ] = requested_sampling_steps

    diagnostic_diffusion_configuration[
        "sampling_timesteps"
    ] = requested_sampling_steps

    length_adapter = (
        SpectrumLengthAdapter
        .from_metadata(
            metadata
        )
    )

    prior_transformer = (
        load_prior_residual_transformer(
            checkpoint_configuration,
            metadata,
        )
    )

    # --------------------------------------------------------
    # 恢复训练16条
    # --------------------------------------------------------

    (
        model_axis,
        train_normalized,
        train_indices,
    ) = load_training_spectra(
        runtime_configuration=(
            runtime_configuration
        ),
        checkpoint_configuration=(
            checkpoint_configuration
        ),
        metadata=metadata,
        length_adapter=(
            length_adapter
        ),
    )

    if (
        train_normalized.shape[0]
        != 16
    ):
        print(
            "警告：当前重建训练集不是16条，"
            f"而是{train_normalized.shape[0]}条。"
            "脚本仍继续，但请核对checkpoint对应数据。"
        )

    # --------------------------------------------------------
    # 真实训练 residual
    # --------------------------------------------------------

    true_reference_priors = (
        prior_transformer
        .reference_priors_for_spectra(
            train_normalized
        )
    )

    true_scaled_residual = (
        prior_transformer.transform(
            train_normalized,
            reference_priors=(
                true_reference_priors
            ),
        )
    )

    true_raw_residual_direct = (
        train_normalized
        - true_reference_priors
    )

    zero_reference_true = np.zeros_like(
        true_scaled_residual,
        dtype=np.float32,
    )

    true_raw_residual_inverse = (
        prior_transformer
        .inverse_transform(
            true_scaled_residual,
            reference_priors=(
                zero_reference_true
            ),
        )
    )

    inverse_roundtrip_error = float(
        np.max(
            np.abs(
                true_raw_residual_direct
                - true_raw_residual_inverse
            )
        )
    )

    if inverse_roundtrip_error > 1.0e-5:
        raise RuntimeError(
            "真实residual经过transform/inverse后"
            "回环误差过大："
            f"{inverse_roundtrip_error:.8g}。"
        )

    true_raw_residual = np.asarray(
        true_raw_residual_inverse,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 加载EMA / raw DDPM
    # --------------------------------------------------------

    device = torch.device(
        arguments.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "PyTorch没有检测到CUDA。"
        )

    diffusion = load_diffusion_model(
        checkpoint=checkpoint,
        checkpoint_configuration=(
            diagnostic_model_configuration
        ),
        length_adapter=(
            length_adapter
        ),
        model_source=(
            arguments.model_source
        ),
        device=device,
    )

    # --------------------------------------------------------
    # 强制验证实际构建出来的sampling模式。
    # 不能只相信命令行参数名称。
    # --------------------------------------------------------

    actual_num_timesteps = int(
        getattr(
            diffusion,
            "num_timesteps",
            -1,
        )
    )

    actual_sampling_timesteps = int(
        getattr(
            diffusion,
            "sampling_timesteps",
            -1,
        )
    )

    actual_is_ddim_sampling = bool(
        getattr(
            diffusion,
            "is_ddim_sampling",
            False,
        )
    )

    if (
        actual_num_timesteps
        != training_diffusion_steps
    ):
        raise RuntimeError(
            "实际扩散步数与checkpoint不一致："
            f"{actual_num_timesteps} != "
            f"{training_diffusion_steps}。"
        )

    if (
        actual_sampling_timesteps
        != requested_sampling_steps
    ):
        raise RuntimeError(
            "实际采样步数与请求值不一致："
            f"{actual_sampling_timesteps} != "
            f"{requested_sampling_steps}。"
        )

    expected_is_ddim = (
        requested_sampling_steps
        < training_diffusion_steps
    )

    if (
        actual_is_ddim_sampling
        != expected_is_ddim
    ):
        raise RuntimeError(
            "扩散库实际sampling模式与预期不一致。"
        )

    sampling_mode = (
        "DDIM"
        if actual_is_ddim_sampling
        else "DDPM"
    )

    print()
    print(
        "===== 本次sampling模式确认 ====="
    )
    print(
        "训练扩散步数："
        f"{actual_num_timesteps}"
    )
    print(
        "实际采样步数："
        f"{actual_sampling_timesteps}"
    )
    print(
        "实际采样器："
        f"{sampling_mode}"
    )
    print(
        "checkpoint原采样步数："
        f"{checkpoint_sampling_steps}"
    )
    print(
        "YAML/checkpoint：未修改"
    )
    print()

    # --------------------------------------------------------
    # 直接生成 scaled residual
    # --------------------------------------------------------

    generated_scaled_residual = (
        sample_scaled_residuals(
            diffusion=diffusion,
            length_adapter=(
                length_adapter
            ),
            number_of_spectra=(
                arguments.number
            ),
            batch_size=(
                arguments.batch_size
            ),
            device=device,
        )
    )

    # --------------------------------------------------------
    # robust_asinh inverse
    #
    # 给 inverse_transform 一个全0 reference prior，
    # 这样得到的就是纯 raw residual，
    # 不会加回 PCA prior。
    # --------------------------------------------------------

    zero_reference_generated = (
        np.zeros_like(
            generated_scaled_residual,
            dtype=np.float32,
        )
    )

    generated_raw_residual = (
        prior_transformer
        .inverse_transform(
            generated_scaled_residual,
            reference_priors=(
                zero_reference_generated
            ),
        )
    )

    generated_raw_residual = (
        validate_2d_array(
            generated_raw_residual,
            "generated_raw_residual",
        )
    )

    # --------------------------------------------------------
    # 固定 PCA mean
    # --------------------------------------------------------

    pca_mean = np.asarray(
        prior_transformer.pca_mean,
        dtype=np.float64,
    ).reshape(-1)

    if pca_mean.size != model_axis.size:
        raise RuntimeError(
            "PCA mean长度与模型Raman轴不一致。"
        )

    generated_fixed_mean_spectra = (
        pca_mean[
            np.newaxis,
            :
        ]
        + generated_raw_residual
    )

    # 辅助对照：
    # 真实训练 residual 也统一加到同一个 PCA mean 上，
    # 这样可以把PCA score真实变化本身排除掉。
    true_fixed_mean_spectra = (
        pca_mean[
            np.newaxis,
            :
        ]
        + true_raw_residual
    )

    # --------------------------------------------------------
    # 自动峰区
    # --------------------------------------------------------

    diagnostic_parameters = (
        read_diagnostic_parameters(
            runtime_configuration
        )
    )

    (
        peak_mask,
        non_peak_mask,
        peak_table,
    ) = detect_peak_mask(
        raman_shift=model_axis,
        reference_spectrum=(
            pca_mean
        ),
        reference_smoothing_sigma_cm1=float(
            diagnostic_parameters[
                "reference_smoothing_sigma_cm1"
            ]
        ),
        minimum_relative_prominence=float(
            diagnostic_parameters[
                "minimum_relative_prominence"
            ]
        ),
        minimum_peak_distance_cm1=float(
            diagnostic_parameters[
                "minimum_peak_distance_cm1"
            ]
        ),
        maximum_peak_count=int(
            diagnostic_parameters[
                "maximum_peak_count"
            ]
        ),
        peak_protection_half_width_cm1=float(
            diagnostic_parameters[
                "peak_protection_half_width_cm1"
            ]
        ),
    )

    spacing_cm1 = estimate_axis_spacing(
        model_axis
    )

    broad_sigma_points = (
        cm1_to_sigma_points(
            float(
                diagnostic_parameters[
                    "broad_sigma_cm1"
                ]
            ),
            spacing_cm1,
        )
    )

    # --------------------------------------------------------
    # ① scaled residual tail
    # ② inverse raw residual tail
    # --------------------------------------------------------

    tail_rows: list[
        dict[str, Any]
    ] = []

    tail_rows.extend(
        build_tail_comparison(
            domain=(
                "scaled_residual"
            ),
            real_values=(
                true_scaled_residual
            ),
            generated_values=(
                generated_scaled_residual
            ),
        )
    )

    tail_rows.extend(
        build_tail_comparison(
            domain=(
                "inverse_raw_residual"
            ),
            real_values=(
                true_raw_residual
            ),
            generated_values=(
                generated_raw_residual
            ),
        )
    )

    # --------------------------------------------------------
    # residual broad component尾部
    # --------------------------------------------------------

    true_scaled_broad = (
        calculate_broad_component(
            values=(
                true_scaled_residual
            ),
            sigma_points=(
                broad_sigma_points
            ),
        )
    )

    generated_scaled_broad = (
        calculate_broad_component(
            values=(
                generated_scaled_residual
            ),
            sigma_points=(
                broad_sigma_points
            ),
        )
    )

    true_raw_broad = (
        calculate_broad_component(
            values=(
                true_raw_residual
            ),
            sigma_points=(
                broad_sigma_points
            ),
        )
    )

    generated_raw_broad = (
        calculate_broad_component(
            values=(
                generated_raw_residual
            ),
            sigma_points=(
                broad_sigma_points
            ),
        )
    )

    tail_rows.extend(
        build_tail_comparison(
            domain=(
                "scaled_residual_broad_nonpeak"
            ),
            real_values=(
                true_scaled_broad[
                    :,
                    non_peak_mask,
                ]
            ),
            generated_values=(
                generated_scaled_broad[
                    :,
                    non_peak_mask,
                ]
            ),
        )
    )

    tail_rows.extend(
        build_tail_comparison(
            domain=(
                "inverse_raw_residual_broad_nonpeak"
            ),
            real_values=(
                true_raw_broad[
                    :,
                    non_peak_mask,
                ]
            ),
            generated_values=(
                generated_raw_broad[
                    :,
                    non_peak_mask,
                ]
            ),
        )
    )

    tail_table = pd.DataFrame(
        tail_rows
    )

    # --------------------------------------------------------
    # inverse amplification诊断
    # --------------------------------------------------------

    ratio_comparison_rows: list[
        dict[str, Any]
    ] = []

    for metric in (
        "abs_p95",
        "abs_p99",
        "abs_p99_5",
        "abs_p99_9",
        "abs_max",
    ):
        scaled_row = tail_table[
            (
                tail_table[
                    "domain"
                ]
                == "scaled_residual"
            )
            & (
                tail_table[
                    "metric"
                ]
                == metric
            )
        ]

        inverse_row = tail_table[
            (
                tail_table[
                    "domain"
                ]
                == "inverse_raw_residual"
            )
            & (
                tail_table[
                    "metric"
                ]
                == metric
            )
        ]

        if (
            len(scaled_row) != 1
            or len(inverse_row) != 1
        ):
            raise RuntimeError(
                "tail comparison内部表结构异常。"
            )

        scaled_ratio = float(
            scaled_row.iloc[
                0
            ][
                "generated_to_real"
            ]
        )

        inverse_ratio = float(
            inverse_row.iloc[
                0
            ][
                "generated_to_real"
            ]
        )

        ratio_comparison_rows.append(
            {
                "metric": metric,
                "scaled_generated_to_real": (
                    scaled_ratio
                ),
                "inverse_generated_to_real": (
                    inverse_ratio
                ),
                "inverse_amplification_of_ratio": (
                    safe_ratio(
                        inverse_ratio,
                        scaled_ratio,
                    )
                ),
            }
        )

    inverse_amplification_table = (
        pd.DataFrame(
            ratio_comparison_rows
        )
    )

    # --------------------------------------------------------
    # full / broad / local 点对点分布
    # --------------------------------------------------------

    component_table = (
        build_component_metric_table(
            datasets={
                "real_training_full": (
                    train_normalized
                ),
                "real_fixed_pca_mean_plus_true_residual": (
                    true_fixed_mean_spectra
                ),
                "generated_fixed_pca_mean_plus_ddpm_residual": (
                    generated_fixed_mean_spectra
                ),
                "true_raw_residual": (
                    true_raw_residual
                ),
                "generated_raw_residual": (
                    generated_raw_residual
                ),
            },
            non_peak_mask=(
                non_peak_mask
            ),
            broad_sigma_points=(
                broad_sigma_points
            ),
        )
    )

    # --------------------------------------------------------
    # ③ matched bootstrap
    #
    # A：
    # 实际16条完整真实训练谱
    # vs
    # 固定PCA mean + DDPM residual
    #
    # B：
    # 固定PCA mean + 真实residual
    # vs
    # 固定PCA mean + DDPM residual
    #
    # B能够进一步剥离真实PCA score变化。
    # --------------------------------------------------------

    (
        bootstrap_full_table,
        bootstrap_full_summary,
    ) = matched_size_bootstrap(
        comparison_name=(
            "real_full_vs_fixed_pca_mean_generated"
        ),
        reference_values=(
            train_normalized
        ),
        generated_values=(
            generated_fixed_mean_spectra
        ),
        non_peak_mask=(
            non_peak_mask
        ),
        broad_sigma_points=(
            broad_sigma_points
        ),
        number_of_repeats=(
            arguments.bootstrap_repeats
        ),
        random_generator=(
            bootstrap_generator
        ),
    )

    (
        bootstrap_residual_table,
        bootstrap_residual_summary,
    ) = matched_size_bootstrap(
        comparison_name=(
            "fixed_pca_mean_true_residual_vs_generated_residual"
        ),
        reference_values=(
            true_fixed_mean_spectra
        ),
        generated_values=(
            generated_fixed_mean_spectra
        ),
        non_peak_mask=(
            non_peak_mask
        ),
        broad_sigma_points=(
            broad_sigma_points
        ),
        number_of_repeats=(
            arguments.bootstrap_repeats
        ),
        random_generator=(
            bootstrap_generator
        ),
    )

    bootstrap_table = pd.concat(
        [
            bootstrap_full_table,
            bootstrap_residual_table,
        ],
        ignore_index=True,
    )

    bootstrap_summary = pd.concat(
        [
            bootstrap_full_summary,
            bootstrap_residual_summary,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    output_directory = Path(
        arguments.output_directory
    ).expanduser()

    if not output_directory.is_absolute():
        output_directory = (
            resolve_project_path(
                runtime_configuration,
                output_directory,
            )
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 导出CSV
    # --------------------------------------------------------

    tail_table.to_csv(
        output_directory
        / "residual_tail_statistics.csv",
        index=False,
    )

    inverse_amplification_table.to_csv(
        output_directory
        / "inverse_amplification_statistics.csv",
        index=False,
    )

    component_table.to_csv(
        output_directory
        / "component_distribution_statistics.csv",
        index=False,
    )

    bootstrap_table.to_csv(
        output_directory
        / "matched_bootstrap_all_iterations.csv",
        index=False,
    )

    bootstrap_summary.to_csv(
        output_directory
        / "matched_bootstrap_summary.csv",
        index=False,
    )

    peak_table.to_csv(
        output_directory
        / "automatically_detected_peak_regions.csv",
        index=False,
    )

    split_table = pd.DataFrame(
        {
            "train_order": np.arange(
                1,
                train_indices.size + 1,
            ),
            "original_spectrum_index": (
                train_indices
            ),
        }
    )

    split_table.to_csv(
        output_directory
        / "training_indices.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 保存所有数值，便于后面继续做频域/PCA/MMD诊断
    # --------------------------------------------------------

    np.savez_compressed(
        output_directory
        / "ddpm_residual_diagnostic_arrays.npz",
        raman_shift_cm1=(
            model_axis.astype(
                np.float64
            )
        ),
        peak_mask=peak_mask,
        non_peak_mask=(
            non_peak_mask
        ),
        pca_mean=(
            pca_mean.astype(
                np.float32
            )
        ),
        real_training_normalized=(
            np.asarray(
                train_normalized,
                dtype=np.float32,
            )
        ),
        real_reference_priors=(
            np.asarray(
                true_reference_priors,
                dtype=np.float32,
            )
        ),
        true_scaled_residual=(
            np.asarray(
                true_scaled_residual,
                dtype=np.float32,
            )
        ),
        generated_scaled_residual=(
            np.asarray(
                generated_scaled_residual,
                dtype=np.float32,
            )
        ),
        true_raw_residual=(
            np.asarray(
                true_raw_residual,
                dtype=np.float32,
            )
        ),
        generated_raw_residual=(
            np.asarray(
                generated_raw_residual,
                dtype=np.float32,
            )
        ),
        true_fixed_mean_spectra=(
            np.asarray(
                true_fixed_mean_spectra,
                dtype=np.float32,
            )
        ),
        generated_fixed_mean_spectra=(
            np.asarray(
                generated_fixed_mean_spectra,
                dtype=np.float32,
            )
        ),
    )

    # --------------------------------------------------------
    # 导出可直接画图的CSV预览
    # generated只导出前100条，避免CSV过大；
    # 完整1000条全部保存在npz中。
    # --------------------------------------------------------

    export_matrix_csv(
        path=(
            output_directory
            / "real_training_normalized.csv"
        ),
        raman_shift=model_axis,
        values=train_normalized,
        prefix="real",
    )

    export_matrix_csv(
        path=(
            output_directory
            / "true_scaled_residual.csv"
        ),
        raman_shift=model_axis,
        values=(
            true_scaled_residual
        ),
        prefix="true_scaled_residual",
    )

    export_matrix_csv(
        path=(
            output_directory
            / "generated_scaled_residual_first100.csv"
        ),
        raman_shift=model_axis,
        values=(
            generated_scaled_residual
        ),
        prefix="generated_scaled_residual",
        maximum_spectra=100,
    )

    export_matrix_csv(
        path=(
            output_directory
            / "generated_raw_residual_first100.csv"
        ),
        raman_shift=model_axis,
        values=(
            generated_raw_residual
        ),
        prefix="generated_raw_residual",
        maximum_spectra=100,
    )

    export_matrix_csv(
        path=(
            output_directory
            / "fixed_pca_mean_generated_first100.csv"
        ),
        raman_shift=model_axis,
        values=(
            generated_fixed_mean_spectra
        ),
        prefix="fixed_mean_generated",
        maximum_spectra=100,
    )

    # --------------------------------------------------------
    # 终端结果
    # --------------------------------------------------------

    print()
    print(
        "============================================"
    )
    print(
        "D2.5 DDPM residual decomposition完成"
    )
    print(
        "============================================"
    )

    print(
        f"checkpoint：{checkpoint_path}"
    )

    print(
        "checkpoint step："
        f"{int(checkpoint.get('step', 0))}"
    )

    print(
        "模型来源："
        f"{arguments.model_source}"
    )

    print(
        "objective："
        f"{objective}"
    )

    print(
        "训练扩散步数："
        f"{actual_num_timesteps}"
    )

    print(
        "诊断采样步数："
        f"{actual_sampling_timesteps}"
    )

    print(
        "诊断采样器："
        f"{sampling_mode}"
    )

    print(
        "prior："
        f"{prior_transformer.prior_method}"
    )

    print(
        "residual normalization："
        f"{prior_transformer.normalization_method}"
    )

    print(
        "PCA主成分数："
        f"{prior_transformer.pca_components.shape[0]}"
    )

    print(
        "训练光谱数："
        f"{train_normalized.shape[0]}"
    )

    print(
        "生成residual数："
        f"{generated_scaled_residual.shape[0]}"
    )

    print(
        "模型Raman轴："
        f"{model_axis[0]:.6g}–"
        f"{model_axis[-1]:.6g} cm^-1，"
        f"{model_axis.size}点"
    )

    print(
        "自动检测峰数："
        f"{len(peak_table)}"
    )

    print(
        "峰区点数："
        f"{int(np.sum(peak_mask))}"
    )

    print(
        "非峰区点数："
        f"{int(np.sum(non_peak_mask))}"
    )

    print(
        "broad sigma："
        f"{diagnostic_parameters['broad_sigma_cm1']:.6g} cm^-1"
    )

    print(
        "robust_asinh residual_scale："
        f"{float(prior_transformer.residual_scale):.8g}"
    )

    print(
        "robust_asinh asinh_normalizer："
        f"{float(prior_transformer.asinh_normalizer):.8g}"
    )

    print(
        "真实residual transform/inverse最大回环误差："
        f"{inverse_roundtrip_error:.8g}"
    )

    print()
    print(
        "===== ① scaled residual 尾部 ====="
    )

    scaled_print = tail_table[
        tail_table[
            "domain"
        ]
        == "scaled_residual"
    ]

    print(
        scaled_print[
            [
                "metric",
                "real",
                "generated",
                "generated_to_real",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== ② robust_asinh inverse后的raw residual ====="
    )

    inverse_print = tail_table[
        tail_table[
            "domain"
        ]
        == "inverse_raw_residual"
    ]

    print(
        inverse_print[
            [
                "metric",
                "real",
                "generated",
                "generated_to_real",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== inverse amplification ====="
    )

    print(
        inverse_amplification_table.to_string(
            index=False
        )
    )

    print()
    print(
        "===== residual broad/nonpeak尾部 ====="
    )

    broad_print = tail_table[
        tail_table[
            "domain"
        ].isin(
            [
                "scaled_residual_broad_nonpeak",
                "inverse_raw_residual_broad_nonpeak",
            ]
        )
    ]

    print(
        broad_print[
            [
                "domain",
                "metric",
                "real",
                "generated",
                "generated_to_real",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== ③ fixed PCA mean + DDPM residual ====="
    )

    print(
        component_table[
            [
                "dataset",
                "number_of_spectra",
                "nonpeak_band95",
                "nonpeak_iqr",
                "nonpeak_std",
                "broad_band95",
                "broad_iqr",
                "broad_std",
                "local_band95",
                "local_iqr",
                "local_std",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== matched-size bootstrap ====="
    )

    print(
        bootstrap_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== 输出文件 ====="
    )

    for filename in (
        "residual_tail_statistics.csv",
        "inverse_amplification_statistics.csv",
        "component_distribution_statistics.csv",
        "matched_bootstrap_summary.csv",
        "matched_bootstrap_all_iterations.csv",
        "automatically_detected_peak_regions.csv",
        "training_indices.csv",
        "ddpm_residual_diagnostic_arrays.npz",
        "real_training_normalized.csv",
        "true_scaled_residual.csv",
        "generated_scaled_residual_first100.csv",
        "generated_raw_residual_first100.csv",
        "fixed_pca_mean_generated_first100.csv",
    ):
        print(
            "  - "
            + str(
                output_directory
                / filename
            )
        )

    print()
    print(
        "本脚本没有执行sampling calibration、"
        "Raman jitter、variation_scale或峰区limiter。"
    )

    print(
        "下一步应根据scaled residual、"
        "inverse residual和fixed PCA mean bootstrap"
        "的结果决定是否修改模型。"
    )


if __name__ == "__main__":
    main()

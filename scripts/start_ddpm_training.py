"""
训练 D0-D3 一维 SERS DDPM。

完整项目测试

echo "===== 1. Python语法检查 ====="
python -m compileall -q \
    src \
    scripts \
    tests \
    run.py

echo "===== 2. 软件包依赖检查 ====="
python -m pip check

echo "===== 3. 项目完整自动化测试 ====="
python -m pytest -v


开始训练指令
    CUDA_VISIBLE_DEVICES=1 \
    python -m scripts.start_ddpm_training \
    --config config/ddpm_training.yaml

    生成指令
    CUDA_VISIBLE_DEVICES=1 \
    python -m scripts.generate_spectra \
    --config config/ddpm_training.yaml \
    --checkpoint outputs/checkpoints/latest.pt \
    --number 50

    监控gpu指令
    watch -n 1 nvidia-smi

    以后添加新模块后，只需：

    git add .
    git commit -m "D1: add new module"
    git push
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Subset,
)

from src.checkpoint_manager import (
    CheckpointManager,
    build_axis_metadata,
    load_checkpoint_file,
)
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.dataset_splitter import (
    split_spectrum_collection,
)
from src.ddpm_trainer import (
    DdpmTrainer,
)
from src.feature_peak_residual_limiter import (
    fit_feature_peak_residual_limiter_state,
)
from src.intensity_normalizer import (
    GlobalMinMaxNormalizer,
)
from src.model_builder import (
    build_diffusion_model,
)
from src.one_dimensional_ddpm import (
    get_backend_version,
)
from src.prior_residual import (
    PriorResidualTransformer,
)
from src.broad_local_residual import (
    BroadLocalResidualDecomposer,
)
from src.random_seed_manager import (
    create_data_loader_generator,
    seed_data_loader_worker,
    set_random_seed,
)
from src.sers_diversity_constraints import (
    fit_sers_diversity_constraint_state,
)
from src.sers_physics_constraints import (
    fit_sers_physics_constraint_state,
)
from src.sers_peak_derivative_constraints import (
    fit_peak_derivative_constraint_state,
)
from src.sers_relative_peak_intensity_constraints import (
    fit_relative_peak_intensity_constraint_state,
)
from src.sers_peak_parameter_constraints import (
    fit_peak_parameter_constraint_state,
)
from src.spectrum_dataset import (
    SpectrumDataset,
)
from src.spectrum_file_reader import (
    SpectrumCollection,
    read_spectrum_collection,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)
from src.training_logger import (
    TrainingLogger,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练一维SERS DDPM。"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML配置文件路径。",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="从同一实验阶段checkpoint继续训练。",
    )

    parser.add_argument(
        "--constraint-check-only",
        action="store_true",
        help=(
            "只执行D3.4真实training batch约束激活和梯度检查，"
            "不开始正式训练。"
        ),
    )

    return parser.parse_args()


def resolve_device(
    device_text: str,
) -> torch.device:
    normalized = str(
        device_text
    ).strip().lower()

    if (
        normalized.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "配置要求使用CUDA，"
            "但PyTorch没有检测到可用GPU。"
        )

    return torch.device(
        normalized
    )


def validate_split_indices(
    *,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    number_of_spectra: int,
) -> None:
    named_indices = {
        "训练集": training_indices,
        "验证集": validation_indices,
        "测试集": test_indices,
    }

    for (
        subset_name,
        indices,
    ) in named_indices.items():
        if indices.size == 0:
            raise RuntimeError(
                f"{subset_name}为空。"
            )

        if (
            np.any(
                indices < 0
            )
            or np.any(
                indices
                >= number_of_spectra
            )
        ):
            raise RuntimeError(
                f"{subset_name}包含越界索引。"
            )

        if (
            np.unique(
                indices
            ).size
            != indices.size
        ):
            raise RuntimeError(
                f"{subset_name}包含重复索引。"
            )

    all_indices = np.concatenate(
        [
            training_indices,
            validation_indices,
            test_indices,
        ]
    )

    if (
        all_indices.size
        != number_of_spectra
        or np.unique(
            all_indices
        ).size
        != number_of_spectra
    ):
        raise RuntimeError(
            "训练、验证和测试没有无重复覆盖全部光谱。"
        )


def resolve_training_log_file(
    configuration: dict,
) -> Path:
    output = configuration[
        "output"
    ]

    if (
        "training_log_file"
        in output
    ):
        value = output[
            "training_log_file"
        ]
    else:
        value = (
            Path(
                output[
                    "log_directory"
                ]
            )
            / str(
                output.get(
                    "training_log_name",
                    "training_log.csv",
                )
            )
        )

    return resolve_project_path(
        configuration,
        value,
    )


def build_single_spectrum_overfit_collection(
    *,
    collection: SpectrumCollection,
    diagnostic_config: dict,
) -> tuple[
    SpectrumCollection,
    dict[str, object] | None,
]:
    """保留已有单谱重复过拟合诊断接口。"""

    if not bool(
        diagnostic_config.get(
            "enabled",
            False,
        )
    ):
        return (
            collection,
            None,
        )

    spectrum_index = int(
        diagnostic_config.get(
            "spectrum_index",
            0,
        )
    )

    repeat_count = int(
        diagnostic_config.get(
            "repeat_count",
            50,
        )
    )

    original_count = len(
        collection.spectrum_names
    )

    if original_count == 0:
        raise RuntimeError(
            "原始数据中没有可用光谱。"
        )

    if not (
        0
        <= spectrum_index
        < original_count
    ):
        raise IndexError(
            "diagnostic_overfit.spectrum_index越界："
            f"有效范围0到{original_count - 1}。"
        )

    if repeat_count < 10:
        raise ValueError(
            "diagnostic_overfit.repeat_count至少为10。"
        )

    selected_spectrum = np.asarray(
        collection.spectra[
            spectrum_index
        ],
        dtype=np.float32,
    ).reshape(-1)

    selected_axis = np.asarray(
        collection.raman_shifts[
            spectrum_index
        ],
        dtype=np.float64,
    ).reshape(-1)

    if (
        selected_spectrum.size
        != selected_axis.size
    ):
        raise RuntimeError(
            "所选光谱和拉曼轴长度不一致。"
        )

    if selected_axis.size < 2:
        raise RuntimeError(
            "所选拉曼轴至少需要2个点。"
        )

    if (
        not np.isfinite(
            selected_spectrum
        ).all()
        or not np.isfinite(
            selected_axis
        ).all()
    ):
        raise RuntimeError(
            "所选光谱或拉曼轴包含NaN/无穷值。"
        )

    if not np.all(
        np.diff(
            selected_axis
        )
        > 0.0
    ):
        raise RuntimeError(
            "所选拉曼轴必须严格递增。"
        )

    selected_name = str(
        collection.spectrum_names[
            spectrum_index
        ]
    )

    selected_source_file = (
        collection.source_files[
            spectrum_index
        ]
    )

    selected_relative_source_file = (
        collection.relative_source_files[
            spectrum_index
        ]
    )

    selected_label = (
        collection.labels[
            spectrum_index
        ]
    )

    repeated_collection = (
        SpectrumCollection(
            raman_shift=selected_axis.copy(),
            spectra=np.repeat(
                selected_spectrum[
                    None,
                    :
                ],
                repeat_count,
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            ),
            raman_shifts=np.repeat(
                selected_axis[
                    None,
                    :
                ],
                repeat_count,
                axis=0,
            ).astype(
                np.float64,
                copy=False,
            ),
            original_lengths=np.full(
                repeat_count,
                selected_axis.size,
                dtype=np.int64,
            ),
            source_files=np.asarray(
                [
                    selected_source_file
                ]
                * repeat_count,
                dtype=object,
            ),
            relative_source_files=np.asarray(
                [
                    selected_relative_source_file
                ]
                * repeat_count,
                dtype=object,
            ),
            spectrum_names=np.asarray(
                [
                    (
                        f"{selected_name}"
                        f"__repeat_{index + 1:04d}"
                    )
                    for index
                    in range(
                        repeat_count
                    )
                ],
                dtype=object,
            ),
            labels=np.asarray(
                [
                    selected_label
                ]
                * repeat_count,
                dtype=object,
            ),
        )
    )

    metadata: dict[str, object] = {
        "enabled": True,
        "formal_validation": False,
        "original_number_of_spectra": int(
            original_count
        ),
        "selected_original_index": int(
            spectrum_index
        ),
        "selected_spectrum_name": (
            selected_name
        ),
        "selected_source_file": str(
            selected_source_file
        ),
        "selected_relative_source_file": str(
            selected_relative_source_file
        ),
        "selected_label": str(
            selected_label
        ),
        "selected_original_length": int(
            selected_axis.size
        ),
        "repeat_count": int(
            repeat_count
        ),
    }

    return (
        repeated_collection,
        metadata,
    )


def resolve_resume_path(
    *,
    configuration: dict,
    resume_argument: str,
) -> Path:
    resume_path = Path(
        resume_argument
    ).expanduser()

    if not resume_path.is_absolute():
        resume_path = resolve_project_path(
            configuration,
            resume_path,
        )

    if not resume_path.is_file():
        raise FileNotFoundError(
            f"找不到resume检查点：{resume_path}"
        )

    return resume_path


def validate_resume_stage(
    *,
    checkpoint_path: Path,
    configuration: dict,
) -> None:
    """
    禁止跨 prior 数据域、D3 模块或 loss 配置直接 resume。

    特别注意：
    D2.4 blended prior 不能从 D2.3 low-frequency checkpoint
    或 D2.1 median checkpoint 直接 --resume。
    """

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    checkpoint_configuration = (
        checkpoint.get(
            "configuration"
        )
    )

    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "resume检查点缺少有效configuration。"
        )

    configuration_pairs = (
        (
            "prior_residual",
            "D2先验残差",
        ),
        (
            "broad_local_residual",
            "D2.6 broad-local residual",
        ),
        (
            "physics_constraints",
            "D3物理约束",
        ),
        (
            "diversity_constraints",
            "D3多样性约束",
        ),
        (
            "feature_peak_residual_limiter",
            "D3特征峰残差limiter",
        ),
        (
            "local_peak_distribution_constraints",
            "D3局部峰分布约束",
        ),
        (
            "peak_derivative_constraints",
            "D3.1自动峰区一阶导数约束",
        ),
        (
            "relative_peak_intensity_constraints",
            "D3.2自动峰区相对峰强约束",
        ),
        (
            "peak_parameter_constraints",
            "D3.4完整谱峰参数物理约束",
        ),
    )

    for (
        key,
        description,
    ) in configuration_pairs:
        current_value = (
            configuration.get(
                key,
                {},
            )
            or {}
        )

        checkpoint_value = (
            checkpoint_configuration.get(
                key,
                {},
            )
            or {}
        )

        if (
            current_value
            != checkpoint_value
        ):
            raise ValueError(
                f"当前{description}配置"
                "与resume检查点不一致。"
                "只能从同一模型阶段、"
                "同一数据域的checkpoint续训。"
            )

    current_residual_aware = (
        configuration.get(
            "diffusion",
            {},
        ).get(
            "residual_aware_loss",
            {
                "enabled": False,
            },
        )
        or {
            "enabled": False,
        }
    )

    checkpoint_residual_aware = (
        checkpoint_configuration.get(
            "diffusion",
            {},
        ).get(
            "residual_aware_loss",
            {
                "enabled": False,
            },
        )
        or {
            "enabled": False,
        }
    )

    if (
        current_residual_aware
        != checkpoint_residual_aware
    ):
        raise ValueError(
            "当前diffusion.residual_aware_loss"
            "与resume检查点不一致。"
        )


def print_prior_residual_summary(
    transformer: PriorResidualTransformer,
    transformed_training_residuals: np.ndarray,
) -> None:
    """打印 D2 状态。"""

    print(
        "\n===== D2先验残差状态 ====="
    )

    print(
        "先验方法："
        f"{transformer.prior_method}"
    )

    print(
        "残差归一化方法："
        f"{transformer.normalization_method}"
    )

    statistics = (
        transformer.training_abs_residual_percentiles
        or {}
    )

    for key in (
        "p50",
        "p90",
        "p95",
        "p99",
        "p99_5",
        "p99_9",
        "max",
    ):
        if key in statistics:
            print(
                f"训练原始残差 {key}: "
                f"{statistics[key]:.8g}"
            )

    if (
        transformer.normalization_method
        == "robust_asinh"
    ):
        print(
            "残差缩放系数："
            f"{transformer.residual_scale:.8g}"
        )

        print(
            "asinh归一化因子："
            f"{transformer.asinh_normalizer:.8g}"
        )

    elif (
        transformer.normalization_method
        == "pointwise_mad_asinh"
    ):
        print(
            "逐波数MAD换算系数："
            f"{transformer.mad_scale_factor:.8g}"
        )

        print(
            "逐波数尺度下限："
            f"{transformer.pointwise_scale_floor:.8g}"
        )

        pointwise_statistics = (
            transformer.training_pointwise_scale_percentiles
            or {}
        )

        for key in (
            "p50",
            "p90",
            "p95",
            "p99",
            "p99_5",
            "p99_9",
            "max",
        ):
            if key in pointwise_statistics:
                print(
                    f"逐波数尺度 {key}: "
                    f"{pointwise_statistics[key]:.8g}"
                )

        print(
            "标准化残差尺度："
            f"{transformer.residual_scale:.8g}"
        )

        print(
            "asinh归一化因子："
            f"{transformer.asinh_normalizer:.8g}"
        )

    if (
        transformer.prior_method
        == "pca_reconstruction"
    ):
        cumulative_ratio = float(
            np.sum(
                transformer.pca_explained_variance_ratio_
            )
        )

        print(
            "D2.2 PCA可变先验：已启用"
        )

        print(
            "PCA主成分数："
            f"{transformer.pca_components.shape[0]}"
        )

        print(
            "累计解释方差比例："
            f"{cumulative_ratio:.6f}"
        )

        print(
            "PCA score采样策略："
            f"{transformer.pca_sampling_strategy}"
        )

        print(
            "PCA score截断范围：±"
            f"{transformer.pca_score_clip_standard_deviations:g}"
            "个标准差"
        )

    if transformer.prior_method in {
        "training_low_frequency_median",
        "training_blended_frequency_median",
    }:
        print(
            "低频先验方法："
            f"{transformer.low_frequency_method}"
        )

        print(
            "Gaussian sigma："
            f"{transformer.low_frequency_sigma_cm1:g} cm^-1"
        )

        print(
            "Gaussian truncate："
            f"{transformer.low_frequency_truncate:g}"
        )

        print(
            "辅助等间距轴间隔："
            f"{transformer.low_frequency_uniform_spacing_cm1:.8g} "
            "cm^-1"
        )

        print(
            "换算sigma_points："
            f"{transformer.low_frequency_sigma_points:.8g}"
        )

        print(
            "低频prior拟合训练谱数量："
            f"{transformer.low_frequency_number_of_training_spectra}"
        )

    if (
        transformer.prior_method
        == "training_blended_frequency_median"
    ):
        print(
            "D2.4频率混合先验：已启用"
        )

        print(
            "共同峰骨架保留比例 alpha："
            f"{transformer.blended_peak_component_ratio:g}"
        )

        print(
            "D2.4公式："
            "P_blend = P_low + alpha * "
            "(P_median - P_low)"
        )

    absolute_values = np.abs(
        np.asarray(
            transformed_training_residuals,
            dtype=np.float64,
        )
    )

    percentiles = np.percentile(
        absolute_values,
        [
            50.0,
            90.0,
            95.0,
            99.0,
            99.5,
            99.9,
            100.0,
        ],
    )

    for (
        name,
        value,
    ) in zip(
        (
            "p50",
            "p90",
            "p95",
            "p99",
            "p99.5",
            "p99.9",
            "max",
        ),
        percentiles,
    ):
        print(
            f"变换后残差 {name}: "
            f"{float(value):.8g}"
        )

    for threshold in (
        0.50,
        0.80,
        0.95,
        1.00,
    ):
        fraction = float(
            np.mean(
                absolute_values
                > threshold
            )
            * 100.0
        )

        print(
            f"|scaled residual| > "
            f"{threshold:.2f} 比例："
            f"{fraction:.4f}%"
        )

    print(
        "==============================\n"
    )


def print_broad_local_residual_summary(
    decomposer: BroadLocalResidualDecomposer | None,
    training_scaled_local_residuals: np.ndarray | None,
) -> None:
    if decomposer is None:
        print(
            "D2.6 broad-local residual：未启用"
        )
        return

    if training_scaled_local_residuals is None:
        raise RuntimeError(
            "D2.6已启用，但缺少训练local residual统计。"
        )

    summary = decomposer.summary()

    print(
        "\n===== D2.6 broad-local residual状态 ====="
    )

    print(
        "broad Gaussian sigma："
        f"{summary['broad_sigma_cm1']:.6g} cm^-1"
    )

    print(
        "broad PCA主成分数："
        f"{summary['broad_pca_components']}"
    )

    print(
        "broad PCA累计解释方差："
        f"{summary['broad_pca_cumulative_explained_variance']:.6f}"
    )

    print(
        "broad PCA score截断：±"
        f"{summary['broad_score_clip_standard_deviations']:.6g}σ"
    )

    print(
        "local normalization：robust_asinh"
    )

    print(
        "local residual scale："
        f"{summary['local_residual_scale']:.8g}"
    )

    print(
        "local asinh normalizer："
        f"{summary['local_asinh_normalizer']:.8g}"
    )

    print(
        "训练raw residual RMS中位数："
        f"{summary['training_raw_rms_median']:.8g}"
    )

    print(
        "训练broad residual RMS中位数："
        f"{summary['training_broad_rms_median']:.8g}"
    )

    print(
        "训练local residual RMS中位数："
        f"{summary['training_local_rms_median']:.8g}"
    )

    absolute = np.abs(
        np.asarray(
            training_scaled_local_residuals,
            dtype=np.float64,
        )
    )

    for name, percentile in (
        ("p50", 50.0),
        ("p90", 90.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("p99.5", 99.5),
        ("p99.9", 99.9),
        ("max", 100.0),
    ):
        value = float(
            np.percentile(
                absolute,
                percentile,
            )
        )

        print(
            "训练scaled local residual "
            f"{name}: {value:.8g}"
        )

    print(
        "D2.6模型输入：只训练local residual；"
        "broad residual由train-only broad PCA生成。"
    )

    print(
        "========================================\n"
    )


def print_physics_summary(
    state: dict | None,
    peak_derivative_enabled: bool,
    relative_peak_intensity_enabled: bool,
) -> None:
    if (
        not state
        or not bool(
            state.get(
                "enabled",
                False,
            )
        )
    ):
        print(
            "旧D3复合物理/峰约束：未启用"
        )

    if peak_derivative_enabled:
        print(
            "D3.1唯一新增变量：train-only自动峰区"
            "一阶导数软约束"
        )

        return

    if relative_peak_intensity_enabled:
        print(
            "D3.2唯一新增变量：train-only自动峰区"
            "相对峰强软约束"
        )

        return

    print(
        "D3物理/峰约束：已启用"
    )


def print_diversity_summary(
    state: dict | None,
) -> None:
    if (
        not state
        or not bool(
            state.get(
                "enabled",
                False,
            )
        )
    ):
        print(
            "D3多样性约束：未启用"
        )

        return

    print(
        "D3多样性约束：已启用"
    )


def print_feature_peak_residual_limiter_summary(
    state: dict | None,
) -> None:
    if (
        not state
        or not bool(
            state.get(
                "enabled",
                False,
            )
        )
    ):
        print(
            "D3特征峰残差limiter：未启用"
        )

        return

    print(
        "D3特征峰残差limiter：已启用"
    )


def main() -> None:
    arguments = parse_arguments()

    configuration = load_configuration(
        arguments.config
    )

    project_config = configuration[
        "project"
    ]

    diagnostic_config = configuration.get(
        "diagnostic_overfit",
        {},
    )

    data_config = configuration[
        "data"
    ]

    model_config = configuration[
        "model"
    ]

    normalization_config = configuration[
        "normalization"
    ]

    prior_config = configuration.get(
        "prior_residual",
        {},
    ) or {}

    broad_local_config = configuration.get(
        "broad_local_residual",
        {
            "enabled": False,
        },
    ) or {
        "enabled": False,
    }

    if not isinstance(
        broad_local_config,
        dict,
    ):
        raise TypeError(
            "broad_local_residual配置必须是字典。"
        )

    broad_local_enabled = bool(
        broad_local_config.get(
            "enabled",
            False,
        )
    )

    physics_config = configuration.get(
        "physics_constraints",
        {
            "enabled": False,
        },
    )

    peak_derivative_config = configuration.get(
        "peak_derivative_constraints",
        {
            "enabled": False,
        },
    ) or {
        "enabled": False,
    }

    if not isinstance(peak_derivative_config, dict):
        raise TypeError("peak_derivative_constraints配置必须是字典。")

    peak_derivative_enabled = bool(
        peak_derivative_config.get("enabled", False)
    )

    relative_peak_config = configuration.get(
        "relative_peak_intensity_constraints",
        {
            "enabled": False,
        },
    ) or {
        "enabled": False,
    }

    if not isinstance(relative_peak_config, dict):
        raise TypeError(
            "relative_peak_intensity_constraints配置必须是字典。"
        )

    relative_peak_enabled = bool(
        relative_peak_config.get("enabled", False)
    )

    peak_parameter_config = configuration.get(
        "peak_parameter_constraints",
        {"enabled": False},
    ) or {"enabled": False}

    if not isinstance(peak_parameter_config, dict):
        raise TypeError(
            "peak_parameter_constraints配置必须是字典。"
        )

    peak_parameter_enabled = bool(
        peak_parameter_config.get("enabled", False)
    )

    enabled_new_d3_modules = sum(
        int(value)
        for value in (
            peak_derivative_enabled,
            relative_peak_enabled,
            peak_parameter_enabled,
        )
    )

    if enabled_new_d3_modules > 1:
        raise ValueError(
            "D3消融每次只允许启用一个主要训练变量；"
            "D3.1 peak_derivative、D3.2 relative_peak_intensity"
            "和D3.4 peak_parameter不能同时启用。"
        )

    if peak_derivative_enabled and not broad_local_enabled:
        raise ValueError(
            "D3.1自动峰区一阶导数约束要求启用"
            "D2.6 broad_local_residual。"
        )

    if relative_peak_enabled and not broad_local_enabled:
        raise ValueError(
            "D3.2自动峰区相对峰强约束要求启用"
            "D2.6 broad_local_residual。"
        )

    if peak_parameter_enabled and not broad_local_enabled:
        raise ValueError(
            "D3.4完整谱峰参数物理约束要求启用"
            "D2.6 broad_local_residual。"
        )

    if broad_local_enabled:
        if not bool(
            prior_config.get(
                "enabled",
                False,
            )
        ):
            raise ValueError(
                "D2.6要求prior_residual.enabled=true。"
            )

        if str(
            prior_config.get(
                "prior_method",
                "",
            )
        ).strip().lower() != "pca_reconstruction":
            raise ValueError(
                "D2.6当前只支持"
                "prior_residual.prior_method=pca_reconstruction。"
            )

        incompatible_sections = (
            "physics_constraints",
            "diversity_constraints",
            "feature_peak_residual_limiter",
            "local_peak_distribution_constraints",
        )

        enabled_incompatible = []

        for section_name in incompatible_sections:
            section = (
                configuration.get(
                    section_name,
                    {},
                )
                or {}
            )

            if (
                isinstance(
                    section,
                    dict,
                )
                and bool(
                    section.get(
                        "enabled",
                        False,
                    )
                )
            ):
                enabled_incompatible.append(
                    section_name
                )

        if enabled_incompatible:
            raise ValueError(
                "D3.1消融只允许启用自动峰区一阶导数约束；"
                "其余旧D3模块必须关闭；"
                "当前仍启用："
                f"{enabled_incompatible}。"
            )

    training_config = configuration[
        "training"
    ]

    output_config = configuration[
        "output"
    ]

    random_seed = int(
        project_config.get(
            "random_seed",
            2026,
        )
    )

    set_random_seed(
        random_seed=random_seed,
    )

    input_directory = (
        resolve_project_path(
            configuration,
            data_config[
                "input_directory"
            ],
        )
    )

    collection = (
        read_spectrum_collection(
            input_directory=input_directory,
            data_config=data_config,
        )
    )

    collection, overfit_metadata = (
        build_single_spectrum_overfit_collection(
            collection=collection,
            diagnostic_config=diagnostic_config,
        )
    )

    number_of_spectra = len(
        collection.spectrum_names
    )

    if number_of_spectra == 0:
        raise RuntimeError(
            "没有读取到任何光谱。"
        )

    # ------------------------------------------------------------------
    # 数据划分
    # ------------------------------------------------------------------

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
    ).reshape(-1)

    validation_indices = np.asarray(
        dataset_split.validation.indices,
        dtype=np.int64,
    ).reshape(-1)

    test_indices = np.asarray(
        dataset_split.test.indices,
        dtype=np.int64,
    ).reshape(-1)

    validate_split_indices(
        training_indices=training_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        number_of_spectra=number_of_spectra,
    )

    # ------------------------------------------------------------------
    # 拉曼轴统一
    # ------------------------------------------------------------------

    length_adapter = (
        SpectrumLengthAdapter.create(
            raman_shifts=(
                collection.raman_shifts
            ),
            dimension_multipliers=(
                model_config[
                    "dimension_multipliers"
                ]
            ),
            model_length=data_config.get(
                "model_spectrum_length",
                "auto",
            ),
            padding_mode=str(
                data_config.get(
                    "padding_mode",
                    "right_zero_padding",
                )
            ),
            padding_value=float(
                data_config.get(
                    "padding_value",
                    0.0,
                )
            ),
            raman_range_tolerance=float(
                data_config.get(
                    "raman_range_tolerance",
                    1.0,
                )
            ),
        )
    )

    spectra_on_model_axis = (
        length_adapter.interpolate_to_model_axis(
            collection.spectra,
            collection.raman_shifts,
        )
    )

    spectra_on_model_axis = np.asarray(
        spectra_on_model_axis,
        dtype=np.float32,
    )

    expected_shape = (
        number_of_spectra,
        length_adapter.original_length,
    )

    if (
        spectra_on_model_axis.shape
        != expected_shape
    ):
        raise RuntimeError(
            "插值后的光谱形状不正确："
            f"实际={spectra_on_model_axis.shape}，"
            f"期望={expected_shape}。"
        )

    if not np.isfinite(
        spectra_on_model_axis
    ).all():
        raise RuntimeError(
            "插值后的光谱包含NaN或无穷值。"
        )

    # ------------------------------------------------------------------
    # train-only global_minmax
    # ------------------------------------------------------------------

    normalizer: (
        GlobalMinMaxNormalizer | None
    ) = None

    normalization_state = None

    if bool(
        normalization_config.get(
            "enabled",
            False,
        )
    ):
        normalizer = (
            GlobalMinMaxNormalizer(
                target_min=float(
                    normalization_config.get(
                        "target_min",
                        -1.0,
                    )
                ),
                target_max=float(
                    normalization_config.get(
                        "target_max",
                        1.0,
                    )
                ),
                epsilon=float(
                    normalization_config.get(
                        "epsilon",
                        1.0e-12,
                    )
                ),
                clip=bool(
                    normalization_config.get(
                        "clip",
                        False,
                    )
                ),
            )
        )

        # 严格只在 train 上 fit。
        normalizer.fit(
            spectra_on_model_axis[
                training_indices
            ]
        )

        normalized_full_spectra = (
            normalizer.transform(
                spectra_on_model_axis
            )
        )

        if bool(
            normalization_config.get(
                "save_in_checkpoint",
                True,
            )
        ):
            normalization_state = (
                normalizer.state_dict()
            )

    else:
        normalized_full_spectra = (
            spectra_on_model_axis.copy()
        )

    # ------------------------------------------------------------------
    # D2 prior-residual
    # ------------------------------------------------------------------

    prior_residual_transformer: (
        PriorResidualTransformer | None
    ) = None

    prior_residual_state = None

    broad_local_residual_decomposer: (
        BroadLocalResidualDecomposer | None
    ) = None

    broad_local_residual_state = None

    constraint_reconstruction_bases_full = None

    spectra_for_model = (
        normalized_full_spectra.copy()
    )

    training_scaled_residuals = (
        spectra_for_model[
            training_indices
        ]
    )

    if bool(
        prior_config.get(
            "enabled",
            False,
        )
    ):
        if (
            normalizer is None
            or normalization_state is None
        ):
            raise ValueError(
                "启用prior_residual时必须启用"
                "并保存train-only global_minmax状态。"
            )

        low_frequency_config = (
            prior_config.get(
                "low_frequency",
                {},
            )
            or {}
        )

        blended_frequency_config = (
            prior_config.get(
                "blended_frequency",
                {},
            )
            or {}
        )

        prior_residual_transformer = (
            PriorResidualTransformer(
                prior_method=str(
                    prior_config.get(
                        "prior_method",
                        "training_pointwise_median",
                    )
                ),
                normalization_method=str(
                    prior_config.get(
                        "residual_normalization",
                        "robust_asinh",
                    )
                ),
                target_abs_max=float(
                    prior_config.get(
                        "target_abs_max",
                        1.0,
                    )
                ),
                residual_quantile=float(
                    prior_config.get(
                        "residual_quantile",
                        99.5,
                    )
                ),
                pointwise_scale_floor_quantile=float(
                    prior_config.get(
                        "pointwise_scale_floor_quantile",
                        10.0,
                    )
                ),
                mad_scale_factor=float(
                    prior_config.get(
                        "mad_scale_factor",
                        1.4826,
                    )
                ),
                epsilon=float(
                    prior_config.get(
                        "epsilon",
                        1.0e-8,
                    )
                ),
                pca_explained_variance_ratio=float(
                    prior_config.get(
                        "pca_explained_variance_ratio",
                        0.95,
                    )
                ),
                pca_max_components=(
                    prior_config.get(
                        "pca_max_components"
                    )
                ),
                pca_sampling_strategy=str(
                    prior_config.get(
                        "pca_sampling_strategy",
                        "truncated_gaussian_scores",
                    )
                ),
                pca_score_clip_standard_deviations=float(
                    prior_config.get(
                        "pca_score_clip_standard_deviations",
                        2.5,
                    )
                ),
                low_frequency_method=str(
                    low_frequency_config.get(
                        "method",
                        "gaussian",
                    )
                ),
                low_frequency_sigma_cm1=float(
                    low_frequency_config.get(
                        "sigma_cm1",
                        40.0,
                    )
                ),
                low_frequency_truncate=float(
                    low_frequency_config.get(
                        "truncate",
                        4.0,
                    )
                ),
                blended_peak_component_ratio=float(
                    blended_frequency_config.get(
                        "peak_component_ratio",
                        0.5,
                    )
                ),
            )
        )

        # 这里始终把真实 Raman shift 传进去。
        # median / PCA 会忽略它；
        # low-frequency / blended-frequency 会使用它。
        prior_residual_transformer.fit(
            normalized_full_spectra[
                training_indices
            ],
            raman_shift=np.asarray(
                length_adapter.model_raman_shift,
                dtype=np.float64,
            ),
        )

        prior_residual_state = (
            prior_residual_transformer.state_dict()
        )

        prior_summary_scaled_residuals = (
            prior_residual_transformer.transform(
                normalized_full_spectra[
                    training_indices
                ]
            )
        )

        if broad_local_enabled:
            if (
                prior_residual_transformer.prior_method
                != "pca_reconstruction"
            ):
                raise RuntimeError(
                    "D2.6要求已拟合PCA outer prior。"
                )

            reference_priors_full = (
                prior_residual_transformer
                .reference_priors_for_spectra(
                    normalized_full_spectra
                )
            )

            raw_residuals_full = (
                normalized_full_spectra
                - reference_priors_full
            ).astype(
                np.float32,
                copy=False,
            )

            broad_local_residual_decomposer = (
                BroadLocalResidualDecomposer
                .from_configuration(
                    broad_local_config
                )
            )

            broad_local_residual_decomposer.fit(
                raw_residuals_full[
                    training_indices
                ],
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
            )

            broad_residuals_full, _ = (
                broad_local_residual_decomposer.split_raw_residuals(
                    raw_residuals_full
                )
            )

            constraint_reconstruction_bases_full = (
                reference_priors_full + broad_residuals_full
            ).astype(
                np.float32,
                copy=False,
            )

            training_scaled_residuals = (
                broad_local_residual_decomposer
                .transform_raw_residuals(
                    raw_residuals_full[
                        training_indices
                    ]
                )
            )

            spectra_for_model = (
                broad_local_residual_decomposer
                .transform_raw_residuals(
                    raw_residuals_full
                )
            )

            broad_local_residual_state = (
                broad_local_residual_decomposer
                .state_dict()
            )

        else:
            training_scaled_residuals = (
                prior_residual_transformer.transform(
                    normalized_full_spectra[
                        training_indices
                    ]
                )
            )

            spectra_for_model = (
                prior_residual_transformer.transform(
                    normalized_full_spectra
                )
            )

        print_prior_residual_summary(
            prior_residual_transformer,
            prior_summary_scaled_residuals,
        )

        print_broad_local_residual_summary(
            broad_local_residual_decomposer,
            (
                training_scaled_residuals
                if broad_local_enabled
                else None
            ),
        )

    # ------------------------------------------------------------------
    # D3 physics
    # ------------------------------------------------------------------

    physics_constraint_state = None

    if bool(
        physics_config.get(
            "enabled",
            False,
        )
    ):
        if prior_residual_state is None:
            raise ValueError(
                "D3物理约束要求先完成D2拟合。"
            )

        physics_constraint_state = (
            fit_sers_physics_constraint_state(
                training_normalized_spectra=(
                    normalized_full_spectra[
                        training_indices
                    ]
                ),
                training_scaled_residuals=(
                    training_scaled_residuals
                ),
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
                configuration=physics_config,
            )
        )

    print_physics_summary(
        physics_constraint_state,
        peak_derivative_enabled=peak_derivative_enabled,
        relative_peak_intensity_enabled=relative_peak_enabled,
    )

    # ------------------------------------------------------------------
    # D3.1 train-only automatic peak derivative constraint
    # ------------------------------------------------------------------

    peak_derivative_constraint_state = None

    if peak_derivative_enabled:
        if broad_local_residual_state is None:
            raise RuntimeError(
                "D3.1缺少D2.6 broad_local_residual_state。"
            )
        if constraint_reconstruction_bases_full is None:
            raise RuntimeError(
                "D3.1缺少outer PCA prior + broad residual恢复基底。"
            )

        peak_derivative_constraint_state = (
            fit_peak_derivative_constraint_state(
                training_normalized_spectra=(
                    normalized_full_spectra[training_indices]
                ),
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
                configuration=peak_derivative_config,
            )
        )

        print(
            "D3.1自动峰区一阶导数约束：已启用；"
            "训练集自动峰数="
            f"{len(peak_derivative_constraint_state['detected_peak_indices'])}；"
            "峰区mask占比="
            f"{100.0 * float(peak_derivative_constraint_state['peak_mask_fraction']):.2f}%；"
            "导数尺度="
            f"{float(peak_derivative_constraint_state['derivative_scale']):.8g}"
        )
    else:
        print("D3.1自动峰区一阶导数约束：未启用")

    # ------------------------------------------------------------------
    # D3.2 train-only automatic relative peak intensity constraint
    # ------------------------------------------------------------------

    relative_peak_intensity_constraint_state = None

    if relative_peak_enabled:
        if broad_local_residual_state is None:
            raise RuntimeError(
                "D3.2缺少D2.6 broad_local_residual_state。"
            )
        if constraint_reconstruction_bases_full is None:
            raise RuntimeError(
                "D3.2缺少outer PCA prior + broad residual恢复基底。"
            )

        relative_peak_intensity_constraint_state = (
            fit_relative_peak_intensity_constraint_state(
                training_normalized_spectra=(
                    normalized_full_spectra[training_indices]
                ),
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
                configuration=relative_peak_config,
            )
        )

        print(
            "D3.2自动峰区相对峰强约束：已启用；"
            "训练集自动峰数="
            f"{len(relative_peak_intensity_constraint_state['detected_peak_indices'])}；"
            "峰中心="
            f"{np.round(relative_peak_intensity_constraint_state['detected_peak_centers_cm1'], 3).tolist()}"
        )
    else:
        print("D3.2自动峰区相对峰强约束：未启用")

    # ------------------------------------------------------------------
    # D3.4 train-only full-spectrum peak parameter constraints
    # ------------------------------------------------------------------

    peak_parameter_constraint_state = None

    if peak_parameter_enabled:
        if broad_local_residual_state is None:
            raise RuntimeError(
                "D3.4缺少D2.6 broad_local_residual_state。"
            )

        if constraint_reconstruction_bases_full is None:
            raise RuntimeError(
                "D3.4缺少outer PCA prior + true broad residual恢复基底。"
            )

        peak_parameter_constraint_state = (
            fit_peak_parameter_constraint_state(
                training_normalized_spectra=(
                    normalized_full_spectra[
                        training_indices
                    ]
                ),
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
                configuration=peak_parameter_config,
            )
        )

        print(
            "D3.4完整重建谱峰参数物理约束：已启用；"
            "training-only自动稳定峰数="
            f"{len(peak_parameter_constraint_state['detected_peak_centers_cm1'])}"
        )

        print(
            "D3.4自动峰中心(cm^-1)：",
            np.round(
                np.asarray(
                    peak_parameter_constraint_state[
                        "detected_peak_centers_cm1"
                    ],
                    dtype=np.float64,
                ),
                3,
            ).tolist(),
        )

        print(
            "D3.4峰位下界：",
            np.round(
                np.asarray(
                    peak_parameter_constraint_state[
                        "position_lower_bound_cm1"
                    ],
                    dtype=np.float64,
                ),
                3,
            ).tolist(),
        )

        print(
            "D3.4峰位上界：",
            np.round(
                np.asarray(
                    peak_parameter_constraint_state[
                        "position_upper_bound_cm1"
                    ],
                    dtype=np.float64,
                ),
                3,
            ).tolist(),
        )

        print(
            "D3.4有效峰宽下界：",
            np.round(
                np.asarray(
                    peak_parameter_constraint_state[
                        "width_lower_bound_cm1"
                    ],
                    dtype=np.float64,
                ),
                3,
            ).tolist(),
        )

        print(
            "D3.4有效峰宽上界：",
            np.round(
                np.asarray(
                    peak_parameter_constraint_state[
                        "width_upper_bound_cm1"
                    ],
                    dtype=np.float64,
                ),
                3,
            ).tolist(),
        )

    else:
        print(
            "D3.4完整重建谱峰参数物理约束：未启用"
        )

    # ------------------------------------------------------------------
    # D3 diversity
    # ------------------------------------------------------------------

    diversity_constraint_state = None

    diversity_config = (
        configuration.get(
            "diversity_constraints",
            {
                "enabled": False,
            },
        )
    )

    if bool(
        diversity_config.get(
            "enabled",
            False,
        )
    ):
        diversity_constraint_state = (
            fit_sers_diversity_constraint_state(
                training_scaled_residuals=(
                    training_scaled_residuals
                ),
                configuration=(
                    diversity_config
                ),
            )
        )

    print_diversity_summary(
        diversity_constraint_state
    )

    # ------------------------------------------------------------------
    # D3 feature limiter
    # ------------------------------------------------------------------

    feature_peak_residual_limiter_state = (
        None
    )

    limiter_config = configuration.get(
        "feature_peak_residual_limiter",
        {
            "enabled": False,
        },
    )

    if bool(
        limiter_config.get(
            "enabled",
            False,
        )
    ):
        if (
            prior_residual_transformer
            is None
        ):
            raise ValueError(
                "feature_peak_residual_limiter"
                "要求先完成prior_residual拟合。"
            )

        feature_peak_residual_limiter_state = (
            fit_feature_peak_residual_limiter_state(
                training_normalized_spectra=(
                    normalized_full_spectra[
                        training_indices
                    ]
                ),
                prior_normalized_intensity=(
                    prior_residual_transformer.prior
                ),
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
                configuration=(
                    limiter_config
                ),
            )
        )

    print_feature_peak_residual_limiter_summary(
        feature_peak_residual_limiter_state
    )

    # ------------------------------------------------------------------
    # 网络长度补齐
    # ------------------------------------------------------------------

    padded_spectra = (
        length_adapter.adapt(
            spectra_for_model
        )
    )

    if not np.isfinite(
        padded_spectra
    ).all():
        raise RuntimeError(
            "模型输入包含NaN或无穷值。"
        )

    # PCA模式约束需要逐样本 reference prior。
    # fixed prior，包括新的 blended prior，不需要。
    padded_constraint_reference_priors = (
        None
    )

    if (
        prior_residual_transformer
        is not None
        and broad_local_residual_decomposer
        is None
        and prior_residual_transformer.prior_method
        == "pca_reconstruction"
    ):
        constraint_reference_priors = (
            prior_residual_transformer
            .reference_priors_for_spectra(
                normalized_full_spectra
            )
        )

        padded_constraint_reference_priors = (
            length_adapter.adapt(
                constraint_reference_priors
            )
        )

    if (
        peak_derivative_enabled
        or relative_peak_enabled
        or peak_parameter_enabled
    ):
        if constraint_reconstruction_bases_full is None:
            raise RuntimeError(
                "D3.1/D3.2/D3.4恢复基底尚未构建。"
            )
        padded_constraint_reference_priors = length_adapter.adapt(
            constraint_reconstruction_bases_full
        )

    spectrum_dataset = (
        SpectrumDataset(
            padded_spectra,
            constraint_reference_priors=(
                padded_constraint_reference_priors
            ),
        )
    )

    training_dataset = Subset(
        spectrum_dataset,
        training_indices.tolist(),
    )

    validation_dataset = Subset(
        spectrum_dataset,
        validation_indices.tolist(),
    )

    test_dataset = Subset(
        spectrum_dataset,
        test_indices.tolist(),
    )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    device = resolve_device(
        str(
            training_config[
                "device"
            ]
        )
    )

    number_of_workers = int(
        training_config.get(
            "number_of_workers",
            0,
        )
    )

    if number_of_workers < 0:
        raise ValueError(
            "training.number_of_workers不能小于0。"
        )

    pin_memory = (
        bool(
            training_config.get(
                "pin_memory",
                False,
            )
        )
        and device.type == "cuda"
    )

    batch_size = int(
        training_config[
            "batch_size"
        ]
    )

    if batch_size <= 0:
        raise ValueError(
            "training.batch_size必须大于0。"
        )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=bool(
            training_config.get(
                "drop_last",
                False,
            )
        ),
        worker_init_fn=(
            seed_data_loader_worker
        ),
        generator=(
            create_data_loader_generator(
                random_seed
            )
        ),
        persistent_workers=(
            number_of_workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(
            seed_data_loader_worker
        ),
        persistent_workers=(
            number_of_workers > 0
        ),
    )

    steps_per_epoch = len(
        training_loader
    )

    if steps_per_epoch <= 0:
        raise RuntimeError(
            "训练集没有产生任何batch；"
            "请检查batch_size和drop_last。"
        )

    # ------------------------------------------------------------------
    # epoch → step
    # ------------------------------------------------------------------

    number_of_epochs = int(
        training_config[
            "number_of_epochs"
        ]
    )

    validate_every_epochs = int(
        training_config[
            "validate_every_epochs"
        ]
    )

    save_every_epochs = int(
        training_config[
            "save_every_epochs"
        ]
    )

    log_every_batches = int(
        training_config[
            "log_every_batches"
        ]
    )

    if min(
        number_of_epochs,
        validate_every_epochs,
        save_every_epochs,
        log_every_batches,
    ) <= 0:
        raise ValueError(
            "训练epoch、验证频率、"
            "保存频率和日志频率必须大于0。"
        )

    training_config[
        "total_training_steps"
    ] = (
        number_of_epochs
        * steps_per_epoch
    )

    training_config[
        "validate_every_steps"
    ] = (
        validate_every_epochs
        * steps_per_epoch
    )

    training_config[
        "checkpoint_every_steps"
    ] = (
        save_every_epochs
        * steps_per_epoch
    )

    training_config[
        "log_every_steps"
    ] = log_every_batches

    # ------------------------------------------------------------------
    # 构建扩散模型
    # ------------------------------------------------------------------

    _, diffusion = (
        build_diffusion_model(
            model_configuration=configuration,
            sequence_length=(
                length_adapter.padded_length
            ),
        )
    )

    if (
        physics_constraint_state
        is not None
    ):
        configure_physics = getattr(
            diffusion,
            "configure_physics_constraints",
            None,
        )

        if not callable(
            configure_physics
        ):
            raise RuntimeError(
                "D3扩散模型缺少"
                "configure_physics_constraints。"
            )

        configure_physics(
            physics_constraint_state=(
                physics_constraint_state
            ),
            prior_residual_state=(
                prior_residual_state
            ),
        )

    if peak_derivative_constraint_state is not None:
        configure_peak_derivative = getattr(
            diffusion,
            "configure_peak_derivative_constraints",
            None,
        )
        if not callable(configure_peak_derivative):
            raise RuntimeError(
                "D3.1扩散模型缺少"
                "configure_peak_derivative_constraints。"
            )
        configure_peak_derivative(
            peak_derivative_constraint_state=(
                peak_derivative_constraint_state
            ),
            broad_local_residual_state=broad_local_residual_state,
        )

    if relative_peak_intensity_constraint_state is not None:
        configure_relative_peak = getattr(
            diffusion,
            "configure_relative_peak_intensity_constraints",
            None,
        )
        if not callable(configure_relative_peak):
            raise RuntimeError(
                "D3.2扩散模型缺少"
                "configure_relative_peak_intensity_constraints。"
            )
        configure_relative_peak(
            relative_peak_intensity_constraint_state=(
                relative_peak_intensity_constraint_state
            ),
            broad_local_residual_state=broad_local_residual_state,
        )

    if peak_parameter_constraint_state is not None:
        configure_peak_parameter = getattr(
            diffusion,
            "configure_peak_parameter_constraints",
            None,
        )

        if not callable(
            configure_peak_parameter
        ):
            raise RuntimeError(
                "D3.4扩散模型缺少"
                "configure_peak_parameter_constraints。"
            )

        configure_peak_parameter(
            peak_parameter_constraint_state=(
                peak_parameter_constraint_state
            ),
            broad_local_residual_state=(
                broad_local_residual_state
            ),
        )

    if (
        diversity_constraint_state
        is not None
    ):
        configure_diversity = getattr(
            diffusion,
            "configure_diversity_constraints",
            None,
        )

        if not callable(
            configure_diversity
        ):
            raise RuntimeError(
                "D3扩散模型缺少"
                "configure_diversity_constraints。"
            )

        configure_diversity(
            diversity_constraint_state=(
                diversity_constraint_state
            ),
        )

    # ------------------------------------------------------------------
    # D3.4真实training batch约束激活/梯度检查
    #
    # 该检查：
    # 1. 不读取training_loader，因此不会消耗正式shuffle顺序；
    # 2. 不执行optimizer.step；
    # 3. 不创建checkpoint/logger/trainer；
    # 4. 只检查D3.4是否真正产生有限、非零梯度。
    # ------------------------------------------------------------------

    if arguments.constraint_check_only:
        if not peak_parameter_enabled:
            raise RuntimeError(
                "--constraint-check-only要求"
                "peak_parameter_constraints.enabled=true。"
            )

        if arguments.resume is not None:
            raise RuntimeError(
                "--constraint-check-only不允许同时使用--resume。"
            )

        if peak_parameter_constraint_state is None:
            raise RuntimeError(
                "D3.4 constraint state尚未配置。"
            )

        peak_parameter_loss_module = getattr(
            diffusion,
            "peak_parameter_loss_module",
            None,
        )

        if peak_parameter_loss_module is None:
            raise RuntimeError(
                "扩散模型中没有D3.4 peak_parameter_loss_module。"
            )

        check_batch_size = min(
            int(batch_size),
            len(training_dataset),
        )

        if check_batch_size <= 0:
            raise RuntimeError(
                "D3.4自检没有可用training样本。"
            )

        check_items = [
            training_dataset[index]
            for index in range(check_batch_size)
        ]

        if not all(
            isinstance(item, dict)
            for item in check_items
        ):
            raise RuntimeError(
                "D3.4自检要求SpectrumDataset返回"
                "spectrum + constraint_reference_prior。"
            )

        if not all(
            "spectrum" in item
            and "constraint_reference_prior" in item
            for item in check_items
        ):
            raise RuntimeError(
                "D3.4自检batch缺少"
                "spectrum或constraint_reference_prior。"
            )

        check_spectra = torch.stack(
            [
                item["spectrum"]
                for item in check_items
            ],
            dim=0,
        ).to(
            device=device,
            dtype=torch.float32,
        )

        check_reference_prior = torch.stack(
            [
                item["constraint_reference_prior"]
                for item in check_items
            ],
            dim=0,
        ).to(
            device=device,
            dtype=torch.float32,
        )

        diffusion = diffusion.to(
            device
        )

        diffusion.train()

        # ------------------------------------------------------
        # 使用低至中噪声时间步。
        #
        # 不调用diffusion.forward()随机抽t，
        # 避免刚好全部抽到D3.4关闭区域而形成假阴性。
        # ------------------------------------------------------

        maximum_active_fraction = float(
            peak_parameter_config[
                "maximum_active_timestep_fraction"
            ]
        )

        maximum_active_timestep = int(
            np.floor(
                (
                    int(diffusion.num_timesteps)
                    - 1
                )
                * maximum_active_fraction
            )
        )

        maximum_active_timestep = max(
            0,
            maximum_active_timestep,
        )

        if check_batch_size == 1:
            check_timesteps = torch.zeros(
                1,
                device=device,
                dtype=torch.long,
            )
        else:
            check_timesteps = torch.linspace(
                0,
                maximum_active_timestep,
                steps=check_batch_size,
                device=device,
            ).round().long()

        # 自检使用独立随机状态。
        # 因为检查完成后立即return，
        # 不影响正式训练随机序列。
        check_seed = int(
            random_seed
        ) + 3404

        torch.manual_seed(
            check_seed
        )

        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                check_seed
            )

        check_noise = torch.randn_like(
            check_spectra
        )

        diffusion.zero_grad(
            set_to_none=True
        )

        check_total_loss = diffusion.p_losses(
            check_spectra,
            check_timesteps,
            noise=check_noise,
            constraint_reference_prior=(
                check_reference_prior
            ),
        )

        if not torch.isfinite(
            check_total_loss
        ):
            raise RuntimeError(
                "D3.4自检total loss出现NaN或无穷值。"
            )

        live_components = getattr(
            diffusion,
            "_latest_loss_components",
            None,
        )

        if not isinstance(
            live_components,
            dict,
        ):
            raise RuntimeError(
                "无法取得D3.4实时loss components。"
            )

        required_component_names = (
            "ddpm_loss",
            "peak_parameter_loss",
            "peak_parameter_candidate_loss",
            "peak_parameter_raw_loss",
            "peak_parameter_position_loss",
            "peak_parameter_width_loss",
            "peak_parameter_loss_cap",
            "peak_parameter_loss_scale",
            "mean_peak_position_violation_cm1",
            "mean_peak_width_violation_cm1",
            "peak_position_violation_fraction",
            "peak_width_violation_fraction",
            "target_peak_position_violation_fraction",
            "target_peak_width_violation_fraction",
            "mean_peak_parameter_timestep_weight",
        )

        missing_components = [
            name
            for name in required_component_names
            if name not in live_components
        ]

        if missing_components:
            raise RuntimeError(
                "D3.4自检缺少loss components："
                f"{missing_components}"
            )

        live_ddpm_loss = (
            live_components[
                "ddpm_loss"
            ]
        )

        live_peak_parameter_loss = (
            live_components[
                "peak_parameter_loss"
            ]
        )

        if not live_ddpm_loss.requires_grad:
            raise RuntimeError(
                "DDPM loss没有梯度图。"
            )

        if not live_peak_parameter_loss.requires_grad:
            raise RuntimeError(
                "D3.4 loss没有梯度图。"
            )

        # ------------------------------------------------------
        # 独立计算某个loss对U-Net参数的梯度norm。
        #
        # 这样不是只检查loss非零，
        # 而是真正确认它能改变模型参数。
        # ------------------------------------------------------

        def calculate_model_gradient_norm(
            loss: torch.Tensor,
            *,
            retain_graph: bool,
        ) -> tuple[
            float,
            bool,
            int,
        ]:
            diffusion.zero_grad(
                set_to_none=True
            )

            loss.backward(
                retain_graph=retain_graph
            )

            squared_norm = 0.0
            all_finite = True
            gradient_parameter_count = 0

            for parameter in diffusion.model.parameters():
                gradient = parameter.grad

                if gradient is None:
                    continue

                gradient_parameter_count += 1

                finite_here = bool(
                    torch.isfinite(
                        gradient
                    ).all().item()
                )

                all_finite = (
                    all_finite
                    and finite_here
                )

                if finite_here:
                    squared_norm += float(
                        gradient
                        .detach()
                        .float()
                        .square()
                        .sum()
                        .item()
                    )

            gradient_norm = (
                squared_norm ** 0.5
            )

            return (
                float(gradient_norm),
                bool(all_finite),
                int(
                    gradient_parameter_count
                ),
            )

        (
            ddpm_gradient_norm,
            ddpm_gradient_finite,
            ddpm_gradient_parameter_count,
        ) = calculate_model_gradient_norm(
            live_ddpm_loss,
            retain_graph=True,
        )

        (
            peak_parameter_gradient_norm,
            peak_parameter_gradient_finite,
            peak_parameter_gradient_parameter_count,
        ) = calculate_model_gradient_norm(
            live_peak_parameter_loss,
            retain_graph=False,
        )

        diffusion.zero_grad(
            set_to_none=True
        )

        def scalar(
            name: str,
        ) -> float:
            return float(
                live_components[
                    name
                ]
                .detach()
                .float()
                .item()
            )

        ddpm_value = scalar(
            "ddpm_loss"
        )

        peak_parameter_value = scalar(
            "peak_parameter_loss"
        )

        peak_parameter_candidate = scalar(
            "peak_parameter_candidate_loss"
        )

        peak_parameter_raw = scalar(
            "peak_parameter_raw_loss"
        )

        epsilon = 1.0e-12

        candidate_to_ddpm_ratio = (
            peak_parameter_candidate
            / max(
                abs(ddpm_value),
                epsilon,
            )
        )

        actual_to_ddpm_ratio = (
            peak_parameter_value
            / max(
                abs(ddpm_value),
                epsilon,
            )
        )

        gradient_norm_ratio = (
            peak_parameter_gradient_norm
            / max(
                ddpm_gradient_norm,
                epsilon,
            )
        )

        detected_peak_count = len(
            peak_parameter_constraint_state[
                "detected_peak_centers_cm1"
            ]
        )

        print()
        print(
            "===== D3.4 constraint activation check ====="
        )

        print(
            "检查用途：真实training batch，"
            "不执行optimizer.step，不开始正式训练"
        )

        print(
            "check batch size：",
            check_batch_size,
        )

        print(
            "check timesteps：",
            check_timesteps
            .detach()
            .cpu()
            .tolist(),
        )

        print(
            "detected peak count：",
            detected_peak_count,
        )

        print(
            "ddpm loss：",
            f"{ddpm_value:.8e}",
        )

        print(
            "peak parameter raw loss：",
            f"{peak_parameter_raw:.8e}",
        )

        print(
            "position loss：",
            f"{scalar('peak_parameter_position_loss'):.8e}",
        )

        print(
            "width loss：",
            f"{scalar('peak_parameter_width_loss'):.8e}",
        )

        print(
            "candidate loss：",
            f"{peak_parameter_candidate:.8e}",
        )

        print(
            "actual capped D3.4 loss：",
            f"{peak_parameter_value:.8e}",
        )

        print(
            "loss cap：",
            f"{scalar('peak_parameter_loss_cap'):.8e}",
        )

        print(
            "loss scale：",
            f"{scalar('peak_parameter_loss_scale'):.8e}",
        )

        print(
            "candidate / DDPM：",
            f"{candidate_to_ddpm_ratio:.8e}",
        )

        print(
            "actual D3.4 / DDPM：",
            f"{actual_to_ddpm_ratio:.8e}",
        )

        print(
            "mean position violation (cm^-1)：",
            f"{scalar('mean_peak_position_violation_cm1'):.8e}",
        )

        print(
            "mean width violation (cm^-1)：",
            f"{scalar('mean_peak_width_violation_cm1'):.8e}",
        )

        print(
            "position violation fraction：",
            f"{scalar('peak_position_violation_fraction'):.8e}",
        )

        print(
            "width violation fraction：",
            f"{scalar('peak_width_violation_fraction'):.8e}",
        )

        print(
            "target position violation fraction：",
            f"{scalar('target_peak_position_violation_fraction'):.8e}",
        )

        print(
            "target width violation fraction：",
            f"{scalar('target_peak_width_violation_fraction'):.8e}",
        )

        print(
            "mean timestep weight：",
            f"{scalar('mean_peak_parameter_timestep_weight'):.8e}",
        )

        print(
            "DDPM gradient norm：",
            f"{ddpm_gradient_norm:.8e}",
        )

        print(
            "D3.4 gradient norm：",
            f"{peak_parameter_gradient_norm:.8e}",
        )

        print(
            "D3.4/DDPM gradient norm ratio：",
            f"{gradient_norm_ratio:.8e}",
        )

        print(
            "DDPM gradient finite：",
            ddpm_gradient_finite,
        )

        print(
            "D3.4 gradient finite：",
            peak_parameter_gradient_finite,
        )

        print(
            "DDPM gradient parameter count：",
            ddpm_gradient_parameter_count,
        )

        print(
            "D3.4 gradient parameter count：",
            peak_parameter_gradient_parameter_count,
        )

        print(
            "D3.4 gradient nonzero：",
            peak_parameter_gradient_norm
            > epsilon,
        )

        print(
            "============================================"
        )
        print()

        failures = []

        if not ddpm_gradient_finite:
            failures.append(
                "DDPM梯度存在NaN/Inf"
            )

        if not peak_parameter_gradient_finite:
            failures.append(
                "D3.4梯度存在NaN/Inf"
            )

        if ddpm_gradient_norm <= epsilon:
            failures.append(
                "DDPM梯度norm接近0"
            )

        if peak_parameter_raw <= epsilon:
            failures.append(
                "D3.4 raw loss接近0，"
                "真实模型输出未激活物理约束"
            )

        if peak_parameter_value <= epsilon:
            failures.append(
                "实际加入total loss的D3.4 loss接近0"
            )

        if peak_parameter_gradient_norm <= epsilon:
            failures.append(
                "D3.4对U-Net的梯度norm接近0"
            )

        configured_maximum_ratio = float(
            peak_parameter_config[
                "maximum_total_ratio_to_ddpm"
            ]
        )

        if (
            actual_to_ddpm_ratio
            >
            configured_maximum_ratio
            + 1.0e-6
        ):
            failures.append(
                "D3.4实际loss超过配置的DDPM占比上限"
            )

        if failures:
            print(
                "D3.4约束激活检查：未通过"
            )

            for failure in failures:
                print(
                    " -",
                    failure,
                )

            raise RuntimeError(
                "D3.4 constraint activation check失败；"
                "不要开始2000-step正式训练。"
            )

        print(
            "D3.4约束激活检查：通过"
        )

        print(
            "注意：通过只代表约束真实参与反向传播，"
            "不代表模型最终生成质量已经改善。"
        )

        return


    # ------------------------------------------------------------------
    # checkpoint / metadata
    # ------------------------------------------------------------------

    checkpoint_manager = (
        CheckpointManager(
            resolve_project_path(
                configuration,
                output_config[
                    "checkpoint_directory"
                ],
            )
        )
    )

    logger = TrainingLogger(
        resolve_training_log_file(
            configuration
        )
    )

    axis_metadata = (
        build_axis_metadata(
            labels=collection.labels,
            relative_source_files=(
                collection.relative_source_files
            ),
            raman_shifts=(
                collection.raman_shifts
            ),
        )
    )

    metadata = {
        "diagnostic_overfit": (
            overfit_metadata
        ),
        "backend_package": (
            "denoising-diffusion-pytorch"
        ),
        "backend_version": (
            get_backend_version()
        ),
        "spectrum_names": [
            str(value)
            for value
            in collection.spectrum_names
        ],
        "source_files": [
            str(value)
            for value
            in collection.source_files
        ],
        "relative_source_files": [
            str(value)
            for value
            in collection.relative_source_files
        ],
        "labels": [
            str(value)
            for value
            in collection.labels
        ],
        "number_of_spectra": int(
            number_of_spectra
        ),
        "training_indices": (
            training_indices.tolist()
        ),
        "validation_indices": (
            validation_indices.tolist()
        ),
        "test_indices": (
            test_indices.tolist()
        ),
        "split_unit": str(
            data_config[
                "split_unit"
            ]
        ),
        "normalization_state": (
            normalization_state
        ),
        "prior_residual_state": (
            prior_residual_state
        ),
        "broad_local_residual_state": (
            broad_local_residual_state
        ),
        "physics_constraint_state": (
            physics_constraint_state
        ),
        "peak_derivative_constraint_state": (
            peak_derivative_constraint_state
        ),
        "relative_peak_intensity_constraint_state": (
            relative_peak_intensity_constraint_state
        ),
        "peak_parameter_constraint_state": (
            peak_parameter_constraint_state
        ),
        "diversity_constraint_state": (
            diversity_constraint_state
        ),
        "feature_peak_residual_limiter_state": (
            feature_peak_residual_limiter_state
        ),
        "axis_metadata": (
            axis_metadata
        ),
        **length_adapter.to_metadata(),
    }

    trainer = DdpmTrainer(
        diffusion=diffusion,
        training_loader=training_loader,
        validation_loader=validation_loader,
        device=device,
        configuration=configuration,
        metadata=metadata,
        checkpoint_manager=(
            checkpoint_manager
        ),
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    if arguments.resume is not None:
        resume_path = (
            resolve_resume_path(
                configuration=configuration,
                resume_argument=arguments.resume,
            )
        )

        validate_resume_stage(
            checkpoint_path=resume_path,
            configuration=configuration,
        )

        trainer.resume(
            resume_path
        )

    # ------------------------------------------------------------------
    # 终端摘要
    # ------------------------------------------------------------------

    labels = sorted(
        {
            str(value)
            for value
            in collection.labels
        }
    )

    original_lengths = sorted(
        {
            int(
                np.asarray(
                    axis
                ).size
            )
            for axis
            in collection.raman_shifts
        }
    )

    print(
        "\n===== 开始训练 ====="
    )

    print(
        f"设备：{device}"
    )

    print(
        "实验名称："
        f"{project_config['name']}"
    )

    print(
        "文件夹标签："
        f"{'、'.join(labels)}"
    )

    print(
        "总光谱数量："
        f"{number_of_spectra}"
    )

    print(
        "训练/验证/测试光谱数量："
        f"{len(training_dataset)}/"
        f"{len(validation_dataset)}/"
        f"{len(test_dataset)}"
    )

    print(
        "各原始拉曼轴点数："
        f"{original_lengths}"
    )

    print(
        "统一训练轴长度："
        f"{length_adapter.original_length}"
    )

    print(
        "模型输入长度："
        f"{length_adapter.padded_length}"
    )

    print(
        "末尾补齐点数："
        f"{length_adapter.padding_size}"
    )

    print(
        "每个epoch的step："
        f"{steps_per_epoch}"
    )

    print(
        "总训练step："
        f"{training_config['total_training_steps']}"
    )

    print(
        "真实输入光谱额外平滑或去基线：不执行"
    )

    if (
        prior_residual_transformer
        is not None
        and prior_residual_transformer.prior_method
        == "training_blended_frequency_median"
    ):
        print(
            "D2.4说明：Gaussian低通只用于"
            "train-only median prior的分解，"
            "不会直接平滑任何真实训练/验证/测试光谱。"
        )

        print(
            "D2.4共同峰骨架保留比例："
            f"{prior_residual_transformer.blended_peak_component_ratio:g}"
        )

    if broad_local_residual_decomposer is not None:
        broad_local_summary = (
            broad_local_residual_decomposer.summary()
        )

        print(
            "D2.6 broad-local residual：已启用"
        )

        print(
            "D2.6 DDPM学习域：scaled local residual"
        )

        print(
            "D2.6 broad sampler：train-only PCA；"
            f"组件数={broad_local_summary['broad_pca_components']}；"
            "累计解释方差="
            f"{broad_local_summary['broad_pca_cumulative_explained_variance']:.6f}"
        )

    if not bool(
        physics_config.get(
            "enabled",
            False,
        )
    ):
        print(
            "D3物理/峰约束：未启用"
        )

    if normalizer is not None:
        print(
            "训练集原始强度范围："
            f"{float(normalizer.data_min):.8g}–"
            f"{float(normalizer.data_max):.8g}"
        )

        print(
            "模型缩放残差输入范围："
            f"{float(padded_spectra.min()):.8g}–"
            f"{float(padded_spectra.max()):.8g}"
        )

    if overfit_metadata is not None:
        print(
            "实验模式：单光谱重复过拟合诊断"
        )

        print(
            "注意：该模式不能评价泛化性能。"
        )

    print(
        "====================\n"
    )

    trainer.train()


if __name__ == "__main__":
    main()

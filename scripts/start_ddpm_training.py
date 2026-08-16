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
            "D3物理/峰约束：未启用"
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
                "D2.6第一轮消融要求关闭全部D3模块；"
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
        physics_constraint_state
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
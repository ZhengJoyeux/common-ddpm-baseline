"""
训练 D0-D3 一维 SERS DDPM。

完整项目测试

set -e

echo "===== 1. Python语法检查 ====="
python -m compileall -q \
    src \
    scripts \
    tests \
    run.py

echo "===== 2. 软件包依赖检查 ====="
python -m pip check

echo "===== 3. 项目自动化测试 ====="
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
from torch.utils.data import DataLoader, Subset

from src.checkpoint_manager import (
    CheckpointManager,
    build_axis_metadata,
    load_checkpoint_file,
)
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.dataset_splitter import split_spectrum_collection
from src.ddpm_trainer import DdpmTrainer
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.model_builder import build_diffusion_model
from src.one_dimensional_ddpm import get_backend_version
from src.prior_residual import PriorResidualTransformer
from src.random_seed_manager import (
    create_data_loader_generator,
    seed_data_loader_worker,
    set_random_seed,
)
from src.sers_physics_constraints import (
    fit_sers_physics_constraint_state,
)
from src.spectrum_dataset import SpectrumDataset
from src.spectrum_file_reader import (
    SpectrumCollection,
    read_spectrum_collection,
)
from src.spectrum_length_adapter import SpectrumLengthAdapter
from src.training_logger import TrainingLogger


def parse_arguments() -> argparse.Namespace:
    """读取训练入口命令行参数。"""

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
        help="可选：从同一模型阶段的checkpoint继续训练。",
    )

    return parser.parse_args()


def resolve_device(device_text: str) -> torch.device:
    """解析训练设备并检查 CUDA 是否可用。"""

    normalized = str(device_text).strip().lower()

    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "配置要求使用CUDA，但PyTorch未检测到可用GPU。"
        )

    return torch.device(normalized)


def validate_split_indices(
    *,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    number_of_spectra: int,
) -> None:
    """确认三个子集非空、无重复并完整覆盖全部光谱。"""

    named_indices = {
        "训练集": training_indices,
        "验证集": validation_indices,
        "测试集": test_indices,
    }

    for subset_name, indices in named_indices.items():
        if indices.size == 0:
            raise RuntimeError(f"{subset_name}为空。")

        if (
            np.any(indices < 0)
            or np.any(indices >= number_of_spectra)
        ):
            raise RuntimeError(f"{subset_name}包含越界索引。")

        if np.unique(indices).size != indices.size:
            raise RuntimeError(f"{subset_name}包含重复索引。")

    all_indices = np.concatenate(
        [
            training_indices,
            validation_indices,
            test_indices,
        ]
    )

    if (
        all_indices.size != number_of_spectra
        or np.unique(all_indices).size != number_of_spectra
    ):
        raise RuntimeError(
            "训练集、验证集和测试集没有无重复覆盖全部光谱。"
        )


def resolve_training_log_file(configuration: dict) -> Path:
    """兼容训练日志的历史路径字段。"""

    output = configuration["output"]

    if "training_log_file" in output:
        value = output["training_log_file"]
    else:
        value = (
            Path(output["log_directory"])
            / str(
                output.get(
                    "training_log_name",
                    "training_log.csv",
                )
            )
        )

    return resolve_project_path(configuration, value)


def build_single_spectrum_overfit_collection(
    *,
    collection: SpectrumCollection,
    diagnostic_config: dict,
) -> tuple[SpectrumCollection, dict[str, object] | None]:
    """
    保留项目原有单光谱重复过拟合诊断。

    该模式只验证数据流、模型和采样能否记住一条光谱，不代表泛化性能。
    """

    if not bool(diagnostic_config.get("enabled", False)):
        return collection, None

    spectrum_index = int(
        diagnostic_config.get("spectrum_index", 0)
    )
    repeat_count = int(
        diagnostic_config.get("repeat_count", 50)
    )
    original_count = len(collection.spectrum_names)

    if original_count == 0:
        raise RuntimeError("原始数据中没有可用光谱。")

    if not 0 <= spectrum_index < original_count:
        raise IndexError(
            "diagnostic_overfit.spectrum_index越界："
            f"当前有效范围为0到{original_count - 1}。"
        )

    if repeat_count < 10:
        raise ValueError(
            "diagnostic_overfit.repeat_count至少为10。"
        )

    selected_spectrum = np.asarray(
        collection.spectra[spectrum_index],
        dtype=np.float32,
    ).reshape(-1)
    selected_axis = np.asarray(
        collection.raman_shifts[spectrum_index],
        dtype=np.float64,
    ).reshape(-1)

    if selected_spectrum.size != selected_axis.size:
        raise RuntimeError("所选光谱和拉曼轴长度不一致。")

    if selected_axis.size < 2:
        raise RuntimeError("所选拉曼轴至少需要2个点。")

    if (
        not np.isfinite(selected_spectrum).all()
        or not np.isfinite(selected_axis).all()
    ):
        raise RuntimeError("所选光谱或拉曼轴包含NaN/无穷值。")

    if not np.all(np.diff(selected_axis) > 0.0):
        raise RuntimeError("所选拉曼轴必须严格递增。")

    selected_name = str(
        collection.spectrum_names[spectrum_index]
    )
    selected_source_file = collection.source_files[spectrum_index]
    selected_relative_source_file = (
        collection.relative_source_files[spectrum_index]
    )
    selected_label = collection.labels[spectrum_index]

    repeated_collection = SpectrumCollection(
        raman_shift=selected_axis.copy(),
        spectra=np.repeat(
            selected_spectrum[None, :],
            repeat_count,
            axis=0,
        ).astype(np.float32, copy=False),
        raman_shifts=np.repeat(
            selected_axis[None, :],
            repeat_count,
            axis=0,
        ).astype(np.float64, copy=False),
        original_lengths=np.full(
            repeat_count,
            selected_axis.size,
            dtype=np.int64,
        ),
        source_files=np.asarray(
            [selected_source_file] * repeat_count,
            dtype=object,
        ),
        relative_source_files=np.asarray(
            [selected_relative_source_file] * repeat_count,
            dtype=object,
        ),
        spectrum_names=np.asarray(
            [
                f"{selected_name}__repeat_{index + 1:04d}"
                for index in range(repeat_count)
            ],
            dtype=object,
        ),
        labels=np.asarray(
            [selected_label] * repeat_count,
            dtype=object,
        ),
    )

    metadata: dict[str, object] = {
        "enabled": True,
        "formal_validation": False,
        "original_number_of_spectra": int(original_count),
        "selected_original_index": int(spectrum_index),
        "selected_spectrum_name": selected_name,
        "selected_source_file": str(selected_source_file),
        "selected_relative_source_file": str(
            selected_relative_source_file
        ),
        "selected_label": str(selected_label),
        "selected_original_length": int(selected_axis.size),
        "repeat_count": int(repeat_count),
    }

    return repeated_collection, metadata


def resolve_resume_path(
    *,
    configuration: dict,
    resume_argument: str,
) -> Path:
    """把 --resume 参数解析为绝对路径。"""

    resume_path = Path(resume_argument).expanduser()

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
    """防止把 D2.1 checkpoint 当作 D3.1 断点直接续训。"""

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )
    checkpoint_configuration = checkpoint.get("configuration")

    if not isinstance(checkpoint_configuration, dict):
        raise RuntimeError("resume检查点缺少有效configuration。")

    current_physics = configuration.get(
        "physics_constraints",
        {},
    ) or {}
    checkpoint_physics = checkpoint_configuration.get(
        "physics_constraints",
        {},
    ) or {}

    current_enabled = bool(current_physics.get("enabled", False))
    checkpoint_enabled = bool(
        checkpoint_physics.get("enabled", False)
    )

    if current_enabled != checkpoint_enabled:
        raise ValueError(
            "当前配置与resume检查点的物理约束阶段不一致。"
            "D3.1不能从D2.1 checkpoint直接--resume；"
            "D3.1断点续训必须使用同一D3.1实验的checkpoint。"
        )

    if current_enabled and current_physics != checkpoint_physics:
        raise ValueError(
            "当前D3.1物理约束配置与resume检查点不一致。"
            "请保持物理损失、阈值和权重完全相同。"
        )


def print_prior_residual_summary(
    transformer: PriorResidualTransformer,
    transformed_training_residuals: np.ndarray,
) -> None:
    """打印 D2.1 训练集残差缩放状态。"""

    print("\n===== D2.1先验残差状态 =====")
    print(f"残差归一化方法：{transformer.normalization_method}")

    statistics = (
        transformer.training_abs_residual_percentiles or {}
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
            print(f"训练原始残差 {key}: {statistics[key]:.8g}")

    if transformer.normalization_method == "pointwise_mad_asinh":
        print(
            "逐波数MAD换算系数："
            f"{transformer.mad_scale_factor:.8g}"
        )
        print(
            "逐波数尺度下限："
            f"{transformer.pointwise_scale_floor:.8g}"
        )

        pointwise_statistics = (
            transformer.training_pointwise_scale_percentiles or {}
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

    absolute_values = np.abs(transformed_training_residuals)
    percentiles = np.percentile(
        absolute_values,
        [50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 100.0],
    )

    for name, value in zip(
        ("p50", "p90", "p95", "p99", "p99.5", "p99.9", "max"),
        percentiles,
        strict=True,
    ):
        print(f"变换后残差 {name}: {float(value):.8g}")

    print("==============================\n")


def print_physics_summary(
    physics_constraint_state: dict[str, object] | None,
) -> None:
    """打印 D3.1 自适应峰检测和异常约束状态。"""

    if not physics_constraint_state or not bool(
        physics_constraint_state.get("enabled", False)
    ):
        print("D3.1物理约束：未启用")
        return

    configuration = physics_constraint_state["configuration"]
    detection = configuration["peak_detection"]

    print("\n===== D3.1目标自适应SERS物理约束 =====")
    print("固定特征峰位置：无")
    print("固定peak_windows：无")
    print("峰选择方式：每条真实目标光谱独立动态检测")
    print(
        "训练光谱数量："
        f"{physics_constraint_state['number_of_training_spectra']}"
    )
    print(
        "峰位零惩罚范围：±"
        f"{configuration['peak_position']['zero_penalty_tolerance_cm1']:g} "
        "cm^-1"
    )
    print(
        "每条光谱最多约束峰数："
        f"{detection['maximum_peaks_per_spectrum']}"
    )
    print(
        "自适应局部分析半宽：±"
        f"{detection['analysis_half_width_cm1']:g} cm^-1"
    )
    print(
        "训练集峰显著性下限："
        f"{physics_constraint_state['training_prominence_floor']:.8g}"
    )
    print(
        "一阶导数软上限："
        f"{physics_constraint_state['first_derivative_abs_limit']:.8g}"
    )
    print(
        "二阶导数软上限："
        f"{physics_constraint_state['second_derivative_abs_limit']:.8g}"
    )
    print(
        "完整归一化强度软范围："
        f"{physics_constraint_state['allowed_intensity_minimum']:.8g}–"
        f"{physics_constraint_state['allowed_intensity_maximum']:.8g}"
    )
    print(
        "D2.1缩放残差绝对值软上限："
        f"{physics_constraint_state['scaled_residual_abs_limit']:.8g}"
    )
    print(f"物理总权重：{configuration['total_weight']:g}")
    print("========================================\n")


def main() -> None:
    """执行数据读取、划分、D2.1拟合、D3.1拟合和正式训练。"""

    arguments = parse_arguments()
    configuration = load_configuration(arguments.config)

    project_config = configuration["project"]
    random_config = configuration.get("random", {})
    data_config = configuration["data"]
    model_config = configuration["model"]
    normalization_config = configuration["normalization"]
    training_config = configuration["training"]
    output_config = configuration["output"]
    diagnostic_config = configuration.get(
        "diagnostic_overfit",
        {},
    )
    physics_config = configuration.get(
        "physics_constraints",
        {"enabled": False},
    )

    if not isinstance(diagnostic_config, dict):
        raise TypeError("diagnostic_overfit必须是字典。")

    random_seed = int(
        random_config.get(
            "seed",
            project_config.get("random_seed", 42),
        )
    )
    set_random_seed(
        random_seed=random_seed,
        deterministic=bool(
            random_config.get("deterministic", False)
        ),
    )

    input_directory = resolve_project_path(
        configuration,
        data_config["input_directory"],
    )
    collection = read_spectrum_collection(
        input_directory=input_directory,
        data_config=data_config,
    )

    collection, overfit_metadata = (
        build_single_spectrum_overfit_collection(
            collection=collection,
            diagnostic_config=diagnostic_config,
        )
    )

    if overfit_metadata is not None:
        if arguments.resume is not None:
            raise ValueError(
                "单光谱过拟合诊断必须从头训练，不能使用--resume。"
            )

        # 只修改内存中的配置，不改写 YAML 文件。
        data_config = dict(data_config)
        data_config["split_unit"] = "spectrum"
        data_config["shuffle"] = False
        configuration["data"] = data_config

    number_of_spectra = len(collection.spectrum_names)

    if number_of_spectra == 0:
        raise RuntimeError("没有读取到任何光谱。")

    dataset_split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=random_seed,
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

    # 建立统一拉曼训练轴，并把网络长度补齐到 U-Net 下采样倍数。
    length_adapter = SpectrumLengthAdapter.create(
        raman_shifts=collection.raman_shifts,
        dimension_multipliers=model_config[
            "dimension_multipliers"
        ],
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
            data_config.get("padding_value", 0.0)
        ),
        raman_range_tolerance=float(
            data_config.get("raman_range_tolerance", 1.0)
        ),
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

    if spectra_on_model_axis.shape != expected_shape:
        raise RuntimeError(
            "插值后的光谱形状不正确："
            f"实际为{spectra_on_model_axis.shape}，"
            f"期望为{expected_shape}。"
        )

    if not np.isfinite(spectra_on_model_axis).all():
        raise RuntimeError("插值后的光谱包含NaN或无穷值。")

    # ------------------------------------------------------------------
    # 训练集 global_minmax 归一化
    # ------------------------------------------------------------------

    normalizer: GlobalMinMaxNormalizer | None = None
    normalization_state = None

    if bool(normalization_config.get("enabled", False)):
        normalizer = GlobalMinMaxNormalizer(
            target_min=float(
                normalization_config.get("target_min", -1.0)
            ),
            target_max=float(
                normalization_config.get("target_max", 1.0)
            ),
            epsilon=float(
                normalization_config.get("epsilon", 1.0e-12)
            ),
            clip=bool(
                normalization_config.get("clip", False)
            ),
        )

        # 归一化参数只在训练集拟合。
        normalizer.fit(
            spectra_on_model_axis[training_indices]
        )
        normalized_full_spectra = normalizer.transform(
            spectra_on_model_axis
        )

        if bool(
            normalization_config.get(
                "save_in_checkpoint",
                True,
            )
        ):
            normalization_state = normalizer.state_dict()
    else:
        normalized_full_spectra = spectra_on_model_axis.copy()

    # ------------------------------------------------------------------
    # D2.1 pointwise MAD-asinh 先验残差变换
    # ------------------------------------------------------------------

    prior_config = configuration.get("prior_residual", {}) or {}
    prior_residual_transformer: PriorResidualTransformer | None = None
    prior_residual_state = None
    spectra_for_model = normalized_full_spectra.copy()
    training_scaled_residuals = spectra_for_model[training_indices]

    if bool(prior_config.get("enabled", False)):
        if normalizer is None or normalization_state is None:
            raise ValueError(
                "启用prior_residual时必须启用并保存训练集归一化状态。"
            )

        prior_residual_transformer = PriorResidualTransformer(
            normalization_method=str(
                prior_config.get(
                    "residual_normalization",
                    "robust_asinh",
                )
            ),
            target_abs_max=float(
                prior_config.get("target_abs_max", 1.0)
            ),
            residual_quantile=float(
                prior_config.get("residual_quantile", 99.5)
            ),
            pointwise_scale_floor_quantile=float(
                prior_config.get(
                    "pointwise_scale_floor_quantile",
                    10.0,
                )
            ),
            mad_scale_factor=float(
                prior_config.get("mad_scale_factor", 1.4826)
            ),
            epsilon=float(
                prior_config.get("epsilon", 1.0e-8)
            ),
        )

        prior_residual_transformer.fit(
            normalized_full_spectra[training_indices]
        )
        training_scaled_residuals = (
            prior_residual_transformer.transform(
                normalized_full_spectra[training_indices]
            )
        )
        spectra_for_model = prior_residual_transformer.transform(
            normalized_full_spectra
        )
        prior_residual_state = (
            prior_residual_transformer.state_dict()
        )

        print_prior_residual_summary(
            prior_residual_transformer,
            training_scaled_residuals,
        )

    # ------------------------------------------------------------------
    # D3.1：只用训练集拟合自适应峰检测阈值和异常范围
    # ------------------------------------------------------------------

    physics_constraint_state = None

    if bool(physics_config.get("enabled", False)):
        if prior_residual_state is None:
            raise ValueError("D3.1物理约束要求先完成D2.1拟合。")

        physics_constraint_state = (
            fit_sers_physics_constraint_state(
                training_normalized_spectra=(
                    normalized_full_spectra[training_indices]
                ),
                training_scaled_residuals=training_scaled_residuals,
                raman_shift=np.asarray(
                    length_adapter.model_raman_shift,
                    dtype=np.float64,
                ),
                configuration=physics_config,
            )
        )
        print_physics_summary(physics_constraint_state)

    # 完成 D2/D3 状态拟合后再补齐网络输入长度。
    padded_spectra = length_adapter.adapt(spectra_for_model)

    if not np.isfinite(padded_spectra).all():
        raise RuntimeError("模型输入包含NaN或无穷值。")

    spectrum_dataset = SpectrumDataset(padded_spectra)
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
    # DataLoader 和 step 频率换算
    # ------------------------------------------------------------------

    device = resolve_device(str(training_config["device"]))
    number_of_workers = int(
        training_config.get("number_of_workers", 0)
    )

    if number_of_workers < 0:
        raise ValueError("training.number_of_workers不能小于0。")

    pin_memory = (
        bool(training_config.get("pin_memory", False))
        and device.type == "cuda"
    )
    batch_size = int(training_config["batch_size"])

    if batch_size <= 0:
        raise ValueError("training.batch_size必须大于0。")

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=bool(
            training_config.get("drop_last", False)
        ),
        worker_init_fn=seed_data_loader_worker,
        generator=create_data_loader_generator(random_seed),
        persistent_workers=number_of_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_data_loader_worker,
        persistent_workers=number_of_workers > 0,
    )

    steps_per_epoch = len(training_loader)

    if steps_per_epoch <= 0:
        raise RuntimeError(
            "训练集没有产生任何batch；"
            "请检查batch_size和drop_last。"
        )

    number_of_epochs = int(training_config["number_of_epochs"])
    validate_every_epochs = int(
        training_config["validate_every_epochs"]
    )
    save_every_epochs = int(
        training_config["save_every_epochs"]
    )
    log_every_batches = int(
        training_config["log_every_batches"]
    )

    if min(
        number_of_epochs,
        validate_every_epochs,
        save_every_epochs,
        log_every_batches,
    ) <= 0:
        raise ValueError(
            "训练epoch、验证频率、保存频率和日志频率必须大于0。"
        )

    training_config["total_training_steps"] = (
        number_of_epochs * steps_per_epoch
    )
    training_config["validate_every_steps"] = (
        validate_every_epochs * steps_per_epoch
    )
    training_config["checkpoint_every_steps"] = (
        save_every_epochs * steps_per_epoch
    )
    training_config["log_every_steps"] = log_every_batches

    # ------------------------------------------------------------------
    # 模型构建和 D3.1 状态注入
    # ------------------------------------------------------------------

    _, diffusion = build_diffusion_model(
        model_configuration=configuration,
        sequence_length=length_adapter.padded_length,
    )

    if physics_constraint_state is not None:
        configure_physics = getattr(
            diffusion,
            "configure_physics_constraints",
            None,
        )

        if not callable(configure_physics):
            raise RuntimeError(
                "D3.1扩散模型缺少configure_physics_constraints。"
            )

        configure_physics(
            physics_constraint_state=physics_constraint_state,
            prior_residual_state=prior_residual_state,
        )

    checkpoint_manager = CheckpointManager(
        resolve_project_path(
            configuration,
            output_config["checkpoint_directory"],
        )
    )
    logger = TrainingLogger(
        resolve_training_log_file(configuration)
    )

    axis_metadata = build_axis_metadata(
        labels=collection.labels,
        relative_source_files=collection.relative_source_files,
        raman_shifts=collection.raman_shifts,
    )

    metadata = {
        "diagnostic_overfit": overfit_metadata,
        "backend_package": "denoising-diffusion-pytorch",
        "backend_version": get_backend_version(),
        "spectrum_names": [
            str(value) for value in collection.spectrum_names
        ],
        "source_files": [
            str(value) for value in collection.source_files
        ],
        "relative_source_files": [
            str(value)
            for value in collection.relative_source_files
        ],
        "labels": [str(value) for value in collection.labels],
        "number_of_spectra": int(number_of_spectra),
        "training_indices": training_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "split_unit": str(data_config["split_unit"]),
        "normalization_state": normalization_state,
        "prior_residual_state": prior_residual_state,
        "physics_constraint_state": physics_constraint_state,
        "axis_metadata": axis_metadata,
        **length_adapter.to_metadata(),
    }

    trainer = DdpmTrainer(
        diffusion=diffusion,
        training_loader=training_loader,
        validation_loader=validation_loader,
        device=device,
        configuration=configuration,
        metadata=metadata,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
    )

    if arguments.resume is not None:
        resume_path = resolve_resume_path(
            configuration=configuration,
            resume_argument=arguments.resume,
        )
        validate_resume_stage(
            checkpoint_path=resume_path,
            configuration=configuration,
        )
        trainer.resume(resume_path)

    # ------------------------------------------------------------------
    # 训练前终端摘要
    # ------------------------------------------------------------------

    labels = sorted({str(value) for value in collection.labels})
    original_lengths = sorted(
        {
            int(np.asarray(axis).size)
            for axis in collection.raman_shifts
        }
    )

    print("\n===== 开始训练 =====")
    print(f"设备：{device}")
    print(f"实验名称：{project_config['name']}")
    print(f"文件夹标签：{'、'.join(labels)}")
    print(f"总光谱数量：{number_of_spectra}")
    print(
        "训练/验证/测试光谱数量："
        f"{len(training_dataset)}/"
        f"{len(validation_dataset)}/"
        f"{len(test_dataset)}"
    )
    print(f"各原始拉曼轴点数：{original_lengths}")
    print(f"统一训练轴长度：{length_adapter.original_length}")
    print(f"模型输入长度：{length_adapter.padded_length}")
    print(f"末尾补齐点数：{length_adapter.padding_size}")
    print(f"每个epoch的step：{steps_per_epoch}")
    print(f"总训练step：{training_config['total_training_steps']}")
    print("额外平滑或去基线：不执行")
    print(
        "D3峰位置来源：每条真实目标光谱动态检测，"
        "不使用固定峰窗口"
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
        print("实验模式：单光谱重复过拟合诊断")
        print(
            "注意：该模式的验证集和测试集也是同一光谱副本，"
            "不能评价泛化性能。"
        )

    print("====================\n")

    trainer.train()


if __name__ == "__main__":
    main()
"""Train the unconditional one-dimensional SERS DDPM.
    
    开始训练指令
    python -m scripts.start_ddpm_training \
    --config config/ddpm_training.yaml

    生成指令
    CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.generate_spectra \
    --config config/ddpm_training.yaml \
    --checkpoint outputs/checkpoints/latest.pt \
    --number 50

    监控gpu指令
    watch -n 1 nvidia-smi
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
)
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.dataset_splitter import (
    split_spectrum_collection,
)
from src.ddpm_trainer import DdpmTrainer
from src.intensity_normalizer import (
    GlobalMinMaxNormalizer,
)
from src.model_builder import (
    build_diffusion_model,
)
from src.one_dimensional_ddpm import (
    get_backend_version,
)
from src.random_seed_manager import (
    create_data_loader_generator,
    seed_data_loader_worker,
    set_random_seed,
)
from src.spectrum_dataset import SpectrumDataset
from src.spectrum_file_reader import (
    read_spectrum_collection,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)
from src.training_logger import TrainingLogger


def parse_arguments() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description="训练无条件一维SERS DDPM。"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML配置文件路径。",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="可选：继续训练使用的检查点路径。",
    )

    return parser.parse_args()


def resolve_device(
    device_text: str,
) -> torch.device:
    """读取并检查训练设备。"""

    device_text = str(
        device_text
    ).strip().lower()

    if device_text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "配置要求使用CUDA，"
                "但PyTorch未检测到GPU。"
            )

    return torch.device(
        device_text
    )


def validate_split_indices(
    *,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    number_of_spectra: int,
) -> None:
    """检查三个数据子集是否无重复地覆盖全部光谱。"""

    named_indices = {
        "训练集": training_indices,
        "验证集": validation_indices,
        "测试集": test_indices,
    }

    for subset_name, indices in named_indices.items():
        if indices.size == 0:
            raise RuntimeError(
                f"{subset_name}为空。"
            )

        if (
            np.any(indices < 0)
            or np.any(
                indices >= number_of_spectra
            )
        ):
            raise RuntimeError(
                f"{subset_name}包含越界索引。"
            )

        if (
            np.unique(indices).size
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
        all_indices.size != number_of_spectra
        or np.unique(all_indices).size
        != number_of_spectra
    ):
        raise RuntimeError(
            "训练集、验证集和测试集"
            "没有无重复地覆盖全部光谱。"
        )


def resolve_training_log_file(
    configuration: dict,
) -> Path:
    """兼容两种训练日志路径写法。"""

    output_config = configuration["output"]

    if "training_log_file" in output_config:
        log_value = output_config[
            "training_log_file"
        ]
    else:
        log_value = (
            Path(
                output_config["log_directory"]
            )
            / str(
                output_config.get(
                    "training_log_name",
                    "training_log.csv",
                )
            )
        )

    return resolve_project_path(
        configuration,
        log_value,
    )


def main() -> None:
    """执行完整训练流程。"""

    arguments = parse_arguments()

    configuration = load_configuration(
        arguments.config
    )

    project_config = configuration["project"]

    random_config = configuration.get(
        "random",
        {},
    )

    data_config = configuration["data"]
    model_config = configuration["model"]

    normalization_config = configuration[
        "normalization"
    ]

    training_config = configuration["training"]
    output_config = configuration["output"]

    random_seed = int(
        random_config.get(
            "seed",
            project_config.get(
                "random_seed",
                42,
            ),
        )
    )

    set_random_seed(
        random_seed=random_seed,
        deterministic=bool(
            random_config.get(
                "deterministic",
                False,
            )
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

    number_of_spectra = len(
        collection.spectrum_names
    )

    if number_of_spectra == 0:
        raise RuntimeError(
            "没有读取到任何光谱。"
        )

    # 必须先完成数据集划分。
    dataset_split = split_spectrum_collection(
        collection=collection,
        data_config=data_config,
        random_seed=random_seed,
    )

    # 直接使用划分器返回的原始索引。
    # 不再通过source_files反向寻找索引。
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

    # 根据各文件的原始位移轴建立统一训练轴。
    length_adapter = SpectrumLengthAdapter.create(
        raman_shifts=collection.raman_shifts,
        dimension_multipliers=model_config[
            "dimension_multipliers"
        ],
        padding_mode=str(
            data_config.get(
                "padding_mode",
                "right_zero_padding",
            )
        ),
    )

    # 把不同长度、不同采样点的光谱
    # 插值到统一训练轴。
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

    if spectra_on_model_axis.ndim != 2:
        raise RuntimeError(
            "插值后的光谱必须为二维数组[N, L]，"
            f"实际为{spectra_on_model_axis.shape}。"
        )

    if (
        spectra_on_model_axis.shape[0]
        != number_of_spectra
    ):
        raise RuntimeError(
            "插值前后的光谱数量不一致。"
        )

    if not np.isfinite(
        spectra_on_model_axis
    ).all():
        raise RuntimeError(
            "插值后的光谱包含NaN或无穷值。"
        )

    configured_model_length = data_config.get(
        "model_spectrum_length",
        "auto",
    )

    model_length_is_auto = (
        configured_model_length is None
        or (
            isinstance(
                configured_model_length,
                str,
            )
            and configured_model_length
            .strip()
            .lower()
            == "auto"
        )
    )

    if (
        not model_length_is_auto
        and int(configured_model_length)
        != length_adapter.padded_length
    ):
        raise ValueError(
            "配置中的model_spectrum_length="
            f"{configured_model_length}，"
            "但程序根据统一位移轴和U-Net"
            "下采样倍数自动得到"
            f"{length_adapter.padded_length}。"
        )

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
        if (
            normalization_config.get("method")
            != "global_minmax"
        ):
            raise ValueError(
                "当前只支持global_minmax归一化。"
            )

        if (
            normalization_config.get("fit_on")
            != "train_only"
        ):
            raise ValueError(
                "归一化器必须只在训练集上拟合，"
                "请设置fit_on: train_only。"
            )

        normalizer = GlobalMinMaxNormalizer(
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

        # 只使用训练集拟合全局最小值和最大值。
        normalizer.fit(
            spectra_on_model_axis[
                training_indices
            ]
        )

        # 三个数据子集使用同一组训练集参数。
        spectra_for_model = (
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
        spectra_for_model = (
            spectra_on_model_axis.copy()
        )

    # 插值和归一化完成后，
    # 再在光谱末尾补齐至U-Net要求的长度。
    padded_spectra = length_adapter.adapt(
        spectra_for_model
    )

    if not np.isfinite(
        padded_spectra
    ).all():
        raise RuntimeError(
            "模型输入包含NaN或无穷值。"
        )

    spectrum_dataset = SpectrumDataset(
        padded_spectra
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

    device = resolve_device(
        str(training_config["device"])
    )

    number_of_workers = int(
        training_config.get(
            "number_of_workers",
            0,
        )
    )

    if number_of_workers < 0:
        raise ValueError(
            "number_of_workers不能小于0。"
        )

    pin_memory = bool(
        training_config.get(
            "pin_memory",
            False,
        )
    ) and device.type == "cuda"

    batch_size = int(
        training_config["batch_size"]
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size必须大于0。"
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
        worker_init_fn=seed_data_loader_worker,
        generator=create_data_loader_generator(
            random_seed
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
        worker_init_fn=seed_data_loader_worker,
        persistent_workers=(
            number_of_workers > 0
        ),
    )

    # 将epoch转换为训练器内部使用的step。
    steps_per_epoch = len(
        training_loader
    )

    if steps_per_epoch <= 0:
        raise RuntimeError(
            "训练集没有产生任何batch；"
            "请检查batch_size和drop_last。"
        )

    number_of_epochs = int(
        training_config["number_of_epochs"]
    )

    validate_every_epochs = int(
        training_config["validate_every_epochs"]
    )

    save_every_epochs = int(
        training_config["save_every_epochs"]
    )

    log_every_batches = int(
        training_config["log_every_batches"]
    )

    if number_of_epochs <= 0:
        raise ValueError(
            "number_of_epochs必须大于0。"
        )

    if validate_every_epochs <= 0:
        raise ValueError(
            "validate_every_epochs必须大于0。"
        )

    if save_every_epochs <= 0:
        raise ValueError(
            "save_every_epochs必须大于0。"
        )

    if log_every_batches <= 0:
        raise ValueError(
            "log_every_batches必须大于0。"
        )

    training_config["total_training_steps"] = (
        number_of_epochs
        * steps_per_epoch
    )

    training_config["validate_every_steps"] = (
        validate_every_epochs
        * steps_per_epoch
    )

    training_config[
        "checkpoint_every_steps"
    ] = (
        save_every_epochs
        * steps_per_epoch
    )

    training_config["log_every_steps"] = (
        log_every_batches
    )

    # 必须传入完整配置。
    # 这样model和diffusion两个配置区段都会生效。
    _, diffusion = build_diffusion_model(
        model_configuration=configuration,
        sequence_length=(
            length_adapter.padded_length
        ),
    )

    checkpoint_directory = resolve_project_path(
        configuration,
        output_config["checkpoint_directory"],
    )

    training_log_file = resolve_training_log_file(
        configuration
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_directory
    )

    logger = TrainingLogger(
        training_log_file
    )

    # 保存标签、源文件和各自原始位移轴之间的映射。
    axis_metadata = build_axis_metadata(
        labels=collection.labels,
        relative_source_files=(
            collection.relative_source_files
        ),
        raman_shifts=collection.raman_shifts,
    )

    metadata = {
        "backend_package": (
            "denoising-diffusion-pytorch"
        ),
        "backend_version": get_backend_version(),
        "spectrum_names": [
            str(value)
            for value in collection.spectrum_names
        ],
        "source_files": [
            str(value)
            for value in collection.source_files
        ],
        "relative_source_files": [
            str(value)
            for value in (
                collection.relative_source_files
            )
        ],
        "labels": [
            str(value)
            for value in collection.labels
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
            data_config["split_unit"]
        ),
        "normalization_state": (
            normalization_state
        ),
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
        resume_path = Path(
            arguments.resume
        ).expanduser()

        if not resume_path.is_absolute():
            resume_path = resolve_project_path(
                configuration,
                resume_path,
            )

        trainer.resume(
            resume_path
        )

    original_lengths = sorted(
        {
            int(
                np.asarray(axis).size
            )
            for axis in collection.raman_shifts
        }
    )

    labels = sorted(
        {
            str(value)
            for value in collection.labels
        }
    )

    print("\n===== 开始训练 =====")
    print(f"设备：{device}")
    print(
        f"文件夹标签："
        f"{'、'.join(labels)}"
    )
    print(
        f"总光谱数量："
        f"{number_of_spectra}"
    )
    print(
        f"训练光谱数量："
        f"{len(training_dataset)}"
    )
    print(
        f"验证光谱数量："
        f"{len(validation_dataset)}"
    )
    print(
        f"测试光谱数量："
        f"{len(test_dataset)}"
    )
    print(
        "各原始位移轴点数："
        + "、".join(
            str(value)
            for value in original_lengths
        )
    )
    print(
        f"统一训练轴长度："
        f"{length_adapter.original_length}"
    )
    print(
        f"模型输入长度："
        f"{length_adapter.padded_length}"
    )
    print(
        f"末尾补齐点数："
        f"{length_adapter.padding_size}"
    )
    print(
        f"总训练step："
        f"{training_config['total_training_steps']}"
    )
    print(
        "平滑、去基线等额外预处理：不执行"
    )

    if normalizer is not None:
        print(
            "强度归一化：训练集global_minmax，"
            f"映射到["
            f"{normalizer.target_min:.6g}, "
            f"{normalizer.target_max:.6g}]"
        )

        print(
            "训练集原始强度范围："
            f"{float(normalizer.data_min):.6g}–"
            f"{float(normalizer.data_max):.6g}"
        )

        print(
            "模型输入范围："
            f"{float(padded_spectra.min()):.6g}–"
            f"{float(padded_spectra.max()):.6g}"
        )
    else:
        print(
            "强度归一化：未启用"
        )

    print("====================\n")

    trainer.train()


if __name__ == "__main__":
    main()
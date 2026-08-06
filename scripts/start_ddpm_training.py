"""Train the unconditional one-dimensional SERS DDPM.
    
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

    用git更新代码


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
from src.prior_residual import (
    PriorResidualTransformer,
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
    SpectrumCollection,
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

    if (
        device_text.startswith("cuda")
        and not torch.cuda.is_available()
    ):
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
    """检查三个子集是否非空、无重复并覆盖全部光谱。"""

    named_indices = {
        "训练集": training_indices,
        "验证集": validation_indices,
        "测试集": test_indices,
    }

    for subset_name, indices in (
        named_indices.items()
    ):
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
    """兼容训练日志的两种路径写法。"""

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


def build_single_spectrum_overfit_collection(
    *,
    collection: SpectrumCollection,
    diagnostic_config: dict,
) -> tuple[
    SpectrumCollection,
    dict[str, object] | None,
]:
    """
    选择一条真实光谱并在内存中复制。

    该模式只用于检查数据流程、模型和采样能否
    记住一条固定光谱，不能评价模型泛化能力。
    """

    enabled = bool(
        diagnostic_config.get(
            "enabled",
            False,
        )
    )

    if not enabled:
        return collection, None

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
            "原始数据中没有可用于诊断的光谱。"
        )

    if (
        spectrum_index < 0
        or spectrum_index >= original_count
    ):
        raise IndexError(
            "diagnostic_overfit.spectrum_index"
            "越界："
            f"当前共有{original_count}条光谱，"
            f"有效索引为0到"
            f"{original_count - 1}，"
            f"但配置值为{spectrum_index}。"
        )

    if repeat_count < 10:
        raise ValueError(
            "diagnostic_overfit.repeat_count"
            "至少为10，以保证按照"
            "0.8/0.1/0.1划分后"
            "三个子集都不为空。"
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
            "所选光谱的强度点数与"
            "拉曼位移点数不一致。"
        )

    if selected_axis.size < 2:
        raise RuntimeError(
            "所选光谱至少需要两个"
            "拉曼位移点。"
        )

    if not np.isfinite(
        selected_spectrum
    ).all():
        raise RuntimeError(
            "所选光谱包含NaN或无穷值。"
        )

    if not np.isfinite(
        selected_axis
    ).all():
        raise RuntimeError(
            "所选拉曼位移轴包含"
            "NaN或无穷值。"
        )

    if not np.all(
        np.diff(selected_axis) > 0.0
    ):
        raise RuntimeError(
            "所选拉曼位移轴不是严格递增。"
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

    selected_label = collection.labels[
        spectrum_index
    ]

    repeated_spectra = np.repeat(
        selected_spectrum[
            np.newaxis,
            :,
        ],
        repeat_count,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    repeated_axes = np.repeat(
        selected_axis[
            np.newaxis,
            :,
        ],
        repeat_count,
        axis=0,
    ).astype(
        np.float64,
        copy=False,
    )

    repeated_collection = SpectrumCollection(
        raman_shift=selected_axis.copy(),
        spectra=repeated_spectra,
        raman_shifts=repeated_axes,
        original_lengths=np.full(
            repeat_count,
            selected_axis.size,
            dtype=np.int64,
        ),
        source_files=np.asarray(
            [
                selected_source_file
                for _ in range(
                    repeat_count
                )
            ],
            dtype=object,
        ),
        relative_source_files=np.asarray(
            [
                selected_relative_source_file
                for _ in range(
                    repeat_count
                )
            ],
            dtype=object,
        ),
        spectrum_names=np.asarray(
            [
                (
                    f"{selected_name}"
                    f"__repeat_{index + 1:04d}"
                )
                for index in range(
                    repeat_count
                )
            ],
            dtype=object,
        ),
        labels=np.asarray(
            [
                selected_label
                for _ in range(
                    repeat_count
                )
            ],
            dtype=object,
        ),
    )

    diagnostic_metadata: dict[
        str,
        object,
    ] = {
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
        diagnostic_metadata,
    )


def main() -> None:
    """执行完整训练流程。"""

    arguments = parse_arguments()

    configuration = load_configuration(
        arguments.config
    )

    project_config = configuration[
        "project"
    ]

    random_config = configuration.get(
        "random",
        {},
    )

    data_config = configuration["data"]
    model_config = configuration["model"]

    normalization_config = configuration[
        "normalization"
    ]

    training_config = configuration[
        "training"
    ]

    output_config = configuration[
        "output"
    ]

    diagnostic_config = configuration.get(
        "diagnostic_overfit",
        {},
    )

    if not isinstance(
        diagnostic_config,
        dict,
    ):
        raise TypeError(
            "diagnostic_overfit必须是字典。"
        )

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

    (
        collection,
        overfit_metadata,
    ) = (
        build_single_spectrum_overfit_collection(
            collection=collection,
            diagnostic_config=(
                diagnostic_config
            ),
        )
    )

    if overfit_metadata is not None:
        if arguments.resume is not None:
            raise ValueError(
                "单光谱过拟合诊断必须从头训练，"
                "不能使用--resume加载旧检查点。"
            )

        # 只修改内存中的配置，不修改YAML文件。
        data_config = dict(
            data_config
        )

        data_config[
            "split_unit"
        ] = "spectrum"

        data_config[
            "shuffle"
        ] = False

        configuration[
            "data"
        ] = data_config

    number_of_spectra = len(
        collection.spectrum_names
    )

    if number_of_spectra == 0:
        raise RuntimeError(
            "没有读取到任何光谱。"
        )

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
        validation_indices=(
            validation_indices
        ),
        test_indices=test_indices,
        number_of_spectra=(
            number_of_spectra
        ),
    )

    # 根据拉曼轴建立统一训练轴。
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
            padding_mode=str(
                data_config.get(
                    "padding_mode",
                    "right_zero_padding",
                )
            ),
        )
    )

    # 按拉曼位移插值，而不是数组下标。
    spectra_on_model_axis = (
        length_adapter
        .interpolate_to_model_axis(
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
            "插值后的光谱必须为二维数组"
            "[N, L]，"
            f"实际为"
            f"{spectra_on_model_axis.shape}。"
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

    configured_model_length = (
        data_config.get(
            "model_spectrum_length",
            "auto",
        )
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
            normalization_config.get(
                "method"
            )
            != "global_minmax"
        ):
            raise ValueError(
                "当前只支持"
                "global_minmax归一化。"
            )

        if (
            normalization_config.get(
                "fit_on"
            )
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

        # 只使用训练集拟合归一化参数。
        normalizer.fit(
            spectra_on_model_axis[
                training_indices
            ]
        )

        # 所有子集使用同一组训练集参数。
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

        # D2：训练集统计先验残差域变换。
    # 先只使用训练集拟合逐点中位数先验和
    # 残差范围参数，再用同一状态变换全部子集。
    prior_residual_config = (
        configuration.get(
            "prior_residual",
            {},
        )
    )

    if prior_residual_config is None:
        prior_residual_config = {}

    if not isinstance(
        prior_residual_config,
        dict,
    ):
        raise TypeError(
            "prior_residual配置必须是字典。"
        )

    prior_residual_state = None

    if bool(
        prior_residual_config.get(
            "enabled",
            False,
        )
    ):
        if normalizer is None:
            raise ValueError(
                "启用prior_residual时，"
                "必须同时启用global_minmax归一化。"
            )

        if normalization_state is None:
            raise ValueError(
                "启用prior_residual时，"
                "normalization.save_in_checkpoint"
                "必须设为true。"
            )

        prior_method = str(
            prior_residual_config.get(
                "prior_method",
                "training_pointwise_median",
            )
        ).strip().lower()

        if (
            prior_method
            != "training_pointwise_median"
        ):
            raise ValueError(
                "当前D2只支持"
                "prior_method: "
                "training_pointwise_median。"
            )

        residual_normalization = str(
            prior_residual_config.get(
                "residual_normalization",
                "robust_asinh",
            )
        ).strip().lower()

        supported_methods = {
            "global_maxabs",
            "robust_asinh",
            "pointwise_mad_asinh",
        }

        if (
            residual_normalization
            not in supported_methods
        ):
            raise ValueError(
                "prior_residual."
                "residual_normalization必须是"
                "global_maxabs、robust_asinh或"
                "pointwise_mad_asinh。"
            )

        prior_residual_transformer = (
            PriorResidualTransformer(
                normalization_method=(
                    residual_normalization
                ),
                target_abs_max=float(
                    prior_residual_config.get(
                        "target_abs_max",
                        1.0,
                    )
                ),
                residual_quantile=float(
                    prior_residual_config.get(
                        "residual_quantile",
                        99.5,
                    )
                ),
                pointwise_scale_floor_quantile=float(
                    prior_residual_config.get(
                        "pointwise_scale_floor_quantile",
                        10.0,
                    )
                ),
                mad_scale_factor=float(
                    prior_residual_config.get(
                        "mad_scale_factor",
                        1.4826,
                    )
                ),
                epsilon=float(
                    prior_residual_config.get(
                        "epsilon",
                        1.0e-8,
                    )
                ),
            )
        )

        # 只用训练集拟合先验和残差范围。
        prior_residual_transformer.fit(
            spectra_for_model[
                training_indices
            ]
        )

        # 在覆盖spectra_for_model之前，
        # 单独统计训练集变换后的残差范围。
        training_model_residuals = (
            prior_residual_transformer
            .transform(
                spectra_for_model[
                    training_indices
                ]
            )
        )

        training_absolute_residuals = (
            np.abs(
                training_model_residuals
            )
        )

        statistics = (
            prior_residual_transformer
            .training_abs_residual_percentiles
            or {}
        )

        print(
            "\n===== D2先验残差范围 ====="
        )

        print(
            "残差归一化方法："
            f"{residual_normalization}"
        )

        print(
            "训练残差绝对值分位数："
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
                    f"  {key}: "
                    f"{statistics[key]:.8g}"
                )

        if (
            residual_normalization
            == "robust_asinh"
        ):
            print(
                "稳健尺度分位数："
                f"{prior_residual_transformer.residual_quantile:g}%"
            )

            print(
                "稳健残差尺度："
                f"{prior_residual_transformer.residual_scale:.8g}"
            )

            print(
                "asinh归一化因子："
                f"{prior_residual_transformer.asinh_normalizer:.8g}"
            )

        elif (
            residual_normalization
            == "pointwise_mad_asinh"
        ):
            print(
                "逐波数MAD换算系数："
                f"{prior_residual_transformer.mad_scale_factor:.8g}"
            )

            print(
                "逐波数尺度下限分位数："
                f"{prior_residual_transformer.pointwise_scale_floor_quantile:g}%"
            )

            print(
                "逐波数尺度下限："
                f"{prior_residual_transformer.pointwise_scale_floor:.8g}"
            )

            print(
                "逐波数尺度分位数："
            )

            pointwise_statistics = (
                prior_residual_transformer
                .training_pointwise_scale_percentiles
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
                        f"  {key}: "
                        f"{pointwise_statistics[key]:.8g}"
                    )

            print(
                "标准化残差稳健分位数："
                f"{prior_residual_transformer.residual_quantile:g}%"
            )

            print(
                "标准化残差尺度："
                f"{prior_residual_transformer.residual_scale:.8g}"
            )

            print(
                "标准化残差最大绝对值："
                f"{prior_residual_transformer.training_max_abs_standardized_residual:.8g}"
            )

            print(
                "asinh归一化因子："
                f"{prior_residual_transformer.asinh_normalizer:.8g}"
            )

        else:
            print(
                "线性残差尺度："
                f"{prior_residual_transformer.residual_scale:.8g}"
            )

        transformed_percentiles = (
            np.percentile(
                training_absolute_residuals,
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
        )

        print(
            "变换后训练残差绝对值分位数："
        )

        for name, value in zip(
            (
                "p50",
                "p90",
                "p95",
                "p99",
                "p99.5",
                "p99.9",
                "max",
            ),
            transformed_percentiles,
            strict=True,
        ):
            print(
                f"  {name}: "
                f"{float(value):.8g}"
            )

        print(
            "==========================\n"
        )

        # 使用训练集拟合出的同一个变换器
        # 处理训练集、验证集和测试集。
        spectra_for_model = (
            prior_residual_transformer
            .transform(
                spectra_for_model
            )
        )

        prior_residual_state = (
            prior_residual_transformer
            .state_dict()
        )

    # 完整光谱模式或D2残差模式完成后，
    # 再补齐到U-Net要求的长度。
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
        str(
            training_config["device"]
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
            "number_of_workers不能小于0。"
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

    _, diffusion = build_diffusion_model(
        model_configuration=configuration,
        sequence_length=(
            length_adapter.padded_length
        ),
    )

    checkpoint_directory = (
        resolve_project_path(
            configuration,
            output_config[
                "checkpoint_directory"
            ],
        )
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_directory
    )

    logger = TrainingLogger(
        resolve_training_log_file(
            configuration
        )
    )

    axis_metadata = build_axis_metadata(
        labels=collection.labels,
        relative_source_files=(
            collection.relative_source_files
        ),
        raman_shifts=(
            collection.raman_shifts
        ),
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
            for value in (
                collection.spectrum_names
            )
        ],
        "source_files": [
            str(value)
            for value in (
                collection.source_files
            )
        ],
        "relative_source_files": [
            str(value)
            for value in (
                collection
                .relative_source_files
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
        "prior_residual_state": (
            prior_residual_state
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
        checkpoint_manager=(
            checkpoint_manager
        ),
        logger=logger,
    )

    if arguments.resume is not None:
        resume_path = Path(
            arguments.resume
        ).expanduser()

        if not resume_path.is_absolute():
            resume_path = (
                resolve_project_path(
                    configuration,
                    resume_path,
                )
            )

        trainer.resume(
            resume_path
        )

    original_lengths = sorted(
        {
            int(
                np.asarray(axis).size
            )
            for axis in (
                collection.raman_shifts
            )
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

    if overfit_metadata is not None:
        original_spectrum_count = overfit_metadata[
            "original_number_of_spectra"
        ]
        selected_original_index = overfit_metadata[
            "selected_original_index"
        ]
        selected_spectrum_name = overfit_metadata[
            "selected_spectrum_name"
        ]
        selected_source_file = overfit_metadata[
            "selected_relative_source_file"
        ]
        repeat_count = overfit_metadata[
            "repeat_count"
        ]

        print(
            "实验模式："
            "单条真实光谱重复过拟合诊断"
        )
        print(
            "注意：验证集和测试集也是"
            "同一条光谱的副本，"
            "不代表泛化性能。"
        )
        print(
            "原始数据光谱总数："
            f"{original_spectrum_count}"
        )
        print(
            "所选原始光谱索引："
            f"{selected_original_index}"
        )
        print(
            "所选原始光谱名称："
            f"{selected_spectrum_name}"
        )
        print(
            "所选原始源文件："
            f"{selected_source_file}"
        )
        print(
            "内存复制数量："
            f"{repeat_count}"
        )

    label_text = "、".join(labels)

    print(
        f"文件夹标签：{label_text}"
    )
    print(
        f"总光谱数量：{number_of_spectra}"
    )
    print(
        "训练光谱数量："
        f"{len(training_dataset)}"
    )
    print(
        "验证光谱数量："
        f"{len(validation_dataset)}"
    )
    print(
        "测试光谱数量："
        f"{len(test_dataset)}"
    )

    original_length_text = "、".join(
        str(value)
        for value in original_lengths
    )

    print(
        "各原始位移轴点数："
        f"{original_length_text}"
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
        f"每个epoch的step：{steps_per_epoch}"
    )

    total_training_steps = training_config[
        "total_training_steps"
    ]

    print(
        f"总训练step：{total_training_steps}"
    )
    print(
        "平滑、去基线等额外预处理："
        "不执行"
    )

    if normalizer is not None:
        normalization_data_min = float(
            normalizer.data_min
        )
        normalization_data_max = float(
            normalizer.data_max
        )
        model_input_min = float(
            padded_spectra.min()
        )
        model_input_max = float(
            padded_spectra.max()
        )

        print(
            "强度归一化："
            "训练集global_minmax，"
            f"映射到[{normalizer.target_min:.6g}, "
            f"{normalizer.target_max:.6g}]"
        )
        print(
            "训练集原始强度范围："
            f"{normalization_data_min:.6g}–"
            f"{normalization_data_max:.6g}"
        )
        print(
            "模型输入范围："
            f"{model_input_min:.6g}–"
            f"{model_input_max:.6g}"
        )
    else:
        print(
            "强度归一化：未启用"
        )

    print("====================\n")

    trainer.train()


if __name__ == "__main__":
    main()
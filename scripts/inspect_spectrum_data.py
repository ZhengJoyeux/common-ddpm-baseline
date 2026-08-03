"""Inspect preprocessed SERS data before DDPM training."""

from __future__ import annotations

import argparse

import numpy as np

from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.dataset_splitter import (
    split_spectrum_collection,
)
from src.intensity_normalizer import (
    GlobalMinMaxNormalizer,
)
from src.one_dimensional_ddpm import (
    get_backend_version,
)
from src.spectrum_file_reader import (
    read_spectrum_collection,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)


def parse_arguments() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "检查一维无条件DDPM使用的SERS输入数据。"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML配置文件路径。",
    )

    return parser.parse_args()


def is_auto_length(
    value: object,
) -> bool:
    """判断配置中的模型长度是否设置为自动。"""

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() == "auto"

    return False


def validate_raw_collection(
    collection,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    float,
    float,
]:
    """
    检查每条光谱及其对应的原始拉曼位移轴。

    返回：
    1. 转换后的光谱列表；
    2. 转换后的位移轴列表；
    3. 全部光谱的最小强度；
    4. 全部光谱的最大强度。
    """

    number_of_spectra = len(
        collection.spectrum_names
    )

    if number_of_spectra == 0:
        raise RuntimeError(
            "没有读取到任何光谱。"
        )

    fields_to_check = {
        "spectra": collection.spectra,
        "raman_shifts": collection.raman_shifts,
        "spectrum_names": collection.spectrum_names,
        "source_files": collection.source_files,
        "relative_source_files": (
            collection.relative_source_files
        ),
        "labels": collection.labels,
    }

    for field_name, values in fields_to_check.items():
        if len(values) != number_of_spectra:
            raise RuntimeError(
                f"collection.{field_name}的数量为"
                f"{len(values)}，"
                f"但总光谱数量为{number_of_spectra}。"
            )

    validated_spectra: list[np.ndarray] = []
    validated_raman_shifts: list[np.ndarray] = []

    global_intensity_min = np.inf
    global_intensity_max = -np.inf

    for index in range(number_of_spectra):
        spectrum_name = str(
            collection.spectrum_names[index]
        )

        spectrum = np.asarray(
            collection.spectra[index],
            dtype=np.float32,
        ).reshape(-1)

        raman_shift = np.asarray(
            collection.raman_shifts[index],
            dtype=np.float64,
        ).reshape(-1)

        if spectrum.size < 2:
            raise RuntimeError(
                f"光谱{spectrum_name!r}少于2个数据点。"
            )

        if raman_shift.size != spectrum.size:
            raise RuntimeError(
                f"光谱{spectrum_name!r}的强度点数为"
                f"{spectrum.size}，"
                f"但拉曼位移点数为"
                f"{raman_shift.size}。"
            )

        if not np.isfinite(spectrum).all():
            raise RuntimeError(
                f"光谱{spectrum_name!r}中包含"
                "NaN或无穷值。"
            )

        if not np.isfinite(raman_shift).all():
            raise RuntimeError(
                f"光谱{spectrum_name!r}的"
                "拉曼位移轴包含NaN或无穷值。"
            )

        if not np.all(
            np.diff(raman_shift) > 0.0
        ):
            raise RuntimeError(
                f"光谱{spectrum_name!r}的"
                "拉曼位移轴不是严格递增的。"
            )

        global_intensity_min = min(
            global_intensity_min,
            float(spectrum.min()),
        )

        global_intensity_max = max(
            global_intensity_max,
            float(spectrum.max()),
        )

        validated_spectra.append(
            spectrum
        )

        validated_raman_shifts.append(
            raman_shift
        )

    return (
        validated_spectra,
        validated_raman_shifts,
        float(global_intensity_min),
        float(global_intensity_max),
    )


def validate_split_indices(
    *,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    number_of_spectra: int,
) -> None:
    """检查三个数据子集是否正确覆盖全部光谱。"""

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

    if all_indices.size != number_of_spectra:
        raise RuntimeError(
            "训练集、验证集和测试集的光谱总数"
            "与原始数据总数不一致。"
        )

    if (
        np.unique(all_indices).size
        != number_of_spectra
    ):
        raise RuntimeError(
            "训练集、验证集和测试集之间"
            "存在重复光谱，或未覆盖全部光谱。"
        )


def validate_source_file_separation(
    *,
    relative_source_files: list[str],
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    split_unit: str,
) -> None:
    """检查按源文件划分时是否存在数据泄漏。"""

    if str(split_unit) != "source_file":
        return

    training_files = {
        relative_source_files[index]
        for index in training_indices.tolist()
    }

    validation_files = {
        relative_source_files[index]
        for index in validation_indices.tolist()
    }

    test_files = {
        relative_source_files[index]
        for index in test_indices.tolist()
    }

    train_validation_overlap = (
        training_files & validation_files
    )

    train_test_overlap = (
        training_files & test_files
    )

    validation_test_overlap = (
        validation_files & test_files
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        overlapping_files = sorted(
            train_validation_overlap
            | train_test_overlap
            | validation_test_overlap
        )

        raise RuntimeError(
            "检测到同一个源文件被划入多个数据子集，"
            "可能造成数据泄漏："
            + "、".join(overlapping_files)
        )


def collect_unique_axis_profiles(
    *,
    indices: list[int],
    raman_shifts: list[np.ndarray],
) -> list[dict]:
    """统计指定光谱使用的不同原始位移轴。"""

    profiles: list[dict] = []

    for index in indices:
        current_axis = raman_shifts[index]

        matched_profile = None

        for profile in profiles:
            saved_axis = profile["axis"]

            if (
                saved_axis.shape == current_axis.shape
                and np.array_equal(
                    saved_axis,
                    current_axis,
                )
            ):
                matched_profile = profile
                break

        if matched_profile is None:
            profiles.append(
                {
                    "axis": current_axis,
                    "count": 1,
                }
            )
        else:
            matched_profile["count"] += 1

    return profiles


def print_label_summary(
    *,
    labels: list[str],
    relative_source_files: list[str],
    raman_shifts: list[np.ndarray],
) -> None:
    """输出每个标签下的光谱和位移轴信息。"""

    unique_labels = list(
        dict.fromkeys(labels)
    )

    print("\n标签和原始位移轴：")

    for label in unique_labels:
        label_indices = [
            index
            for index, current_label in enumerate(
                labels
            )
            if current_label == label
        ]

        label_files = {
            relative_source_files[index]
            for index in label_indices
        }

        axis_profiles = collect_unique_axis_profiles(
            indices=label_indices,
            raman_shifts=raman_shifts,
        )

        print(
            f"  标签 {label!r}："
            f"{len(label_indices)}条光谱，"
            f"{len(label_files)}个源文件，"
            f"{len(axis_profiles)}种原始位移轴"
        )

        for profile_number, profile in enumerate(
            axis_profiles,
            start=1,
        ):
            axis = profile["axis"]

            print(
                f"    - 位移轴{profile_number}："
                f"{axis.size}点，"
                f"{axis[0]:.6g}–"
                f"{axis[-1]:.6g} cm⁻¹，"
                f"用于{profile['count']}条光谱"
            )


def main() -> None:
    """执行完整的输入数据检查。"""

    arguments = parse_arguments()

    configuration = load_configuration(
        arguments.config
    )

    project_config = configuration["project"]
    data_config = configuration["data"]
    model_config = configuration["model"]

    normalization_config = configuration.get(
        "normalization",
        {},
    )

    random_config = configuration.get(
        "random",
        {},
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

    input_directory = resolve_project_path(
        configuration,
        data_config["input_directory"],
    )

    collection = read_spectrum_collection(
        input_directory=input_directory,
        data_config=data_config,
    )

    (
        original_spectra,
        original_raman_shifts,
        original_intensity_min,
        original_intensity_max,
    ) = validate_raw_collection(
        collection
    )

    number_of_spectra = len(
        original_spectra
    )

    relative_source_files = [
        str(value)
        for value in collection.relative_source_files
    ]

    labels = [
        str(value)
        for value in collection.labels
    ]

    unique_source_files = list(
        dict.fromkeys(
            relative_source_files
        )
    )

    # 必须先按照原始样品文件划分数据集，
    # 然后才能拟合强度归一化参数。
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

    validate_source_file_separation(
        relative_source_files=relative_source_files,
        training_indices=training_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        split_unit=str(
            data_config["split_unit"]
        ),
    )

    # 根据全部原始位移轴建立统一训练轴。
    length_adapter = SpectrumLengthAdapter.create(
        raman_shifts=original_raman_shifts,
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

    # 将不同点数、不同采样轴的光谱
    # 插值到统一训练轴。
    spectra_on_training_axis = (
        length_adapter.interpolate_to_model_axis(
            original_spectra,
            original_raman_shifts,
        )
    )

    spectra_on_training_axis = np.asarray(
        spectra_on_training_axis,
        dtype=np.float32,
    )

    if spectra_on_training_axis.ndim != 2:
        raise RuntimeError(
            "插值后的光谱必须是二维数组[N, L]，"
            f"实际形状为"
            f"{spectra_on_training_axis.shape}。"
        )

    if (
        spectra_on_training_axis.shape[0]
        != number_of_spectra
    ):
        raise RuntimeError(
            "插值前后的光谱数量不一致。"
        )

    if (
        spectra_on_training_axis.shape[1]
        != length_adapter.original_length
    ):
        raise RuntimeError(
            "插值后的光谱长度与统一训练轴长度不一致。"
        )

    if not np.isfinite(
        spectra_on_training_axis
    ).all():
        raise RuntimeError(
            "插值后的光谱中包含NaN或无穷值。"
        )

    configured_model_length = data_config.get(
        "model_spectrum_length",
        "auto",
    )

    if (
        not is_auto_length(
            configured_model_length
        )
        and int(configured_model_length)
        != length_adapter.padded_length
    ):
        raise ValueError(
            "配置文件中的model_spectrum_length为"
            f"{configured_model_length}，"
            "但程序根据统一训练轴和U-Net"
            "下采样倍数自动计算得到"
            f"{length_adapter.padded_length}。"
        )

    normalizer = None

    if bool(
        normalization_config.get(
            "enabled",
            False,
        )
    ):
        normalization_method = str(
            normalization_config.get(
                "method",
                "",
            )
        )

        if normalization_method != "global_minmax":
            raise ValueError(
                "当前程序只支持"
                "normalization.method: global_minmax。"
            )

        if str(
            normalization_config.get(
                "fit_on",
                "",
            )
        ) != "train_only":
            raise ValueError(
                "归一化器必须只使用训练集拟合，"
                "请设置normalization.fit_on: train_only。"
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

        # 只使用训练集强度拟合归一化器。
        normalizer.fit(
            spectra_on_training_axis[
                training_indices
            ]
        )

        spectra_for_model = (
            normalizer.transform(
                spectra_on_training_axis
            )
        )

    else:
        spectra_for_model = (
            spectra_on_training_axis.copy()
        )

    # 插值和归一化后，再在末尾补齐。
    padded_spectra = length_adapter.adapt(
        spectra_for_model
    )

    padded_spectra = np.asarray(
        padded_spectra,
        dtype=np.float32,
    )

    if padded_spectra.ndim != 2:
        raise RuntimeError(
            "补齐后的模型输入必须为二维数组。"
        )

    if (
        padded_spectra.shape[1]
        != length_adapter.padded_length
    ):
        raise RuntimeError(
            "实际模型输入长度与长度适配器记录不一致。"
        )

    if not np.isfinite(
        padded_spectra
    ).all():
        raise RuntimeError(
            "最终模型输入包含NaN或无穷值。"
        )

    original_lengths = sorted(
        {
            int(axis.size)
            for axis in original_raman_shifts
        }
    )

    original_shift_min = min(
        float(axis[0])
        for axis in original_raman_shifts
    )

    original_shift_max = max(
        float(axis[-1])
        for axis in original_raman_shifts
    )

    print("\n===== SERS输入数据检查结果 =====")

    print(
        f"第三方扩散包版本："
        f"{get_backend_version()}"
    )

    print(f"输入目录：{input_directory}")
    print(
        f"输入文件数量："
        f"{len(unique_source_files)}"
    )

    for source_file in unique_source_files:
        print(f"  - {source_file}")

    print(
        f"总光谱数量："
        f"{number_of_spectra}"
    )

    print(
        "各原始光谱点数："
        + "、".join(
            str(value)
            for value in original_lengths
        )
    )

    print(
        "所有原始位移轴覆盖范围："
        f"{original_shift_min:.6g}–"
        f"{original_shift_max:.6g} cm⁻¹"
    )

    print(
        "原始光谱强度范围："
        f"{original_intensity_min:.6g}–"
        f"{original_intensity_max:.6g}"
    )

    print_label_summary(
        labels=labels,
        relative_source_files=relative_source_files,
        raman_shifts=original_raman_shifts,
    )

    print("\n统一训练轴和模型输入：")

    print(
        f"统一训练轴点数："
        f"{length_adapter.original_length}"
    )

    print(
        f"U-Net要求的长度倍数："
        f"{length_adapter.required_multiple}"
    )

    print(
        f"模型输入点数："
        f"{length_adapter.padded_length}"
    )

    print(
        f"末尾补齐点数："
        f"{length_adapter.padding_size}"
    )

    print(
        "插值后、归一化前的强度范围："
        f"{float(spectra_on_training_axis.min()):.6g}–"
        f"{float(spectra_on_training_axis.max()):.6g}"
    )

    if normalizer is not None:
        print(
            "强度归一化："
            "训练集global_minmax"
        )

        print(
            "训练集归一化拟合范围："
            f"{float(normalizer.data_min):.6g}–"
            f"{float(normalizer.data_max):.6g}"
        )

        print(
            "归一化目标范围："
            f"{normalizer.target_min:.6g}–"
            f"{normalizer.target_max:.6g}"
        )

        print(
            "归一化后、补齐前的实际范围："
            f"{float(spectra_for_model.min()):.6g}–"
            f"{float(spectra_for_model.max()):.6g}"
        )
    else:
        print(
            "强度归一化：未启用"
        )

    print(
        "最终模型输入范围："
        f"{float(padded_spectra.min()):.6g}–"
        f"{float(padded_spectra.max()):.6g}"
    )

    print("\n数据集划分：")

    print(
        f"训练集光谱数量："
        f"{training_indices.size}"
    )

    print(
        f"验证集光谱数量："
        f"{validation_indices.size}"
    )

    print(
        f"测试集光谱数量："
        f"{test_indices.size}"
    )

    print(
        f"数据集划分单位："
        f"{data_config['split_unit']}"
    )

    if str(
        data_config["split_unit"]
    ) == "source_file":
        print(
            "源文件交叉检查：未发现数据泄漏"
        )

    print("\n其他检查：")
    print("缺失值/无穷值：未发现")
    print("拉曼位移轴与强度长度：一一对应")
    print("拉曼位移轴顺序：全部严格递增")
    print("程序执行的平滑、去基线和去噪：无")
    print(
        "本脚本仅在内存中模拟插值、归一化和补齐，"
        "不会修改原始文件。"
    )
    print(
        "数据检查通过，可以进行短训练测试。"
    )


if __name__ == "__main__":
    main()
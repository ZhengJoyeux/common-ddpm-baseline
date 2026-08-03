"""Split a spectrum collection into training, validation, and test sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.spectrum_file_reader import (
    SpectrumCollection,
)


@dataclass(frozen=True)
class SpectrumSubset:
    """One subset of the complete spectrum collection."""

    spectra: np.ndarray | list[np.ndarray]
    source_files: np.ndarray
    spectrum_names: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class SpectrumDatasetSplit:
    """Training, validation, and test subsets."""

    train: SpectrumSubset
    validation: SpectrumSubset
    test: SpectrumSubset


# 保留兼容名称，避免其他旧代码导入时报错。
DatasetSplit = SpectrumDatasetSplit
SpectrumSplit = SpectrumDatasetSplit

@dataclass(frozen=True)
class DatasetIndexSplit:
    """
    训练集和验证集的光谱索引。

    这是为旧测试和旧代码保留的兼容数据结构。
    正式训练仍然使用SpectrumDatasetSplit。
    """

    training_indices: np.ndarray
    validation_indices: np.ndarray


def split_dataset_indices(
    *,
    number_of_spectra: int,
    validation_fraction: float,
    random_seed: int,
) -> DatasetIndexSplit:
    """
    将光谱索引划分为训练集和验证集。

    这是旧版接口的兼容函数，仅用于原有测试或旧代码。
    正式训练的数据划分由split_spectrum_collection负责。
    """

    number_of_spectra = int(
        number_of_spectra
    )

    validation_fraction = float(
        validation_fraction
    )

    random_seed = int(
        random_seed
    )

    if number_of_spectra < 2:
        raise ValueError(
            "至少需要2条光谱，才能划分训练集和验证集。"
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction必须大于0且小于1。"
        )

    validation_count = int(
        round(
            number_of_spectra
            * validation_fraction
        )
    )

    # 确保训练集和验证集都不为空。
    validation_count = max(
        1,
        min(
            validation_count,
            number_of_spectra - 1,
        ),
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    shuffled_indices = random_generator.permutation(
        number_of_spectra
    ).astype(
        np.int64,
        copy=False,
    )

    validation_indices = shuffled_indices[
        :validation_count
    ]

    training_indices = shuffled_indices[
        validation_count:
    ]

    return DatasetIndexSplit(
        training_indices=training_indices,
        validation_indices=validation_indices,
    )


def _read_split_ratios(
    data_config: dict,
) -> tuple[float, float, float]:
    """Read and validate train/validation/test ratios."""

    train_ratio = float(
        data_config["train_ratio"]
    )

    if "validation_ratio" in data_config:
        validation_ratio = float(
            data_config["validation_ratio"]
        )
    elif "val_ratio" in data_config:
        validation_ratio = float(
            data_config["val_ratio"]
        )
    else:
        raise KeyError(
            "data配置中缺少validation_ratio。"
        )

    test_ratio = float(
        data_config["test_ratio"]
    )

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if any(
        ratio <= 0.0
        for ratio in ratios
    ):
        raise ValueError(
            "train_ratio、validation_ratio和"
            "test_ratio都必须大于0。"
        )

    ratio_sum = sum(ratios)

    if not np.isclose(
        ratio_sum,
        1.0,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            "train_ratio、validation_ratio和"
            "test_ratio之和必须等于1，"
            f"当前之和为{ratio_sum:.12g}。"
        )

    return ratios


def _calculate_split_counts(
    number_of_items: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """
    Calculate the number of items assigned to each subset.

    前两个子集向下取整，剩余项目放入测试集，
    从而保证三个子集的总数量与原始数量完全一致。
    """

    if number_of_items < 3:
        raise ValueError(
            "至少需要3个可划分单位，才能建立"
            "非空的训练集、验证集和测试集。"
        )

    train_count = int(
        number_of_items * train_ratio
    )

    validation_count = int(
        number_of_items * validation_ratio
    )

    test_count = (
        number_of_items
        - train_count
        - validation_count
    )

    if train_count == 0:
        raise ValueError(
            "训练集划分结果为空。请增加数据量，"
            "或者调整train_ratio。"
        )

    if validation_count == 0:
        raise ValueError(
            "验证集划分结果为空。请增加数据量，"
            "或者调整validation_ratio。"
        )

    if test_count == 0:
        raise ValueError(
            "测试集划分结果为空。请增加数据量，"
            "或者调整test_ratio。"
        )

    return (
        train_count,
        validation_count,
        test_count,
    )


def _select_spectra(
    spectra: object,
    indices: np.ndarray,
) -> np.ndarray | list[np.ndarray]:
    """
    Select spectra by their original indices.

    同时兼容：
    1. 相同长度光谱组成的二维NumPy数组；
    2. 不同长度光谱组成的列表或元组；
    3. object类型的NumPy数组。
    """

    if isinstance(spectra, np.ndarray):
        return spectra[indices]

    return [
        spectra[int(index)]
        for index in indices
    ]


def _make_subset(
    collection: SpectrumCollection,
    indices: np.ndarray,
) -> SpectrumSubset:
    """Create one subset from selected spectrum indices."""

    indices = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(-1)

    if indices.size == 0:
        raise ValueError(
            "不能创建空的数据子集。"
        )

    number_of_spectra = len(
        collection.spectra
    )

    if (
        np.any(indices < 0)
        or np.any(
            indices >= number_of_spectra
        )
    ):
        raise IndexError(
            "创建数据子集时检测到越界索引。"
        )

    source_files = np.asarray(
        collection.source_files,
        dtype=object,
    )

    spectrum_names = np.asarray(
        collection.spectrum_names,
        dtype=object,
    )

    return SpectrumSubset(
        spectra=_select_spectra(
            collection.spectra,
            indices,
        ),
        source_files=source_files[indices],
        spectrum_names=spectrum_names[indices],
        indices=indices.copy(),
    )


def _ordered_unique_values(
    values: np.ndarray,
) -> list[object]:
    """Return unique values while preserving first appearance order."""

    unique_values: list[object] = []
    seen_keys: set[str] = set()

    for value in values.tolist():
        value_key = str(value)

        if value_key in seen_keys:
            continue

        seen_keys.add(
            value_key
        )

        unique_values.append(
            value
        )

    return unique_values


def _indices_for_source_files(
    *,
    all_source_files: np.ndarray,
    selected_source_files: list[object],
) -> np.ndarray:
    """
    Return spectrum indices belonging to selected source files.

    索引顺序按照selected_source_files中的文件顺序排列；
    同一个文件内部保持原始光谱顺序。
    """

    source_file_to_indices: dict[
        str,
        list[int],
    ] = {}

    for spectrum_index, source_file in enumerate(
        all_source_files.tolist()
    ):
        source_key = str(
            source_file
        )

        source_file_to_indices.setdefault(
            source_key,
            [],
        ).append(
            spectrum_index
        )

    selected_indices: list[int] = []

    for source_file in selected_source_files:
        source_key = str(
            source_file
        )

        if source_key not in source_file_to_indices:
            raise RuntimeError(
                f"没有找到源文件{source_key!r}对应的光谱。"
            )

        selected_indices.extend(
            source_file_to_indices[source_key]
        )

    return np.asarray(
        selected_indices,
        dtype=np.int64,
    )


def _split_by_spectrum(
    *,
    collection: SpectrumCollection,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
    shuffle: bool,
) -> SpectrumDatasetSplit:
    """Split individual spectra into three subsets."""

    number_of_spectra = len(
        collection.spectra
    )

    (
        train_count,
        validation_count,
        _,
    ) = _calculate_split_counts(
        number_of_items=number_of_spectra,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    all_indices = np.arange(
        number_of_spectra,
        dtype=np.int64,
    )

    if shuffle:
        random_generator = np.random.default_rng(
            random_seed
        )

        all_indices = random_generator.permutation(
            all_indices
        )

    validation_end = (
        train_count
        + validation_count
    )

    train_indices = all_indices[
        :train_count
    ]

    validation_indices = all_indices[
        train_count:validation_end
    ]

    test_indices = all_indices[
        validation_end:
    ]

    return SpectrumDatasetSplit(
        train=_make_subset(
            collection,
            train_indices,
        ),
        validation=_make_subset(
            collection,
            validation_indices,
        ),
        test=_make_subset(
            collection,
            test_indices,
        ),
    )


def _split_by_source_file(
    *,
    collection: SpectrumCollection,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
    shuffle: bool,
) -> SpectrumDatasetSplit:
    """
    Split complete source files into three subsets.

    同一个Excel、CSV或其他源文件中的全部光谱，
    只会进入训练集、验证集或测试集中的一个子集。
    """

    all_source_files = np.asarray(
        collection.source_files,
        dtype=object,
    )

    unique_source_files = _ordered_unique_values(
        all_source_files
    )

    number_of_source_files = len(
        unique_source_files
    )

    (
        train_file_count,
        validation_file_count,
        _,
    ) = _calculate_split_counts(
        number_of_items=number_of_source_files,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    if shuffle:
        random_generator = np.random.default_rng(
            random_seed
        )

        shuffled_positions = (
            random_generator.permutation(
                number_of_source_files
            )
        )

        unique_source_files = [
            unique_source_files[int(position)]
            for position in shuffled_positions
        ]

    validation_file_end = (
        train_file_count
        + validation_file_count
    )

    train_source_files = unique_source_files[
        :train_file_count
    ]

    validation_source_files = unique_source_files[
        train_file_count:validation_file_end
    ]

    test_source_files = unique_source_files[
        validation_file_end:
    ]

    train_indices = _indices_for_source_files(
        all_source_files=all_source_files,
        selected_source_files=train_source_files,
    )

    validation_indices = _indices_for_source_files(
        all_source_files=all_source_files,
        selected_source_files=(
            validation_source_files
        ),
    )

    test_indices = _indices_for_source_files(
        all_source_files=all_source_files,
        selected_source_files=test_source_files,
    )

    return SpectrumDatasetSplit(
        train=_make_subset(
            collection,
            train_indices,
        ),
        validation=_make_subset(
            collection,
            validation_indices,
        ),
        test=_make_subset(
            collection,
            test_indices,
        ),
    )


def _validate_complete_split(
    *,
    dataset_split: SpectrumDatasetSplit,
    number_of_spectra: int,
) -> None:
    """Check that all spectra appear exactly once."""

    train_indices = dataset_split.train.indices
    validation_indices = (
        dataset_split.validation.indices
    )
    test_indices = dataset_split.test.indices

    all_indices = np.concatenate(
        [
            train_indices,
            validation_indices,
            test_indices,
        ]
    )

    if all_indices.size != number_of_spectra:
        raise RuntimeError(
            "划分后的光谱总数与原始光谱总数不一致。"
        )

    if (
        np.unique(all_indices).size
        != number_of_spectra
    ):
        raise RuntimeError(
            "训练集、验证集和测试集之间存在"
            "重复光谱，或者有光谱未被划分。"
        )

    expected_indices = np.arange(
        number_of_spectra,
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(all_indices),
        expected_indices,
    ):
        raise RuntimeError(
            "数据集划分没有完整覆盖原始光谱索引。"
        )


def split_spectrum_collection(
    *,
    collection: SpectrumCollection,
    data_config: dict,
    random_seed: int,
) -> SpectrumDatasetSplit:
    """
    Split a complete spectrum collection.

    支持两种划分方式：

    source_file:
        按源文件划分，同一个文件中的光谱不会泄漏到
        不同数据子集中。这是正式训练推荐使用的方式。

    spectrum:
        按单条光谱划分，仅用于特殊测试；当同一文件包含
        多条重复测量光谱时，可能造成数据泄漏。
    """

    number_of_spectra = len(
        collection.spectra
    )

    if number_of_spectra == 0:
        raise ValueError(
            "不能划分空的光谱数据集。"
        )

    if (
        len(collection.source_files)
        != number_of_spectra
    ):
        raise ValueError(
            "source_files数量与光谱数量不一致。"
        )

    if (
        len(collection.spectrum_names)
        != number_of_spectra
    ):
        raise ValueError(
            "spectrum_names数量与光谱数量不一致。"
        )

    (
        train_ratio,
        validation_ratio,
        test_ratio,
    ) = _read_split_ratios(
        data_config
    )

    split_unit = str(
        data_config.get(
            "split_unit",
            "source_file",
        )
    ).strip().lower()

    shuffle = bool(
        data_config.get(
            "shuffle",
            True,
        )
    )

    if split_unit == "source_file":
        dataset_split = _split_by_source_file(
            collection=collection,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            random_seed=int(random_seed),
            shuffle=shuffle,
        )

    elif split_unit == "spectrum":
        dataset_split = _split_by_spectrum(
            collection=collection,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            random_seed=int(random_seed),
            shuffle=shuffle,
        )

    else:
        raise ValueError(
            "data.split_unit只支持"
            "'source_file'或'spectrum'，"
            f"当前值为{split_unit!r}。"
        )

    _validate_complete_split(
        dataset_split=dataset_split,
        number_of_spectra=number_of_spectra,
    )

    return dataset_split
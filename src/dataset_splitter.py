"""Split SERS spectra into training, validation, and test subsets.

The project supports three split modes:

``source_file``
    Treat every source CSV or Excel file as one indivisible split unit.
    This is suitable when one source file contains several mapping spectra
    belonging to the same independent sample.

``spectrum``
    Split all spectra globally, without considering source files or folders.
    This mode is retained mainly for diagnostics because it can introduce
    leakage when related spectra come from the same experimental sample.

``spectrum_within_folder``
    Group spectra by their parent-folder label and independently split the
    spectra inside every folder. This mode is intended for the directory form
    in which every sample-combination folder contains many files and every file
    contains exactly one spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from src.spectrum_file_reader import (
    ROOT_LABEL,
    SpectrumCollection,
)


@dataclass(frozen=True)
class SpectrumSubset:
    """One subset selected from the complete spectrum collection."""

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


# Compatibility names retained for older imports.
DatasetSplit = SpectrumDatasetSplit
SpectrumSplit = SpectrumDatasetSplit


@dataclass(frozen=True)
class DatasetIndexSplit:
    """Compatibility structure containing training and validation indices."""

    training_indices: np.ndarray
    validation_indices: np.ndarray


def split_dataset_indices(
    *,
    number_of_spectra: int,
    validation_fraction: float,
    random_seed: int,
) -> DatasetIndexSplit:
    """Split indices into training and validation subsets reproducibly.

    This is an older compatibility interface used by the original tests.
    Formal training should use :func:`split_spectrum_collection`.
    """

    number_of_spectra = int(number_of_spectra)
    validation_fraction = float(validation_fraction)
    random_seed = int(random_seed)

    if number_of_spectra < 2:
        raise ValueError(
            "至少需要2条光谱，才能划分训练集和验证集。"
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction必须大于0且小于1。"
        )

    validation_count = int(
        round(number_of_spectra * validation_fraction)
    )

    # Keep both subsets non-empty.
    validation_count = max(
        1,
        min(validation_count, number_of_spectra - 1),
    )

    random_generator = np.random.default_rng(random_seed)
    shuffled_indices = random_generator.permutation(
        number_of_spectra
    ).astype(
        np.int64,
        copy=False,
    )

    validation_indices = shuffled_indices[:validation_count]
    training_indices = shuffled_indices[validation_count:]

    return DatasetIndexSplit(
        training_indices=training_indices,
        validation_indices=validation_indices,
    )


def _read_split_ratios(
    data_config: dict,
) -> tuple[float, float, float]:
    """Read and validate train, validation, and test ratios."""

    train_ratio = float(data_config["train_ratio"])

    if "validation_ratio" in data_config:
        validation_ratio = float(data_config["validation_ratio"])
    elif "val_ratio" in data_config:
        validation_ratio = float(data_config["val_ratio"])
    else:
        raise KeyError(
            "data配置中缺少validation_ratio。"
        )

    test_ratio = float(data_config["test_ratio"])

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(
            "train_ratio、validation_ratio和test_ratio都必须大于0。"
        )

    ratio_sum = sum(ratios)

    if not np.isclose(
        ratio_sum,
        1.0,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            "train_ratio、validation_ratio和test_ratio之和必须等于1，"
            f"当前之和为{ratio_sum:.12g}。"
        )

    return ratios


def _calculate_split_counts(
    number_of_items: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    *,
    unit_description: str = "可划分单位",
) -> tuple[int, int, int]:
    """Calculate non-empty subset counts for one group of items.

    The train and validation counts are obtained by flooring their products.
    The remainder is assigned to the test subset, so all items are used once.
    """

    number_of_items = int(number_of_items)

    if number_of_items < 3:
        raise ValueError(
            f"{unit_description}只有{number_of_items}个，"
            "至少需要3个才能建立非空的训练、验证和测试子集。"
        )

    train_count = int(number_of_items * train_ratio)
    validation_count = int(number_of_items * validation_ratio)
    test_count = number_of_items - train_count - validation_count

    if train_count == 0:
        raise ValueError(
            f"{unit_description}按当前比例划分后训练集为空。"
            "请增加数据量或调整train_ratio。"
        )

    if validation_count == 0:
        raise ValueError(
            f"{unit_description}按当前比例划分后验证集为空。"
            "请增加数据量或调整validation_ratio。"
        )

    if test_count == 0:
        raise ValueError(
            f"{unit_description}按当前比例划分后测试集为空。"
            "请增加数据量或调整test_ratio。"
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
    """Select spectra while supporting arrays and variable-length lists."""

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
    """Create a validated subset from original spectrum indices."""

    indices = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(-1)

    if indices.size == 0:
        raise ValueError("不能创建空的数据子集。")

    number_of_spectra = len(collection.spectra)

    if (
        np.any(indices < 0)
        or np.any(indices >= number_of_spectra)
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
    """Return unique values while preserving first-appearance order."""

    unique_values: list[object] = []
    seen_keys: set[str] = set()

    for value in values.tolist():
        key = str(value)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_values.append(value)

    return unique_values


def _indices_for_source_files(
    *,
    all_source_files: np.ndarray,
    selected_source_files: list[object],
) -> np.ndarray:
    """Return all spectrum indices belonging to selected source files."""

    source_file_to_indices: dict[str, list[int]] = {}

    for spectrum_index, source_file in enumerate(
        all_source_files.tolist()
    ):
        source_key = str(source_file)
        source_file_to_indices.setdefault(
            source_key,
            [],
        ).append(spectrum_index)

    selected_indices: list[int] = []

    for source_file in selected_source_files:
        source_key = str(source_file)

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
    """Split all individual spectra globally."""

    number_of_spectra = len(collection.spectra)

    (
        train_count,
        validation_count,
        _,
    ) = _calculate_split_counts(
        number_of_items=number_of_spectra,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        unit_description="全部光谱",
    )

    all_indices = np.arange(
        number_of_spectra,
        dtype=np.int64,
    )

    if shuffle:
        random_generator = np.random.default_rng(random_seed)
        all_indices = random_generator.permutation(all_indices)

    validation_end = train_count + validation_count

    return SpectrumDatasetSplit(
        train=_make_subset(
            collection,
            all_indices[:train_count],
        ),
        validation=_make_subset(
            collection,
            all_indices[train_count:validation_end],
        ),
        test=_make_subset(
            collection,
            all_indices[validation_end:],
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
    """Split complete source files as indivisible units."""

    all_source_files = np.asarray(
        collection.source_files,
        dtype=object,
    )
    unique_source_files = _ordered_unique_values(all_source_files)

    (
        train_file_count,
        validation_file_count,
        _,
    ) = _calculate_split_counts(
        number_of_items=len(unique_source_files),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        unit_description="源文件",
    )

    if shuffle:
        random_generator = np.random.default_rng(random_seed)
        shuffled_positions = random_generator.permutation(
            len(unique_source_files)
        )
        unique_source_files = [
            unique_source_files[int(position)]
            for position in shuffled_positions
        ]

    validation_file_end = train_file_count + validation_file_count

    train_source_files = unique_source_files[:train_file_count]
    validation_source_files = unique_source_files[
        train_file_count:validation_file_end
    ]
    test_source_files = unique_source_files[validation_file_end:]

    return SpectrumDatasetSplit(
        train=_make_subset(
            collection,
            _indices_for_source_files(
                all_source_files=all_source_files,
                selected_source_files=train_source_files,
            ),
        ),
        validation=_make_subset(
            collection,
            _indices_for_source_files(
                all_source_files=all_source_files,
                selected_source_files=validation_source_files,
            ),
        ),
        test=_make_subset(
            collection,
            _indices_for_source_files(
                all_source_files=all_source_files,
                selected_source_files=test_source_files,
            ),
        ),
    )


def _validate_spectrum_within_folder_input(
    collection: SpectrumCollection,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the strict input contract for folder-internal splitting.

    This mode requires:

    1. every spectrum to be located below a named subfolder;
    2. every source file to contribute exactly one spectrum;
    3. every spectrum to have a folder label recorded by the reader.
    """

    labels = np.asarray(
        collection.labels,
        dtype=object,
    ).reshape(-1)
    source_files = np.asarray(
        collection.source_files,
        dtype=object,
    ).reshape(-1)

    number_of_spectra = len(collection.spectra)

    if labels.size != number_of_spectra:
        raise ValueError(
            "labels数量与光谱数量不一致，"
            "无法执行文件夹内部划分。"
        )

    if source_files.size != number_of_spectra:
        raise ValueError(
            "source_files数量与光谱数量不一致。"
        )

    root_indices = np.flatnonzero(
        np.asarray(
            [str(value) == ROOT_LABEL for value in labels],
            dtype=bool,
        )
    )

    if root_indices.size > 0:
        first_index = int(root_indices[0])
        raise ValueError(
            "split_unit=spectrum_within_folder要求所有光谱文件"
            "位于data.input_directory的子文件夹中。"
            f"检测到根目录文件：{source_files[first_index]}。"
        )

    source_counts: dict[str, int] = {}

    for source_file in source_files.tolist():
        source_key = str(source_file)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

    repeated_sources = [
        (source_file, count)
        for source_file, count in source_counts.items()
        if count != 1
    ]

    if repeated_sources:
        source_file, count = repeated_sources[0]
        raise ValueError(
            "split_unit=spectrum_within_folder要求每个源文件"
            "恰好包含一条光谱。"
            f"文件{source_file!r}被读取为{count}条光谱。"
            "请把每条光谱保存为独立文件，"
            "或改用split_unit=source_file。"
        )

    return labels, source_files


def _derive_folder_random_seed(
    *,
    base_seed: int,
    folder_label: str,
) -> int:
    """为每个文件夹生成跨运行稳定且相互独立的随机种子。

    不能使用Python内置hash，因为其结果可能随解释器进程变化。
    使用SHA-256后，即使以后新增其他文件夹，已有文件夹内部的
    train、validation和test归属也不会被连带改变。
    """

    seed_material = (
        f"{int(base_seed)}\n{str(folder_label)}"
    ).encode("utf-8")

    digest = hashlib.sha256(seed_material).digest()

    # NumPy Generator接受非负整数种子。取前8字节即可提供
    # 足够的文件夹间区分度，同时保持实现清晰。
    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def _split_spectra_within_folder(
    *,
    collection: SpectrumCollection,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
    shuffle: bool,
) -> SpectrumDatasetSplit:
    """Independently split spectra inside every sample-combination folder.

    For example, when one folder contains 20 single-spectrum files and the
    ratios are 0.8/0.1/0.1, that folder contributes 16 train spectra,
    2 validation spectra, and 2 test spectra. The same calculation is applied
    independently to every other folder before the indices are merged.
    """

    labels, _ = _validate_spectrum_within_folder_input(collection)
    folder_labels = _ordered_unique_values(labels)

    if not folder_labels:
        raise ValueError(
            "没有找到可用于文件夹内部划分的样品文件夹。"
        )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for folder_label in folder_labels:
        folder_key = str(folder_label)
        folder_indices = np.flatnonzero(
            np.asarray(
                [str(value) == folder_key for value in labels],
                dtype=bool,
            )
        ).astype(
            np.int64,
            copy=False,
        )

        (
            train_count,
            validation_count,
            _,
        ) = _calculate_split_counts(
            number_of_items=folder_indices.size,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            unit_description=(
                f"文件夹{folder_key!r}中的单光谱文件"
            ),
        )

        if shuffle:
            folder_seed = _derive_folder_random_seed(
                base_seed=int(random_seed),
                folder_label=folder_key,
            )
            folder_generator = np.random.default_rng(folder_seed)
            folder_indices = folder_generator.permutation(
                folder_indices
            ).astype(
                np.int64,
                copy=False,
            )

        validation_end = train_count + validation_count

        train_indices.extend(
            folder_indices[:train_count].tolist()
        )
        validation_indices.extend(
            folder_indices[
                train_count:validation_end
            ].tolist()
        )
        test_indices.extend(
            folder_indices[validation_end:].tolist()
        )

    return SpectrumDatasetSplit(
        train=_make_subset(
            collection,
            np.asarray(train_indices, dtype=np.int64),
        ),
        validation=_make_subset(
            collection,
            np.asarray(validation_indices, dtype=np.int64),
        ),
        test=_make_subset(
            collection,
            np.asarray(test_indices, dtype=np.int64),
        ),
    )


def _validate_complete_split(
    *,
    dataset_split: SpectrumDatasetSplit,
    number_of_spectra: int,
) -> None:
    """Check that every original spectrum appears exactly once."""

    all_indices = np.concatenate(
        [
            dataset_split.train.indices,
            dataset_split.validation.indices,
            dataset_split.test.indices,
        ]
    )

    if all_indices.size != number_of_spectra:
        raise RuntimeError(
            "划分后的光谱总数与原始光谱总数不一致。"
        )

    if np.unique(all_indices).size != number_of_spectra:
        raise RuntimeError(
            "训练集、验证集和测试集之间存在重复光谱，"
            "或者有光谱未被划分。"
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
    """Split a complete spectrum collection using the configured strategy.

    Supported ``data.split_unit`` values:

    ``source_file``
        Keep all spectra from each source file in one subset.

    ``spectrum``
        Split all spectra globally. This mode is mainly for diagnostics.

    ``spectrum_within_folder``
        Independently split the single-spectrum files inside every folder.
        Every folder therefore contributes spectra to train, validation,
        and test according to the configured ratios.
    """

    number_of_spectra = len(collection.spectra)

    if number_of_spectra == 0:
        raise ValueError(
            "不能划分空的光谱数据集。"
        )

    if len(collection.source_files) != number_of_spectra:
        raise ValueError(
            "source_files数量与光谱数量不一致。"
        )

    if len(collection.spectrum_names) != number_of_spectra:
        raise ValueError(
            "spectrum_names数量与光谱数量不一致。"
        )

    (
        train_ratio,
        validation_ratio,
        test_ratio,
    ) = _read_split_ratios(data_config)

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

    elif split_unit == "spectrum_within_folder":
        dataset_split = _split_spectra_within_folder(
            collection=collection,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            random_seed=int(random_seed),
            shuffle=shuffle,
        )

    else:
        raise ValueError(
            "data.split_unit只支持source_file、spectrum或"
            "spectrum_within_folder，"
            f"当前值为{split_unit!r}。"
        )

    _validate_complete_split(
        dataset_split=dataset_split,
        number_of_spectra=number_of_spectra,
    )

    return dataset_split
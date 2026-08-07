"""Tests for reproducible and folder-aware dataset splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dataset_splitter import (
    split_dataset_indices,
    split_spectrum_collection,
)
from src.spectrum_file_reader import SpectrumCollection


def test_dataset_split_is_reproducible() -> None:
    """Retain the original compatibility test."""

    split_a = split_dataset_indices(
        number_of_spectra=20,
        validation_fraction=0.2,
        random_seed=42,
    )

    split_b = split_dataset_indices(
        number_of_spectra=20,
        validation_fraction=0.2,
        random_seed=42,
    )

    np.testing.assert_array_equal(
        split_a.training_indices,
        split_b.training_indices,
    )
    np.testing.assert_array_equal(
        split_a.validation_indices,
        split_b.validation_indices,
    )

    assert len(split_a.training_indices) == 16
    assert len(split_a.validation_indices) == 4


def _build_folder_collection(
    folder_sizes: dict[str, int],
    *,
    spectra_per_source_file: int = 1,
    include_root_file: bool = False,
) -> SpectrumCollection:
    """Build a synthetic collection matching the real reader output."""

    spectra: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    lengths: list[int] = []
    source_files: list[str] = []
    relative_source_files: list[str] = []
    spectrum_names: list[str] = []
    labels: list[str] = []

    axis = np.asarray(
        [600.0, 601.0, 602.0, 603.0],
        dtype=np.float32,
    )

    spectrum_counter = 0

    for folder_name, number_of_files in folder_sizes.items():
        for file_index in range(number_of_files):
            file_name = f"spectrum_{file_index + 1:03d}.csv"
            source_path = Path("/tmp/data/input") / folder_name / file_name
            relative_path = Path(folder_name) / file_name

            for column_index in range(spectra_per_source_file):
                value = float(spectrum_counter + column_index)

                spectra.append(
                    np.asarray(
                        [value, value + 1.0, value + 2.0, value + 3.0],
                        dtype=np.float32,
                    )
                )
                axes.append(axis.copy())
                lengths.append(axis.size)
                source_files.append(str(source_path))
                relative_source_files.append(relative_path.as_posix())
                spectrum_names.append(
                    f"{folder_name}_{file_index:03d}_{column_index:02d}"
                )
                labels.append(folder_name)

            spectrum_counter += spectra_per_source_file

    if include_root_file:
        spectra.append(
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )
        axes.append(axis.copy())
        lengths.append(axis.size)
        source_files.append("/tmp/data/input/root_spectrum.csv")
        relative_source_files.append("root_spectrum.csv")
        spectrum_names.append("root_spectrum")
        labels.append("__root__")

    spectra_array = np.empty(len(spectra), dtype=object)
    spectra_array[:] = spectra

    axes_array = np.empty(len(axes), dtype=object)
    axes_array[:] = axes

    return SpectrumCollection(
        raman_shift=axis.copy(),
        spectra=spectra_array,
        raman_shifts=axes_array,
        original_lengths=np.asarray(lengths, dtype=np.int64),
        source_files=np.asarray(source_files, dtype=object),
        relative_source_files=np.asarray(
            relative_source_files,
            dtype=object,
        ),
        spectrum_names=np.asarray(spectrum_names, dtype=object),
        labels=np.asarray(labels, dtype=object),
    )


def _folder_counts(
    collection: SpectrumCollection,
    indices: np.ndarray,
) -> dict[str, int]:
    """Count selected spectra for every folder label."""

    selected_labels = np.asarray(
        collection.labels,
        dtype=object,
    )[indices]

    counts: dict[str, int] = {}

    for label in selected_labels.tolist():
        key = str(label)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _folder_split_config() -> dict[str, object]:
    return {
        "split_unit": "spectrum_within_folder",
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "shuffle": True,
    }


def test_spectrum_within_folder_splits_every_folder_independently() -> None:
    """Every 20-spectrum folder must contribute 16/2/2 spectra."""

    collection = _build_folder_collection(
        {
            "sample_001_DEL_water_1e-6": 20,
            "sample_002_CHL_soil_1e-8": 20,
            "sample_003_DEL_CHL_water": 20,
        }
    )

    dataset_split = split_spectrum_collection(
        collection=collection,
        data_config=_folder_split_config(),
        random_seed=2026,
    )

    expected_train = {
        "sample_001_DEL_water_1e-6": 16,
        "sample_002_CHL_soil_1e-8": 16,
        "sample_003_DEL_CHL_water": 16,
    }
    expected_validation = {
        key: 2
        for key in expected_train
    }
    expected_test = {
        key: 2
        for key in expected_train
    }

    assert _folder_counts(
        collection,
        dataset_split.train.indices,
    ) == expected_train

    assert _folder_counts(
        collection,
        dataset_split.validation.indices,
    ) == expected_validation

    assert _folder_counts(
        collection,
        dataset_split.test.indices,
    ) == expected_test

    all_indices = np.concatenate(
        [
            dataset_split.train.indices,
            dataset_split.validation.indices,
            dataset_split.test.indices,
        ]
    )

    assert all_indices.size == 60
    assert np.unique(all_indices).size == 60


def test_spectrum_within_folder_is_reproducible() -> None:
    """The same seed must produce exactly the same three subsets."""

    collection = _build_folder_collection(
        {
            "folder_a": 20,
            "folder_b": 20,
        }
    )
    configuration = _folder_split_config()

    split_a = split_spectrum_collection(
        collection=collection,
        data_config=configuration,
        random_seed=42,
    )
    split_b = split_spectrum_collection(
        collection=collection,
        data_config=configuration,
        random_seed=42,
    )

    np.testing.assert_array_equal(
        split_a.train.indices,
        split_b.train.indices,
    )
    np.testing.assert_array_equal(
        split_a.validation.indices,
        split_b.validation.indices,
    )
    np.testing.assert_array_equal(
        split_a.test.indices,
        split_b.test.indices,
    )


def test_spectrum_within_folder_rejects_multi_spectrum_source_file() -> None:
    """One source file must not contain several intensity columns."""

    collection = _build_folder_collection(
        {"folder_a": 10},
        spectra_per_source_file=2,
    )

    with pytest.raises(
        ValueError,
        match="每个源文件.*恰好包含一条光谱",
    ):
        split_spectrum_collection(
            collection=collection,
            data_config=_folder_split_config(),
            random_seed=42,
        )


def test_spectrum_within_folder_rejects_root_level_files() -> None:
    """All files must belong to an explicit sample-combination folder."""

    collection = _build_folder_collection(
        {"folder_a": 20},
        include_root_file=True,
    )

    with pytest.raises(
        ValueError,
        match="位于data.input_directory的子文件夹",
    ):
        split_spectrum_collection(
            collection=collection,
            data_config=_folder_split_config(),
            random_seed=42,
        )


def test_spectrum_within_folder_requires_non_empty_subsets_per_folder() -> None:
    """A folder with too few files must fail instead of silently vanishing."""

    collection = _build_folder_collection(
        {
            "folder_with_20": 20,
            "folder_with_9": 9,
        }
    )

    with pytest.raises(
        ValueError,
        match="folder_with_9.*验证集为空",
    ):
        split_spectrum_collection(
            collection=collection,
            data_config=_folder_split_config(),
            random_seed=42,
        )



def _indices_for_label(
    collection: SpectrumCollection,
    indices: np.ndarray,
    label: str,
) -> np.ndarray:
    """Return selected original indices belonging to one folder."""

    labels = np.asarray(collection.labels, dtype=object)

    return np.asarray(
        [
            int(index)
            for index in indices.tolist()
            if str(labels[int(index)]) == label
        ],
        dtype=np.int64,
    )


def test_existing_folder_split_is_stable_when_new_folder_is_added() -> None:
    """Adding another folder must not reshuffle an existing folder."""

    original_collection = _build_folder_collection(
        {
            "folder_a": 20,
            "folder_b": 20,
        }
    )
    expanded_collection = _build_folder_collection(
        {
            "folder_a": 20,
            "folder_b": 20,
            "folder_c": 20,
        }
    )

    configuration = _folder_split_config()

    original_split = split_spectrum_collection(
        collection=original_collection,
        data_config=configuration,
        random_seed=2026,
    )
    expanded_split = split_spectrum_collection(
        collection=expanded_collection,
        data_config=configuration,
        random_seed=2026,
    )

    for subset_name in ("train", "validation", "test"):
        original_indices = _indices_for_label(
            original_collection,
            getattr(original_split, subset_name).indices,
            "folder_a",
        )
        expanded_indices = _indices_for_label(
            expanded_collection,
            getattr(expanded_split, subset_name).indices,
            "folder_a",
        )

        np.testing.assert_array_equal(
            np.sort(original_indices),
            np.sort(expanded_indices),
        )
import numpy as np

from src.dataset_splitter import (
    split_dataset_indices,
)


def test_dataset_split_is_reproducible():
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
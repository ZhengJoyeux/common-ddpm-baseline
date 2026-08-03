import numpy as np

from src.spectrum_dataset import SpectrumDataset


def test_spectrum_dataset_shape():
    spectra = np.random.default_rng(1).random(
        (5, 16),
        dtype=np.float32,
    )

    dataset = SpectrumDataset(spectra)

    assert len(dataset) == 5
    assert dataset[0].shape == (1, 16)
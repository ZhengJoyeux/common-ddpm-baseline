import numpy as np
import pandas as pd

from src.spectrum_file_reader import (
    read_one_spectrum_file,
)


def test_read_wide_csv(tmp_path):
    file_path = tmp_path / "spectra.csv"

    dataframe = pd.DataFrame(
        {
            "Raman_shift_cm-1": [400, 401, 402],
            "sample_001": [0.1, 0.2, 0.3],
            "sample_002": [0.4, 0.5, 0.6],
        }
    )
    dataframe.to_csv(file_path, index=False)

    data_config = {
        "csv_encoding": "utf-8-sig",
        "enforce_expected_range": True,
        "expected_min": 0.0,
        "expected_max": 1.0,
    }

    raman_shift, spectra, names = (
        read_one_spectrum_file(
            file_path,
            data_config,
        )
    )

    assert raman_shift.shape == (3,)
    assert spectra.shape == (2, 3)
    assert len(names) == 2

    np.testing.assert_allclose(
        spectra[0],
        [0.1, 0.2, 0.3],
    )
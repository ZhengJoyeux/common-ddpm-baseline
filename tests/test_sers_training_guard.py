import numpy as np

from src.sers_training_guard import (
    SersTrainingGuard,
    fit_sers_training_guard_state,
)


def _synthetic_training_set():
    axis = np.linspace(600.0, 900.0, 301, dtype=np.float64)
    rng = np.random.default_rng(2026)
    rows = []

    for _ in range(16):
        spectrum = np.zeros_like(axis)
        for center, amplitude, width in (
            (700.0, 100.0, 5.0),
            (800.0, 70.0, 7.0),
        ):
            shifted_center = center + rng.normal(0.0, 0.7)
            shifted_amplitude = amplitude * (
                1.0 + rng.normal(0.0, 0.04)
            )
            shifted_width = width * (
                1.0 + rng.normal(0.0, 0.03)
            )
            spectrum += shifted_amplitude * np.exp(
                -0.5
                * (
                    (axis - shifted_center)
                    / shifted_width
                )
                ** 2
            )
        spectrum += rng.normal(
            0.0,
            0.5,
            size=axis.size,
        )
        rows.append(spectrum)

    return axis, np.stack(rows, axis=0)


def _configuration():
    return {
        "reference_smoothing_sigma_cm1": 20.0,
        "minimum_relative_prominence": 0.05,
        "minimum_peak_distance_cm1": 20.0,
        "maximum_peak_count": 10,
        "metric_half_width_cm1": 18.0,
        "position_margin_cm1": 2.0,
        "bound_margin_iqr": 1.5,
        "minimum_valid_training_count": 8,
        "check_peak_position": True,
        "check_peak_height": True,
        "check_peak_width": True,
        "check_negative_minimum": True,
        "maximum_peak_violation_fraction": 0.20,
    }


def test_training_guard_accepts_training_like_sample():
    axis, training = _synthetic_training_set()

    state = fit_sers_training_guard_state(
        training_spectra=training,
        raman_shift=axis,
        configuration=_configuration(),
    )

    assert len(state["peaks"]) >= 2

    guard = SersTrainingGuard(
        raman_shift=axis,
        state=state,
        configuration=_configuration(),
    )

    accepted, rows, _ = guard.evaluate(
        training[:1],
        spectrum_names=["training_like"],
    )

    assert bool(accepted[0])
    assert bool(rows[0]["accepted"])


def test_training_guard_rejects_extreme_negative_valley():
    axis, training = _synthetic_training_set()

    state = fit_sers_training_guard_state(
        training_spectra=training,
        raman_shift=axis,
        configuration=_configuration(),
    )

    guard = SersTrainingGuard(
        raman_shift=axis,
        state=state,
        configuration=_configuration(),
    )

    abnormal = training[:1].copy()
    abnormal[0, 150] = (
        float(state["negative_minimum_lower_bound"])
        - 100.0
    )

    accepted, rows, _ = guard.evaluate(
        abnormal,
        spectrum_names=["negative_outlier"],
    )

    assert not bool(accepted[0])
    assert bool(rows[0]["negative_violation"])

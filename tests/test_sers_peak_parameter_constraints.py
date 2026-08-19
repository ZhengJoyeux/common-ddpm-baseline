from __future__ import annotations

import numpy as np
import pytest
import torch

from src.sers_peak_parameter_constraints import (
    DifferentiablePeakParameterLoss,
    fit_peak_parameter_constraint_state,
    normalize_peak_parameter_configuration,
)


def _configuration() -> dict:
    return {
        "enabled": True,
        "reference_baseline_sigma_cm1": 30.0,
        "reference_peak_smoothing_sigma_cm1": 1.5,
        "minimum_relative_prominence": 0.05,
        "minimum_peak_distance_cm1": 80.0,
        "maximum_peak_count": 3,
        "minimum_peak_count": 3,
        "edge_exclusion_cm1": 20.0,
        "maximum_measurement_half_window_cm1": 20.0,
        "minimum_measurement_half_window_cm1": 4.0,
        "neighbor_window_fraction": 0.45,
        "minimum_median_peak_height_fraction": 0.10,
        "softplus_temperature_fraction": 0.03,
        "minimum_softplus_temperature": 1.0e-4,
        "lower_quantile": 5.0,
        "upper_quantile": 95.0,
        "position_margin_mad": 0.25,
        "width_margin_mad": 0.25,
        "minimum_position_half_range_cm1": 0.5,
        "minimum_width_half_range_cm1": 0.5,
        "position_scale_cm1": 1.0,
        "width_scale_cm1": 2.0,
        "position_loss_weight": 1.0,
        "width_loss_weight": 0.5,
        "smooth_l1_beta": 0.5,
        "maximum_active_timestep_fraction": 1.0,
        "timestep_weight_power": 1.0,
        "minimum_timestep_weight": 0.0,
        "total_weight": 0.10,
        "maximum_total_ratio_to_ddpm": 0.02,
        "numerical_safety_limit": 12.0,
        "epsilon": 1.0e-8,
    }


def _gaussian(
    axis: np.ndarray,
    *,
    center: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -0.5
        * np.square(
            (axis - center) / sigma
        )
    )


def _build_spectrum(
    axis: np.ndarray,
    *,
    centers: tuple[float, float, float] = (
        724.0,
        988.0,
        1143.0,
    ),
    sigmas: tuple[float, float, float] = (
        4.0,
        5.5,
        6.0,
    ),
    amplitudes: tuple[float, float, float] = (
        0.80,
        0.65,
        0.70,
    ),
) -> np.ndarray:
    spectrum = (
        0.08
        + 0.00002
        * (axis - axis[0])
    )

    for (
        center,
        sigma,
        amplitude,
    ) in zip(
        centers,
        sigmas,
        amplitudes,
        strict=True,
    ):
        spectrum = (
            spectrum
            + _gaussian(
                axis,
                center=center,
                sigma=sigma,
                amplitude=amplitude,
            )
        )

    return spectrum.astype(
        np.float64,
        copy=False,
    )


def _training_spectra(
    axis: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(
        2026
    )

    spectra = []

    nominal_centers = np.asarray(
        [
            724.0,
            988.0,
            1143.0,
        ],
        dtype=np.float64,
    )

    nominal_sigmas = np.asarray(
        [
            4.0,
            5.5,
            6.0,
        ],
        dtype=np.float64,
    )

    nominal_amplitudes = np.asarray(
        [
            0.80,
            0.65,
            0.70,
        ],
        dtype=np.float64,
    )

    for _ in range(16):
        centers = (
            nominal_centers
            + rng.normal(
                0.0,
                0.30,
                size=3,
            )
        )

        sigmas = (
            nominal_sigmas
            * (
                1.0
                + rng.normal(
                    0.0,
                    0.025,
                    size=3,
                )
            )
        )

        amplitudes = (
            nominal_amplitudes
            * (
                1.0
                + rng.normal(
                    0.0,
                    0.03,
                    size=3,
                )
            )
        )

        spectrum = _build_spectrum(
            axis,
            centers=tuple(
                float(value)
                for value
                in centers
            ),
            sigmas=tuple(
                float(value)
                for value
                in sigmas
            ),
            amplitudes=tuple(
                float(value)
                for value
                in amplitudes
            ),
        )

        spectrum = (
            spectrum
            + rng.normal(
                0.0,
                2.0e-4,
                size=axis.size,
            )
        )

        spectra.append(
            spectrum
        )

    return np.stack(
        spectra,
        axis=0,
    )


def _fit_state():
    axis = np.linspace(
        600.0,
        1300.0,
        701,
        dtype=np.float64,
    )

    training = _training_spectra(
        axis
    )

    state = (
        fit_peak_parameter_constraint_state(
            training_normalized_spectra=training,
            raman_shift=axis,
            configuration=_configuration(),
        )
    )

    return (
        axis,
        training,
        state,
    )


def _broad_local_state() -> dict:
    return {
        "enabled": True,
        "local_normalization": {
            "method": "robust_asinh",
            "target_abs_max": 1.0,
            "scale": 0.10,
            "asinh_normalizer": 1.20,
        },
    }


def _encode_local(
    local: np.ndarray,
    *,
    scale: float = 0.10,
    target_abs_max: float = 1.0,
    asinh_normalizer: float = 1.20,
) -> np.ndarray:
    return (
        target_abs_max
        * np.arcsinh(
            local / scale
        )
        / asinh_normalizer
    )


def test_configuration_forbids_fixed_peak_positions():
    with pytest.raises(
        ValueError,
        match="禁止人工固定峰位",
    ):
        normalize_peak_parameter_configuration(
            {
                "enabled": True,
                "peak_positions": [
                    724.0,
                    988.0,
                ],
            }
        )


def test_fit_state_uses_training_only_and_detects_expected_peaks():
    (
        axis,
        training,
        state,
    ) = _fit_state()

    assert state[
        "enabled"
    ] is True

    assert state[
        "fit_on"
    ] == "train_only"

    assert state[
        "contains_fixed_peak_positions"
    ] is False

    assert state[
        "number_of_training_spectra"
    ] == training.shape[0]

    assert state[
        "original_length"
    ] == axis.size

    centers = np.asarray(
        state[
            "detected_peak_centers_cm1"
        ],
        dtype=np.float64,
    )

    assert centers.size == 3

    np.testing.assert_allclose(
        centers,
        np.asarray(
            [
                724.0,
                988.0,
                1143.0,
            ]
        ),
        atol=3.0,
    )

    position_lower = np.asarray(
        state[
            "position_lower_bound_cm1"
        ]
    )

    position_upper = np.asarray(
        state[
            "position_upper_bound_cm1"
        ]
    )

    width_lower = np.asarray(
        state[
            "width_lower_bound_cm1"
        ]
    )

    width_upper = np.asarray(
        state[
            "width_upper_bound_cm1"
        ]
    )

    assert np.all(
        position_lower
        < position_upper
    )

    assert np.all(
        width_lower
        < width_upper
    )

    assert np.all(
        width_lower > 0.0
    )


def test_parameter_values_inside_bounds_have_zero_boundary_loss():
    (
        axis,
        _,
        state,
    ) = _fit_state()

    module = (
        DifferentiablePeakParameterLoss(
            peak_parameter_constraint_state=state,
            broad_local_residual_state=(
                _broad_local_state()
            ),
            padded_length=axis.size,
        )
    )

    positions = (
        0.5
        * (
            module.position_lower
            + module.position_upper
        )
    )

    widths = (
        0.5
        * (
            module.width_lower
            + module.width_upper
        )
    )

    result = module._boundary_loss(
        positions=positions,
        widths=widths,
    )

    assert torch.allclose(
        result[
            "position_per_sample"
        ],
        torch.zeros_like(
            result[
                "position_per_sample"
            ]
        ),
        atol=1.0e-8,
    )

    assert torch.allclose(
        result[
            "width_per_sample"
        ],
        torch.zeros_like(
            result[
                "width_per_sample"
            ]
        ),
        atol=1.0e-8,
    )


def test_shift_and_broadening_activate_loss_and_gradient():
    (
        axis,
        _,
        state,
    ) = _fit_state()

    module = (
        DifferentiablePeakParameterLoss(
            peak_parameter_constraint_state=state,
            broad_local_residual_state=(
                _broad_local_state()
            ),
            padded_length=axis.size,
        )
    )

    base = _build_spectrum(
        axis
    )

    # 人为将第一个峰明显右移并展宽。
    predicted_full = _build_spectrum(
        axis,
        centers=(
            733.0,
            988.0,
            1143.0,
        ),
        sigmas=(
            8.0,
            5.5,
            6.0,
        ),
    )

    local = (
        predicted_full
        - base
    )

    predicted_scaled = (
        _encode_local(
            local
        )
    )

    predicted_tensor = torch.tensor(
        predicted_scaled,
        dtype=torch.float32,
    ).view(
        1,
        1,
        -1,
    )

    predicted_tensor.requires_grad_()

    target_tensor = torch.zeros_like(
        predicted_tensor
    )

    base_tensor = torch.tensor(
        base,
        dtype=torch.float32,
    ).view(
        1,
        1,
        -1,
    )

    result = module(
        predicted_scaled_local_residual=(
            predicted_tensor
        ),
        target_scaled_local_residual=(
            target_tensor
        ),
        reconstruction_base=(
            base_tensor
        ),
        timesteps=torch.zeros(
            1,
            dtype=torch.long,
        ),
        alphas_cumprod=torch.ones(
            100,
            dtype=torch.float32,
        ),
    )

    loss = result[
        "peak_parameter_timestep_weighted_loss"
    ]

    assert torch.isfinite(
        loss
    )

    assert float(
        loss.detach()
    ) > 0.0

    assert (
        float(
            result[
                "peak_position_violation_fraction"
            ].detach()
        )
        > 0.0
        or float(
            result[
                "peak_width_violation_fraction"
            ].detach()
        )
        > 0.0
    )

    loss.backward()

    assert (
        predicted_tensor.grad
        is not None
    )

    assert torch.isfinite(
        predicted_tensor.grad
    ).all()

    assert float(
        predicted_tensor.grad
        .abs()
        .sum()
    ) > 0.0


def test_padding_is_ignored_by_physics_loss():
    (
        axis,
        _,
        state,
    ) = _fit_state()

    padding = 7

    module = (
        DifferentiablePeakParameterLoss(
            peak_parameter_constraint_state=state,
            broad_local_residual_state=(
                _broad_local_state()
            ),
            padded_length=(
                axis.size
                + padding
            ),
        )
    )

    base = _build_spectrum(
        axis
    )

    predicted_full = _build_spectrum(
        axis,
        centers=(
            733.0,
            988.0,
            1143.0,
        ),
        sigmas=(
            8.0,
            5.5,
            6.0,
        ),
    )

    local = (
        predicted_full
        - base
    )

    encoded = _encode_local(
        local
    )

    predicted = np.pad(
        encoded,
        (0, padding),
        mode="constant",
        constant_values=0.0,
    )

    target = np.zeros_like(
        predicted
    )

    base_padded = np.pad(
        base,
        (0, padding),
        mode="constant",
        constant_values=123.0,
    )

    predicted_tensor = torch.tensor(
        predicted,
        dtype=torch.float32,
    ).view(
        1,
        1,
        -1,
    )

    predicted_tensor.requires_grad_()

    result = module(
        predicted_scaled_local_residual=(
            predicted_tensor
        ),
        target_scaled_local_residual=(
            torch.tensor(
                target,
                dtype=torch.float32,
            ).view(
                1,
                1,
                -1,
            )
        ),
        reconstruction_base=(
            torch.tensor(
                base_padded,
                dtype=torch.float32,
            ).view(
                1,
                1,
                -1,
            )
        ),
        timesteps=torch.zeros(
            1,
            dtype=torch.long,
        ),
        alphas_cumprod=torch.ones(
            100,
            dtype=torch.float32,
        ),
    )

    loss = result[
        "peak_parameter_timestep_weighted_loss"
    ]

    loss.backward()

    tail_gradient = (
        predicted_tensor.grad[
            ...,
            axis.size:
        ]
    )

    assert torch.allclose(
        tail_gradient,
        torch.zeros_like(
            tail_gradient
        ),
        atol=0.0,
        rtol=0.0,
    )

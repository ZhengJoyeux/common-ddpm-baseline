"""D3.2 目标自适应局部病态峰约束测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.sers_physics_constraints import (
    DifferentiableSersPhysicsLoss,
    fit_sers_physics_constraint_state,
    normalize_physics_configuration,
)


def _physics_configuration() -> dict:
    return {
        "enabled": True,
        "strategy": "target_adaptive_peak_morphology",
        "total_weight": 0.02,
        "inverse_transform": {
            "numerical_safety_limit": 15.0,
        },
        "timestep_weighting": {
            "mode": "sqrt_alpha_cumprod",
            "minimum_weight": 0.05,
            "pathology_minimum_weight": 0.50,
        },
        "peak_detection": {
            "smoothing_sigma_cm1": 1.5,
            "local_maximum_radius_cm1": 3.0,
            "prominence_radius_cm1": 10.0,
            "minimum_peak_separation_cm1": 8.0,
            "analysis_half_width_cm1": 16.0,
            "maximum_peaks_per_spectrum": 8,
            "minimum_relative_prominence": 0.05,
            "minimum_prominence_noise_multiplier": 3.0,
            "training_prominence_floor_quantile": 20.0,
            "minimum_absolute_prominence": 1.0e-4,
            "edge_exclusion_cm1": 18.0,
            "positive_softplus_temperature": 0.005,
            "epsilon": 1.0e-8,
        },
        "peak_position": {
            "zero_penalty_tolerance_cm1": 5.0,
            "transition_width_cm1": 5.0,
        },
        "peak_width": {
            "narrow_relative_tolerance": 0.30,
            "broad_relative_tolerance": 0.50,
            "absolute_tolerance_cm1": 1.5,
            "transition_fraction": 0.25,
            "narrow_penalty_multiplier": 2.5,
            "broad_penalty_multiplier": 0.5,
        },
        "peak_sharpness": {
            "upper_relative_tolerance": 0.25,
            "transition_fraction": 0.20,
        },
        "peak_presence": {
            "minimum_area_ratio": 0.45,
            "transition_fraction": 0.25,
        },
        "local_shape": {
            "profile_tolerance": 0.10,
            "first_derivative_tolerance": 0.25,
            "second_derivative_tolerance": 0.30,
            "unimodality_tolerance": 0.05,
            "total_variation_tolerance": 0.20,
            "target_support_fraction": 0.08,
            "transition_fraction": 0.20,
            "profile_weight": 1.5,
            "first_derivative_weight": 1.0,
            "second_derivative_weight": 1.0,
            "unimodality_weight": 2.0,
            "total_variation_weight": 1.0,
        },
        "negative_valley": {
            "enabled": True,
            "noise_margin_multiplier": 4.0,
            "minimum_margin_fraction": 0.02,
            "center_weight_sigma_fraction": 0.40,
            "transition_fraction": 0.20,
            "weight_within_extreme": 2.0,
        },
        "pathology_aggregation": {
            "mean_weight": 0.50,
            "topk_weight": 0.50,
            "topk_fraction": 0.01,
        },
        "roughness": {
            "first_derivative_quantile": 99.5,
            "second_derivative_quantile": 99.5,
            "limit_multiplier": 1.15,
            "first_transition_fraction": 0.20,
            "second_transition_fraction": 0.20,
            "first_derivative_weight": 1.0,
            "second_derivative_weight": 1.5,
        },
        "pointwise_envelope": {
            "enabled": True,
            "lower_quantile": 1.0,
            "upper_quantile": 99.0,
            "mad_margin_multiplier": 3.0,
            "transition_fraction": 0.20,
        },
        "extreme_intensity": {
            "lower_margin_fraction": 0.03,
            "upper_margin_fraction": 0.03,
            "transition_fraction": 0.08,
        },
        "scaled_residual_guard": {
            "enabled": True,
            "absolute_quantile": 99.5,
            "limit_multiplier": 1.10,
            "transition_fraction": 0.15,
        },
        "component_weights": {
            "position": 1.0,
            "width": 1.5,
            "sharpness": 1.5,
            "presence": 0.75,
            "local_shape": 2.0,
            "roughness": 2.0,
            "extreme": 3.0,
            "scaled_residual_guard": 1.5,
        },
    }


def _gaussian(
    axis: np.ndarray,
    center: float,
    sigma: float,
    height: float = 0.6,
) -> np.ndarray:
    return height * np.exp(-0.5 * ((axis - center) / sigma) ** 2)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(600.0, 800.0, 1.0, dtype=np.float64)
    rows = []

    for index in range(16):
        first_center = 650.0 + (index % 3 - 1)
        second_center = 735.0 + (index % 2)
        baseline_noise = 0.0015 * np.sin(
            axis * (0.20 + index * 0.001)
        )
        rows.append(
            -0.20
            + _gaussian(axis, first_center, 5.0, 0.55)
            + _gaussian(axis, second_center, 7.0, 0.35)
            + baseline_noise
        )

    return axis, np.asarray(rows, dtype=np.float32)


def _prior_state(original_length: int) -> dict:
    return {
        "schema_version": 3,
        "enabled": True,
        "domain": "spectrum_global_minmax_normalized",
        "prior_method": "training_pointwise_median",
        "prior_normalized_intensity": [0.0] * original_length,
        "residual_normalization": {
            "method": "pointwise_mad_asinh",
            "target_abs_max": 1.0,
            "pointwise_scale": [1.0] * original_length,
            "standardized_residual_scale": 1.0,
            "asinh_normalizer": 1.0,
        },
    }


def _forward_transform(spectra: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(
        np.arcsinh(spectra),
        dtype=torch.float32,
    ).unsqueeze(1)


def _build_physics_module():
    axis, training = _training_data()
    scaled_residuals = np.arcsinh(training).astype(np.float32)
    state = fit_sers_physics_constraint_state(
        training_normalized_spectra=training,
        training_scaled_residuals=scaled_residuals,
        raman_shift=axis,
        configuration=_physics_configuration(),
    )
    module = DifferentiableSersPhysicsLoss(
        physics_constraint_state=state,
        prior_residual_state=_prior_state(axis.size),
        padded_length=axis.size,
    )

    return axis, training, state, module


def _physics_forward(
    module: DifferentiableSersPhysicsLoss,
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    timestep: int = 0,
) -> dict[str, torch.Tensor]:
    batch_size = predicted.shape[0]

    return module(
        predicted_scaled_residual=_forward_transform(predicted),
        target_scaled_residual=_forward_transform(target),
        timesteps=torch.full(
            (batch_size,),
            int(timestep),
            dtype=torch.long,
        ),
        alphas_cumprod=torch.linspace(
            0.99,
            0.01,
            100,
            dtype=torch.float32,
        ),
    )


def test_configuration_rejects_fixed_peak_windows() -> None:
    configuration = _physics_configuration()
    configuration["peak_windows"] = [
        {
            "minimum_shift": 640.0,
            "maximum_shift": 660.0,
        }
    ]

    with pytest.raises(ValueError, match="不允许配置peak_windows"):
        normalize_physics_configuration(configuration)


def test_state_is_d3_2_and_contains_no_fixed_peak_positions() -> None:
    _, _, state, _ = _build_physics_module()

    assert state["schema_version"] == 4
    assert state["method_version"] == "d3_2_local_pathology_guard"
    assert state["contains_fixed_peak_positions"] is False
    assert "peak_windows" not in state
    assert "peak_centers" not in state
    assert state["training_prominence_floor"] > 0.0


def test_different_samples_use_their_own_target_peaks() -> None:
    axis, _, _, module = _build_physics_module()
    target = np.stack(
        [
            -0.20 + _gaussian(axis, 645.0, 5.0),
            -0.20 + _gaussian(axis, 755.0, 5.0),
        ],
        axis=0,
    ).astype(np.float32)

    result = _physics_forward(module, target, target)

    assert result["mean_detected_peaks"].item() >= 1.0
    assert result["position_loss"].item() < 1.0e-5


def test_five_cm1_position_dead_zone() -> None:
    axis, _, _, module = _build_physics_module()
    target = (
        -0.20 + _gaussian(axis, 700.0, 5.0)
    ).astype(np.float32)[None, :]
    within_tolerance = (
        -0.20 + _gaussian(axis, 704.0, 5.0)
    ).astype(np.float32)[None, :]
    outside_tolerance = (
        -0.20 + _gaussian(axis, 711.0, 5.0)
    ).astype(np.float32)[None, :]

    inside = _physics_forward(
        module,
        within_tolerance,
        target,
    )["position_loss"]
    outside = _physics_forward(
        module,
        outside_tolerance,
        target,
    )["position_loss"]

    assert inside.item() < 1.0e-3
    assert outside.item() > inside.item() + 1.0e-3


def test_narrow_sharp_peak_is_penalized() -> None:
    axis, _, _, module = _build_physics_module()
    target = (
        -0.20 + _gaussian(axis, 700.0, 6.0)
    ).astype(np.float32)[None, :]
    narrow = (
        -0.20 + _gaussian(axis, 700.0, 1.0)
    ).astype(np.float32)[None, :]

    result = _physics_forward(module, narrow, target)

    assert result["width_loss"].item() > 0.0
    assert result["sharpness_loss"].item() > 0.0
    assert result["local_shape_loss"].item() > 0.0


def test_deep_negative_valley_inside_target_peak_is_strongly_penalized() -> None:
    axis, _, _, module = _build_physics_module()
    target = (
        -0.20 + _gaussian(axis, 700.0, 6.0, 0.70)
    ).astype(np.float32)[None, :]
    acceptable = (
        -0.20 + _gaussian(axis, 702.0, 6.0, 0.68)
    ).astype(np.float32)[None, :]
    pathological = acceptable.copy()
    pathological[0, np.argmin(np.abs(axis - 700.0))] -= 1.5

    acceptable_result = _physics_forward(module, acceptable, target)
    pathological_result = _physics_forward(module, pathological, target)

    assert pathological_result["negative_valley_loss"].item() > 0.0
    assert (
        pathological_result["extreme_loss"].item()
        > acceptable_result["extreme_loss"].item() + 0.1
    )


def test_one_peak_split_into_three_narrow_peaks_is_penalized() -> None:
    axis, _, _, module = _build_physics_module()
    target = (
        -0.20 + _gaussian(axis, 700.0, 6.0, 0.75)
    ).astype(np.float32)[None, :]
    acceptable = (
        -0.20 + _gaussian(axis, 701.0, 6.0, 0.73)
    ).astype(np.float32)[None, :]
    split = (
        -0.20
        + _gaussian(axis, 696.0, 0.8, 0.45)
        + _gaussian(axis, 700.0, 0.8, 0.55)
        + _gaussian(axis, 704.0, 0.8, 0.45)
    ).astype(np.float32)[None, :]

    acceptable_result = _physics_forward(module, acceptable, target)
    split_result = _physics_forward(module, split, target)

    assert (
        split_result["local_shape_loss"].item()
        > acceptable_result["local_shape_loss"].item() + 0.05
    )
    assert split_result["roughness_loss"].item() > 0.0


def test_sparse_single_point_pathology_is_not_diluted_by_full_spectrum() -> None:
    axis, training, _, module = _build_physics_module()
    target = training[:1]
    abnormal = target.copy()
    abnormal[0, np.argmin(np.abs(axis - 735.0))] -= 2.0

    normal_result = _physics_forward(module, target, target)
    abnormal_result = _physics_forward(module, abnormal, target)

    assert (
        abnormal_result["extreme_loss"].item()
        > normal_result["extreme_loss"].item() + 0.05
    )
    assert (
        abnormal_result["roughness_loss"].item()
        > normal_result["roughness_loss"].item()
    )


def test_extreme_bipolar_oscillation_and_residual_are_penalized() -> None:
    _, training, _, module = _build_physics_module()
    target = training[:1]
    abnormal = target.copy()
    abnormal[:, 60:80] += np.where(
        np.arange(20) % 2 == 0,
        4.0,
        -4.0,
    )

    result = _physics_forward(module, abnormal, target)

    assert result["roughness_loss"].item() > 0.0
    assert result["extreme_loss"].item() > 0.0
    assert result["scaled_residual_guard_loss"].item() > 0.0


def test_pathology_weight_remains_high_at_high_noise_timestep() -> None:
    _, training, _, module = _build_physics_module()
    target = training[:1]
    abnormal = target.copy()
    abnormal[:, 80] -= 2.0

    result = _physics_forward(
        module,
        abnormal,
        target,
        timestep=99,
    )

    assert result["mean_pathology_timestep_weight"].item() >= 0.50
    assert result["mean_timestep_weight"].item() <= 0.11


def test_inverse_transform_is_finite_and_differentiable() -> None:
    axis, _, _, module = _build_physics_module()
    scaled = torch.full(
        (1, 1, axis.size),
        3.0,
        dtype=torch.float32,
        requires_grad=True,
    )

    restored = module.inverse_scaled_residual(scaled)
    restored.mean().backward()

    assert torch.isfinite(restored).all()
    assert scaled.grad is not None
    assert torch.isfinite(scaled.grad).all()
    assert scaled.grad.abs().sum().item() > 0.0


def test_complete_forward_is_finite_and_differentiable() -> None:
    _, training, _, module = _build_physics_module()
    predicted = _forward_transform(training[:2]).requires_grad_(True)
    target = _forward_transform(training[:2])
    result = module(
        predicted_scaled_residual=predicted,
        target_scaled_residual=target,
        timesteps=torch.tensor([0, 50], dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )

    loss = result["physics_timestep_weighted_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
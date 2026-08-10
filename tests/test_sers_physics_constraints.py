"""D3 SERS 物理约束：病态保护 + 训练集稳定峰分布测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.sers_physics_constraints import (
    PHYSICS_STATE_SCHEMA_VERSION,
    TRAINING_PEAK_METHOD_VERSION,
    DifferentiableSersPhysicsLoss,
    fit_sers_physics_constraint_state,
    normalize_physics_configuration,
)


def _gaussian(
    axis: np.ndarray,
    center: float,
    sigma: float,
    height: float = 0.6,
) -> np.ndarray:
    return height * np.exp(-0.5 * ((axis - center) / sigma) ** 2)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(600.0, 901.0, 1.0, dtype=np.float64)
    generator = np.random.default_rng(2026)
    rows = []
    for index in range(16):
        first_center = 680.0 + float(index % 5 - 2)
        second_center = 790.0 + float(index % 3 - 1)
        first_width = 5.0 + 0.15 * float(index % 3 - 1)
        second_width = 7.0 + 0.20 * float(index % 4 - 1)
        first_height = 0.50 + 0.015 * float(index % 4)
        second_height = 0.34 + 0.012 * float(index % 5)
        baseline = -0.30 + 0.00010 * (axis - axis[0])
        noise = generator.normal(0.0, 0.0025, size=axis.size)
        rows.append(
            baseline
            + _gaussian(axis, first_center, first_width, first_height)
            + _gaussian(axis, second_center, second_width, second_height)
            + noise
        )
    return axis, np.asarray(rows, dtype=np.float32)


def _configuration() -> dict:
    return {
        "enabled": True,
        "strategy": "target_adaptive_peak_morphology",
        "total_weight": 0.02,
        "inverse_transform": {"numerical_safety_limit": 15.0},
        "timestep_weighting": {
            "mode": "sqrt_alpha_cumprod",
            "minimum_weight": 0.20,
            "pathology_minimum_weight": 0.50,
        },
        "peak_detection": {
            "smoothing_sigma_cm1": 2.0,
            "local_maximum_radius_cm1": 4.0,
            "prominence_radius_cm1": 12.0,
            "minimum_peak_separation_cm1": 10.0,
            "analysis_half_width_cm1": 20.0,
            "maximum_peaks_per_spectrum": 12,
            "minimum_relative_prominence": 0.08,
            "minimum_prominence_noise_multiplier": 4.0,
            "training_prominence_floor_quantile": 25.0,
            "minimum_absolute_prominence": 1.0e-4,
            "edge_exclusion_cm1": 24.0,
            "positive_softplus_temperature": 0.01,
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
            "profile_weight": 0.0,
            "first_derivative_weight": 0.0,
            "second_derivative_weight": 0.0,
            "unimodality_weight": 2.0,
            "total_variation_weight": 1.0,
        },
        "training_peak_distribution": {
            "enabled": True,
            "maximum_reference_peaks": 8,
            "minimum_relative_prominence": 0.05,
            "training_prominence_floor_multiplier": 0.50,
            "analysis_half_width_cm1": 30.0,
            "maximum_absolute_shift_cm1": 5.0,
            "shift_transition_cm1": 1.0,
            "coherence_tolerance_cm1": 1.0,
            "coherence_transition_cm1": 0.5,
            "apex_softmax_temperature_fraction": 0.05,
            "height_lower_quantile": 5.0,
            "height_upper_quantile": 95.0,
            "height_mad_margin_multiplier": 0.50,
            "height_transition_fraction": 0.20,
            "width_lower_quantile": 5.0,
            "width_upper_quantile": 95.0,
            "width_mad_margin_multiplier": 0.50,
            "width_transition_fraction": 0.20,
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
            "mad_margin_multiplier": 2.0,
            "transition_fraction": 0.20,
        },
        "extreme_intensity": {
            "lower_margin_fraction": 0.02,
            "upper_margin_fraction": 0.02,
            "transition_fraction": 0.08,
        },
        "scaled_residual_guard": {
            "enabled": True,
            "absolute_quantile": 99.5,
            "limit_multiplier": 1.10,
            "transition_fraction": 0.15,
        },
        "component_weights": {
            "position": 0.0,
            "width": 0.0,
            "sharpness": 0.0,
            "presence": 0.0,
            "local_shape": 0.0,
            "stable_peak_shift": 1.5,
            "stable_peak_coherence": 2.0,
            "stable_peak_height": 2.0,
            "stable_peak_width": 1.0,
            "roughness": 2.0,
            "extreme": 4.0,
            "scaled_residual_guard": 2.0,
        },
    }


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


def _build_module():
    axis, training = _training_data()
    scaled_residuals = np.arcsinh(training).astype(np.float32)
    state = fit_sers_physics_constraint_state(
        training_normalized_spectra=training,
        training_scaled_residuals=scaled_residuals,
        raman_shift=axis,
        configuration=_configuration(),
    )
    module = DifferentiableSersPhysicsLoss(
        physics_constraint_state=state,
        prior_residual_state=_prior_state(axis.size),
        padded_length=axis.size,
    )
    return axis, training, state, module


def _forward(
    module: DifferentiableSersPhysicsLoss,
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    timestep: int = 0,
) -> dict[str, torch.Tensor]:
    predicted_tensor = torch.as_tensor(
        np.arcsinh(predicted), dtype=torch.float32
    ).unsqueeze(1)
    target_tensor = torch.as_tensor(
        np.arcsinh(target), dtype=torch.float32
    ).unsqueeze(1)
    return module(
        predicted_scaled_residual=predicted_tensor,
        target_scaled_residual=target_tensor,
        timesteps=torch.full(
            (predicted.shape[0],), int(timestep), dtype=torch.long
        ),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )


def _shift_spectrum(
    axis: np.ndarray,
    spectra: np.ndarray,
    shift_cm1: float,
) -> np.ndarray:
    return np.stack(
        [
            np.interp(
                axis - shift_cm1,
                axis,
                row,
                left=float(row[0]),
                right=float(row[-1]),
            )
            for row in spectra
        ],
        axis=0,
    ).astype(np.float32)


def test_configuration_rejects_fixed_peak_positions() -> None:
    config = _configuration()
    config["peak_centers"] = [680.0]
    with pytest.raises(ValueError, match="不允许配置peak_centers"):
        normalize_physics_configuration(config)


def test_training_peak_state_is_fitted_only_from_training_spectra() -> None:
    _, training, state, _ = _build_module()
    assert state["schema_version"] == PHYSICS_STATE_SCHEMA_VERSION
    assert state["method_version"] == TRAINING_PEAK_METHOD_VERSION
    assert state["contains_fixed_peak_positions"] is False
    stable = state["training_peak_distribution"]
    assert stable["enabled"] is True
    assert stable["number_of_training_spectra"] == training.shape[0]
    shifts = np.asarray(stable["selected_peak_raman_shifts"])
    assert shifts.size >= 2
    assert np.min(np.abs(shifts - 680.0)) <= 3.0
    assert np.min(np.abs(shifts - 790.0)) <= 3.0


def test_real_training_spectra_have_small_stable_peak_penalty() -> None:
    _, training, _, module = _build_module()
    result = _forward(module, training, training)
    assert result["stable_peak_shift_loss"].item() < 0.2
    assert result["stable_peak_coherence_loss"].item() < 0.2
    assert result["stable_peak_height_loss"].item() < 0.2
    assert result["stable_peak_width_loss"].item() < 0.2


def test_whole_peak_shift_inside_five_cm1_is_allowed_for_reference_like_spectrum() -> None:
    axis, training, _, module = _build_module()
    reference = np.median(training, axis=0, keepdims=True).astype(np.float32)
    shifted = _shift_spectrum(axis, reference, 4.0)
    result = _forward(module, shifted, reference)
    assert result["stable_peak_shift_loss"].item() < 0.1
    assert result["stable_peak_coherence_loss"].item() < 0.1
    assert result["stable_peak_mean_absolute_shift_cm1"].item() < 5.0


def test_whole_peak_shift_beyond_five_cm1_is_penalized() -> None:
    axis, training, _, module = _build_module()
    reference = np.median(training, axis=0, keepdims=True).astype(np.float32)
    shifted = _shift_spectrum(axis, reference, 9.0)
    result = _forward(module, shifted, reference)
    assert result["stable_peak_shift_loss"].item() > 0.1
    assert result["stable_peak_mean_absolute_shift_cm1"].item() > 5.0


def test_apex_only_motion_is_not_treated_as_whole_peak_translation() -> None:
    axis, training, _, module = _build_module()
    reference = np.median(training, axis=0, keepdims=True).astype(np.float32)
    abnormal = reference.copy()
    # 峰体仍在约 680 cm^-1，只在 +4 cm^-1 处制造窄而高的峰头。
    abnormal[0] += _gaussian(axis, 684.0, 0.55, 0.55).astype(np.float32)
    result = _forward(module, abnormal, reference)
    assert (
        result["stable_peak_coherence_loss"].item() > 0.0
        or result["stable_peak_height_loss"].item() > 0.1
        or result["stable_peak_width_loss"].item() > 0.1
    )


def test_excessive_peak_height_is_penalized_by_training_band() -> None:
    axis, training, _, module = _build_module()
    reference = np.median(training, axis=0, keepdims=True).astype(np.float32)
    abnormal = reference.copy()
    abnormal[0] += _gaussian(axis, 680.0, 5.0, 0.80).astype(np.float32)
    result = _forward(module, abnormal, reference)
    assert result["stable_peak_height_loss"].item() > 0.1


def test_deep_negative_valley_remains_penalized() -> None:
    axis, training, _, module = _build_module()
    target = training[:1]
    abnormal = target.copy()
    abnormal[0, np.argmin(np.abs(axis - 680.0))] -= 1.5
    normal = _forward(module, target, target)
    bad = _forward(module, abnormal, target)
    assert bad["negative_valley_loss"].item() > 0.0
    assert bad["extreme_loss"].item() > normal["extreme_loss"].item()


def test_full_spectrum_lower_envelope_penalizes_large_nonpeak_negative_valley() -> None:
    axis, training, _, module = _build_module()
    target = training[:1]
    abnormal = target.copy()
    # 选择没有主峰的 860 cm^-1 附近，验证不是只保护特征峰窗口。
    abnormal[0, np.argmin(np.abs(axis - 860.0))] -= 1.2
    normal = _forward(module, target, target)
    bad = _forward(module, abnormal, target)
    assert bad["extreme_loss"].item() > normal["extreme_loss"].item() + 0.01


def test_pathology_weight_remains_at_high_noise() -> None:
    axis, training, _, module = _build_module()
    abnormal = training[:1].copy()
    abnormal[0, np.argmin(np.abs(axis - 860.0))] -= 1.2
    result = _forward(module, abnormal, training[:1], timestep=99)
    assert result["mean_pathology_timestep_weight"].item() >= 0.50
    assert result["mean_timestep_weight"].item() >= 0.20


def test_loss_is_finite_and_differentiable() -> None:
    _, training, _, module = _build_module()
    predicted = torch.as_tensor(
        np.arcsinh(training[:4]), dtype=torch.float32
    ).unsqueeze(1).requires_grad_(True)
    target = torch.as_tensor(
        np.arcsinh(training[:4]), dtype=torch.float32
    ).unsqueeze(1)
    result = module(
        predicted_scaled_residual=predicted,
        target_scaled_residual=target,
        timesteps=torch.zeros(4, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )
    loss = result["physics_timestep_weighted_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
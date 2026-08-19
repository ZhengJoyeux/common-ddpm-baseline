"""D3.2自动峰区相对峰强软约束测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model_builder import build_diffusion_model
from src.sers_relative_peak_intensity_constraints import (
    DifferentiableRelativePeakIntensityLoss,
    fit_relative_peak_intensity_constraint_state,
    normalize_relative_peak_intensity_configuration,
)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(600.0, 901.0, 1.0, dtype=np.float64)
    generator = np.random.default_rng(20260817)
    rows = []
    for _ in range(16):
        first_center = 705.0 + generator.normal(0.0, 0.7)
        second_center = 812.0 + generator.normal(0.0, 0.6)
        first_width = 5.0 * (1.0 + generator.normal(0.0, 0.02))
        second_width = 7.0 * (1.0 + generator.normal(0.0, 0.02))
        first_height = 0.55 * (1.0 + generator.normal(0.0, 0.025))
        second_height = 0.36 * (1.0 + generator.normal(0.0, 0.025))
        baseline = -0.25 + 0.00015 * (axis - axis[0])
        spectrum = (
            baseline
            + first_height
            * np.exp(-0.5 * ((axis - first_center) / first_width) ** 2)
            + second_height
            * np.exp(-0.5 * ((axis - second_center) / second_width) ** 2)
            + generator.normal(0.0, 0.001, size=axis.size)
        )
        rows.append(spectrum)
    return axis, np.asarray(rows, dtype=np.float32)


def _configuration() -> dict:
    return {
        "enabled": True,
        "reference_baseline_sigma_cm1": 30.0,
        "reference_peak_smoothing_sigma_cm1": 2.0,
        "minimum_relative_prominence": 0.06,
        "minimum_peak_distance_cm1": 12.0,
        "maximum_peak_count": 10,
        "minimum_peak_count": 2,
        "edge_exclusion_cm1": 20.0,
        "peak_half_width_cm1": 12.0,
        "baseline_inner_half_width_cm1": 18.0,
        "baseline_outer_half_width_cm1": 30.0,
        "minimum_relative_height_fraction": 0.03,
        "log_ratio_margin_iqr": 1.5,
        "minimum_log_ratio_width": 0.08,
        "smooth_l1_beta": 0.25,
        "timestep_weight_power": 1.0,
        "minimum_timestep_weight": 0.0,
        "total_weight": 1.0,
        "maximum_total_ratio_to_ddpm": 0.05,
        "numerical_safety_limit": 12.0,
        "epsilon": 1.0e-8,
    }


def _broad_local_state() -> dict:
    return {
        "local_normalization": {
            "method": "robust_asinh",
            "target_abs_max": 1.0,
            "scale": 0.05,
            "asinh_normalizer": float(np.arcsinh(20.0)),
        }
    }


def _build_module():
    axis, training = _training_data()
    state = fit_relative_peak_intensity_constraint_state(
        training_normalized_spectra=training,
        raman_shift=axis,
        configuration=_configuration(),
    )
    module = DifferentiableRelativePeakIntensityLoss(
        relative_peak_intensity_constraint_state=state,
        broad_local_residual_state=_broad_local_state(),
        padded_length=axis.size + 3,
    )
    return axis, training, state, module


def test_configuration_rejects_fixed_peak_positions() -> None:
    configuration = _configuration()
    configuration["peak_centers_cm1"] = [705.0]
    with pytest.raises(ValueError, match="禁止配置固定峰位"):
        normalize_relative_peak_intensity_configuration(configuration)


def test_state_detects_training_peaks_without_fixed_positions() -> None:
    _, _, state, _ = _build_module()
    centers = np.asarray(state["detected_peak_centers_cm1"])
    assert state["fit_on"] == "train_only"
    assert not state["contains_fixed_peak_positions"]
    assert np.any(np.abs(centers - 705.0) <= 3.0)
    assert np.any(np.abs(centers - 812.0) <= 3.0)
    assert len(state["lower_log_relative_peak_intensity"]) >= 2
    assert np.isfinite(state["training_relative_peak_intensities"]).all()


def test_identical_prediction_within_training_bounds_has_zero_boundary_loss() -> None:
    axis, training, _, module = _build_module()
    batch_size = 2
    padded_length = axis.size + 3
    target = torch.zeros(
        (batch_size, 1, padded_length), dtype=torch.float32
    )
    base = torch.as_tensor(
        training[:batch_size], dtype=torch.float32
    ).unsqueeze(1)
    base = torch.nn.functional.pad(base, (0, 3))
    result = module(
        predicted_scaled_local_residual=target.clone(),
        target_scaled_local_residual=target,
        reconstruction_base=base,
        timesteps=torch.zeros(batch_size, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.999, 0.001, 100),
    )
    assert float(
        result["relative_peak_intensity_boundary_loss"]
    ) == pytest.approx(0.0, abs=1.0e-10)


def test_distorted_relative_peak_intensity_has_gradient() -> None:
    axis, training, _, module = _build_module()
    padded_length = axis.size + 3
    target = torch.zeros((1, 1, padded_length), dtype=torch.float32)
    predicted = target.clone()
    distortion = 1.50 * np.exp(-0.5 * ((axis - 705.0) / 4.0) ** 2)
    predicted[0, 0, : axis.size] = torch.as_tensor(
        distortion, dtype=torch.float32
    )
    predicted.requires_grad_(True)
    base = torch.as_tensor(training[:1], dtype=torch.float32).unsqueeze(1)
    base = torch.nn.functional.pad(base, (0, 3))
    result = module(
        predicted_scaled_local_residual=predicted,
        target_scaled_local_residual=target,
        reconstruction_base=base,
        timesteps=torch.tensor([10], dtype=torch.long),
        alphas_cumprod=torch.linspace(0.999, 0.001, 100),
    )
    loss = result["relative_peak_intensity_loss"]
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0.0


def test_high_noise_timestep_receives_smaller_weight() -> None:
    _, _, _, module = _build_module()
    predicted = torch.zeros((2, 1, 304), dtype=torch.float32)
    target = torch.zeros_like(predicted)
    base = torch.zeros_like(predicted)
    result_low = module(
        predicted_scaled_local_residual=predicted,
        target_scaled_local_residual=target,
        reconstruction_base=base,
        timesteps=torch.tensor([0, 0]),
        alphas_cumprod=torch.linspace(0.999, 0.001, 100),
    )
    result_high = module(
        predicted_scaled_local_residual=predicted,
        target_scaled_local_residual=target,
        reconstruction_base=base,
        timesteps=torch.tensor([99, 99]),
        alphas_cumprod=torch.linspace(0.999, 0.001, 100),
    )
    assert float(
        result_high["mean_relative_peak_intensity_timestep_weight"]
    ) < float(result_low["mean_relative_peak_intensity_timestep_weight"])


def test_diffusion_integration_caps_constraint_at_five_percent() -> None:
    axis, training = _training_data()
    configuration = {
        "model": {
            "channels": 1,
            "model_dimension": 8,
            "dimension_multipliers": [1, 2],
            "dropout": 0.0,
            "self_condition": False,
        },
        "diffusion": {
            "diffusion_steps": 10,
            "sampling_steps": 5,
            "objective": "pred_x0",
            "beta_schedule": "cosine",
            "ddim_sampling_eta": 0.0,
            "auto_normalize": False,
            "loss_weighting": "uniform",
            "residual_aware_loss": {"enabled": False},
        },
        "prior_residual": {
            "enabled": True,
            "residual_normalization": "robust_asinh",
        },
        "broad_local_residual": {
            "enabled": True,
            "local_normalization": {"method": "robust_asinh"},
        },
        "physics_constraints": {"enabled": False},
        "diversity_constraints": {"enabled": False},
        "local_peak_distribution_constraints": {"enabled": False},
        "peak_derivative_constraints": {"enabled": False},
        "relative_peak_intensity_constraints": _configuration(),
    }
    _, diffusion = build_diffusion_model(
        model_configuration=configuration,
        sequence_length=axis.size + 3,
    )
    state = fit_relative_peak_intensity_constraint_state(
        training_normalized_spectra=training,
        raman_shift=axis,
        configuration=_configuration(),
    )
    diffusion.configure_relative_peak_intensity_constraints(
        relative_peak_intensity_constraint_state=state,
        broad_local_residual_state=_broad_local_state(),
    )

    target = torch.zeros((2, 1, axis.size + 3), dtype=torch.float32)
    base = torch.as_tensor(training[:2], dtype=torch.float32).unsqueeze(1)
    base = torch.nn.functional.pad(base, (0, 3))
    loss = diffusion.p_losses(
        target,
        torch.tensor([0, 5], dtype=torch.long),
        constraint_reference_prior=base,
    )
    components = diffusion.get_latest_loss_components()

    assert torch.isfinite(loss)
    assert float(components["relative_peak_intensity_loss"]) >= 0.0
    assert float(components["relative_peak_intensity_loss"]) <= (
        0.05 * float(components["ddpm_loss"]) + 1.0e-7
    )
    assert not any(
        "relative_peak_intensity_loss_module" in key
        for key in diffusion.state_dict()
    )

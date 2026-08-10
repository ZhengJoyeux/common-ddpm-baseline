from __future__ import annotations

import numpy as np
import pytest
import torch

from src.sers_local_peak_distribution_constraints import (
    DifferentiableSersLocalPeakDistributionLoss,
    normalize_local_peak_distribution_configuration,
)


def _gaussian(axis: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((axis - center) / width) ** 2)


def _synthetic_training() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(600.0, 721.0, 1.0, dtype=np.float32)
    rows = []
    for index in range(8):
        shift1 = (-1.0, -0.5, 0.0, 0.5)[index % 4]
        shift2 = (-0.8, -0.2, 0.3, 0.9)[index % 4]
        width1 = 5.0 + 0.25 * ((index % 3) - 1)
        width2 = 6.0 + 0.30 * ((index % 3) - 1)
        height1 = 0.88 + 0.025 * ((index % 5) - 2)
        height2 = 0.72 + 0.020 * ((index % 5) - 2)
        spectrum = (
            0.04
            + _gaussian(axis, 640.0 + shift1, width1, height1)
            + _gaussian(axis, 680.0 + shift2, width2, height2)
        )
        rows.append(spectrum.astype(np.float32))
    return axis, np.stack(rows, axis=0)


def _physics_state(axis: np.ndarray) -> dict:
    original_length = axis.size
    lower = np.full(original_length, -0.08, dtype=np.float32)
    # 三个“候选峰”，中间一个故意设置为不稳定/太弱，应被过滤掉。
    peak_indices = [40, 60, 80]
    return {
        "enabled": True,
        "original_length": original_length,
        "raman_shift": axis.tolist(),
        "pointwise_lower_envelope": lower.tolist(),
        "allowed_intensity_minimum": -0.10,
        "allowed_intensity_maximum": 1.10,
        "training_peak_distribution": {
            "enabled": True,
            "analysis_half_width_points": 10,
            "selected_peak_indices": peak_indices,
            "training_position_center_std_cm1": [0.65, 8.0, 0.75],
            "height_median": [0.90, 0.03, 0.74],
            "height_std": [0.055, 0.02, 0.050],
            "width_std_cm1": [0.45, 3.0, 0.55],
        },
    }


def _prior_state(axis: np.ndarray) -> dict:
    return {
        "enabled": True,
        "domain": "spectrum_global_minmax_normalized",
        "prior_normalized_intensity": np.zeros(axis.size, dtype=np.float32).tolist(),
        "residual_normalization": {
            "method": "pointwise_mad_asinh",
            "pointwise_scale": np.ones(axis.size, dtype=np.float32).tolist(),
            "target_abs_max": 1.0,
            "standardized_residual_scale": 1.0,
            "asinh_normalizer": 1.0,
        },
    }


def _configuration() -> dict:
    return {
        "enabled": True,
        "minimum_batch_samples": 3,
        "low_noise_gate": {
            "full_weight_max_timestep": 20,
            "zero_weight_min_timestep": 45,
        },
        "stable_peak_filter": {
            "maximum_position_std_cm1": 2.0,
            "minimum_height_median": 0.05,
        },
    }


def _module() -> tuple[np.ndarray, np.ndarray, DifferentiableSersLocalPeakDistributionLoss]:
    axis, training = _synthetic_training()
    module = DifferentiableSersLocalPeakDistributionLoss(
        configuration=_configuration(),
        physics_constraint_state=_physics_state(axis),
        prior_residual_state=_prior_state(axis),
        padded_length=axis.size,
    )
    return axis, training, module


def _scaled(values: np.ndarray) -> torch.Tensor:
    # synthetic prior-state 的 pointwise_scale=1, standardized_scale=1,
    # asinh_normalizer=1, target_abs_max=1，因此该形式与 inverse 对应。
    return torch.as_tensor(np.arcsinh(values), dtype=torch.float32).unsqueeze(1)


def _forward(
    module: DifferentiableSersLocalPeakDistributionLoss,
    predicted: np.ndarray,
    target: np.ndarray,
    timestep: int = 10,
) -> dict[str, torch.Tensor]:
    batch = predicted.shape[0]
    return module(
        predicted_scaled_residual=_scaled(predicted),
        target_scaled_residual=_scaled(target),
        timesteps=torch.full((batch,), timestep, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )


def test_default_configuration_is_disabled() -> None:
    config = normalize_local_peak_distribution_configuration(None)
    assert config["enabled"] is False


def test_invalid_low_noise_gate_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero_weight_min_timestep"):
        normalize_local_peak_distribution_configuration(
            {
                "enabled": True,
                "low_noise_gate": {
                    "full_weight_max_timestep": 30,
                    "zero_weight_min_timestep": 20,
                },
            }
        )


def test_unstable_weak_peak_is_filtered() -> None:
    _, _, module = _module()
    assert module.stable_peak_indices.tolist() == [40, 80]


def test_deep_negative_tail_is_penalized_more_than_normal_spectrum() -> None:
    _, training, module = _module()
    target = training[:4]
    normal = _forward(module, target, target)

    abnormal = target.copy()
    abnormal[:, 20:23] = -0.50
    deep = _forward(module, abnormal, target)

    assert deep["negative_tail_guard_loss"].item() > normal[
        "negative_tail_guard_loss"
    ].item() + 1.0e-4
    assert deep["negative_tail_guard_contribution"].item() > 0.0


def test_collapsed_peak_shape_triggers_local_distribution_losses() -> None:
    _, training, module = _module()
    target = training[:6]
    median = np.median(target, axis=0, keepdims=True).astype(np.float32)
    predicted = np.repeat(median, target.shape[0], axis=0)

    result = _forward(module, predicted, target)

    assert result["wing_shape_tracking_loss"].item() > 0.0
    assert result["wing_dispersion_loss"].item() > 0.0
    assert result["shift_dispersion_loss"].item() > 0.0
    assert result["width_dispersion_loss"].item() > 0.0


def test_excessive_peak_height_cv_is_penalized() -> None:
    axis, training, module = _module()
    predicted = training[:6].copy()
    multipliers = np.asarray([0.55, 0.70, 0.90, 1.10, 1.35, 1.55], dtype=np.float32)
    for row, multiplier in enumerate(multipliers):
        predicted[row] = (
            0.04
            + multiplier * _gaussian(axis, 640.0, 5.0, 0.90)
            + (2.0 - multiplier) * _gaussian(axis, 680.0, 6.0, 0.74)
        )

    result = _forward(module, predicted, training[:6])
    assert result["height_cv_upper_loss"].item() > 0.0
    assert result["height_cv_upper_contribution"].item() > 0.0


def test_high_noise_disables_peak_distribution_but_keeps_negative_guard() -> None:
    _, training, module = _module()
    target = training[:6]
    predicted = target.copy()
    predicted[:, 15:18] = -0.50

    result = _forward(module, predicted, target, timestep=90)

    assert result["mean_local_peak_timestep_weight"].item() == 0.0
    assert result["wing_shape_tracking_contribution"].item() == 0.0
    assert result["wing_dispersion_contribution"].item() == 0.0
    assert result["shift_dispersion_contribution"].item() == 0.0
    assert result["width_dispersion_contribution"].item() == 0.0
    assert result["height_cv_upper_contribution"].item() == 0.0
    assert result["negative_tail_guard_contribution"].item() > 0.0


def test_total_loss_is_finite_and_differentiable() -> None:
    _, training, module = _module()
    target = _scaled(training[:6])
    predicted = (_scaled(training[:6]) * 0.97).detach().requires_grad_(True)
    result = module(
        predicted_scaled_residual=predicted,
        target_scaled_residual=target,
        timesteps=torch.full((6,), 10, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )
    loss = result["local_peak_distribution_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum().item() > 0.0
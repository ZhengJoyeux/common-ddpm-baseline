"""D3.3-A 训练集峰分布与多样性约束测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.sers_distribution_constraints import (
    DifferentiableSersDistributionLoss,
    fit_sers_distribution_constraint_state,
    normalize_distribution_configuration,
)


def _configuration() -> dict:
    return {
        "enabled": True,
        "total_weight": 0.005,
        "epsilon": 1.0e-8,
        "stable_peak_detection": {
            "smoothing_sigma_cm1": 1.5,
            "prominence_noise_multiplier": 2.5,
            "minimum_relative_prominence": 0.03,
            "consensus_tolerance_cm1": 4.0,
            "minimum_prevalence": 0.60,
            "minimum_separation_cm1": 12.0,
            "analysis_half_width_cm1": 18.0,
            "maximum_stable_peaks": 6,
            "edge_exclusion_cm1": 20.0,
        },
        "feature_range": {
            "lower_quantile": 2.5,
            "upper_quantile": 97.5,
            "iqr_margin_multiplier": 0.25,
            "minimum_position_margin_cm1": 1.0,
            "transition_fraction": 0.20,
        },
        "batch_moments": {
            "mean_tolerance_fraction": 0.20,
            "minimum_std_ratio": 0.70,
            "maximum_std_ratio": 1.50,
            "reference_std_floor_fraction": 0.25,
        },
        "pointwise_diversity": {
            "enabled": True,
            "active_std_quantile": 25.0,
            "minimum_std_ratio": 0.65,
            "maximum_std_ratio": 1.60,
            "reference_std_floor_fraction": 0.25,
        },
        "timestep_weighting": {
            "mode": "sqrt_alpha_cumprod",
            "minimum_weight": 0.05,
        },
        "component_weights": {
            "feature_range": 1.0,
            "feature_mean": 0.5,
            "feature_std": 2.0,
            "pointwise_std": 2.0,
        },
    }


def _gaussian(
    axis: np.ndarray,
    center: float,
    sigma: float,
    height: float,
) -> np.ndarray:
    return height * np.exp(-0.5 * ((axis - center) / sigma) ** 2)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(600.0, 901.0, 1.0, dtype=np.float64)
    rows = []

    for index in range(24):
        center_1 = 680.0 + float(index % 5 - 2)
        center_2 = 790.0 + float(index % 3 - 1)
        height_1 = 0.55 + 0.04 * float(index % 4)
        height_2 = 0.32 + 0.03 * float(index % 5)
        baseline = -0.35 + 0.00015 * (axis - axis[0])
        texture = 0.003 * np.sin(axis * (0.12 + 0.001 * index))
        rows.append(
            baseline
            + _gaussian(axis, center_1, 5.0, height_1)
            + _gaussian(axis, center_2, 7.0, height_2)
            + texture
        )

    return axis, np.asarray(rows, dtype=np.float32)


def _build_module():
    axis, training = _training_data()
    state = fit_sers_distribution_constraint_state(
        training_normalized_spectra=training,
        raman_shift=axis,
        configuration=_configuration(),
    )

    return axis, training, state, DifferentiableSersDistributionLoss(state)


def _forward(
    module: DifferentiableSersDistributionLoss,
    predicted: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_tensor = torch.as_tensor(
        predicted,
        dtype=torch.float32,
    ).unsqueeze(1)
    target_tensor = torch.as_tensor(
        target,
        dtype=torch.float32,
    ).unsqueeze(1)
    batch_size = predicted_tensor.shape[0]

    return module(
        predicted_spectra=predicted_tensor,
        target_spectra=target_tensor,
        timesteps=torch.zeros(batch_size, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )


def test_configuration_rejects_manually_fixed_peaks() -> None:
    configuration = _configuration()
    configuration["peak_positions"] = [680.0, 790.0]

    with pytest.raises(ValueError, match="不允许配置peak_positions"):
        normalize_distribution_configuration(configuration)


def test_state_contains_only_training_learned_stable_peaks() -> None:
    _, _, state, _ = _build_module()

    positions = np.asarray(state["stable_peak_positions_cm1"])

    assert state["schema_version"] == 1
    assert state["method_version"] == (
        "d3_3_a_peak_distribution_diversity"
    )
    assert state["contains_manually_fixed_peak_positions"] is False
    assert state["number_of_training_spectra"] == 24
    assert np.any(np.abs(positions - 680.0) <= 5.0)
    assert np.any(np.abs(positions - 790.0) <= 5.0)
    assert all(value >= 0.60 for value in state["stable_peak_prevalence"])


def test_matching_real_batch_has_lower_loss_than_collapsed_batch() -> None:
    _, training, _, module = _build_module()
    target = training[:16]
    matching = _forward(module, target, target)
    collapsed = np.repeat(
        np.median(target, axis=0, keepdims=True),
        target.shape[0],
        axis=0,
    )
    collapsed_result = _forward(module, collapsed, target)

    assert collapsed_result["distribution_feature_std_loss"].item() > (
        matching["distribution_feature_std_loss"].item() + 1.0e-4
    )
    assert collapsed_result["distribution_pointwise_std_loss"].item() > (
        matching["distribution_pointwise_std_loss"].item() + 1.0e-4
    )
    assert collapsed_result[
        "distribution_timestep_weighted_loss"
    ].item() > matching["distribution_timestep_weighted_loss"].item()


def test_excessive_batch_dispersion_is_penalized() -> None:
    _, training, _, module = _build_module()
    target = training[:16]
    excessive = target.copy()
    scale = np.linspace(0.2, 2.5, excessive.shape[0], dtype=np.float32)
    median = np.median(target, axis=0, keepdims=True)
    excessive = median + scale[:, None] * (excessive - median)
    result = _forward(module, excessive, target)

    assert result["distribution_feature_std_loss"].item() > 0.0
    assert result["distribution_pointwise_std_loss"].item() > 0.0


def test_narrow_split_peak_leaves_training_feature_range() -> None:
    axis, training, _, module = _build_module()
    target = training[:8]
    abnormal = target.copy()
    region = np.abs(axis - 680.0) <= 12.0
    abnormal[:, region] = -0.33

    for row_index in range(abnormal.shape[0]):
        abnormal[row_index] += _gaussian(
            axis,
            676.0,
            0.7,
            0.55,
        )
        abnormal[row_index] += _gaussian(
            axis,
            684.0,
            0.7,
            0.55,
        )

    normal_result = _forward(module, target, target)
    abnormal_result = _forward(module, abnormal, target)

    assert abnormal_result["distribution_feature_range_loss"].item() > (
        normal_result["distribution_feature_range_loss"].item() + 1.0e-3
    )


def test_complete_loss_is_finite_and_differentiable() -> None:
    _, training, _, module = _build_module()
    predicted = torch.as_tensor(
        training[:8],
        dtype=torch.float32,
    ).unsqueeze(1)
    predicted = (predicted + 0.01).requires_grad_(True)
    target = torch.as_tensor(
        training[:8],
        dtype=torch.float32,
    ).unsqueeze(1)
    result = module(
        predicted_spectra=predicted,
        target_spectra=target,
        timesteps=torch.arange(8, dtype=torch.long),
        alphas_cumprod=torch.linspace(0.99, 0.01, 100),
    )
    loss = result["distribution_timestep_weighted_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_single_item_batch_skips_variance_terms() -> None:
    _, training, _, module = _build_module()
    result = _forward(module, training[:1], training[:1])

    assert result["distribution_feature_std_loss"].item() == 0.0
    assert result["distribution_pointwise_std_loss"].item() == 0.0
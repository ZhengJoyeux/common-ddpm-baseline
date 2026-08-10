from __future__ import annotations

import pytest
import torch

from src.residual_aware_loss import (
    build_residual_aware_pointwise_weights,
    normalize_residual_aware_loss_configuration,
)


def configuration() -> dict:
    return {
        "enabled": True,
        "reference_quantile": 0.90,
        "strength": 2.0,
        "power": 1.0,
        "maximum_relative_magnitude": 3.0,
        "minimum_reference_scale": 0.02,
        "normalize_per_sample": True,
    }


def test_configuration_defaults_are_valid() -> None:
    cfg = normalize_residual_aware_loss_configuration(
        {"enabled": True}
    )

    assert cfg["reference_quantile"] == pytest.approx(
        0.90
    )
    assert cfg["strength"] == pytest.approx(
        2.0
    )
    assert cfg["normalize_per_sample"] is True


def test_invalid_quantile_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_residual_aware_loss_configuration(
            {
                "enabled": True,
                "reference_quantile": 1.0,
            }
        )


def test_disabled_returns_unit_weights() -> None:
    target = torch.randn(
        4,
        1,
        32,
    )

    weights, diagnostics = (
        build_residual_aware_pointwise_weights(
            target_scaled_residual=target,
            configuration={
                "enabled": False
            },
        )
    )

    assert torch.allclose(
        weights,
        torch.ones_like(target),
    )

    assert float(
        diagnostics["weight_mean"]
    ) == pytest.approx(1.0)


def test_large_residual_gets_larger_weight() -> None:
    target = torch.tensor(
        [[[
            0.0,
            0.01,
            0.02,
            0.04,
            0.08,
            0.16,
            0.32,
            0.64,
        ]]],
        dtype=torch.float32,
    )

    weights, _ = (
        build_residual_aware_pointwise_weights(
            target_scaled_residual=target,
            configuration=configuration(),
        )
    )

    assert (
        float(weights[0, 0, -1])
        >
        float(weights[0, 0, 0])
    )


def test_weights_mean_one_per_sample() -> None:
    target = torch.tensor(
        [
            [[
                0.0,
                0.01,
                0.02,
                0.05,
                0.10,
                0.20,
            ]],
            [[
                0.0,
                0.02,
                0.04,
                0.10,
                0.20,
                0.40,
            ]],
        ],
        dtype=torch.float32,
    )

    weights, diagnostics = (
        build_residual_aware_pointwise_weights(
            target_scaled_residual=target,
            configuration=configuration(),
        )
    )

    means = (
        weights
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    assert torch.allclose(
        means,
        torch.ones_like(means),
        atol=1.0e-6,
    )

    assert float(
        diagnostics["weight_mean"]
    ) == pytest.approx(
        1.0,
        abs=1.0e-6,
    )


def test_extreme_residual_is_capped() -> None:
    target = torch.tensor(
        [[[
            0.0,
            0.01,
            0.02,
            0.05,
            0.10,
            100.0,
        ]]],
        dtype=torch.float32,
    )

    weights, diagnostics = (
        build_residual_aware_pointwise_weights(
            target_scaled_residual=target,
            configuration=configuration(),
        )
    )

    assert torch.isfinite(
        weights
    ).all()

    assert torch.isfinite(
        diagnostics["weight_max"]
    )

    assert float(
        diagnostics["weight_max"]
    ) < 10.0

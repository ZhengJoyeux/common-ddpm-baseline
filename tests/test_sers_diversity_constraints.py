"""D3.4.2 SERS 高频残差与峰级形态多样性约束测试。"""

from __future__ import annotations

import numpy as np
import torch

from src.sers_diversity_constraints import (
    D3_4_2_METHOD_VERSION,
    DifferentiableSersDiversityLoss,
    fit_sers_diversity_constraint_state,
    normalize_diversity_configuration,
)


def _configuration() -> dict:
    return {
        "enabled": True,
        "strategy": (
            "low_noise_pairwise_residual_geometry"
        ),
        "total_weight": 0.01,
        "epsilon": 1.0e-8,

        "low_noise_gate": {
            "minimum_alpha_cumprod": 0.50,
            "minimum_samples": 4,
        },

        "high_frequency_filter": {
            "smoothing_sigma_points": 1.0,
            "kernel_truncate": 3.0,
            "distance_reference_quantile": 50.0,
            "correlation_reference_quantile": 90.0,
        },

        "pairwise_distance": {
            "enabled": True,
            "metric": "whitened_l2",
            "minimum_distance_ratio": 0.50,
            "maximum_distance_ratio": 2.00,
            "transition_fraction": 0.20,
            "weight": 1.0,
        },

        "pairwise_correlation": {
            "enabled": True,
            "maximum_excess_correlation": 0.005,
            "transition_fraction": 0.20,
            "weight": 0.5,
        },

        "pointwise_variance_floor": {
            "enabled": True,
            "active_std_quantile": 20.0,
            "minimum_std_ratio": 0.30,
            "maximum_std_ratio": 1.70,
            "reference_std_floor_fraction": 0.25,
            "weight": 0.2,
        },

        "peak_morphology": {
            "enabled": True,
            "weight": 1.0,

            "smoothing_sigma_cm1": 2.0,
            "local_maximum_radius_cm1": 4.0,
            "prominence_radius_cm1": 12.0,
            "minimum_peak_separation_cm1": 10.0,
            "analysis_half_width_cm1": 20.0,
            "edge_exclusion_cm1": 24.0,
            "maximum_peaks": 6,

            "minimum_relative_prominence": 0.03,
            "minimum_absolute_prominence": 1.0e-4,
            "positive_softplus_temperature": 0.01,

            "position_dispersion": {
                "minimum_std_ratio": 0.60,
                "maximum_std_ratio": 1.60,
                "scale_floor_cm1": 0.10,
                "weight": 1.0,
            },

            "coherent_shift": {
                "maximum_incoherence_ratio": 1.50,
                "scale_floor_cm1": 0.10,
                "weight": 0.75,
            },

            "height_dispersion": {
                "minimum_std_ratio": 0.60,
                "maximum_std_ratio": 1.60,
                "scale_floor_fraction": 0.02,
                "weight": 0.75,
            },

            "width_dispersion": {
                "minimum_std_ratio": 0.60,
                "maximum_std_ratio": 1.60,
                "scale_floor_fraction": 0.02,
                "weight": 0.75,
            },
        },
    }


def _gaussian(
    axis: np.ndarray,
    center: float,
    sigma: float,
    height: float,
) -> np.ndarray:
    return (
        height
        * np.exp(
            -0.5
            * (
                (
                    axis
                    - center
                )
                / sigma
            )
            ** 2
        )
    )


def _training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    axis = np.arange(
        600.0,
        901.0,
        1.0,
        dtype=np.float64,
    )

    generator = (
        np.random.default_rng(
            2026
        )
    )

    rows = []

    for index in range(
        16
    ):
        first_center = (
            680.0
            + float(
                index % 5
                - 2
            )
        )

        second_center = (
            790.0
            + float(
                index % 3
                - 1
            )
        )

        first_width = (
            5.0
            + 0.25
            * float(
                index % 3
                - 1
            )
        )

        second_width = (
            7.0
            + 0.30
            * float(
                index % 4
            )
        )

        first_height = (
            0.50
            + 0.03
            * float(
                index % 4
            )
        )

        second_height = (
            0.30
            + 0.02
            * float(
                index % 5
            )
        )

        baseline = (
            -0.30
            + 0.00015
            * (
                axis
                - axis[0]
            )
        )

        noise = generator.normal(
            0.0,
            0.004,
            size=axis.size,
        )

        rows.append(
            baseline
            + _gaussian(
                axis,
                first_center,
                first_width,
                first_height,
            )
            + _gaussian(
                axis,
                second_center,
                second_width,
                second_height,
            )
            + noise
        )

    return (
        axis,
        np.asarray(
            rows,
            dtype=np.float32,
        ),
    )


def _build_module():
    axis, training = (
        _training_data()
    )

    # 测试中构造一个简化但可逆的
    # asinh scaled residual。
    scaled = np.arcsinh(
        training
    ).astype(
        np.float32
    )

    state = (
        fit_sers_diversity_constraint_state(
            training_scaled_residuals=(
                scaled
            ),
            configuration=(
                _configuration()
            ),
        )
    )

    module = (
        DifferentiableSersDiversityLoss(
            diversity_constraint_state=(
                state
            ),
            padded_length=(
                axis.size
            ),
        )
    )

    # 简化D2.1状态：
    #
    # prior = 0
    # pointwise_scale = 1
    # standardized_residual_scale = 1
    # asinh_normalizer = 1
    #
    # 因而 scaled=asinh(normalized)
    # 可以恢复到原normalized测试谱。
    module.configure_inverse_reference(
        prior_normalized_intensity=(
            np.zeros(
                axis.size,
                dtype=np.float32,
            )
        ),
        pointwise_scale=(
            np.ones(
                axis.size,
                dtype=np.float32,
            )
        ),
        raman_shift=axis,
        target_abs_max=1.0,
        standardized_residual_scale=1.0,
        asinh_normalizer=1.0,
        numerical_safety_limit=15.0,
    )

    return (
        axis,
        training,
        state,
        module,
    )


def _forward(
    module: DifferentiableSersDiversityLoss,
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[
    str,
    torch.Tensor,
]:
    predicted_tensor = (
        torch.as_tensor(
            np.arcsinh(
                predicted
            ),
            dtype=torch.float32,
        )
        .unsqueeze(
            1
        )
    )

    target_tensor = (
        torch.as_tensor(
            np.arcsinh(
                target
            ),
            dtype=torch.float32,
        )
        .unsqueeze(
            1
        )
    )

    return module(
        predicted_scaled_residual=(
            predicted_tensor
        ),
        target_scaled_residual=(
            target_tensor
        ),
        timesteps=torch.zeros(
            predicted.shape[0],
            dtype=torch.long,
        ),
        alphas_cumprod=torch.linspace(
            0.99,
            0.01,
            100,
        ),
    )


def test_configuration_normalizes_d3_4_2_parameters() -> None:
    config = (
        normalize_diversity_configuration(
            _configuration()
        )
    )

    assert (
        config[
            "method_version"
        ]
        == D3_4_2_METHOD_VERSION
    )

    assert (
        config[
            "peak_morphology"
        ][
            "enabled"
        ]
        is True
    )

    assert (
        config[
            "pointwise_variance_floor"
        ][
            "maximum_std_ratio"
        ]
        == 1.70
    )


def test_state_is_fitted_from_sixteen_training_spectra() -> None:
    _, _, state, _ = (
        _build_module()
    )

    assert (
        state[
            "schema_version"
        ]
        == 1
    )

    assert (
        state[
            "number_of_training_spectra"
        ]
        == 16
    )

    assert (
        state[
            "method_version"
        ]
        == D3_4_2_METHOD_VERSION
    )

    assert (
        state[
            "high_frequency_distance_reference"
        ]
        > 0.0
    )


def test_matching_real_batch_has_near_zero_loss() -> None:
    _, training, _, module = (
        _build_module()
    )

    result = _forward(
        module,
        training[:12],
        training[:12],
    )

    assert (
        result[
            "diversity_raw_loss"
        ].item()
        < 1.0e-5
    )

    assert (
        result[
            "peak_morphology_loss"
        ].item()
        < 1.0e-5
    )


def test_collapsed_predictions_are_penalized_in_hf_and_peak_domains() -> None:
    _, training, _, module = (
        _build_module()
    )

    target = (
        training[:12]
    )

    collapsed = np.repeat(
        np.median(
            target,
            axis=0,
            keepdims=True,
        ),
        target.shape[0],
        axis=0,
    ).astype(
        np.float32
    )

    result = _forward(
        module,
        collapsed,
        target,
    )

    assert (
        result[
            "pairwise_distance_loss"
        ].item()
        > 0.0
    )

    assert (
        result[
            "pointwise_variance_floor_loss"
        ].item()
        > 0.0
    )

    assert (
        result[
            "peak_position_dispersion_loss"
        ].item()
        > 0.0
    )

    assert (
        result[
            "peak_height_dispersion_loss"
        ].item()
        > 0.0
    )

    assert (
        result[
            "peak_width_dispersion_loss"
        ].item()
        > 0.0
    )


def test_apex_only_motion_is_not_accepted_as_coherent_peak_shift() -> None:
    axis, training, _, module = (
        _build_module()
    )

    target = (
        training[:12]
    )

    base = np.median(
        target,
        axis=0,
    )

    abnormal = []

    for index in range(
        target.shape[0]
    ):
        offset = (
            float(
                index % 5
                - 2
            )
            * 2.0
        )

        row = base.copy()

        # 在峰顶附近加入非常窄的局部小峰，
        # 模拟“apex在动而峰体不动”的错误多样性。
        row += _gaussian(
            axis,
            680.0 + offset,
            0.7,
            0.12,
        )

        row += _gaussian(
            axis,
            790.0 - offset,
            0.7,
            0.08,
        )

        abnormal.append(
            row
        )

    abnormal = np.asarray(
        abnormal,
        dtype=np.float32,
    )

    result = _forward(
        module,
        abnormal,
        target,
    )

    assert (
        result[
            "peak_shift_coherence_loss"
        ].item()
        > 0.0
    )

    assert (
        result[
            "peak_position_dispersion_loss"
        ].item()
        > 0.0
    )


def test_excessive_high_frequency_noise_is_penalized() -> None:
    _, training, _, module = (
        _build_module()
    )

    target = (
        training[:12]
    )

    generator = (
        np.random.default_rng(
            2027
        )
    )

    abnormal = (
        target
        + generator.normal(
            0.0,
            0.08,
            size=target.shape,
        ).astype(
            np.float32
        )
    )

    result = _forward(
        module,
        abnormal,
        target,
    )

    assert (
        result[
            "pairwise_distance_loss"
        ].item()
        > 0.0
        or result[
            "pointwise_variance_floor_loss"
        ].item()
        > 0.0
    )


def test_low_noise_gate_skips_insufficient_active_samples() -> None:
    _, training, _, module = (
        _build_module()
    )

    tensor = (
        torch.as_tensor(
            np.arcsinh(
                training[:4]
            ),
            dtype=torch.float32,
        )
        .unsqueeze(
            1
        )
    )

    result = module(
        predicted_scaled_residual=(
            tensor
        ),
        target_scaled_residual=(
            tensor
        ),
        timesteps=torch.tensor(
            [
                0,
                90,
                91,
                92,
            ],
            dtype=torch.long,
        ),
        alphas_cumprod=torch.linspace(
            0.99,
            0.01,
            100,
        ),
    )

    assert (
        result[
            "diversity_active_samples"
        ].item()
        == 0.0
    )

    assert (
        result[
            "diversity_raw_loss"
        ].item()
        == 0.0
    )


def test_complete_d3_4_2_loss_is_finite_and_differentiable() -> None:
    _, training, _, module = (
        _build_module()
    )

    target = (
        training[:8]
    )

    collapsed = np.repeat(
        np.median(
            target,
            axis=0,
            keepdims=True,
        ),
        target.shape[0],
        axis=0,
    ).astype(
        np.float32
    )

    predicted = (
        torch.as_tensor(
            np.arcsinh(
                collapsed
            ),
            dtype=torch.float32,
        )
        .unsqueeze(
            1
        )
        .requires_grad_(
            True
        )
    )

    target_tensor = (
        torch.as_tensor(
            np.arcsinh(
                target
            ),
            dtype=torch.float32,
        )
        .unsqueeze(
            1
        )
    )

    result = module(
        predicted_scaled_residual=(
            predicted
        ),
        target_scaled_residual=(
            target_tensor
        ),
        timesteps=torch.zeros(
            8,
            dtype=torch.long,
        ),
        alphas_cumprod=torch.linspace(
            0.99,
            0.01,
            100,
        ),
    )

    loss = result[
        "diversity_timestep_weighted_loss"
    ]

    loss.backward()

    assert torch.isfinite(
        loss
    )

    assert (
        predicted.grad
        is not None
    )

    assert torch.isfinite(
        predicted.grad
    ).all()

    assert (
        predicted.grad
        .abs()
        .sum()
        .item()
        > 0.0
    )
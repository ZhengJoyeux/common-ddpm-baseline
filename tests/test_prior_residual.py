"""测试D2/D2.1/D2.2/D2.3先验残差范围变换。"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.prior_residual import (
    PriorResidualTransformer,
)


def _build_test_spectra() -> np.ndarray:
    random_generator = (
        np.random.default_rng(
            2026
        )
    )

    spectra = (
        random_generator.normal(
            loc=0.0,
            scale=0.03,
            size=(
                42,
                1901,
            ),
        )
        .astype(
            np.float32
        )
    )

    spectra[
        0,
        100,
    ] = 1.2

    spectra[
        1,
        300,
    ] = -0.9

    return spectra


def _build_heteroscedastic_spectra() -> np.ndarray:
    """构造安静区和真实高变异峰区尺度明显不同的训练谱。"""

    random_generator = (
        np.random.default_rng(
            3107
        )
    )

    number_of_spectra = 42
    length = 64

    x = np.arange(
        length,
        dtype=np.float64,
    )

    prior = (
        0.15
        + 0.35
        * np.exp(
            -0.5
            * (
                (
                    x
                    - 31.0
                )
                / 3.0
            )
            ** 2
        )
    )

    pointwise_sigma = np.full(
        length,
        0.002,
        dtype=np.float64,
    )

    pointwise_sigma[
        27:36
    ] = 0.08

    residuals = (
        random_generator.normal(
            loc=0.0,
            scale=pointwise_sigma,
            size=(
                number_of_spectra,
                length,
            ),
        )
    )

    return (
        prior[
            np.newaxis,
            :
        ]
        + residuals
    ).astype(
        np.float32
    )


def _build_low_frequency_test_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """构造带缓慢背景和窄SERS峰的测试数据。"""

    random_generator = (
        np.random.default_rng(
            6203
        )
    )

    axis = np.linspace(
        600.0,
        1000.0,
        401,
        dtype=np.float64,
    )

    baseline = (
        0.05
        + 0.02
        * np.sin(
            (
                axis
                - 600.0
            )
            / 90.0
        )
        + 0.00005
        * (
            axis
            - 800.0
        )
    )

    spectra = []

    for _ in range(
        32
    ):
        amplitude = (
            0.75
            + 0.08
            * random_generator.normal()
        )

        shift = (
            random_generator.normal(
                scale=1.0
            )
        )

        shifted_peak = np.exp(
            -0.5
            * (
                (
                    axis
                    - (
                        812.0
                        + shift
                    )
                )
                / 5.0
            )
            ** 2
        )

        noise = (
            random_generator.normal(
                scale=0.004,
                size=axis.size,
            )
        )

        spectra.append(
            baseline
            + amplitude
            * shifted_peak
            + noise
        )

    return (
        axis,
        np.asarray(
            spectra,
            dtype=np.float32,
        ),
    )


def test_robust_asinh_is_bounded_and_reversible() -> None:
    spectra = (
        _build_test_spectra()
    )

    transformer = (
        PriorResidualTransformer(
            normalization_method=(
                "robust_asinh"
            ),
            residual_quantile=99.5,
            target_abs_max=1.0,
        )
        .fit(
            spectra
        )
    )

    transformed = (
        transformer.transform(
            spectra
        )
    )

    restored = (
        transformer.inverse_transform(
            transformed
        )
    )

    assert (
        np.max(
            np.abs(
                transformed
            )
        )
        <= 1.0
        + 1.0e-6
    )

    np.testing.assert_allclose(
        restored,
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    assert (
        transformer.residual_scale
        < transformer.training_max_abs_residual
    )


def test_pointwise_mad_asinh_is_bounded_and_reversible() -> None:
    spectra = (
        _build_heteroscedastic_spectra()
    )

    transformer = (
        PriorResidualTransformer(
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            residual_quantile=99.5,
            pointwise_scale_floor_quantile=10.0,
            mad_scale_factor=1.4826,
        )
        .fit(
            spectra
        )
    )

    transformed = (
        transformer.transform(
            spectra
        )
    )

    restored = (
        transformer.inverse_transform(
            transformed
        )
    )

    assert (
        transformed.shape
        == spectra.shape
    )

    assert (
        np.max(
            np.abs(
                transformed
            )
        )
        <= 1.0
        + 1.0e-6
    )

    assert (
        transformer.pointwise_scale.shape
        == (
            spectra.shape[1],
        )
    )

    assert np.all(
        transformer.pointwise_scale
        >= (
            transformer.pointwise_scale_floor
            - transformer.epsilon
        )
    )

    np.testing.assert_allclose(
        restored,
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_pointwise_scale_suppresses_quiet_region_decoding() -> None:
    """同样模型输出在安静区应恢复为更小的光谱残差。"""

    spectra = (
        _build_heteroscedastic_spectra()
    )

    transformer = (
        PriorResidualTransformer(
            normalization_method=(
                "pointwise_mad_asinh"
            ),
        )
        .fit(
            spectra
        )
    )

    quiet_index = 5
    active_peak_index = 31

    assert (
        transformer.pointwise_scale[
            active_peak_index
        ]
        > 10.0
        * transformer.pointwise_scale[
            quiet_index
        ]
    )

    model_output = np.zeros(
        (
            2,
            spectra.shape[1],
        ),
        dtype=np.float32,
    )

    model_output[
        0,
        quiet_index,
    ] = 0.5

    model_output[
        1,
        active_peak_index,
    ] = 0.5

    restored = (
        transformer.inverse_transform(
            model_output
        )
    )

    restored_residuals = (
        restored
        - transformer.prior[
            np.newaxis,
            :
        ]
    )

    quiet_amplitude = abs(
        restored_residuals[
            0,
            quiet_index,
        ]
    )

    active_amplitude = abs(
        restored_residuals[
            1,
            active_peak_index,
        ]
    )

    assert (
        active_amplitude
        > 10.0
        * quiet_amplitude
    )


@pytest.mark.parametrize(
    "method",
    [
        "robust_asinh",
        "pointwise_mad_asinh",
    ],
)
def test_checkpoint_round_trip(
    method: str,
) -> None:
    spectra = (
        _build_heteroscedastic_spectra()
        if method
        == "pointwise_mad_asinh"
        else _build_test_spectra()
    )

    original = (
        PriorResidualTransformer(
            normalization_method=method,
            residual_quantile=99.5,
        )
        .fit(
            spectra
        )
    )

    state = (
        original.state_dict()
    )

    # 旧median/PCA路径继续保持schema v4。
    assert (
        state[
            "schema_version"
        ]
        == 4
    )

    assert (
        state[
            "residual_normalization"
        ][
            "method"
        ]
        == method
    )

    if (
        method
        == "pointwise_mad_asinh"
    ):
        assert len(
            state[
                "residual_normalization"
            ][
                "pointwise_scale"
            ]
        ) == spectra.shape[1]

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored_transformer.transform(
            spectra
        ),
        original.transform(
            spectra
        ),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            restored_transformer.transform(
                spectra
            )
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_version_2_robust_asinh_checkpoint_is_still_supported() -> None:
    spectra = (
        _build_test_spectra()
    )

    original = (
        PriorResidualTransformer(
            normalization_method=(
                "robust_asinh"
            )
        )
        .fit(
            spectra
        )
    )

    state = copy.deepcopy(
        original.state_dict()
    )

    state[
        "schema_version"
    ] = 2

    restored = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored.inverse_transform(
            restored.transform(
                spectra
            )
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_version_1_checkpoint_is_still_supported() -> None:
    spectra = (
        _build_test_spectra()
    )

    original = (
        PriorResidualTransformer(
            normalization_method=(
                "global_maxabs"
            )
        )
        .fit(
            spectra
        )
    )

    state = copy.deepcopy(
        original.state_dict()
    )

    state[
        "schema_version"
    ] = 1

    state[
        "residual_normalization"
    ].pop(
        "training_abs_residual_percentiles"
    )

    restored = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored.inverse_transform(
            restored.transform(
                spectra
            )
        ),
        spectra,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_pointwise_checkpoint_rejects_wrong_scale_length() -> None:
    spectra = (
        _build_heteroscedastic_spectra()
    )

    transformer = (
        PriorResidualTransformer(
            normalization_method=(
                "pointwise_mad_asinh"
            )
        )
        .fit(
            spectra
        )
    )

    state = copy.deepcopy(
        transformer.state_dict()
    )

    state[
        "residual_normalization"
    ][
        "pointwise_scale"
    ] = (
        state[
            "residual_normalization"
        ][
            "pointwise_scale"
        ][
            :-1
        ]
    )

    with pytest.raises(
        ValueError,
        match="长度与先验不一致",
    ):
        PriorResidualTransformer.from_state_dict(
            state
        )


def test_pca_variable_prior_is_reversible_and_samples_new_priors() -> None:
    spectra = (
        _build_heteroscedastic_spectra()
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "pca_reconstruction"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            pca_explained_variance_ratio=0.90,
            pca_max_components=6,
        )
        .fit(
            spectra
        )
    )

    reference_priors = (
        transformer.reference_priors_for_spectra(
            spectra
        )
    )

    transformed = (
        transformer.transform(
            spectra,
            reference_priors=(
                reference_priors
            ),
        )
    )

    restored = (
        transformer.inverse_transform(
            transformed,
            reference_priors=(
                reference_priors
            ),
        )
    )

    assert (
        transformer.pca_components.shape[0]
        <= 6
    )

    assert (
        reference_priors.shape
        == spectra.shape
    )

    np.testing.assert_allclose(
        restored,
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    with pytest.raises(
        ValueError,
        match="必须显式提供",
    ):
        transformer.inverse_transform(
            transformed
        )

    sampled = (
        transformer.sample_reference_priors(
            5,
            random_generator=(
                np.random.default_rng(
                    2026
                )
            ),
        )
    )

    assert (
        sampled.shape
        == (
            5,
            spectra.shape[1],
        )
    )

    assert np.isfinite(
        sampled
    ).all()

    state = (
        transformer.state_dict()
    )

    assert (
        state[
            "schema_version"
        ]
        == 4
    )

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            transformed,
            reference_priors=(
                reference_priors
            ),
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_low_frequency_prior_is_smoother_and_suppresses_narrow_peak() -> None:
    axis, spectra = (
        _build_low_frequency_test_data()
    )

    full_median = np.median(
        spectra,
        axis=0,
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            low_frequency_sigma_cm1=40.0,
            low_frequency_truncate=4.0,
        )
        .fit(
            spectra,
            raman_shift=axis,
        )
    )

    low_frequency_prior = (
        transformer.prior
    )

    full_curvature = np.mean(
        np.abs(
            np.diff(
                full_median,
                n=2,
            )
        )
    )

    low_frequency_curvature = np.mean(
        np.abs(
            np.diff(
                low_frequency_prior,
                n=2,
            )
        )
    )

    assert (
        low_frequency_curvature
        < 0.20
        * full_curvature
    )

    peak_index = int(
        np.argmin(
            np.abs(
                axis
                - 812.0
            )
        )
    )

    left_index = int(
        np.argmin(
            np.abs(
                axis
                - 760.0
            )
        )
    )

    right_index = int(
        np.argmin(
            np.abs(
                axis
                - 864.0
            )
        )
    )

    full_local_baseline = (
        0.5
        * (
            full_median[
                left_index
            ]
            + full_median[
                right_index
            ]
        )
    )

    low_frequency_local_baseline = (
        0.5
        * (
            low_frequency_prior[
                left_index
            ]
            + low_frequency_prior[
                right_index
            ]
        )
    )

    full_peak_height = (
        full_median[
            peak_index
        ]
        - full_local_baseline
    )

    low_frequency_peak_height = (
        low_frequency_prior[
            peak_index
        ]
        - low_frequency_local_baseline
    )

    assert (
        low_frequency_peak_height
        < 0.25
        * full_peak_height
    )


def test_low_frequency_prior_round_trip_uses_v5_checkpoint() -> None:
    axis, spectra = (
        _build_low_frequency_test_data()
    )

    original = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            residual_quantile=99.5,
            low_frequency_sigma_cm1=40.0,
        )
        .fit(
            spectra,
            raman_shift=axis,
        )
    )

    transformed = (
        original.transform(
            spectra
        )
    )

    restored = (
        original.inverse_transform(
            transformed
        )
    )

    np.testing.assert_allclose(
        restored,
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    state = (
        original.state_dict()
    )

    assert (
        state[
            "schema_version"
        ]
        == 5
    )

    assert (
        state[
            "prior_method"
        ]
        == "training_low_frequency_median"
    )

    assert (
        state[
            "low_frequency_prior"
        ][
            "method"
        ]
        == "gaussian"
    )

    assert (
        state[
            "low_frequency_prior"
        ][
            "sigma_cm1"
        ]
        == 40.0
    )

    loaded = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        loaded.prior,
        original.prior,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        loaded.transform(
            spectra
        ),
        transformed,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    sampled = (
        loaded.sample_reference_priors(
            4,
            random_generator=(
                np.random.default_rng(
                    2026
                )
            ),
        )
    )

    expected = np.repeat(
        loaded.prior[
            None,
            :
        ],
        4,
        axis=0,
    )

    np.testing.assert_allclose(
        sampled,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_low_frequency_prior_supports_nonuniform_raman_axis() -> None:
    axis, spectra = (
        _build_low_frequency_test_data()
    )

    nonuniform_axis = (
        axis.copy()
    )

    nonuniform_axis[
        1:-1
    ] += (
        0.15
        * np.sin(
            np.linspace(
                0.0,
                4.0
                * np.pi,
                axis.size - 2,
            )
        )
    )

    assert np.all(
        np.diff(
            nonuniform_axis
        )
        > 0.0
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            low_frequency_sigma_cm1=35.0,
        )
        .fit(
            spectra,
            raman_shift=(
                nonuniform_axis
            ),
        )
    )

    assert (
        transformer.prior.shape
        == (
            spectra.shape[1],
        )
    )

    assert np.isfinite(
        transformer.prior
    ).all()

    expected_spacing = (
        (
            nonuniform_axis[-1]
            - nonuniform_axis[0]
        )
        / (
            nonuniform_axis.size
            - 1
        )
    )

    assert np.isclose(
        transformer.low_frequency_uniform_spacing_cm1,
        expected_spacing,
    )


def test_low_frequency_prior_requires_matching_raman_axis() -> None:
    axis, spectra = (
        _build_low_frequency_test_data()
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="必须提供统一训练raman_shift",
    ):
        transformer.fit(
            spectra
        )

    with pytest.raises(
        ValueError,
        match="长度与训练光谱不一致",
    ):
        transformer.fit(
            spectra,
            raman_shift=(
                axis[:-1]
            ),
        )


def test_low_frequency_sigma_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="low_frequency_sigma_cm1",
    ):
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            low_frequency_sigma_cm1=0.0,
        )


def test_blended_frequency_prior_formula_and_v6_round_trip() -> None:
    """D2.4状态必须继续兼容，避免D2.5修改破坏旧checkpoint。"""

    axis, spectra = _build_low_frequency_test_data()

    low_frequency_transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method="robust_asinh",
            low_frequency_sigma_cm1=40.0,
            low_frequency_truncate=4.0,
        )
        .fit(
            spectra,
            raman_shift=axis,
        )
    )

    blended_transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_blended_frequency_median"
            ),
            normalization_method="robust_asinh",
            low_frequency_sigma_cm1=40.0,
            low_frequency_truncate=4.0,
            blended_peak_component_ratio=0.5,
        )
        .fit(
            spectra,
            raman_shift=axis,
        )
    )

    full_median = np.median(
        spectra,
        axis=0,
    )

    expected_prior = (
        low_frequency_transformer.prior
        + 0.5
        * (
            full_median
            - low_frequency_transformer.prior
        )
    )

    np.testing.assert_allclose(
        blended_transformer.prior,
        expected_prior,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    transformed = blended_transformer.transform(
        spectra
    )

    state = blended_transformer.state_dict()

    assert state["schema_version"] == 6

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            transformed
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_pca3_robust_asinh_checkpoint_and_seeded_sampling() -> None:
    """验证D2.5 PCA3、robust_asinh和固定种子采样的完整状态。"""

    random_generator = np.random.default_rng(
        8125
    )

    spectra = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(16, 64),
    ).astype(np.float32)

    transformer = (
        PriorResidualTransformer(
            prior_method="pca_reconstruction",
            normalization_method="robust_asinh",
            residual_quantile=99.5,
            target_abs_max=1.0,
            pca_explained_variance_ratio=0.95,
            pca_max_components=3,
            pca_sampling_strategy=(
                "independent_truncated_gaussian_scores"
            ),
            pca_score_clip_standard_deviations=1.5,
        )
        .fit(spectra)
    )

    assert transformer.pca_components.shape == (
        3,
        spectra.shape[1],
    )

    reference_priors = (
        transformer.reference_priors_for_spectra(
            spectra
        )
    )

    transformed = transformer.transform(
        spectra,
        reference_priors=reference_priors,
    )

    state = transformer.state_dict()

    assert state["schema_version"] == 4
    assert state["pca_prior"]["max_components"] == 3
    assert (
        state["pca_prior"]["sampling_strategy"]
        == "independent_truncated_gaussian_scores"
    )
    assert (
        state["pca_prior"]
        ["score_clip_standard_deviations"]
        == 1.5
    )

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            transformed,
            reference_priors=reference_priors,
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    sampled_a = (
        restored_transformer.sample_reference_priors(
            32,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    sampled_b = (
        restored_transformer.sample_reference_priors(
            32,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    sampled_c = (
        restored_transformer.sample_reference_priors(
            32,
            random_generator=np.random.default_rng(
                2027
            ),
        )
    )

    np.testing.assert_array_equal(
        sampled_a,
        sampled_b,
    )

    assert not np.array_equal(
        sampled_a,
        sampled_c,
    )

    sampled_scores = (
        (
            sampled_a
            - restored_transformer.pca_mean[
                np.newaxis,
                :,
            ]
        )
        @ restored_transformer.pca_components.T
    )

    score_distance = np.abs(
        sampled_scores
        - restored_transformer.pca_training_score_mean[
            np.newaxis,
            :,
        ]
    )

    score_limit = (
        1.5
        * restored_transformer.pca_score_standard_deviation[
            np.newaxis,
            :,
        ]
    )

    assert np.all(
        score_distance
        <= score_limit + 1.0e-6
    )

    # 真正截断采样不应像np.clip那样把大量score精确堆在边界。
    assert not np.any(
        np.isclose(
            score_distance,
            score_limit,
            rtol=1.0e-5,
            atol=1.0e-7,
        )
    )
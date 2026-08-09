"""测试D2/D2.1先验残差范围变换。"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.prior_residual import PriorResidualTransformer


def _build_test_spectra() -> np.ndarray:
    random_generator = np.random.default_rng(2026)
    spectra = random_generator.normal(
        loc=0.0,
        scale=0.03,
        size=(42, 1901),
    ).astype(np.float32)

    # 模拟少量远大于正常残差的极端点。
    spectra[0, 100] = 1.2
    spectra[1, 300] = -0.9
    return spectra


def _build_heteroscedastic_spectra() -> np.ndarray:
    """构造安静区和真实高变异峰区尺度明显不同的训练谱。"""

    random_generator = np.random.default_rng(3107)
    number_of_spectra = 42
    length = 64
    x = np.arange(length, dtype=np.float64)

    prior = (
        0.15
        + 0.35 * np.exp(-0.5 * ((x - 31.0) / 3.0) ** 2)
    )
    pointwise_sigma = np.full(length, 0.002, dtype=np.float64)
    pointwise_sigma[27:36] = 0.08

    residuals = random_generator.normal(
        loc=0.0,
        scale=pointwise_sigma,
        size=(number_of_spectra, length),
    )
    return (prior[np.newaxis, :] + residuals).astype(np.float32)


def test_robust_asinh_is_bounded_and_reversible() -> None:
    spectra = _build_test_spectra()
    transformer = PriorResidualTransformer(
        normalization_method="robust_asinh",
        residual_quantile=99.5,
        target_abs_max=1.0,
    ).fit(spectra)

    transformed = transformer.transform(spectra)
    restored = transformer.inverse_transform(transformed)

    assert np.max(np.abs(transformed)) <= 1.0 + 1.0e-6
    np.testing.assert_allclose(
        restored,
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert transformer.residual_scale < transformer.training_max_abs_residual


def test_pointwise_mad_asinh_is_bounded_and_reversible() -> None:
    spectra = _build_heteroscedastic_spectra()
    transformer = PriorResidualTransformer(
        normalization_method="pointwise_mad_asinh",
        residual_quantile=99.5,
        pointwise_scale_floor_quantile=10.0,
        mad_scale_factor=1.4826,
    ).fit(spectra)

    transformed = transformer.transform(spectra)
    restored = transformer.inverse_transform(transformed)

    assert transformed.shape == spectra.shape
    assert np.max(np.abs(transformed)) <= 1.0 + 1.0e-6
    assert transformer.pointwise_scale.shape == (spectra.shape[1],)
    assert np.all(
        transformer.pointwise_scale
        >= transformer.pointwise_scale_floor - transformer.epsilon
    )
    np.testing.assert_allclose(
        restored,
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_pointwise_scale_suppresses_quiet_region_decoding() -> None:
    """同样模型输出在安静区应恢复为更小的光谱残差。"""

    spectra = _build_heteroscedastic_spectra()
    transformer = PriorResidualTransformer(
        normalization_method="pointwise_mad_asinh",
    ).fit(spectra)

    quiet_index = 5
    active_peak_index = 31
    assert (
        transformer.pointwise_scale[active_peak_index]
        > 10.0 * transformer.pointwise_scale[quiet_index]
    )

    model_output = np.zeros((2, spectra.shape[1]), dtype=np.float32)
    model_output[0, quiet_index] = 0.5
    model_output[1, active_peak_index] = 0.5

    restored = transformer.inverse_transform(model_output)
    restored_residuals = restored - transformer.prior[np.newaxis, :]

    quiet_amplitude = abs(restored_residuals[0, quiet_index])
    active_amplitude = abs(restored_residuals[1, active_peak_index])
    assert active_amplitude > 10.0 * quiet_amplitude


@pytest.mark.parametrize(
    "method",
    ["robust_asinh", "pointwise_mad_asinh"],
)
def test_checkpoint_round_trip(method: str) -> None:
    spectra = (
        _build_heteroscedastic_spectra()
        if method == "pointwise_mad_asinh"
        else _build_test_spectra()
    )
    original = PriorResidualTransformer(
        normalization_method=method,
        residual_quantile=99.5,
    ).fit(spectra)

    state = original.state_dict()
    assert state["schema_version"] == 4
    assert state["residual_normalization"]["method"] == method

    if method == "pointwise_mad_asinh":
        assert len(
            state["residual_normalization"]["pointwise_scale"]
        ) == spectra.shape[1]

    restored_transformer = PriorResidualTransformer.from_state_dict(state)
    np.testing.assert_allclose(
        restored_transformer.transform(spectra),
        original.transform(spectra),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            restored_transformer.transform(spectra)
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_version_2_robust_asinh_checkpoint_is_still_supported() -> None:
    spectra = _build_test_spectra()
    original = PriorResidualTransformer(
        normalization_method="robust_asinh"
    ).fit(spectra)
    state = copy.deepcopy(original.state_dict())
    state["schema_version"] = 2

    restored = PriorResidualTransformer.from_state_dict(state)
    np.testing.assert_allclose(
        restored.inverse_transform(restored.transform(spectra)),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_version_1_checkpoint_is_still_supported() -> None:
    spectra = _build_test_spectra()
    original = PriorResidualTransformer(
        normalization_method="global_maxabs"
    ).fit(spectra)
    state = copy.deepcopy(original.state_dict())

    # 模拟D2最早期schema_version=1 checkpoint。
    state["schema_version"] = 1
    state["residual_normalization"].pop(
        "training_abs_residual_percentiles"
    )

    restored = PriorResidualTransformer.from_state_dict(state)
    np.testing.assert_allclose(
        restored.inverse_transform(restored.transform(spectra)),
        spectra,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_pointwise_checkpoint_rejects_wrong_scale_length() -> None:
    spectra = _build_heteroscedastic_spectra()
    transformer = PriorResidualTransformer(
        normalization_method="pointwise_mad_asinh"
    ).fit(spectra)
    state = copy.deepcopy(transformer.state_dict())
    state["residual_normalization"]["pointwise_scale"] = (
        state["residual_normalization"]["pointwise_scale"][:-1]
    )

    with pytest.raises(ValueError, match="长度与先验不一致"):
        PriorResidualTransformer.from_state_dict(state)


def test_pca_variable_prior_is_reversible_and_samples_new_priors() -> None:
    spectra = _build_heteroscedastic_spectra()
    transformer = PriorResidualTransformer(
        prior_method="pca_reconstruction",
        normalization_method="pointwise_mad_asinh",
        pca_explained_variance_ratio=0.90,
        pca_max_components=6,
    ).fit(spectra)

    reference_priors = transformer.reference_priors_for_spectra(spectra)
    transformed = transformer.transform(
        spectra,
        reference_priors=reference_priors,
    )
    restored = transformer.inverse_transform(
        transformed,
        reference_priors=reference_priors,
    )

    assert transformer.pca_components.shape[0] <= 6
    assert reference_priors.shape == spectra.shape
    np.testing.assert_allclose(restored, spectra, rtol=2.0e-5, atol=2.0e-6)

    with pytest.raises(ValueError, match="必须显式提供"):
        transformer.inverse_transform(transformed)

    sampled = transformer.sample_reference_priors(
        5,
        random_generator=np.random.default_rng(2026),
    )
    assert sampled.shape == (5, spectra.shape[1])
    assert np.isfinite(sampled).all()

    state = transformer.state_dict()
    assert state["schema_version"] == 4
    restored_transformer = PriorResidualTransformer.from_state_dict(state)
    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            transformed,
            reference_priors=reference_priors,
        ),
        spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
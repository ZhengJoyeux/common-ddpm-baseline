"""测试D2先验残差范围变换。"""

from __future__ import annotations

import numpy as np

from src.prior_residual import (
    PriorResidualTransformer,
)


def _build_test_spectra() -> np.ndarray:
    random_generator = (
        np.random.default_rng(
            2026
        )
    )

    spectra = random_generator.normal(
        loc=0.0,
        scale=0.03,
        size=(
            42,
            1901,
        ),
    ).astype(
        np.float32
    )

    # 模拟少量远大于正常残差的极端点。
    spectra[0, 100] = 1.2
    spectra[1, 300] = -0.9

    return spectra


def test_robust_asinh_is_bounded_and_reversible() -> None:
    spectra = _build_test_spectra()

    transformer = (
        PriorResidualTransformer(
            normalization_method=(
                "robust_asinh"
            ),
            residual_quantile=99.5,
            target_abs_max=1.0,
        )
    )

    transformer.fit(
        spectra
    )

    transformed = transformer.transform(
        spectra
    )

    restored = transformer.inverse_transform(
        transformed
    )

    assert (
        np.max(
            np.abs(
                transformed
            )
        )
        <= 1.0 + 1.0e-6
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


def test_robust_asinh_checkpoint_round_trip() -> None:
    spectra = _build_test_spectra()

    original = (
        PriorResidualTransformer(
            normalization_method=(
                "robust_asinh"
            ),
            residual_quantile=99.5,
        )
    )

    original.fit(
        spectra
    )

    state = original.state_dict()

    assert (
        state["schema_version"]
        == 2
    )

    assert (
        state[
            "residual_normalization"
        ]["method"]
        == "robust_asinh"
    )

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


def test_version_1_checkpoint_is_still_supported() -> None:
    spectra = _build_test_spectra()

    original = (
        PriorResidualTransformer(
            normalization_method=(
                "global_maxabs"
            )
        )
    )

    original.fit(
        spectra
    )

    state = original.state_dict()

    # 模拟旧版schema_version=1 checkpoint。
    state["schema_version"] = 1

    state[
        "residual_normalization"
    ].pop(
        "training_abs_residual_percentiles"
    )

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    np.testing.assert_allclose(
        restored_transformer.inverse_transform(
            restored_transformer.transform(
                spectra
            )
        ),
        spectra,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
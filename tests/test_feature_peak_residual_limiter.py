import numpy as np

from src.feature_peak_residual_limiter import (
    FeaturePeakResidualLimiter,
    fit_feature_peak_residual_limiter_state,
    normalize_feature_peak_residual_limiter_configuration,
)


def _make_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    axis = np.arange(
        600.0,
        1001.0,
        1.0,
    )

    prior = (
        0.05 * np.sin(axis / 35.0)
        + 0.90
        * np.exp(
            -0.5
            * ((axis - 700.0) / 7.0) ** 2
        )
        + 0.65
        * np.exp(
            -0.5
            * ((axis - 870.0) / 10.0) ** 2
        )
    )

    generator = np.random.default_rng(
        2026
    )

    spectra = []

    for amplitude in (
        -0.05,
        -0.02,
        0.01,
        0.04,
        0.07,
    ):
        spectra.append(
            prior
            + amplitude
            * np.exp(
                -0.5
                * ((axis - 700.0) / 7.0) ** 2
            )
            + generator.normal(
                0.0,
                0.01,
                size=axis.size,
            )
        )

    return (
        axis,
        prior,
        np.asarray(
            spectra,
            dtype=np.float32,
        ),
    )


def test_limiter_only_changes_feature_peak_windows_and_soft_compresses_extreme_residuals():
    axis, prior, training_spectra = (
        _make_training_data()
    )

    state = (
        fit_feature_peak_residual_limiter_state(
            training_normalized_spectra=(
                training_spectra
            ),
            prior_normalized_intensity=prior,
            raman_shift=axis,
            configuration={
                "enabled": True,
                "limit_multiplier": 1.20,
                "maximum_peak_count": 4,
            },
        )
    )

    limiter = (
        FeaturePeakResidualLimiter.from_state_dict(
            state
        )
    )

    mask = np.asarray(
        state["feature_peak_mask"],
        dtype=np.uint8,
    ).astype(bool)

    generated = np.repeat(
        prior[np.newaxis, :],
        2,
        axis=0,
    )

    generated[0, mask] += 10.0
    generated[1, mask] -= 10.0

    generated[:, ~mask] += 0.123

    limited = (
        limiter.apply_to_normalized_spectra(
            generated
        )
    )

    residuals = (
        limited
        - prior[np.newaxis, :]
    )

    # 非特征峰区域不能被 D3.5 修改。
    np.testing.assert_allclose(
        residuals[:, ~mask],
        0.123,
        rtol=0.0,
        atol=1.0e-6,
    )

    # 极端残差应该被明显压缩，
    # 但 D3.5.1 不再要求压缩后必须严格小于
    # 原始 pointwise limit。
    assert np.all(
        np.abs(
            residuals[:, mask]
        )
        < 10.0
    )


def test_limiter_preserves_residuals_inside_pointwise_limits():
    axis, prior, training_spectra = (
        _make_training_data()
    )

    state = (
        fit_feature_peak_residual_limiter_state(
            training_normalized_spectra=(
                training_spectra
            ),
            prior_normalized_intensity=prior,
            raman_shift=axis,
            configuration={
                "enabled": True,
                "limit_multiplier": 1.20,
                "maximum_peak_count": 4,
            },
        )
    )

    limiter = (
        FeaturePeakResidualLimiter.from_state_dict(
            state
        )
    )

    mask = np.asarray(
        state["feature_peak_mask"],
        dtype=np.uint8,
    ).astype(bool)

    limits = np.asarray(
        state["pointwise_soft_limit"],
        dtype=np.float64,
    )

    generated = (
        prior[np.newaxis, :].copy()
    )

    # 构造一个完全处在 D3.5 逐点上限以内的
    # 合理峰区残差。
    #
    # D3.5.1 的关键要求：
    # 这种正常差异必须原样保留，
    # 不能再向训练集先验收缩。
    generated[:, mask] += (
        0.50 * limits[mask]
    )

    limited = (
        limiter.apply_to_normalized_spectra(
            generated
        )
    )

    np.testing.assert_allclose(
        limited,
        generated,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_limiter_preserves_order_and_separation_of_large_peak_residuals():
    axis, prior, training_spectra = (
        _make_training_data()
    )

    state = (
        fit_feature_peak_residual_limiter_state(
            training_normalized_spectra=(
                training_spectra
            ),
            prior_normalized_intensity=prior,
            raman_shift=axis,
            configuration={
                "enabled": True,
                "limit_multiplier": 1.20,
                "maximum_peak_count": 4,
            },
        )
    )

    limiter = (
        FeaturePeakResidualLimiter.from_state_dict(
            state
        )
    )

    mask = np.asarray(
        state["feature_peak_mask"],
        dtype=np.uint8,
    ).astype(bool)

    limits = np.asarray(
        state["pointwise_soft_limit"],
        dtype=np.float64,
    )

    generated = np.repeat(
        prior[np.newaxis, :],
        2,
        axis=0,
    )

    # 两条生成谱都明显超过训练集拟合的正常
    # 峰区残差上限，但第二条明显强于第一条。
    generated[0, mask] += (
        2.0 * limits[mask]
    )

    generated[1, mask] += (
        4.0 * limits[mask]
    )

    limited = (
        limiter.apply_to_normalized_spectra(
            generated
        )
    )

    residuals = (
        limited
        - prior[np.newaxis, :]
    )

    smaller = residuals[
        0,
        mask,
    ]

    larger = residuals[
        1,
        mask,
    ]

    # 即使发生了软压缩，
    # 原始较强的光谱仍然必须保持更强。
    assert np.all(
        larger > smaller
    )

    input_gap = (
        2.0 * limits[mask]
    )

    output_gap = (
        larger - smaller
    )

    # 不能像完全饱和的 tanh 上限一样，
    # 把两个明显不同的超限峰压成几乎相同。
    #
    # 当前要求至少保留输入峰强差异的 10%。
    assert np.all(
        output_gap
        >= 0.10 * input_gap
    )


def test_disabled_limiter_configuration_needs_no_training_statistics():
    config = (
        normalize_feature_peak_residual_limiter_configuration(
            {
                "enabled": False,
            }
        )
    )

    assert config == {
        "enabled": False,
    }
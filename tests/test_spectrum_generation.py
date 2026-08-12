import numpy as np
import torch
from torch import nn

from src.model_builder import (
    build_diffusion_model,
)
from src.prior_residual import (
    PriorResidualTransformer,
)
from src.spectrum_generator import (
    generate_spectra,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)
from src.sers_sampling_calibrator import (
    SersSamplingCalibrator,
)


class FixedSampleDiffusion(nn.Module):
    """
    测试专用扩散模型。

    不执行随机采样，而是按照顺序返回预先设置好的
    缩放残差，用于准确验证D2生成恢复流程。
    """

    def __init__(
        self,
        generated_samples: np.ndarray,
    ) -> None:
        super().__init__()

        samples = torch.as_tensor(
            generated_samples,
            dtype=torch.float32,
        )

        if samples.ndim != 2:
            raise ValueError(
                "generated_samples必须是二维数组[N,L]。"
            )

        self.register_buffer(
            "generated_samples",
            samples[
                :,
                None,
                :,
            ],
        )

        self.sample_offset = 0

    def sample(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """返回指定数量的固定生成结果。"""

        batch_size = int(
            batch_size
        )

        start = (
            self.sample_offset
        )

        end = (
            start
            + batch_size
        )

        if (
            end
            > self.generated_samples.shape[0]
        ):
            raise RuntimeError(
                "测试扩散模型中的固定样本数量不足。"
            )

        batch = (
            self.generated_samples[
                start:end
            ]
        )

        self.sample_offset = (
            end
        )

        return batch


def test_spectrum_generation_shape():
    """验证普通DDPM能够生成正确形状的张量。"""

    model_configuration = {
        "channels": 1,
        "base_dimension": 8,
        "dimension_multipliers": [
            1,
            2,
        ],
        "self_condition": False,
        "dropout": 0.0,
        "diffusion_timesteps": 10,
        "sampling_timesteps": 5,
        "objective": "pred_noise",
        "beta_schedule": "cosine",
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
    }

    (
        _,
        diffusion,
    ) = build_diffusion_model(
        model_configuration=(
            model_configuration
        ),
        sequence_length=16,
    )

    with torch.inference_mode():
        generated = diffusion.sample(
            batch_size=2
        )

    assert (
        generated.shape
        == (
            2,
            1,
            16,
        )
    )


def test_d2_generation_restores_prior_residual():
    """
    验证D2生成端能够：

    1. 删除U-Net末尾补齐点；
    2. 恢复checkpoint中的先验残差变换器；
    3. 取消残差缩放；
    4. 加回训练集逐点中位数先验；
    5. 插值恢复到目标拉曼位移轴。
    """

    model_axis = np.asarray(
        [
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
        ],
        dtype=np.float64,
    )

    training_spectra = np.asarray(
        [
            [
                -0.8,
                -0.4,
                0.0,
                0.4,
                0.8,
            ],
            [
                -0.6,
                -0.2,
                0.2,
                0.6,
                0.6,
            ],
            [
                -0.4,
                0.0,
                0.4,
                0.8,
                0.4,
            ],
        ],
        dtype=np.float32,
    )

    expected_model_axis_spectra = np.asarray(
        [
            [
                -0.55,
                -0.10,
                0.25,
                0.70,
                0.50,
            ],
            [
                -0.65,
                -0.30,
                0.10,
                0.50,
                0.70,
            ],
        ],
        dtype=np.float32,
    )

    fitted_transformer = (
        PriorResidualTransformer(
            target_abs_max=0.8,
        )
        .fit(
            training_spectra
        )
    )

    prior_residual_state = (
        fitted_transformer.state_dict()
    )

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            prior_residual_state
        )
    )

    scaled_residuals = (
        restored_transformer.transform(
            expected_model_axis_spectra
        )
    )

    length_adapter = (
        SpectrumLengthAdapter.create(
            dimension_multipliers=[
                1,
                2,
                4,
            ],
            raman_shifts=[
                model_axis
            ],
            model_length="auto",
            padding_mode=(
                "right_zero_padding"
            ),
            padding_value=0.0,
            raman_range_tolerance=1.0,
        )
    )

    assert (
        length_adapter.original_length
        == 5
    )

    assert (
        length_adapter.padded_length
        == 8
    )

    assert (
        length_adapter.padding_size
        == 3
    )

    padded_scaled_residuals = (
        length_adapter.adapt(
            scaled_residuals
        )
    )

    assert (
        padded_scaled_residuals.shape
        == (
            2,
            8,
        )
    )

    fixed_diffusion = (
        FixedSampleDiffusion(
            padded_scaled_residuals
        )
    )

    output_axis = np.asarray(
        [
            100.0,
            300.0,
            500.0,
        ],
        dtype=np.float64,
    )

    generated = generate_spectra(
        diffusion=(
            fixed_diffusion
        ),
        number_of_spectra=2,
        generation_batch_size=1,
        device=torch.device(
            "cpu"
        ),
        length_adapter=(
            length_adapter
        ),
        output_raman_shifts=(
            output_axis
        ),
        prior_residual_transformer=(
            restored_transformer
        ),
    )

    expected_output_spectra = (
        expected_model_axis_spectra[
            :,
            [
                0,
                2,
                4,
            ],
        ]
    )

    assert (
        generated.shape
        == (
            2,
            3,
        )
    )

    np.testing.assert_allclose(
        generated,
        expected_output_spectra,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_2_generation_uses_sampled_pca_priors():
    model_axis = np.arange(
        5,
        dtype=np.float64,
    )

    training_spectra = np.asarray(
        [
            [
                -0.8,
                -0.4,
                0.0,
                0.4,
                0.8,
            ],
            [
                -0.5,
                -0.2,
                0.2,
                0.5,
                0.7,
            ],
            [
                -0.3,
                0.1,
                0.4,
                0.7,
                0.6,
            ],
            [
                -0.6,
                -0.1,
                0.1,
                0.6,
                0.5,
            ],
        ],
        dtype=np.float32,
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
            pca_max_components=2,
        )
        .fit(
            training_spectra
        )
    )

    adapter = (
        SpectrumLengthAdapter.create(
            dimension_multipliers=[
                1,
                2,
                4,
            ],
            raman_shifts=[
                model_axis
            ],
            model_length="auto",
            padding_mode=(
                "right_zero_padding"
            ),
            padding_value=0.0,
            raman_range_tolerance=1.0,
        )
    )

    fixed_diffusion = (
        FixedSampleDiffusion(
            np.zeros(
                (
                    3,
                    adapter.padded_length,
                ),
                dtype=np.float32,
            )
        )
    )

    expected = (
        transformer.sample_reference_priors(
            3,
            random_generator=(
                np.random.default_rng(
                    2026
                )
            ),
        )
    )

    generated = generate_spectra(
        diffusion=(
            fixed_diffusion
        ),
        number_of_spectra=3,
        generation_batch_size=2,
        device=torch.device(
            "cpu"
        ),
        length_adapter=(
            adapter
        ),
        prior_residual_transformer=(
            transformer
        ),
        prior_random_seed=2026,
    )

    np.testing.assert_allclose(
        generated,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_3_generation_restores_low_frequency_prior():
    """验证D2.3 fixed low-frequency prior完整生成恢复链。"""

    model_axis = np.asarray(
        [
            100.0,
            170.0,
            300.0,
            410.0,
            500.0,
        ],
        dtype=np.float64,
    )

    # 刻意使用非严格等间距Raman轴，
    # 验证D2.3以真实cm^-1轴构造低频prior。
    training_spectra = np.asarray(
        [
            [
                -0.75,
                -0.42,
                0.10,
                0.45,
                0.70,
            ],
            [
                -0.65,
                -0.30,
                0.22,
                0.58,
                0.62,
            ],
            [
                -0.55,
                -0.18,
                0.35,
                0.68,
                0.54,
            ],
            [
                -0.72,
                -0.24,
                0.18,
                0.52,
                0.66,
            ],
            [
                -0.60,
                -0.36,
                0.30,
                0.63,
                0.58,
            ],
        ],
        dtype=np.float32,
    )

    transformer = (
        PriorResidualTransformer(
            prior_method=(
                "training_low_frequency_median"
            ),
            normalization_method=(
                "pointwise_mad_asinh"
            ),
            low_frequency_sigma_cm1=90.0,
            low_frequency_truncate=4.0,
        )
        .fit(
            training_spectra,
            raman_shift=(
                model_axis
            ),
        )
    )

    state = (
        transformer.state_dict()
    )

    assert (
        state[
            "schema_version"
        ]
        == 5
    )

    restored_transformer = (
        PriorResidualTransformer
        .from_state_dict(
            state
        )
    )

    expected_model_axis_spectra = np.asarray(
        [
            [
                -0.67,
                -0.28,
                0.26,
                0.56,
                0.61,
            ],
            [
                -0.58,
                -0.34,
                0.16,
                0.64,
                0.57,
            ],
        ],
        dtype=np.float32,
    )

    scaled_residuals = (
        restored_transformer.transform(
            expected_model_axis_spectra
        )
    )

    adapter = (
        SpectrumLengthAdapter.create(
            dimension_multipliers=[
                1,
                2,
                4,
            ],
            raman_shifts=[
                model_axis
            ],
            model_length="auto",
            padding_mode=(
                "right_zero_padding"
            ),
            padding_value=0.0,
            raman_range_tolerance=1.0,
        )
    )

    padded_scaled_residuals = (
        adapter.adapt(
            scaled_residuals
        )
    )

    fixed_diffusion = (
        FixedSampleDiffusion(
            padded_scaled_residuals
        )
    )

    generated = generate_spectra(
        diffusion=(
            fixed_diffusion
        ),
        number_of_spectra=2,
        generation_batch_size=1,
        device=torch.device(
            "cpu"
        ),
        length_adapter=(
            adapter
        ),
        output_raman_shifts=(
            model_axis
        ),
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
    )

    assert (
        generated.shape
        == expected_model_axis_spectra.shape
    )

    np.testing.assert_allclose(
        generated,
        expected_model_axis_spectra,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_d2_5_pca3_robust_asinh_generation_is_reproducible():
    """验证D2.5 checkpoint恢复、PCA prior采样和生成反变换。"""

    model_axis = np.arange(
        16,
        dtype=np.float64,
    )

    random_generator = np.random.default_rng(
        9157
    )

    training_spectra = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(16, model_axis.size),
    ).astype(np.float32)

    fitted_transformer = (
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
        .fit(training_spectra)
    )

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            fitted_transformer.state_dict()
        )
    )

    adapter = SpectrumLengthAdapter.create(
        dimension_multipliers=[
            1,
            2,
            4,
        ],
        raman_shifts=[
            model_axis
        ],
        model_length="auto",
        padding_mode="right_zero_padding",
        padding_value=0.0,
        raman_range_tolerance=1.0,
    )

    zero_scaled_residuals = np.zeros(
        (
            7,
            adapter.padded_length,
        ),
        dtype=np.float32,
    )

    expected_priors = (
        restored_transformer.sample_reference_priors(
            7,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    generated_batch_3 = generate_spectra(
        diffusion=FixedSampleDiffusion(
            zero_scaled_residuals
        ),
        number_of_spectra=7,
        generation_batch_size=3,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=model_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
    )

    generated_batch_1 = generate_spectra(
        diffusion=FixedSampleDiffusion(
            zero_scaled_residuals
        ),
        number_of_spectra=7,
        generation_batch_size=1,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=model_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
    )

    np.testing.assert_allclose(
        generated_batch_3,
        expected_priors,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        generated_batch_1,
        expected_priors,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_5_generation_variation_scale_contracts_around_pca_mean():
    """验证0.90校准只围绕checkpoint中的PCA均值谱收缩。"""

    model_axis = np.arange(
        16,
        dtype=np.float64,
    )

    random_generator = np.random.default_rng(
        9157
    )

    training_spectra = random_generator.normal(
        loc=0.0,
        scale=0.25,
        size=(16, model_axis.size),
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
        .fit(training_spectra)
    )

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            transformer.state_dict()
        )
    )

    adapter = SpectrumLengthAdapter.create(
        dimension_multipliers=[
            1,
            2,
            4,
        ],
        raman_shifts=[
            model_axis
        ],
        model_length="auto",
        padding_mode="right_zero_padding",
        padding_value=0.0,
        raman_range_tolerance=1.0,
    )

    zero_scaled_residuals = np.zeros(
        (
            7,
            adapter.padded_length,
        ),
        dtype=np.float32,
    )

    sampled_priors = (
        restored_transformer.sample_reference_priors(
            7,
            random_generator=np.random.default_rng(
                2026
            ),
        )
    )

    variation_scale = 0.90
    variation_center = np.asarray(
        restored_transformer.pca_mean,
        dtype=np.float32,
    )[np.newaxis, :]
    expected = (
        variation_center
        + variation_scale
        * (
            sampled_priors
            - variation_center
        )
    )

    generated = generate_spectra(
        diffusion=FixedSampleDiffusion(
            zero_scaled_residuals
        ),
        number_of_spectra=7,
        generation_batch_size=3,
        device=torch.device("cpu"),
        length_adapter=adapter,
        output_raman_shifts=model_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
        prior_random_seed=2026,
        variation_scale=variation_scale,
    )

    np.testing.assert_allclose(
        generated,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def _sampling_calibration_reference():
    axis = np.arange(600.0, 801.0, 1.0, dtype=np.float64)
    first_peak = np.exp(-0.5 * ((axis - 660.0) / 4.0) ** 2)
    second_peak = 0.75 * np.exp(-0.5 * ((axis - 742.0) / 5.0) ** 2)
    reference = 0.1 + first_peak + second_peak
    return axis, reference.astype(np.float32)


def test_sampling_calibration_reduces_only_non_peak_wide_noise():
    """验证宽杂讯被定向压缩，而自动峰区中心保持不变。"""

    axis, reference = _sampling_calibration_reference()
    phases = np.linspace(0.0, np.pi, 8, endpoint=False)
    spectra = np.stack(
        [
            reference
            + 0.12 * np.sin((axis - 600.0) / 4.0 + phase)
            + 0.025 * np.sin((axis - 600.0) * 2.1 + phase)
            for phase in phases
        ]
    ).astype(np.float32)
    configuration = {
        "enabled": True,
        "non_peak_noise": {
            "enabled": True,
            "reference_smoothing_sigma_cm1": 24.0,
            "minimum_relative_prominence": 0.05,
            "minimum_peak_distance_cm1": 20.0,
            "maximum_peak_count": 6,
            "peak_protection_half_width_cm1": 10.0,
            "transition_width_cm1": 4.0,
            "fine_sigma_cm1": 1.5,
            "broad_sigma_cm1": 7.0,
            "middle_component_scale": 0.40,
            "fine_component_scale": 0.85,
        },
        "raman_shift_jitter": {"enabled": False},
    }
    calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    calibrated = calibrator.apply(spectra)

    non_peak = (
        (np.abs(axis - 660.0) > 16.0)
        & (np.abs(axis - 742.0) > 16.0)
    )
    before_width = np.median(
        np.percentile(spectra[:, non_peak], 97.5, axis=1)
        - np.percentile(spectra[:, non_peak], 2.5, axis=1)
    )
    after_width = np.median(
        np.percentile(calibrated[:, non_peak], 97.5, axis=1)
        - np.percentile(calibrated[:, non_peak], 2.5, axis=1)
    )
    assert after_width < 0.75 * before_width

    for peak_position in (660.0, 742.0):
        protected = np.abs(axis - peak_position) <= 8.0
        np.testing.assert_allclose(
            calibrated[:, protected],
            spectra[:, protected],
            rtol=0.0,
            atol=1.0e-6,
        )

    summary = calibrator.summary()
    assert summary["detected_peak_count"] == 2


def test_sampling_calibration_adds_coherent_bounded_peak_shift():
    """验证整谱峰位共同漂移约±5 cm^-1且峰间距不被破坏。"""

    axis, reference = _sampling_calibration_reference()
    spectra = np.repeat(reference[np.newaxis, :], 100, axis=0)
    configuration = {
        "enabled": True,
        "non_peak_noise": {"enabled": False},
        "raman_shift_jitter": {
            "enabled": True,
            "distribution": "truncated_normal",
            "standard_deviation_cm1": 3.0,
            "maximum_absolute_shift_cm1": 5.0,
        },
    }
    first_calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    shifted = first_calibrator.apply(spectra)
    summary = first_calibrator.summary()

    assert -5.0 <= summary["sampled_shift_min_cm1"] < -4.0
    assert 4.0 < summary["sampled_shift_max_cm1"] <= 5.0
    assert 1.5 < summary["sampled_shift_std_cm1"] < 3.0

    first_window = (axis >= 650.0) & (axis <= 670.0)
    second_window = (axis >= 732.0) & (axis <= 752.0)
    first_indices = np.flatnonzero(first_window)
    second_indices = np.flatnonzero(second_window)
    first_centers = axis[
        first_indices[np.argmax(shifted[:, first_window], axis=1)]
    ]
    second_centers = axis[
        second_indices[np.argmax(shifted[:, second_window], axis=1)]
    ]
    np.testing.assert_allclose(
        second_centers - first_centers,
        82.0,
        rtol=0.0,
        atol=1.0,
    )

    second_calibrator = SersSamplingCalibrator(
        raman_shift=axis,
        reference_spectrum=reference,
        configuration=configuration,
        random_seed=2026,
    )
    np.testing.assert_allclose(
        shifted,
        second_calibrator.apply(spectra),
        rtol=0.0,
        atol=0.0,
    )

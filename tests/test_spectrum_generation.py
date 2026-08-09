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
            samples[:, None, :],
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

        start = self.sample_offset
        end = start + batch_size

        if end > self.generated_samples.shape[0]:
            raise RuntimeError(
                "测试扩散模型中的固定样本数量不足。"
            )

        batch = self.generated_samples[
            start:end
        ]

        self.sample_offset = end

        return batch


def test_spectrum_generation_shape():
    """验证普通DDPM能够生成正确形状的张量。"""

    model_configuration = {
        "channels": 1,
        "base_dimension": 8,
        "dimension_multipliers": [1, 2],
        "self_condition": False,
        "dropout": 0.0,
        "diffusion_timesteps": 10,
        "sampling_timesteps": 5,
        "objective": "pred_noise",
        "beta_schedule": "cosine",
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
    }

    _, diffusion = build_diffusion_model(
        model_configuration=model_configuration,
        sequence_length=16,
    )

    with torch.inference_mode():
        generated = diffusion.sample(
            batch_size=2
        )

    assert generated.shape == (
        2,
        1,
        16,
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

    # 三条训练集完整归一化光谱。
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

    # 假设DDPM最终应该恢复出的两条完整归一化光谱。
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
        ).fit(
            training_spectra
        )
    )

    # 模拟训练时保存到checkpoint的状态，
    # 再模拟生成时从checkpoint恢复。
    prior_residual_state = (
        fitted_transformer.state_dict()
    )

    restored_transformer = (
        PriorResidualTransformer.from_state_dict(
            prior_residual_state
        )
    )

    # 将预期完整光谱转换为DDPM实际学习和生成的
    # 缩放残差。
    scaled_residuals = (
        restored_transformer.transform(
            expected_model_axis_spectra
        )
    )

    # 统一训练轴为5点，U-Net要求长度为4的倍数，
    # 因此这里会自动在末尾补齐到8点。
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

    assert length_adapter.original_length == 5
    assert length_adapter.padded_length == 8
    assert length_adapter.padding_size == 3

    padded_scaled_residuals = (
        length_adapter.adapt(
            scaled_residuals
        )
    )

    assert padded_scaled_residuals.shape == (
        2,
        8,
    )

    fixed_diffusion = FixedSampleDiffusion(
        padded_scaled_residuals
    )

    # 使用与统一训练轴不同的3点输出轴，
    # 同时验证D2逆变换发生在轴插值之前。
    output_axis = np.asarray(
        [
            100.0,
            300.0,
            500.0,
        ],
        dtype=np.float64,
    )

    generated = generate_spectra(
        diffusion=fixed_diffusion,
        number_of_spectra=2,
        generation_batch_size=1,
        device=torch.device("cpu"),
        length_adapter=length_adapter,
        output_raman_shifts=output_axis,
        prior_residual_transformer=(
            restored_transformer
        ),
    )

    # 输出轴正好对应统一训练轴的第0、2、4个点。
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

    assert generated.shape == (
        2,
        3,
    )

    np.testing.assert_allclose(
        generated,
        expected_output_spectra,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_d2_2_generation_uses_sampled_pca_priors():
    model_axis = np.arange(5, dtype=np.float64)
    training_spectra = np.asarray(
        [
            [-0.8, -0.4, 0.0, 0.4, 0.8],
            [-0.5, -0.2, 0.2, 0.5, 0.7],
            [-0.3, 0.1, 0.4, 0.7, 0.6],
            [-0.6, -0.1, 0.1, 0.6, 0.5],
        ],
        dtype=np.float32,
    )
    transformer = PriorResidualTransformer(
        prior_method="pca_reconstruction",
        normalization_method="pointwise_mad_asinh",
        pca_explained_variance_ratio=0.90,
        pca_max_components=2,
    ).fit(training_spectra)
    adapter = SpectrumLengthAdapter.create(
        dimension_multipliers=[1, 2, 4],
        raman_shifts=[model_axis],
        model_length="auto",
        padding_mode="right_zero_padding",
        padding_value=0.0,
        raman_range_tolerance=1.0,
    )
    fixed_diffusion = FixedSampleDiffusion(
        np.zeros((3, adapter.padded_length), dtype=np.float32)
    )
    expected = transformer.sample_reference_priors(
        3,
        random_generator=np.random.default_rng(2026),
    )
    generated = generate_spectra(
        diffusion=fixed_diffusion,
        number_of_spectra=3,
        generation_batch_size=2,
        device=torch.device("cpu"),
        length_adapter=adapter,
        prior_residual_transformer=transformer,
        prior_random_seed=2026,
    )
    np.testing.assert_allclose(generated, expected, rtol=1.0e-6, atol=1.0e-6)
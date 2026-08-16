"""Generate spectra from a trained one-dimensional diffusion model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.prior_residual import (
    PriorResidualTransformer,
)
from src.broad_local_residual import (
    BroadLocalResidualDecomposer,
)
from src.feature_peak_residual_limiter import (
    FeaturePeakResidualLimiter,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)
from src.sers_sampling_calibrator import (
    SersSamplingCalibrator,
)


@torch.inference_mode()
def generate_spectra(
    *,
    diffusion: nn.Module,
    number_of_spectra: int,
    generation_batch_size: int,
    device: torch.device,
    length_adapter: SpectrumLengthAdapter,
    output_raman_shifts: np.ndarray | None = None,
    prior_residual_transformer: (
        PriorResidualTransformer | None
    ) = None,
    broad_local_residual_decomposer: (
        BroadLocalResidualDecomposer | None
    ) = None,
    feature_peak_residual_limiter: (
        FeaturePeakResidualLimiter | None
    ) = None,
    prior_random_seed: int | None = None,
    variation_scale: float = 1.0,
    sampling_calibrator: (
        SersSamplingCalibrator | None
    ) = None,
) -> np.ndarray:
    """
    分批生成光谱。

    首先删除模型输入末尾的补齐点；如果提供了
    output_raman_shifts，再将光谱插值恢复到指定的
    原始拉曼位移轴。
    """

    if number_of_spectra <= 0:
        raise ValueError(
            "number_of_spectra必须大于0。"
        )

    if generation_batch_size <= 0:
        raise ValueError(
            "generation_batch_size必须大于0。"
        )

    if (
        feature_peak_residual_limiter is not None
        and prior_residual_transformer is None
    ):
        raise ValueError(
            "D3.5特征峰残差软上限必须与D2先验残差变换器一起使用。"
        )

    if broad_local_residual_decomposer is not None:
        if prior_residual_transformer is None:
            raise ValueError(
                "D2.6 broad-local residual必须与"
                "PCA prior_residual_transformer一起使用。"
            )

        if (
            prior_residual_transformer.prior_method
            != "pca_reconstruction"
        ):
            raise ValueError(
                "D2.6当前只支持pca_reconstruction outer prior。"
            )

        if feature_peak_residual_limiter is not None:
            raise ValueError(
                "D2.6第一轮消融不与D3.5 limiter组合。"
            )

    variation_scale = float(variation_scale)

    if (
        not np.isfinite(variation_scale)
        or variation_scale <= 0.0
        or variation_scale > 1.0
    ):
        raise ValueError(
            "variation_scale必须位于(0, 1]范围内。"
        )

    variation_center: np.ndarray | None = None

    if variation_scale < 1.0:
        if prior_residual_transformer is None:
            raise ValueError(
                "variation_scale小于1时，必须启用"
                "prior_residual_transformer。"
            )

        if (
            prior_residual_transformer.prior_method
            != "pca_reconstruction"
        ):
            raise ValueError(
                "当前variation_scale校准只支持"
                "pca_reconstruction先验。"
            )

        if prior_residual_transformer.pca_mean is None:
            raise RuntimeError(
                "checkpoint中的PCA状态缺少pca_mean。"
            )

        variation_center = np.asarray(
            prior_residual_transformer.pca_mean,
            dtype=np.float32,
        ).reshape(-1)

        if variation_center.size < 2:
            raise RuntimeError(
                "checkpoint中的pca_mean长度无效。"
            )

        if not np.isfinite(variation_center).all():
            raise RuntimeError(
                "checkpoint中的pca_mean包含NaN或无穷值。"
            )

    target_axis: np.ndarray | None = None

    if output_raman_shifts is not None:
        target_axis = np.asarray(
            output_raman_shifts,
            dtype=np.float64,
        ).reshape(-1)

        if target_axis.size < 2:
            raise ValueError(
                "输出拉曼位移轴至少需要包含两个点。"
            )

        if not np.isfinite(target_axis).all():
            raise ValueError(
                "输出拉曼位移轴包含NaN或无穷值。"
            )

        if not np.all(
            np.diff(target_axis) > 0.0
        ):
            raise ValueError(
                "输出拉曼位移轴必须严格递增。"
            )

    diffusion = diffusion.to(device)
    diffusion.eval()

    prior_random_generator = np.random.default_rng(
        prior_random_seed
    )

    broad_random_generator = None

    if broad_local_residual_decomposer is not None:
        broad_random_seed = (
            None
            if prior_random_seed is None
            else int(prior_random_seed)
            + 1_000_003
        )

        broad_random_generator = (
            np.random.default_rng(
                broad_random_seed
            )
        )

    generated_batches: list[np.ndarray] = []
    number_generated = 0

    while number_generated < number_of_spectra:
        current_batch_size = min(
            generation_batch_size,
            number_of_spectra - number_generated,
        )

        generated = diffusion.sample(
            batch_size=current_batch_size,
        )

        if generated.ndim != 3:
            raise RuntimeError(
                "生成张量应为[B,C,L]，实际为"
                f"{tuple(generated.shape)}。"
            )

        if generated.shape[0] != current_batch_size:
            raise RuntimeError(
                "模型返回的生成光谱数量与请求的"
                "批次大小不一致。"
            )

        if generated.shape[1] != 1:
            raise RuntimeError(
                "当前项目要求生成结果只有一个光谱通道。"
            )

        generated_numpy = (
            generated[:, 0, :]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        # 删除模型输入末尾的补齐点，
        # 恢复到训练使用的统一拉曼位移轴。
        restored = length_adapter.restore(
            generated_numpy
        )

        # D2：此时数据已经删除末尾补齐点，
        # 但仍位于统一训练拉曼轴上。
        # 先取消残差缩放并加回逐点中位数先验，
        # 再恢复到标签对应的原始拉曼轴。
        if (
            prior_residual_transformer
            is not None
        ):
            reference_priors = None

            if (
                prior_residual_transformer.prior_method
                == "pca_reconstruction"
            ):
                reference_priors = (
                    prior_residual_transformer.sample_reference_priors(
                        current_batch_size,
                        random_generator=prior_random_generator,
                    )
                )

            if broad_local_residual_decomposer is not None:
                if reference_priors is None:
                    raise RuntimeError(
                        "D2.6生成缺少outer PCA reference priors。"
                    )

                if broad_random_generator is None:
                    raise RuntimeError(
                        "D2.6生成缺少broad随机数生成器。"
                    )

                local_raw_residuals = (
                    broad_local_residual_decomposer
                    .inverse_local_transform(
                        restored
                    )
                )

                sampled_broad_residuals = (
                    broad_local_residual_decomposer
                    .sample_broad_residuals(
                        current_batch_size,
                        random_generator=(
                            broad_random_generator
                        ),
                    )
                )

                restored = (
                    reference_priors
                    + sampled_broad_residuals
                    + local_raw_residuals
                ).astype(
                    np.float32,
                    copy=False,
                )

            else:
                restored = (
                    prior_residual_transformer
                    .inverse_transform(
                        restored,
                        reference_priors=reference_priors,
                    )
                )

        # D2.5最终生成离散度校准：
        # 完整归一化光谱重建完成后，以checkpoint中训练集
        # 拟合的PCA均值谱为中心，温和收缩样本间离散度。
        # 不进行平滑，也不写死任何特征峰位置。
        if variation_center is not None:
            if restored.ndim != 2:
                raise RuntimeError(
                    "离散度校准要求光谱数组为[N,L]，"
                    f"实际形状为{restored.shape}。"
                )

            if restored.shape[1] != variation_center.size:
                raise RuntimeError(
                    "重建光谱与PCA均值谱长度不一致："
                    f"{restored.shape[1]} != "
                    f"{variation_center.size}。"
                )

            restored = (
                variation_center[np.newaxis, :]
                + variation_scale
                * (
                    restored
                    - variation_center[np.newaxis, :]
                )
            ).astype(
                np.float32,
                copy=False,
            )

        # D3.5：先恢复到完整归一化光谱，再相对训练集先验
        # 只软限制自动识别的特征峰窗口中的残差幅度。
        # 此步骤必须发生在轴插值和全局反归一化之前。
        if feature_peak_residual_limiter is not None:
            restored = (
                feature_peak_residual_limiter
                .apply_to_normalized_spectra(restored)
            )

        # 如果指定了标签或模板文件的原始位移轴，
        # 再从统一训练轴插值回该输出轴。
        if target_axis is not None:
            restored = (
                length_adapter.interpolate_from_model_axis(
                    restored,
                    target_axis,
                )
            )

        restored = np.asarray(
            restored,
            dtype=np.float32,
        )

        if restored.ndim != 2:
            raise RuntimeError(
                "恢复后的生成光谱应为二维数组"
                "[光谱数量, 光谱点数]，实际为"
                f"{restored.shape}。"
            )

        if restored.shape[0] != current_batch_size:
            raise RuntimeError(
                "恢复后的生成光谱数量与当前"
                "生成批次大小不一致。"
            )

        if not np.isfinite(restored).all():
            raise RuntimeError(
                "生成结果包含NaN或无穷值。"
            )

        generated_batches.append(
            restored
        )

        number_generated += (
            current_batch_size
        )

    generated_spectra = np.concatenate(
        generated_batches,
        axis=0,
    )

    if (
        generated_spectra.shape[0]
        != number_of_spectra
    ):
        raise RuntimeError(
            "最终生成的光谱数量与请求数量不一致。"
        )

    # D2.5 targeted sampling calibration：
    # 在所有批次完成并恢复到最终输出轴后统一处理，
    # 使随机峰位微漂移不受generation batch size影响。
    if sampling_calibrator is not None:
        generated_spectra = sampling_calibrator.apply(
            generated_spectra
        )

    return generated_spectra

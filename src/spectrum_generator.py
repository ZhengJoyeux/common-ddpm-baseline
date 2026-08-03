"""Generate spectra from a trained one-dimensional diffusion model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
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

    return generated_spectra
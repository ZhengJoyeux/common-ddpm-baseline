"""D2/D3 残差感知 pred_x0 主损失的配置与逐点权重。"""

from __future__ import annotations

from typing import Any

import torch


def normalize_residual_aware_loss_configuration(
    configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """规范化 residual-aware pointwise MSE 配置。

    本模块只改变 DDPM 主损失在 Raman 轴上的逐点权重，
    不改变网络结构、扩散时间步、前向加噪过程或采样公式。
    """

    if configuration is None:
        configuration = {}

    if not isinstance(configuration, dict):
        raise TypeError(
            "diffusion.residual_aware_loss必须是字典。"
        )

    normalized = dict(configuration)

    normalized["enabled"] = bool(
        normalized.get("enabled", False)
    )

    normalized["reference_quantile"] = float(
        normalized.get(
            "reference_quantile",
            0.90,
        )
    )

    normalized["strength"] = float(
        normalized.get(
            "strength",
            2.0,
        )
    )

    normalized["power"] = float(
        normalized.get(
            "power",
            1.0,
        )
    )

    normalized["maximum_relative_magnitude"] = float(
        normalized.get(
            "maximum_relative_magnitude",
            3.0,
        )
    )

    normalized["minimum_reference_scale"] = float(
        normalized.get(
            "minimum_reference_scale",
            0.02,
        )
    )

    normalized["normalize_per_sample"] = bool(
        normalized.get(
            "normalize_per_sample",
            True,
        )
    )

    quantile = normalized["reference_quantile"]

    if not 0.0 < quantile < 1.0:
        raise ValueError(
            "diffusion.residual_aware_loss."
            "reference_quantile必须在(0,1)内。"
        )

    strength = normalized["strength"]

    if strength < 0.0:
        raise ValueError(
            "diffusion.residual_aware_loss."
            "strength不能小于0。"
        )

    if (
        normalized["enabled"]
        and strength <= 0.0
    ):
        raise ValueError(
            "启用residual_aware_loss时"
            "strength必须大于0。"
        )

    if normalized["power"] <= 0.0:
        raise ValueError(
            "diffusion.residual_aware_loss."
            "power必须大于0。"
        )

    if (
        normalized[
            "maximum_relative_magnitude"
        ]
        <= 0.0
    ):
        raise ValueError(
            "diffusion.residual_aware_loss."
            "maximum_relative_magnitude必须大于0。"
        )

    if (
        normalized[
            "minimum_reference_scale"
        ]
        <= 0.0
    ):
        raise ValueError(
            "diffusion.residual_aware_loss."
            "minimum_reference_scale必须大于0。"
        )

    return normalized


def build_residual_aware_pointwise_weights(
    target_scaled_residual: torch.Tensor,
    configuration: dict[str, Any],
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """根据真实 scaled residual 幅度建立逐 Raman 点权重。

    每条光谱使用自身 |R_scaled| 的指定分位数作为参考尺度。

    大 residual 点获得更高权重，但通过
    maximum_relative_magnitude 限制极端点影响。

    默认把每条光谱的权重重新归一化到均值 1，
    从而避免整体 DDPM loss 尺度因加权系统性增大。

    权重完全由 target 构建并 detach，
    不通过权重计算传播梯度。
    """

    normalized = (
        normalize_residual_aware_loss_configuration(
            configuration
        )
    )

    target = target_scaled_residual

    if target.ndim < 2:
        raise ValueError(
            "target_scaled_residual至少需要"
            "batch维和特征维。"
        )

    if not torch.is_floating_point(target):
        raise TypeError(
            "target_scaled_residual必须是"
            "浮点Tensor。"
        )

    absolute_target = (
        target
        .detach()
        .abs()
    )

    batch_size = target.shape[0]

    if not normalized["enabled"]:
        weights = torch.ones_like(target)

        zero = torch.zeros(
            (),
            device=target.device,
            dtype=target.dtype,
        )

        one = torch.ones_like(zero)

        diagnostics = {
            "reference_scale_mean": zero,
            "weight_mean": one,
            "weight_min": one,
            "weight_max": one,
        }

        return weights, diagnostics

    flattened = absolute_target.flatten(
        start_dim=1
    )

    reference_scale = torch.quantile(
        flattened,
        q=normalized[
            "reference_quantile"
        ],
        dim=1,
        keepdim=True,
    )

    reference_scale = (
        reference_scale.clamp_min(
            normalized[
                "minimum_reference_scale"
            ]
        )
    )

    broadcast_shape = (
        batch_size,
        *((1,) * (target.ndim - 1)),
    )

    relative_magnitude = (
        absolute_target
        / reference_scale.reshape(
            broadcast_shape
        )
    )

    relative_magnitude = (
        relative_magnitude.clamp(
            min=0.0,
            max=normalized[
                "maximum_relative_magnitude"
            ],
        )
    )

    weights = (
        1.0
        + normalized["strength"]
        * relative_magnitude.pow(
            normalized["power"]
        )
    )

    if normalized[
        "normalize_per_sample"
    ]:
        per_sample_mean = (
            weights
            .flatten(start_dim=1)
            .mean(
                dim=1,
                keepdim=True,
            )
        )

        eps = torch.finfo(
            weights.dtype
        ).eps

        weights = (
            weights
            / per_sample_mean
            .clamp_min(eps)
            .reshape(
                broadcast_shape
            )
        )

    weights = weights.detach()

    diagnostics = {
        "reference_scale_mean":
            reference_scale.mean().detach(),
        "weight_mean":
            weights.mean().detach(),
        "weight_min":
            weights.min().detach(),
        "weight_max":
            weights.max().detach(),
    }

    return weights, diagnostics

"""构建一维 U-Net、D0-D2 扩散模型或 D3 物理/多样性扩散模型。"""

from __future__ import annotations

from typing import Any

import torch

from src.one_dimensional_ddpm import (
    GaussianDiffusion1D,
    Unet1D,
    check_backend_version,
)
from src.physics_guided_diffusion import (
    SersPhysicsGuidedGaussianDiffusion1D,
)
from src.sers_diversity_constraints import (
    normalize_diversity_configuration,
)
from src.sers_physics_constraints import (
    normalize_physics_configuration,
)


def _get_required_value(
    configuration: dict[str, Any],
    *possible_keys: str,
) -> Any:
    """读取必需配置，同时兼容项目历史字段名。"""

    for key in possible_keys:
        if key in configuration:
            return configuration[key]

    joined_keys = " 或 ".join(repr(key) for key in possible_keys)

    raise KeyError(f"模型配置中缺少必要参数：{joined_keys}")


def _validate_positive_integer(value: Any, field_name: str) -> int:
    """读取大于 0 的整数参数。"""

    parsed = int(value)

    if parsed <= 0:
        raise ValueError(f"{field_name}必须大于0。")

    return parsed


def _configure_high_noise_loss_weight(
    diffusion_model: GaussianDiffusion1D,
    diffusion_configuration: dict[str, Any],
    diffusion_timesteps: int,
) -> None:
    """为高噪声时间步建立递增的DDPM主损失权重。

    权重按训练时间步从低噪声到高噪声递增，并在最后按平均值归一化，
    这样可以提高中高噪声时间步的重要性，同时避免整体损失尺度发生
    不必要的大幅改变。
    """

    high_noise = diffusion_configuration.get("high_noise", {})

    if high_noise is None:
        high_noise = {}

    if not isinstance(high_noise, dict):
        raise TypeError("diffusion.high_noise必须是字典。")

    minimum_weight = float(
        high_noise.get("minimum_weight", 0.50)
    )
    maximum_weight = float(
        high_noise.get("maximum_weight", 2.00)
    )
    ramp_power = float(
        high_noise.get("ramp_power", 1.0)
    )

    if minimum_weight <= 0.0:
        raise ValueError(
            "diffusion.high_noise.minimum_weight必须大于0。"
        )

    if maximum_weight <= minimum_weight:
        raise ValueError(
            "diffusion.high_noise.maximum_weight必须大于"
            "minimum_weight。"
        )

    if ramp_power <= 0.0:
        raise ValueError(
            "diffusion.high_noise.ramp_power必须大于0。"
        )

    progress = torch.linspace(
        0.0,
        1.0,
        steps=diffusion_timesteps,
        dtype=diffusion_model.loss_weight.dtype,
        device=diffusion_model.loss_weight.device,
    )
    weights = minimum_weight + (
        maximum_weight - minimum_weight
    ) * progress.pow(ramp_power)
    weights = weights / weights.mean().clamp_min(
        torch.finfo(weights.dtype).eps
    )

    if tuple(weights.shape) != tuple(
        diffusion_model.loss_weight.shape
    ):
        raise RuntimeError(
            "高噪声损失权重的长度与扩散时间步数量不一致："
            f"weights={tuple(weights.shape)}，"
            f"loss_weight={tuple(diffusion_model.loss_weight.shape)}。"
        )

    with torch.no_grad():
        diffusion_model.loss_weight.copy_(weights)


def build_diffusion_model(
    model_configuration: dict[str, Any],
    sequence_length: int,
) -> tuple[Unet1D, GaussianDiffusion1D]:
    """
    构建并返回一维 U-Net 和扩散模型。

    当 physics_constraints.enabled=false 且
    diversity_constraints.enabled=false 时，仍构建第三方原始
    GaussianDiffusion1D，保持 D0-D2.1 行为不变。

    当 physics_constraints 或 diversity_constraints 任意一个启用时，
    构建 SersPhysicsGuidedGaussianDiffusion1D。该子类只改变训练损失，
    不改变采样接口和网络参数结构。
    """

    check_backend_version()

    if not isinstance(model_configuration, dict):
        raise TypeError("model_configuration必须是字典。")

    sequence_length = _validate_positive_integer(
        sequence_length,
        "sequence_length",
    )

    architecture_configuration = model_configuration.get(
        "model",
        model_configuration,
    )
    diffusion_configuration = model_configuration.get(
        "diffusion",
        model_configuration,
    )

    if not isinstance(architecture_configuration, dict):
        raise TypeError("model配置必须是字典。")

    if not isinstance(diffusion_configuration, dict):
        raise TypeError("diffusion配置必须是字典。")

    dimension_multipliers = tuple(
        int(value)
        for value in _get_required_value(
            architecture_configuration,
            "dimension_multipliers",
        )
    )

    if not dimension_multipliers:
        raise ValueError("dimension_multipliers不能为空。")

    if any(value <= 0 for value in dimension_multipliers):
        raise ValueError("dimension_multipliers中的数值必须大于0。")

    downsample_factor = 2 ** (len(dimension_multipliers) - 1)

    if sequence_length % downsample_factor != 0:
        raise ValueError(
            f"模型输入长度{sequence_length}必须能被"
            f"{downsample_factor}整除。"
        )

    base_dimension = _validate_positive_integer(
        _get_required_value(
            architecture_configuration,
            "base_dimension",
            "model_dimension",
        ),
        "model.base_dimension",
    )
    channels = _validate_positive_integer(
        _get_required_value(
            architecture_configuration,
            "channels",
        ),
        "model.channels",
    )

    dropout = float(
        architecture_configuration.get("dropout", 0.0)
    )

    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout必须在[0,1)范围内。")

    unet = Unet1D(
        dim=base_dimension,
        dim_mults=dimension_multipliers,
        channels=channels,
        dropout=dropout,
        self_condition=bool(
            architecture_configuration.get("self_condition", False)
        ),
    )

    diffusion_timesteps = _validate_positive_integer(
        _get_required_value(
            diffusion_configuration,
            "diffusion_timesteps",
            "diffusion_steps",
        ),
        "diffusion.diffusion_steps",
    )
    sampling_timesteps = _validate_positive_integer(
        _get_required_value(
            diffusion_configuration,
            "sampling_timesteps",
            "sampling_steps",
        ),
        "diffusion.sampling_steps",
    )

    if sampling_timesteps > diffusion_timesteps:
        raise ValueError(
            "diffusion.sampling_steps不能大于diffusion.diffusion_steps。"
        )

    objective = str(
        _get_required_value(
            diffusion_configuration,
            "objective",
        )
    ).strip().lower()

    if objective not in {"pred_noise", "pred_x0", "pred_v"}:
        raise ValueError(
            "diffusion.objective必须为pred_noise、pred_x0或pred_v。"
        )

    loss_weighting = str(
        diffusion_configuration.get(
            "loss_weighting",
            "library_default",
        )
    ).strip().lower()

    if loss_weighting not in {
        "library_default",
        "snr",
        "uniform",
        "high_noise",
    }:
        raise ValueError(
            "diffusion.loss_weighting必须为"
            "library_default、snr、uniform或high_noise。"
        )

    if loss_weighting == "snr" and objective != "pred_x0":
        raise ValueError(
            "diffusion.loss_weighting=snr目前只允许与"
            "diffusion.objective=pred_x0配合使用。"
        )

    common_arguments = {
        "model": unet,
        "seq_length": sequence_length,
        "timesteps": diffusion_timesteps,
        "sampling_timesteps": sampling_timesteps,
        "objective": objective,
        "beta_schedule": str(
            _get_required_value(
                diffusion_configuration,
                "beta_schedule",
            )
        ),
        "ddim_sampling_eta": float(
            diffusion_configuration.get("ddim_sampling_eta", 0.0)
        ),
        "auto_normalize": bool(
            diffusion_configuration.get("auto_normalize", True)
        ),
    }

    raw_physics_configuration = model_configuration.get(
        "physics_constraints",
        {"enabled": False},
    )

    if raw_physics_configuration is None:
        raw_physics_configuration = {"enabled": False}

    physics_configuration = normalize_physics_configuration(
        raw_physics_configuration
    )

    raw_diversity_configuration = model_configuration.get(
        "diversity_constraints",
        {"enabled": False},
    )

    if raw_diversity_configuration is None:
        raw_diversity_configuration = {"enabled": False}

    diversity_configuration = normalize_diversity_configuration(
        raw_diversity_configuration
    )

    if (
        bool(physics_configuration.get("enabled", False))
        or bool(diversity_configuration.get("enabled", False))
    ):
        diffusion_model = SersPhysicsGuidedGaussianDiffusion1D(
            physics_configuration=physics_configuration,
            diversity_configuration=diversity_configuration,
            **common_arguments,
        )
    else:
        diffusion_model = GaussianDiffusion1D(**common_arguments)

    if loss_weighting == "uniform":
        diffusion_model.loss_weight.fill_(1.0)
    elif loss_weighting == "high_noise":
        _configure_high_noise_loss_weight(
            diffusion_model,
            diffusion_configuration,
            diffusion_timesteps,
        )

    diffusion_model.configured_loss_weighting = loss_weighting

    return unet, diffusion_model
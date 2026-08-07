"""构建一维 U-Net、D0-D2 扩散模型或 D3.1 物理引导扩散模型。"""

from __future__ import annotations

from typing import Any

from src.one_dimensional_ddpm import (
    GaussianDiffusion1D,
    Unet1D,
    check_backend_version,
)
from src.physics_guided_diffusion import (
    SersPhysicsGuidedGaussianDiffusion1D,
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


def build_diffusion_model(
    model_configuration: dict[str, Any],
    sequence_length: int,
) -> tuple[Unet1D, GaussianDiffusion1D]:
    """
    构建并返回一维 U-Net 和扩散模型。

    当 ``physics_constraints.enabled=false`` 时，仍构建第三方原始
    ``GaussianDiffusion1D``，保持 D0-D2.1 行为不变。

    当 ``physics_constraints.enabled=true`` 时，构建
    ``SersPhysicsGuidedGaussianDiffusion1D``。该子类只改变训练损失，
    不改变采样接口和网络参数结构。
    """

    check_backend_version()

    if not isinstance(model_configuration, dict):
        raise TypeError("model_configuration必须是字典。")

    sequence_length = _validate_positive_integer(
        sequence_length,
        "sequence_length",
    )

    # 兼容直接传入扁平配置和传入完整 YAML 配置两种情况。
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

    if loss_weighting not in {"library_default", "snr", "uniform"}:
        raise ValueError(
            "diffusion.loss_weighting必须为"
            "library_default、snr或uniform。"
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

    if bool(physics_configuration.get("enabled", False)):
        diffusion_model = SersPhysicsGuidedGaussianDiffusion1D(
            physics_configuration=physics_configuration,
            **common_arguments,
        )
    else:
        diffusion_model = GaussianDiffusion1D(**common_arguments)

    # denoising-diffusion-pytorch 2.2.6 的默认 pred_x0 权重是 SNR。
    # D2.1 和 D3.1 的 uniform 模式把所有时间步权重覆盖成 1，
    # 避免高噪声时间步几乎不参与训练。
    if loss_weighting == "uniform":
        diffusion_model.loss_weight.fill_(1.0)

    # 普通 Python 属性不会进入 state_dict，仅用于记录实际配置。
    diffusion_model.configured_loss_weighting = loss_weighting

    return unet, diffusion_model
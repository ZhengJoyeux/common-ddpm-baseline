"""Build the one-dimensional U-Net and DDPM model."""

from __future__ import annotations

from typing import Any

from src.one_dimensional_ddpm import (
    GaussianDiffusion1D,
    Unet1D,
    check_backend_version,
)


def _get_required_value(
    configuration: dict[str, Any],
    *possible_keys: str,
) -> Any:
    """Read a required value while supporting compatible key names."""

    for key in possible_keys:
        if key in configuration:
            return configuration[key]

    joined_keys = " 或 ".join(
        repr(key)
        for key in possible_keys
    )

    raise KeyError(
        f"模型配置中缺少必要参数：{joined_keys}"
    )


def build_diffusion_model(
    model_configuration: dict[str, Any],
    sequence_length: int,
) -> tuple[Unet1D, GaussianDiffusion1D]:
    """
    Build and return the one-dimensional U-Net and diffusion model.

    The function accepts the flat model configuration used by the
    training scripts and tests. It also supports the previous compatible
    key names such as model_dimension and diffusion_steps.

    Supported loss-weighting modes:

    - library_default:
      Keep the loss weighting defined by the installed diffusion library.

    - snr:
      Explicitly use the library's SNR weighting for pred_x0.

    - uniform:
      Give every diffusion timestep the same loss weight of 1.0.
    """

    check_backend_version()

    if not isinstance(model_configuration, dict):
        raise TypeError(
            "model_configuration必须是字典。"
        )

    sequence_length = int(sequence_length)

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length必须大于0。"
        )

    # 兼容直接传入扁平配置，以及传入完整配置字典两种情况。
    architecture_configuration = model_configuration.get(
        "model",
        model_configuration,
    )

    diffusion_configuration = model_configuration.get(
        "diffusion",
        model_configuration,
    )

    if not isinstance(architecture_configuration, dict):
        raise TypeError(
            "model配置必须是字典。"
        )

    if not isinstance(diffusion_configuration, dict):
        raise TypeError(
            "diffusion配置必须是字典。"
        )

    dimension_multipliers = tuple(
        int(value)
        for value in _get_required_value(
            architecture_configuration,
            "dimension_multipliers",
        )
    )

    if not dimension_multipliers:
        raise ValueError(
            "dimension_multipliers不能为空。"
        )

    if any(
        value <= 0
        for value in dimension_multipliers
    ):
        raise ValueError(
            "dimension_multipliers中的数值必须大于0。"
        )

    downsample_factor = 2 ** (
        len(dimension_multipliers) - 1
    )

    if sequence_length % downsample_factor != 0:
        raise ValueError(
            f"模型输入长度{sequence_length}必须能被"
            f"{downsample_factor}整除。"
        )

    base_dimension = int(
        _get_required_value(
            architecture_configuration,
            "base_dimension",
            "model_dimension",
        )
    )

    if base_dimension <= 0:
        raise ValueError(
            "base_dimension必须大于0。"
        )

    channels = int(
        _get_required_value(
            architecture_configuration,
            "channels",
        )
    )

    if channels <= 0:
        raise ValueError(
            "channels必须大于0。"
        )

    dropout = float(
        architecture_configuration.get(
            "dropout",
            0.0,
        )
    )

    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(
            "dropout必须大于等于0且小于1。"
        )

    unet = Unet1D(
        dim=base_dimension,
        dim_mults=dimension_multipliers,
        channels=channels,
        dropout=dropout,
        self_condition=bool(
            architecture_configuration.get(
                "self_condition",
                False,
            )
        ),
    )

    diffusion_timesteps = int(
        _get_required_value(
            diffusion_configuration,
            "diffusion_timesteps",
            "diffusion_steps",
        )
    )

    sampling_timesteps = int(
        _get_required_value(
            diffusion_configuration,
            "sampling_timesteps",
            "sampling_steps",
        )
    )

    if diffusion_timesteps <= 0:
        raise ValueError(
            "diffusion_timesteps必须大于0。"
        )

    if sampling_timesteps <= 0:
        raise ValueError(
            "sampling_timesteps必须大于0。"
        )

    if sampling_timesteps > diffusion_timesteps:
        raise ValueError(
            "sampling_timesteps不能大于"
            "diffusion_timesteps。"
        )

    objective = str(
        _get_required_value(
            diffusion_configuration,
            "objective",
        )
    ).strip().lower()

    supported_objectives = {
        "pred_noise",
        "pred_x0",
        "pred_v",
    }

    if objective not in supported_objectives:
        raise ValueError(
            "diffusion.objective必须为"
            "pred_noise、pred_x0或pred_v。"
        )

    loss_weighting = str(
        diffusion_configuration.get(
            "loss_weighting",
            "library_default",
        )
    ).strip().lower()

    supported_loss_weightings = {
        "library_default",
        "snr",
        "uniform",
    }

    if loss_weighting not in supported_loss_weightings:
        raise ValueError(
            "diffusion.loss_weighting必须为"
            "library_default、snr或uniform。"
        )

    if loss_weighting == "snr" and objective != "pred_x0":
        raise ValueError(
            "diffusion.loss_weighting=snr目前只允许与"
            "diffusion.objective=pred_x0配合使用。"
        )

    diffusion_model = GaussianDiffusion1D(
        model=unet,
        seq_length=sequence_length,
        timesteps=diffusion_timesteps,
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        beta_schedule=str(
            _get_required_value(
                diffusion_configuration,
                "beta_schedule",
            )
        ),
        ddim_sampling_eta=float(
            diffusion_configuration.get(
                "ddim_sampling_eta",
                0.0,
            )
        ),
        auto_normalize=bool(
            diffusion_configuration.get(
                "auto_normalize",
                True,
            )
        ),
    )

    # denoising-diffusion-pytorch 2.2.6中：
    #
    # pred_noise -> 权重为1
    # pred_x0    -> 权重为SNR
    # pred_v     -> 权重为SNR / (SNR + 1)
    #
    # uniform模式会在模型创建完成后，将所有扩散时间步的
    # 损失权重覆盖为1，从而避免pred_x0严重忽略高噪声时间步。
    if loss_weighting == "uniform":
        diffusion_model.loss_weight.fill_(1.0)

    # 记录项目配置采用的权重模式。
    # 该普通属性不会改变第三方库的state_dict结构。
    diffusion_model.configured_loss_weighting = (
        loss_weighting
    )

    return unet, diffusion_model
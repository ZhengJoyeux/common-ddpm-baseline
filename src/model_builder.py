"""构建一维 U-Net 和 D0-D3 扩散模型。"""

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
from src.sers_diversity_constraints import (
    normalize_diversity_configuration,
)
from src.sers_local_peak_distribution_constraints import (
    normalize_local_peak_distribution_configuration,
)
from src.sers_physics_constraints import (
    normalize_physics_configuration,
)


def _get_required_value(
    configuration: dict[str, Any],
    *possible_keys: str,
) -> Any:
    for key in possible_keys:
        if key in configuration:
            return configuration[key]
    joined = " 或 ".join(repr(key) for key in possible_keys)
    raise KeyError(f"模型配置中缺少必要参数：{joined}")


def _as_optional_dictionary(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name}必须是字典。")
    return value


def build_diffusion_model(
    model_configuration: dict[str, Any],
    sequence_length: int,
) -> tuple[Unet1D, GaussianDiffusion1D]:
    """
    构建 D0-D2 普通 GaussianDiffusion1D，或带 D3 额外损失的子类。

    本轮新增的 local_peak_distribution_constraints 只改变训练损失，
    不改变 U-Net 参数结构、DDPM/DDIM 采样接口或 checkpoint 参数键。
    """

    check_backend_version()

    if not isinstance(model_configuration, dict):
        raise TypeError("model_configuration必须是字典。")

    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("sequence_length必须大于0。")

    architecture = _as_optional_dictionary(
        model_configuration.get("model", model_configuration),
        name="model",
    )
    diffusion_config = _as_optional_dictionary(
        model_configuration.get("diffusion", model_configuration),
        name="diffusion",
    )
    physics_raw = _as_optional_dictionary(
        model_configuration.get("physics_constraints", {}),
        name="physics_constraints",
    )
    diversity_raw = _as_optional_dictionary(
        model_configuration.get("diversity_constraints", {}),
        name="diversity_constraints",
    )
    local_peak_raw = _as_optional_dictionary(
        model_configuration.get("local_peak_distribution_constraints", {}),
        name="local_peak_distribution_constraints",
    )
    prior_config = _as_optional_dictionary(
        model_configuration.get("prior_residual", {}),
        name="prior_residual",
    )
    residual_aware_raw = _as_optional_dictionary(
        diffusion_config.get("residual_aware_loss", {}),
        name="diffusion.residual_aware_loss",
    )

    dimension_multipliers = tuple(
        int(value)
        for value in _get_required_value(
            architecture,
            "dimension_multipliers",
        )
    )
    if not dimension_multipliers or any(
        value <= 0 for value in dimension_multipliers
    ):
        raise ValueError("dimension_multipliers必须包含正整数。")

    downsample_factor = 2 ** (len(dimension_multipliers) - 1)
    if sequence_length % downsample_factor != 0:
        raise ValueError(
            f"模型输入长度{sequence_length}必须能被"
            f"{downsample_factor}整除。"
        )

    base_dimension = int(
        _get_required_value(
            architecture,
            "base_dimension",
            "model_dimension",
        )
    )
    channels = int(_get_required_value(architecture, "channels"))
    dropout = float(architecture.get("dropout", 0.0))

    if base_dimension <= 0:
        raise ValueError("base_dimension/model_dimension必须大于0。")
    if channels <= 0:
        raise ValueError("channels必须大于0。")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout必须在[0,1)范围内。")

    unet = Unet1D(
        dim=base_dimension,
        dim_mults=dimension_multipliers,
        channels=channels,
        dropout=dropout,
        self_condition=bool(
            architecture.get("self_condition", False)
        ),
    )

    diffusion_timesteps = int(
        _get_required_value(
            diffusion_config,
            "diffusion_timesteps",
            "diffusion_steps",
        )
    )
    sampling_timesteps = int(
        _get_required_value(
            diffusion_config,
            "sampling_timesteps",
            "sampling_steps",
        )
    )
    if diffusion_timesteps <= 0 or sampling_timesteps <= 0:
        raise ValueError("扩散和采样时间步必须大于0。")
    if sampling_timesteps > diffusion_timesteps:
        raise ValueError("sampling_timesteps不能大于diffusion_timesteps。")

    objective = str(
        _get_required_value(diffusion_config, "objective")
    ).strip().lower()
    if objective not in {"pred_noise", "pred_x0", "pred_v"}:
        raise ValueError("objective必须为pred_noise、pred_x0或pred_v。")

    loss_weighting = str(
        diffusion_config.get("loss_weighting", "library_default")
    ).strip().lower()
    if loss_weighting not in {"library_default", "snr", "uniform"}:
        raise ValueError(
            "loss_weighting必须为library_default、snr或uniform。"
        )
    if loss_weighting == "snr" and objective != "pred_x0":
        raise ValueError("loss_weighting=snr当前只允许与pred_x0配合。")

    common_arguments = {
        "model": unet,
        "seq_length": sequence_length,
        "timesteps": diffusion_timesteps,
        "sampling_timesteps": sampling_timesteps,
        "objective": objective,
        "beta_schedule": str(
            _get_required_value(diffusion_config, "beta_schedule")
        ),
        "ddim_sampling_eta": float(
            diffusion_config.get("ddim_sampling_eta", 0.0)
        ),
        "auto_normalize": bool(
            diffusion_config.get("auto_normalize", True)
        ),
    }

    physics_config = normalize_physics_configuration(physics_raw)
    diversity_config = normalize_diversity_configuration(diversity_raw)
    local_peak_config = normalize_local_peak_distribution_configuration(
        local_peak_raw
    )

    physics_enabled = bool(physics_config.get("enabled", False))
    diversity_enabled = bool(diversity_config.get("enabled", False))
    local_peak_enabled = bool(local_peak_config.get("enabled", False))
    residual_aware_enabled = bool(
        residual_aware_raw.get("enabled", False)
    )

    extra_loss_enabled = any(
        (
            physics_enabled,
            diversity_enabled,
            local_peak_enabled,
            residual_aware_enabled,
        )
    )

    if extra_loss_enabled:
        if objective != "pred_x0":
            raise ValueError(
                "D3/residual-aware额外损失当前必须使用objective=pred_x0。"
            )
        if common_arguments["auto_normalize"]:
            raise ValueError(
                "D2/D3使用项目外部归一化，auto_normalize必须为false。"
            )

    if physics_enabled or local_peak_enabled or residual_aware_enabled:
        if not bool(prior_config.get("enabled", False)):
            raise ValueError(
                "physics/local-peak/residual-aware要求启用prior_residual。"
            )
        residual_method = str(
            prior_config.get("residual_normalization", "")
        ).strip().lower()
        if residual_method != "pointwise_mad_asinh":
            raise ValueError(
                "physics/local-peak/residual-aware当前要求"
                "prior_residual.residual_normalization="
                "pointwise_mad_asinh。"
            )

    if local_peak_enabled:
        if not physics_enabled:
            raise ValueError(
                "local_peak_distribution_constraints依赖"
                "physics_constraint_state，因此必须同时启用"
                "physics_constraints。"
            )
        stable = physics_config.get("training_peak_distribution", {})
        if not bool(stable.get("enabled", False)):
            raise ValueError(
                "local_peak_distribution_constraints要求"
                "physics_constraints.training_peak_distribution.enabled=true，"
                "用于只从训练集拟合稳定峰参考。"
            )

    if extra_loss_enabled:
        diffusion_model = SersPhysicsGuidedGaussianDiffusion1D(
            physics_configuration=physics_config,
            diversity_configuration=diversity_config,
            local_peak_distribution_configuration=local_peak_config,
            residual_aware_configuration=residual_aware_raw,
            **common_arguments,
        )
    else:
        diffusion_model = GaussianDiffusion1D(**common_arguments)

    if loss_weighting == "uniform":
        diffusion_model.loss_weight.fill_(1.0)

    # 普通Python属性不进入state_dict，只用于日志/检查点元数据和调试。
    diffusion_model.configured_loss_weighting = loss_weighting
    diffusion_model.configured_physics_enabled = physics_enabled
    diffusion_model.configured_diversity_enabled = diversity_enabled
    diffusion_model.configured_local_peak_distribution_enabled = (
        local_peak_enabled
    )
    diffusion_model.configured_residual_aware_enabled = (
        residual_aware_enabled
    )

    return unet, diffusion_model
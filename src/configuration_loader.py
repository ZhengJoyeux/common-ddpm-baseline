"""读取、解析并校验项目 YAML 配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.sers_physics_constraints import (
    normalize_physics_configuration,
)


def _auto_or_positive_integer(
    value: Any,
    field_name: str,
) -> int | None:
    """允许长度字段为正整数、None 或字符串 auto。"""

    if value is None or (
        isinstance(value, str)
        and value.strip().lower() == "auto"
    ):
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name}只能是正整数或auto。"
        ) from error

    if parsed <= 0:
        raise ValueError(f"{field_name}必须大于0。")

    return parsed


def resolve_project_path(
    configuration: dict,
    path_value: str | Path,
) -> Path:
    """把配置中的相对路径转换为项目根目录下的绝对路径。"""

    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    project_root = Path(
        configuration.get(
            "_paths",
            {},
        ).get(
            "project_root",
            Path(__file__).resolve().parents[1],
        )
    )

    return (project_root / path).resolve()


def project_path(
    configuration: dict[str, Any],
    value: str | Path,
) -> Path:
    """保留项目原有公开路径函数。"""

    return resolve_project_path(configuration, value)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML、记录项目路径并执行完整校验。"""

    path = Path(config_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("YAML顶层必须是字典结构。")

    configuration["_paths"] = {
        "config_path": str(path),
        "project_root": str(path.parent.parent),
    }

    validate_config(configuration)

    return configuration


def _validate_required_sections(configuration: dict[str, Any]) -> None:
    """检查训练与生成流程必需的顶层区段。"""

    required_sections = {
        "project",
        "data",
        "normalization",
        "model",
        "diffusion",
        "training",
        "output",
        "generation",
    }
    missing = sorted(required_sections.difference(configuration))

    if missing:
        raise KeyError(f"YAML缺少配置区段：{missing}")

    for section_name in required_sections:
        if not isinstance(configuration[section_name], dict):
            raise TypeError(f"YAML区段{section_name}必须是字典。")


def _validate_data_configuration(
    data: dict[str, Any],
    model: dict[str, Any],
) -> None:
    """校验数据划分、拉曼轴适配和网络长度。"""

    original_length = _auto_or_positive_integer(
        data.get("original_spectrum_length", "auto"),
        "data.original_spectrum_length",
    )
    model_length = _auto_or_positive_integer(
        data.get("model_spectrum_length", "auto"),
        "data.model_spectrum_length",
    )

    if (
        original_length is not None
        and model_length is not None
        and model_length < original_length
    ):
        raise ValueError(
            "data.model_spectrum_length不能小于"
            "data.original_spectrum_length。"
        )

    dimension_multipliers = tuple(
        int(value)
        for value in model["dimension_multipliers"]
    )

    if len(dimension_multipliers) < 2:
        raise ValueError(
            "model.dimension_multipliers至少需要两个层级。"
        )

    if any(value <= 0 for value in dimension_multipliers):
        raise ValueError(
            "model.dimension_multipliers必须全部为正整数。"
        )

    downsample_factor = 2 ** (len(dimension_multipliers) - 1)

    if (
        model_length is not None
        and model_length % downsample_factor != 0
    ):
        raise ValueError(
            f"data.model_spectrum_length={model_length}不能被"
            f"U-Net下采样倍数{downsample_factor}整除。"
        )

    ratios = [
        float(data["train_ratio"]),
        float(data["validation_ratio"]),
        float(data["test_ratio"]),
    ]

    if any(value <= 0.0 for value in ratios):
        raise ValueError("训练、验证和测试比例都必须大于0。")

    if abs(sum(ratios) - 1.0) > 1.0e-8:
        raise ValueError(
            "data.train_ratio、validation_ratio和test_ratio之和必须为1。"
        )

    split_unit = str(
        data.get(
            "split_unit",
            "source_file",
        )
    ).strip().lower()

    supported_split_units = {
        "source_file",
        "spectrum",
        "spectrum_within_folder",
    }

    if split_unit not in supported_split_units:
        raise ValueError(
            "data.split_unit只能是source_file、spectrum或"
            "spectrum_within_folder。"
        )

    data["split_unit"] = split_unit

    if split_unit == "spectrum_within_folder" and not bool(
        data.get(
            "recursive",
            False,
        )
    ):
        raise ValueError(
            "使用data.split_unit=spectrum_within_folder时，"
            "必须设置data.recursive: true，"
            "以递归读取data.input_directory下的样品文件夹。"
        )

    if str(
        data.get("length_adaptation", "raman_axis_interpolation")
    ) != "raman_axis_interpolation":
        raise ValueError(
            "data.length_adaptation必须为raman_axis_interpolation。"
        )

    if str(
        data.get("padding_mode", "right_zero_padding")
    ) != "right_zero_padding":
        raise ValueError(
            "data.padding_mode必须为right_zero_padding。"
        )

    raman_range_tolerance = float(
        data.get("raman_range_tolerance", 1.0)
    )

    if raman_range_tolerance < 0.0:
        raise ValueError("data.raman_range_tolerance不能小于0。")


def _validate_normalization_configuration(
    normalization: dict[str, Any],
    diffusion: dict[str, Any],
) -> None:
    """校验训练集归一化和第三方库自动归一化的关系。"""

    if normalization["method"] != "global_minmax":
        raise ValueError(
            "当前项目只支持normalization.method=global_minmax。"
        )

    if normalization["fit_on"] != "train_only":
        raise ValueError(
            "归一化参数只能在训练集拟合，"
            "normalization.fit_on必须为train_only。"
        )

    if (
        float(normalization["target_max"])
        <= float(normalization["target_min"])
    ):
        raise ValueError(
            "normalization.target_max必须大于target_min。"
        )

    if (
        bool(normalization["enabled"])
        and bool(diffusion["auto_normalize"])
    ):
        raise ValueError(
            "启用项目外部global_minmax时，"
            "diffusion.auto_normalize必须为false。"
        )


def _validate_diffusion_configuration(
    diffusion: dict[str, Any],
) -> None:
    """校验扩散步数、预测目标和损失权重。"""

    diffusion_steps = int(
        diffusion.get(
            "diffusion_steps",
            diffusion.get("diffusion_timesteps", 0),
        )
    )
    sampling_steps = int(
        diffusion.get(
            "sampling_steps",
            diffusion.get("sampling_timesteps", 0),
        )
    )

    if diffusion_steps <= 0 or sampling_steps <= 0:
        raise ValueError("扩散步数和采样步数必须大于0。")

    if sampling_steps > diffusion_steps:
        raise ValueError("diffusion.sampling_steps不能大于diffusion_steps。")

    objective = str(diffusion["objective"]).strip().lower()

    if objective not in {"pred_noise", "pred_x0", "pred_v"}:
        raise ValueError(
            "diffusion.objective必须为pred_noise、pred_x0或pred_v。"
        )

    loss_weighting = str(
        diffusion.get("loss_weighting", "library_default")
    ).strip().lower()

    if loss_weighting not in {"library_default", "snr", "uniform"}:
        raise ValueError(
            "diffusion.loss_weighting必须为"
            "library_default、snr或uniform。"
        )

    if loss_weighting == "snr" and objective != "pred_x0":
        raise ValueError(
            "diffusion.loss_weighting=snr目前只允许用于pred_x0。"
        )


def _validate_prior_residual_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """校验 D2 先验残差配置，并返回规范化字典。"""

    prior_residual = configuration.get("prior_residual", {})

    if prior_residual is None:
        prior_residual = {}

    if not isinstance(prior_residual, dict):
        raise TypeError("prior_residual必须是字典。")

    prior_residual["enabled"] = bool(
        prior_residual.get("enabled", False)
    )

    if prior_residual["enabled"]:
        if prior_residual.get(
            "prior_method",
            "training_pointwise_median",
        ) != "training_pointwise_median":
            raise ValueError(
                "当前只支持prior_method=training_pointwise_median。"
            )

        method = str(
            prior_residual.get(
                "residual_normalization",
                "robust_asinh",
            )
        ).strip().lower()

        if method not in {
            "global_maxabs",
            "robust_asinh",
            "pointwise_mad_asinh",
        }:
            raise ValueError(
                "prior_residual.residual_normalization必须为"
                "global_maxabs、robust_asinh或pointwise_mad_asinh。"
            )

        prior_residual["residual_normalization"] = method

    configuration["prior_residual"] = prior_residual

    return prior_residual


def _validate_physics_configuration(
    configuration: dict[str, Any],
    *,
    prior_residual: dict[str, Any],
) -> None:
    """校验 D3.1 与 D2.1、pred_x0 和归一化流程的一致性。"""

    raw_physics = configuration.get(
        "physics_constraints",
        {"enabled": False},
    )

    if raw_physics is None:
        raw_physics = {"enabled": False}

    physics = normalize_physics_configuration(raw_physics)
    configuration["physics_constraints"] = physics

    if not bool(physics.get("enabled", False)):
        return

    if not bool(prior_residual.get("enabled", False)):
        raise ValueError("启用D3.1时必须同时启用prior_residual。")

    if prior_residual.get("residual_normalization") != (
        "pointwise_mad_asinh"
    ):
        raise ValueError(
            "D3.1当前要求"
            "prior_residual.residual_normalization="
            "pointwise_mad_asinh。"
        )

    if not bool(configuration["normalization"]["enabled"]):
        raise ValueError("启用D3.1时必须启用global_minmax归一化。")

    if not bool(
        configuration["normalization"].get(
            "save_in_checkpoint",
            True,
        )
    ):
        raise ValueError(
            "启用D3.1时normalization.save_in_checkpoint必须为true。"
        )

    if str(
        configuration["diffusion"]["objective"]
    ).strip().lower() != "pred_x0":
        raise ValueError("D3.1当前只支持diffusion.objective=pred_x0。")

    if bool(configuration["diffusion"]["auto_normalize"]):
        raise ValueError("启用D3.1时diffusion.auto_normalize必须为false。")


def validate_config(configuration: dict[str, Any]) -> None:
    """执行完整项目配置校验。"""

    _validate_required_sections(configuration)

    data = configuration["data"]
    model = configuration["model"]
    normalization = configuration["normalization"]
    diffusion = configuration["diffusion"]

    _validate_data_configuration(data, model)
    _validate_normalization_configuration(normalization, diffusion)
    _validate_diffusion_configuration(diffusion)

    prior_residual = _validate_prior_residual_configuration(configuration)
    _validate_physics_configuration(
        configuration,
        prior_residual=prior_residual,
    )


def load_configuration(config_path: str | Path) -> dict[str, Any]:
    """项目公开配置读取入口。"""

    return load_config(config_path)
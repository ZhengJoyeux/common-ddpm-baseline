from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _auto_or_positive_integer(
    value: Any,
    field_name: str,
) -> int | None:
    """允许长度设置为正整数或者auto。"""

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
        raise ValueError(
            f"{field_name}必须大于0。"
        )

    return parsed


def resolve_project_path(
    configuration: dict,
    path_value: str | Path,
) -> Path:
    """
    将配置中的相对路径转换为相对于项目根目录的绝对路径。
    绝对路径保持不变。
    """

    del configuration

    path = Path(
        path_value
    ).expanduser()

    if path.is_absolute():
        return path.resolve()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    return (
        project_root / path
    ).resolve()


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """读取并校验YAML配置文件。"""

    path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到配置文件：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "YAML 顶层必须是字典结构。"
        )

    config["_paths"] = {
        "config_path": str(path),
        "project_root": str(
            path.parent.parent
        ),
    }

    validate_config(
        config
    )

    return config


def project_path(
    config: dict[str, Any],
    value: str | Path,
) -> Path:
    """把相对路径转换为项目根目录下的路径。"""

    path = Path(
        value
    ).expanduser()

    if path.is_absolute():
        return path

    return (
        Path(
            config["_paths"][
                "project_root"
            ]
        )
        / path
    )


def validate_config(
    config: dict[str, Any],
) -> None:
    """校验项目配置内容。"""

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

    missing = sorted(
        required_sections.difference(
            config
        )
    )

    if missing:
        raise KeyError(
            f"YAML 缺少配置区段：{missing}"
        )

    data = config["data"]
    model = config["model"]
    normalization = config[
        "normalization"
    ]
    diffusion = config["diffusion"]

    original_length = (
        _auto_or_positive_integer(
            data.get(
                "original_spectrum_length",
                "auto",
            ),
            "original_spectrum_length",
        )
    )

    model_length = (
        _auto_or_positive_integer(
            data.get(
                "model_spectrum_length",
                "auto",
            ),
            "model_spectrum_length",
        )
    )

    if (
        original_length is not None
        and model_length is not None
        and model_length < original_length
    ):
        raise ValueError(
            "model_spectrum_length不能小于"
            "original_spectrum_length。"
        )

    dim_mults = tuple(
        int(value)
        for value in model[
            "dimension_multipliers"
        ]
    )

    if len(dim_mults) < 2:
        raise ValueError(
            "dimension_multipliers "
            "至少需要两个层级。"
        )

    downsample_factor = (
        2 ** (len(dim_mults) - 1)
    )

    if (
        model_length is not None
        and model_length
        % downsample_factor
        != 0
    ):
        raise ValueError(
            f"model_spectrum_length="
            f"{model_length}不能被"
            f"U-Net下采样倍数"
            f"{downsample_factor}整除。"
        )

    ratios = [
        float(
            data["train_ratio"]
        ),
        float(
            data["validation_ratio"]
        ),
        float(
            data["test_ratio"]
        ),
    ]

    if any(
        value <= 0
        for value in ratios
    ):
        raise ValueError(
            "训练、验证、测试比例"
            "都必须大于0。"
        )

    if (
        abs(sum(ratios) - 1.0)
        > 1.0e-8
    ):
        raise ValueError(
            "train_ratio、"
            "validation_ratio、"
            "test_ratio之和必须为1。"
        )

    if data["split_unit"] not in {
        "source_file",
        "sample_folder",
        "spectrum",
    }:
        raise ValueError(
            "split_unit只能是source_file、"
            "sample_folder或spectrum。"
        )

    length_adaptation = str(
        data.get(
            "length_adaptation",
            "raman_axis_interpolation",
        )
    )

    if (
        length_adaptation
        != "raman_axis_interpolation"
    ):
        raise ValueError(
            "自适应长度模式下"
            "length_adaptation必须设置为"
            "raman_axis_interpolation。"
        )

    padding_mode = str(
        data.get(
            "padding_mode",
            "right_zero_padding",
        )
    )

    if (
        padding_mode
        != "right_zero_padding"
    ):
        raise ValueError(
            "padding_mode必须为"
            "right_zero_padding。"
        )

    raman_range_tolerance = float(
        data.get(
            "raman_range_tolerance",
            1.0,
        )
    )

    if raman_range_tolerance < 0.0:
        raise ValueError(
            "raman_range_tolerance"
            "不能小于0。"
        )

    if (
        normalization["method"]
        != "global_minmax"
    ):
        raise ValueError(
            "当前代码只支持"
            "global_minmax。"
        )

    if (
        normalization["fit_on"]
        != "train_only"
    ):
        raise ValueError(
            "归一化参数只能用训练集拟合，"
            "fit_on必须为train_only。"
        )

    if (
        float(
            normalization["target_max"]
        )
        <= float(
            normalization["target_min"]
        )
    ):
        raise ValueError(
            "target_max必须大于"
            "target_min。"
        )

    if (
        bool(
            normalization["enabled"]
        )
        and bool(
            diffusion["auto_normalize"]
        )
    ):
        raise ValueError(
            "已经启用手动全局归一化时，"
            "diffusion.auto_normalize"
            "必须设为false。"
        )


def load_configuration(
    config_path: str | Path,
) -> dict[str, Any]:
    """
    读取并校验YAML配置。

    这是数据检查、模型训练和光谱生成脚本
    共同使用的公开函数。
    """

    return load_config(
        config_path
    )
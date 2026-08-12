"""读取、解析并校验项目 YAML 配置。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from src.feature_peak_residual_limiter import (
    normalize_feature_peak_residual_limiter_configuration,
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


SUPPORTED_PRIOR_METHODS = {
    "training_pointwise_median",
    "pca_reconstruction",
    "training_low_frequency_median",
    "training_blended_frequency_median",
}


SUPPORTED_RESIDUAL_NORMALIZATIONS = {
    "global_maxabs",
    "robust_asinh",
    "pointwise_mad_asinh",
}


LOW_FREQUENCY_PRIOR_METHODS = {
    "training_low_frequency_median",
    "training_blended_frequency_median",
}


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
        raise ValueError(
            f"{field_name}必须大于0。"
        )

    return parsed


def _finite_positive(
    value: Any,
    field_name: str,
) -> float:
    parsed = float(
        value
    )

    if (
        not math.isfinite(parsed)
        or parsed <= 0.0
    ):
        raise ValueError(
            f"{field_name}必须是有限的正数。"
        )

    return parsed


def resolve_project_path(
    configuration: dict,
    path_value: str | Path,
) -> Path:
    """把配置中的相对路径转换为项目根目录下的绝对路径。"""

    path = Path(
        path_value
    ).expanduser()

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

    return (
        project_root
        / path
    ).resolve()


def project_path(
    configuration: dict[str, Any],
    value: str | Path,
) -> Path:
    """保留项目已有公开函数。"""

    return resolve_project_path(
        configuration,
        value,
    )


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """读取 YAML、记录项目根目录并执行完整校验。"""

    path = Path(
        config_path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到配置文件：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        configuration = yaml.safe_load(
            file
        )

    if not isinstance(
        configuration,
        dict,
    ):
        raise ValueError(
            "YAML顶层必须是字典结构。"
        )

    configuration[
        "_paths"
    ] = {
        "config_path": str(
            path
        ),
        "project_root": str(
            path.parent.parent
        ),
    }

    validate_config(
        configuration
    )

    return configuration


def _validate_required_sections(
    configuration: dict[str, Any],
) -> None:
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
            configuration
        )
    )

    if missing:
        raise KeyError(
            f"YAML缺少配置区段：{missing}"
        )

    for section_name in required_sections:
        if not isinstance(
            configuration[
                section_name
            ],
            dict,
        ):
            raise TypeError(
                f"YAML区段{section_name}必须是字典。"
            )


def _validate_data_configuration(
    data: dict[str, Any],
    model: dict[str, Any],
) -> None:
    """校验划分、拉曼轴适配和模型长度。"""

    original_length = (
        _auto_or_positive_integer(
            data.get(
                "original_spectrum_length",
                "auto",
            ),
            "data.original_spectrum_length",
        )
    )

    model_length = (
        _auto_or_positive_integer(
            data.get(
                "model_spectrum_length",
                "auto",
            ),
            "data.model_spectrum_length",
        )
    )

    if (
        original_length is not None
        and model_length is not None
        and model_length
        < original_length
    ):
        raise ValueError(
            "data.model_spectrum_length"
            "不能小于data.original_spectrum_length。"
        )

    dimension_multipliers = tuple(
        int(value)
        for value
        in model[
            "dimension_multipliers"
        ]
    )

    if len(
        dimension_multipliers
    ) < 2:
        raise ValueError(
            "model.dimension_multipliers"
            "至少需要两个层级。"
        )

    if any(
        value <= 0
        for value
        in dimension_multipliers
    ):
        raise ValueError(
            "model.dimension_multipliers"
            "必须全部为正整数。"
        )

    downsample_factor = (
        2
        ** (
            len(
                dimension_multipliers
            )
            - 1
        )
    )

    if (
        model_length is not None
        and model_length
        % downsample_factor
        != 0
    ):
        raise ValueError(
            "data.model_spectrum_length="
            f"{model_length}"
            "不能被U-Net下采样倍数"
            f"{downsample_factor}整除。"
        )

    ratios = [
        float(
            data[
                "train_ratio"
            ]
        ),
        float(
            data[
                "validation_ratio"
            ]
        ),
        float(
            data[
                "test_ratio"
            ]
        ),
    ]

    if any(
        value <= 0.0
        for value in ratios
    ):
        raise ValueError(
            "训练、验证和测试比例都必须大于0。"
        )

    if abs(
        sum(
            ratios
        )
        - 1.0
    ) > 1.0e-8:
        raise ValueError(
            "data.train_ratio、validation_ratio"
            "和test_ratio之和必须为1。"
        )

    split_unit = str(
        data.get(
            "split_unit",
            "source_file",
        )
    ).strip().lower()

    supported_split_units = {
        "source_file",
        "sample_folder",
        "spectrum",
        "spectrum_within_folder",
    }

    if (
        split_unit
        not in supported_split_units
    ):
        raise ValueError(
            "data.split_unit必须为"
            "source_file、sample_folder、"
            "spectrum或spectrum_within_folder。"
        )

    data[
        "split_unit"
    ] = split_unit

    if (
        split_unit
        == "spectrum_within_folder"
        and not bool(
            data.get(
                "recursive",
                False,
            )
        )
    ):
        raise ValueError(
            "使用spectrum_within_folder时"
            "data.recursive必须为true。"
        )

    if str(
        data.get(
            "length_adaptation",
            "raman_axis_interpolation",
        )
    ) != "raman_axis_interpolation":
        raise ValueError(
            "data.length_adaptation必须为"
            "raman_axis_interpolation。"
        )

    if str(
        data.get(
            "padding_mode",
            "right_zero_padding",
        )
    ) != "right_zero_padding":
        raise ValueError(
            "data.padding_mode必须为"
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
            "data.raman_range_tolerance不能小于0。"
        )


def _validate_normalization_configuration(
    normalization: dict[str, Any],
    diffusion: dict[str, Any],
) -> None:
    if (
        normalization[
            "method"
        ]
        != "global_minmax"
    ):
        raise ValueError(
            "当前项目只支持"
            "normalization.method=global_minmax。"
        )

    if (
        normalization[
            "fit_on"
        ]
        != "train_only"
    ):
        raise ValueError(
            "归一化参数只能在训练集拟合，"
            "normalization.fit_on必须为train_only。"
        )

    if (
        float(
            normalization[
                "target_max"
            ]
        )
        <= float(
            normalization[
                "target_min"
            ]
        )
    ):
        raise ValueError(
            "normalization.target_max"
            "必须大于target_min。"
        )

    if (
        bool(
            normalization[
                "enabled"
            ]
        )
        and bool(
            diffusion[
                "auto_normalize"
            ]
        )
    ):
        raise ValueError(
            "启用项目外部global_minmax时，"
            "diffusion.auto_normalize必须为false。"
        )


def _validate_diffusion_configuration(
    diffusion: dict[str, Any],
) -> None:
    diffusion_steps = int(
        diffusion.get(
            "diffusion_steps",
            diffusion.get(
                "diffusion_timesteps",
                0,
            ),
        )
    )

    sampling_steps = int(
        diffusion.get(
            "sampling_steps",
            diffusion.get(
                "sampling_timesteps",
                0,
            ),
        )
    )

    if (
        diffusion_steps <= 0
        or sampling_steps <= 0
    ):
        raise ValueError(
            "扩散步数和采样步数必须大于0。"
        )

    if (
        sampling_steps
        > diffusion_steps
    ):
        raise ValueError(
            "diffusion.sampling_steps"
            "不能大于diffusion.diffusion_steps。"
        )

    objective = str(
        diffusion[
            "objective"
        ]
    ).strip().lower()

    if objective not in {
        "pred_noise",
        "pred_x0",
        "pred_v",
    }:
        raise ValueError(
            "diffusion.objective必须为"
            "pred_noise、pred_x0或pred_v。"
        )

    diffusion[
        "objective"
    ] = objective

    loss_weighting = str(
        diffusion.get(
            "loss_weighting",
            "library_default",
        )
    ).strip().lower()

    if loss_weighting not in {
        "library_default",
        "snr",
        "uniform",
    }:
        raise ValueError(
            "diffusion.loss_weighting必须为"
            "library_default、snr或uniform。"
        )

    if (
        loss_weighting == "snr"
        and objective != "pred_x0"
    ):
        raise ValueError(
            "diffusion.loss_weighting=snr"
            "当前只允许用于pred_x0。"
        )

    diffusion[
        "loss_weighting"
    ] = loss_weighting

    residual_aware = diffusion.get(
        "residual_aware_loss",
        {
            "enabled": False,
        },
    )

    if residual_aware is None:
        residual_aware = {
            "enabled": False,
        }

    if not isinstance(
        residual_aware,
        dict,
    ):
        raise TypeError(
            "diffusion.residual_aware_loss必须是字典。"
        )

    residual_aware[
        "enabled"
    ] = bool(
        residual_aware.get(
            "enabled",
            False,
        )
    )

    diffusion[
        "residual_aware_loss"
    ] = residual_aware


def _validate_prior_residual_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """校验 D2 prior-residual 配置。"""

    prior_residual = configuration.get(
        "prior_residual",
        {},
    )

    if prior_residual is None:
        prior_residual = {}

    if not isinstance(
        prior_residual,
        dict,
    ):
        raise TypeError(
            "prior_residual必须是字典。"
        )

    prior_residual[
        "enabled"
    ] = bool(
        prior_residual.get(
            "enabled",
            False,
        )
    )

    if not prior_residual[
        "enabled"
    ]:
        configuration[
            "prior_residual"
        ] = prior_residual

        return prior_residual

    # ------------------------------------------------------------------
    # prior_method
    # ------------------------------------------------------------------

    prior_method = str(
        prior_residual.get(
            "prior_method",
            "training_pointwise_median",
        )
    ).strip().lower()

    if (
        prior_method
        not in SUPPORTED_PRIOR_METHODS
    ):
        raise ValueError(
            "prior_residual.prior_method必须为："
            "training_pointwise_median、"
            "pca_reconstruction、"
            "training_low_frequency_median或"
            "training_blended_frequency_median。"
        )

    prior_residual[
        "prior_method"
    ] = prior_method

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    if (
        prior_method
        == "pca_reconstruction"
    ):
        explained_ratio = float(
            prior_residual.get(
                "pca_explained_variance_ratio",
                0.95,
            )
        )

        if not (
            0.0
            < explained_ratio
            <= 1.0
        ):
            raise ValueError(
                "prior_residual."
                "pca_explained_variance_ratio"
                "必须在(0,1]范围内。"
            )

        max_components = (
            prior_residual.get(
                "pca_max_components"
            )
        )

        if (
            max_components is not None
            and int(
                max_components
            ) <= 0
        ):
            raise ValueError(
                "prior_residual.pca_max_components"
                "必须为正整数或null。"
            )

        if max_components is not None:
            max_components = int(
                max_components
            )

        sampling_strategy = str(
            prior_residual.get(
                "pca_sampling_strategy",
                "truncated_gaussian_scores",
            )
        ).strip().lower()

        if sampling_strategy not in {
            "truncated_gaussian_scores",
            "independent_truncated_gaussian_scores",
        }:
            raise ValueError(
                "prior_residual.pca_sampling_strategy必须为"
                "truncated_gaussian_scores或"
                "independent_truncated_gaussian_scores。"
            )

        score_clip = _finite_positive(
            prior_residual.get(
                "pca_score_clip_standard_deviations",
                2.5,
            ),
            "prior_residual."
            "pca_score_clip_standard_deviations",
        )

        prior_residual[
            "pca_explained_variance_ratio"
        ] = explained_ratio

        prior_residual[
            "pca_max_components"
        ] = max_components

        prior_residual[
            "pca_sampling_strategy"
        ] = sampling_strategy

        prior_residual[
            "pca_score_clip_standard_deviations"
        ] = score_clip

    # ------------------------------------------------------------------
    # D2.3 / D2.4 low-frequency
    # ------------------------------------------------------------------

    if (
        prior_method
        in LOW_FREQUENCY_PRIOR_METHODS
    ):
        low_frequency = prior_residual.get(
            "low_frequency",
            {},
        )

        if low_frequency is None:
            low_frequency = {}

        if not isinstance(
            low_frequency,
            dict,
        ):
            raise TypeError(
                "prior_residual.low_frequency必须是字典。"
            )

        low_method = str(
            low_frequency.get(
                "method",
                "gaussian",
            )
        ).strip().lower()

        if low_method != "gaussian":
            raise ValueError(
                "prior_residual.low_frequency.method"
                "当前只支持gaussian。"
            )

        sigma_cm1 = _finite_positive(
            low_frequency.get(
                "sigma_cm1",
                40.0,
            ),
            "prior_residual."
            "low_frequency.sigma_cm1",
        )

        truncate = _finite_positive(
            low_frequency.get(
                "truncate",
                4.0,
            ),
            "prior_residual."
            "low_frequency.truncate",
        )

        low_frequency[
            "method"
        ] = low_method

        low_frequency[
            "sigma_cm1"
        ] = sigma_cm1

        low_frequency[
            "truncate"
        ] = truncate

        prior_residual[
            "low_frequency"
        ] = low_frequency

    # ------------------------------------------------------------------
    # D2.4 blended ratio
    # ------------------------------------------------------------------

    if (
        prior_method
        == "training_blended_frequency_median"
    ):
        blended = prior_residual.get(
            "blended_frequency",
            {},
        )

        if blended is None:
            blended = {}

        if not isinstance(
            blended,
            dict,
        ):
            raise TypeError(
                "prior_residual.blended_frequency"
                "必须是字典。"
            )

        peak_component_ratio = float(
            blended.get(
                "peak_component_ratio",
                0.5,
            )
        )

        if (
            not math.isfinite(
                peak_component_ratio
            )
            or not (
                0.0
                < peak_component_ratio
                < 1.0
            )
        ):
            raise ValueError(
                "prior_residual.blended_frequency."
                "peak_component_ratio必须在(0,1)范围内。"
            )

        blended[
            "peak_component_ratio"
        ] = peak_component_ratio

        prior_residual[
            "blended_frequency"
        ] = blended

    # ------------------------------------------------------------------
    # residual normalization
    # ------------------------------------------------------------------

    method = str(
        prior_residual.get(
            "residual_normalization",
            "robust_asinh",
        )
    ).strip().lower()

    if (
        method
        not in SUPPORTED_RESIDUAL_NORMALIZATIONS
    ):
        raise ValueError(
            "prior_residual.residual_normalization必须为"
            "global_maxabs、robust_asinh或"
            "pointwise_mad_asinh。"
        )

    # 这是本轮必须加入的安全检查。
    #
    # 纯 low-frequency / blended prior 的残差中包含
    # 非零的共同峰结构，而 pointwise MAD 只衡量样本间离散。
    # 两者组合已经实验证明会产生严重 inverse amplification。
    if (
        prior_method
        in LOW_FREQUENCY_PRIOR_METHODS
        and method
        == "pointwise_mad_asinh"
    ):
        raise ValueError(
            "training_low_frequency_median或"
            "training_blended_frequency_median"
            "不能与pointwise_mad_asinh组合。"
            "该组合会把共同峰残差除以过小的逐点MAD，"
            "并在inverse_transform时放大为异常负峰。"
            "本轮请使用robust_asinh。"
        )

    prior_residual[
        "residual_normalization"
    ] = method

    residual_quantile = float(
        prior_residual.get(
            "residual_quantile",
            99.5,
        )
    )

    if not (
        0.0
        < residual_quantile
        < 100.0
    ):
        raise ValueError(
            "prior_residual.residual_quantile"
            "必须在(0,100)范围内。"
        )

    prior_residual[
        "residual_quantile"
    ] = residual_quantile

    target_abs_max = float(
        prior_residual.get(
            "target_abs_max",
            1.0,
        )
    )

    if not (
        0.0
        < target_abs_max
        <= 1.0
    ):
        raise ValueError(
            "prior_residual.target_abs_max"
            "必须在(0,1]范围内。"
        )

    prior_residual[
        "target_abs_max"
    ] = target_abs_max

    epsilon = _finite_positive(
        prior_residual.get(
            "epsilon",
            1.0e-8,
        ),
        "prior_residual.epsilon",
    )

    prior_residual[
        "epsilon"
    ] = epsilon

    mad_scale_factor = _finite_positive(
        prior_residual.get(
            "mad_scale_factor",
            1.4826,
        ),
        "prior_residual.mad_scale_factor",
    )

    prior_residual[
        "mad_scale_factor"
    ] = mad_scale_factor

    pointwise_floor_quantile = float(
        prior_residual.get(
            "pointwise_scale_floor_quantile",
            10.0,
        )
    )

    if not (
        0.0
        <= pointwise_floor_quantile
        < 100.0
    ):
        raise ValueError(
            "prior_residual."
            "pointwise_scale_floor_quantile"
            "必须在[0,100)范围内。"
        )

    prior_residual[
        "pointwise_scale_floor_quantile"
    ] = pointwise_floor_quantile

    configuration[
        "prior_residual"
    ] = prior_residual

    return prior_residual


def _validate_d3_configuration(
    configuration: dict[str, Any],
    *,
    prior_residual: dict[str, Any],
) -> None:
    """
    保留已有 D3 配置接口。

    D2.4 本轮全部关闭这些模块。
    """

    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------

    raw_physics = configuration.get(
        "physics_constraints",
        {
            "enabled": False,
        },
    )

    if raw_physics is None:
        raw_physics = {
            "enabled": False,
        }

    if not isinstance(
        raw_physics,
        dict,
    ):
        raise TypeError(
            "physics_constraints必须是字典。"
        )

    if (
        "distribution_preservation"
        in raw_physics
    ):
        raise ValueError(
            "当前版本不再支持"
            "physics_constraints.distribution_preservation；"
            "请使用顶层diversity_constraints。"
        )

    physics = (
        normalize_physics_configuration(
            raw_physics
        )
    )

    configuration[
        "physics_constraints"
    ] = physics

    # ------------------------------------------------------------------
    # diversity
    # ------------------------------------------------------------------

    raw_diversity = configuration.get(
        "diversity_constraints",
        {
            "enabled": False,
        },
    )

    if raw_diversity is None:
        raw_diversity = {
            "enabled": False,
        }

    diversity = (
        normalize_diversity_configuration(
            raw_diversity
        )
    )

    configuration[
        "diversity_constraints"
    ] = diversity

    # ------------------------------------------------------------------
    # feature limiter
    # ------------------------------------------------------------------

    raw_limiter = configuration.get(
        "feature_peak_residual_limiter",
        {
            "enabled": False,
        },
    )

    if raw_limiter is None:
        raw_limiter = {
            "enabled": False,
        }

    limiter = (
        normalize_feature_peak_residual_limiter_configuration(
            raw_limiter
        )
    )

    configuration[
        "feature_peak_residual_limiter"
    ] = limiter

    # ------------------------------------------------------------------
    # local peak distribution
    # ------------------------------------------------------------------

    raw_local = configuration.get(
        "local_peak_distribution_constraints",
        {
            "enabled": False,
        },
    )

    if raw_local is None:
        raw_local = {
            "enabled": False,
        }

    local_peak = (
        normalize_local_peak_distribution_configuration(
            raw_local
        )
    )

    configuration[
        "local_peak_distribution_constraints"
    ] = local_peak

    residual_aware = configuration[
        "diffusion"
    ].get(
        "residual_aware_loss",
        {
            "enabled": False,
        },
    )

    physics_enabled = bool(
        physics.get(
            "enabled",
            False,
        )
    )

    diversity_enabled = bool(
        diversity.get(
            "enabled",
            False,
        )
    )

    limiter_enabled = bool(
        limiter.get(
            "enabled",
            False,
        )
    )

    local_enabled = bool(
        local_peak.get(
            "enabled",
            False,
        )
    )

    residual_aware_enabled = bool(
        residual_aware.get(
            "enabled",
            False,
        )
    )

    any_d3 = any(
        (
            physics_enabled,
            diversity_enabled,
            limiter_enabled,
            local_enabled,
            residual_aware_enabled,
        )
    )

    if not any_d3:
        return

    if not bool(
        prior_residual.get(
            "enabled",
            False,
        )
    ):
        raise ValueError(
            "启用D3/residual-aware模块时"
            "必须同时启用prior_residual。"
        )

    # 当前 D3 物理模块的可微 inverse 明确依赖
    # pointwise_mad_asinh checkpoint 字段。
    if (
        prior_residual.get(
            "residual_normalization"
        )
        != "pointwise_mad_asinh"
    ):
        raise ValueError(
            "当前D3物理/局部峰/residual-aware代码"
            "要求prior_residual.residual_normalization="
            "pointwise_mad_asinh。"
            "D2.4 blended-frequency + robust_asinh"
            "本轮必须关闭全部D3额外损失。"
        )

    if (
        physics_enabled
        or limiter_enabled
        or local_enabled
        or residual_aware_enabled
    ):
        if (
            prior_residual.get(
                "prior_method"
            )
            != "training_pointwise_median"
        ):
            raise ValueError(
                "当前D3物理/limiter/local-peak/residual-aware"
                "实现仍要求prior_method="
                "training_pointwise_median。"
                "D2.4实验请关闭这些模块。"
            )

    if not bool(
        configuration[
            "normalization"
        ][
            "enabled"
        ]
    ):
        raise ValueError(
            "启用D3时必须启用global_minmax。"
        )

    if not bool(
        configuration[
            "normalization"
        ].get(
            "save_in_checkpoint",
            True,
        )
    ):
        raise ValueError(
            "启用D3时normalization."
            "save_in_checkpoint必须为true。"
        )

    if str(
        configuration[
            "diffusion"
        ][
            "objective"
        ]
    ).strip().lower() != "pred_x0":
        raise ValueError(
            "D3/residual-aware当前只支持pred_x0。"
        )

    if bool(
        configuration[
            "diffusion"
        ][
            "auto_normalize"
        ]
    ):
        raise ValueError(
            "启用D3时diffusion.auto_normalize"
            "必须为false。"
        )

    if (
        local_enabled
        and not physics_enabled
    ):
        raise ValueError(
            "local_peak_distribution_constraints"
            "依赖physics_constraint_state，"
            "因此必须同时启用physics_constraints。"
        )


def validate_config(
    configuration: dict[str, Any],
) -> None:
    """执行完整项目配置校验。"""

    _validate_required_sections(
        configuration
    )

    data = configuration[
        "data"
    ]

    model = configuration[
        "model"
    ]

    normalization = configuration[
        "normalization"
    ]

    diffusion = configuration[
        "diffusion"
    ]

    _validate_data_configuration(
        data,
        model,
    )

    _validate_normalization_configuration(
        normalization,
        diffusion,
    )

    _validate_diffusion_configuration(
        diffusion
    )

    prior_residual = (
        _validate_prior_residual_configuration(
            configuration
        )
    )

    _validate_d3_configuration(
        configuration,
        prior_residual=prior_residual,
    )


def load_configuration(
    config_path: str | Path,
) -> dict[str, Any]:
    """项目公开配置入口。"""

    return load_config(
        config_path
    )
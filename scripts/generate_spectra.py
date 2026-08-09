"""Generate new spectra from a trained unconditional DDPM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from typing import Any

from src.checkpoint_manager import (
    load_checkpoint_file,
    resolve_label_axis,
)
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.feature_peak_residual_limiter import (
    FeaturePeakResidualLimiter,
)
from src.intensity_normalizer import (
    GlobalMinMaxNormalizer,
)
from src.model_builder import (
    build_diffusion_model,
)
from src.prior_residual import (
    PriorResidualTransformer,
)
from src.random_seed_manager import (
    set_random_seed,
)
from src.spectrum_exporter import (
    export_generated_spectra,
)
from src.spectrum_generator import (
    generate_spectra,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)


def parse_arguments() -> argparse.Namespace:
    """读取生成光谱所需的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用训练好的一维无条件DDPM生成SERS光谱。"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML配置文件路径。",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "检查点路径；未提供时使用配置中"
            "checkpoint_directory下的latest.pt。"
        ),
    )

    parser.add_argument(
        "--label",
        default=None,
        help=(
            "指定输出光谱使用的文件夹标签。"
            "如果检查点只有一个标签，可以省略；"
            "如果包含多个标签，则必须指定。"
        ),
    )

    parser.add_argument(
        "--number",
        type=int,
        default=None,
        help="覆盖配置中的生成光谱数量。",
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help="自定义输出文件基础名称。",
    )

    parser.add_argument(
        "--model-source",
        choices=(
            "raw",
            "ema",
        ),
        default=None,
        help=(
            "选择生成时使用的模型："
            "raw为普通训练模型，ema为EMA模型。"
            "默认读取generation.model_source；"
            "若配置中未设置，则默认使用raw。"
        ),
    )

    return parser.parse_args()


def resolve_device(
    device_text: str,
) -> torch.device:
    """读取并检查生成光谱使用的设备。"""

    normalized_device = str(
        device_text
    ).strip().lower()

    if normalized_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "配置要求使用CUDA，"
                "但PyTorch未检测到可用GPU。"
            )

    return torch.device(
        normalized_device
    )


def resolve_checkpoint_path(
    *,
    configuration: dict,
    checkpoint_argument: str | None,
) -> Path:
    """确定实际使用的检查点路径。"""

    output_config = configuration["output"]

    if checkpoint_argument is None:
        return resolve_project_path(
            configuration,
            Path(
                output_config[
                    "checkpoint_directory"
                ]
            )
            / "latest.pt",
        )

    checkpoint_path = Path(
        checkpoint_argument
    ).expanduser()

    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(
            configuration,
            checkpoint_path,
        )

    return checkpoint_path


def resolve_requested_model_source(
    *,
    arguments: argparse.Namespace,
    generation_config: dict,
) -> str:
    """确定使用普通模型还是EMA模型。"""

    if arguments.model_source is not None:
        model_source = arguments.model_source
    else:
        model_source = str(
            generation_config.get(
                "model_source",
                "raw",
            )
        ).strip().lower()

    if model_source not in {
        "raw",
        "ema",
    }:
        raise ValueError(
            "generation.model_source只能是"
            "raw或ema，"
            f"实际为{model_source!r}。"
        )

    return model_source


def build_safe_label_name(
    label: str,
) -> str:
    """将标签转换为适合文件名使用的文本。"""

    safe_characters = []

    for character in str(label):
        if (
            character.isalnum()
            or character in {
                "-",
                "_",
                ".",
            }
        ):
            safe_characters.append(
                character
            )
        else:
            safe_characters.append(
                "_"
            )

    safe_name = "".join(
        safe_characters
    ).strip("._")

    if not safe_name:
        safe_name = "label"

    return safe_name


def configure_checkpoint_diversity_constraints(
    diffusion: torch.nn.Module,
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    """按checkpoint状态注册D3.4多样性约束模块。"""

    diversity_configuration = checkpoint_configuration.get(
        "diversity_constraints",
        {},
    )

    diversity_enabled_in_configuration = (
        isinstance(diversity_configuration, dict)
        and bool(diversity_configuration.get("enabled", False))
    )

    diversity_constraint_state = metadata.get(
        "diversity_constraint_state"
    )

    # D0–D3.3：既没有配置，也没有保存状态，直接正常跳过。
    if diversity_constraint_state is None:
        if diversity_enabled_in_configuration:
            raise KeyError(
                "checkpoint配置启用了D3.4多样性约束，"
                "但metadata中缺少diversity_constraint_state。"
            )

        return False

    if not isinstance(diversity_constraint_state, dict):
        raise TypeError(
            "checkpoint metadata中的diversity_constraint_state"
            "必须是字典。"
        )

    diversity_enabled_in_state = bool(
        diversity_constraint_state.get("enabled", False)
    )

    if diversity_enabled_in_state != diversity_enabled_in_configuration:
        raise ValueError(
            "checkpoint配置与metadata中的D3.4多样性约束启用状态"
            "不一致，无法安全加载checkpoint。"
        )

    if not diversity_enabled_in_state:
        return False

    configure_diversity = getattr(
        diffusion,
        "configure_diversity_constraints",
        None,
    )

    if not callable(configure_diversity):
        raise RuntimeError(
            "当前扩散模型缺少configure_diversity_constraints，"
            "无法加载D3.4 checkpoint。"
        )

    configure_diversity(
        diversity_constraint_state=diversity_constraint_state,
    )

    return True


def load_checkpoint_feature_peak_residual_limiter(
    *,
    checkpoint_configuration: dict[str, Any],
    metadata: dict[str, Any],
) -> FeaturePeakResidualLimiter | None:
    """按checkpoint中的D3.5状态恢复生成端峰区残差软上限。"""

    limiter_configuration = checkpoint_configuration.get(
        "feature_peak_residual_limiter",
        {"enabled": False},
    )

    if limiter_configuration is None:
        limiter_configuration = {"enabled": False}

    if not isinstance(limiter_configuration, dict):
        raise TypeError(
            "checkpoint中的feature_peak_residual_limiter必须是字典。"
        )

    enabled_in_configuration = bool(
        limiter_configuration.get("enabled", False)
    )
    limiter_state = metadata.get(
        "feature_peak_residual_limiter_state"
    )

    # D0–D3.4旧检查点既没有该配置也没有该状态，保持可用。
    if limiter_state is None:
        if enabled_in_configuration:
            raise KeyError(
                "checkpoint配置启用了D3.5特征峰残差软上限，"
                "但metadata中缺少feature_peak_residual_limiter_state。"
            )
        return None

    if not isinstance(limiter_state, dict):
        raise TypeError(
            "checkpoint metadata中的feature_peak_residual_limiter_state"
            "必须是字典。"
        )

    enabled_in_state = bool(limiter_state.get("enabled", False))
    if enabled_in_state != enabled_in_configuration:
        raise ValueError(
            "checkpoint配置与metadata中的D3.5特征峰残差软上限启用状态"
            "不一致，无法安全生成。"
        )

    if not enabled_in_state:
        return None

    return FeaturePeakResidualLimiter.from_state_dict(limiter_state)

def main() -> None:
    """执行完整的光谱生成和导出流程。"""

    arguments = parse_arguments()

    configuration = load_configuration(
        arguments.config
    )

    project_config = configuration["project"]
    training_config = configuration["training"]
    generation_config = configuration["generation"]
    output_config = configuration["output"]

    random_config = configuration.get(
        "random",
        {},
    )

    random_seed = int(
        random_config.get(
            "seed",
            project_config.get(
                "random_seed",
                42,
            ),
        )
    )

    set_random_seed(
        random_seed=random_seed,
        deterministic=bool(
            random_config.get(
                "deterministic",
                False,
            )
        ),
    )

    checkpoint_path = resolve_checkpoint_path(
        configuration=configuration,
        checkpoint_argument=(
            arguments.checkpoint
        ),
    )

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    metadata = checkpoint.get(
        "metadata"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise RuntimeError(
            "检查点中没有有效的metadata。"
        )

    # 根据文件夹标签选择最终输出的原始位移轴。
    (
        resolved_label,
        output_raman_shifts,
        profile_id,
    ) = resolve_label_axis(
        metadata=metadata,
        label=arguments.label,
    )

    output_raman_shifts = np.asarray(
        output_raman_shifts,
        dtype=np.float64,
    ).reshape(-1)

    if output_raman_shifts.size < 2:
        raise RuntimeError(
            "检查点中的输出拉曼位移轴无效。"
        )

    if not np.isfinite(
        output_raman_shifts
    ).all():
        raise RuntimeError(
            "输出拉曼位移轴包含NaN或无穷值。"
        )

    if not np.all(
        np.diff(output_raman_shifts) > 0.0
    ):
        raise RuntimeError(
            "输出拉曼位移轴必须严格递增。"
        )

    # 从检查点元数据恢复训练时使用的
    # 统一位移轴、补齐长度和插值信息。
    length_adapter = (
        SpectrumLengthAdapter.from_metadata(
            metadata
        )
    )

    checkpoint_configuration = checkpoint.get(
        "configuration"
    )

    if not isinstance(
        checkpoint_configuration,
        dict,
    ):
        raise RuntimeError(
            "检查点中没有有效的configuration，"
            "无法确定训练时的模型结构。"
        )

    # 根据检查点训练配置判断该模型是否启用了D2。
    # 必须读取检查点内的配置，不能使用当前YAML判断，
    # 因为当前YAML可能已经被修改。
    prior_residual_config = (
        checkpoint_configuration.get(
            "prior_residual",
            {},
        )
    )

    if prior_residual_config is None:
        prior_residual_config = {}

    if not isinstance(
        prior_residual_config,
        dict,
    ):
        raise RuntimeError(
            "检查点中的prior_residual配置无效。"
        )

    prior_residual_enabled = bool(
        prior_residual_config.get(
            "enabled",
            False,
        )
    )

    prior_residual_transformer = None

    if prior_residual_enabled:
        prior_residual_state = metadata.get(
            "prior_residual_state"
        )

        if not isinstance(
            prior_residual_state,
            dict,
        ):
            raise RuntimeError(
                "该检查点启用了D2先验残差模型，"
                "但metadata中没有有效的"
                "prior_residual_state。"
                "请确认该检查点由完整的D2训练流程生成。"
            )

        try:
            prior_residual_transformer = (
                PriorResidualTransformer
                .from_state_dict(
                    prior_residual_state
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            raise RuntimeError(
                "检查点中的prior_residual_state"
                "无法恢复，可能缺少字段或内容损坏。"
            ) from error

    # D0和D1旧检查点没有prior_residual配置，
    # 此时prior_residual_transformer保持为None。

    # 必须传入检查点中的完整配置，
    # 使model和diffusion两个区段同时生效。
    _, diffusion = build_diffusion_model(
    model_configuration=(
        checkpoint_configuration
    ),
    sequence_length=(
        length_adapter.padded_length
    ),
)

    configure_checkpoint_diversity_constraints(
    diffusion=diffusion,
    checkpoint_configuration=checkpoint_configuration,
    metadata=metadata,
)

    feature_peak_residual_limiter = (
        load_checkpoint_feature_peak_residual_limiter(
            checkpoint_configuration=checkpoint_configuration,
            metadata=metadata,
        )
    )

    requested_model_source = (
        resolve_requested_model_source(
            arguments=arguments,
            generation_config=generation_config,
        )
    )

    if requested_model_source == "raw":
        diffusion_state = checkpoint.get(
            "diffusion_state"
        )

        if not isinstance(
            diffusion_state,
            dict,
        ):
            raise RuntimeError(
                "检查点中没有有效的diffusion_state。"
            )

        diffusion.load_state_dict(
            diffusion_state
        )

        model_source_text = "普通训练模型"

    else:
        ema_state = checkpoint.get(
            "ema_state"
        )

        if (
            not isinstance(
                ema_state,
                dict,
            )
            or not isinstance(
                ema_state.get("ema_model"),
                dict,
            )
        ):
            raise RuntimeError(
                "指定了EMA模型，但检查点中没有"
                "有效的ema_state['ema_model']。"
            )

        diffusion.load_state_dict(
            ema_state["ema_model"]
        )

        model_source_text = "EMA模型"

    device = resolve_device(
        str(
            training_config["device"]
        )
    )

    if arguments.number is None:
        number_of_spectra = int(
            generation_config[
                "number_of_spectra"
            ]
        )
    else:
        number_of_spectra = int(
            arguments.number
        )

    if number_of_spectra <= 0:
        raise ValueError(
            "生成光谱数量必须大于0。"
        )

    generation_batch_size = int(
        generation_config["batch_size"]
    )

    if generation_batch_size <= 0:
        raise ValueError(
            "generation.batch_size必须大于0。"
        )

    # 生成器先删除模型末尾补齐点。
    # D2检查点随后在统一训练轴上恢复完整归一化光谱，
    # 最后插值回当前标签的原始位移轴。
    spectra = generate_spectra(
        diffusion=diffusion,
        number_of_spectra=number_of_spectra,
        generation_batch_size=(
            generation_batch_size
        ),
        device=device,
        length_adapter=length_adapter,
        output_raman_shifts=(
            output_raman_shifts
        ),
        prior_residual_transformer=(
            prior_residual_transformer
        ),
        feature_peak_residual_limiter=(
            feature_peak_residual_limiter
        ),
        prior_random_seed=random_seed,
    )

    inverse_normalizer = None

    if bool(
        generation_config.get(
            "inverse_normalize",
            False,
        )
    ):
        normalization_state = metadata.get(
            "normalization_state"
        )

        if not isinstance(
            normalization_state,
            dict,
        ):
            raise RuntimeError(
                "检查点中没有normalization_state。"
                "不能将生成结果恢复到原始强度尺度；"
                "请使用修改自适应流程后重新训练的检查点。"
            )

        inverse_normalizer = (
            GlobalMinMaxNormalizer.from_state_dict(
                normalization_state
            )
        )

        spectra = (
            inverse_normalizer.inverse_transform(
                spectra
            )
        )

    spectra = np.asarray(
        spectra,
        dtype=np.float32,
    )

    if spectra.ndim != 2:
        raise RuntimeError(
            "最终生成光谱必须为二维数组"
            "[光谱数量, 光谱点数]。"
        )

    if spectra.shape[0] != number_of_spectra:
        raise RuntimeError(
            "实际生成光谱数量与请求数量不一致。"
        )

    if (
        spectra.shape[1]
        != output_raman_shifts.size
    ):
        raise RuntimeError(
            "最终生成光谱长度与输出位移轴不一致："
            f"光谱长度为{spectra.shape[1]}，"
            f"位移轴长度为"
            f"{output_raman_shifts.size}。"
        )

    if not np.isfinite(
        spectra
    ).all():
        raise RuntimeError(
            "最终生成结果包含NaN或无穷值。"
        )

    checkpoint_step = int(
        checkpoint.get(
            "step",
            0,
        )
    )

    if arguments.output_name:
        base_name = str(
            arguments.output_name
        ).strip()

        if not base_name:
            raise ValueError(
                "output-name不能为空。"
            )
    else:
        safe_label = build_safe_label_name(
            resolved_label
        )

        base_name = (
            f"generated_{safe_label}_"
            f"step_{checkpoint_step:08d}"
        )

    spectrum_output_directory = (
        resolve_project_path(
            configuration,
            output_config[
                "generated_spectrum_directory"
            ],
        )
    )

    plot_output_directory = (
        resolve_project_path(
            configuration,
            output_config[
                "preview_plot_directory"
            ],
        )
    )

    exported_paths = export_generated_spectra(
        raman_shift=output_raman_shifts,
        spectra=spectra,
        spectrum_output_directory=(
            spectrum_output_directory
        ),
        plot_output_directory=(
            plot_output_directory
        ),
        base_name=base_name,
        output_formats=list(
            generation_config[
                "output_formats"
            ]
        ),
    )

    print("\n===== 光谱生成完成 =====")
    print(f"检查点：{checkpoint_path}")
    print(f"检查点步数：{checkpoint_step}")
    print(f"使用模型：{model_source_text}")
    print(f"输出标签：{resolved_label}")
    print(f"位移轴配置ID：{profile_id}")
    print(
        "输出位移范围："
        f"{output_raman_shifts[0]:.6g}–"
        f"{output_raman_shifts[-1]:.6g} cm⁻¹"
    )
    print(
        f"输出位移点数："
        f"{output_raman_shifts.size}"
    )
    print(
        f"统一训练轴点数："
        f"{length_adapter.original_length}"
    )
    print(
        f"模型输入点数："
        f"{length_adapter.padded_length}"
    )
    print(
        f"生成数量："
        f"{spectra.shape[0]}"
    )
    if feature_peak_residual_limiter is not None:
        limiter_summary = feature_peak_residual_limiter.summary()
        print(
            "D3.5峰区残差软上限：已启用；"
            f"峰区点数{limiter_summary['feature_peak_point_count']}；"
            "残差上限倍数"
            f"{limiter_summary['limit_multiplier']:.3f}"
        )
    else:
        print("D3.5峰区残差软上限：未启用")
    print(
        "生成强度范围："
        f"{float(spectra.min()):.6g}–"
        f"{float(spectra.max()):.6g}"
    )

    if inverse_normalizer is not None:
        print(
            "生成强度：已恢复到原始强度尺度"
        )
    else:
        print(
            "生成强度：保持模型归一化尺度"
        )

    print("输出文件：")

    for path in exported_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
"""Apply the Experiment-C training-set guard to generated SERS spectra."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.checkpoint_manager import load_checkpoint_file
from src.configuration_loader import (
    load_configuration,
    resolve_project_path,
)
from src.sers_training_guard import (
    SersTrainingGuard,
    fit_sers_training_guard_state,
)
from src.spectrum_file_reader import read_spectrum_collection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用checkpoint记录的训练集索引自动拟合峰位/峰高/FWHM/"
            "负谷软边界，并筛选已生成的SERS光谱。"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args()


def _read_generated(
    path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("generated只支持xlsx/xls/csv。")

    if frame.shape[1] < 2:
        raise ValueError("生成文件至少需要Raman shift列和1条光谱列。")

    axis = pd.to_numeric(
        frame.iloc[:, 0],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(axis).all():
        raise ValueError("生成文件第一列Raman shift包含非数值。")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("生成文件Raman shift必须严格递增。")

    intensity_frame = frame.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce",
    )
    matrix = intensity_frame.to_numpy(dtype=np.float64).T

    if not np.isfinite(matrix).all():
        raise ValueError("生成光谱强度包含非数值/NaN/无穷值。")

    names = [str(value) for value in frame.columns[1:]]
    return frame, axis, matrix, names


def _normalize_path_text(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _load_training_on_output_axis(
    *,
    configuration: dict,
    checkpoint: dict,
    output_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("checkpoint缺少有效metadata。")

    checkpoint_configuration = checkpoint.get("configuration")
    if not isinstance(checkpoint_configuration, dict):
        raise RuntimeError("checkpoint缺少有效configuration。")

    data_config = checkpoint_configuration.get("data")
    if not isinstance(data_config, dict):
        raise RuntimeError("checkpoint缺少有效data配置。")

    input_directory = resolve_project_path(
        configuration,
        data_config["input_directory"],
    )

    collection = read_spectrum_collection(
        input_directory=input_directory,
        data_config=data_config,
    )

    expected_count = int(
        metadata.get(
            "number_of_spectra",
            len(collection.spectrum_names),
        )
    )
    if len(collection.spectrum_names) != expected_count:
        raise RuntimeError(
            "当前训练数据数量与checkpoint记录不一致："
            f"{len(collection.spectrum_names)} != {expected_count}。"
        )

    checkpoint_relative = metadata.get("relative_source_files")
    if isinstance(checkpoint_relative, list):
        current_relative = [
            _normalize_path_text(value)
            for value in collection.relative_source_files
        ]
        expected_relative = [
            _normalize_path_text(value)
            for value in checkpoint_relative
        ]
        if current_relative != expected_relative:
            raise RuntimeError(
                "当前data/input中的源文件顺序与checkpoint不一致；"
                "为避免错误使用training_indices，停止筛选。"
            )

    training_indices = np.asarray(
        metadata.get("training_indices"),
        dtype=np.int64,
    ).reshape(-1)

    if training_indices.size == 0:
        raise RuntimeError("checkpoint没有training_indices。")
    if (
        np.any(training_indices < 0)
        or np.any(training_indices >= expected_count)
    ):
        raise RuntimeError("checkpoint training_indices越界。")

    training_rows: list[np.ndarray] = []

    tolerance = float(
        data_config.get("raman_range_tolerance", 1.0)
    )

    for index in training_indices:
        source_axis = np.asarray(
            collection.raman_shifts[int(index)],
            dtype=np.float64,
        ).reshape(-1)
        source_spectrum = np.asarray(
            collection.spectra[int(index)],
            dtype=np.float64,
        ).reshape(-1)

        if source_axis.size != source_spectrum.size:
            raise RuntimeError("训练光谱与Raman shift长度不一致。")
        if not np.all(np.diff(source_axis) > 0.0):
            raise RuntimeError("训练Raman shift轴不是严格递增。")

        if (
            output_axis[0] < source_axis[0] - tolerance
            or output_axis[-1] > source_axis[-1] + tolerance
        ):
            raise RuntimeError(
                "生成输出轴超出训练光谱覆盖范围，停止盲目插值。"
            )

        training_rows.append(
            np.interp(
                output_axis,
                source_axis,
                source_spectrum,
            )
        )

    training = np.stack(training_rows, axis=0)

    if not np.isfinite(training).all():
        raise RuntimeError("插值后的训练光谱包含NaN或无穷值。")

    return training, training_indices


def _guard_configuration(configuration: dict) -> dict[str, Any]:
    generation = configuration.get("generation", {})
    if not isinstance(generation, dict):
        raise TypeError("generation配置必须是字典。")

    guard = generation.get("training_guard", {}) or {}
    if not isinstance(guard, dict):
        raise TypeError("generation.training_guard必须是字典。")
    if not bool(guard.get("enabled", False)):
        raise RuntimeError(
            "generation.training_guard.enabled=false；"
            "本实验不会在未显式启用时筛选光谱。"
        )
    return guard


def _write_filtered_frame(
    *,
    source_frame: pd.DataFrame,
    mask: np.ndarray,
    xlsx_path: Path,
    csv_path: Path,
) -> None:
    selected_positions = [
        0,
        *[
            column_index + 1
            for column_index, keep in enumerate(mask)
            if bool(keep)
        ],
    ]
    selected = source_frame.iloc[:, selected_positions].copy()
    selected.to_excel(xlsx_path, index=False)
    selected.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_arguments()

    configuration = load_configuration(args.config)
    guard_config = _guard_configuration(configuration)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(
            configuration,
            checkpoint_path,
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    generated_path = Path(args.generated).expanduser()
    if not generated_path.is_absolute():
        generated_path = resolve_project_path(
            configuration,
            generated_path,
        )
    if not generated_path.is_file():
        raise FileNotFoundError(generated_path)

    output_directory = Path(args.output_directory).expanduser()
    if not output_directory.is_absolute():
        output_directory = resolve_project_path(
            configuration,
            output_directory,
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint_file(
        checkpoint_path,
        map_location="cpu",
    )

    (
        generated_frame,
        output_axis,
        generated_spectra,
        generated_names,
    ) = _read_generated(generated_path)

    training_spectra, training_indices = (
        _load_training_on_output_axis(
            configuration=configuration,
            checkpoint=checkpoint,
            output_axis=output_axis,
        )
    )

    state = fit_sers_training_guard_state(
        training_spectra=training_spectra,
        raman_shift=output_axis,
        configuration=guard_config,
    )

    guard = SersTrainingGuard(
        raman_shift=output_axis,
        state=state,
        configuration=guard_config,
    )

    accepted, spectrum_rows, peak_rows = guard.evaluate(
        generated_spectra,
        spectrum_names=generated_names,
    )

    spectrum_frame = pd.DataFrame(spectrum_rows)
    peak_frame = pd.DataFrame(peak_rows)
    bounds_frame = pd.DataFrame(state["peaks"])

    spectrum_frame.to_csv(
        output_directory / "generated_guard_summary.csv",
        index=False,
    )
    peak_frame.to_csv(
        output_directory / "generated_peak_metrics_long.csv",
        index=False,
    )
    bounds_frame.to_csv(
        output_directory / "training_guard_peak_bounds.csv",
        index=False,
    )

    _write_filtered_frame(
        source_frame=generated_frame,
        mask=accepted,
        xlsx_path=(
            output_directory
            / "accepted_generated_spectra.xlsx"
        ),
        csv_path=(
            output_directory
            / "accepted_generated_spectra.csv"
        ),
    )
    _write_filtered_frame(
        source_frame=generated_frame,
        mask=~accepted,
        xlsx_path=(
            output_directory
            / "rejected_generated_spectra.xlsx"
        ),
        csv_path=(
            output_directory
            / "rejected_generated_spectra.csv"
        ),
    )

    accepted_count = int(np.count_nonzero(accepted))
    rejected_count = int(accepted.size - accepted_count)
    acceptance_rate = 100.0 * accepted_count / accepted.size

    summary_lines = [
        "===== SERS training guard =====",
        f"checkpoint: {checkpoint_path}",
        f"generated: {generated_path}",
        f"training spectra: {training_spectra.shape[0]}",
        f"training indices: {training_indices.tolist()}",
        f"generated spectra: {generated_spectra.shape[0]}",
        f"automatically fitted guard peaks: {len(state['peaks'])}",
        (
            "negative minimum lower bound: "
            f"{state['negative_minimum_lower_bound']:.8g}"
        ),
        f"accepted: {accepted_count}",
        f"rejected: {rejected_count}",
        f"acceptance rate: {acceptance_rate:.2f}%",
        "",
        "说明：",
        "1. 峰位全部由当前训练集自动检测，不写死农药峰位。",
        "2. 本脚本只做接受/拒绝，不修改任何生成光谱强度。",
        "3. broad calibration、Raman jitter、variation_scale均不在本脚本中执行。",
        "4. 如果拒绝率很高，说明模型分布仍有问题，不能靠筛选掩盖。",
    ]

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    (output_directory / "guard_summary.txt").write_text(
        summary_text + "\n",
        encoding="utf-8",
    )

    print()
    print("输出：")
    for name in (
        "training_guard_peak_bounds.csv",
        "generated_guard_summary.csv",
        "generated_peak_metrics_long.csv",
        "accepted_generated_spectra.xlsx",
        "accepted_generated_spectra.csv",
        "rejected_generated_spectra.xlsx",
        "rejected_generated_spectra.csv",
        "guard_summary.txt",
    ):
        print("  -", output_directory / name)


if __name__ == "__main__":
    main()

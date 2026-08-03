"""Export generated spectra to Excel, CSV, NPZ and PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUPPORTED_OUTPUT_FORMATS = {
    "xlsx",
    "csv",
    "npz",
    "png",
}


def _validate_export_arrays(
    raman_shift: np.ndarray,
    spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """检查并统一导出使用的拉曼位移轴和光谱数组。"""

    shift = np.asarray(
        raman_shift,
        dtype=np.float64,
    ).reshape(-1)

    spectrum_array = np.asarray(
        spectra,
        dtype=np.float32,
    )

    if shift.size < 2:
        raise ValueError(
            "raman_shift至少需要包含两个点。"
        )

    if not np.isfinite(shift).all():
        raise ValueError(
            "raman_shift包含NaN或无穷值。"
        )

    if not np.all(
        np.diff(shift) > 0.0
    ):
        raise ValueError(
            "raman_shift必须严格递增。"
        )

    if spectrum_array.ndim != 2:
        raise ValueError(
            "spectra必须为二维数组"
            "[光谱数量, 光谱点数]。"
        )

    if spectrum_array.shape[0] == 0:
        raise ValueError(
            "spectra中没有可导出的光谱。"
        )

    if spectrum_array.shape[1] != shift.size:
        raise ValueError(
            "光谱长度与Raman shift长度不一致："
            f"光谱长度为{spectrum_array.shape[1]}，"
            f"位移轴长度为{shift.size}。"
        )

    if not np.isfinite(
        spectrum_array
    ).all():
        raise ValueError(
            "spectra包含NaN或无穷值。"
        )

    return (
        shift,
        spectrum_array,
    )


def _normalize_output_formats(
    output_formats: Iterable[str],
) -> set[str]:
    """规范并检查输出格式。"""

    normalized_formats = {
        str(value).strip().lower().lstrip(".")
        for value in output_formats
        if str(value).strip()
    }

    if not normalized_formats:
        raise ValueError(
            "output_formats不能为空。"
        )

    unsupported_formats = (
        normalized_formats
        - SUPPORTED_OUTPUT_FORMATS
    )

    if unsupported_formats:
        unsupported_text = "、".join(
            sorted(unsupported_formats)
        )

        supported_text = "、".join(
            sorted(SUPPORTED_OUTPUT_FORMATS)
        )

        raise ValueError(
            f"不支持的输出格式：{unsupported_text}。"
            f"当前支持：{supported_text}。"
        )

    return normalized_formats


def build_spectrum_dataframe(
    raman_shift: np.ndarray,
    spectra: np.ndarray,
) -> pd.DataFrame:
    """
    建立宽格式拉曼光谱表格。

    第一列为拉曼位移，后续每一列为一条生成光谱。
    """

    shift, spectrum_array = (
        _validate_export_arrays(
            raman_shift=raman_shift,
            spectra=spectra,
        )
    )

    data: dict[
        str,
        np.ndarray,
    ] = {
        "Raman_shift_cm-1": shift,
    }

    number_width = max(
        4,
        len(
            str(
                spectrum_array.shape[0]
            )
        ),
    )

    for index, spectrum in enumerate(
        spectrum_array,
        start=1,
    ):
        column_name = (
            f"generated_"
            f"{index:0{number_width}d}"
        )

        data[column_name] = spectrum

    return pd.DataFrame(
        data
    )


def export_generated_spectra(
    *,
    raman_shift: np.ndarray,
    spectra: np.ndarray,
    spectrum_output_directory: str | Path,
    plot_output_directory: str | Path,
    base_name: str,
    output_formats: list[str],
) -> list[Path]:
    """
    按照指定格式导出生成光谱。

    raman_shift应当是最终输出轴，例如通过--label
    或--template确定的原始拉曼位移轴。spectra必须已经
    由SpectrumLengthAdapter恢复到该位移轴。
    """

    shift, spectrum_array = (
        _validate_export_arrays(
            raman_shift=raman_shift,
            spectra=spectra,
        )
    )

    normalized_formats = (
        _normalize_output_formats(
            output_formats
        )
    )

    clean_base_name = str(
        base_name
    ).strip()

    if not clean_base_name:
        raise ValueError(
            "base_name不能为空。"
        )

    spectrum_directory = Path(
        spectrum_output_directory
    )

    plot_directory = Path(
        plot_output_directory
    )

    spectrum_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = build_spectrum_dataframe(
        raman_shift=shift,
        spectra=spectrum_array,
    )

    exported_paths: list[Path] = []

    if "xlsx" in normalized_formats:
        path = (
            spectrum_directory
            / f"{clean_base_name}.xlsx"
        )

        dataframe.to_excel(
            path,
            index=False,
        )

        exported_paths.append(
            path
        )

    if "csv" in normalized_formats:
        path = (
            spectrum_directory
            / f"{clean_base_name}.csv"
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        exported_paths.append(
            path
        )

    if "npz" in normalized_formats:
        path = (
            spectrum_directory
            / f"{clean_base_name}.npz"
        )

        np.savez_compressed(
            path,
            raman_shift=shift,
            spectra=spectrum_array,
        )

        exported_paths.append(
            path
        )

    if "png" in normalized_formats:
        path = (
            plot_directory
            / f"{clean_base_name}.png"
        )

        figure, axis = plt.subplots(
            figsize=(10, 6),
            dpi=150,
        )

        number_to_preview = min(
            10,
            spectrum_array.shape[0],
        )

        number_width = max(
            4,
            len(
                str(
                    spectrum_array.shape[0]
                )
            ),
        )

        for index in range(
            number_to_preview
        ):
            axis.plot(
                shift,
                spectrum_array[index],
                linewidth=1.0,
                alpha=0.8,
                label=(
                    f"generated_"
                    f"{index + 1:0{number_width}d}"
                ),
            )

        axis.set_xlabel(
            "Raman shift (cm$^{-1}$)"
        )

        axis.set_ylabel(
            "Preprocessed intensity"
        )

        axis.set_title(
            "Generated SERS spectra"
        )

        axis.grid(
            alpha=0.2
        )

        axis.margins(
            x=0.01
        )

        axis.legend(
            fontsize=7,
            ncol=2,
        )

        figure.tight_layout()

        figure.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        exported_paths.append(
            path
        )

    return exported_paths
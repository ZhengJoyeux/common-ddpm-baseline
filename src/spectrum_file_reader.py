"""读取具有不同拉曼位移轴和不同点数的SERS光谱。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT_LABEL = "__root__"


@dataclass(frozen=True)
class SpectrumCollection:
    """保存全部光谱及其原始信息。"""

    # 兼容旧代码使用的参考轴。
    # 正式训练轴由SpectrumLengthAdapter确定。
    raman_shift: np.ndarray

    # 每个元素是一条一维光谱，因此允许长度不同。
    spectra: np.ndarray

    # 每条光谱对应的原始拉曼位移轴。
    raman_shifts: np.ndarray

    # 每条光谱的原始数据点数。
    original_lengths: np.ndarray

    # 绝对源文件路径。
    source_files: np.ndarray

    # 相对于input_directory的源文件路径。
    relative_source_files: np.ndarray

    spectrum_names: np.ndarray

    # 文件夹标签，例如1e-6、1e-7。
    labels: np.ndarray


def _normalise_extension(
    extension: str,
) -> str:
    extension = str(extension).strip().lower()

    if not extension:
        raise ValueError(
            "文件扩展名不能为空。"
        )

    if not extension.startswith("."):
        extension = "." + extension

    return extension


def _is_auto(
    value: Any,
) -> bool:
    return value is None or (
        isinstance(value, str)
        and value.strip().lower() == "auto"
    )


def _build_object_array(
    values: list[np.ndarray],
) -> np.ndarray:
    """建立一维object数组，避免NumPy强制合并不同长度。"""

    result = np.empty(
        len(values),
        dtype=object,
    )

    result[:] = values

    return result


def discover_input_files(
    input_directory: str | Path,
    allowed_extensions: Sequence[str],
    recursive: bool,
) -> list[Path]:
    directory = Path(
        input_directory
    ).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(
            f"输入目录不存在：{directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"输入路径不是目录：{directory}"
        )

    extensions = {
        _normalise_extension(value)
        for value in allowed_extensions
    }

    if not extensions:
        raise ValueError(
            "allowed_extensions至少要包含一种格式。"
        )

    iterator = (
        directory.rglob("*")
        if recursive
        else directory.glob("*")
    )

    paths = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in extensions
    ]

    paths.sort(
        key=lambda path: path.as_posix().lower()
    )

    if not paths:
        raise FileNotFoundError(
            f"在输入目录中没有找到支持的光谱文件："
            f"{directory}"
        )

    return paths


def _read_table(
    file_path: str | Path,
    *,
    sheet_name: str | int = 0,
    header_row: int = 0,
    csv_encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(
            path,
            header=header_row,
            encoding=csv_encoding,
        )

    elif path.suffix.lower() in {
        ".xlsx",
        ".xls",
    }:
        frame = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header_row,
        )

    else:
        raise ValueError(
            f"不支持的光谱文件格式：{path.suffix}"
        )

    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
    ):
        raise ValueError(
            f"光谱文件没有有效数据：{path}"
        )

    return frame


def _to_numeric_array(
    frame: pd.DataFrame,
    file_path: str | Path,
    field_name: str,
) -> np.ndarray:
    path = Path(file_path)

    if frame.empty:
        raise ValueError(
            f"{path.name} 的{field_name}为空。"
        )

    numeric = frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any().any():
        invalid_positions = np.argwhere(
            numeric.isna().to_numpy()
        )

        row_index, column_index = (
            invalid_positions[0]
        )

        raise ValueError(
            f"{path.name} 的{field_name}中存在空值或"
            f"非数值内容，位置约为数据行"
            f"{int(row_index) + 1}、所选列"
            f"{int(column_index) + 1}。"
        )

    values = numeric.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{path.name} 的{field_name}中存在"
            "NaN或无穷值。"
        )

    return values


def _ensure_increasing_axis(
    raman_shift: np.ndarray,
    spectra: np.ndarray,
    file_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """保证拉曼位移轴严格递增。"""

    differences = np.diff(
        raman_shift.astype(np.float64)
    )

    if np.all(differences > 0.0):
        return raman_shift, spectra

    if np.all(differences < 0.0):
        return (
            raman_shift[::-1].copy(),
            spectra[:, ::-1].copy(),
        )

    raise ValueError(
        f"{file_name} 的拉曼位移轴必须严格单调，"
        "不能包含重复点或乱序点。"
    )


def read_one_spectrum_file(
    file_path: str | Path,
    data_config: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    读取一个宽格式光谱文件。

    第一列为拉曼位移，后面的每一列为一条强度光谱。
    """

    path = Path(
        file_path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到光谱文件：{path}"
        )

    frame = _read_table(
        path,
        sheet_name=data_config.get(
            "sheet_name",
            0,
        ),
        header_row=int(
            data_config.get(
                "header_row",
                0,
            )
        ),
        csv_encoding=str(
            data_config.get(
                "csv_encoding",
                "utf-8-sig",
            )
        ),
    )

    shift_column = int(
        data_config.get(
            "raman_shift_column",
            0,
        )
    )

    intensity_start = int(
        data_config.get(
            "intensity_start_column",
            1,
        )
    )

    if shift_column < 0:
        raise ValueError(
            "raman_shift_column不能小于0。"
        )

    if intensity_start < 0:
        raise ValueError(
            "intensity_start_column不能小于0。"
        )

    if shift_column >= frame.shape[1]:
        raise ValueError(
            f"{path.name} 中不存在拉曼位移列"
            f"{shift_column}。"
        )

    if intensity_start >= frame.shape[1]:
        raise ValueError(
            f"{path.name} 至少应包含一列拉曼位移和"
            "一列光谱强度。"
        )

    raman_shift = _to_numeric_array(
        frame.iloc[
            :,
            shift_column:
            shift_column + 1,
        ],
        path,
        "拉曼位移列",
    ).reshape(-1)

    spectra = _to_numeric_array(
        frame.iloc[
            :,
            intensity_start:,
        ],
        path,
        "光谱强度列",
    ).T

    spectrum_names = [
        str(value)
        for value in frame.columns[
            intensity_start:
        ]
    ]

    if spectra.shape[1] != raman_shift.size:
        raise ValueError(
            f"{path.name} 的位移点数与强度点数"
            "不一致。"
        )

    if raman_shift.size < 2:
        raise ValueError(
            f"{path.name} 至少需要2个拉曼位移点。"
        )

    raman_shift, spectra = (
        _ensure_increasing_axis(
            raman_shift,
            spectra,
            path.name,
        )
    )

    configured_length = data_config.get(
        "original_spectrum_length",
        "auto",
    )

    # auto模式不检查固定长度。
    if not _is_auto(configured_length):
        expected_length = int(
            configured_length
        )

        if raman_shift.size != expected_length:
            raise ValueError(
                f"{path.name} 包含"
                f"{raman_shift.size}个数据点，"
                f"但配置要求{expected_length}个数据点。"
            )

    if bool(
        data_config.get(
            "enforce_expected_range",
            False,
        )
    ):
        expected_minimum = float(
            data_config["expected_min"]
        )

        expected_maximum = float(
            data_config["expected_max"]
        )

        if expected_minimum >= expected_maximum:
            raise ValueError(
                "expected_min必须小于expected_max。"
            )

        actual_minimum = float(
            spectra.min()
        )

        actual_maximum = float(
            spectra.max()
        )

        if (
            actual_minimum < expected_minimum
            or actual_maximum > expected_maximum
        ):
            raise ValueError(
                f"{path.name} 的强度范围"
                f"{actual_minimum}–{actual_maximum}"
                f"超出允许范围"
                f"{expected_minimum}–{expected_maximum}。"
            )

    return (
        raman_shift.astype(
            np.float32,
            copy=False,
        ),
        spectra.astype(
            np.float32,
            copy=False,
        ),
        spectrum_names,
    )


def _get_folder_label(
    file_path: Path,
    input_directory: Path,
) -> str:
    """使用文件相对于input目录的父文件夹作为标签。"""

    relative_parent = file_path.parent.relative_to(
        input_directory
    )

    if str(relative_parent) == ".":
        return ROOT_LABEL

    return relative_parent.as_posix()


def read_spectrum_collection(
    input_directory: str | Path,
    data_config: dict[str, Any],
) -> SpectrumCollection:
    """读取全部光谱，同时保留每条光谱的原始轴。"""

    directory = Path(
        input_directory
    ).expanduser().resolve()

    input_files = discover_input_files(
        input_directory=directory,
        allowed_extensions=data_config.get(
            "allowed_extensions",
            [
                ".csv",
                ".xlsx",
                ".xls",
            ],
        ),
        recursive=bool(
            data_config.get(
                "recursive",
                False,
            )
        ),
    )

    require_same_axis = bool(
        data_config.get(
            "require_same_raman_shift_axis",
            True,
        )
    )

    tolerance = float(
        data_config.get(
            "raman_shift_tolerance",
            1.0e-6,
        )
    )

    if tolerance < 0.0:
        raise ValueError(
            "raman_shift_tolerance不能小于0。"
        )

    reference_axis: np.ndarray | None = None
    longest_axis: np.ndarray | None = None

    all_spectra: list[np.ndarray] = []
    all_axes: list[np.ndarray] = []
    all_lengths: list[int] = []
    all_source_files: list[str] = []
    all_relative_files: list[str] = []
    all_names: list[str] = []
    all_labels: list[str] = []

    for file_path in input_files:
        (
            current_axis,
            current_spectra,
            current_names,
        ) = read_one_spectrum_file(
            file_path=file_path,
            data_config=data_config,
        )

        if reference_axis is None:
            reference_axis = current_axis.copy()

        elif require_same_axis:
            shapes_match = (
                current_axis.shape
                == reference_axis.shape
            )

            axes_match = (
                shapes_match
                and np.allclose(
                    current_axis,
                    reference_axis,
                    rtol=0.0,
                    atol=tolerance,
                )
            )

            if not axes_match:
                raise ValueError(
                    f"{file_path.name} 的拉曼位移轴与"
                    "第一个输入文件不一致。"
                    "若要使用不同点数，请设置"
                    "require_same_raman_shift_axis: false。"
                )

        if (
            longest_axis is None
            or current_axis.size > longest_axis.size
        ):
            longest_axis = current_axis.copy()

        label = _get_folder_label(
            file_path,
            directory,
        )

        relative_file = file_path.relative_to(
            directory
        ).as_posix()

        for row_index, spectrum in enumerate(
            current_spectra
        ):
            all_spectra.append(
                spectrum.copy()
            )

            all_axes.append(
                current_axis.copy()
            )

            all_lengths.append(
                int(current_axis.size)
            )

            all_source_files.append(
                str(file_path)
            )

            all_relative_files.append(
                relative_file
            )

            all_names.append(
                current_names[row_index]
            )

            all_labels.append(
                label
            )

    if (
        longest_axis is None
        or not all_spectra
    ):
        raise RuntimeError(
            "没有读取到任何光谱数据。"
        )

    return SpectrumCollection(
        raman_shift=longest_axis.astype(
            np.float32,
            copy=False,
        ),
        spectra=_build_object_array(
            all_spectra
        ),
        raman_shifts=_build_object_array(
            all_axes
        ),
        original_lengths=np.asarray(
            all_lengths,
            dtype=np.int64,
        ),
        source_files=np.asarray(
            all_source_files,
            dtype=object,
        ),
        relative_source_files=np.asarray(
            all_relative_files,
            dtype=object,
        ),
        spectrum_names=np.asarray(
            all_names,
            dtype=object,
        ),
        labels=np.asarray(
            all_labels,
            dtype=object,
        ),
    )
"""按拉曼位移插值，并自动适配一维DDPM的长度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _is_auto(
    value: Any,
) -> bool:
    return value is None or (
        isinstance(value, str)
        and value.strip().lower() == "auto"
    )


def _validate_axis(
    axis: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(
        axis,
        dtype=np.float64,
    ).reshape(-1)

    if values.size < 2:
        raise ValueError(
            f"{name}至少需要2个点。"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name}包含NaN或无穷值。"
        )

    if not np.all(
        np.diff(values) > 0.0
    ):
        raise ValueError(
            f"{name}必须严格递增。"
        )

    return values


def adapt_spectrum_length(
    spectra: np.ndarray,
    target_length: int,
    method: str = "right_zero_padding",
    padding_value: float = 0.0,
) -> np.ndarray:
    """将已经插值到统一轴的光谱右侧补齐。"""

    values = np.asarray(
        spectra,
        dtype=np.float32,
    )

    if values.ndim != 2:
        raise ValueError(
            "spectra必须是二维数组[N,L]，"
            f"实际形状为{values.shape}。"
        )

    target_length = int(
        target_length
    )

    if target_length <= 0:
        raise ValueError(
            "target_length必须大于0。"
        )

    if target_length < values.shape[1]:
        raise ValueError(
            f"目标长度{target_length}小于"
            f"当前长度{values.shape[1]}。"
        )

    if method != "right_zero_padding":
        raise ValueError(
            "当前只支持right_zero_padding。"
        )

    if target_length == values.shape[1]:
        return values.copy()

    padding_width = (
        target_length
        - values.shape[1]
    )

    return np.pad(
        values,
        (
            (0, 0),
            (0, padding_width),
        ),
        mode="constant",
        constant_values=float(
            padding_value
        ),
    ).astype(
        np.float32,
        copy=False,
    )


def crop_spectrum_length(
    spectra: np.ndarray,
    original_length: int,
) -> np.ndarray:
    """去掉网络输入末尾的补齐点。"""

    values = np.asarray(
        spectra,
        dtype=np.float32,
    )

    if values.ndim != 2:
        raise ValueError(
            "spectra必须是二维数组[N,L]，"
            f"实际形状为{values.shape}。"
        )

    original_length = int(
        original_length
    )

    if (
        original_length <= 0
        or original_length > values.shape[1]
    ):
        raise ValueError(
            "original_length不合法。"
        )

    return values[
        :,
        :original_length,
    ].copy()


@dataclass(frozen=True)
class SpectrumLengthAdapter:
    """保存统一训练轴及网络长度信息。"""

    # 统一插值轴的长度，不代表所有输入文件的原始长度。
    original_length: int

    required_multiple: int
    padded_length: int
    padding_size: int

    padding_mode: str = (
        "right_zero_padding"
    )

    padding_value: float = 0.0

    # 检查点中以列表形式保存，类中用tuple保持不可变。
    model_raman_shift: tuple[
        float,
        ...,
    ] = ()

    raman_range_tolerance: float = 1.0

    def __post_init__(self) -> None:
        if self.original_length <= 0:
            raise ValueError(
                "original_length必须大于0。"
            )

        if self.required_multiple <= 0:
            raise ValueError(
                "required_multiple必须大于0。"
            )

        if self.padded_length < self.original_length:
            raise ValueError(
                "padded_length不能小于"
                "original_length。"
            )

        if (
            self.padded_length
            % self.required_multiple
            != 0
        ):
            raise ValueError(
                "padded_length必须能被"
                "required_multiple整除。"
            )

        expected_padding_size = (
            self.padded_length
            - self.original_length
        )

        if (
            self.padding_size
            != expected_padding_size
        ):
            raise ValueError(
                "padding_size与两个长度不一致。"
            )

        if (
            self.padding_mode
            != "right_zero_padding"
        ):
            raise ValueError(
                "当前只支持right_zero_padding。"
            )

        if not np.isfinite(
            self.padding_value
        ):
            raise ValueError(
                "padding_value必须为有限数值。"
            )

        if self.raman_range_tolerance < 0.0:
            raise ValueError(
                "raman_range_tolerance不能小于0。"
            )

        if self.model_raman_shift:
            model_axis = _validate_axis(
                np.asarray(
                    self.model_raman_shift
                ),
                "model_raman_shift",
            )

            if (
                model_axis.size
                != self.original_length
            ):
                raise ValueError(
                    "model_raman_shift长度与"
                    "original_length不一致。"
                )

    @classmethod
    def create(
        cls,
        *,
        dimension_multipliers: Sequence[int],
        raman_shifts: (
            Sequence[np.ndarray] | None
        ) = None,
        original_length: int | None = None,
        model_length: (
            int | str | None
        ) = "auto",
        padding_mode: str = (
            "right_zero_padding"
        ),
        padding_value: float = 0.0,
        raman_range_tolerance: float = 1.0,
    ) -> "SpectrumLengthAdapter":
        multipliers = tuple(
            int(value)
            for value in dimension_multipliers
        )

        if (
            not multipliers
            or any(
                value <= 0
                for value in multipliers
            )
        ):
            raise ValueError(
                "dimension_multipliers必须包含正整数。"
            )

        required_multiple = 2 ** (
            len(multipliers) - 1
        )

        if raman_shifts is not None:
            axes = [
                _validate_axis(
                    axis,
                    f"raman_shifts[{index}]",
                )
                for index, axis in enumerate(
                    raman_shifts
                )
            ]

            if not axes:
                raise ValueError(
                    "raman_shifts不能为空。"
                )

            # 使用训练集中点数最多的真实位移轴。
            model_axis = max(
                axes,
                key=lambda axis: axis.size,
            ).copy()

            tolerance = float(
                raman_range_tolerance
            )

            for index, axis in enumerate(
                axes
            ):
                start_difference = abs(
                    axis[0] - model_axis[0]
                )

                end_difference = abs(
                    axis[-1] - model_axis[-1]
                )

                if (
                    max(
                        start_difference,
                        end_difference,
                    )
                    > tolerance
                ):
                    raise ValueError(
                        f"raman_shifts[{index}]与统一训练轴"
                        "的起止范围相差过大；"
                        f"起点差{start_difference:.6g} cm-1，"
                        f"终点差{end_difference:.6g} cm-1，"
                        f"允许值为{tolerance:.6g} cm-1。"
                        "不同点数可以自动适配，"
                        "但测量范围必须基本一致。"
                    )

            resolved_original_length = int(
                model_axis.size
            )

        else:
            if (
                original_length is None
                or int(original_length) <= 0
            ):
                raise ValueError(
                    "未提供raman_shifts时必须提供"
                    "original_length。"
                )

            resolved_original_length = int(
                original_length
            )

            model_axis = np.arange(
                resolved_original_length,
                dtype=np.float64,
            )

        automatic_padded_length = (
            (
                resolved_original_length
                + required_multiple
                - 1
            )
            // required_multiple
        ) * required_multiple

        if _is_auto(model_length):
            padded_length = (
                automatic_padded_length
            )

        else:
            padded_length = int(
                model_length
            )

            if (
                padded_length
                < resolved_original_length
            ):
                raise ValueError(
                    f"model_spectrum_length="
                    f"{padded_length}小于统一轴长度"
                    f"{resolved_original_length}。"
                )

            if (
                padded_length
                % required_multiple
                != 0
            ):
                raise ValueError(
                    f"model_spectrum_length="
                    f"{padded_length}不能被"
                    f"U-Net下采样倍数"
                    f"{required_multiple}整除。"
                )

        return cls(
            original_length=(
                resolved_original_length
            ),
            required_multiple=required_multiple,
            padded_length=padded_length,
            padding_size=(
                padded_length
                - resolved_original_length
            ),
            padding_mode=padding_mode,
            padding_value=float(
                padding_value
            ),
            model_raman_shift=tuple(
                float(value)
                for value in model_axis
            ),
            raman_range_tolerance=float(
                raman_range_tolerance
            ),
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
    ) -> "SpectrumLengthAdapter":
        """从检查点恢复适配器。"""

        model_axis = metadata.get(
            "model_raman_shift",
            metadata.get(
                "raman_shift",
                [],
            ),
        )

        return cls(
            original_length=int(
                metadata["original_length"]
            ),
            required_multiple=int(
                metadata["required_multiple"]
            ),
            padded_length=int(
                metadata["padded_length"]
            ),
            padding_size=int(
                metadata["padding_size"]
            ),
            padding_mode=str(
                metadata.get(
                    "padding_mode",
                    "right_zero_padding",
                )
            ),
            padding_value=float(
                metadata.get(
                    "padding_value",
                    0.0,
                )
            ),
            model_raman_shift=tuple(
                float(value)
                for value in model_axis
            ),
            raman_range_tolerance=float(
                metadata.get(
                    "raman_range_tolerance",
                    1.0,
                )
            ),
        )

    @property
    def model_axis(
        self,
    ) -> np.ndarray:
        return np.asarray(
            self.model_raman_shift,
            dtype=np.float64,
        )

    def interpolate_to_model_axis(
        self,
        spectra: Sequence[np.ndarray],
        raman_shifts: Sequence[np.ndarray],
    ) -> np.ndarray:
        """将全部光谱按真实拉曼位移插值到统一轴。"""

        spectrum_items = list(
            spectra
        )

        axis_items = list(
            raman_shifts
        )

        if (
            len(spectrum_items)
            != len(axis_items)
        ):
            raise ValueError(
                "spectra与raman_shifts数量不一致。"
            )

        if not spectrum_items:
            raise ValueError(
                "spectra不能为空。"
            )

        model_axis = self.model_axis

        output = np.empty(
            (
                len(spectrum_items),
                self.original_length,
            ),
            dtype=np.float32,
        )

        for index, (
            spectrum,
            source_axis,
        ) in enumerate(
            zip(
                spectrum_items,
                axis_items,
            )
        ):
            values = np.asarray(
                spectrum,
                dtype=np.float64,
            ).reshape(-1)

            source_axis = _validate_axis(
                source_axis,
                f"raman_shifts[{index}]",
            )

            if values.size != source_axis.size:
                raise ValueError(
                    f"第{index}条光谱的强度点数"
                    "与位移点数不一致。"
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"第{index}条光谱含NaN或无穷值。"
                )

            start_difference = abs(
                source_axis[0]
                - model_axis[0]
            )

            end_difference = abs(
                source_axis[-1]
                - model_axis[-1]
            )

            if (
                max(
                    start_difference,
                    end_difference,
                )
                > self.raman_range_tolerance
            ):
                raise ValueError(
                    f"第{index}条光谱与训练轴的"
                    "起止范围相差过大；"
                    f"起点差{start_difference:.6g} cm-1，"
                    f"终点差{end_difference:.6g} cm-1。"
                )

            output[index] = np.interp(
                model_axis,
                source_axis,
                values,
            ).astype(
                np.float32
            )

        return output

    def adapt(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """插值和归一化完成后，再补到网络长度。"""

        values = np.asarray(
            spectra,
            dtype=np.float32,
        )

        if (
            values.ndim != 2
            or values.shape[1]
            != self.original_length
        ):
            raise ValueError(
                "adapt接收的光谱必须已经插值到"
                "统一训练轴，"
                f"期望[N,{self.original_length}]，"
                f"实际为{values.shape}。"
            )

        return adapt_spectrum_length(
            spectra=values,
            target_length=self.padded_length,
            method=self.padding_mode,
            padding_value=self.padding_value,
        )

    def interpolate_from_model_axis(
        self,
        spectra: np.ndarray,
        target_raman_shift: np.ndarray,
    ) -> np.ndarray:
        """
        将统一模型拉曼轴上的光谱，
        插值恢复到指定的目标拉曼位移轴。
        """

        values = np.asarray(
            spectra,
            dtype=np.float32,
        )

        model_axis = self.model_axis

        if (
            values.ndim != 2
            or values.shape[1] != model_axis.size
        ):
            raise ValueError(
                "interpolate_from_model_axis接收的光谱"
                "必须位于统一模型拉曼轴上，"
                f"期望[N,{model_axis.size}]，"
                f"实际为{values.shape}。"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "待恢复的生成光谱含NaN或无穷值。"
            )

        target_axis = _validate_axis(
            target_raman_shift,
            "target_raman_shift",
        )

        left_excess = max(
            0.0,
            model_axis[0] - target_axis[0],
        )

        right_excess = max(
            0.0,
            target_axis[-1] - model_axis[-1],
        )

        if (
            max(
                left_excess,
                right_excess,
            )
            > self.raman_range_tolerance
        ):
            raise ValueError(
                "目标输出轴超出了模型训练轴范围，"
                "无法可靠恢复；"
                f"左侧超出{left_excess:.6g} cm-1，"
                f"右侧超出{right_excess:.6g} cm-1。"
            )

        restored = np.empty(
            (
                values.shape[0],
                target_axis.size,
            ),
            dtype=np.float32,
        )

        for index, spectrum in enumerate(values):
            restored[index] = np.interp(
                target_axis,
                model_axis,
                spectrum.astype(
                    np.float64
                ),
            ).astype(
                np.float32
            )

        return restored

    def restore(
        self,
        spectra: np.ndarray,
        target_raman_shift: (
            np.ndarray | None
        ) = None,
    ) -> np.ndarray:
        """
        先去除网络补点，再根据需要恢复到目标拉曼位移轴。
        """

        cropped = crop_spectrum_length(
            spectra=spectra,
            original_length=self.original_length,
        )

        if target_raman_shift is None:
            return cropped

        return self.interpolate_from_model_axis(
            spectra=cropped,
            target_raman_shift=target_raman_shift,
        )

    def to_metadata(
        self,
    ) -> dict[str, Any]:
        """生成保存到检查点中的长度信息。"""

        return {
            "original_length": (
                self.original_length
            ),
            "required_multiple": (
                self.required_multiple
            ),
            "padded_length": (
                self.padded_length
            ),
            "padding_size": (
                self.padding_size
            ),
            "padding_mode": (
                self.padding_mode
            ),
            "padding_value": (
                self.padding_value
            ),
            "model_raman_shift": list(
                self.model_raman_shift
            ),
            "raman_range_tolerance": (
                self.raman_range_tolerance
            ),
        }
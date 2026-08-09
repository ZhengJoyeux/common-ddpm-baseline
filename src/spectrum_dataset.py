from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(
    Dataset[torch.Tensor]
):
    """
    保存DDPM输入张量，同时保存原始轴、长度和标签。

    __getitem__仍然只返回光谱张量，
    因此不需要修改DdpmTrainer。
    """

    def __init__(
        self,
        spectra: np.ndarray,
        *,
        original_lengths: (
            Sequence[int] | None
        ) = None,
        original_raman_shifts: (
            Sequence[np.ndarray] | None
        ) = None,
        labels: (
            Sequence[str] | None
        ) = None,
        source_files: (
            Sequence[str] | None
        ) = None,
        constraint_reference_priors: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(
            spectra,
            dtype=np.float32,
        )

        if values.ndim != 2:
            raise ValueError(
                "spectra必须是二维数组[N,L]，"
                f"实际为{values.shape}。"
            )

        if values.shape[0] == 0:
            raise ValueError(
                "数据集不能为空。"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "光谱中存在NaN或无穷大。"
            )

        number_of_spectra = int(
            values.shape[0]
        )

        def check_count(
            name: str,
            metadata: (
                Sequence[object] | None
            ),
        ) -> None:
            if (
                metadata is not None
                and len(metadata)
                != number_of_spectra
            ):
                raise ValueError(
                    f"{name}数量{len(metadata)}"
                    f"与光谱数量{number_of_spectra}"
                    "不一致。"
                )

        check_count(
            "original_lengths",
            original_lengths,
        )

        check_count(
            "original_raman_shifts",
            original_raman_shifts,
        )

        check_count(
            "labels",
            labels,
        )

        check_count(
            "source_files",
            source_files,
        )

        reference_priors = None
        if constraint_reference_priors is not None:
            reference_priors = np.asarray(
                constraint_reference_priors,
                dtype=np.float32,
            )
            if reference_priors.shape != values.shape:
                raise ValueError(
                    "constraint_reference_priors形状必须与spectra一致："
                    f"先验为{reference_priors.shape}，"
                    f"光谱为{values.shape}。"
                )
            if not np.isfinite(reference_priors).all():
                raise ValueError(
                    "constraint_reference_priors中存在NaN或无穷大。"
                )

        # [N,L]转换为DDPM使用的[N,1,L]。
        self.spectra = torch.from_numpy(
            values
        ).unsqueeze(1)

        self.original_lengths = (
            None
            if original_lengths is None
            else np.asarray(
                original_lengths,
                dtype=np.int64,
            )
        )

        self.original_raman_shifts = (
            None
            if original_raman_shifts is None
            else tuple(
                np.asarray(
                    axis,
                    dtype=np.float64,
                ).copy()
                for axis in original_raman_shifts
            )
        )

        self.labels = (
            None
            if labels is None
            else np.asarray(
                labels,
                dtype=object,
            )
        )

        self.source_files = (
            None
            if source_files is None
            else np.asarray(
                source_files,
                dtype=object,
            )
        )

        self.constraint_reference_priors = (
            None
            if reference_priors is None
            else torch.from_numpy(reference_priors).unsqueeze(1)
        )

    def __len__(
        self,
    ) -> int:
        return int(
            self.spectra.shape[0]
        )

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        spectrum = self.spectra[index]
        if self.constraint_reference_priors is None:
            return spectrum
        return {
            "spectrum": spectrum,
            "constraint_reference_prior": (
                self.constraint_reference_priors[index]
            ),
        }

    def get_original_metadata(
        self,
        index: int,
    ) -> dict[str, object]:
        """获取指定光谱的原始信息。"""

        return {
            "original_length": (
                None
                if self.original_lengths is None
                else int(
                    self.original_lengths[index]
                )
            ),
            "raman_shift": (
                None
                if self.original_raman_shifts
                is None
                else self.original_raman_shifts[
                    index
                ].copy()
            ),
            "label": (
                None
                if self.labels is None
                else str(
                    self.labels[index]
                )
            ),
            "source_file": (
                None
                if self.source_files is None
                else str(
                    self.source_files[index]
                )
            ),
        }
"""将训练总损失和 D3.2 / D3.4.x 分项损失写入终端与 CSV。"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any


COMPONENT_FIELDS = (
    "ddpm_loss",
    "physics_loss",
    "physics_raw_loss",
    "position_loss",
    "width_loss",
    "sharpness_loss",
    "presence_loss",
    "local_shape_loss",
    "roughness_loss",
    "extreme_loss",
    "negative_valley_loss",
    "scaled_residual_guard_loss",
    "mean_timestep_weight",
    "mean_pathology_timestep_weight",
    "mean_detected_peaks",
    "diversity_loss",
    "diversity_raw_loss",
    "pairwise_distance_loss",
    "pairwise_correlation_loss",
    "pointwise_variance_floor_loss",
    "peak_morphology_loss",
    "peak_position_dispersion_loss",
    "peak_shift_coherence_loss",
    "peak_height_dispersion_loss",
    "peak_width_dispersion_loss",
    "mean_morphology_peaks",
    "diversity_active_samples",
    "mean_diversity_timestep_weight",
)


class TrainingLogger:
    """记录训练过程，并把 D3.4.2 新增峰级分量完整写入日志。"""

    FIELD_NAMES = (
        "step",
        "training_loss",
        *(
            f"training_{name}"
            for name in COMPONENT_FIELDS
        ),
        "validation_loss",
        *(
            f"validation_{name}"
            for name in COMPONENT_FIELDS
        ),
        "learning_rate",
        "elapsed_seconds",
    )

    def __init__(
        self,
        log_file: str | Path,
    ) -> None:
        self.log_file = Path(
            log_file
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.start_time = time.time()

        if self.log_file.exists():
            with self.log_file.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.reader(
                    file
                )

                existing_header = next(
                    reader,
                    [],
                )

            if (
                tuple(
                    existing_header
                )
                != tuple(
                    self.FIELD_NAMES
                )
            ):
                raise RuntimeError(
                    "现有训练日志表头与当前"
                    "D3.4.2日志格式不一致。"
                    "请使用新的output目录，"
                    "不要覆盖旧D3日志。"
                )

        else:
            with self.log_file.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                csv.DictWriter(
                    file,
                    fieldnames=(
                        self.FIELD_NAMES
                    ),
                ).writeheader()

    @staticmethod
    def _value(
        mapping: dict[
            str,
            Any,
        ]
        | None,
        name: str,
    ) -> float | str:
        if (
            not mapping
            or name not in mapping
        ):
            return ""

        return float(
            mapping[
                name
            ]
        )

    def record(
        self,
        *,
        step: int,
        training_loss: float,
        validation_loss: (
            float
            | None
        ),
        learning_rate: float,
        training_components: dict[
            str,
            float,
        ]
        | None = None,
        validation_components: dict[
            str,
            float,
        ]
        | None = None,
    ) -> None:
        elapsed_seconds = (
            time.time()
            - self.start_time
        )

        row: dict[
            str,
            Any,
        ] = {
            "step": int(
                step
            ),
            "training_loss": float(
                training_loss
            ),
            "validation_loss": (
                ""
                if validation_loss
                is None
                else float(
                    validation_loss
                )
            ),
            "learning_rate": float(
                learning_rate
            ),
            "elapsed_seconds": float(
                elapsed_seconds
            ),
        }

        for name in COMPONENT_FIELDS:
            row[
                f"training_{name}"
            ] = self._value(
                training_components,
                name,
            )

            row[
                f"validation_{name}"
            ] = self._value(
                validation_components,
                name,
            )

        with self.log_file.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as file:
            csv.DictWriter(
                file,
                fieldnames=(
                    self.FIELD_NAMES
                ),
            ).writerow(
                row
            )

        validation_text = (
            "未计算"
            if validation_loss is None
            else f"{validation_loss:.6f}"
        )

        ddpm_text = (
            "未记录"
            if not training_components
            else (
                f"{training_components.get('ddpm_loss', float('nan')):.6f}"
            )
        )

        physics_text = (
            "未记录"
            if not training_components
            else (
                f"{training_components.get('physics_loss', 0.0):.6f}"
            )
        )

        diversity_text = (
            "未记录"
            if not training_components
            else (
                f"{training_components.get('diversity_loss', 0.0):.6f}"
            )
        )

        extra_text = ""

        if training_components:
            extra_text = (
                " | hf_dist="
                f"{training_components.get('pairwise_distance_loss', 0.0):.4f}"
                " | peak_morph="
                f"{training_components.get('peak_morphology_loss', 0.0):.4f}"
                " | peak_pos="
                f"{training_components.get('peak_position_dispersion_loss', 0.0):.4f}"
                " | coherent="
                f"{training_components.get('peak_shift_coherence_loss', 0.0):.4f}"
                " | peak_h="
                f"{training_components.get('peak_height_dispersion_loss', 0.0):.4f}"
                " | peak_w="
                f"{training_components.get('peak_width_dispersion_loss', 0.0):.4f}"
                " | morph_peaks="
                f"{training_components.get('mean_morphology_peaks', 0.0):.2f}"
            )

        print(
            f"step={step}"
            f" | train_total={training_loss:.6f}"
            f" | train_ddpm={ddpm_text}"
            f" | train_physics={physics_text}"
            f" | train_diversity={diversity_text}"
            f"{extra_text}"
            f" | val_total={validation_text}"
            f" | lr={learning_rate:.3e}"
            f" | elapsed={elapsed_seconds:.1f}s"
        )
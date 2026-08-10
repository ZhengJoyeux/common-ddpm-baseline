"""记录 D0-D3 训练、验证及各类 SERS 约束分项损失。"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any


COMPONENT_FIELDS = (
    "ddpm_loss",
    "ddpm_uniform_loss",
    "ddpm_residual_aware_loss",
    "residual_reference_scale",
    "residual_weight_mean",
    "residual_weight_min",
    "residual_weight_max",
    "physics_loss",
    "physics_raw_loss",
    "position_loss",
    "width_loss",
    "sharpness_loss",
    "presence_loss",
    "local_shape_loss",
    "stable_peak_shift_loss",
    "stable_peak_coherence_loss",
    "stable_peak_height_loss",
    "stable_peak_width_loss",
    "stable_peak_mean_absolute_shift_cm1",
    "stable_peak_count",
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
    "local_peak_distribution_loss",
    "local_peak_distribution_raw_loss",
    "local_peak_loss_cap",
    "local_peak_loss_scale",
    "negative_tail_guard_loss",
    "negative_tail_guard_contribution",
    "wing_shape_tracking_loss",
    "wing_shape_tracking_contribution",
    "wing_dispersion_loss",
    "wing_dispersion_contribution",
    "shift_dispersion_loss",
    "shift_dispersion_contribution",
    "width_dispersion_loss",
    "width_dispersion_contribution",
    "height_cv_upper_loss",
    "height_cv_upper_contribution",
    "local_peak_active_samples",
    "local_peak_count",
    "mean_local_peak_timestep_weight",
)


class TrainingLogger:
    """把训练过程写入 CSV，并在终端打印关键分项。"""

    FIELD_NAMES = (
        "step",
        "training_loss",
        *(f"training_{name}" for name in COMPONENT_FIELDS),
        "validation_loss",
        *(f"validation_{name}" for name in COMPONENT_FIELDS),
        "learning_rate",
        "elapsed_seconds",
    )

    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        if self.log_file.exists():
            with self.log_file.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                existing_header = next(csv.reader(file), [])

            if tuple(existing_header) != tuple(self.FIELD_NAMES):
                raise RuntimeError(
                    "现有训练日志表头与当前日志格式不一致。"
                    "本次局部峰分布约束增加了新的日志字段；"
                    "请使用新的output目录，不要覆盖旧实验日志。"
                )
        else:
            with self.log_file.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                csv.DictWriter(
                    file,
                    fieldnames=self.FIELD_NAMES,
                ).writeheader()

    @staticmethod
    def _value(
        mapping: dict[str, Any] | None,
        name: str,
    ) -> float | str:
        if not mapping or name not in mapping:
            return ""
        return float(mapping[name])

    def record(
        self,
        *,
        step: int,
        training_loss: float,
        validation_loss: float | None,
        learning_rate: float,
        training_components: dict[str, float] | None = None,
        validation_components: dict[str, float] | None = None,
    ) -> None:
        elapsed_seconds = time.time() - self.start_time

        row: dict[str, Any] = {
            "step": int(step),
            "training_loss": float(training_loss),
            "validation_loss": (
                "" if validation_loss is None else float(validation_loss)
            ),
            "learning_rate": float(learning_rate),
            "elapsed_seconds": float(elapsed_seconds),
        }

        for name in COMPONENT_FIELDS:
            row[f"training_{name}"] = self._value(
                training_components,
                name,
            )
            row[f"validation_{name}"] = self._value(
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
                fieldnames=self.FIELD_NAMES,
            ).writerow(row)

        validation_text = (
            "未计算"
            if validation_loss is None
            else f"{validation_loss:.6f}"
        )

        if training_components:
            ddpm_text = (
                f"{training_components.get('ddpm_loss', float('nan')):.6f}"
            )
            uniform_text = (
                f"{training_components.get('ddpm_uniform_loss', float('nan')):.6f}"
            )
            physics_text = (
                f"{training_components.get('physics_loss', 0.0):.6f}"
            )
            diversity_text = (
                f"{training_components.get('diversity_loss', 0.0):.6f}"
            )
            local_text = (
                f"{training_components.get('local_peak_distribution_loss', 0.0):.6f}"
            )
            local_scale_text = (
                f"{training_components.get('local_peak_loss_scale', 0.0):.3f}"
            )
            neg_text = (
                f"{training_components.get('negative_tail_guard_loss', 0.0):.4f}"
            )
            wing_track_text = (
                f"{training_components.get('wing_shape_tracking_loss', 0.0):.4f}"
            )
            wing_disp_text = (
                f"{training_components.get('wing_dispersion_loss', 0.0):.4f}"
            )
            shift_text = (
                f"{training_components.get('shift_dispersion_loss', 0.0):.4f}"
            )
            width_text = (
                f"{training_components.get('width_dispersion_loss', 0.0):.4f}"
            )
            height_text = (
                f"{training_components.get('height_cv_upper_loss', 0.0):.4f}"
            )
            active_text = (
                f"{training_components.get('local_peak_active_samples', 0.0):.1f}"
            )

            extra_text = (
                f" | ddpm_uniform={uniform_text}"
                f" | local_peak={local_text}"
                f" | local_scale={local_scale_text}"
                f" | neg_tail={neg_text}"
                f" | wing_track={wing_track_text}"
                f" | wing_disp={wing_disp_text}"
                f" | shift_disp={shift_text}"
                f" | width_disp={width_text}"
                f" | height_cv={height_text}"
                f" | local_n={active_text}"
            )
        else:
            ddpm_text = "未记录"
            physics_text = "未记录"
            diversity_text = "未记录"
            extra_text = ""

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
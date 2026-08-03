"""Write training progress to both terminal and CSV."""

import csv
from pathlib import Path
import time


class TrainingLogger:
    """CSV and terminal training logger."""

    FIELD_NAMES = (
        "step",
        "training_loss",
        "validation_loss",
        "learning_rate",
        "elapsed_seconds",
    )

    def __init__(self, log_file: str | Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        if not self.log_file.exists():
            with self.log_file.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.FIELD_NAMES,
                )
                writer.writeheader()

    def record(
        self,
        step: int,
        training_loss: float,
        validation_loss: float | None,
        learning_rate: float,
    ) -> None:
        elapsed_seconds = time.time() - self.start_time

        row = {
            "step": int(step),
            "training_loss": float(training_loss),
            "validation_loss": (
                ""
                if validation_loss is None
                else float(validation_loss)
            ),
            "learning_rate": float(learning_rate),
            "elapsed_seconds": float(elapsed_seconds),
        }

        with self.log_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELD_NAMES,
            )
            writer.writerow(row)

        validation_text = (
            "未计算"
            if validation_loss is None
            else f"{validation_loss:.6f}"
        )

        print(
            f"step={step} | "
            f"train_loss={training_loss:.6f} | "
            f"val_loss={validation_text} | "
            f"lr={learning_rate:.3e} | "
            f"elapsed={elapsed_seconds:.1f}s"
        )
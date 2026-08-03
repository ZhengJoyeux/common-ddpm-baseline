from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _axis_profile_id(
    axis: np.ndarray,
) -> str:
    """根据拉曼位移轴生成唯一标识。"""

    values = np.asarray(
        axis,
        dtype=np.float64,
    ).reshape(-1)

    digest = hashlib.sha256(
        values.tobytes()
    ).hexdigest()[:12]

    return (
        f"axis_{values.size}_{digest}"
    )


def build_axis_metadata(
    *,
    labels: np.ndarray,
    relative_source_files: np.ndarray,
    raman_shifts: np.ndarray,
) -> dict[str, Any]:
    """建立标签、源文件与原始位移轴之间的映射。"""

    if not (
        len(labels)
        == len(relative_source_files)
        == len(raman_shifts)
    ):
        raise ValueError(
            "标签、源文件和位移轴数量不一致。"
        )

    source_records: dict[
        str,
        dict[str, Any],
    ] = {}

    label_profiles: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for (
        label_value,
        source_value,
        axis_value,
    ) in zip(
        labels,
        relative_source_files,
        raman_shifts,
    ):
        label = str(
            label_value
        )

        source_file = str(
            source_value
        )

        axis = np.asarray(
            axis_value,
            dtype=np.float64,
        ).reshape(-1)

        if (
            axis.size < 2
            or not np.isfinite(axis).all()
        ):
            raise ValueError(
                f"{source_file}的拉曼位移轴无效。"
            )

        if not np.all(
            np.diff(axis) > 0.0
        ):
            raise ValueError(
                f"{source_file}的拉曼位移轴"
                "不是严格递增。"
            )

        profile_id = _axis_profile_id(
            axis
        )

        existing_source = (
            source_records.get(
                source_file
            )
        )

        if existing_source is not None:
            if (
                existing_source["label"]
                != label
                or existing_source["profile_id"]
                != profile_id
            ):
                raise ValueError(
                    f"同一源文件{source_file}"
                    "出现了不一致的轴或标签。"
                )

            continue

        source_records[source_file] = {
            "label": label,
            "profile_id": profile_id,
            "length": int(
                axis.size
            ),
            "raman_shift": axis.tolist(),
        }

        profiles = (
            label_profiles.setdefault(
                label,
                {},
            )
        )

        profile = profiles.setdefault(
            profile_id,
            {
                "length": int(
                    axis.size
                ),
                "raman_shift": axis.tolist(),
                "source_files": [],
            },
        )

        profile["source_files"].append(
            source_file
        )

    labels_output: dict[
        str,
        Any,
    ] = {}

    for label in sorted(
        label_profiles
    ):
        profiles = label_profiles[
            label
        ]

        # 优先选择源文件数量最多的位移轴。
        # 文件数相同时，选择点数更多的位移轴。
        default_profile_id = sorted(
            profiles,
            key=lambda current_id: (
                -len(
                    profiles[current_id][
                        "source_files"
                    ]
                ),
                -int(
                    profiles[current_id][
                        "length"
                    ]
                ),
                current_id,
            ),
        )[0]

        labels_output[label] = {
            "default_profile_id": (
                default_profile_id
            ),
            "profiles": profiles,
        }

    return {
        "labels": labels_output,
        "source_files": source_records,
    }


def resolve_label_axis(
    metadata: dict[str, Any],
    label: str | None,
) -> tuple[
    str,
    np.ndarray,
    str,
]:
    """从检查点读取指定文件夹标签的输出轴。"""

    axis_metadata = metadata.get(
        "axis_metadata"
    )

    if not isinstance(
        axis_metadata,
        dict,
    ):
        raise RuntimeError(
            "检查点不含axis_metadata；"
            "请使用自适应代码重新训练。"
        )

    labels = axis_metadata.get(
        "labels"
    )

    if (
        not isinstance(labels, dict)
        or not labels
    ):
        raise RuntimeError(
            "检查点中的标签轴信息为空。"
        )

    if label is None:
        if len(labels) != 1:
            available_labels = "、".join(
                sorted(labels)
            )

            raise ValueError(
                "检查点包含多个文件夹标签，"
                "请用--label指定输出轴。"
                f"可用标签：{available_labels}"
            )

        resolved_label = next(
            iter(labels)
        )

    else:
        resolved_label = str(
            label
        ).strip()

        if resolved_label not in labels:
            available_labels = "、".join(
                sorted(labels)
            )

            raise ValueError(
                f"找不到标签{resolved_label!r}。"
                f"可用标签：{available_labels}"
            )

    label_record = labels[
        resolved_label
    ]

    profile_id = str(
        label_record[
            "default_profile_id"
        ]
    )

    profile = label_record[
        "profiles"
    ][profile_id]

    axis = np.asarray(
        profile["raman_shift"],
        dtype=np.float64,
    )

    return (
        resolved_label,
        axis,
        profile_id,
    )


def save_checkpoint(
    checkpoint_path: str | Path,
    payload: dict[str, Any],
) -> Path:
    """以临时文件替换的方式安全保存检查点。"""

    path = Path(
        checkpoint_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        path
    )

    return path


def load_checkpoint_file(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """读取并检查一个DDPM检查点文件。"""

    path = Path(
        checkpoint_path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到检查点：{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            f"检查点格式不正确：{path}"
        )

    return checkpoint


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """兼容旧接口。"""

    return load_checkpoint_file(
        checkpoint_path=checkpoint_path,
        map_location=map_location,
    )


class CheckpointManager:
    """管理训练检查点、latest.pt和best.pt。"""

    def __init__(
        self,
        checkpoint_directory: str | Path,
    ) -> None:
        self.checkpoint_directory = Path(
            checkpoint_directory
        )

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        *,
        step: int,
        diffusion_state: dict[str, Any],
        ema_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        scaler_state: dict[str, Any],
        configuration: dict[str, Any],
        metadata: dict[str, Any],
        best_validation_loss: float,
        file_name: str | None = None,
        update_latest: bool = True,
    ) -> Path:
        """保存完整训练状态并返回检查点路径。"""

        if step < 0:
            raise ValueError(
                "step不能小于0。"
            )

        # 检查metadata基本格式。
        if not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "metadata必须是字典。"
            )

        # 如果包含自适应位移轴信息，
        # 检查其中是否具有有效的标签映射。
        if "axis_metadata" in metadata:
            axis_metadata = metadata[
                "axis_metadata"
            ]

            if (
                not isinstance(
                    axis_metadata,
                    dict,
                )
                or not axis_metadata.get(
                    "labels"
                )
            ):
                raise ValueError(
                    "axis_metadata缺少有效的"
                    "labels映射。"
                )

        payload = {
            "step": int(
                step
            ),
            "diffusion_state": (
                diffusion_state
            ),
            "ema_state": ema_state,
            "optimizer_state": (
                optimizer_state
            ),
            "scaler_state": scaler_state,
            "configuration": configuration,
            "metadata": metadata,
            "best_validation_loss": float(
                best_validation_loss
            ),
        }

        if file_name is None:
            file_name = (
                f"step_{step:08d}.pt"
            )

        checkpoint_path = (
            self.checkpoint_directory
            / file_name
        )

        save_checkpoint(
            checkpoint_path=checkpoint_path,
            payload=payload,
        )

        if update_latest:
            latest_path = (
                self.checkpoint_directory
                / "latest.pt"
            )

            if latest_path != checkpoint_path:
                save_checkpoint(
                    checkpoint_path=latest_path,
                    payload=payload,
                )

        return checkpoint_path
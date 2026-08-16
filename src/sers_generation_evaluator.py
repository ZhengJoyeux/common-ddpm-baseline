"""Final evaluation utilities for generated one-dimensional SERS spectra.

Experiment D evaluates:
1. train-driven peak position / height / FWHM distributions;
2. whole-spectrum similarity to real spectra;
3. PCA / Wasserstein / MMD distribution consistency;
4. broad/local non-peak distribution widths;
5. diversity and nearest-training-distance screening.

This module never changes generated spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


EPSILON = 1.0e-12


def as_2d_finite(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"{name}必须是二维数组[N,L]，实际={array.shape}。"
        )
    if array.shape[0] < 1 or array.shape[1] < 3:
        raise ValueError(f"{name}形状无效：{array.shape}。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}包含NaN或无穷值。")
    return array


def as_axis(
    values: np.ndarray,
    expected_length: int | None = None,
) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 3:
        raise ValueError("Raman shift轴至少需要3个点。")
    if expected_length is not None and axis.size != expected_length:
        raise ValueError(
            "Raman shift轴长度与光谱长度不一致："
            f"{axis.size} != {expected_length}。"
        )
    if not np.isfinite(axis).all():
        raise ValueError("Raman shift轴包含NaN或无穷值。")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("Raman shift轴必须严格递增。")
    return axis


def _safe_std(
    values: np.ndarray,
    axis: int,
    ddof: int = 0,
) -> np.ndarray:
    result = np.std(values, axis=axis, ddof=ddof)
    return np.maximum(result, EPSILON)


def row_standardize(values: np.ndarray) -> np.ndarray:
    array = as_2d_finite(values, "spectra")
    mean = np.mean(array, axis=1, keepdims=True)
    std = _safe_std(array, axis=1, ddof=0)[:, None]
    return (array - mean) / std


def row_pearson(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_values = as_2d_finite(left, "left")
    right_values = as_2d_finite(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError("row_pearson两侧形状必须一致。")
    left_z = row_standardize(left_values)
    right_z = row_standardize(right_values)
    correlation = np.mean(left_z * right_z, axis=1)
    return np.clip(correlation, -1.0, 1.0)


def row_cosine(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_values = as_2d_finite(left, "left")
    right_values = as_2d_finite(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError("row_cosine两侧形状必须一致。")
    numerator = np.sum(left_values * right_values, axis=1)
    denominator = (
        np.linalg.norm(left_values, axis=1)
        * np.linalg.norm(right_values, axis=1)
    )
    denominator = np.maximum(denominator, EPSILON)
    cosine = numerator / denominator
    return np.clip(cosine, -1.0, 1.0)


def row_rmse(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_values = as_2d_finite(left, "left")
    right_values = as_2d_finite(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError("row_rmse两侧形状必须一致。")
    return np.sqrt(
        np.mean(
            np.square(left_values - right_values),
            axis=1,
        )
    )


def row_mae(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_values = as_2d_finite(left, "left")
    right_values = as_2d_finite(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError("row_mae两侧形状必须一致。")
    return np.mean(
        np.abs(left_values - right_values),
        axis=1,
    )


def derivative_pearson(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return row_pearson(
        np.diff(as_2d_finite(left, "left"), axis=1),
        np.diff(as_2d_finite(right, "right"), axis=1),
    )


def summarize_vector(
    values: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        names = (
            "mean",
            "std",
            "p05",
            "p25",
            "median",
            "p75",
            "p95",
            "minimum",
            "maximum",
        )
        return {
            f"{prefix}{name}": np.nan
            for name in names
        }

    return {
        f"{prefix}mean": float(np.mean(array)),
        f"{prefix}std": float(np.std(array, ddof=0)),
        f"{prefix}p05": float(np.percentile(array, 5.0)),
        f"{prefix}p25": float(np.percentile(array, 25.0)),
        f"{prefix}median": float(np.median(array)),
        f"{prefix}p75": float(np.percentile(array, 75.0)),
        f"{prefix}p95": float(np.percentile(array, 95.0)),
        f"{prefix}minimum": float(np.min(array)),
        f"{prefix}maximum": float(np.max(array)),
    }


def nearest_reference_metrics(
    *,
    query_spectra: np.ndarray,
    reference_spectra: np.ndarray,
    query_names: list[str] | None = None,
    reference_names: list[str] | None = None,
) -> pd.DataFrame:
    query = as_2d_finite(query_spectra, "query_spectra")
    reference = as_2d_finite(
        reference_spectra,
        "reference_spectra",
    )
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference光谱长度不一致。")

    query_z = row_standardize(query)
    reference_z = row_standardize(reference)

    distance_matrix = cdist(
        query_z,
        reference_z,
        metric="euclidean",
    ) / np.sqrt(query.shape[1])

    nearest_indices = np.argmin(distance_matrix, axis=1)
    nearest_distance = distance_matrix[
        np.arange(query.shape[0]),
        nearest_indices,
    ]
    matched = reference[nearest_indices]

    pearson = row_pearson(query, matched)
    cosine = row_cosine(query, matched)
    rmse = row_rmse(query, matched)
    mae = row_mae(query, matched)
    derivative_corr = derivative_pearson(query, matched)
    angle = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    if query_names is None:
        query_names = [
            f"query_{index:04d}"
            for index in range(query.shape[0])
        ]
    if reference_names is None:
        reference_names = [
            f"reference_{index:04d}"
            for index in range(reference.shape[0])
        ]

    if len(query_names) != query.shape[0]:
        raise ValueError("query_names数量不正确。")
    if len(reference_names) != reference.shape[0]:
        raise ValueError("reference_names数量不正确。")

    return pd.DataFrame(
        {
            "query_index": np.arange(query.shape[0], dtype=int),
            "query_name": [str(v) for v in query_names],
            "nearest_reference_index": nearest_indices.astype(int),
            "nearest_reference_name": [
                str(reference_names[index])
                for index in nearest_indices
            ],
            "standardized_shape_rmse": nearest_distance,
            "pearson": pearson,
            "cosine": cosine,
            "spectral_angle_degree": angle,
            "raw_rmse": rmse,
            "raw_mae": mae,
            "first_derivative_pearson": derivative_corr,
        }
    )


def training_nearest_other_metrics(
    spectra: np.ndarray,
    names: list[str] | None = None,
) -> pd.DataFrame:
    values = as_2d_finite(spectra, "training_spectra")
    if values.shape[0] < 2:
        raise ValueError("至少需要2条训练谱计算nearest-other。")

    z = row_standardize(values)
    distances = cdist(z, z, metric="euclidean") / np.sqrt(
        values.shape[1]
    )
    np.fill_diagonal(distances, np.inf)
    nearest_indices = np.argmin(distances, axis=1)
    matched = values[nearest_indices]

    if names is None:
        names = [
            f"training_{index:04d}"
            for index in range(values.shape[0])
        ]

    pearson = row_pearson(values, matched)
    cosine = row_cosine(values, matched)

    return pd.DataFrame(
        {
            "query_index": np.arange(values.shape[0], dtype=int),
            "query_name": [str(v) for v in names],
            "nearest_reference_index": nearest_indices.astype(int),
            "nearest_reference_name": [
                str(names[index])
                for index in nearest_indices
            ],
            "standardized_shape_rmse": distances[
                np.arange(values.shape[0]),
                nearest_indices,
            ],
            "pearson": pearson,
            "cosine": cosine,
            "spectral_angle_degree": np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            ),
            "raw_rmse": row_rmse(values, matched),
            "raw_mae": row_mae(values, matched),
            "first_derivative_pearson": derivative_pearson(
                values,
                matched,
            ),
        }
    )


def nearest_metric_summary(
    *,
    training_nearest: pd.DataFrame,
    generated_nearest: pd.DataFrame,
    heldout_nearest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metrics = [
        "standardized_shape_rmse",
        "pearson",
        "cosine",
        "spectral_angle_degree",
        "raw_rmse",
        "raw_mae",
        "first_derivative_pearson",
    ]
    datasets: list[tuple[str, pd.DataFrame]] = [
        ("training_to_nearest_other_training", training_nearest),
        ("generated_to_nearest_training", generated_nearest),
    ]
    if heldout_nearest is not None and not heldout_nearest.empty:
        datasets.append(
            ("heldout_to_nearest_training", heldout_nearest)
        )

    rows: list[dict[str, Any]] = []
    for dataset_name, frame in datasets:
        for metric in metrics:
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "metric": metric,
                "number": int(frame.shape[0]),
            }
            row.update(summarize_vector(frame[metric].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PcaModel:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    score_std: np.ndarray


def fit_training_pca(
    training_spectra: np.ndarray,
    number_of_components: int,
) -> PcaModel:
    training = as_2d_finite(
        training_spectra,
        "training_spectra",
    )
    if training.shape[0] < 2:
        raise ValueError("PCA至少需要2条训练光谱。")

    requested = int(number_of_components)
    if requested <= 0:
        raise ValueError("PCA主成分数必须大于0。")

    mean = np.mean(training, axis=0)
    centered = training - mean
    _, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    maximum = min(
        requested,
        training.shape[0] - 1,
        vt.shape[0],
    )
    components = vt[:maximum].copy()

    total_variance = float(
        np.sum(np.square(singular_values))
    )
    if total_variance <= EPSILON:
        raise RuntimeError("训练光谱总方差过小，无法进行PCA。")

    explained_ratio = (
        np.square(singular_values[:maximum])
        / total_variance
    )

    training_scores = centered @ components.T
    score_std = _safe_std(
        training_scores,
        axis=0,
        ddof=1,
    )

    return PcaModel(
        mean=mean,
        components=components,
        explained_variance_ratio=explained_ratio,
        score_std=score_std,
    )


def transform_pca(
    model: PcaModel,
    spectra: np.ndarray,
) -> np.ndarray:
    values = as_2d_finite(spectra, "spectra")
    if values.shape[1] != model.mean.size:
        raise ValueError("PCA输入长度不正确。")
    return (values - model.mean) @ model.components.T


def pca_reconstruction_rmse(
    model: PcaModel,
    spectra: np.ndarray,
) -> np.ndarray:
    values = as_2d_finite(spectra, "spectra")
    scores = transform_pca(model, values)
    reconstructed = (
        model.mean[None, :]
        + scores @ model.components
    )
    return row_rmse(values, reconstructed)


def _pairwise_squared_distance(
    values: np.ndarray,
) -> np.ndarray:
    array = as_2d_finite(values, "values")
    squared_norm = np.sum(
        np.square(array),
        axis=1,
        keepdims=True,
    )
    distances = (
        squared_norm
        + squared_norm.T
        - 2.0 * array @ array.T
    )
    return np.maximum(distances, 0.0)


def training_mmd_bandwidth_squared(
    standardized_training_scores: np.ndarray,
) -> float:
    distances = _pairwise_squared_distance(
        standardized_training_scores
    )
    triangle = distances[
        np.triu_indices(
            distances.shape[0],
            k=1,
        )
    ]
    positive = triangle[
        np.isfinite(triangle)
        & (triangle > EPSILON)
    ]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def rbf_mmd2_biased(
    left: np.ndarray,
    right: np.ndarray,
    bandwidth_squared: float,
) -> float:
    x = as_2d_finite(left, "left")
    y = as_2d_finite(right, "right")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD两侧特征数不一致。")

    bandwidth_squared = max(
        float(bandwidth_squared),
        EPSILON,
    )

    xx = cdist(x, x, metric="sqeuclidean")
    yy = cdist(y, y, metric="sqeuclidean")
    xy = cdist(x, y, metric="sqeuclidean")

    kernel_xx = np.exp(
        -xx / (2.0 * bandwidth_squared)
    )
    kernel_yy = np.exp(
        -yy / (2.0 * bandwidth_squared)
    )
    kernel_xy = np.exp(
        -xy / (2.0 * bandwidth_squared)
    )

    value = (
        float(np.mean(kernel_xx))
        + float(np.mean(kernel_yy))
        - 2.0 * float(np.mean(kernel_xy))
    )
    return max(0.0, value)


def pca_distribution_evaluation(
    *,
    training_spectra: np.ndarray,
    generated_spectra: np.ndarray,
    heldout_spectra: np.ndarray | None,
    pca_components: int,
    bootstrap_repeats: int,
    random_seed: int,
) -> tuple[
    PcaModel,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    training = as_2d_finite(
        training_spectra,
        "training_spectra",
    )
    generated = as_2d_finite(
        generated_spectra,
        "generated_spectra",
    )
    if training.shape[1] != generated.shape[1]:
        raise ValueError("training/generated光谱长度不一致。")

    heldout: np.ndarray | None
    if heldout_spectra is None:
        heldout = None
    else:
        heldout = as_2d_finite(
            heldout_spectra,
            "heldout_spectra",
        )

    model = fit_training_pca(
        training,
        pca_components,
    )
    train_scores = transform_pca(model, training)
    gen_scores = transform_pca(model, generated)
    heldout_scores = (
        None
        if heldout is None
        else transform_pca(model, heldout)
    )

    rows: list[dict[str, Any]] = []
    for component_index in range(
        model.components.shape[0]
    ):
        train_pc = train_scores[:, component_index]
        gen_pc = gen_scores[:, component_index]
        scale = max(
            float(model.score_std[component_index]),
            EPSILON,
        )

        row: dict[str, Any] = {
            "pc": int(component_index + 1),
            "explained_variance_ratio": float(
                model.explained_variance_ratio[
                    component_index
                ]
            ),
            "training_mean": float(np.mean(train_pc)),
            "training_std": float(np.std(train_pc, ddof=1)),
            "generated_mean": float(np.mean(gen_pc)),
            "generated_std": float(np.std(gen_pc, ddof=1)),
            "generated_std_to_training": float(
                np.std(gen_pc, ddof=1) / scale
            ),
            "generated_mean_shift_in_training_std": float(
                (np.mean(gen_pc) - np.mean(train_pc))
                / scale
            ),
            "generated_wasserstein_in_training_std": float(
                wasserstein_distance(train_pc, gen_pc)
                / scale
            ),
        }

        if heldout_scores is not None:
            heldout_pc = heldout_scores[:, component_index]
            row["heldout_mean"] = float(
                np.mean(heldout_pc)
            )
            row["heldout_std"] = float(
                np.std(heldout_pc, ddof=0)
            )
            row["heldout_wasserstein_in_training_std"] = float(
                wasserstein_distance(
                    train_pc,
                    heldout_pc,
                )
                / scale
            )
        rows.append(row)

    pca_summary = pd.DataFrame(rows)

    score_columns = [
        f"PC{index + 1}"
        for index in range(model.components.shape[0])
    ]
    score_frames: list[pd.DataFrame] = []
    for dataset_name, scores in [
        ("training", train_scores),
        ("generated", gen_scores),
        ("heldout", heldout_scores),
    ]:
        if scores is None:
            continue
        frame = pd.DataFrame(scores, columns=score_columns)
        frame.insert(0, "dataset", dataset_name)
        frame.insert(
            1,
            "spectrum_index",
            np.arange(scores.shape[0], dtype=int),
        )
        score_frames.append(frame)
    pca_scores = pd.concat(
        score_frames,
        ignore_index=True,
    )

    train_standardized = (
        train_scores / model.score_std[None, :]
    )
    gen_standardized = (
        gen_scores / model.score_std[None, :]
    )
    bandwidth_squared = training_mmd_bandwidth_squared(
        train_standardized
    )

    repeats = int(bootstrap_repeats)
    if repeats <= 0:
        raise ValueError("bootstrap_repeats必须大于0。")

    rng = np.random.default_rng(int(random_seed))
    sample_size = training.shape[0]
    replace = generated.shape[0] < sample_size

    bootstrap_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        indices = rng.choice(
            generated.shape[0],
            size=sample_size,
            replace=replace,
        )
        sampled = gen_standardized[indices]

        mmd2 = rbf_mmd2_biased(
            train_standardized,
            sampled,
            bandwidth_squared,
        )

        wasserstein_values = [
            wasserstein_distance(
                train_standardized[:, pc],
                sampled[:, pc],
            )
            for pc in range(train_standardized.shape[1])
        ]

        bootstrap_rows.append(
            {
                "repeat": int(repeat),
                "mmd2_rbf_biased": float(mmd2),
                "wasserstein_pc_mean": float(
                    np.mean(wasserstein_values)
                ),
                "wasserstein_pc_median": float(
                    np.median(wasserstein_values)
                ),
            }
        )

    bootstrap_frame = pd.DataFrame(
        bootstrap_rows
    )

    distribution_rows: list[dict[str, Any]] = []
    for metric in (
        "mmd2_rbf_biased",
        "wasserstein_pc_mean",
        "wasserstein_pc_median",
    ):
        row = {
            "metric": metric,
            "matched_generated_sample_size": int(
                sample_size
            ),
            "bootstrap_repeats": int(repeats),
        }
        row.update(
            summarize_vector(
                bootstrap_frame[metric].to_numpy()
            )
        )
        distribution_rows.append(row)

    distribution_summary = pd.DataFrame(
        distribution_rows
    )

    reconstruction_rows: list[dict[str, Any]] = []
    for dataset_name, spectra in [
        ("training", training),
        ("generated", generated),
        ("heldout", heldout),
    ]:
        if spectra is None:
            continue
        values = pca_reconstruction_rmse(
            model,
            spectra,
        )
        row = {
            "dataset": dataset_name,
            "number": int(spectra.shape[0]),
        }
        row.update(summarize_vector(values))
        reconstruction_rows.append(row)

    reconstruction_summary = pd.DataFrame(
        reconstruction_rows
    )

    return (
        model,
        pca_summary,
        pca_scores,
        distribution_summary,
        reconstruction_summary,
    )


def build_nonpeak_mask(
    *,
    raman_shift: np.ndarray,
    peak_centers_cm1: list[float],
    half_width_cm1: float,
) -> np.ndarray:
    axis = as_axis(raman_shift)
    mask = np.ones(axis.size, dtype=bool)
    half_width = float(half_width_cm1)
    if half_width <= 0.0:
        raise ValueError("half_width_cm1必须大于0。")
    for center in peak_centers_cm1:
        mask &= np.abs(axis - float(center)) > half_width
    if not np.any(mask):
        raise RuntimeError("非峰区mask为空。")
    return mask


def pointwise_distribution_metrics(
    *,
    spectra: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    values = as_2d_finite(spectra, "spectra")
    point_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if point_mask.size != values.shape[1]:
        raise ValueError("pointwise distribution mask长度不正确。")
    if not np.any(point_mask):
        raise ValueError("pointwise distribution mask为空。")

    selected = values[:, point_mask]
    band95 = (
        np.percentile(selected, 97.5, axis=0)
        - np.percentile(selected, 2.5, axis=0)
    )
    iqr = (
        np.percentile(selected, 75.0, axis=0)
        - np.percentile(selected, 25.0, axis=0)
    )
    std = np.std(selected, axis=0, ddof=0)

    return {
        "band95_median": float(np.median(band95)),
        "iqr_median": float(np.median(iqr)),
        "pointwise_std_mean": float(np.mean(std)),
    }


def broad_local_distribution_evaluation(
    *,
    training_spectra: np.ndarray,
    generated_spectra: np.ndarray,
    raman_shift: np.ndarray,
    nonpeak_mask: np.ndarray,
    broad_sigma_cm1: float,
    bootstrap_repeats: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = as_2d_finite(
        training_spectra,
        "training_spectra",
    )
    generated = as_2d_finite(
        generated_spectra,
        "generated_spectra",
    )
    axis = as_axis(
        raman_shift,
        expected_length=training.shape[1],
    )
    if generated.shape[1] != training.shape[1]:
        raise ValueError("training/generated光谱长度不一致。")

    spacing = float(np.median(np.diff(axis)))
    sigma_points = float(broad_sigma_cm1) / spacing
    if sigma_points <= 0.0:
        raise ValueError("broad_sigma_cm1必须大于0。")

    training_broad = gaussian_filter1d(
        training,
        sigma=sigma_points,
        axis=1,
        mode="nearest",
    )
    generated_broad = gaussian_filter1d(
        generated,
        sigma=sigma_points,
        axis=1,
        mode="nearest",
    )
    training_local = training - training_broad
    generated_local = generated - generated_broad

    real_metrics = {
        "nonpeak": pointwise_distribution_metrics(
            spectra=training,
            mask=nonpeak_mask,
        ),
        "broad": pointwise_distribution_metrics(
            spectra=training_broad,
            mask=nonpeak_mask,
        ),
        "local": pointwise_distribution_metrics(
            spectra=training_local,
            mask=nonpeak_mask,
        ),
    }

    generated_metrics = {
        "nonpeak": pointwise_distribution_metrics(
            spectra=generated,
            mask=nonpeak_mask,
        ),
        "broad": pointwise_distribution_metrics(
            spectra=generated_broad,
            mask=nonpeak_mask,
        ),
        "local": pointwise_distribution_metrics(
            spectra=generated_local,
            mask=nonpeak_mask,
        ),
    }

    rows: list[dict[str, Any]] = []
    for domain in ("nonpeak", "broad", "local"):
        for metric in (
            "band95_median",
            "iqr_median",
            "pointwise_std_mean",
        ):
            real_value = real_metrics[domain][metric]
            generated_value = generated_metrics[domain][metric]
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "training_real": real_value,
                    "generated_all": generated_value,
                    "generated_to_real_ratio": (
                        generated_value
                        / max(abs(real_value), EPSILON)
                    ),
                }
            )

    full_summary = pd.DataFrame(rows)

    repeats = int(bootstrap_repeats)
    rng = np.random.default_rng(int(random_seed))
    sample_size = training.shape[0]
    replace = generated.shape[0] < sample_size

    bootstrap_records: list[dict[str, Any]] = []
    for repeat in range(repeats):
        indices = rng.choice(
            generated.shape[0],
            size=sample_size,
            replace=replace,
        )
        sampled = generated[indices]
        sampled_broad = generated_broad[indices]
        sampled_local = generated_local[indices]

        sampled_domains = {
            "nonpeak": sampled,
            "broad": sampled_broad,
            "local": sampled_local,
        }

        for domain in ("nonpeak", "broad", "local"):
            metrics = pointwise_distribution_metrics(
                spectra=sampled_domains[domain],
                mask=nonpeak_mask,
            )
            for metric, value in metrics.items():
                real_value = real_metrics[domain][metric]
                bootstrap_records.append(
                    {
                        "repeat": int(repeat),
                        "domain": domain,
                        "metric": metric,
                        "generated_value": float(value),
                        "real_value": float(real_value),
                        "ratio": float(
                            value
                            / max(abs(real_value), EPSILON)
                        ),
                    }
                )

    bootstrap_all = pd.DataFrame(
        bootstrap_records
    )

    bootstrap_summary_rows: list[dict[str, Any]] = []
    for (domain, metric), group in bootstrap_all.groupby(
        ["domain", "metric"],
        sort=False,
    ):
        ratio = group["ratio"].to_numpy()
        row = {
            "domain": domain,
            "metric": metric,
            "training_real": float(
                group["real_value"].iloc[0]
            ),
            "bootstrap_repeats": int(repeats),
            "generated_matched_median": float(
                np.median(
                    group["generated_value"].to_numpy()
                )
            ),
        }
        summary = summarize_vector(ratio, prefix="ratio_")
        row.update(summary)
        bootstrap_summary_rows.append(row)

    bootstrap_summary = pd.DataFrame(
        bootstrap_summary_rows
    )
    return full_summary, bootstrap_summary


def _crossing_position(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
) -> float:
    denominator = y1 - y0
    if abs(denominator) <= EPSILON:
        return 0.5 * (x0 + x1)
    fraction = np.clip(
        (target - y0) / denominator,
        0.0,
        1.0,
    )
    return float(
        x0 + float(fraction) * (x1 - x0)
    )


def measure_peak(
    *,
    raman_shift: np.ndarray,
    peak_signal: np.ndarray,
    center_cm1: float,
    half_width_cm1: float,
) -> dict[str, float]:
    axis = as_axis(raman_shift)
    signal = np.asarray(
        peak_signal,
        dtype=np.float64,
    ).reshape(-1)
    if signal.size != axis.size:
        raise ValueError("peak_signal长度不正确。")

    mask = (
        (axis >= center_cm1 - half_width_cm1)
        & (axis <= center_cm1 + half_width_cm1)
    )
    indices = np.flatnonzero(mask)
    if indices.size < 5:
        return {
            "position_cm1": np.nan,
            "height": np.nan,
            "fwhm_cm1": np.nan,
        }

    x = axis[indices]
    y = signal[indices]
    local_index = int(np.argmax(y))
    height = float(y[local_index])
    position = float(x[local_index])

    if not np.isfinite(height) or height <= 0.0:
        return {
            "position_cm1": position,
            "height": height,
            "fwhm_cm1": np.nan,
        }

    half_height = 0.5 * height

    left = local_index
    while left > 0 and y[left] > half_height:
        left -= 1

    right = local_index
    while (
        right < y.size - 1
        and y[right] > half_height
    ):
        right += 1

    if (
        (left == 0 and y[left] > half_height)
        or (
            right == y.size - 1
            and y[right] > half_height
        )
    ):
        width = np.nan
    else:
        left_cross = _crossing_position(
            float(x[left]),
            float(y[left]),
            float(x[left + 1]),
            float(y[left + 1]),
            half_height,
        )
        right_cross = _crossing_position(
            float(x[right - 1]),
            float(y[right - 1]),
            float(x[right]),
            float(y[right]),
            half_height,
        )
        width = max(
            0.0,
            right_cross - left_cross,
        )

    return {
        "position_cm1": position,
        "height": height,
        "fwhm_cm1": float(width),
    }


def measure_peak_table(
    *,
    spectra: np.ndarray,
    spectrum_names: list[str],
    dataset_name: str,
    raman_shift: np.ndarray,
    peak_centers_cm1: list[float],
    smoothing_sigma_cm1: float,
    half_width_cm1: float,
) -> pd.DataFrame:
    values = as_2d_finite(spectra, "spectra")
    axis = as_axis(
        raman_shift,
        expected_length=values.shape[1],
    )
    if len(spectrum_names) != values.shape[0]:
        raise ValueError("spectrum_names数量不正确。")

    spacing = float(np.median(np.diff(axis)))
    sigma_points = float(smoothing_sigma_cm1) / spacing
    baseline = gaussian_filter1d(
        values,
        sigma=sigma_points,
        axis=1,
        mode="nearest",
    )
    peak_signal = values - baseline

    records: list[dict[str, Any]] = []
    for spectrum_index in range(values.shape[0]):
        for peak_index, center in enumerate(
            peak_centers_cm1,
            start=1,
        ):
            measured = measure_peak(
                raman_shift=axis,
                peak_signal=peak_signal[spectrum_index],
                center_cm1=float(center),
                half_width_cm1=float(half_width_cm1),
            )
            records.append(
                {
                    "dataset": str(dataset_name),
                    "spectrum_index": int(spectrum_index),
                    "spectrum_name": str(
                        spectrum_names[spectrum_index]
                    ),
                    "peak_id": int(peak_index),
                    "reference_position_cm1": float(center),
                    **measured,
                }
            )
    return pd.DataFrame(records)


def peak_distribution_comparison(
    *,
    training_peak_table: pd.DataFrame,
    generated_peak_table: pd.DataFrame,
    heldout_peak_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    peak_ids = sorted(
        set(training_peak_table["peak_id"].tolist())
    )

    rows: list[dict[str, Any]] = []
    for peak_id in peak_ids:
        train = training_peak_table[
            training_peak_table["peak_id"] == peak_id
        ]
        generated = generated_peak_table[
            generated_peak_table["peak_id"] == peak_id
        ]
        center = float(
            train["reference_position_cm1"].iloc[0]
        )

        row: dict[str, Any] = {
            "peak_id": int(peak_id),
            "reference_position_cm1": center,
        }

        for metric in (
            "position_cm1",
            "height",
            "fwhm_cm1",
        ):
            train_values = (
                train[metric]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=np.float64)
            )
            gen_values = (
                generated[metric]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=np.float64)
            )

            row[f"training_{metric}_valid_rate"] = float(
                train_values.size
                / max(train.shape[0], 1)
            )
            row[f"generated_{metric}_valid_rate"] = float(
                gen_values.size
                / max(generated.shape[0], 1)
            )

            if train_values.size and gen_values.size:
                train_median = float(
                    np.median(train_values)
                )
                gen_median = float(
                    np.median(gen_values)
                )
                train_std = float(
                    np.std(train_values, ddof=0)
                )
                gen_std = float(
                    np.std(gen_values, ddof=0)
                )

                row[f"training_{metric}_median"] = train_median
                row[f"generated_{metric}_median"] = gen_median
                row[f"training_{metric}_std"] = train_std
                row[f"generated_{metric}_std"] = gen_std
                row[f"{metric}_median_difference"] = (
                    gen_median - train_median
                )
                row[f"{metric}_std_ratio"] = (
                    gen_std / max(train_std, EPSILON)
                )

                if metric != "position_cm1":
                    row[f"{metric}_median_ratio"] = (
                        gen_median
                        / max(abs(train_median), EPSILON)
                    )

        if heldout_peak_table is not None:
            heldout = heldout_peak_table[
                heldout_peak_table["peak_id"] == peak_id
            ]
            for metric in (
                "position_cm1",
                "height",
                "fwhm_cm1",
            ):
                values = (
                    heldout[metric]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )
                row[f"heldout_{metric}_valid_rate"] = float(
                    values.size
                    / max(heldout.shape[0], 1)
                )
                if values.size:
                    row[f"heldout_{metric}_median"] = float(
                        np.median(values)
                    )
                    row[f"heldout_{metric}_std"] = float(
                        np.std(values, ddof=0)
                    )

        rows.append(row)

    return pd.DataFrame(rows)


def sample_pair_metrics(
    *,
    spectra: np.ndarray,
    pair_count: int,
    random_seed: int,
    all_pairs_if_small: bool,
) -> pd.DataFrame:
    values = as_2d_finite(spectra, "spectra")
    number = values.shape[0]
    if number < 2:
        raise ValueError("pair metrics至少需要2条光谱。")

    if all_pairs_if_small:
        left_indices, right_indices = np.triu_indices(
            number,
            k=1,
        )
    else:
        requested = int(pair_count)
        if requested <= 0:
            raise ValueError("pair_count必须大于0。")
        rng = np.random.default_rng(int(random_seed))
        left_indices = rng.integers(
            0,
            number,
            size=requested,
        )
        right_indices = rng.integers(
            0,
            number - 1,
            size=requested,
        )
        right_indices = np.where(
            right_indices >= left_indices,
            right_indices + 1,
            right_indices,
        )

    # 生成谱可能需要抽取数万对。这里分块计算，
    # 避免一次复制 pair_count × Raman_length 的巨大数组。
    chunk_size = 1024
    frames: list[pd.DataFrame] = []

    for start in range(
        0,
        left_indices.size,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            left_indices.size,
        )

        current_left_indices = left_indices[
            start:end
        ]
        current_right_indices = right_indices[
            start:end
        ]

        left = values[
            current_left_indices
        ]
        right = values[
            current_right_indices
        ]

        left_z = row_standardize(left)
        right_z = row_standardize(right)

        standardized_rmse = row_rmse(
            left_z,
            right_z,
        )
        pearson = row_pearson(
            left,
            right,
        )

        frames.append(
            pd.DataFrame(
                {
                    "left_index": (
                        current_left_indices.astype(int)
                    ),
                    "right_index": (
                        current_right_indices.astype(int)
                    ),
                    "standardized_shape_rmse": (
                        standardized_rmse
                    ),
                    "pearson": pearson,
                    "pearson_distance": 1.0 - pearson,
                    "raw_rmse": row_rmse(
                        left,
                        right,
                    ),
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def diversity_summary(
    *,
    training_spectra: np.ndarray,
    generated_spectra: np.ndarray,
    generated_pair_count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_pairs = sample_pair_metrics(
        spectra=training_spectra,
        pair_count=1,
        random_seed=random_seed,
        all_pairs_if_small=True,
    )
    generated_pairs = sample_pair_metrics(
        spectra=generated_spectra,
        pair_count=generated_pair_count,
        random_seed=random_seed,
        all_pairs_if_small=False,
    )

    rows: list[dict[str, Any]] = []
    for metric in (
        "standardized_shape_rmse",
        "pearson",
        "pearson_distance",
        "raw_rmse",
    ):
        train_values = training_pairs[metric].to_numpy()
        gen_values = generated_pairs[metric].to_numpy()

        row: dict[str, Any] = {
            "metric": metric,
            "training_pair_count": int(
                training_pairs.shape[0]
            ),
            "generated_pair_count": int(
                generated_pairs.shape[0]
            ),
        }
        row.update(
            {
                f"training_{key}": value
                for key, value in summarize_vector(
                    train_values
                ).items()
            }
        )
        row.update(
            {
                f"generated_{key}": value
                for key, value in summarize_vector(
                    gen_values
                ).items()
            }
        )
        train_median = float(np.median(train_values))
        gen_median = float(np.median(gen_values))
        row["generated_median_to_training_median"] = (
            gen_median
            / max(abs(train_median), EPSILON)
        )
        rows.append(row)

    return (
        pd.DataFrame(rows),
        training_pairs,
        generated_pairs,
    )


def replication_screening_summary(
    *,
    training_nearest: pd.DataFrame,
    generated_nearest: pd.DataFrame,
) -> pd.DataFrame:
    train_distance = training_nearest[
        "standardized_shape_rmse"
    ].to_numpy(dtype=np.float64)
    gen_distance = generated_nearest[
        "standardized_shape_rmse"
    ].to_numpy(dtype=np.float64)

    real_min = float(np.min(train_distance))
    real_p05 = float(np.percentile(train_distance, 5.0))
    real_median = float(np.median(train_distance))
    gen_median = float(np.median(gen_distance))

    generated_below_real_min = int(
        np.count_nonzero(gen_distance < real_min)
    )
    generated_below_real_p05 = int(
        np.count_nonzero(gen_distance < real_p05)
    )
    exact_like = int(
        np.count_nonzero(gen_distance <= 1.0e-6)
    )

    rows = [
        {
            "metric": "training_nearest_other_shape_rmse_min",
            "value": real_min,
            "note": "训练谱之间最近邻距离下界，仅用于距离筛查。",
        },
        {
            "metric": "training_nearest_other_shape_rmse_p05",
            "value": real_p05,
            "note": "训练谱最近邻距离5百分位，仅用于距离筛查。",
        },
        {
            "metric": "training_nearest_other_shape_rmse_median",
            "value": real_median,
            "note": "训练谱最近邻距离中位数。",
        },
        {
            "metric": "generated_nearest_training_shape_rmse_median",
            "value": gen_median,
            "note": "生成谱到最近训练谱的标准化形状RMSE中位数。",
        },
        {
            "metric": "generated_to_training_nearest_median_ratio",
            "value": (
                gen_median
                / max(real_median, EPSILON)
            ),
            "note": "过小可能提示复制风险；过大可能提示偏离真实流形，不能单独作为判据。",
        },
        {
            "metric": "generated_count_below_training_nearest_min",
            "value": float(generated_below_real_min),
            "note": "不是复制证明，仅表示比任意训练-训练最近邻更接近训练样本。",
        },
        {
            "metric": "generated_percent_below_training_nearest_min",
            "value": float(
                100.0
                * generated_below_real_min
                / max(gen_distance.size, 1)
            ),
            "note": "距离筛查比例。",
        },
        {
            "metric": "generated_count_below_training_nearest_p05",
            "value": float(generated_below_real_p05),
            "note": "距离筛查数量。",
        },
        {
            "metric": "generated_percent_below_training_nearest_p05",
            "value": float(
                100.0
                * generated_below_real_p05
                / max(gen_distance.size, 1)
            ),
            "note": "距离筛查比例。",
        },
        {
            "metric": "generated_exact_like_count_shape_rmse_le_1e-6",
            "value": float(exact_like),
            "note": "极严格的数值近重复筛查。",
        },
    ]
    return pd.DataFrame(rows)


def mean_spectrum_comparison(
    *,
    training_spectra: np.ndarray,
    generated_spectra: np.ndarray,
    heldout_spectra: np.ndarray | None,
) -> pd.DataFrame:
    training = as_2d_finite(
        training_spectra,
        "training_spectra",
    )
    generated = as_2d_finite(
        generated_spectra,
        "generated_spectra",
    )

    train_mean = np.mean(training, axis=0)
    generated_mean = np.mean(generated, axis=0)

    def compare(
        name: str,
        first: np.ndarray,
        second: np.ndarray,
    ) -> dict[str, Any]:
        left = first[None, :]
        right = second[None, :]
        cosine = float(row_cosine(left, right)[0])
        return {
            "comparison": name,
            "pearson": float(row_pearson(left, right)[0]),
            "cosine": cosine,
            "spectral_angle_degree": float(
                np.degrees(
                    np.arccos(
                        np.clip(cosine, -1.0, 1.0)
                    )
                )
            ),
            "rmse": float(row_rmse(left, right)[0]),
            "mae": float(row_mae(left, right)[0]),
            "first_derivative_pearson": float(
                derivative_pearson(left, right)[0]
            ),
        }

    rows = [
        compare(
            "training_mean_vs_generated_mean",
            train_mean,
            generated_mean,
        )
    ]

    if heldout_spectra is not None:
        heldout = as_2d_finite(
            heldout_spectra,
            "heldout_spectra",
        )
        heldout_mean = np.mean(heldout, axis=0)
        rows.append(
            compare(
                "training_mean_vs_heldout_mean",
                train_mean,
                heldout_mean,
            )
        )
        rows.append(
            compare(
                "heldout_mean_vs_generated_mean",
                heldout_mean,
                generated_mean,
            )
        )

    return pd.DataFrame(rows)

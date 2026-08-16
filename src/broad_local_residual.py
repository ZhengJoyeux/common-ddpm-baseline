"""D2.6 broad-local residual decomposition for SERS spectra.

Purpose
-------
D2.5 models a full residual with one DDPM:

    spectrum = PCA prior + residual

Experiment D showed that the local/peak part is already reasonable while the
broad/background tail distribution is still too wide.  D2.6 therefore splits
the train-only residual into:

    residual = broad_residual + local_residual

The smooth broad residual is modeled by a train-only PCA score distribution.
The DDPM learns only the local residual.

Important
---------
1. No pesticide peak position is hard-coded.
2. The Gaussian low-pass width is defined in physical Raman-shift units.
3. Every fitted statistic comes from the training split only.
4. This module does not change the outer PCA prior used by D2.5.
5. This module does not perform post-hoc spectrum calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr, ndtri


_SCHEMA_VERSION = 1
_DOMAIN = "pca_raw_residual_on_global_minmax_normalized_spectrum"


def _as_2d_finite(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name}必须是二维数组[N,L]，实际={array.shape}。"
        )

    if (
        array.shape[0] < 1
        or array.shape[1] < 3
    ):
        raise ValueError(
            f"{name}形状无效：{array.shape}。"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}包含NaN或无穷值。"
        )

    return array


def _as_axis(
    values: np.ndarray,
    *,
    expected_length: int,
) -> np.ndarray:
    axis = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if axis.size != expected_length:
        raise ValueError(
            "Raman shift轴长度与光谱长度不一致："
            f"{axis.size} != {expected_length}。"
        )

    if axis.size < 3:
        raise ValueError(
            "Raman shift轴至少需要3个点。"
        )

    if not np.isfinite(axis).all():
        raise ValueError(
            "Raman shift轴包含NaN或无穷值。"
        )

    if not np.all(
        np.diff(axis) > 0.0
    ):
        raise ValueError(
            "Raman shift轴必须严格递增。"
        )

    return axis


def _positive_float(
    value: Any,
    name: str,
) -> float:
    parsed = float(value)

    if (
        not np.isfinite(parsed)
        or parsed <= 0.0
    ):
        raise ValueError(
            f"{name}必须是有限正数。"
        )

    return parsed


def _probability(
    value: Any,
    name: str,
) -> float:
    parsed = float(value)

    if (
        not np.isfinite(parsed)
        or not 0.0 < parsed <= 1.0
    ):
        raise ValueError(
            f"{name}必须位于(0,1]。"
        )

    return parsed


def _percentile_dictionary(
    values: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "无法统计空数组。"
        )

    percentiles = np.percentile(
        array,
        [
            50.0,
            90.0,
            95.0,
            99.0,
            99.5,
            99.9,
            100.0,
        ],
    )

    return {
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99_5": float(percentiles[4]),
        "p99_9": float(percentiles[5]),
        "max": float(percentiles[6]),
    }


@dataclass
class BroadLocalResidualDecomposer:
    """Train-only broad/local residual model used by D2.6."""

    broad_sigma_cm1: float = 7.0
    broad_truncate: float = 4.0

    broad_pca_explained_variance_ratio: float = 0.95
    broad_pca_max_components: int | None = 6
    broad_score_clip_standard_deviations: float = 2.5

    local_residual_quantile: float = 99.5
    local_target_abs_max: float = 1.0

    epsilon: float = 1.0e-8

    # fitted state
    raman_shift: np.ndarray | None = None
    uniform_spacing_cm1: float | None = None
    broad_sigma_points: float | None = None
    number_of_training_spectra: int | None = None

    broad_pca_mean: np.ndarray | None = None
    broad_pca_components: np.ndarray | None = None
    broad_training_score_mean: np.ndarray | None = None
    broad_score_standard_deviation: np.ndarray | None = None
    broad_explained_variance: np.ndarray | None = None
    broad_explained_variance_ratio_: np.ndarray | None = None

    local_residual_scale: float | None = None
    local_asinh_normalizer: float | None = None
    local_training_max_abs_residual: float | None = None

    training_raw_abs_percentiles: dict[str, float] | None = None
    training_broad_abs_percentiles: dict[str, float] | None = None
    training_local_abs_percentiles: dict[str, float] | None = None

    training_raw_rms_median: float | None = None
    training_broad_rms_median: float | None = None
    training_local_rms_median: float | None = None

    def __post_init__(self) -> None:
        self.broad_sigma_cm1 = _positive_float(
            self.broad_sigma_cm1,
            "broad_sigma_cm1",
        )

        self.broad_truncate = _positive_float(
            self.broad_truncate,
            "broad_truncate",
        )

        self.broad_pca_explained_variance_ratio = (
            _probability(
                self.broad_pca_explained_variance_ratio,
                "broad_pca_explained_variance_ratio",
            )
        )

        if self.broad_pca_max_components is not None:
            self.broad_pca_max_components = int(
                self.broad_pca_max_components
            )

            if self.broad_pca_max_components <= 0:
                raise ValueError(
                    "broad_pca_max_components必须为正整数或null。"
                )

        self.broad_score_clip_standard_deviations = (
            _positive_float(
                self.broad_score_clip_standard_deviations,
                "broad_score_clip_standard_deviations",
            )
        )

        self.local_residual_quantile = float(
            self.local_residual_quantile
        )

        if not (
            0.0
            < self.local_residual_quantile
            < 100.0
        ):
            raise ValueError(
                "local_residual_quantile必须位于(0,100)。"
            )

        self.local_target_abs_max = float(
            self.local_target_abs_max
        )

        if not (
            0.0
            < self.local_target_abs_max
            <= 1.0
        ):
            raise ValueError(
                "local_target_abs_max必须位于(0,1]。"
            )

        self.epsilon = _positive_float(
            self.epsilon,
            "epsilon",
        )

    @classmethod
    def from_configuration(
        cls,
        configuration: dict[str, Any],
    ) -> "BroadLocalResidualDecomposer":
        if not isinstance(
            configuration,
            dict,
        ):
            raise TypeError(
                "broad_local_residual配置必须是字典。"
            )

        if not bool(
            configuration.get(
                "enabled",
                False,
            )
        ):
            raise ValueError(
                "broad_local_residual没有启用。"
            )

        broad_filter = (
            configuration.get(
                "broad_filter",
                {},
            )
            or {}
        )

        broad_prior = (
            configuration.get(
                "broad_prior",
                {},
            )
            or {}
        )

        local_normalization = (
            configuration.get(
                "local_normalization",
                {},
            )
            or {}
        )

        if not isinstance(
            broad_filter,
            dict,
        ):
            raise TypeError(
                "broad_filter必须是字典。"
            )

        if not isinstance(
            broad_prior,
            dict,
        ):
            raise TypeError(
                "broad_prior必须是字典。"
            )

        if not isinstance(
            local_normalization,
            dict,
        ):
            raise TypeError(
                "local_normalization必须是字典。"
            )

        method = str(
            broad_filter.get(
                "method",
                "gaussian",
            )
        ).strip().lower()

        if method != "gaussian":
            raise ValueError(
                "D2.6 broad_filter当前只支持gaussian。"
            )

        broad_prior_method = str(
            broad_prior.get(
                "method",
                "pca",
            )
        ).strip().lower()

        if broad_prior_method != "pca":
            raise ValueError(
                "D2.6 broad_prior当前只支持pca。"
            )

        sampling_strategy = str(
            broad_prior.get(
                "sampling_strategy",
                "independent_truncated_gaussian_scores",
            )
        ).strip().lower()

        if (
            sampling_strategy
            != "independent_truncated_gaussian_scores"
        ):
            raise ValueError(
                "D2.6 broad_prior.sampling_strategy当前只支持"
                "independent_truncated_gaussian_scores。"
            )

        local_method = str(
            local_normalization.get(
                "method",
                "robust_asinh",
            )
        ).strip().lower()

        if local_method != "robust_asinh":
            raise ValueError(
                "D2.6 local_normalization当前只支持robust_asinh。"
            )

        return cls(
            broad_sigma_cm1=float(
                broad_filter.get(
                    "sigma_cm1",
                    7.0,
                )
            ),
            broad_truncate=float(
                broad_filter.get(
                    "truncate",
                    4.0,
                )
            ),
            broad_pca_explained_variance_ratio=float(
                broad_prior.get(
                    "pca_explained_variance_ratio",
                    0.95,
                )
            ),
            broad_pca_max_components=(
                broad_prior.get(
                    "pca_max_components",
                    6,
                )
            ),
            broad_score_clip_standard_deviations=float(
                broad_prior.get(
                    "score_clip_standard_deviations",
                    2.5,
                )
            ),
            local_residual_quantile=float(
                local_normalization.get(
                    "residual_quantile",
                    99.5,
                )
            ),
            local_target_abs_max=float(
                local_normalization.get(
                    "target_abs_max",
                    1.0,
                )
            ),
            epsilon=float(
                configuration.get(
                    "epsilon",
                    1.0e-8,
                )
            ),
        )

    def fit(
        self,
        training_raw_residuals: np.ndarray,
        *,
        raman_shift: np.ndarray,
    ) -> "BroadLocalResidualDecomposer":
        values = _as_2d_finite(
            training_raw_residuals,
            "training_raw_residuals",
        )

        if values.shape[0] < 4:
            raise ValueError(
                "D2.6至少需要4条训练残差。"
            )

        axis = _as_axis(
            raman_shift,
            expected_length=values.shape[1],
        )

        broad, local = self._split_with_axis(
            values,
            axis,
        )

        self.raman_shift = axis.copy()
        self.number_of_training_spectra = int(
            values.shape[0]
        )

        self._fit_broad_pca(
            broad
        )

        self._fit_local_robust_asinh(
            local
        )

        self.training_raw_abs_percentiles = (
            _percentile_dictionary(
                np.abs(values)
            )
        )

        self.training_broad_abs_percentiles = (
            _percentile_dictionary(
                np.abs(broad)
            )
        )

        self.training_local_abs_percentiles = (
            _percentile_dictionary(
                np.abs(local)
            )
        )

        self.training_raw_rms_median = float(
            np.median(
                np.sqrt(
                    np.mean(
                        np.square(values),
                        axis=1,
                    )
                )
            )
        )

        self.training_broad_rms_median = float(
            np.median(
                np.sqrt(
                    np.mean(
                        np.square(broad),
                        axis=1,
                    )
                )
            )
        )

        self.training_local_rms_median = float(
            np.median(
                np.sqrt(
                    np.mean(
                        np.square(local),
                        axis=1,
                    )
                )
            )
        )

        return self

    def split_raw_residuals(
        self,
        raw_residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()

        values = _as_2d_finite(
            raw_residuals,
            "raw_residuals",
        )

        if values.shape[1] != self.raman_shift.size:
            raise ValueError(
                "raw_residuals长度与D2.6训练Raman轴不一致。"
            )

        return self._split_with_axis(
            values,
            self.raman_shift,
        )

    def transform_raw_residuals(
        self,
        raw_residuals: np.ndarray,
    ) -> np.ndarray:
        """Return only the scaled local residual learned by the DDPM."""

        _, local = self.split_raw_residuals(
            raw_residuals
        )

        scaled = (
            self.local_target_abs_max
            * np.arcsinh(
                local
                / float(
                    self.local_residual_scale
                )
            )
            / float(
                self.local_asinh_normalizer
            )
        )

        if not np.isfinite(
            scaled
        ).all():
            raise RuntimeError(
                "D2.6 local residual变换出现NaN或无穷值。"
            )

        return scaled.astype(
            np.float32,
            copy=False,
        )

    def inverse_local_transform(
        self,
        scaled_local_residuals: np.ndarray,
    ) -> np.ndarray:
        """Decode the DDPM output to an unscaled local residual."""

        self._check_fitted()

        values = _as_2d_finite(
            scaled_local_residuals,
            "scaled_local_residuals",
        )

        if values.shape[1] != self.raman_shift.size:
            raise ValueError(
                "scaled_local_residuals长度与D2.6训练轴不一致。"
            )

        argument = (
            values
            / self.local_target_abs_max
            * float(
                self.local_asinh_normalizer
            )
        )

        local = (
            float(
                self.local_residual_scale
            )
            * np.sinh(
                argument
            )
        )

        if not np.isfinite(
            local
        ).all():
            raise RuntimeError(
                "D2.6 local residual逆变换出现NaN或无穷值。"
            )

        return local.astype(
            np.float32,
            copy=False,
        )

    def sample_broad_residuals(
        self,
        number_of_spectra: int,
        *,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Sample smooth broad residuals from train-only broad PCA scores."""

        self._check_fitted()

        count = int(
            number_of_spectra
        )

        if count <= 0:
            raise ValueError(
                "number_of_spectra必须大于0。"
            )

        if not isinstance(
            random_generator,
            np.random.Generator,
        ):
            raise TypeError(
                "random_generator必须是numpy.random.Generator。"
            )

        clip = float(
            self.broad_score_clip_standard_deviations
        )

        lower_cdf = float(
            ndtr(-clip)
        )

        upper_cdf = float(
            ndtr(clip)
        )

        probabilities = (
            random_generator.uniform(
                low=lower_cdf,
                high=upper_cdf,
                size=(
                    count,
                    self.broad_pca_components.shape[0],
                ),
            )
        )

        standardized_scores = (
            ndtri(
                probabilities
            )
        )

        scores = (
            self.broad_training_score_mean[
                np.newaxis,
                :
            ]
            + standardized_scores
            * self.broad_score_standard_deviation[
                np.newaxis,
                :
            ]
        )

        broad = (
            self.broad_pca_mean[
                np.newaxis,
                :
            ]
            + scores
            @ self.broad_pca_components
        )

        if not np.isfinite(
            broad
        ).all():
            raise RuntimeError(
                "D2.6 broad residual采样包含NaN或无穷值。"
            )

        return broad.astype(
            np.float32,
            copy=False,
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        self._check_fitted()

        return {
            "schema_version": _SCHEMA_VERSION,
            "enabled": True,
            "domain": _DOMAIN,
            "number_of_training_spectra": int(
                self.number_of_training_spectra
            ),
            "raman_shift": (
                self.raman_shift.tolist()
            ),
            "broad_filter": {
                "method": "gaussian",
                "sigma_cm1": float(
                    self.broad_sigma_cm1
                ),
                "truncate": float(
                    self.broad_truncate
                ),
                "uniform_spacing_cm1": float(
                    self.uniform_spacing_cm1
                ),
                "sigma_points": float(
                    self.broad_sigma_points
                ),
            },
            "broad_prior": {
                "method": "pca",
                "sampling_strategy": (
                    "independent_truncated_gaussian_scores"
                ),
                "pca_explained_variance_ratio_target": float(
                    self.broad_pca_explained_variance_ratio
                ),
                "pca_max_components": (
                    self.broad_pca_max_components
                ),
                "score_clip_standard_deviations": float(
                    self.broad_score_clip_standard_deviations
                ),
                "mean": (
                    self.broad_pca_mean.tolist()
                ),
                "components": (
                    self.broad_pca_components.tolist()
                ),
                "training_score_mean": (
                    self.broad_training_score_mean.tolist()
                ),
                "score_standard_deviation": (
                    self.broad_score_standard_deviation.tolist()
                ),
                "explained_variance": (
                    self.broad_explained_variance.tolist()
                ),
                "explained_variance_ratio": (
                    self.broad_explained_variance_ratio_.tolist()
                ),
            },
            "local_normalization": {
                "method": "robust_asinh",
                "residual_quantile": float(
                    self.local_residual_quantile
                ),
                "target_abs_max": float(
                    self.local_target_abs_max
                ),
                "scale": float(
                    self.local_residual_scale
                ),
                "asinh_normalizer": float(
                    self.local_asinh_normalizer
                ),
                "training_max_abs_residual": float(
                    self.local_training_max_abs_residual
                ),
            },
            "epsilon": float(
                self.epsilon
            ),
            "training_statistics": {
                "raw_abs_percentiles": dict(
                    self.training_raw_abs_percentiles
                    or {}
                ),
                "broad_abs_percentiles": dict(
                    self.training_broad_abs_percentiles
                    or {}
                ),
                "local_abs_percentiles": dict(
                    self.training_local_abs_percentiles
                    or {}
                ),
                "raw_rms_median": float(
                    self.training_raw_rms_median
                ),
                "broad_rms_median": float(
                    self.training_broad_rms_median
                ),
                "local_rms_median": float(
                    self.training_local_rms_median
                ),
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "BroadLocalResidualDecomposer":
        if not isinstance(
            state,
            dict,
        ):
            raise TypeError(
                "broad_local_residual_state必须是字典。"
            )

        if int(
            state.get(
                "schema_version",
                0,
            )
        ) != _SCHEMA_VERSION:
            raise ValueError(
                "不支持的broad_local_residual_state版本。"
            )

        if not bool(
            state.get(
                "enabled",
                False,
            )
        ):
            raise ValueError(
                "broad_local_residual_state没有启用。"
            )

        if state.get(
            "domain"
        ) != _DOMAIN:
            raise ValueError(
                "broad_local_residual_state数据域无效。"
            )

        broad_filter = state.get(
            "broad_filter"
        )

        broad_prior = state.get(
            "broad_prior"
        )

        local_normalization = state.get(
            "local_normalization"
        )

        if not isinstance(
            broad_filter,
            dict,
        ):
            raise ValueError(
                "state缺少broad_filter。"
            )

        if not isinstance(
            broad_prior,
            dict,
        ):
            raise ValueError(
                "state缺少broad_prior。"
            )

        if not isinstance(
            local_normalization,
            dict,
        ):
            raise ValueError(
                "state缺少local_normalization。"
            )

        instance = cls(
            broad_sigma_cm1=float(
                broad_filter["sigma_cm1"]
            ),
            broad_truncate=float(
                broad_filter["truncate"]
            ),
            broad_pca_explained_variance_ratio=float(
                broad_prior[
                    "pca_explained_variance_ratio_target"
                ]
            ),
            broad_pca_max_components=(
                broad_prior.get(
                    "pca_max_components"
                )
            ),
            broad_score_clip_standard_deviations=float(
                broad_prior[
                    "score_clip_standard_deviations"
                ]
            ),
            local_residual_quantile=float(
                local_normalization[
                    "residual_quantile"
                ]
            ),
            local_target_abs_max=float(
                local_normalization[
                    "target_abs_max"
                ]
            ),
            epsilon=float(
                state.get(
                    "epsilon",
                    1.0e-8,
                )
            ),
        )

        axis = np.asarray(
            state["raman_shift"],
            dtype=np.float64,
        ).reshape(-1)

        _as_axis(
            axis,
            expected_length=axis.size,
        )

        instance.raman_shift = axis

        instance.uniform_spacing_cm1 = _positive_float(
            broad_filter[
                "uniform_spacing_cm1"
            ],
            "uniform_spacing_cm1",
        )

        instance.broad_sigma_points = _positive_float(
            broad_filter[
                "sigma_points"
            ],
            "sigma_points",
        )

        instance.number_of_training_spectra = int(
            state[
                "number_of_training_spectra"
            ]
        )

        instance.broad_pca_mean = np.asarray(
            broad_prior["mean"],
            dtype=np.float64,
        ).reshape(-1)

        instance.broad_pca_components = np.asarray(
            broad_prior["components"],
            dtype=np.float64,
        )

        instance.broad_training_score_mean = np.asarray(
            broad_prior[
                "training_score_mean"
            ],
            dtype=np.float64,
        ).reshape(-1)

        instance.broad_score_standard_deviation = np.asarray(
            broad_prior[
                "score_standard_deviation"
            ],
            dtype=np.float64,
        ).reshape(-1)

        instance.broad_explained_variance = np.asarray(
            broad_prior[
                "explained_variance"
            ],
            dtype=np.float64,
        ).reshape(-1)

        instance.broad_explained_variance_ratio_ = np.asarray(
            broad_prior[
                "explained_variance_ratio"
            ],
            dtype=np.float64,
        ).reshape(-1)

        instance.local_residual_scale = _positive_float(
            local_normalization[
                "scale"
            ],
            "local residual scale",
        )

        instance.local_asinh_normalizer = _positive_float(
            local_normalization[
                "asinh_normalizer"
            ],
            "local asinh normalizer",
        )

        instance.local_training_max_abs_residual = (
            _positive_float(
                local_normalization[
                    "training_max_abs_residual"
                ],
                "local training max abs residual",
            )
        )

        statistics = state.get(
            "training_statistics",
            {},
        )

        if not isinstance(
            statistics,
            dict,
        ):
            raise ValueError(
                "training_statistics必须是字典。"
            )

        instance.training_raw_abs_percentiles = dict(
            statistics.get(
                "raw_abs_percentiles",
                {},
            )
        )

        instance.training_broad_abs_percentiles = dict(
            statistics.get(
                "broad_abs_percentiles",
                {},
            )
        )

        instance.training_local_abs_percentiles = dict(
            statistics.get(
                "local_abs_percentiles",
                {},
            )
        )

        instance.training_raw_rms_median = float(
            statistics.get(
                "raw_rms_median",
                np.nan,
            )
        )

        instance.training_broad_rms_median = float(
            statistics.get(
                "broad_rms_median",
                np.nan,
            )
        )

        instance.training_local_rms_median = float(
            statistics.get(
                "local_rms_median",
                np.nan,
            )
        )

        instance._check_fitted()

        return instance

    def summary(
        self,
    ) -> dict[str, Any]:
        self._check_fitted()

        return {
            "enabled": True,
            "broad_sigma_cm1": float(
                self.broad_sigma_cm1
            ),
            "broad_pca_components": int(
                self.broad_pca_components.shape[0]
            ),
            "broad_pca_cumulative_explained_variance": float(
                np.sum(
                    self.broad_explained_variance_ratio_
                )
            ),
            "broad_score_clip_standard_deviations": float(
                self.broad_score_clip_standard_deviations
            ),
            "local_residual_quantile": float(
                self.local_residual_quantile
            ),
            "local_residual_scale": float(
                self.local_residual_scale
            ),
            "local_asinh_normalizer": float(
                self.local_asinh_normalizer
            ),
            "training_raw_rms_median": float(
                self.training_raw_rms_median
            ),
            "training_broad_rms_median": float(
                self.training_broad_rms_median
            ),
            "training_local_rms_median": float(
                self.training_local_rms_median
            ),
        }

    def _split_with_axis(
        self,
        values: np.ndarray,
        axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        uniform_axis = np.linspace(
            float(axis[0]),
            float(axis[-1]),
            axis.size,
            dtype=np.float64,
        )

        spacing = float(
            (
                uniform_axis[-1]
                - uniform_axis[0]
            )
            / (
                uniform_axis.size
                - 1
            )
        )

        if (
            not np.isfinite(spacing)
            or spacing <= 0.0
        ):
            raise RuntimeError(
                "D2.6辅助等间距Raman轴间隔无效。"
            )

        sigma_points = float(
            self.broad_sigma_cm1
            / spacing
        )

        if (
            not np.isfinite(sigma_points)
            or sigma_points <= 0.0
        ):
            raise RuntimeError(
                "D2.6 broad sigma_points无效。"
            )

        broad_on_uniform = np.empty_like(
            values,
            dtype=np.float64,
        )

        if np.allclose(
            axis,
            uniform_axis,
            rtol=0.0,
            atol=max(
                1.0e-10,
                1.0e-8 * spacing,
            ),
        ):
            broad_on_uniform = gaussian_filter1d(
                values,
                sigma=sigma_points,
                axis=1,
                mode="reflect",
                truncate=self.broad_truncate,
            )

            broad = broad_on_uniform

        else:
            for row_index in range(
                values.shape[0]
            ):
                on_uniform = np.interp(
                    uniform_axis,
                    axis,
                    values[row_index],
                )

                smoothed = gaussian_filter1d(
                    on_uniform,
                    sigma=sigma_points,
                    mode="reflect",
                    truncate=self.broad_truncate,
                )

                broad_on_uniform[
                    row_index
                ] = smoothed

            broad = np.stack(
                [
                    np.interp(
                        axis,
                        uniform_axis,
                        broad_on_uniform[
                            row_index
                        ],
                    )
                    for row_index in range(
                        values.shape[0]
                    )
                ],
                axis=0,
            )

        local = (
            values
            - broad
        )

        if (
            not np.isfinite(broad).all()
            or not np.isfinite(local).all()
        ):
            raise RuntimeError(
                "D2.6 broad/local分解包含NaN或无穷值。"
            )

        self.uniform_spacing_cm1 = spacing
        self.broad_sigma_points = sigma_points

        return (
            broad.astype(
                np.float64,
                copy=False,
            ),
            local.astype(
                np.float64,
                copy=False,
            ),
        )

    def _fit_broad_pca(
        self,
        broad: np.ndarray,
    ) -> None:
        mean = np.mean(
            broad,
            axis=0,
            dtype=np.float64,
        )

        centered = (
            broad
            - mean[
                np.newaxis,
                :
            ]
        )

        (
            _,
            singular_values,
            right_vectors,
        ) = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        maximum_rank = min(
            broad.shape[0] - 1,
            right_vectors.shape[0],
        )

        if maximum_rank <= 0:
            raise ValueError(
                "D2.6 broad PCA没有有效秩。"
            )

        explained_variance_all = (
            np.square(
                singular_values[
                    :maximum_rank
                ]
            )
            / max(
                broad.shape[0] - 1,
                1,
            )
        )

        total_variance = float(
            np.sum(
                explained_variance_all
            )
        )

        if (
            not np.isfinite(total_variance)
            or total_variance <= self.epsilon
        ):
            raise ValueError(
                "D2.6 broad residual总体方差几乎为零。"
            )

        explained_ratio_all = (
            explained_variance_all
            / total_variance
        )

        cumulative = np.cumsum(
            explained_ratio_all
        )

        component_count = int(
            np.searchsorted(
                cumulative,
                self.broad_pca_explained_variance_ratio,
                side="left",
            )
            + 1
        )

        component_count = min(
            component_count,
            maximum_rank,
        )

        if self.broad_pca_max_components is not None:
            component_count = min(
                component_count,
                int(
                    self.broad_pca_max_components
                ),
            )

        component_count = max(
            component_count,
            1,
        )

        components = (
            right_vectors[
                :component_count
            ]
        )

        scores = (
            centered
            @ components.T
        )

        score_mean = np.mean(
            scores,
            axis=0,
        )

        score_std = np.std(
            scores,
            axis=0,
            ddof=1,
        )

        if (
            not np.isfinite(score_std).all()
            or np.any(
                score_std
                <= self.epsilon
            )
        ):
            raise ValueError(
                "D2.6 broad PCA score标准差无效。"
            )

        self.broad_pca_mean = (
            mean.astype(
                np.float64,
                copy=False,
            )
        )

        self.broad_pca_components = (
            components.astype(
                np.float64,
                copy=False,
            )
        )

        self.broad_training_score_mean = (
            score_mean.astype(
                np.float64,
                copy=False,
            )
        )

        self.broad_score_standard_deviation = (
            score_std.astype(
                np.float64,
                copy=False,
            )
        )

        self.broad_explained_variance = (
            explained_variance_all[
                :component_count
            ].astype(
                np.float64,
                copy=False,
            )
        )

        self.broad_explained_variance_ratio_ = (
            explained_ratio_all[
                :component_count
            ].astype(
                np.float64,
                copy=False,
            )
        )

    def _fit_local_robust_asinh(
        self,
        local: np.ndarray,
    ) -> None:
        absolute = np.abs(
            local
        )

        maximum = float(
            np.max(
                absolute
            )
        )

        scale = float(
            np.percentile(
                absolute,
                self.local_residual_quantile,
            )
        )

        if (
            not np.isfinite(maximum)
            or maximum <= self.epsilon
        ):
            raise ValueError(
                "D2.6 local residual最大值无效。"
            )

        if (
            not np.isfinite(scale)
            or scale <= self.epsilon
        ):
            raise ValueError(
                "D2.6 local residual稳健尺度无效。"
            )

        normalizer = float(
            np.arcsinh(
                maximum
                / scale
            )
        )

        if (
            not np.isfinite(normalizer)
            or normalizer <= self.epsilon
        ):
            raise ValueError(
                "D2.6 local asinh normalizer无效。"
            )

        self.local_residual_scale = scale
        self.local_asinh_normalizer = normalizer
        self.local_training_max_abs_residual = maximum

    def _check_fitted(
        self,
    ) -> None:
        required = {
            "raman_shift": self.raman_shift,
            "number_of_training_spectra": (
                self.number_of_training_spectra
            ),
            "broad_pca_mean": self.broad_pca_mean,
            "broad_pca_components": (
                self.broad_pca_components
            ),
            "broad_training_score_mean": (
                self.broad_training_score_mean
            ),
            "broad_score_standard_deviation": (
                self.broad_score_standard_deviation
            ),
            "broad_explained_variance": (
                self.broad_explained_variance
            ),
            "broad_explained_variance_ratio_": (
                self.broad_explained_variance_ratio_
            ),
            "local_residual_scale": (
                self.local_residual_scale
            ),
            "local_asinh_normalizer": (
                self.local_asinh_normalizer
            ),
            "local_training_max_abs_residual": (
                self.local_training_max_abs_residual
            ),
        }

        missing = [
            key
            for key, value in required.items()
            if value is None
        ]

        if missing:
            raise RuntimeError(
                "D2.6 broad-local residual尚未拟合："
                f"{missing}。"
            )

        if (
            self.broad_pca_components.ndim != 2
            or self.broad_pca_components.shape[1]
            != self.raman_shift.size
        ):
            raise RuntimeError(
                "D2.6 broad PCA components形状无效。"
            )

        component_count = (
            self.broad_pca_components.shape[0]
        )

        for vector_name, vector in (
            (
                "broad_training_score_mean",
                self.broad_training_score_mean,
            ),
            (
                "broad_score_standard_deviation",
                self.broad_score_standard_deviation,
            ),
            (
                "broad_explained_variance",
                self.broad_explained_variance,
            ),
            (
                "broad_explained_variance_ratio_",
                self.broad_explained_variance_ratio_,
            ),
        ):
            array = np.asarray(
                vector,
                dtype=np.float64,
            ).reshape(-1)

            if array.size != component_count:
                raise RuntimeError(
                    f"{vector_name}长度与PCA组件数量不一致。"
                )

            if not np.isfinite(
                array
            ).all():
                raise RuntimeError(
                    f"{vector_name}包含NaN或无穷值。"
                )

"""训练集先验与SERS光谱残差变换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr, ndtri


# ---------------------------------------------------------------------------
# 残差归一化方法
# ---------------------------------------------------------------------------

GLOBAL_MAXABS = "global_maxabs"
ROBUST_ASINH = "robust_asinh"
POINTWISE_MAD_ASINH = "pointwise_mad_asinh"


# ---------------------------------------------------------------------------
# 先验方法
# ---------------------------------------------------------------------------

TRAINING_POINTWISE_MEDIAN = "training_pointwise_median"

PCA_RECONSTRUCTION = "pca_reconstruction"

TRAINING_LOW_FREQUENCY_MEDIAN = (
    "training_low_frequency_median"
)

TRAINING_BLENDED_FREQUENCY_MEDIAN = (
    "training_blended_frequency_median"
)


SUPPORTED_PRIOR_METHODS = {
    TRAINING_POINTWISE_MEDIAN,
    PCA_RECONSTRUCTION,
    TRAINING_LOW_FREQUENCY_MEDIAN,
    TRAINING_BLENDED_FREQUENCY_MEDIAN,
}


# 除 PCA 外，其余方法生成时都使用 checkpoint 中保存的一条固定 prior。
FIXED_PRIOR_METHODS = {
    TRAINING_POINTWISE_MEDIAN,
    TRAINING_LOW_FREQUENCY_MEDIAN,
    TRAINING_BLENDED_FREQUENCY_MEDIAN,
}


SUPPORTED_NORMALIZATION_METHODS = {
    GLOBAL_MAXABS,
    ROBUST_ASINH,
    POINTWISE_MAD_ASINH,
}


# 历史D2.2策略：先从多元高斯采样，再逐维clip。
# 为保持旧checkpoint生成行为，不修改其既有语义。
LEGACY_CLIPPED_GAUSSIAN_SCORES = (
    "truncated_gaussian_scores"
)

# D2.5策略：利用PCA score互不相关的性质，逐主成分从真正的
# 截断高斯分布采样，避免clip造成概率质量堆积在边界。
INDEPENDENT_TRUNCATED_GAUSSIAN_SCORES = (
    "independent_truncated_gaussian_scores"
)

SUPPORTED_PCA_SAMPLING_STRATEGIES = {
    LEGACY_CLIPPED_GAUSSIAN_SCORES,
    INDEPENDENT_TRUNCATED_GAUSSIAN_SCORES,
}


_PERCENTILE_LEVELS = np.asarray(
    [
        50.0,
        90.0,
        95.0,
        99.0,
        99.5,
        99.9,
    ],
    dtype=np.float64,
)


@dataclass
class PriorResidualTransformer:
    """
    使用训练集拟合的先验，将完整归一化 SERS 光谱变换到残差域。

    输入和输出均为二维数组 [N, L]。

    输入光谱必须已经：

    1. 插值到统一训练拉曼轴；
    2. 使用仅由训练集拟合的 global_minmax 完成强度归一化；
    3. 尚未进行 U-Net 末尾长度补齐。

    支持的先验：

    training_pointwise_median
        D2/D2.1 固定训练集中位数光谱。

    pca_reconstruction
        D2.2 PCA 可变先验。

    training_low_frequency_median
        D2.3。对训练集中位数光谱做 Gaussian 低通，
        仅保留低频结构。

    training_blended_frequency_median
        D2.4。在低频中位数 prior 的基础上，
        加回一定比例的训练集共同高频峰骨架：

        P_blend =
            P_low
            + alpha * (P_median - P_low)

        alpha=0：
            等价于纯 low-frequency prior。

        alpha=1：
            等价于完整 median prior。

        当前实验默认 alpha=0.5。

    注意：
    D2.4 不检测、不写死任何农药峰位。
    所有频率结构均由当前训练集自动拟合。
    """

    # ------------------------------------------------------------------
    # 基础配置
    # ------------------------------------------------------------------

    prior_method: str = TRAINING_POINTWISE_MEDIAN
    normalization_method: str = GLOBAL_MAXABS

    target_abs_max: float = 1.0
    residual_quantile: float = 99.5

    pointwise_scale_floor_quantile: float = 10.0
    mad_scale_factor: float = 1.4826

    epsilon: float = 1.0e-8

    # ------------------------------------------------------------------
    # D2.2 PCA prior
    # ------------------------------------------------------------------

    pca_explained_variance_ratio: float = 0.95
    pca_max_components: int | None = None

    pca_sampling_strategy: str = (
        "truncated_gaussian_scores"
    )

    pca_score_clip_standard_deviations: float = 2.5

    # ------------------------------------------------------------------
    # D2.3 / D2.4 低频 prior 参数
    # ------------------------------------------------------------------

    low_frequency_method: str = "gaussian"

    # 单位是 cm^-1，而不是数组点数。
    low_frequency_sigma_cm1: float = 40.0

    low_frequency_truncate: float = 4.0

    # ------------------------------------------------------------------
    # D2.4 blended prior
    # ------------------------------------------------------------------

    # 0 < alpha < 1。
    # 当前第一轮实验固定使用 0.5。
    blended_peak_component_ratio: float = 0.5

    # ------------------------------------------------------------------
    # 公共拟合状态
    # ------------------------------------------------------------------

    prior: np.ndarray | None = None

    residual_scale: float | None = None

    training_max_abs_residual: float | None = None

    training_abs_residual_quantile: float | None = None

    asinh_normalizer: float | None = None

    training_abs_residual_percentiles: (
        dict[str, float] | None
    ) = None

    # ------------------------------------------------------------------
    # D2.1 pointwise MAD 状态
    # ------------------------------------------------------------------

    pointwise_scale: np.ndarray | None = None

    pointwise_scale_floor: float | None = None

    training_max_abs_standardized_residual: (
        float | None
    ) = None

    training_abs_standardized_residual_quantile: (
        float | None
    ) = None

    training_pointwise_scale_percentiles: (
        dict[str, float] | None
    ) = None

    # ------------------------------------------------------------------
    # D2.2 PCA 状态
    # ------------------------------------------------------------------

    pca_mean: np.ndarray | None = None

    pca_components: np.ndarray | None = None

    pca_training_score_mean: np.ndarray | None = None

    pca_training_score_covariance: np.ndarray | None = None

    pca_score_standard_deviation: np.ndarray | None = None

    pca_explained_variance: np.ndarray | None = None

    pca_explained_variance_ratio_: np.ndarray | None = None

    pca_number_of_training_spectra: int | None = None

    # ------------------------------------------------------------------
    # D2.3 / D2.4 低频 prior 元数据
    # ------------------------------------------------------------------

    low_frequency_axis_start_cm1: float | None = None

    low_frequency_axis_end_cm1: float | None = None

    low_frequency_axis_length: int | None = None

    low_frequency_uniform_spacing_cm1: float | None = None

    low_frequency_sigma_points: float | None = None

    low_frequency_number_of_training_spectra: (
        int | None
    ) = None

    # ------------------------------------------------------------------
    # 初始化校验
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.prior_method = str(
            self.prior_method
        ).strip().lower()

        self.normalization_method = str(
            self.normalization_method
        ).strip().lower()

        self.target_abs_max = float(
            self.target_abs_max
        )

        self.residual_quantile = float(
            self.residual_quantile
        )

        self.pointwise_scale_floor_quantile = float(
            self.pointwise_scale_floor_quantile
        )

        self.mad_scale_factor = float(
            self.mad_scale_factor
        )

        self.epsilon = float(
            self.epsilon
        )

        self.pca_explained_variance_ratio = float(
            self.pca_explained_variance_ratio
        )

        self.pca_max_components = (
            None
            if self.pca_max_components is None
            else int(self.pca_max_components)
        )

        self.pca_sampling_strategy = str(
            self.pca_sampling_strategy
        ).strip().lower()

        self.pca_score_clip_standard_deviations = float(
            self.pca_score_clip_standard_deviations
        )

        self.low_frequency_method = str(
            self.low_frequency_method
        ).strip().lower()

        self.low_frequency_sigma_cm1 = float(
            self.low_frequency_sigma_cm1
        )

        self.low_frequency_truncate = float(
            self.low_frequency_truncate
        )

        self.blended_peak_component_ratio = float(
            self.blended_peak_component_ratio
        )

        if self.prior_method not in SUPPORTED_PRIOR_METHODS:
            supported = ", ".join(
                sorted(SUPPORTED_PRIOR_METHODS)
            )

            raise ValueError(
                "不支持的prior_method："
                f"{self.prior_method}。"
                f"可用方法：{supported}。"
            )

        if (
            self.normalization_method
            not in SUPPORTED_NORMALIZATION_METHODS
        ):
            supported = ", ".join(
                sorted(
                    SUPPORTED_NORMALIZATION_METHODS
                )
            )

            raise ValueError(
                "不支持的残差缩放方法："
                f"{self.normalization_method}。"
                f"可用方法：{supported}。"
            )

        if not 0.0 < self.target_abs_max <= 1.0:
            raise ValueError(
                "target_abs_max必须大于0且不大于1。"
            )

        if not 0.0 < self.residual_quantile < 100.0:
            raise ValueError(
                "residual_quantile必须大于0且小于100。"
            )

        if not (
            0.0
            <= self.pointwise_scale_floor_quantile
            < 100.0
        ):
            raise ValueError(
                "pointwise_scale_floor_quantile"
                "必须大于等于0且小于100。"
            )

        if self.mad_scale_factor <= 0.0:
            raise ValueError(
                "mad_scale_factor必须大于0。"
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon必须大于0。"
            )

        if not (
            0.0
            < self.pca_explained_variance_ratio
            <= 1.0
        ):
            raise ValueError(
                "pca_explained_variance_ratio"
                "必须在(0,1]范围内。"
            )

        if (
            self.pca_max_components is not None
            and self.pca_max_components <= 0
        ):
            raise ValueError(
                "pca_max_components必须为正整数或null。"
            )

        if (
            self.pca_sampling_strategy
            not in SUPPORTED_PCA_SAMPLING_STRATEGIES
        ):
            raise ValueError(
                "pca_sampling_strategy必须为"
                "truncated_gaussian_scores或"
                "independent_truncated_gaussian_scores。"
            )

        if (
            self.pca_score_clip_standard_deviations
            <= 0.0
        ):
            raise ValueError(
                "pca_score_clip_standard_deviations"
                "必须大于0。"
            )

        if self.low_frequency_method != "gaussian":
            raise ValueError(
                "low_frequency_method当前只支持gaussian。"
            )

        if (
            not np.isfinite(
                self.low_frequency_sigma_cm1
            )
            or self.low_frequency_sigma_cm1 <= 0.0
        ):
            raise ValueError(
                "low_frequency_sigma_cm1"
                "必须是有限的正数。"
            )

        if (
            not np.isfinite(
                self.low_frequency_truncate
            )
            or self.low_frequency_truncate <= 0.0
        ):
            raise ValueError(
                "low_frequency_truncate"
                "必须是有限的正数。"
            )

        if not (
            0.0
            < self.blended_peak_component_ratio
            < 1.0
        ):
            raise ValueError(
                "blended_peak_component_ratio"
                "必须在(0,1)范围内。"
            )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        training_spectra: np.ndarray,
        *,
        raman_shift: np.ndarray | None = None,
    ) -> "PriorResidualTransformer":
        """
        仅使用训练集拟合先验和残差变换参数。

        raman_shift：
            training_low_frequency_median 和
            training_blended_frequency_median 必须提供。

        Gaussian sigma 始终以 cm^-1 定义，
        不按数组下标直接平滑。
        """

        values = self._validate_array(
            training_spectra,
            "training_spectra",
        )

        if values.shape[0] < 2:
            raise ValueError(
                "先验残差模型至少需要2条训练光谱。"
            )

        self._clear_pca_state()
        self._clear_low_frequency_state()

        # --------------------------------------------------------------
        # D2 / D2.1：完整训练集中位数 prior
        # --------------------------------------------------------------

        if (
            self.prior_method
            == TRAINING_POINTWISE_MEDIAN
        ):
            prior = np.median(
                values,
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )

            reference_priors = np.repeat(
                prior[np.newaxis, :],
                values.shape[0],
                axis=0,
            )

        # --------------------------------------------------------------
        # D2.3：纯低频训练中位数 prior
        # --------------------------------------------------------------

        elif (
            self.prior_method
            == TRAINING_LOW_FREQUENCY_MEDIAN
        ):
            prior = (
                self._fit_low_frequency_median_prior(
                    values,
                    raman_shift=raman_shift,
                )
            )

            reference_priors = np.repeat(
                prior[np.newaxis, :],
                values.shape[0],
                axis=0,
            )

        # --------------------------------------------------------------
        # D2.4：低频 + 部分共同峰骨架
        # --------------------------------------------------------------

        elif (
            self.prior_method
            == TRAINING_BLENDED_FREQUENCY_MEDIAN
        ):
            full_median = np.median(
                values,
                axis=0,
            ).astype(
                np.float64,
                copy=False,
            )

            low_frequency_prior = (
                self._fit_low_frequency_median_prior(
                    values,
                    raman_shift=raman_shift,
                )
            ).astype(
                np.float64,
                copy=False,
            )

            # 训练集中位数中相对于低频结构的共同峰/高频部分。
            peak_component = (
                full_median
                - low_frequency_prior
            )

            # D2.4核心公式。
            blended_prior = (
                low_frequency_prior
                + self.blended_peak_component_ratio
                * peak_component
            )

            prior = blended_prior.astype(
                np.float32,
                copy=False,
            )

            reference_priors = np.repeat(
                prior[np.newaxis, :],
                values.shape[0],
                axis=0,
            )

        # --------------------------------------------------------------
        # D2.2：PCA可变 prior
        # --------------------------------------------------------------

        else:
            reference_priors = (
                self._fit_pca_prior(
                    values
                )
            )

            # checkpoint 默认 prior 仍保存 PCA mean；
            # 每条训练光谱实际使用的是自身 PCA reconstruction。
            prior = self.pca_mean.astype(
                np.float32,
                copy=False,
            )

        # --------------------------------------------------------------
        # 计算残差
        # --------------------------------------------------------------

        residuals = (
            values.astype(
                np.float64,
                copy=False,
            )
            - reference_priors.astype(
                np.float64,
                copy=False,
            )
        )

        absolute_residuals = np.abs(
            residuals
        )

        maximum = float(
            np.max(
                absolute_residuals
            )
        )

        if maximum <= self.epsilon:
            raise ValueError(
                "训练光谱相对于当前先验的残差几乎为零。"
                "请确认训练光谱没有被重复复制，"
                "以及prior配置是否正确。"
            )

        self.prior = prior.copy()

        self.training_max_abs_residual = (
            maximum
        )

        self.training_abs_residual_percentiles = (
            self._summarize_nonnegative(
                absolute_residuals,
                include_max=True,
            )
        )

        # 每次重新 fit 都先清除旧 normalization 状态。
        self.residual_scale = None

        self.training_abs_residual_quantile = (
            None
        )

        self.asinh_normalizer = None

        self.pointwise_scale = None

        self.pointwise_scale_floor = None

        self.training_max_abs_standardized_residual = (
            None
        )

        self.training_abs_standardized_residual_quantile = (
            None
        )

        self.training_pointwise_scale_percentiles = (
            None
        )

        # --------------------------------------------------------------
        # global_maxabs
        # --------------------------------------------------------------

        if (
            self.normalization_method
            == GLOBAL_MAXABS
        ):
            self.residual_scale = (
                maximum
                / self.target_abs_max
            )

            return self

        # --------------------------------------------------------------
        # robust_asinh
        # --------------------------------------------------------------

        if (
            self.normalization_method
            == ROBUST_ASINH
        ):
            quantile_value = float(
                np.percentile(
                    absolute_residuals,
                    self.residual_quantile,
                )
            )

            self._validate_positive_finite(
                quantile_value,
                "训练残差的稳健分位数尺度",
            )

            self.residual_scale = (
                quantile_value
            )

            self.training_abs_residual_quantile = (
                quantile_value
            )

            self.asinh_normalizer = (
                self._build_asinh_normalizer(
                    maximum=maximum,
                    scale=quantile_value,
                    description="训练残差",
                )
            )

            return self

        # --------------------------------------------------------------
        # pointwise_mad_asinh
        # --------------------------------------------------------------

        self._fit_pointwise_mad_asinh(
            residuals
        )

        return self

    # ------------------------------------------------------------------
    # D2.3 / D2.4 low-frequency prior
    # ------------------------------------------------------------------

    def _fit_low_frequency_median_prior(
        self,
        values: np.ndarray,
        *,
        raman_shift: np.ndarray | None,
    ) -> np.ndarray:
        """
        在物理 Raman shift 轴上建立低频训练中位数 prior。

        处理步骤：

        1. 计算训练集逐点中位数；
        2. 将实际训练轴映射到同范围等间距辅助轴；
        3. 在等间距轴上按 cm^-1 → point 换算 sigma；
        4. Gaussian low-pass；
        5. 再插值回实际模型 Raman 轴。

        不写死任何峰位。
        """

        axis = self._validate_raman_shift_axis(
            raman_shift,
            expected_length=values.shape[1],
        )

        full_median = np.median(
            values,
            axis=0,
        ).astype(
            np.float64,
            copy=False,
        )

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

        self._validate_positive_finite(
            spacing,
            "低频prior辅助等间距轴间隔",
        )

        sigma_points = float(
            self.low_frequency_sigma_cm1
            / spacing
        )

        self._validate_positive_finite(
            sigma_points,
            "低频prior Gaussian sigma_points",
        )

        median_on_uniform_axis = np.interp(
            uniform_axis,
            axis,
            full_median,
        )

        smoothed_on_uniform_axis = (
            gaussian_filter1d(
                median_on_uniform_axis,
                sigma=sigma_points,
                mode="reflect",
                truncate=self.low_frequency_truncate,
            )
        )

        low_frequency_prior = np.interp(
            axis,
            uniform_axis,
            smoothed_on_uniform_axis,
        )

        if not np.isfinite(
            low_frequency_prior
        ).all():
            raise RuntimeError(
                "低频prior包含NaN或无穷值。"
            )

        self.low_frequency_axis_start_cm1 = (
            float(axis[0])
        )

        self.low_frequency_axis_end_cm1 = (
            float(axis[-1])
        )

        self.low_frequency_axis_length = int(
            axis.size
        )

        self.low_frequency_uniform_spacing_cm1 = (
            spacing
        )

        self.low_frequency_sigma_points = (
            sigma_points
        )

        self.low_frequency_number_of_training_spectra = (
            int(values.shape[0])
        )

        return low_frequency_prior.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # D2.1 pointwise MAD
    # ------------------------------------------------------------------

    def _fit_pointwise_mad_asinh(
        self,
        residuals: np.ndarray,
    ) -> None:
        """
        拟合逐波数 MAD 尺度。

        注意：
        该方法保留用于历史 D2.1/D3 模型兼容。

        D2.3/D2.4 当前实验不应使用该方法，
        configuration_loader.py 会阻止该错误组合。
        """

        residual_median = np.median(
            residuals,
            axis=0,
        )

        pointwise_mad = np.median(
            np.abs(
                residuals
                - residual_median[
                    np.newaxis,
                    :
                ]
            ),
            axis=0,
        )

        raw_pointwise_scale = (
            self.mad_scale_factor
            * pointwise_mad
        )

        positive_scales = (
            raw_pointwise_scale[
                np.isfinite(
                    raw_pointwise_scale
                )
                & (
                    raw_pointwise_scale
                    > self.epsilon
                )
            ]
        )

        if positive_scales.size == 0:
            raise ValueError(
                "所有波数点的MAD尺度都几乎为零，"
                "无法拟合pointwise_mad_asinh。"
                "请检查训练光谱是否被复制。"
            )

        scale_floor = float(
            np.percentile(
                positive_scales,
                self.pointwise_scale_floor_quantile,
            )
        )

        scale_floor = max(
            scale_floor,
            self.epsilon,
        )

        pointwise_scale = np.maximum(
            raw_pointwise_scale,
            scale_floor,
        )

        self._validate_vector(
            pointwise_scale,
            expected_length=residuals.shape[1],
            name="逐波数MAD尺度",
            strictly_positive=True,
        )

        standardized_residuals = (
            residuals
            / pointwise_scale[
                np.newaxis,
                :
            ]
        )

        absolute_standardized = np.abs(
            standardized_residuals
        )

        maximum_standardized = float(
            np.max(
                absolute_standardized
            )
        )

        quantile_standardized = float(
            np.percentile(
                absolute_standardized,
                self.residual_quantile,
            )
        )

        self._validate_positive_finite(
            maximum_standardized,
            "训练标准化残差最大值",
        )

        self._validate_positive_finite(
            quantile_standardized,
            "训练标准化残差稳健分位数尺度",
        )

        self.pointwise_scale = (
            pointwise_scale.astype(
                np.float32,
                copy=False,
            )
        )

        self.pointwise_scale_floor = (
            scale_floor
        )

        self.training_pointwise_scale_percentiles = (
            self._summarize_nonnegative(
                pointwise_scale,
                include_max=True,
            )
        )

        self.training_max_abs_standardized_residual = (
            maximum_standardized
        )

        self.training_abs_standardized_residual_quantile = (
            quantile_standardized
        )

        # 历史接口：
        # D2.1中的 residual_scale 表示
        # pointwise 标准化后的全局稳健分位数。
        self.residual_scale = (
            quantile_standardized
        )

        self.asinh_normalizer = (
            self._build_asinh_normalizer(
                maximum=maximum_standardized,
                scale=quantile_standardized,
                description="训练标准化残差",
            )
        )

    # ------------------------------------------------------------------
    # transform / inverse_transform
    # ------------------------------------------------------------------

    def transform(
        self,
        spectra: np.ndarray,
        *,
        reference_priors: np.ndarray | None = None,
    ) -> np.ndarray:
        """将完整归一化光谱转换为 DDPM 学习的残差域。"""

        self._check_fitted()

        values = self._validate_array(
            spectra,
            "spectra",
        )

        self._check_length(
            values
        )

        references = (
            self._resolve_reference_priors(
                values,
                reference_priors=reference_priors,
                operation="transform",
            )
        )

        residuals = (
            values.astype(
                np.float64,
                copy=False,
            )
            - references.astype(
                np.float64,
                copy=False,
            )
        )

        if (
            self.normalization_method
            == GLOBAL_MAXABS
        ):
            scaled = (
                residuals
                / float(
                    self.residual_scale
                )
            )

        elif (
            self.normalization_method
            == ROBUST_ASINH
        ):
            scaled = self._asinh_forward(
                residuals
                / float(
                    self.residual_scale
                )
            )

        else:
            standardized = (
                residuals
                / self.pointwise_scale[
                    np.newaxis,
                    :
                ]
            )

            scaled = self._asinh_forward(
                standardized
                / float(
                    self.residual_scale
                )
            )

        if not np.isfinite(
            scaled
        ).all():
            raise RuntimeError(
                "残差变换结果包含NaN或无穷值。"
            )

        return scaled.astype(
            np.float32,
            copy=False,
        )

    def inverse_transform(
        self,
        scaled_residuals: np.ndarray,
        *,
        reference_priors: np.ndarray | None = None,
    ) -> np.ndarray:
        """取消残差变换并加回对应 prior。"""

        self._check_fitted()

        values = self._validate_array(
            scaled_residuals,
            "scaled_residuals",
        )

        self._check_length(
            values
        )

        if (
            self.normalization_method
            == GLOBAL_MAXABS
        ):
            residuals = (
                values.astype(
                    np.float64,
                    copy=False,
                )
                * float(
                    self.residual_scale
                )
            )

        elif (
            self.normalization_method
            == ROBUST_ASINH
        ):
            residuals = (
                float(
                    self.residual_scale
                )
                * self._asinh_inverse(
                    values
                )
            )

        else:
            standardized = (
                float(
                    self.residual_scale
                )
                * self._asinh_inverse(
                    values
                )
            )

            residuals = (
                self.pointwise_scale[
                    np.newaxis,
                    :
                ]
                * standardized
            )

        references = (
            self._resolve_reference_priors(
                values,
                reference_priors=reference_priors,
                operation="inverse_transform",
            )
        )

        restored = (
            references
            + residuals
        )

        if not np.isfinite(
            restored
        ).all():
            raise RuntimeError(
                "残差逆变换结果包含NaN或无穷值。"
                "请检查生成残差是否远超训练范围。"
            )

        return restored.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # asinh
    # ------------------------------------------------------------------

    def _asinh_forward(
        self,
        values_divided_by_scale: np.ndarray,
    ) -> np.ndarray:
        return (
            self.target_abs_max
            * np.arcsinh(
                values_divided_by_scale
            )
            / float(
                self.asinh_normalizer
            )
        )

    def _asinh_inverse(
        self,
        scaled_values: np.ndarray,
    ) -> np.ndarray:
        sinh_argument = (
            scaled_values.astype(
                np.float64,
                copy=False,
            )
            / self.target_abs_max
            * float(
                self.asinh_normalizer
            )
        )

        return np.sinh(
            sinh_argument
        )

    # ------------------------------------------------------------------
    # prior API
    # ------------------------------------------------------------------

    def prior_batch(
        self,
        number_of_spectra: int,
    ) -> np.ndarray:
        """返回固定 prior 的多份副本。"""

        self._check_fitted()

        count = int(
            number_of_spectra
        )

        if count <= 0:
            raise ValueError(
                "number_of_spectra必须大于0。"
            )

        return np.repeat(
            self.prior[
                np.newaxis,
                :
            ],
            count,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

    def reference_priors_for_spectra(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        为真实光谱计算训练阶段对应 prior。

        固定 prior：
            直接返回固定 prior。

        PCA：
            根据每条真实光谱 PCA projection 计算对应 prior。
        """

        self._check_fitted()

        values = self._validate_array(
            spectra,
            "spectra",
        )

        self._check_length(
            values
        )

        return self._resolve_reference_priors(
            values,
            reference_priors=None,
            operation="reference_priors_for_spectra",
        )

    def sample_reference_priors(
        self,
        number_of_spectra: int,
        *,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """
        为生成端抽取 prior。

        fixed prior：
            所有生成样本共享 checkpoint 中保存的 prior。

        PCA：
            从训练 PCA score 分布抽样。
        """

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
                "random_generator必须是"
                "numpy.random.Generator。"
            )

        if (
            self.prior_method
            in FIXED_PRIOR_METHODS
        ):
            return self.prior_batch(
                count
            )

        if (
            self.pca_sampling_strategy
            == LEGACY_CLIPPED_GAUSSIAN_SCORES
        ):
            # 历史D2.2行为：保留旧checkpoint生成兼容性。
            scores = (
                random_generator.multivariate_normal(
                    mean=self.pca_training_score_mean,
                    cov=self.pca_training_score_covariance,
                    size=count,
                    check_valid="raise",
                )
            )

            scores = np.asarray(
                scores,
                dtype=np.float64,
            ).reshape(
                count,
                self.pca_components.shape[0],
            )

            limit = (
                self.pca_score_clip_standard_deviations
                * self.pca_score_standard_deviation
            )

            lower = (
                self.pca_training_score_mean
                - limit
            )

            upper = (
                self.pca_training_score_mean
                + limit
            )

            scores = np.clip(
                scores,
                lower[
                    np.newaxis,
                    :
                ],
                upper[
                    np.newaxis,
                    :
                ],
            )

        else:
            # PCA由SVD拟合，各score轴在训练集上互不相关。
            # 对每个标准化score直接进行逆CDF采样，得到严格位于
            # [-clip, +clip]内的独立截断标准高斯，避免np.clip
            # 在上下边界制造离散概率质量。
            clip = float(
                self.pca_score_clip_standard_deviations
            )

            lower_cdf = float(
                ndtr(-clip)
            )

            upper_cdf = float(
                ndtr(clip)
            )

            uniform_probabilities = (
                random_generator.uniform(
                    low=lower_cdf,
                    high=upper_cdf,
                    size=(
                        count,
                        self.pca_components.shape[0],
                    ),
                )
            )

            standardized_scores = ndtri(
                uniform_probabilities
            )

            scores = (
                self.pca_training_score_mean[
                    np.newaxis,
                    :
                ]
                + standardized_scores
                * self.pca_score_standard_deviation[
                    np.newaxis,
                    :
                ]
            )

        priors = (
            self.pca_mean[
                np.newaxis,
                :
            ]
            + scores
            @ self.pca_components
        )

        if not np.isfinite(
            priors
        ).all():
            raise RuntimeError(
                "PCA生成先验包含NaN或无穷值。"
            )

        return priors.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def _clear_pca_state(
        self,
    ) -> None:
        self.pca_mean = None
        self.pca_components = None

        self.pca_training_score_mean = None

        self.pca_training_score_covariance = (
            None
        )

        self.pca_score_standard_deviation = (
            None
        )

        self.pca_explained_variance = None

        self.pca_explained_variance_ratio_ = (
            None
        )

        self.pca_number_of_training_spectra = (
            None
        )

    def _fit_pca_prior(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """在训练集拟合 PCA，并返回逐样本重建 prior。"""

        values64 = values.astype(
            np.float64,
            copy=False,
        )

        mean = values64.mean(
            axis=0
        )

        centered = (
            values64
            - mean[
                np.newaxis,
                :
            ]
        )

        _, singular_values, right_vectors = (
            np.linalg.svd(
                centered,
                full_matrices=False,
            )
        )

        explained_variance_all = (
            np.square(
                singular_values
            )
            / max(
                values64.shape[0] - 1,
                1,
            )
        )

        total_variance = float(
            explained_variance_all.sum()
        )

        if total_variance <= self.epsilon:
            raise ValueError(
                "训练光谱总体方差几乎为零，"
                "无法拟合PCA先验。"
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
                self.pca_explained_variance_ratio,
                side="left",
            )
            + 1
        )

        maximum_available = min(
            values64.shape[0] - 1,
            values64.shape[1],
        )

        if (
            self.pca_max_components
            is not None
        ):
            maximum_available = min(
                maximum_available,
                self.pca_max_components,
            )

        component_count = max(
            1,
            min(
                component_count,
                maximum_available,
            ),
        )

        components = right_vectors[
            :component_count
        ]

        scores = (
            centered
            @ components.T
        )

        score_mean = scores.mean(
            axis=0
        )

        if component_count == 1:
            score_covariance = np.asarray(
                [
                    [
                        float(
                            np.var(
                                scores[:, 0],
                                ddof=1,
                            )
                        )
                    ]
                ],
                dtype=np.float64,
            )
        else:
            score_covariance = np.cov(
                scores,
                rowvar=False,
                ddof=1,
            )

        score_covariance = np.asarray(
            score_covariance,
            dtype=np.float64,
        )

        diagonal = np.diag(
            score_covariance
        ).copy()

        diagonal = np.maximum(
            diagonal,
            self.epsilon**2,
        )

        score_covariance[
            np.diag_indices_from(
                score_covariance
            )
        ] = diagonal

        score_standard_deviation = np.sqrt(
            diagonal
        )

        reconstructed = (
            mean[
                np.newaxis,
                :
            ]
            + scores
            @ components
        )

        self.pca_mean = mean.astype(
            np.float32,
            copy=False,
        )

        self.pca_components = components.astype(
            np.float32,
            copy=False,
        )

        self.pca_training_score_mean = (
            score_mean.astype(
                np.float64,
                copy=False,
            )
        )

        self.pca_training_score_covariance = (
            score_covariance
        )

        self.pca_score_standard_deviation = (
            score_standard_deviation
        )

        self.pca_explained_variance = (
            explained_variance_all[
                :component_count
            ]
        )

        self.pca_explained_variance_ratio_ = (
            explained_ratio_all[
                :component_count
            ]
        )

        self.pca_number_of_training_spectra = (
            int(
                values64.shape[0]
            )
        )

        return reconstructed.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # Reference prior resolution
    # ------------------------------------------------------------------

    def _resolve_reference_priors(
        self,
        values: np.ndarray,
        *,
        reference_priors: np.ndarray | None,
        operation: str,
    ) -> np.ndarray:
        """规范化或自动计算每条光谱的 reference prior。"""

        if reference_priors is not None:
            references = self._validate_array(
                reference_priors,
                "reference_priors",
            )

            if (
                references.shape
                != values.shape
            ):
                raise ValueError(
                    "reference_priors形状必须与光谱一致："
                    f"先验为{references.shape}，"
                    f"光谱为{values.shape}。"
                )

            return references.astype(
                np.float32,
                copy=False,
            )

        if (
            self.prior_method
            in FIXED_PRIOR_METHODS
        ):
            return self.prior_batch(
                values.shape[0]
            )

        if (
            operation
            == "inverse_transform"
        ):
            raise ValueError(
                "PCA可变先验的inverse_transform"
                "必须显式提供reference_priors；"
                "生成端请使用sample_reference_priors()。"
            )

        centered = (
            values.astype(
                np.float64,
                copy=False,
            )
            - self.pca_mean[
                np.newaxis,
                :
            ].astype(
                np.float64,
                copy=False,
            )
        )

        scores = (
            centered
            @ self.pca_components.astype(
                np.float64,
                copy=False,
            ).T
        )

        priors = (
            self.pca_mean[
                np.newaxis,
                :
            ]
            + scores
            @ self.pca_components
        )

        return priors.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # checkpoint state
    # ------------------------------------------------------------------

    def state_dict(
        self,
    ) -> dict[str, Any]:
        """生成可以保存到 checkpoint metadata 中的状态。"""

        self._check_fitted()

        residual_state: dict[str, Any] = {
            "method": self.normalization_method,
            "training_max_abs_residual": float(
                self.training_max_abs_residual
            ),
            "target_abs_max": float(
                self.target_abs_max
            ),
            "epsilon": float(
                self.epsilon
            ),
            "training_abs_residual_percentiles": dict(
                self.training_abs_residual_percentiles
                or {}
            ),
        }

        if (
            self.normalization_method
            == GLOBAL_MAXABS
        ):
            residual_state[
                "scale"
            ] = float(
                self.residual_scale
            )

        elif (
            self.normalization_method
            == ROBUST_ASINH
        ):
            residual_state.update(
                {
                    "scale": float(
                        self.residual_scale
                    ),
                    "residual_quantile": float(
                        self.residual_quantile
                    ),
                    "training_abs_residual_quantile": float(
                        self.training_abs_residual_quantile
                    ),
                    "asinh_normalizer": float(
                        self.asinh_normalizer
                    ),
                }
            )

        else:
            residual_state.update(
                {
                    "residual_quantile": float(
                        self.residual_quantile
                    ),
                    "mad_scale_factor": float(
                        self.mad_scale_factor
                    ),
                    "pointwise_scale_floor_quantile": float(
                        self.pointwise_scale_floor_quantile
                    ),
                    "pointwise_scale_floor": float(
                        self.pointwise_scale_floor
                    ),
                    "pointwise_scale": (
                        self.pointwise_scale.tolist()
                    ),
                    "standardized_residual_scale": float(
                        self.residual_scale
                    ),
                    "training_abs_standardized_residual_quantile": float(
                        self.training_abs_standardized_residual_quantile
                    ),
                    "training_max_abs_standardized_residual": float(
                        self.training_max_abs_standardized_residual
                    ),
                    "asinh_normalizer": float(
                        self.asinh_normalizer
                    ),
                    "training_pointwise_scale_percentiles": dict(
                        self.training_pointwise_scale_percentiles
                        or {}
                    ),
                }
            )

        # 保持旧版本兼容：
        #
        # v4：median / PCA
        # v5：low-frequency median
        # v6：blended-frequency median
        if (
            self.prior_method
            == TRAINING_BLENDED_FREQUENCY_MEDIAN
        ):
            schema_version = 6

        elif (
            self.prior_method
            == TRAINING_LOW_FREQUENCY_MEDIAN
        ):
            schema_version = 5

        else:
            schema_version = 4

        state: dict[str, Any] = {
            "schema_version": schema_version,
            "enabled": True,
            "domain": (
                "spectrum_global_minmax_normalized"
            ),
            "prior_method": self.prior_method,
            "prior_normalized_intensity": (
                self.prior.tolist()
            ),
            "residual_normalization": (
                residual_state
            ),
        }

        # --------------------------------------------------------------
        # PCA state
        # --------------------------------------------------------------

        if (
            self.prior_method
            == PCA_RECONSTRUCTION
        ):
            state["pca_prior"] = {
                "explained_variance_ratio_target": float(
                    self.pca_explained_variance_ratio
                ),
                "max_components": (
                    self.pca_max_components
                ),
                "sampling_strategy": (
                    self.pca_sampling_strategy
                ),
                "score_clip_standard_deviations": float(
                    self.pca_score_clip_standard_deviations
                ),
                "number_of_training_spectra": int(
                    self.pca_number_of_training_spectra
                ),
                "mean": (
                    self.pca_mean.tolist()
                ),
                "components": (
                    self.pca_components.tolist()
                ),
                "training_score_mean": (
                    self.pca_training_score_mean.tolist()
                ),
                "training_score_covariance": (
                    self.pca_training_score_covariance.tolist()
                ),
                "score_standard_deviation": (
                    self.pca_score_standard_deviation.tolist()
                ),
                "explained_variance": (
                    self.pca_explained_variance.tolist()
                ),
                "explained_variance_ratio": (
                    self.pca_explained_variance_ratio_.tolist()
                ),
            }

        # --------------------------------------------------------------
        # Low-frequency metadata
        # --------------------------------------------------------------

        if self.prior_method in {
            TRAINING_LOW_FREQUENCY_MEDIAN,
            TRAINING_BLENDED_FREQUENCY_MEDIAN,
        }:
            state["low_frequency_prior"] = {
                "method": (
                    self.low_frequency_method
                ),
                "sigma_cm1": float(
                    self.low_frequency_sigma_cm1
                ),
                "truncate": float(
                    self.low_frequency_truncate
                ),
                "axis_start_cm1": float(
                    self.low_frequency_axis_start_cm1
                ),
                "axis_end_cm1": float(
                    self.low_frequency_axis_end_cm1
                ),
                "axis_length": int(
                    self.low_frequency_axis_length
                ),
                "uniform_spacing_cm1": float(
                    self.low_frequency_uniform_spacing_cm1
                ),
                "sigma_points": float(
                    self.low_frequency_sigma_points
                ),
                "number_of_training_spectra": int(
                    self.low_frequency_number_of_training_spectra
                ),
            }

        # --------------------------------------------------------------
        # Blended metadata
        # --------------------------------------------------------------

        if (
            self.prior_method
            == TRAINING_BLENDED_FREQUENCY_MEDIAN
        ):
            state[
                "blended_frequency_prior"
            ] = {
                "peak_component_ratio": float(
                    self.blended_peak_component_ratio
                ),
            }

        return state

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "PriorResidualTransformer":
        """
        从 v1-v6 checkpoint metadata 恢复 transformer。

        兼容：
            v1-v3：早期 fixed median
            v4：median / PCA
            v5：low-frequency
            v6：blended-frequency
        """

        if not isinstance(
            state,
            dict,
        ):
            raise TypeError(
                "prior_residual_state必须是字典。"
            )

        if not bool(
            state.get(
                "enabled",
                False,
            )
        ):
            raise ValueError(
                "prior_residual_state没有启用。"
            )

        schema_version = int(
            state.get(
                "schema_version",
                0,
            )
        )

        if schema_version not in {
            1,
            2,
            3,
            4,
            5,
            6,
        }:
            raise ValueError(
                "不支持的prior_residual_state版本。"
            )

        if (
            state.get("domain")
            != "spectrum_global_minmax_normalized"
        ):
            raise ValueError(
                "检查点中的先验残差数据域无效。"
            )

        prior_method = str(
            state.get(
                "prior_method",
                TRAINING_POINTWISE_MEDIAN,
            )
        ).strip().lower()

        if (
            prior_method
            not in SUPPORTED_PRIOR_METHODS
        ):
            raise ValueError(
                "检查点中的prior_method不受支持："
                f"{prior_method}。"
            )

        if (
            schema_version < 4
            and prior_method
            != TRAINING_POINTWISE_MEDIAN
        ):
            raise ValueError(
                "v1-v3检查点只支持"
                "training_pointwise_median。"
            )

        if (
            prior_method
            == PCA_RECONSTRUCTION
            and schema_version < 4
        ):
            raise ValueError(
                "PCA先验必须使用v4或更高checkpoint状态。"
            )

        if (
            prior_method
            == TRAINING_LOW_FREQUENCY_MEDIAN
            and schema_version < 5
        ):
            raise ValueError(
                "training_low_frequency_median"
                "必须使用v5或更高checkpoint状态。"
            )

        if (
            prior_method
            == TRAINING_BLENDED_FREQUENCY_MEDIAN
            and schema_version < 6
        ):
            raise ValueError(
                "training_blended_frequency_median"
                "必须使用v6 checkpoint状态。"
            )

        residual_state = state.get(
            "residual_normalization"
        )

        if not isinstance(
            residual_state,
            dict,
        ):
            raise ValueError(
                "检查点缺少residual_normalization。"
            )

        method = str(
            residual_state.get(
                "method",
                "",
            )
        ).strip().lower()

        if (
            method
            not in SUPPORTED_NORMALIZATION_METHODS
        ):
            raise ValueError(
                "检查点中的残差归一化方法不受支持。"
            )

        if (
            schema_version == 1
            and method != GLOBAL_MAXABS
        ):
            raise ValueError(
                "v1检查点只支持global_maxabs。"
            )

        if (
            schema_version == 2
            and method
            == POINTWISE_MAD_ASINH
        ):
            raise ValueError(
                "pointwise_mad_asinh"
                "至少需要v3 checkpoint状态。"
            )

        constructor_keywords: dict[str, Any] = {}

        constructor_keywords.update(
            cls._pca_constructor_keywords(
                state,
                prior_method,
            )
        )

        constructor_keywords.update(
            cls._low_frequency_constructor_keywords(
                state,
                prior_method,
            )
        )

        constructor_keywords.update(
            cls._blended_constructor_keywords(
                state,
                prior_method,
            )
        )

        transformer = cls(
            prior_method=prior_method,
            normalization_method=method,
            target_abs_max=float(
                residual_state[
                    "target_abs_max"
                ]
            ),
            residual_quantile=float(
                residual_state.get(
                    "residual_quantile",
                    99.5,
                )
            ),
            pointwise_scale_floor_quantile=float(
                residual_state.get(
                    "pointwise_scale_floor_quantile",
                    10.0,
                )
            ),
            mad_scale_factor=float(
                residual_state.get(
                    "mad_scale_factor",
                    1.4826,
                )
            ),
            epsilon=float(
                residual_state.get(
                    "epsilon",
                    1.0e-8,
                )
            ),
            **constructor_keywords,
        )

        prior = np.asarray(
            state[
                "prior_normalized_intensity"
            ],
            dtype=np.float32,
        ).reshape(-1)

        transformer._validate_vector(
            prior,
            expected_length=None,
            name="检查点先验光谱",
            strictly_positive=False,
            minimum_length=2,
        )

        maximum = float(
            residual_state[
                "training_max_abs_residual"
            ]
        )

        transformer._validate_positive_finite(
            maximum,
            "检查点中的训练残差最大值",
        )

        percentiles = (
            transformer._load_percentile_dictionary(
                residual_state.get(
                    "training_abs_residual_percentiles",
                    {},
                ),
                "训练残差分位数统计",
            )
        )

        transformer.prior = (
            prior.copy()
        )

        transformer.training_max_abs_residual = (
            maximum
        )

        transformer.training_abs_residual_percentiles = (
            percentiles
        )

        if method == GLOBAL_MAXABS:
            transformer._restore_global_maxabs(
                residual_state
            )

        elif method == ROBUST_ASINH:
            transformer._restore_robust_asinh(
                residual_state
            )

        else:
            transformer._restore_pointwise_mad_asinh(
                residual_state
            )

        if (
            prior_method
            == PCA_RECONSTRUCTION
        ):
            transformer._restore_pca_prior(
                state
            )

        if prior_method in {
            TRAINING_LOW_FREQUENCY_MEDIAN,
            TRAINING_BLENDED_FREQUENCY_MEDIAN,
        }:
            transformer._restore_low_frequency_metadata(
                state
            )

        transformer._check_fitted()

        return transformer

    # ------------------------------------------------------------------
    # checkpoint constructor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pca_constructor_keywords(
        state: dict[str, Any],
        prior_method: str,
    ) -> dict[str, Any]:
        if (
            prior_method
            != PCA_RECONSTRUCTION
        ):
            return {}

        pca_state = state.get(
            "pca_prior"
        )

        if not isinstance(
            pca_state,
            dict,
        ):
            raise ValueError(
                "PCA checkpoint缺少pca_prior状态。"
            )

        return {
            "pca_explained_variance_ratio": float(
                pca_state[
                    "explained_variance_ratio_target"
                ]
            ),
            "pca_max_components": (
                pca_state.get(
                    "max_components"
                )
            ),
            "pca_sampling_strategy": str(
                pca_state[
                    "sampling_strategy"
                ]
            ),
            "pca_score_clip_standard_deviations": float(
                pca_state[
                    "score_clip_standard_deviations"
                ]
            ),
        }

    @staticmethod
    def _low_frequency_constructor_keywords(
        state: dict[str, Any],
        prior_method: str,
    ) -> dict[str, Any]:
        if prior_method not in {
            TRAINING_LOW_FREQUENCY_MEDIAN,
            TRAINING_BLENDED_FREQUENCY_MEDIAN,
        }:
            return {}

        low_state = state.get(
            "low_frequency_prior"
        )

        if not isinstance(
            low_state,
            dict,
        ):
            raise ValueError(
                "低频prior checkpoint缺少"
                "low_frequency_prior状态。"
            )

        return {
            "low_frequency_method": str(
                low_state.get(
                    "method",
                    "gaussian",
                )
            ),
            "low_frequency_sigma_cm1": float(
                low_state[
                    "sigma_cm1"
                ]
            ),
            "low_frequency_truncate": float(
                low_state.get(
                    "truncate",
                    4.0,
                )
            ),
        }

    @staticmethod
    def _blended_constructor_keywords(
        state: dict[str, Any],
        prior_method: str,
    ) -> dict[str, Any]:
        if (
            prior_method
            != TRAINING_BLENDED_FREQUENCY_MEDIAN
        ):
            return {}

        blended_state = state.get(
            "blended_frequency_prior"
        )

        if not isinstance(
            blended_state,
            dict,
        ):
            raise ValueError(
                "D2.4 checkpoint缺少"
                "blended_frequency_prior状态。"
            )

        return {
            "blended_peak_component_ratio": float(
                blended_state[
                    "peak_component_ratio"
                ]
            )
        }

    # ------------------------------------------------------------------
    # checkpoint restore - PCA
    # ------------------------------------------------------------------

    def _restore_pca_prior(
        self,
        state: dict[str, Any],
    ) -> None:
        pca_state = state.get(
            "pca_prior"
        )

        if not isinstance(
            pca_state,
            dict,
        ):
            raise ValueError(
                "PCA checkpoint缺少pca_prior状态。"
            )

        mean = np.asarray(
            pca_state["mean"],
            dtype=np.float32,
        ).reshape(-1)

        components = np.asarray(
            pca_state["components"],
            dtype=np.float32,
        )

        score_mean = np.asarray(
            pca_state[
                "training_score_mean"
            ],
            dtype=np.float64,
        ).reshape(-1)

        covariance = np.asarray(
            pca_state[
                "training_score_covariance"
            ],
            dtype=np.float64,
        )

        score_std = np.asarray(
            pca_state[
                "score_standard_deviation"
            ],
            dtype=np.float64,
        ).reshape(-1)

        explained_variance = np.asarray(
            pca_state[
                "explained_variance"
            ],
            dtype=np.float64,
        ).reshape(-1)

        explained_ratio = np.asarray(
            pca_state[
                "explained_variance_ratio"
            ],
            dtype=np.float64,
        ).reshape(-1)

        if (
            mean.size
            != self.prior.size
            or components.ndim != 2
            or components.shape[1]
            != self.prior.size
            or components.shape[0] < 1
        ):
            raise ValueError(
                "PCA checkpoint中的mean/components"
                "形状无效。"
            )

        component_count = int(
            components.shape[0]
        )

        if (
            score_mean.size
            != component_count
            or score_std.size
            != component_count
            or covariance.shape
            != (
                component_count,
                component_count,
            )
            or explained_variance.size
            != component_count
            or explained_ratio.size
            != component_count
        ):
            raise ValueError(
                "PCA checkpoint的统计量形状无效。"
            )

        if (
            not np.isfinite(
                mean
            ).all()
            or not np.isfinite(
                components
            ).all()
            or not np.isfinite(
                score_mean
            ).all()
            or not np.isfinite(
                covariance
            ).all()
            or not np.isfinite(
                score_std
            ).all()
            or np.any(
                score_std
                <= self.epsilon
            )
            or np.any(
                explained_variance
                < 0.0
            )
            or np.any(
                explained_ratio
                < 0.0
            )
        ):
            raise ValueError(
                "PCA checkpoint包含无效数值。"
            )

        if not np.allclose(
            mean,
            self.prior,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "PCA checkpoint的默认prior"
                "与PCA mean不一致。"
            )

        if not np.allclose(
            covariance,
            covariance.T,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "PCA score covariance必须对称。"
            )

        self.pca_mean = (
            mean.copy()
        )

        self.pca_components = (
            components.copy()
        )

        self.pca_training_score_mean = (
            score_mean.copy()
        )

        self.pca_training_score_covariance = (
            covariance.copy()
        )

        self.pca_score_standard_deviation = (
            score_std.copy()
        )

        self.pca_explained_variance = (
            explained_variance.copy()
        )

        self.pca_explained_variance_ratio_ = (
            explained_ratio.copy()
        )

        self.pca_number_of_training_spectra = int(
            pca_state[
                "number_of_training_spectra"
            ]
        )

        if (
            self.pca_number_of_training_spectra
            < 2
        ):
            raise ValueError(
                "PCA checkpoint训练光谱数量"
                "至少应为2。"
            )

    # ------------------------------------------------------------------
    # checkpoint restore - low-frequency
    # ------------------------------------------------------------------

    def _restore_low_frequency_metadata(
        self,
        state: dict[str, Any],
    ) -> None:
        low_state = state.get(
            "low_frequency_prior"
        )

        if not isinstance(
            low_state,
            dict,
        ):
            raise ValueError(
                "低频prior checkpoint缺少"
                "low_frequency_prior。"
            )

        axis_start = float(
            low_state[
                "axis_start_cm1"
            ]
        )

        axis_end = float(
            low_state[
                "axis_end_cm1"
            ]
        )

        axis_length = int(
            low_state[
                "axis_length"
            ]
        )

        spacing = float(
            low_state[
                "uniform_spacing_cm1"
            ]
        )

        sigma_points = float(
            low_state[
                "sigma_points"
            ]
        )

        number_training = int(
            low_state[
                "number_of_training_spectra"
            ]
        )

        if (
            not np.isfinite(
                axis_start
            )
            or not np.isfinite(
                axis_end
            )
            or axis_end <= axis_start
        ):
            raise ValueError(
                "低频prior checkpoint中的拉曼轴范围无效。"
            )

        if (
            axis_length != self.prior.size
            or axis_length < 2
        ):
            raise ValueError(
                "低频prior checkpoint中的axis_length"
                "与prior长度不一致。"
            )

        self._validate_positive_finite(
            spacing,
            "低频prior checkpoint的轴间隔",
        )

        self._validate_positive_finite(
            sigma_points,
            "低频prior checkpoint的sigma_points",
        )

        if number_training < 2:
            raise ValueError(
                "低频prior checkpoint训练光谱数量"
                "至少应为2。"
            )

        self.low_frequency_axis_start_cm1 = (
            axis_start
        )

        self.low_frequency_axis_end_cm1 = (
            axis_end
        )

        self.low_frequency_axis_length = (
            axis_length
        )

        self.low_frequency_uniform_spacing_cm1 = (
            spacing
        )

        self.low_frequency_sigma_points = (
            sigma_points
        )

        self.low_frequency_number_of_training_spectra = (
            number_training
        )

    # ------------------------------------------------------------------
    # checkpoint restore - residual normalization
    # ------------------------------------------------------------------

    def _restore_global_maxabs(
        self,
        residual_state: dict[str, Any],
    ) -> None:
        scale = float(
            residual_state[
                "scale"
            ]
        )

        self._validate_positive_finite(
            scale,
            "检查点中的残差scale",
        )

        expected_scale = (
            self.training_max_abs_residual
            / self.target_abs_max
        )

        if not np.isclose(
            scale,
            expected_scale,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "检查点中的残差scale与"
                "training_max_abs_residual/"
                "target_abs_max不一致。"
            )

        self.residual_scale = (
            scale
        )

    def _restore_robust_asinh(
        self,
        residual_state: dict[str, Any],
    ) -> None:
        scale = float(
            residual_state[
                "scale"
            ]
        )

        quantile_value = float(
            residual_state[
                "training_abs_residual_quantile"
            ]
        )

        normalizer = float(
            residual_state[
                "asinh_normalizer"
            ]
        )

        self._validate_positive_finite(
            scale,
            "检查点中的残差scale",
        )

        self._validate_positive_finite(
            quantile_value,
            "检查点中的残差分位数尺度",
        )

        if not np.isclose(
            scale,
            quantile_value,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "检查点中的scale与"
                "training_abs_residual_quantile不一致。"
            )

        expected_normalizer = float(
            np.arcsinh(
                self.training_max_abs_residual
                / scale
            )
        )

        self._validate_normalizer(
            normalizer,
            expected_normalizer,
        )

        self.residual_scale = (
            scale
        )

        self.training_abs_residual_quantile = (
            quantile_value
        )

        self.asinh_normalizer = (
            normalizer
        )

    def _restore_pointwise_mad_asinh(
        self,
        residual_state: dict[str, Any],
    ) -> None:
        pointwise_scale = np.asarray(
            residual_state[
                "pointwise_scale"
            ],
            dtype=np.float32,
        ).reshape(-1)

        self._validate_vector(
            pointwise_scale,
            expected_length=self.prior.size,
            name="检查点逐波数MAD尺度",
            strictly_positive=True,
        )

        scale_floor = float(
            residual_state[
                "pointwise_scale_floor"
            ]
        )

        standardized_scale = float(
            residual_state[
                "standardized_residual_scale"
            ]
        )

        maximum_standardized = float(
            residual_state[
                "training_max_abs_standardized_residual"
            ]
        )

        quantile_standardized = float(
            residual_state[
                "training_abs_standardized_residual_quantile"
            ]
        )

        normalizer = float(
            residual_state[
                "asinh_normalizer"
            ]
        )

        self._validate_positive_finite(
            scale_floor,
            "检查点中的pointwise_scale_floor",
        )

        self._validate_positive_finite(
            standardized_scale,
            "检查点中的standardized_residual_scale",
        )

        self._validate_positive_finite(
            maximum_standardized,
            "检查点中的标准化残差最大值",
        )

        self._validate_positive_finite(
            quantile_standardized,
            "检查点中的标准化残差分位数",
        )

        if not np.isclose(
            standardized_scale,
            quantile_standardized,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "检查点中的standardized_residual_scale"
                "与训练标准化残差分位数不一致。"
            )

        expected_normalizer = float(
            np.arcsinh(
                maximum_standardized
                / standardized_scale
            )
        )

        self._validate_normalizer(
            normalizer,
            expected_normalizer,
        )

        self.pointwise_scale = (
            pointwise_scale.copy()
        )

        self.pointwise_scale_floor = (
            scale_floor
        )

        self.residual_scale = (
            standardized_scale
        )

        self.training_max_abs_standardized_residual = (
            maximum_standardized
        )

        self.training_abs_standardized_residual_quantile = (
            quantile_standardized
        )

        self.asinh_normalizer = (
            normalizer
        )

        self.training_pointwise_scale_percentiles = (
            self._load_percentile_dictionary(
                residual_state.get(
                    "training_pointwise_scale_percentiles",
                    {},
                ),
                "逐波数尺度分位数统计",
            )
        )

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------

    def _clear_low_frequency_state(
        self,
    ) -> None:
        self.low_frequency_axis_start_cm1 = None
        self.low_frequency_axis_end_cm1 = None

        self.low_frequency_axis_length = None

        self.low_frequency_uniform_spacing_cm1 = (
            None
        )

        self.low_frequency_sigma_points = None

        self.low_frequency_number_of_training_spectra = (
            None
        )

    def _build_asinh_normalizer(
        self,
        *,
        maximum: float,
        scale: float,
        description: str,
    ) -> float:
        normalizer = float(
            np.arcsinh(
                maximum
                / scale
            )
        )

        self._validate_positive_finite(
            normalizer,
            f"{description}的asinh归一化因子",
        )

        return normalizer

    def _validate_normalizer(
        self,
        actual: float,
        expected: float,
    ) -> None:
        self._validate_positive_finite(
            actual,
            "检查点中的asinh_normalizer",
        )

        self._validate_positive_finite(
            expected,
            "根据检查点统计重算的asinh_normalizer",
        )

        if not np.isclose(
            actual,
            expected,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "检查点中的asinh_normalizer"
                "与训练统计不一致。"
            )

    @staticmethod
    def _validate_array(
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                f"{name}必须是二维数组[N,L]，"
                f"实际为{array.shape}。"
            )

        if (
            array.shape[0] == 0
            or array.shape[1] < 2
        ):
            raise ValueError(
                f"{name}不能为空且光谱长度至少为2。"
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"{name}中存在NaN或无穷值。"
            )

        return array

    def _validate_raman_shift_axis(
        self,
        raman_shift: np.ndarray | None,
        *,
        expected_length: int,
    ) -> np.ndarray:
        if raman_shift is None:
            raise ValueError(
                f"{self.prior_method}必须提供统一训练raman_shift，"
                "以cm^-1定义Gaussian sigma。"
            )

        axis = np.asarray(
            raman_shift,
            dtype=np.float64,
        ).reshape(-1)

        if axis.size != int(
            expected_length
        ):
            raise ValueError(
                "raman_shift长度与训练光谱不一致："
                f"轴长度={axis.size}，"
                f"光谱长度={expected_length}。"
            )

        if axis.size < 2:
            raise ValueError(
                "raman_shift至少需要2个点。"
            )

        if not np.isfinite(
            axis
        ).all():
            raise ValueError(
                "raman_shift包含NaN或无穷值。"
            )

        differences = np.diff(
            axis
        )

        if not np.all(
            differences > 0.0
        ):
            raise ValueError(
                "raman_shift必须严格递增。"
            )

        return axis

    @staticmethod
    def _validate_vector(
        values: np.ndarray,
        *,
        expected_length: int | None,
        name: str,
        strictly_positive: bool,
        minimum_length: int = 1,
    ) -> None:
        vector = np.asarray(
            values
        ).reshape(-1)

        if vector.size < int(
            minimum_length
        ):
            raise ValueError(
                f"{name}长度无效。"
            )

        if (
            expected_length is not None
            and vector.size
            != int(expected_length)
        ):
            raise ValueError(
                f"{name}长度与先验不一致："
                f"实际={vector.size}，"
                f"期望={expected_length}。"
            )

        if not np.isfinite(
            vector
        ).all():
            raise ValueError(
                f"{name}包含NaN或无穷值。"
            )

        if (
            strictly_positive
            and np.any(
                vector <= 0.0
            )
        ):
            raise ValueError(
                f"{name}必须全部大于0。"
            )

    def _validate_positive_finite(
        self,
        value: float,
        description: str,
    ) -> None:
        parsed = float(
            value
        )

        if (
            not np.isfinite(
                parsed
            )
            or parsed <= self.epsilon
        ):
            raise ValueError(
                f"{description}必须是有限正数。"
            )

    def _check_fitted(
        self,
    ) -> None:
        if (
            self.prior is None
            or self.residual_scale is None
            or self.training_max_abs_residual
            is None
        ):
            raise RuntimeError(
                "先验残差变换器尚未使用训练集fit()。"
            )

        if (
            self.normalization_method
            in {
                ROBUST_ASINH,
                POINTWISE_MAD_ASINH,
            }
            and self.asinh_normalizer
            is None
        ):
            raise RuntimeError(
                "asinh残差变换状态不完整。"
            )

        if (
            self.normalization_method
            == POINTWISE_MAD_ASINH
            and self.pointwise_scale
            is None
        ):
            raise RuntimeError(
                "pointwise_mad_asinh状态不完整。"
            )

        if (
            self.prior_method
            == PCA_RECONSTRUCTION
            and (
                self.pca_mean is None
                or self.pca_components is None
                or self.pca_training_score_mean
                is None
                or self.pca_training_score_covariance
                is None
                or self.pca_score_standard_deviation
                is None
            )
        ):
            raise RuntimeError(
                "PCA prior状态不完整。"
            )

    def _check_length(
        self,
        values: np.ndarray,
    ) -> None:
        self._check_fitted()

        if (
            values.shape[1]
            != self.prior.size
        ):
            raise ValueError(
                "输入光谱长度与prior长度不一致："
                f"输入={values.shape[1]}，"
                f"prior={self.prior.size}。"
            )

    @staticmethod
    def _percentile_key(
        percentile: float,
    ) -> str:
        text = (
            f"{float(percentile):g}"
            .replace(
                ".",
                "_",
            )
        )

        return f"p{text}"

    @classmethod
    def _summarize_nonnegative(
        cls,
        values: np.ndarray,
        *,
        include_max: bool,
    ) -> dict[str, float]:
        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if array.size == 0:
            raise ValueError(
                "无法统计空数组。"
            )

        if (
            not np.isfinite(
                array
            ).all()
            or np.any(
                array < 0.0
            )
        ):
            raise ValueError(
                "统计数组必须是有限非负数。"
            )

        percentiles = np.percentile(
            array,
            _PERCENTILE_LEVELS,
        )

        summary = {
            cls._percentile_key(level): float(
                value
            )
            for level, value in zip(
                _PERCENTILE_LEVELS,
                percentiles,
            )
        }

        if include_max:
            summary[
                "max"
            ] = float(
                np.max(
                    array
                )
            )

        return summary

    def _load_percentile_dictionary(
        self,
        values: Any,
        description: str,
    ) -> dict[str, float]:
        if values is None:
            return {}

        if not isinstance(
            values,
            dict,
        ):
            raise ValueError(
                f"{description}必须是字典。"
            )

        result: dict[str, float] = {}

        for key, value in values.items():
            parsed = float(
                value
            )

            if (
                not np.isfinite(
                    parsed
                )
                or parsed < 0.0
            ):
                raise ValueError(
                    f"{description}包含无效数值。"
                )

            result[
                str(key)
            ] = parsed

        return result
"""训练集先验与SERS光谱残差变换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GLOBAL_MAXABS = "global_maxabs"
ROBUST_ASINH = "robust_asinh"
POINTWISE_MAD_ASINH = "pointwise_mad_asinh"

SUPPORTED_NORMALIZATION_METHODS = {
    GLOBAL_MAXABS,
    ROBUST_ASINH,
    POINTWISE_MAD_ASINH,
}

_PERCENTILE_LEVELS = np.asarray(
    [50.0, 90.0, 95.0, 99.0, 99.5, 99.9],
    dtype=np.float64,
)


@dataclass
class PriorResidualTransformer:
    """使用训练集逐点中位数先验，把完整光谱变换到残差域。

    输入和输出均为二维数组 ``[N, L]``。输入光谱必须已经：

    1. 插值到统一拉曼位移轴；
    2. 使用只在训练集上拟合的全局参数完成强度归一化；
    3. 尚未进行U-Net长度补齐。

    支持三种残差归一化方法：

    ``global_maxabs``
        D2旧版全局最大绝对值线性缩放。

    ``robust_asinh``
        D2旧版全局分位数asinh缩放。所有波数点共享一个标量尺度。

    ``pointwise_mad_asinh``
        D2.1逐波数稳健缩放。先用训练集在每个波数点计算
        ``1.4826 * MAD``，再对标准化残差使用全局可逆asinh变换。
        因此，同样大小的模型输出在安静区会恢复成较小的原始残差，
        在真实高变异峰区则允许恢复成较大的残差。
    """

    normalization_method: str = GLOBAL_MAXABS
    target_abs_max: float = 1.0
    residual_quantile: float = 99.5
    pointwise_scale_floor_quantile: float = 10.0
    mad_scale_factor: float = 1.4826
    epsilon: float = 1.0e-8

    prior: np.ndarray | None = None

    # global_maxabs中为线性残差尺度；robust_asinh中为原始残差分位数；
    # pointwise_mad_asinh中为逐点标准化后残差的分位数。
    residual_scale: float | None = None

    training_max_abs_residual: float | None = None
    training_abs_residual_quantile: float | None = None
    asinh_normalizer: float | None = None

    # D2.1专用状态。
    pointwise_scale: np.ndarray | None = None
    pointwise_scale_floor: float | None = None
    training_max_abs_standardized_residual: float | None = None
    training_abs_standardized_residual_quantile: float | None = None

    training_abs_residual_percentiles: dict[str, float] | None = None
    training_pointwise_scale_percentiles: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.normalization_method = str(
            self.normalization_method
        ).strip().lower()
        self.target_abs_max = float(self.target_abs_max)
        self.residual_quantile = float(self.residual_quantile)
        self.pointwise_scale_floor_quantile = float(
            self.pointwise_scale_floor_quantile
        )
        self.mad_scale_factor = float(self.mad_scale_factor)
        self.epsilon = float(self.epsilon)

        if self.normalization_method not in SUPPORTED_NORMALIZATION_METHODS:
            supported = ", ".join(sorted(SUPPORTED_NORMALIZATION_METHODS))
            raise ValueError(
                "不支持的残差缩放方法："
                f"{self.normalization_method}。可用方法：{supported}。"
            )

        if not 0.0 < self.target_abs_max <= 1.0:
            raise ValueError("target_abs_max必须大于0且不大于1。")

        if not 0.0 < self.residual_quantile < 100.0:
            raise ValueError("residual_quantile必须大于0且小于100。")

        if not 0.0 <= self.pointwise_scale_floor_quantile < 100.0:
            raise ValueError(
                "pointwise_scale_floor_quantile必须大于等于0且小于100。"
            )

        if self.mad_scale_factor <= 0.0:
            raise ValueError("mad_scale_factor必须大于0。")

        if self.epsilon <= 0.0:
            raise ValueError("epsilon必须大于0。")

    def fit(
        self,
        training_spectra: np.ndarray,
    ) -> "PriorResidualTransformer":
        """仅使用训练集拟合先验和残差变换参数。"""

        values = self._validate_array(training_spectra, "training_spectra")
        if values.shape[0] < 2:
            raise ValueError("先验残差模型至少需要2条训练光谱。")

        prior = np.median(values, axis=0).astype(np.float32, copy=False)
        residuals = values.astype(np.float64, copy=False) - prior[
            np.newaxis, :
        ].astype(np.float64, copy=False)
        absolute_residuals = np.abs(residuals)
        maximum = float(np.max(absolute_residuals))

        if maximum <= self.epsilon:
            raise ValueError(
                "训练光谱相对于逐点中位数先验的残差几乎为零。"
                "请确认没有继续使用复制光谱。"
            )

        self.prior = prior.copy()
        self.training_max_abs_residual = maximum
        self.training_abs_residual_percentiles = self._summarize_nonnegative(
            absolute_residuals,
            include_max=True,
        )

        # 每次fit前清空所有方法专用状态，避免同一对象重复fit后残留旧值。
        self.residual_scale = None
        self.training_abs_residual_quantile = None
        self.asinh_normalizer = None
        self.pointwise_scale = None
        self.pointwise_scale_floor = None
        self.training_max_abs_standardized_residual = None
        self.training_abs_standardized_residual_quantile = None
        self.training_pointwise_scale_percentiles = None

        if self.normalization_method == GLOBAL_MAXABS:
            self.residual_scale = maximum / self.target_abs_max
            return self

        if self.normalization_method == ROBUST_ASINH:
            quantile_value = float(
                np.percentile(absolute_residuals, self.residual_quantile)
            )
            self._validate_positive_finite(
                quantile_value,
                "训练残差的稳健分位数尺度",
            )

            self.residual_scale = quantile_value
            self.training_abs_residual_quantile = quantile_value
            self.asinh_normalizer = self._build_asinh_normalizer(
                maximum=maximum,
                scale=quantile_value,
                description="训练残差",
            )
            return self

        self._fit_pointwise_mad_asinh(residuals)
        return self

    def _fit_pointwise_mad_asinh(self, residuals: np.ndarray) -> None:
        """拟合D2.1逐波数MAD尺度和标准化残差asinh参数。"""

        residual_median = np.median(residuals, axis=0)
        pointwise_mad = np.median(
            np.abs(residuals - residual_median[np.newaxis, :]),
            axis=0,
        )
        raw_pointwise_scale = self.mad_scale_factor * pointwise_mad

        positive_scales = raw_pointwise_scale[
            np.isfinite(raw_pointwise_scale)
            & (raw_pointwise_scale > self.epsilon)
        ]
        if positive_scales.size == 0:
            raise ValueError(
                "所有波数点的MAD尺度都几乎为零，无法拟合"
                "pointwise_mad_asinh。请检查训练光谱是否被复制。"
            )

        scale_floor = float(
            np.percentile(
                positive_scales,
                self.pointwise_scale_floor_quantile,
            )
        )
        scale_floor = max(scale_floor, self.epsilon)

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

        standardized_residuals = residuals / pointwise_scale[np.newaxis, :]
        absolute_standardized = np.abs(standardized_residuals)
        maximum_standardized = float(np.max(absolute_standardized))
        quantile_standardized = float(
            np.percentile(absolute_standardized, self.residual_quantile)
        )

        self._validate_positive_finite(
            maximum_standardized,
            "训练标准化残差最大值",
        )
        self._validate_positive_finite(
            quantile_standardized,
            "训练标准化残差稳健分位数尺度",
        )

        self.pointwise_scale = pointwise_scale.astype(
            np.float32,
            copy=False,
        )
        self.pointwise_scale_floor = scale_floor
        self.training_pointwise_scale_percentiles = (
            self._summarize_nonnegative(pointwise_scale, include_max=True)
        )
        self.training_max_abs_standardized_residual = maximum_standardized
        self.training_abs_standardized_residual_quantile = (
            quantile_standardized
        )

        # 保留residual_scale这个标量属性，便于现有训练和诊断代码读取。
        # 在D2.1中它表示“逐点标准化之后”的稳健分位数，而不是原始残差尺度。
        self.residual_scale = quantile_standardized
        self.asinh_normalizer = self._build_asinh_normalizer(
            maximum=maximum_standardized,
            scale=quantile_standardized,
            description="训练标准化残差",
        )

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """将完整归一化光谱转换为DDPM学习的残差域。"""

        self._check_fitted()
        values = self._validate_array(spectra, "spectra")
        self._check_length(values)

        residuals = values.astype(np.float64, copy=False) - self.prior[
            np.newaxis, :
        ].astype(np.float64, copy=False)

        if self.normalization_method == GLOBAL_MAXABS:
            scaled = residuals / float(self.residual_scale)
        elif self.normalization_method == ROBUST_ASINH:
            scaled = self._asinh_forward(
                residuals / float(self.residual_scale)
            )
        else:
            standardized = residuals / self.pointwise_scale[np.newaxis, :]
            scaled = self._asinh_forward(
                standardized / float(self.residual_scale)
            )

        if not np.isfinite(scaled).all():
            raise RuntimeError("残差变换结果包含NaN或无穷值。")

        return scaled.astype(np.float32, copy=False)

    def inverse_transform(self, scaled_residuals: np.ndarray) -> np.ndarray:
        """取消残差变换并加回训练集逐点中位数先验。"""

        self._check_fitted()
        values = self._validate_array(scaled_residuals, "scaled_residuals")
        self._check_length(values)

        if self.normalization_method == GLOBAL_MAXABS:
            residuals = values.astype(np.float64, copy=False) * float(
                self.residual_scale
            )
        elif self.normalization_method == ROBUST_ASINH:
            residuals = float(self.residual_scale) * self._asinh_inverse(values)
        else:
            standardized = float(self.residual_scale) * self._asinh_inverse(
                values
            )
            residuals = self.pointwise_scale[np.newaxis, :] * standardized

        restored = self.prior[np.newaxis, :] + residuals
        if not np.isfinite(restored).all():
            raise RuntimeError(
                "残差逆变换结果包含NaN或无穷值。"
                "请检查生成残差是否远超训练范围。"
            )

        return restored.astype(np.float32, copy=False)

    def _asinh_forward(self, values_divided_by_scale: np.ndarray) -> np.ndarray:
        return (
            self.target_abs_max
            * np.arcsinh(values_divided_by_scale)
            / float(self.asinh_normalizer)
        )

    def _asinh_inverse(self, scaled_values: np.ndarray) -> np.ndarray:
        sinh_argument = (
            scaled_values.astype(np.float64, copy=False)
            / self.target_abs_max
            * float(self.asinh_normalizer)
        )
        return np.sinh(sinh_argument)

    def prior_batch(self, number_of_spectra: int) -> np.ndarray:
        """返回多份训练集先验。"""

        self._check_fitted()
        count = int(number_of_spectra)
        if count <= 0:
            raise ValueError("number_of_spectra必须大于0。")

        return np.repeat(
            self.prior[np.newaxis, :],
            count,
            axis=0,
        ).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        """生成可保存到checkpoint metadata中的状态。"""

        self._check_fitted()
        residual_state: dict[str, Any] = {
            "method": self.normalization_method,
            "training_max_abs_residual": float(
                self.training_max_abs_residual
            ),
            "target_abs_max": float(self.target_abs_max),
            "epsilon": float(self.epsilon),
            "training_abs_residual_percentiles": dict(
                self.training_abs_residual_percentiles or {}
            ),
        }

        if self.normalization_method == GLOBAL_MAXABS:
            residual_state["scale"] = float(self.residual_scale)
        elif self.normalization_method == ROBUST_ASINH:
            residual_state.update(
                {
                    "scale": float(self.residual_scale),
                    "residual_quantile": float(self.residual_quantile),
                    "training_abs_residual_quantile": float(
                        self.training_abs_residual_quantile
                    ),
                    "asinh_normalizer": float(self.asinh_normalizer),
                }
            )
        else:
            residual_state.update(
                {
                    "residual_quantile": float(self.residual_quantile),
                    "mad_scale_factor": float(self.mad_scale_factor),
                    "pointwise_scale_floor_quantile": float(
                        self.pointwise_scale_floor_quantile
                    ),
                    "pointwise_scale_floor": float(self.pointwise_scale_floor),
                    "pointwise_scale": self.pointwise_scale.tolist(),
                    "standardized_residual_scale": float(self.residual_scale),
                    "training_abs_standardized_residual_quantile": float(
                        self.training_abs_standardized_residual_quantile
                    ),
                    "training_max_abs_standardized_residual": float(
                        self.training_max_abs_standardized_residual
                    ),
                    "asinh_normalizer": float(self.asinh_normalizer),
                    "training_pointwise_scale_percentiles": dict(
                        self.training_pointwise_scale_percentiles or {}
                    ),
                }
            )

        return {
            "schema_version": 3,
            "enabled": True,
            "domain": "spectrum_global_minmax_normalized",
            "prior_method": "training_pointwise_median",
            "prior_normalized_intensity": self.prior.tolist(),
            "residual_normalization": residual_state,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "PriorResidualTransformer":
        """从v1、v2或v3 checkpoint metadata恢复变换器。"""

        if not isinstance(state, dict):
            raise TypeError("prior_residual_state必须是字典。")
        if not bool(state.get("enabled", False)):
            raise ValueError("prior_residual_state没有启用。")

        schema_version = int(state.get("schema_version", 0))
        if schema_version not in {1, 2, 3}:
            raise ValueError("不支持的prior_residual_state版本。")
        if state.get("domain") != "spectrum_global_minmax_normalized":
            raise ValueError("检查点中的先验残差数据域无效。")
        if state.get("prior_method") != "training_pointwise_median":
            raise ValueError(
                "检查点中的先验方法不是training_pointwise_median。"
            )

        residual_state = state.get("residual_normalization")
        if not isinstance(residual_state, dict):
            raise ValueError("检查点缺少residual_normalization。")

        method = str(residual_state.get("method", "")).strip().lower()
        if schema_version == 1 and method != GLOBAL_MAXABS:
            raise ValueError("v1检查点只支持global_maxabs残差缩放。")
        if schema_version == 2 and method == POINTWISE_MAD_ASINH:
            raise ValueError("pointwise_mad_asinh必须使用v3检查点状态。")

        transformer = cls(
            normalization_method=method,
            target_abs_max=float(residual_state["target_abs_max"]),
            residual_quantile=float(
                residual_state.get("residual_quantile", 99.5)
            ),
            pointwise_scale_floor_quantile=float(
                residual_state.get("pointwise_scale_floor_quantile", 10.0)
            ),
            mad_scale_factor=float(
                residual_state.get("mad_scale_factor", 1.4826)
            ),
            epsilon=float(residual_state.get("epsilon", 1.0e-8)),
        )

        prior = np.asarray(
            state["prior_normalized_intensity"],
            dtype=np.float32,
        ).reshape(-1)
        transformer._validate_vector(
            prior,
            expected_length=None,
            name="检查点先验光谱",
            strictly_positive=False,
            minimum_length=2,
        )

        maximum = float(residual_state["training_max_abs_residual"])
        transformer._validate_positive_finite(
            maximum,
            "检查点中的训练残差最大值",
        )

        percentiles = transformer._load_percentile_dictionary(
            residual_state.get("training_abs_residual_percentiles", {}),
            "训练残差分位数统计",
        )

        transformer.prior = prior.copy()
        transformer.training_max_abs_residual = maximum
        transformer.training_abs_residual_percentiles = percentiles

        if method == GLOBAL_MAXABS:
            transformer._restore_global_maxabs(residual_state)
        elif method == ROBUST_ASINH:
            transformer._restore_robust_asinh(residual_state)
        else:
            transformer._restore_pointwise_mad_asinh(residual_state)

        transformer._check_fitted()
        return transformer

    def _restore_global_maxabs(self, residual_state: dict[str, Any]) -> None:
        scale = float(residual_state["scale"])
        self._validate_positive_finite(scale, "检查点中的残差scale")
        expected_scale = self.training_max_abs_residual / self.target_abs_max
        if not np.isclose(
            scale,
            expected_scale,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "检查点中的残差scale与training_max_abs_residual/"
                "target_abs_max不一致。"
            )
        self.residual_scale = scale

    def _restore_robust_asinh(self, residual_state: dict[str, Any]) -> None:
        scale = float(residual_state["scale"])
        quantile_value = float(
            residual_state["training_abs_residual_quantile"]
        )
        normalizer = float(residual_state["asinh_normalizer"])

        self._validate_positive_finite(scale, "检查点中的残差scale")
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
            raise ValueError("检查点中的scale与分位数尺度不一致。")

        expected_normalizer = float(
            np.arcsinh(self.training_max_abs_residual / scale)
        )
        self._validate_normalizer(normalizer, expected_normalizer)

        self.residual_scale = scale
        self.training_abs_residual_quantile = quantile_value
        self.asinh_normalizer = normalizer

    def _restore_pointwise_mad_asinh(
        self,
        residual_state: dict[str, Any],
    ) -> None:
        pointwise_scale = np.asarray(
            residual_state["pointwise_scale"],
            dtype=np.float32,
        ).reshape(-1)
        self._validate_vector(
            pointwise_scale,
            expected_length=self.prior.size,
            name="检查点中的逐波数MAD尺度",
            strictly_positive=True,
        )

        scale_floor = float(residual_state["pointwise_scale_floor"])
        standardized_scale = float(
            residual_state["standardized_residual_scale"]
        )
        quantile_standardized = float(
            residual_state["training_abs_standardized_residual_quantile"]
        )
        maximum_standardized = float(
            residual_state["training_max_abs_standardized_residual"]
        )
        normalizer = float(residual_state["asinh_normalizer"])

        self._validate_positive_finite(
            scale_floor,
            "检查点中的逐波数尺度下限",
        )
        self._validate_positive_finite(
            standardized_scale,
            "检查点中的标准化残差scale",
        )
        self._validate_positive_finite(
            quantile_standardized,
            "检查点中的标准化残差分位数",
        )
        self._validate_positive_finite(
            maximum_standardized,
            "检查点中的标准化残差最大值",
        )

        if np.any(pointwise_scale < scale_floor - self.epsilon):
            raise ValueError("检查点中的逐波数尺度小于保存的尺度下限。")
        if not np.isclose(
            standardized_scale,
            quantile_standardized,
            rtol=1.0e-5,
            atol=self.epsilon,
        ):
            raise ValueError(
                "检查点中的标准化残差scale与分位数尺度不一致。"
            )

        expected_normalizer = float(
            np.arcsinh(maximum_standardized / standardized_scale)
        )
        self._validate_normalizer(normalizer, expected_normalizer)

        self.pointwise_scale = pointwise_scale.copy()
        self.pointwise_scale_floor = scale_floor
        self.residual_scale = standardized_scale
        self.training_abs_standardized_residual_quantile = (
            quantile_standardized
        )
        self.training_max_abs_standardized_residual = maximum_standardized
        self.asinh_normalizer = normalizer
        self.training_pointwise_scale_percentiles = (
            self._load_percentile_dictionary(
                residual_state.get("training_pointwise_scale_percentiles", {}),
                "逐波数尺度分位数统计",
            )
        )

    def _build_asinh_normalizer(
        self,
        maximum: float,
        scale: float,
        description: str,
    ) -> float:
        normalizer = float(np.arcsinh(maximum / scale))
        if not np.isfinite(normalizer) or normalizer <= self.epsilon:
            raise ValueError(f"{description}的asinh归一化因子无效。")
        return normalizer

    def _validate_normalizer(
        self,
        normalizer: float,
        expected_normalizer: float,
    ) -> None:
        if (
            not np.isfinite(normalizer)
            or not np.isclose(
                normalizer,
                expected_normalizer,
                rtol=1.0e-5,
                atol=self.epsilon,
            )
        ):
            raise ValueError("检查点中的asinh_normalizer无效。")

    @staticmethod
    def _validate_array(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"{name}必须是二维数组[N, L]，实际为{array.shape}。"
            )
        if array.shape[0] == 0 or array.shape[1] < 2:
            raise ValueError(f"{name}不能为空且光谱长度至少为2。")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}中存在NaN或无穷值。")
        return array

    @staticmethod
    def _validate_vector(
        values: np.ndarray,
        expected_length: int | None,
        name: str,
        strictly_positive: bool,
        minimum_length: int = 1,
    ) -> None:
        if values.ndim != 1 or values.size < minimum_length:
            raise ValueError(f"{name}长度无效。")
        if expected_length is not None and values.size != expected_length:
            raise ValueError(
                f"{name}长度与先验不一致："
                f"实际为{values.size}，期望为{expected_length}。"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name}包含NaN或无穷值。")
        if strictly_positive and np.any(values <= 0.0):
            raise ValueError(f"{name}必须全部大于0。")

    def _validate_positive_finite(self, value: float, name: str) -> None:
        if not np.isfinite(value) or value <= self.epsilon:
            raise ValueError(f"{name}无效。")

    @classmethod
    def _summarize_nonnegative(
        cls,
        values: np.ndarray,
        include_max: bool,
    ) -> dict[str, float]:
        percentile_values = np.percentile(values, _PERCENTILE_LEVELS)
        result = {
            cls._percentile_key(level): float(value)
            for level, value in zip(
                _PERCENTILE_LEVELS,
                percentile_values,
                strict=True,
            )
        }
        if include_max:
            result["max"] = float(np.max(values))
        return result

    @staticmethod
    def _load_percentile_dictionary(
        raw_percentiles: Any,
        name: str,
    ) -> dict[str, float]:
        if not isinstance(raw_percentiles, dict):
            raise ValueError(f"检查点中的{name}必须是字典。")
        percentiles = {
            str(key): float(value)
            for key, value in raw_percentiles.items()
        }
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in percentiles.values()
        ):
            raise ValueError(f"检查点中的{name}无效。")
        return percentiles

    @staticmethod
    def _percentile_key(level: float) -> str:
        text = f"{float(level):g}".replace(".", "_")
        return f"p{text}"

    def _check_fitted(self) -> None:
        base_fitted = (
            self.prior is not None
            and self.residual_scale is not None
            and self.training_max_abs_residual is not None
        )
        robust_fitted = (
            self.normalization_method == GLOBAL_MAXABS
            or self.asinh_normalizer is not None
        )
        pointwise_fitted = (
            self.normalization_method != POINTWISE_MAD_ASINH
            or (
                self.pointwise_scale is not None
                and self.pointwise_scale_floor is not None
                and self.training_max_abs_standardized_residual is not None
                and self.training_abs_standardized_residual_quantile is not None
            )
        )

        if not base_fitted or not robust_fitted or not pointwise_fitted:
            raise RuntimeError("先验残差变换器尚未使用训练集fit()。")

    def _check_length(self, values: np.ndarray) -> None:
        self._check_fitted()
        if values.shape[1] != self.prior.size:
            raise ValueError(
                "输入光谱长度与先验长度不一致："
                f"输入为{values.shape[1]}，先验为{self.prior.size}。"
            )
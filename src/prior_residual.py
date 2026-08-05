"""训练集先验与SERS光谱残差变换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GLOBAL_MAXABS = "global_maxabs"
ROBUST_ASINH = "robust_asinh"

SUPPORTED_NORMALIZATION_METHODS = {
    GLOBAL_MAXABS,
    ROBUST_ASINH,
}


@dataclass
class PriorResidualTransformer:
    """
    使用训练集逐点中位数建立先验，并缩放残差。

    输入和输出均为二维数组[N, L]。这里的输入必须已经：

    1. 插值到统一拉曼位移轴；
    2. 使用只在训练集上拟合的全局参数完成强度归一化；
    3. 尚未进行U-Net长度补齐。

    global_maxabs保留D2旧版线性缩放。

    robust_asinh使用训练残差绝对值分位数作为稳健尺度，
    再通过严格可逆的asinh变换压缩极端残差。
    """

    normalization_method: str = GLOBAL_MAXABS
    target_abs_max: float = 1.0
    residual_quantile: float = 99.5
    epsilon: float = 1.0e-8

    prior: np.ndarray | None = None
    residual_scale: float | None = None
    training_max_abs_residual: float | None = None
    training_abs_residual_quantile: float | None = None
    asinh_normalizer: float | None = None

    training_abs_residual_percentiles: (
        dict[str, float] | None
    ) = None

    def __post_init__(self) -> None:
        self.normalization_method = str(
            self.normalization_method
        ).strip().lower()

        self.target_abs_max = float(
            self.target_abs_max
        )

        self.residual_quantile = float(
            self.residual_quantile
        )

        self.epsilon = float(
            self.epsilon
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

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon必须大于0。"
            )

    def fit(
        self,
        training_spectra: np.ndarray,
    ) -> "PriorResidualTransformer":
        """
        仅使用训练集建立先验，
        并拟合残差范围参数。
        """

        values = self._validate_array(
            training_spectra,
            "training_spectra",
        )

        if values.shape[0] < 2:
            raise ValueError(
                "先验残差模型至少需要2条训练光谱。"
            )

        prior = np.median(
            values,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        residuals = (
            values
            - prior[np.newaxis, :]
        )

        absolute_residuals = np.abs(
            residuals.astype(
                np.float64,
                copy=False,
            )
        )

        maximum = float(
            np.max(
                absolute_residuals
            )
        )

        if maximum <= self.epsilon:
            raise ValueError(
                "训练光谱相对于逐点中位数先验的残差"
                "几乎为零。请确认没有继续使用复制光谱。"
            )

        percentile_levels = np.asarray(
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

        percentile_values = np.percentile(
            absolute_residuals,
            percentile_levels,
        )

        self.prior = prior.copy()

        self.training_max_abs_residual = (
            maximum
        )

        self.training_abs_residual_percentiles = {
            self._percentile_key(
                level
            ): float(value)
            for level, value in zip(
                percentile_levels,
                percentile_values,
                strict=True,
            )
        }

        self.training_abs_residual_percentiles[
            "max"
        ] = maximum

        if (
            self.normalization_method
            == GLOBAL_MAXABS
        ):
            self.residual_scale = (
                maximum
                / self.target_abs_max
            )

            self.training_abs_residual_quantile = (
                None
            )

            self.asinh_normalizer = None

            return self

        quantile_value = float(
            np.percentile(
                absolute_residuals,
                self.residual_quantile,
            )
        )

        if quantile_value <= self.epsilon:
            raise ValueError(
                "训练残差的稳健分位数几乎为零。"
                "请降低residual_quantile或检查训练光谱。"
            )

        normalizer = float(
            np.arcsinh(
                maximum
                / quantile_value
            )
        )

        if (
            not np.isfinite(normalizer)
            or normalizer <= self.epsilon
        ):
            raise ValueError(
                "训练残差的asinh归一化因子无效。"
            )

        self.residual_scale = quantile_value

        self.training_abs_residual_quantile = (
            quantile_value
        )

        self.asinh_normalizer = normalizer

        return self

    def transform(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        将完整归一化光谱转换为
        DDPM学习的残差域。
        """

        self._check_fitted()

        values = self._validate_array(
            spectra,
            "spectra",
        )

        self._check_length(
            values
        )

        residuals = (
            values
            - self.prior[np.newaxis, :]
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

        else:
            scaled = (
                self.target_abs_max
                * np.arcsinh(
                    residuals.astype(
                        np.float64,
                        copy=False,
                    )
                    / float(
                        self.residual_scale
                    )
                )
                / float(
                    self.asinh_normalizer
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
    ) -> np.ndarray:
        """
        取消残差范围变换并加回先验。
        """

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
                values
                * float(
                    self.residual_scale
                )
            )

        else:
            sinh_argument = (
                values.astype(
                    np.float64,
                    copy=False,
                )
                / self.target_abs_max
                * float(
                    self.asinh_normalizer
                )
            )

            residuals = (
                float(
                    self.residual_scale
                )
                * np.sinh(
                    sinh_argument
                )
            )

        restored = (
            self.prior[np.newaxis, :]
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

    def prior_batch(
        self,
        number_of_spectra: int,
    ) -> np.ndarray:
        """
        返回多份训练集先验。
        """

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

    def state_dict(
        self,
    ) -> dict[str, Any]:
        """
        生成可以保存到checkpoint metadata中的状态。
        """

        self._check_fitted()

        residual_state: dict[str, Any] = {
            "method": (
                self.normalization_method
            ),
            "scale": float(
                self.residual_scale
            ),
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
            == ROBUST_ASINH
        ):
            residual_state.update(
                {
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

        return {
            "schema_version": 2,
            "enabled": True,
            "domain": (
                "spectrum_global_minmax_normalized"
            ),
            "prior_method": (
                "training_pointwise_median"
            ),
            "prior_normalized_intensity": (
                self.prior.tolist()
            ),
            "residual_normalization": (
                residual_state
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "PriorResidualTransformer":
        """
        从v1或v2 checkpoint metadata恢复变换器。
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
        }:
            raise ValueError(
                "不支持的prior_residual_state版本。"
            )

        if (
            state.get(
                "domain"
            )
            != "spectrum_global_minmax_normalized"
        ):
            raise ValueError(
                "检查点中的先验残差数据域无效。"
            )

        if (
            state.get(
                "prior_method"
            )
            != "training_pointwise_median"
        ):
            raise ValueError(
                "检查点中的先验方法不是"
                "training_pointwise_median。"
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
            schema_version == 1
            and method != GLOBAL_MAXABS
        ):
            raise ValueError(
                "v1检查点只支持global_maxabs残差缩放。"
            )

        transformer = cls(
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
            epsilon=float(
                residual_state.get(
                    "epsilon",
                    1.0e-8,
                )
            ),
        )

        prior = np.asarray(
            state[
                "prior_normalized_intensity"
            ],
            dtype=np.float32,
        ).reshape(-1)

        if prior.size < 2:
            raise ValueError(
                "检查点中的先验光谱长度无效。"
            )

        if not np.isfinite(
            prior
        ).all():
            raise ValueError(
                "检查点中的先验光谱包含NaN或无穷值。"
            )

        scale = float(
            residual_state[
                "scale"
            ]
        )

        maximum = float(
            residual_state[
                "training_max_abs_residual"
            ]
        )

        if (
            not np.isfinite(scale)
            or scale <= transformer.epsilon
        ):
            raise ValueError(
                "检查点中的残差scale无效。"
            )

        if (
            not np.isfinite(maximum)
            or maximum <= transformer.epsilon
        ):
            raise ValueError(
                "检查点中的训练残差最大值无效。"
            )

        if method == GLOBAL_MAXABS:
            expected_scale = (
                maximum
                / transformer.target_abs_max
            )

            if not np.isclose(
                scale,
                expected_scale,
                rtol=1.0e-5,
                atol=transformer.epsilon,
            ):
                raise ValueError(
                    "检查点中的残差scale与"
                    "training_max_abs_residual/"
                    "target_abs_max不一致。"
                )

            quantile_value = None
            asinh_normalizer = None

        else:
            quantile_value = float(
                residual_state[
                    "training_abs_residual_quantile"
                ]
            )

            asinh_normalizer = float(
                residual_state[
                    "asinh_normalizer"
                ]
            )

            if (
                not np.isfinite(
                    quantile_value
                )
                or quantile_value
                <= transformer.epsilon
            ):
                raise ValueError(
                    "检查点中的残差分位数尺度无效。"
                )

            if not np.isclose(
                scale,
                quantile_value,
                rtol=1.0e-5,
                atol=transformer.epsilon,
            ):
                raise ValueError(
                    "检查点中的scale与分位数尺度不一致。"
                )

            expected_normalizer = float(
                np.arcsinh(
                    maximum
                    / scale
                )
            )

            if (
                not np.isfinite(
                    asinh_normalizer
                )
                or not np.isclose(
                    asinh_normalizer,
                    expected_normalizer,
                    rtol=1.0e-5,
                    atol=transformer.epsilon,
                )
            ):
                raise ValueError(
                    "检查点中的asinh_normalizer无效。"
                )

        raw_percentiles = residual_state.get(
            "training_abs_residual_percentiles",
            {},
        )

        if not isinstance(
            raw_percentiles,
            dict,
        ):
            raise ValueError(
                "检查点中的残差分位数统计必须是字典。"
            )

        percentiles = {
            str(key): float(value)
            for key, value in (
                raw_percentiles.items()
            )
        }

        if not all(
            np.isfinite(value)
            and value >= 0.0
            for value in percentiles.values()
        ):
            raise ValueError(
                "检查点中的残差分位数统计无效。"
            )

        transformer.prior = (
            prior.copy()
        )

        transformer.residual_scale = (
            scale
        )

        transformer.training_max_abs_residual = (
            maximum
        )

        transformer.training_abs_residual_quantile = (
            quantile_value
        )

        transformer.asinh_normalizer = (
            asinh_normalizer
        )

        transformer.training_abs_residual_percentiles = (
            percentiles
        )

        return transformer

    @staticmethod
    def _percentile_key(
        level: float,
    ) -> str:
        text = (
            f"{float(level):g}"
            .replace(
                ".",
                "_",
            )
        )

        return f"p{text}"

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
                f"{name}必须是二维数组[N, L]，"
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

    def _check_fitted(
        self,
    ) -> None:
        base_fitted = (
            self.prior is not None
            and self.residual_scale is not None
            and self.training_max_abs_residual
            is not None
        )

        robust_fitted = (
            self.normalization_method
            != ROBUST_ASINH
            or (
                self.training_abs_residual_quantile
                is not None
                and self.asinh_normalizer
                is not None
            )
        )

        if (
            not base_fitted
            or not robust_fitted
        ):
            raise RuntimeError(
                "先验残差变换器尚未使用"
                "训练集fit()。"
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
                "输入光谱长度与先验长度不一致："
                f"输入为{values.shape[1]}，"
                f"先验为{self.prior.size}。"
            )
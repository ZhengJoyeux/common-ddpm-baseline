import numpy as np
import torch
from torch import nn

from src.broad_local_residual import (
    BroadLocalResidualDecomposer,
)
from src.prior_residual import (
    PriorResidualTransformer,
)
from src.spectrum_generator import (
    generate_spectra,
)
from src.spectrum_length_adapter import (
    SpectrumLengthAdapter,
)


class _FixedDiffusion(nn.Module):
    def __init__(
        self,
        samples: np.ndarray,
    ) -> None:
        super().__init__()

        tensor = torch.as_tensor(
            samples,
            dtype=torch.float32,
        )

        self.register_buffer(
            "samples",
            tensor[:, None, :],
        )

        self.offset = 0

    def sample(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        start = self.offset
        end = start + int(
            batch_size
        )

        if end > self.samples.shape[0]:
            raise RuntimeError(
                "测试样本不足。"
            )

        result = self.samples[
            start:end
        ]

        self.offset = end

        return result


def _synthetic_residuals():
    axis = np.arange(
        600.0,
        1001.0,
        1.0,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        2026
    )

    rows = []

    for index in range(16):
        phase = (
            2.0
            * np.pi
            * index
            / 16.0
        )

        broad = (
            0.055
            * np.sin(
                (axis - 600.0)
                / 90.0
                + phase
            )
            + 0.025
            * np.cos(
                (axis - 600.0)
                / 45.0
                - 0.5 * phase
            )
        )

        peak = (
            (
                0.040
                + rng.normal(
                    0.0,
                    0.004,
                )
            )
            * np.exp(
                -0.5
                * (
                    (
                        axis
                        - (
                            760.0
                            + rng.normal(
                                0.0,
                                0.7,
                            )
                        )
                    )
                    / 4.0
                )
                ** 2
            )
        )

        fine = rng.normal(
            0.0,
            0.004,
            size=axis.size,
        )

        rows.append(
            broad
            + peak
            + fine
        )

    return (
        axis,
        np.stack(
            rows,
            axis=0,
        ).astype(
            np.float32
        ),
    )


def _configuration():
    return {
        "enabled": True,
        "broad_filter": {
            "method": "gaussian",
            "sigma_cm1": 7.0,
            "truncate": 4.0,
        },
        "broad_prior": {
            "method": "pca",
            "pca_explained_variance_ratio": 0.95,
            "pca_max_components": 6,
            "sampling_strategy": (
                "independent_truncated_gaussian_scores"
            ),
            "score_clip_standard_deviations": 2.5,
        },
        "local_normalization": {
            "method": "robust_asinh",
            "residual_quantile": 99.5,
            "target_abs_max": 1.0,
        },
        "epsilon": 1.0e-8,
    }


def test_split_reconstructs_original_residual():
    axis, residuals = (
        _synthetic_residuals()
    )

    decomposer = (
        BroadLocalResidualDecomposer
        .from_configuration(
            _configuration()
        )
        .fit(
            residuals,
            raman_shift=axis,
        )
    )

    broad, local = (
        decomposer.split_raw_residuals(
            residuals
        )
    )

    np.testing.assert_allclose(
        broad + local,
        residuals,
        rtol=0.0,
        atol=2.0e-7,
    )


def test_local_transform_round_trip():
    axis, residuals = (
        _synthetic_residuals()
    )

    decomposer = (
        BroadLocalResidualDecomposer
        .from_configuration(
            _configuration()
        )
        .fit(
            residuals,
            raman_shift=axis,
        )
    )

    _, expected_local = (
        decomposer.split_raw_residuals(
            residuals
        )
    )

    scaled = (
        decomposer.transform_raw_residuals(
            residuals
        )
    )

    restored_local = (
        decomposer.inverse_local_transform(
            scaled
        )
    )

    np.testing.assert_allclose(
        restored_local,
        expected_local,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_state_round_trip_and_seeded_broad_sampling():
    axis, residuals = (
        _synthetic_residuals()
    )

    fitted = (
        BroadLocalResidualDecomposer
        .from_configuration(
            _configuration()
        )
        .fit(
            residuals,
            raman_shift=axis,
        )
    )

    restored = (
        BroadLocalResidualDecomposer
        .from_state_dict(
            fitted.state_dict()
        )
    )

    first = (
        fitted.sample_broad_residuals(
            32,
            random_generator=(
                np.random.default_rng(
                    9157
                )
            ),
        )
    )

    second = (
        restored.sample_broad_residuals(
            32,
            random_generator=(
                np.random.default_rng(
                    9157
                )
            ),
        )
    )

    np.testing.assert_allclose(
        first,
        second,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    assert (
        restored.broad_pca_components.shape[0]
        <= 6
    )

    assert (
        0.0
        < np.sum(
            restored.broad_explained_variance_ratio_
        )
        <= 1.0
    )


def test_sampled_broad_is_smoother_than_local_training_residual():
    axis, residuals = (
        _synthetic_residuals()
    )

    decomposer = (
        BroadLocalResidualDecomposer
        .from_configuration(
            _configuration()
        )
        .fit(
            residuals,
            raman_shift=axis,
        )
    )

    _, local = (
        decomposer.split_raw_residuals(
            residuals
        )
    )

    broad = (
        decomposer.sample_broad_residuals(
            256,
            random_generator=(
                np.random.default_rng(
                    2026
                )
            ),
        )
    )

    broad_first_difference = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.diff(
                        broad,
                        axis=1,
                    )
                )
            )
        )
    )

    local_first_difference = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.diff(
                        local,
                        axis=1,
                    )
                )
            )
        )
    )

    assert (
        broad_first_difference
        < local_first_difference
    )


def test_generate_spectra_reconstructs_outer_prior_plus_broad_plus_local():
    axis = np.arange(
        32,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        20260816
    )

    training = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(16, axis.size),
    ).astype(np.float32)

    outer = (
        PriorResidualTransformer(
            prior_method="pca_reconstruction",
            normalization_method="robust_asinh",
            residual_quantile=99.5,
            target_abs_max=1.0,
            pca_explained_variance_ratio=0.95,
            pca_max_components=6,
            pca_sampling_strategy=(
                "independent_truncated_gaussian_scores"
            ),
            pca_score_clip_standard_deviations=2.5,
        )
        .fit(training)
    )

    references = (
        outer.reference_priors_for_spectra(
            training
        )
    )

    raw_residuals = (
        training
        - references
    )

    decomposer = (
        BroadLocalResidualDecomposer
        .from_configuration(
            {
                **_configuration(),
                "broad_filter": {
                    "method": "gaussian",
                    "sigma_cm1": 3.0,
                    "truncate": 4.0,
                },
            }
        )
        .fit(
            raw_residuals,
            raman_shift=axis,
        )
    )

    adapter = SpectrumLengthAdapter.create(
        dimension_multipliers=[
            1,
            2,
            4,
        ],
        raman_shifts=[
            axis
        ],
        model_length="auto",
        padding_mode=(
            "right_zero_padding"
        ),
        padding_value=0.0,
        raman_range_tolerance=1.0,
    )

    number = 7

    zero_local_scaled = np.zeros(
        (
            number,
            adapter.padded_length,
        ),
        dtype=np.float32,
    )

    prior_rng = np.random.default_rng(
        2026
    )

    broad_rng = np.random.default_rng(
        2026 + 1_000_003
    )

    expected = (
        outer.sample_reference_priors(
            number,
            random_generator=prior_rng,
        )
        + decomposer.sample_broad_residuals(
            number,
            random_generator=broad_rng,
        )
    )

    generated = generate_spectra(
        diffusion=_FixedDiffusion(
            zero_local_scaled
        ),
        number_of_spectra=number,
        generation_batch_size=3,
        device=torch.device(
            "cpu"
        ),
        length_adapter=adapter,
        output_raman_shifts=axis,
        prior_residual_transformer=outer,
        broad_local_residual_decomposer=(
            decomposer
        ),
        prior_random_seed=2026,
    )

    np.testing.assert_allclose(
        generated,
        expected,
        rtol=2.0e-6,
        atol=2.0e-6,
    )

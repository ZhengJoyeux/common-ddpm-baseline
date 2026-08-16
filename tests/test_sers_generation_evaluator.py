import numpy as np

from src.sers_generation_evaluator import (
    broad_local_distribution_evaluation,
    build_nonpeak_mask,
    diversity_summary,
    nearest_reference_metrics,
    pca_distribution_evaluation,
    row_pearson,
    training_nearest_other_metrics,
)


def _synthetic_sers(
    *,
    number: int,
    seed: int,
    variation_scale: float = 1.0,
):
    axis = np.linspace(
        600.0,
        900.0,
        301,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(number):
        spectrum = np.zeros_like(axis)

        for center, amplitude, width in (
            (700.0, 100.0, 5.0),
            (800.0, 70.0, 7.0),
        ):
            shifted_center = (
                center
                + rng.normal(
                    0.0,
                    0.7 * variation_scale,
                )
            )
            shifted_amplitude = (
                amplitude
                * (
                    1.0
                    + rng.normal(
                        0.0,
                        0.04 * variation_scale,
                    )
                )
            )
            shifted_width = (
                width
                * (
                    1.0
                    + rng.normal(
                        0.0,
                        0.03 * variation_scale,
                    )
                )
            )

            spectrum += (
                shifted_amplitude
                * np.exp(
                    -0.5
                    * (
                        (
                            axis
                            - shifted_center
                        )
                        / shifted_width
                    )
                    ** 2
                )
            )

        spectrum += rng.normal(
            0.0,
            0.5,
            size=axis.size,
        )

        rows.append(spectrum)

    return axis, np.stack(rows, axis=0)


def test_nearest_metrics_are_finite():
    _, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=32,
        seed=2027,
    )

    generated_nearest = (
        nearest_reference_metrics(
            query_spectra=generated,
            reference_spectra=training,
        )
    )

    training_nearest = (
        training_nearest_other_metrics(
            training
        )
    )

    assert generated_nearest.shape[0] == 32
    assert training_nearest.shape[0] == 16
    assert np.isfinite(
        generated_nearest[
            "standardized_shape_rmse"
        ].to_numpy()
    ).all()
    assert np.isfinite(
        generated_nearest[
            "pearson"
        ].to_numpy()
    ).all()


def test_identical_spectra_have_unit_pearson():
    _, spectra = _synthetic_sers(
        number=4,
        seed=2026,
    )

    result = row_pearson(
        spectra,
        spectra,
    )

    assert np.allclose(
        result,
        1.0,
        atol=1.0e-10,
    )


def test_pca_distribution_evaluation_runs():
    _, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=40,
        seed=2027,
    )
    _, heldout = _synthetic_sers(
        number=4,
        seed=2028,
    )

    (
        model,
        pca_summary,
        pca_scores,
        distribution_summary,
        reconstruction_summary,
    ) = pca_distribution_evaluation(
        training_spectra=training,
        generated_spectra=generated,
        heldout_spectra=heldout,
        pca_components=6,
        bootstrap_repeats=10,
        random_seed=2026,
    )

    assert model.components.shape[0] == 6
    assert pca_summary.shape[0] == 6
    assert set(
        pca_scores["dataset"].unique()
    ) == {
        "training",
        "generated",
        "heldout",
    }
    assert distribution_summary.shape[0] == 3
    assert set(
        reconstruction_summary["dataset"]
    ) == {
        "training",
        "generated",
        "heldout",
    }


def test_broad_local_matched_bootstrap_runs():
    axis, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=40,
        seed=2027,
    )

    mask = build_nonpeak_mask(
        raman_shift=axis,
        peak_centers_cm1=[
            700.0,
            800.0,
        ],
        half_width_cm1=15.0,
    )

    full_summary, bootstrap_summary = (
        broad_local_distribution_evaluation(
            training_spectra=training,
            generated_spectra=generated,
            raman_shift=axis,
            nonpeak_mask=mask,
            broad_sigma_cm1=7.0,
            bootstrap_repeats=10,
            random_seed=2026,
        )
    )

    assert full_summary.shape[0] == 9
    assert bootstrap_summary.shape[0] == 9
    assert np.isfinite(
        bootstrap_summary[
            "ratio_median"
        ].to_numpy()
    ).all()


def test_diversity_summary_uses_requested_pairs():
    _, training = _synthetic_sers(
        number=16,
        seed=2026,
    )
    _, generated = _synthetic_sers(
        number=32,
        seed=2027,
    )

    (
        summary,
        training_pairs,
        generated_pairs,
    ) = diversity_summary(
        training_spectra=training,
        generated_spectra=generated,
        generated_pair_count=500,
        random_seed=2026,
    )

    assert training_pairs.shape[0] == 120
    assert generated_pairs.shape[0] == 500
    assert summary.shape[0] == 4

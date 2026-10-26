"""Tests for mvn_math. These must pass before any scene is rendered."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Tests run from the project root or from tests/; make the root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import mvn_math as mm  # noqa: E402

ALL_SIGMAS = (
    [config.MAIN_SIGMA]
    + [sigma for _, sigma in config.DEMO_STRUCTURES]
    + [sigma for _, sigma in config.SIGMA_WALK]
)


def random_spd(size: int, rng: np.random.Generator) -> np.ndarray:
    """A random symmetric positive definite matrix: B B^T plus a ridge."""
    base = rng.standard_normal((size, size))
    return base @ base.T + size * np.eye(size)


@pytest.mark.parametrize("sigma", ALL_SIGMAS)
def test_factor_reproduces_sigma(sigma: list[list[float]]) -> None:
    chol = mm.cholesky_factor(sigma)
    np.testing.assert_allclose(chol @ chol.T, sigma, atol=1e-12)


@pytest.mark.parametrize("size", [2, 3, 3, 4, 4, 6])
def test_manual_matches_numpy(size: int) -> None:
    rng = np.random.default_rng(size * 7)
    sigma = random_spd(size, rng)
    np.testing.assert_allclose(mm.cholesky_manual(sigma), np.linalg.cholesky(sigma), atol=1e-10)


def test_main_factor_is_exact() -> None:
    np.testing.assert_array_equal(mm.cholesky_factor(config.MAIN_SIGMA), [[2.0, 0.0], [1.0, 2.0]])


def test_2x2_steps_match_factor() -> None:
    steps = mm.cholesky_2x2_steps(config.MAIN_SIGMA)
    assert (steps.l11, steps.l21, steps.l22) == (2.0, 1.0, 2.0)
    np.testing.assert_array_equal(steps.factor, mm.cholesky_factor(config.MAIN_SIGMA))


def test_worked_point() -> None:
    chol = mm.cholesky_factor(config.MAIN_SIGMA)
    np.testing.assert_allclose(mm.transform(config.WORKED_Z, config.MAIN_MU, chol), [2.0, 0.5])


def test_worked_point_mahalanobis_equals_norm_of_z() -> None:
    chol = mm.cholesky_factor(config.MAIN_SIGMA)
    x = mm.transform(config.WORKED_Z, config.MAIN_MU, chol)
    quad = mm.mahalanobis_sq(x, config.MAIN_MU, config.MAIN_SIGMA)
    assert quad == pytest.approx(np.dot(config.WORKED_Z, config.WORKED_Z))
    assert quad == pytest.approx(1.25)


def test_worked_point_density() -> None:
    x = mm.transform(config.WORKED_Z, config.MAIN_MU, mm.cholesky_factor(config.MAIN_SIGMA))
    assert mm.mvn_density(x, config.MAIN_MU, config.MAIN_SIGMA) == pytest.approx(0.0213, abs=5e-5)


def test_elementwise_sqrt_attempt() -> None:
    np.testing.assert_allclose(
        mm.elementwise_sqrt_covariance(config.MAIN_SIGMA), [[6.0, 5.99], [5.99, 7.0]], atol=0.01
    )


def test_sigma_squared() -> None:
    np.testing.assert_allclose(
        mm.covariance_after_matrix(config.MAIN_SIGMA), [[20.0, 18.0], [18.0, 29.0]]
    )


def test_sample_covariance_close_to_sigma() -> None:
    z = mm.standard_normal_samples(config.N_STATS, 2, config.RANDOM_SEED)
    x = mm.transform(z, config.MAIN_MU, mm.cholesky_factor(config.MAIN_SIGMA))
    np.testing.assert_allclose(mm.sample_cov(x), config.MAIN_SIGMA, atol=0.15)
    np.testing.assert_allclose(mm.sample_mean(x), config.MAIN_MU, atol=0.1)


def test_display_samples_are_prefix_of_stats_samples() -> None:
    small = mm.standard_normal_samples(config.N_DISPLAY, 2, config.RANDOM_SEED)
    large = mm.standard_normal_samples(config.N_STATS, 2, config.RANDOM_SEED)
    np.testing.assert_array_equal(small, large[: config.N_DISPLAY])


def test_cholesky_raises_at_rho_one() -> None:
    with pytest.raises(mm.NotPositiveDefiniteError):
        mm.cholesky_factor([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(mm.NotPositiveDefiniteError):
        mm.cholesky_manual([[1.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.5, 0.99])
def test_mixing_factor_is_cholesky_of_correlation(rho: float) -> None:
    np.testing.assert_allclose(
        mm.mixing_factor(rho), mm.cholesky_factor(mm.correlation_matrix(rho)), atol=1e-12
    )


def test_mixing_factor_finite_at_rho_one() -> None:
    np.testing.assert_array_equal(mm.mixing_factor(1.0), [[1.0, 0.0], [1.0, 0.0]])


def test_eigen_axes_main() -> None:
    values, vectors = mm.eigen_axes(config.MAIN_SIGMA)
    np.testing.assert_allclose(values, [6.56, 2.44], atol=0.01)
    np.testing.assert_allclose(vectors @ np.diag(values) @ vectors.T, config.MAIN_SIGMA)


def test_ellipse_params_axis_aligned() -> None:
    width, height, angle = mm.covariance_ellipse_params([[4.0, 0.0], [0.0, 1.0]], k=2.0)
    assert (width, height) == pytest.approx((8.0, 4.0))
    assert angle == pytest.approx(0.0)


def test_mvn_density_matches_1d_in_one_dimension() -> None:
    assert mm.mvn_density([1.0], [0.0], [[4.0]]) == pytest.approx(
        float(mm.normal_pdf_1d(1.0, 0.0, 2.0))
    )


def test_axis_limits_contain_clouds() -> None:
    x_lim, y_lim = mm.axis_limits(
        [config.MAIN_MU], [config.MAIN_SIGMA], config.AXIS_N_SD, config.AXIS_ROUND_STEP
    )
    sds = np.sqrt(np.diag(config.MAIN_SIGMA))
    assert x_lim[0] <= config.MAIN_MU[0] - config.AXIS_N_SD * sds[0]
    assert y_lim[1] >= config.MAIN_MU[1] + config.AXIS_N_SD * sds[1]


def test_3d_bonus_sigma_is_valid() -> None:
    mm.cholesky_factor(config.THREE_D_SIGMA)


def test_speaker_notes_file_is_up_to_date() -> None:
    """notes/speaker_notes.md must match the notes in the scene files (run `make notes`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notes"))
    import build_notes

    assert build_notes.OUTPUT.read_text() == build_notes.build()
    assert "(no notes)" not in build_notes.build()


def test_no_em_dashes_in_slides_or_notes() -> None:
    root = Path(__file__).resolve().parent.parent
    for path in [*root.glob("scenes/*.py"), root / "notes" / "speaker_notes.md"]:
        assert "—" not in path.read_text(), path.name

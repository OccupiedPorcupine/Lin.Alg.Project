"""Tests for mvn_math. These must pass before any scene is rendered."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Tests run from the project root or from tests/; make the root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import mvn_math as mm  # noqa: E402

ALL_SIGMAS = (
    [config.MAIN_SIGMA, config.EXAMPLE_3X3_SIGMA]
    + [sigma for _, sigma in config.DEMO_STRUCTURES]
    + [config.EIGEN_EXAMPLE_SIGMA, config.EIGEN_EXAMPLE_SIGMA_2]
)


def random_spd(size: int, rng: np.random.Generator) -> np.ndarray:
    """A random symmetric positive definite matrix: B B^T plus a ridge."""
    base = rng.standard_normal((size, size))
    return base @ base.T + size * np.eye(size)


# --- Cholesky ------------------------------------------------------------------


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


@pytest.mark.parametrize(
    "sigma", [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.2], [1.2, 1.0]]], ids=["rho=1", "rho=1.2"]
)
def test_cholesky_raises_when_not_positive_definite(sigma: list[list[float]]) -> None:
    for factor in (mm.cholesky_factor, mm.cholesky_manual):
        with pytest.raises(mm.NotPositiveDefiniteError):
            factor(sigma)


# --- The 2.3 hand route ----------------------------------------------------------


def test_ldl_steps_3x3() -> None:
    route = mm.ldl_steps(config.EXAMPLE_3X3_SIGMA)
    assert [s.multiplier for s in route.steps] == [0.5, 0.5, 0.5]
    assert [(s.target_row, s.source_row) for s in route.steps] == [(1, 0), (2, 0), (2, 1)]
    np.testing.assert_array_equal(route.steps[0].result_row, [0, 4, 2])
    np.testing.assert_array_equal(route.steps[1].result_row, [0, 2, 2])
    np.testing.assert_array_equal(route.steps[2].result_row, [0, 0, 1])
    np.testing.assert_array_equal(route.lower, [[1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]])
    np.testing.assert_array_equal(route.upper, [[4, 2, 2], [0, 4, 2], [0, 0, 1]])
    np.testing.assert_array_equal(route.diag, np.diag([4.0, 4.0, 1.0]))
    np.testing.assert_array_equal(route.v, route.lower.T)
    np.testing.assert_array_equal(route.chol, [[2, 0, 0], [1, 2, 0], [1, 1, 1]])
    np.testing.assert_allclose(route.chol, np.linalg.cholesky(config.EXAMPLE_3X3_SIGMA))
    np.testing.assert_allclose(route.lower @ route.upper, config.EXAMPLE_3X3_SIGMA)


def test_main_factor_is_corner_of_3x3_factor() -> None:
    """Slide 2.4 takes the 2x2 G straight from the corner of the 3x3 G worked out on 2.3."""
    corner = np.asarray(config.EXAMPLE_3X3_SIGMA)[:2, :2]
    np.testing.assert_array_equal(corner, config.MAIN_SIGMA)
    chol_3x3 = mm.ldl_steps(config.EXAMPLE_3X3_SIGMA).chol
    np.testing.assert_allclose(chol_3x3[:2, :2], mm.cholesky_factor(config.MAIN_SIGMA))


# --- Sampling -------------------------------------------------------------------


def test_worked_point() -> None:
    chol = mm.cholesky_factor(config.MAIN_SIGMA)
    np.testing.assert_allclose(mm.transform(config.WORKED_Z, config.MAIN_MU, chol), [2.0, 0.5])


def test_worked_point_mahalanobis_and_density() -> None:
    chol = mm.cholesky_factor(config.MAIN_SIGMA)
    x = mm.transform(config.WORKED_Z, config.MAIN_MU, chol)
    quad = mm.mahalanobis_sq(x, config.MAIN_MU, config.MAIN_SIGMA)
    assert quad == pytest.approx(np.dot(config.WORKED_Z, config.WORKED_Z))
    assert quad == pytest.approx(1.25)
    assert mm.mvn_density(x, config.MAIN_MU, config.MAIN_SIGMA) == pytest.approx(0.0213, abs=5e-5)


def test_formula_points_along_eigenvectors() -> None:
    """Slide 1.4: same ruler distance, different Sigma distance."""
    _, vectors = mm.eigen_axes(config.MAIN_SIGMA)
    mu = np.asarray(config.MAIN_MU)
    distance = config.FORMULA_POINT_DISTANCE
    long_axis = mm.mahalanobis_sq(mu + distance * vectors[:, 0], mu, config.MAIN_SIGMA)
    short_axis = mm.mahalanobis_sq(mu + distance * vectors[:, 1], mu, config.MAIN_SIGMA)
    assert long_axis == pytest.approx(0.61, abs=0.005)
    assert short_axis == pytest.approx(1.64, abs=0.005)


@pytest.mark.parametrize("sigma", [config.MAIN_SIGMA, config.EXAMPLE_3X3_SIGMA])
def test_sample_covariance_close_to_sigma(sigma: list[list[float]]) -> None:
    dim = len(sigma)
    z = mm.standard_normal_samples(config.N_STATS, dim, config.RANDOM_SEED)
    x = mm.transform(z, np.zeros(dim), mm.cholesky_factor(sigma))
    np.testing.assert_allclose(mm.sample_cov(x), sigma, atol=0.15)


def test_standard_normal_correlation_rounds_to_zero() -> None:
    z = mm.standard_normal_samples(config.N_STATS, 2, config.RANDOM_SEED)
    assert abs(mm.sample_corr(z)[0, 1]) < 0.005


def test_display_samples_are_prefix_of_stats_samples() -> None:
    small = mm.standard_normal_samples(config.N_DISPLAY, 2, config.RANDOM_SEED)
    large = mm.standard_normal_samples(config.N_STATS, 2, config.RANDOM_SEED)
    np.testing.assert_array_equal(small, large[: config.N_DISPLAY])


# --- Failed square roots (1.7 and 2.1) ---------------------------------------------


def test_sigma_squared() -> None:
    np.testing.assert_allclose(
        mm.covariance_of_linear_map(config.MAIN_SIGMA), [[20.0, 18.0], [18.0, 29.0]]
    )


def test_elementwise_sqrt_attempt() -> None:
    off = 2 * math.sqrt(2) + math.sqrt(10)  # the symbolic value typed on slide 2.1
    assert off == pytest.approx(5.9907, abs=1e-4)
    np.testing.assert_allclose(
        mm.covariance_of_linear_map(mm.elementwise_sqrt(config.MAIN_SIGMA)),
        [[6.0, off], [off, 7.0]],
    )


# --- Correlation pivot (3.3) ---------------------------------------------------------


def test_unit_corr_pivot() -> None:
    assert mm.unit_corr_pivot(0.999) == pytest.approx(0.002, abs=1e-4)
    assert mm.unit_corr_factor(0.999)[1, 1] == pytest.approx(0.0447, abs=1e-4)
    assert mm.unit_corr_pivot(1.0) == 0.0
    assert mm.unit_corr_pivot(1.2) == pytest.approx(-0.44)


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.5, 0.99])
def test_unit_corr_factor_is_cholesky(rho: float) -> None:
    np.testing.assert_allclose(
        mm.unit_corr_factor(rho), mm.cholesky_factor(mm.correlation_matrix(rho)), atol=1e-12
    )


def test_unit_corr_factor_edge_cases() -> None:
    np.testing.assert_array_equal(mm.unit_corr_factor(1.0), [[1.0, 0.0], [1.0, 0.0]])
    assert mm.unit_corr_factor(1.2) is None


# --- Eigen view (appendix A.2) ----------------------------------------------------------------


def test_eigen_examples() -> None:
    values, _ = mm.eigen_axes(config.EIGEN_EXAMPLE_SIGMA)
    np.testing.assert_allclose(values, [9.0, 1.0])
    values, _ = mm.eigen_axes(config.EIGEN_EXAMPLE_SIGMA_2)
    np.testing.assert_allclose(values, [9.0, 4.0])
    np.testing.assert_array_equal(np.asarray(config.EIGEN_EXAMPLE_SIGMA) @ [1, 0], [5, 4])


def test_ellipse_params_axis_aligned() -> None:
    width, height, angle = mm.covariance_ellipse_params([[4.0, 0.0], [0.0, 1.0]], k=2.0)
    assert (width, height) == pytest.approx((8.0, 4.0))
    assert angle == pytest.approx(0.0)


# --- Applications (1.1) --------------------------------------------------------


def test_portfolio_risk_ratio() -> None:
    def spread(rho: float) -> float:
        returns = mm.portfolio_returns(
            config.PORTFOLIO_WEIGHTS, rho, config.N_STATS, config.RANDOM_SEED
        )
        return float(np.std(returns))

    ratio = spread(config.PORTFOLIO_RHO_CORRELATED) / spread(config.PORTFOLIO_RHO_INDEPENDENT)
    assert ratio == pytest.approx(1.34, abs=0.02)


def test_value_at_risk_is_a_loss() -> None:
    returns = mm.portfolio_returns(config.PORTFOLIO_WEIGHTS, 0.0, config.N_STATS, 1)
    assert mm.value_at_risk(returns, config.VAR_LEVEL) < 0


def test_synth_correlation_is_positive_definite() -> None:
    mm.cholesky_factor(config.SYNTH_CORRELATION)


def test_synth_copy_matches_original() -> None:
    z = mm.standard_normal_samples(config.N_STATS, 3, config.RANDOM_SEED)
    rows = mm.transform(z, np.zeros(3), mm.cholesky_factor(config.SYNTH_CORRELATION))
    np.testing.assert_allclose(mm.sample_corr(rows), config.SYNTH_CORRELATION, atol=0.02)


# --- Misc ----------------------------------------------------------------------------


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


def test_speaker_notes_file_is_up_to_date() -> None:
    """notes/speaker_notes.md must match the notes in the scene files (run `make notes`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notes"))
    import build_notes

    assert build_notes.OUTPUT.read_text() == build_notes.build()
    assert "(no notes)" not in build_notes.build()


def test_no_em_dashes_in_slides_or_notes() -> None:
    root = Path(__file__).resolve().parent.parent
    files = [*root.glob("scenes/*.py"), root / "notes" / "speaker_notes.md", root / "config.py"]
    for path in files:
        assert "—" not in path.read_text(), path.name

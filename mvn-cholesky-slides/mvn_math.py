"""Pure numpy math behind the deck. No Manim imports.

Every number that appears on a slide is computed by a function in this file,
so changing a matrix in ``config.py`` changes the whole deck consistently.

Conventions:
    * Samples are stored as rows: an (n, dim) array where each row is a vector.
    * Because of that, the column-vector formula x = mu + L z becomes
      X = mu + Z @ L.T when applied to every row at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


# --- Sampling -----------------------------------------------------------------


def standard_normal_samples(n: int, dim: int, seed: int) -> Array:
    """Draw n independent vectors from N(0, I) in ``dim`` dimensions.

    This is the only randomness in the deck. Every other cloud is a
    deterministic transform of these samples, which is what makes the
    "same z, different Sigma" comparisons honest.

    The generator fills the array row by row, so the first m rows of a draw of
    size n equal a draw of size m with the same seed. That lets the deck show
    ``N_DISPLAY`` points that are a subset of the ``N_STATS`` used for checks.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim))


def transform(z: ArrayLike, mu: ArrayLike, matrix: ArrayLike) -> Array:
    """Map standard normal samples through x = mu + A z.

    With A = L, the Cholesky factor of Sigma, the result is N(mu, Sigma). Any
    other matrix also works here (slide 1.8 tries a few), giving N(mu, A A^T).
    Works for a single vector of shape (dim,) or many rows of shape (n, dim).
    """
    z = np.asarray(z, dtype=float)
    return np.asarray(mu, dtype=float) + z @ np.asarray(matrix, dtype=float).T


def sample_mean(samples: ArrayLike) -> Array:
    """Average of the rows: the estimate mu-hat of the mean vector."""
    return np.asarray(samples, dtype=float).mean(axis=0)


def sample_cov(samples: ArrayLike) -> Array:
    """Unbiased sample covariance Sigma-hat of the rows."""
    return np.cov(np.asarray(samples, dtype=float), rowvar=False)


# --- Cholesky -----------------------------------------------------------------


class NotPositiveDefiniteError(ValueError):
    """Raised when a covariance matrix has no Cholesky factor."""


def cholesky_factor(sigma: ArrayLike) -> Array:
    """Lower-triangular L with positive diagonal such that L L^T = Sigma.

    Wraps ``np.linalg.cholesky`` so that a bad covariance matrix produces a
    message a human can act on, instead of a bare LinAlgError.
    """
    sigma = np.asarray(sigma, dtype=float)
    if not np.allclose(sigma, sigma.T):
        raise NotPositiveDefiniteError(f"Sigma must be symmetric, got:\n{sigma}")
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as err:
        raise NotPositiveDefiniteError(
            "Sigma is not positive definite, so it has no Cholesky factor. "
            "Check that every variance is positive and every |correlation| < 1. "
            f"Got:\n{sigma}"
        ) from err


def cholesky_manual(sigma: ArrayLike) -> Array:
    """Teaching version of Cholesky, entry by entry (the formulas on slide 2.4).

    Column j is filled top to bottom:
        L_jj = sqrt(Sigma_jj - sum_{k<j} L_jk^2)
        L_ij = (Sigma_ij - sum_{k<j} L_ik L_jk) / L_jj      for i > j
    Each column only needs the columns to its left, which is why one pass works.
    """
    sigma = np.asarray(sigma, dtype=float)
    size = sigma.shape[0]
    chol = np.zeros_like(sigma)
    for j in range(size):
        # What is left of the variance after earlier columns took their share.
        remaining = sigma[j, j] - np.dot(chol[j, :j], chol[j, :j])
        if remaining <= 0:
            raise NotPositiveDefiniteError(
                f"Pivot {j} is {remaining:.3g}; Sigma is not positive definite."
            )
        chol[j, j] = math.sqrt(remaining)
        for i in range(j + 1, size):
            chol[i, j] = (sigma[i, j] - np.dot(chol[i, :j], chol[j, :j])) / chol[j, j]
    return chol


@dataclass(frozen=True)
class Cholesky2x2Steps:
    """Every intermediate value of the 2x2 hand calculation on slide 2.4."""

    sigma11: float
    sigma21: float
    sigma22: float
    l11: float  # sqrt(sigma11)
    l21: float  # sigma21 / l11
    l21_sq: float  # l21 squared, subtracted from sigma22
    l22: float  # sqrt(sigma22 - l21^2)

    @property
    def factor(self) -> Array:
        return np.array([[self.l11, 0.0], [self.l21, self.l22]])


def cholesky_2x2_steps(sigma: ArrayLike) -> Cholesky2x2Steps:
    """Cholesky of a 2x2 matrix, keeping each step so a slide can show it."""
    sigma = np.asarray(sigma, dtype=float)
    l11 = math.sqrt(sigma[0, 0])
    l21 = sigma[1, 0] / l11
    l22_sq = sigma[1, 1] - l21**2
    if l22_sq <= 0:
        raise NotPositiveDefiniteError("Sigma is not positive definite.")
    return Cholesky2x2Steps(
        sigma11=float(sigma[0, 0]),
        sigma21=float(sigma[1, 0]),
        sigma22=float(sigma[1, 1]),
        l11=l11,
        l21=l21,
        l21_sq=l21**2,
        l22=math.sqrt(l22_sq),
    )


def mixing_factor(rho: float) -> Array:
    """Closed-form Cholesky factor of [[1, rho], [rho, 1]].

    L = [[1, 0], [rho, sqrt(1 - rho^2)]], so x1 = z1 and
    x2 = rho z1 + sqrt(1 - rho^2) z2.
    The max(0, ...) keeps it finite at |rho| = 1, where the true Cholesky
    factorization fails. That lets the rho sweep animate right up to the
    breaking point without raising.
    """
    return np.array([[1.0, 0.0], [rho, math.sqrt(max(0.0, 1.0 - rho**2))]])


def correlation_matrix(rho: float) -> Array:
    """The unit-variance covariance [[1, rho], [rho, 1]]."""
    return np.array([[1.0, rho], [rho, 1.0]])


def variance_shares(rho: float) -> tuple[float, float]:
    """Split Var(x2) = 1 into the part shared with x1 and fresh noise.

    From x2 = rho z1 + sqrt(1 - rho^2) z2 the shares are rho^2 and 1 - rho^2.
    """
    return rho**2, 1.0 - rho**2


# --- Failed "square roots" from slide 1.8 --------------------------------------


def elementwise_sqrt(sigma: ArrayLike) -> Array:
    """Entry-by-entry square root of Sigma (the naive guess on slide 1.8).

    The sign is kept so a negative covariance gives a negative entry instead of
    NaN. For the main example all entries are positive, so this is plain sqrt.
    """
    sigma = np.asarray(sigma, dtype=float)
    return np.sign(sigma) * np.sqrt(np.abs(sigma))


def elementwise_sqrt_covariance(sigma: ArrayLike) -> Array:
    """Covariance produced by the naive guess: A A^T with A = elementwise sqrt.

    This is what the measured covariance on slide 1.8 converges to, and slide
    2.1 explains why.
    """
    root = elementwise_sqrt(sigma)
    return root @ root.T


def covariance_after_matrix(matrix: ArrayLike) -> Array:
    """Covariance of A z when z ~ N(0, I): exactly A A^T (slide 2.1)."""
    matrix = np.asarray(matrix, dtype=float)
    return matrix @ matrix.T


# --- Geometry of the covariance ellipse ----------------------------------------


def eigen_axes(sigma: ArrayLike) -> tuple[Array, Array]:
    """Eigenvalues and unit eigenvectors of Sigma, largest eigenvalue first.

    Eigenvectors are the columns of the returned matrix. Each is flipped, if
    needed, so its first nonzero component is positive; this keeps arrows on
    the slide pointing in a stable direction when Sigma changes.
    """
    values, vectors = np.linalg.eigh(np.asarray(sigma, dtype=float))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for col in range(vectors.shape[1]):
        lead = vectors[np.flatnonzero(np.abs(vectors[:, col]) > 1e-12)[0], col]
        if lead < 0:
            vectors[:, col] *= -1
    return values, vectors


def covariance_ellipse_params(sigma: ArrayLike, k: float) -> tuple[float, float, float]:
    """Width, height and rotation angle (radians) of the k-sigma ellipse.

    The ellipse {x : (x - mu)^T Sigma^-1 (x - mu) = k^2} has semi-axes
    k sqrt(lambda_i) along the eigenvectors, so the full width and height are
    2 k sqrt(lambda_1) and 2 k sqrt(lambda_2), and the rotation is the angle of
    the first eigenvector.
    """
    values, vectors = eigen_axes(sigma)
    width = 2.0 * k * math.sqrt(values[0])
    height = 2.0 * k * math.sqrt(values[1])
    angle = math.atan2(vectors[1, 0], vectors[0, 0])
    return width, height, angle


# --- Density ------------------------------------------------------------------


def mahalanobis_sq(x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> float:
    """Squared Mahalanobis distance (x - mu)^T Sigma^-1 (x - mu).

    Uses solve instead of an explicit inverse, which is the numerically sound
    way to apply Sigma^-1.
    """
    diff = np.asarray(x, dtype=float) - np.asarray(mu, dtype=float)
    return float(diff @ np.linalg.solve(np.asarray(sigma, dtype=float), diff))


def mvn_density(x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> float:
    """Multivariate normal pdf.

    f(x) = (2 pi)^(-k/2) |Sigma|^(-1/2) exp(-1/2 (x - mu)^T Sigma^-1 (x - mu)).
    """
    sigma = np.asarray(sigma, dtype=float)
    dim = sigma.shape[0]
    norm_const = (2.0 * math.pi) ** (-dim / 2) * np.linalg.det(sigma) ** -0.5
    return float(norm_const * math.exp(-0.5 * mahalanobis_sq(x, mu, sigma)))


def normal_pdf_1d(x: ArrayLike, mu: float, sd: float) -> Array:
    """The 1D normal pdf, used to draw bell curves on slide 1.6."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))


# --- Axis ranges --------------------------------------------------------------


def _round_out(low: float, high: float, step: float) -> tuple[float, float]:
    return math.floor(low / step) * step, math.ceil(high / step) * step


def axis_limits(
    means: list[ArrayLike], sigmas: list[ArrayLike], n_sd: float, step: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """x and y limits that hold every listed distribution out to n_sd std devs.

    Taking the union over several (mu, Sigma) pairs lets one set of axes show,
    for example, both the round z cloud at 0 and the shifted x cloud at mu.
    """
    lows, highs = [], []
    for mu, sigma in zip(means, sigmas, strict=True):
        spread = n_sd * np.sqrt(np.diag(np.asarray(sigma, dtype=float)))
        lows.append(np.asarray(mu, dtype=float)[:2] - spread[:2])
        highs.append(np.asarray(mu, dtype=float)[:2] + spread[:2])
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    x_lim = _round_out(low[0], high[0], step)
    y_lim = _round_out(low[1], high[1], step)
    return x_lim, y_lim


def axis_limits_1d(
    means: list[float], sds: list[float], n_sd: float, step: float
) -> tuple[float, float]:
    """x limits that hold every listed 1D normal out to n_sd std devs."""
    low = min(m - n_sd * s for m, s in zip(means, sds, strict=True))
    high = max(m + n_sd * s for m, s in zip(means, sds, strict=True))
    return _round_out(low, high, step)


def shared_half_width(sigmas: list[ArrayLike], n_sd: float, step: float) -> float:
    """One half-width big enough for every Sigma listed.

    Side-by-side panels use the same half-width (and so the same scale), so a
    wider cloud really does look wider than its neighbour.
    """
    largest_sd = max(math.sqrt(np.max(np.diag(np.asarray(s, dtype=float)))) for s in sigmas)
    return math.ceil(n_sd * largest_sd / step) * step

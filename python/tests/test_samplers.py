"""Seeded statistical tests for the random samplers.

Each test fixes the generator seed, so the tolerances only have to cover the
sampling error at that one seed. They were chosen with a healthy margin over
the observed deviation, so the suite is deterministic and does not flake.

The deterministic (degenerate / short-circuit) branches are checked without
the ``statistical`` marker so they still run under ``-m "not statistical"``.
"""

import numpy as np
import pytest
from scipy import stats

from realtimecoin.samplers import (
    beta_sample,
    binomial_sample,
    dirichlet_sample,
    gamma_sample,
    sample_bivariate_truncated,
    sample_matrix_normal,
    sample_scalar_normal,
    sample_stable_theta,
    trandn,
)

EPS = float(np.finfo(float).eps)


# --------------------------------------------------------------------------
# Degenerate / deterministic branches
# --------------------------------------------------------------------------


def test_gamma_sample_non_positive_shapes_are_zero():
    rng = np.random.default_rng(0)
    g = gamma_sample(rng, np.array([-1.0, 0.0, np.nan, 2.0]))
    assert g[0] == 0.0 and g[1] == 0.0 and g[2] == 0.0
    assert g[3] > 0


def test_gamma_sample_preserves_shape():
    rng = np.random.default_rng(0)
    g = gamma_sample(rng, np.ones((3, 4, 2)))
    assert g.shape == (3, 4, 2)


def test_beta_sample_degenerate_fallback_is_one():
    rng = np.random.default_rng(0)
    b = beta_sample(rng, np.array([0.0, -1.0]), np.array([0.0, 0.0]))
    assert np.array_equal(b, np.array([1.0, 1.0]))


def test_beta_sample_broadcasts():
    rng = np.random.default_rng(0)
    b = beta_sample(rng, 1.0, np.ones(5))
    assert b.shape == (5,)
    assert np.all((b >= 0) & (b <= 1))


def test_dirichlet_sample_all_zero_alpha_puts_mass_on_first_entry():
    rng = np.random.default_rng(0)
    x = dirichlet_sample(rng, np.zeros(4))
    assert np.array_equal(x, np.array([1.0, 0.0, 0.0, 0.0]))


def test_dirichlet_sample_sums_to_one():
    rng = np.random.default_rng(0)
    x = dirichlet_sample(rng, np.array([0.5, 1.0, 2.0, 4.0]))
    assert x.sum() == pytest.approx(1.0)
    assert np.all(x >= 0)


def test_binomial_sample_short_circuits():
    rng = np.random.default_rng(0)
    assert binomial_sample(rng, 0, 0.5) == 0
    assert binomial_sample(rng, 10, 0.0) == 0
    assert binomial_sample(rng, 10, 1.0) == 10
    # None of the above consumed a draw, so the stream is untouched.
    assert rng.random() == np.random.default_rng(0).random()


def test_binomial_sample_consumes_one_uniform_per_trial():
    counted = np.random.default_rng(11)
    binomial_sample(counted, 7, 0.5)
    reference = np.random.default_rng(11)
    reference.random(7)
    assert counted.random() == reference.random()


def test_trandn_respects_the_bounds():
    rng = np.random.default_rng(5)
    lower = np.array([-np.inf, -1.0, 0.7, -5.0, 2.0])
    upper = np.array([1.0, 1.0, 3.0, np.inf, 2.5])
    x = trandn(rng, np.repeat(lower, 200), np.repeat(upper, 200))
    assert np.all(x >= np.repeat(lower, 200))
    assert np.all(x <= np.repeat(upper, 200))


def test_trandn_rejects_mismatched_or_empty_intervals():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="limitLength"):
        trandn(rng, np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="emptyInterval"):
        trandn(rng, np.array([1.0]), np.array([0.0]))


def test_sample_scalar_normal_deterministic_for_non_positive_variance():
    rng = np.random.default_rng(0)
    x = sample_scalar_normal(rng, 0.4, 0.0, (3, 2), 0.0, 1.0)
    assert np.allclose(x, 0.4)


def test_sample_scalar_normal_clamps_strictly_below_the_upper_bound():
    rng = np.random.default_rng(0)
    x = sample_scalar_normal(rng, 5.0, 0.0, (4,), 0.0, 1.0)
    assert np.all(x < 1.0)
    assert np.allclose(x, 1.0 - EPS)


def test_sample_scalar_normal_draws_within_bounds():
    rng = np.random.default_rng(6)
    x = sample_scalar_normal(rng, 0.5, 0.25, (500,), 0.0, 1.0)
    assert x.shape == (500,)
    assert np.all(x >= 0.0) and np.all(x < 1.0)


def test_sample_bivariate_truncated_falls_back_on_non_finite_input():
    rng = np.random.default_rng(0)
    x = sample_bivariate_truncated(rng, np.array([2.0, -3.0]), np.full((2, 2), np.nan))
    assert x[0] == pytest.approx(1.0 - EPS)
    assert x[1] == -3.0


def test_sample_bivariate_truncated_clamps_a_nan_retention_to_zero():
    # MATLAB's min(max(a, 0), 1 - eps) sends nan to 0, so no nan retention can
    # escape into the particle state.
    rng = np.random.default_rng(0)
    x = sample_bivariate_truncated(rng, np.array([np.nan, 2.0]), np.eye(2))
    assert x[0] == 0.0


def test_sample_scalar_normal_clamps_a_nan_mean_to_the_lower_bound():
    rng = np.random.default_rng(0)
    x = sample_scalar_normal(rng, np.full(3, np.nan), 0.0, (3,), 0.0, 1.0)
    assert np.array_equal(x, np.zeros(3))


def test_sample_stable_theta_validates_against_an_explicit_state_dim():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="invalidMean"):
        sample_stable_theta(rng, np.zeros((2, 3)), np.eye(2), np.eye(3), state_dim=3)


def test_beta_sample_shares_one_gamma_draw_for_a_scalar_argument():
    # MATLAB draws Y at bpar's own size, so a scalar bpar means a single
    # shared Y and hence correlated entries with a common denominator.
    a = np.array([2.0, 2.0, 2.0])
    x = beta_sample(np.random.default_rng(12), a, 3.0)
    reference_rng = np.random.default_rng(12)
    gx = gamma_sample(reference_rng, a)
    gy = gamma_sample(reference_rng, np.asarray(3.0))
    assert np.allclose(x, gx / (gx + gy))


def test_sample_bivariate_truncated_keeps_retention_in_unit_interval():
    rng = np.random.default_rng(8)
    mu = np.array([0.9, 0.05])
    covar = np.array([[0.04, 0.01], [0.01, 0.02]])
    draws = np.array([sample_bivariate_truncated(rng, mu, covar) for _ in range(400)])
    assert np.all(draws[:, 0] >= 0.0)
    assert np.all(draws[:, 0] < 1.0)


def test_sample_matrix_normal_rejects_mismatched_covariances():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="invalidRowCov"):
        sample_matrix_normal(rng, np.zeros((2, 3)), np.eye(3), np.eye(3))
    with pytest.raises(ValueError, match="invalidColCov"):
        sample_matrix_normal(rng, np.zeros((2, 3)), np.eye(2), np.eye(2))


def test_sample_matrix_normal_zero_covariance_returns_the_mean():
    rng = np.random.default_rng(0)
    mean = np.arange(6.0).reshape(2, 3)
    # choljitter regularises the singular row covariance with a 1e-12 ridge,
    # so the draw sits within ~1e-6 of the mean rather than exactly on it.
    x = sample_matrix_normal(rng, mean, np.zeros((2, 2)), np.eye(3))
    assert np.allclose(x, mean, atol=1e-4)


def test_sample_stable_theta_rejects_a_badly_shaped_mean():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="invalidMean"):
        sample_stable_theta(rng, np.zeros((2, 2)), np.eye(2), np.eye(2))


def test_sample_stable_theta_projects_an_unstable_mean():
    # Zero covariance means every draw equals the (unstable) mean, so the
    # spectral-scaling fallback must fire.
    rng = np.random.default_rng(0)
    mean = np.array([[3.0, 0.0, 0.1], [0.0, 3.0, 0.2]])
    theta = sample_stable_theta(rng, mean, np.zeros((2, 2)), np.zeros((3, 3)))
    rho = np.max(np.abs(np.linalg.eigvals(theta[:, :2])))
    assert rho < 1.0
    # The drift column is untouched by the projection.
    assert np.allclose(theta[:, 2], mean[:, 2])


# --------------------------------------------------------------------------
# Seeded statistical checks
# --------------------------------------------------------------------------


@pytest.mark.statistical
def test_gamma_sample_moments():
    rng = np.random.default_rng(20)
    shape = 3.0
    n = 200_000
    g = gamma_sample(rng, np.full(n, shape))
    # Gamma(k, 1) has mean k and variance k.
    assert g.mean() == pytest.approx(shape, abs=0.05)
    assert g.var() == pytest.approx(shape, abs=0.10)


@pytest.mark.statistical
def test_gamma_sample_matches_the_reference_distribution():
    rng = np.random.default_rng(21)
    g = gamma_sample(rng, np.full(5000, 2.5))
    ks = stats.kstest(g, stats.gamma(2.5).cdf)
    assert ks.pvalue > 0.01


@pytest.mark.statistical
def test_beta_sample_moments():
    rng = np.random.default_rng(22)
    a, b = 2.0, 5.0
    n = 200_000
    draws = beta_sample(rng, np.full(n, a), np.full(n, b))
    # Beta(a, b) has mean a / (a + b) and variance ab / ((a+b)^2 (a+b+1)).
    assert draws.mean() == pytest.approx(a / (a + b), abs=0.005)
    expected_var = a * b / ((a + b) ** 2 * (a + b + 1))
    assert draws.var() == pytest.approx(expected_var, abs=0.005)


@pytest.mark.statistical
def test_beta_sample_matches_the_reference_distribution():
    rng = np.random.default_rng(23)
    draws = beta_sample(rng, np.full(5000, 2.0), np.full(5000, 5.0))
    ks = stats.kstest(draws, stats.beta(2.0, 5.0).cdf)
    assert ks.pvalue > 0.01


@pytest.mark.statistical
def test_dirichlet_sample_marginal_means():
    rng = np.random.default_rng(24)
    alpha = np.array([1.0, 2.0, 3.0, 4.0])
    draws = np.array([dirichlet_sample(rng, alpha) for _ in range(20_000)])
    # E[x_i] = alpha_i / sum(alpha).
    assert np.allclose(draws.mean(axis=0), alpha / alpha.sum(), atol=0.01)
    assert np.allclose(draws.sum(axis=1), 1.0)


@pytest.mark.statistical
def test_binomial_sample_mean():
    rng = np.random.default_rng(25)
    trials, prob = 20, 0.3
    draws = np.array([binomial_sample(rng, trials, prob) for _ in range(20_000)])
    # E[n] = trials * prob, Var[n] = trials * prob * (1 - prob).
    assert draws.mean() == pytest.approx(trials * prob, abs=0.05)
    assert draws.var() == pytest.approx(trials * prob * (1 - prob), abs=0.15)
    assert np.all((draws >= 0) & (draws <= trials))


@pytest.mark.statistical
@pytest.mark.parametrize(
    "lower, upper",
    [
        (-np.inf, np.inf),   # untruncated (central sampler, wide interval)
        (-1.0, 1.0),         # narrow central interval (inverse transform)
        (1.0, np.inf),       # right tail (Rayleigh proposal)
        (-np.inf, -1.5),     # left tail (mirrored Rayleigh proposal)
        (2.0, 2.4),          # narrow far tail
    ],
)
def test_trandn_matches_the_truncated_normal(lower, upper):
    rng = np.random.default_rng(26)
    n = 5000
    x = trandn(rng, np.full(n, lower), np.full(n, upper))
    reference = stats.truncnorm(lower, upper)
    ks = stats.kstest(x, reference.cdf)
    assert ks.pvalue > 0.01


@pytest.mark.statistical
def test_sample_scalar_normal_matches_the_truncated_normal():
    rng = np.random.default_rng(27)
    mu, variance = 0.6, 0.09
    sigma = np.sqrt(variance)
    x = sample_scalar_normal(rng, mu, variance, (5000,), 0.0, 1.0)
    reference = stats.truncnorm((0.0 - mu) / sigma, (1.0 - mu) / sigma, mu, sigma)
    ks = stats.kstest(x, reference.cdf)
    assert ks.pvalue > 0.01


@pytest.mark.statistical
def test_sample_matrix_normal_moments():
    rng = np.random.default_rng(28)
    mean = np.array([[1.0, -2.0], [0.5, 3.0]])
    row_cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    col_cov = np.array([[0.5, -0.2], [-0.2, 1.5]])
    n = 40_000

    draws = np.array(
        [sample_matrix_normal(rng, mean, row_cov, col_cov) for _ in range(n)]
    )                                            # (n, 2, 2)
    assert np.allclose(draws.mean(axis=0), mean, atol=0.03)

    # vec(X) ~ N(vec(M), col_cov kron row_cov), with column-major vec.
    vec = draws.transpose(0, 2, 1).reshape(n, 4)
    empirical = np.cov(vec, rowvar=False)
    expected = np.kron(col_cov, row_cov)
    assert np.allclose(empirical, expected, atol=0.05)


@pytest.mark.statistical
def test_sample_stable_theta_draws_are_stable():
    rng = np.random.default_rng(29)
    mean = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.0]])
    row_cov = 0.25 * np.eye(2)
    col_cov = np.diag([0.3, 0.3, 0.1])
    for _ in range(500):
        theta = sample_stable_theta(rng, mean, row_cov, col_cov)
        assert theta.shape == (2, 3)
        rho = np.max(np.abs(np.linalg.eigvals(theta[:, :2])))
        assert rho < 1.0

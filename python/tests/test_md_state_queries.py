"""Multi-dimensional grid-based predictive density queries.

Python port of ``tests/test_md_state_queries.m``. Builds a 2-D model with
CORRELATED process and observation noise - so the full-covariance
Gaussian-mixture machinery is genuinely exercised rather than collapsing to a
product of independent marginals - drives it for a few trials, then checks that

* every grid density is non-negative and integrates to ~1,
* the per-context densities are proper densities keyed by global label,
* the marginal predictive CDF is a valid, monotone distribution function whose
  tails reach 0 and 1.

A scalar section confirms the unchanged scalar path still produces normalised
densities and a ``[0, 1]`` CDF.

Grid convention reminder: this package feeds MD grids as ``(K, N)`` with one
query point per ROW, the transpose of the MATLAB ``N``-by-``K`` convention, and
the density comes back ``(K,)`` (MATLAB returns a ``1``-by-``K`` row).
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import integrate_2d
from realtimecoin import RealTimeCOIN

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: Correlated process / observation noise of the MATLAB fixture.
_Q = np.array([[1.0e-4, 3.0e-5], [3.0e-5, 1.2e-4]])
_R = np.array([[4.0e-4, -1.0e-4], [-1.0e-4, 5.0e-4]])
_A_DIAG = 0.8
_DRIFT = 0.03


@pytest.fixture(scope="module")
def md_model():
    """2-D model driven for eight trials from its stationary distribution.

    The dynamics priors are pinned with a huge precision so every particle
    shares (near enough) the generative ``A`` and ``d``; that makes the density
    integrals tight enough to assert on.

    Returns
    -------
    RealTimeCOIN
        The model after eight cued observations.
    """
    n = 2
    model = RealTimeCOIN(
        num_particles=200,
        max_contexts=2,
        state_dim=n,
        prior_mean_retention=_A_DIAG,
        prior_precision_retention=1e10,
        prior_mean_drift=_DRIFT,
        prior_precision_drift=1e10,
        process_noise_covariance=_Q,
        observation_noise_covariance=_R,
        rng=11,
    )
    rng = np.random.default_rng(11)
    lq = np.linalg.cholesky(_Q)
    lr = np.linalg.cholesky(_R)
    s = np.full(n, _DRIFT / (1.0 - _A_DIAG))                             # (N,)
    for _ in range(8):
        s = _A_DIAG * s + _DRIFT + lq @ rng.standard_normal(n)
        model.observe_q(1.0)
        model.observe_y(s + lr @ rng.standard_normal(n))
    return model


@pytest.fixture(scope="module")
def scalar_model():
    """Scalar model driven for eight cued trials.

    Returns
    -------
    RealTimeCOIN
        The model after eight observations.
    """
    model = RealTimeCOIN(num_particles=50, max_contexts=3, rng=3)
    rng = np.random.default_rng(3)
    for _ in range(8):
        model.observe_q(1.0)
        model.observe_y(0.1 * float(rng.standard_normal()))
    return model


# ----------------------------------------------------------------------
# Multi-dimensional densities
# ----------------------------------------------------------------------


def test_md_state_probability_is_a_proper_density(md_model):
    """``state_probability`` is non-negative and integrates to ~1 in 2-D."""
    mu, cov = md_model.state_moments()
    integral, vals = integrate_2d(
        md_model.state_probability, mu, np.sqrt(np.diag(cov))
    )
    assert np.all(vals >= 0), "state_probability returned a negative density"
    assert abs(integral - 1.0) < 0.05, (
        "state_probability does not integrate to 1 (got %.4f)" % integral
    )


def test_md_state_probability_returns_one_value_per_grid_row(md_model):
    """A ``(K, N)`` grid yields a ``(K,)`` density - one value per ROW."""
    mu, _cov = md_model.state_moments()
    grid = mu + np.array([[0.0, 0.0], [0.01, -0.01], [-0.01, 0.01]])   # (3, 2)
    dens = md_model.state_probability(grid)
    assert dens.shape == (3,), "state_probability output is not (K,)"


def test_md_state_feedback_probability_is_a_proper_density(md_model):
    """``state_feedback_probability`` is non-negative and integrates to ~1."""
    mu, cov = md_model.predictive_feedback_moments(0)
    integral, vals = integrate_2d(
        md_model.state_feedback_probability, mu, np.sqrt(np.diag(cov))
    )
    assert np.all(vals >= 0), (
        "state_feedback_probability returned a negative density"
    )
    assert abs(integral - 1.0) < 0.05, (
        "state_feedback_probability does not integrate to 1 (got %.4f)" % integral
    )


def test_md_novel_densities_are_proper_densities(md_model):
    """The novel-context densities integrate to ~1 over a wide grid.

    The novel context is seeded from the STATIONARY distribution, which is far
    broader than the filtered posterior, so the integration window is widened
    accordingly.
    """
    mu, cov = md_model.state_moments()
    scale = 3.0 * np.sqrt(np.diag(cov))
    for name in ("novel_state_probability", "novel_state_feedback_probability"):
        integral, vals = integrate_2d(getattr(md_model, name), mu, scale)
        assert np.all(vals >= 0), "%s returned a negative density" % name
        assert abs(integral - 1.0) < 0.05, (
            "%s does not integrate to 1 (got %.4f)" % (name, integral)
        )


def test_md_state_given_context_probability_is_a_map_of_proper_densities(md_model):
    """Per-context densities: non-empty dict, ``(K,)`` values, unit integral."""
    mu, cov = md_model.state_moments()
    grid = mu + np.array([[0.0, 0.0], [0.01, -0.01], [-0.01, 0.01]])   # (3, 2)
    ctx = md_model.state_given_context_probability(grid)
    assert isinstance(ctx, dict) and ctx, (
        "state_given_context_probability returned no contexts"
    )
    for label, row in ctx.items():
        assert isinstance(label, int) and label >= 0, "keys must be 0-based ints"
        assert row.shape == (3,), "per-context density is not (K,)"
        assert np.all(row >= 0), "per-context density is negative"

    first = sorted(ctx)[0]
    integral, _vals = integrate_2d(
        lambda pts: md_model.state_given_context_probability(pts)[first],
        mu,
        np.sqrt(np.diag(cov)),
    )
    assert abs(integral - 1.0) < 0.08, (
        "per-context density does not integrate to 1 (got %.4f)" % integral
    )


def test_md_predictive_cdf_is_a_valid_monotone_marginal_cdf(md_model):
    """The MD CDF is an ``(N,)`` vector of monotone marginals with 0/1 tails."""
    mu, cov = md_model.predictive_feedback_moments(0)
    sigma = np.sqrt(np.diag(cov))                                        # (N,)
    md_model.observe_q(1.0)

    p_mid = md_model.predictive_state_feedback_cdf(mu, 1.0)
    assert p_mid.shape == (2,), "MD predictive CDF is not (N,)"
    assert np.all((p_mid >= 0) & (p_mid <= 1)), "MD predictive CDF outside [0,1]"

    p_low = md_model.predictive_state_feedback_cdf(mu - 8 * sigma, 1.0)
    p_high = md_model.predictive_state_feedback_cdf(mu + 8 * sigma, 1.0)
    assert np.all(p_low < 1e-3), "MD predictive CDF does not vanish in the tail"
    assert np.all(p_high > 1 - 1e-3), "MD predictive CDF does not approach 1"
    assert np.all(p_low <= p_mid + 1e-12) and np.all(p_mid <= p_high + 1e-12), (
        "MD predictive CDF is not monotone in y"
    )


def test_md_predictive_cdf_rejects_a_mis_sized_y(md_model):
    """A ``y`` whose length is not ``state_dim`` is a dimension mismatch."""
    with pytest.raises(ValueError, match="RealTimeCOIN:FeedbackDimensionMismatch"):
        md_model.predictive_state_feedback_cdf([0.0, 0.0, 0.0], 1.0)


# ----------------------------------------------------------------------
# Scalar re-checks (the unchanged path)
# ----------------------------------------------------------------------


def test_scalar_densities_still_normalise(scalar_model):
    """The scalar densities keep their shape and unit integral."""
    grid = np.linspace(-3.0, 3.0, 601)
    dens = scalar_model.state_probability(grid)
    assert dens.shape == grid.shape, "scalar state_probability changed shape"
    assert abs(float(_trapezoid(dens, grid)) - 1.0) < 0.05, (
        "scalar state_probability no longer integrates to 1"
    )
    feedback = scalar_model.state_feedback_probability(grid)
    assert abs(float(_trapezoid(feedback, grid)) - 1.0) < 0.05, (
        "scalar state_feedback_probability no longer integrates to 1"
    )


def test_scalar_predictive_cdf_is_a_monotone_scalar_in_unit_interval(scalar_model):
    """The scalar CDF is a plain float in ``[0, 1]``, monotone in ``y``."""
    scalar_model.observe_q(1.0)
    p_mid = scalar_model.predictive_state_feedback_cdf(0.0, 1.0)
    assert isinstance(p_mid, float) and 0.0 <= p_mid <= 1.0, (
        "scalar predictive CDF is not a float in [0, 1]"
    )
    p_low = scalar_model.predictive_state_feedback_cdf(-5.0, 1.0)
    p_high = scalar_model.predictive_state_feedback_cdf(5.0, 1.0)
    assert p_low < p_mid + 1e-12 < p_high + 1e-12, (
        "scalar predictive CDF is not monotone"
    )
    assert p_low == pytest.approx(0.0, abs=1e-9)
    assert p_high == pytest.approx(1.0, abs=1e-9)


def test_predictive_cdf_defaults_to_the_pending_cue(scalar_model):
    """Omitting ``q`` uses the cue staged by ``observe_q``, as MATLAB does."""
    scalar_model.observe_q(1.0)
    assert scalar_model.predictive_state_feedback_cdf(0.1) == pytest.approx(
        scalar_model.predictive_state_feedback_cdf(0.1, 1.0)
    )

"""The shared mixture-on-grid helper does not false-trip on real model state.

Python port of ``tests/test_marginalize_validator.m``.

``mustMarginalize`` and ``mustBeCovarianceMatrix`` are PRIVATE ``@RealTimeCOIN``
validators re-applied inside ``mixtureDensityOnGrid`` - the shared helper behind
``state_probability`` / ``state_feedback_probability`` / the ``novel_*``
densities. They guard an internal invariant (the per-particle mixing weights the
pipeline feeds them are always normalised), so their throw path is a defensive
assertion that the public API cannot reach.

What IS observable - and what this file checks - is that routing through that
helper leaves the density read-outs proper: non-negative, finite and of unit
integral, for the scalar and the multi-dimensional model alike. The Python
:func:`realtimecoin.numerics.mixture_density_on_grid` carries no validators (the
covariance screening lives in ``ensure_pd``), so here the invariant is asserted
directly on the outputs rather than through a validator that might silently
reject good state.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import integrate_2d
from realtimecoin import RealTimeCOIN

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: Every marginal density the shared mixture helper backs.
_DENSITIES = (
    "state_probability",
    "state_feedback_probability",
    "novel_state_probability",
    "novel_state_feedback_probability",
)


@pytest.fixture(scope="module")
def scalar_model():
    """Scalar model driven for eight alternating-cue trials.

    Returns
    -------
    RealTimeCOIN
        The driven model.
    """
    model = RealTimeCOIN(num_particles=40, max_contexts=3, rng=11)
    for t in range(1, 9):
        model.observe_q(float(1 + t % 2))
        model.observe_y(0.05 * t)
    return model


@pytest.fixture(scope="module")
def md_model():
    """2-D model driven for eight alternating-cue trials.

    Returns
    -------
    RealTimeCOIN
        The driven model.
    """
    model = RealTimeCOIN(num_particles=40, max_contexts=3, state_dim=2, rng=11)
    for t in range(1, 9):
        model.observe_q(float(1 + t % 2))
        model.observe_y([0.05 * t, -0.03 * t])
    return model


@pytest.mark.parametrize("name", _DENSITIES)
def test_scalar_densities_are_non_negative_finite_and_normalised(scalar_model, name):
    """Each scalar density integrates to one over a wide grid."""
    grid = np.linspace(-3.0, 3.0, 1201)
    density = getattr(scalar_model, name)(grid)
    assert density.shape == grid.shape
    assert np.all(density >= 0), "%s returned a negative density" % name
    assert np.all(np.isfinite(density)), "%s returned a non-finite density" % name
    assert float(_trapezoid(density, grid)) == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize("name", _DENSITIES)
def test_md_densities_are_non_negative_finite_and_normalised(md_model, name):
    """Each MD density is proper on a 2-D grid centred on its own mixture.

    The window follows the quantity being integrated: the filtered posterior for
    the state densities, the (wider) predictive feedback moments for the
    feedback ones, and a deliberately generous multiple of those for the
    novel-context densities, which are seeded from the far broader stationary
    distribution.
    """
    if name.startswith("novel"):
        center, cov = md_model.predictive_feedback_moments(0)
        scale = 4.0 * np.sqrt(np.diag(cov))
    elif "feedback" in name:
        center, cov = md_model.predictive_feedback_moments(0)
        scale = np.sqrt(np.diag(cov))
    else:
        center, cov = md_model.state_moments()
        scale = np.sqrt(np.diag(cov))

    integral, vals = integrate_2d(getattr(md_model, name), center, scale)
    assert np.all(vals >= 0), "%s returned a negative density" % name
    assert np.all(np.isfinite(vals)), "%s returned a non-finite density" % name
    assert integral == pytest.approx(1.0, abs=5e-2)


@pytest.mark.parametrize("name", _DENSITIES)
def test_densities_are_deterministic_and_consume_no_randomness(scalar_model, name):
    """Repeated evaluation is bit-identical and never touches ``model.rng``."""
    grid = np.linspace(-2.0, 2.0, 257)
    state = scalar_model.rng.bit_generator.state
    first = getattr(scalar_model, name)(grid)
    second = getattr(scalar_model, name)(grid)
    np.testing.assert_array_equal(first, second)
    assert scalar_model.rng.bit_generator.state == state, (
        "%s consumed randomness" % name
    )


def test_densities_reject_a_non_finite_grid(scalar_model):
    """A ``nan``/``inf`` query point is a grid validation error, as in MATLAB."""
    for name in _DENSITIES:
        with pytest.raises(ValueError, match="RealTimeCOIN:GridNotFinite"):
            getattr(scalar_model, name)([0.0, np.inf])


def test_md_grid_dimension_is_checked(md_model):
    """An MD grid whose rows are not ``state_dim``-vectors is rejected."""
    with pytest.raises(ValueError, match="RealTimeCOIN:GridDimensionMismatch"):
        md_model.state_probability(np.zeros((5, 3)))

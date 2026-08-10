"""Invariants for the COIN-plottable query methods.

Python port of ``tests/test_plot_query_methods.m``. Exercises the queries that
exist so ``RealTimeCOIN`` can reproduce every quantity ``COIN.m`` can plot: the
c*/component scalars, the per-context parameter densities, and the
transition/cue/stationary distributions.

Rather than a noisy cross-model comparison this asserts deterministic invariants
that pin the formulas down: probability vectors normalise, the stationary
distribution is a fixed point of the transition matrix, each per-context density
peaks at that context's aligned prototype moment, the explicit/implicit
identities hold, and the scalar-only / bias-only guards fire on the models that
cannot answer them.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import assert_density_peaks, assert_in_range, must_error
from realtimecoin import RealTimeCOIN

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_TOL = 1e-9

#: Grids wide enough that each parameter density is fully contained.
_RETENTION_GRID = np.linspace(0.7, 1.0, 4001)
_DRIFT_GRID = np.linspace(-0.15, 0.15, 4001)
_BIAS_GRID = np.linspace(-1.2, 1.2, 8001)


@pytest.fixture(scope="module")
def scalar_model():
    """Three-phase scalar model with cues and bias inference on.

    Eight null trials, twelve at ``+0.3`` and twelve at ``-0.3``, each phase
    carrying its own cue - enough structure for the model to instantiate
    several contexts with distinguishable biases.

    Returns
    -------
    RealTimeCOIN
        The driven model.
    """
    model = RealTimeCOIN(
        num_particles=60, max_contexts=5, infer_bias=True, rng=7
    )
    perturb = np.concatenate([np.zeros(8), 0.3 * np.ones(12), -0.3 * np.ones(12)])
    cues = np.concatenate([np.ones(8), 2.0 * np.ones(12), 3.0 * np.ones(12)])
    for cue, y in zip(cues, perturb):
        model.observe_q(float(cue))
        model.observe_y(float(y))
    return model


@pytest.fixture(scope="module")
def md_model():
    """Cue-free 2-D model driven for 25 trials.

    Returns
    -------
    RealTimeCOIN
        The driven model.
    """
    model = RealTimeCOIN(num_particles=40, max_contexts=4, state_dim=2, rng=8)
    rng = np.random.default_rng(8)
    for _ in range(25):
        model.observe_y(0.2 * rng.standard_normal(2))
    return model


# ----------------------------------------------------------------------
# Group 1: component identities and bounds
# ----------------------------------------------------------------------


def test_explicit_component_equals_state_cstar1(scalar_model):
    """``explicit_component`` is by definition the ``c*1`` state."""
    assert scalar_model.explicit_component() == pytest.approx(
        scalar_model.state_cstar1(), abs=_TOL
    )


def test_implicit_component_is_motor_output_minus_mean_state(scalar_model):
    """``implicit_component == motor_output - E[state]``."""
    mu, _var = scalar_model.state_moments()
    assert scalar_model.implicit_component() == pytest.approx(
        scalar_model.motor_output() - mu, abs=_TOL
    )


def test_cstar_probabilities_and_kalman_gains_are_in_the_unit_interval(scalar_model):
    """Probabilities and scalar Kalman gains are bounded by ``[0, 1]``."""
    assert_in_range(
        "predicted_probability_cstar1",
        scalar_model.predicted_probability_cstar1(),
        0.0,
        1.0,
    )
    assert_in_range(
        "predicted_probability_cstar3",
        scalar_model.predicted_probability_cstar3(),
        0.0,
        1.0,
    )
    assert_in_range("kalman_gain_cstar1", scalar_model.kalman_gain_cstar1(), 0.0, 1.0)
    assert_in_range("kalman_gain_cstar2", scalar_model.kalman_gain_cstar2(), 0.0, 1.0)


# ----------------------------------------------------------------------
# Group 2: per-context parameter densities
# ----------------------------------------------------------------------


def test_parameter_densities_peak_at_the_aligned_prototype_means(scalar_model):
    """Each per-context density peaks at that context's prototype mean."""
    proto = scalar_model.context_alignment()["global_contexts"]
    assert_density_peaks(
        "retention",
        scalar_model.retention_given_context_probability(_RETENTION_GRID),
        _RETENTION_GRID,
        proto["dynamics_mean"][:, 0],
    )
    assert_density_peaks(
        "drift",
        scalar_model.drift_given_context_probability(_DRIFT_GRID),
        _DRIFT_GRID,
        proto["dynamics_mean"][:, 1],
    )
    assert_density_peaks(
        "bias|context",
        scalar_model.bias_given_context_probability(_BIAS_GRID),
        _BIAS_GRID,
        proto["bias_mean"],
    )


def test_parameter_densities_are_keyed_by_zero_based_global_labels(scalar_model):
    """Keys are 0-based global labels, shared by all three parameter queries."""
    k = int(scalar_model.context_alignment()["K"])
    maps = (
        scalar_model.retention_given_context_probability(_RETENTION_GRID),
        scalar_model.drift_given_context_probability(_DRIFT_GRID),
        scalar_model.bias_given_context_probability(_BIAS_GRID),
    )
    keys = {frozenset(m) for m in maps}
    assert len(keys) == 1, "the parameter densities disagree on their context set"
    labels = sorted(next(iter(keys)))
    assert labels, "no active contexts were summarised"
    assert all(isinstance(c, int) and 0 <= c < k for c in labels), labels


def test_per_context_parameter_densities_integrate_to_one(scalar_model):
    """Each per-context parameter density is a proper density."""
    for name, grid in (
        ("retention_given_context_probability", _RETENTION_GRID),
        ("drift_given_context_probability", _DRIFT_GRID),
        ("bias_given_context_probability", _BIAS_GRID),
    ):
        for label, density in getattr(scalar_model, name)(grid).items():
            assert np.all(density >= 0) and np.all(np.isfinite(density)), (
                "%s[%d] is not a non-negative finite density" % (name, label)
            )
            assert float(_trapezoid(density, grid)) == pytest.approx(1.0, abs=5e-3)


def test_bias_probability_is_a_proper_marginal_density(scalar_model):
    """The across-context bias density is non-negative and integrates to 1."""
    density = scalar_model.bias_probability(_BIAS_GRID)
    assert np.all(density >= 0) and np.all(np.isfinite(density))
    assert float(_trapezoid(density, _BIAS_GRID)) == pytest.approx(1.0, abs=5e-3)


# ----------------------------------------------------------------------
# Group 3: transition / cue / stationary normalisation
# ----------------------------------------------------------------------


def test_local_transition_and_cue_rows_normalise(scalar_model):
    """Both aligned emission matrices are row-stochastic."""
    k = int(scalar_model.context_alignment()["K"])
    ltp = scalar_model.local_transition_probabilities()
    assert ltp.shape == (k, k + 1)
    assert np.max(np.abs(ltp.sum(axis=1) - 1.0)) < 1e-9

    lcp = scalar_model.local_cue_probabilities()
    assert lcp.shape[0] == k
    assert np.max(np.abs(lcp.sum(axis=1) - 1.0)) < 1e-9


def test_stationary_probabilities_are_a_fixed_point_of_the_transition_matrix(
    scalar_model,
):
    """``sp @ T == sp`` for the row-normalised known-context transition block."""
    k = int(scalar_model.context_alignment()["K"])
    sp = scalar_model.stationary_context_probabilities()
    assert sp.shape == (k,)
    assert float(sp.sum()) == pytest.approx(1.0, abs=1e-9)

    block = scalar_model.local_transition_probabilities()[:, :k]
    block = block / block.sum(axis=1, keepdims=True)
    assert np.max(np.abs(sp @ block - sp)) < 1e-6


def test_global_transition_and_cue_distributions_sum_to_one(scalar_model):
    """The franchise-level distributions normalise."""
    assert float(np.sum(scalar_model.global_transition_probabilities())) == (
        pytest.approx(1.0, abs=1e-9)
    )
    assert float(np.sum(scalar_model.global_cue_probabilities())) == (
        pytest.approx(1.0, abs=1e-9)
    )


# ----------------------------------------------------------------------
# Error guards
# ----------------------------------------------------------------------


def test_cue_queries_require_an_observed_cue():
    """Cue read-outs raise ``NoCues`` on a model that never saw one."""
    model = RealTimeCOIN(num_particles=20, max_contexts=3, rng=1)
    model.observe_y(0.1)
    model.observe_y(0.2)
    must_error(
        "local_cue_probabilities without cues",
        model.local_cue_probabilities,
        "RealTimeCOIN:NoCues",
    )
    must_error(
        "global_cue_probabilities without cues",
        model.global_cue_probabilities,
        "RealTimeCOIN:NoCues",
    )


def test_bias_queries_require_bias_inference():
    """Both bias read-outs raise ``BiasNotInferred`` when it is switched off."""
    model = RealTimeCOIN(num_particles=20, max_contexts=3, rng=1)
    model.observe_y(0.1)
    model.observe_y(0.2)
    assert not model.infer_bias, "the fixture assumes infer_bias defaults to False"
    with pytest.raises(Exception, match="RealTimeCOIN:BiasNotInferred"):
        model.bias_given_context_probability(_BIAS_GRID)
    with pytest.raises(Exception, match="RealTimeCOIN:BiasNotInferred"):
        model.bias_probability(_BIAS_GRID)


def test_scalar_only_queries_reject_the_multi_dimensional_model(md_model):
    """Every scalar-only read-out raises ``ScalarModelOnly`` on an MD model."""
    for call in (
        md_model.kalman_gain_cstar1,
        md_model.kalman_gain_cstar2,
        lambda: md_model.retention_given_context_probability(_RETENTION_GRID),
        lambda: md_model.drift_given_context_probability(_DRIFT_GRID),
        lambda: md_model.bias_given_context_probability(_BIAS_GRID),
        lambda: md_model.bias_probability(_BIAS_GRID),
    ):
        with pytest.raises(Exception, match="RealTimeCOIN:ScalarModelOnly"):
            call()


def test_scalar_model_check_precedes_the_bias_inference_check(md_model):
    """``bias_probability`` reports the dimension problem first, as MATLAB does.

    ``bias_probability.m`` runs ``mustBeScalarModel`` before the ``infer_bias``
    test, so an MD model with bias inference off must surface
    ``ScalarModelOnly`` rather than ``BiasNotInferred``.
    """
    assert not md_model.infer_bias
    with pytest.raises(Exception, match="RealTimeCOIN:ScalarModelOnly"):
        md_model.bias_probability(_BIAS_GRID)


# ----------------------------------------------------------------------
# Multi-dimensional shapes
# ----------------------------------------------------------------------


def test_md_component_queries_return_state_dim_vectors(md_model):
    """The MD component read-outs are ``(N,)`` vectors."""
    assert md_model.explicit_component().shape == (2,)
    assert md_model.implicit_component().shape == (2,)
    assert md_model.state_cstar2().shape == (2,)


def test_md_context_distributions_normalise(md_model):
    """The MD stationary and transition distributions keep their invariants."""
    k = int(md_model.context_alignment()["K"])
    assert float(np.sum(md_model.stationary_context_probabilities())) == (
        pytest.approx(1.0, abs=1e-9)
    )
    assert md_model.local_transition_probabilities().shape == (k, k + 1)


# ----------------------------------------------------------------------
# predictive_cue_p_value
# ----------------------------------------------------------------------


def test_predictive_cue_p_value_is_the_randomised_pit(scalar_model):
    """``p = F(q-) + u f(q)``: linear in ``u``, spanning the cue's atom."""
    at_zero = scalar_model.predictive_cue_p_value(2.0, 0.0)
    at_one = scalar_model.predictive_cue_p_value(2.0, 1.0)
    at_half = scalar_model.predictive_cue_p_value(2.0, 0.5)
    assert 0.0 <= at_zero < at_one <= 1.0
    assert at_half == pytest.approx(0.5 * (at_zero + at_one))


def test_predictive_cue_p_value_never_registers_an_unseen_cue(scalar_model):
    """A novel cue value is peeked at, not consumed: the registry is untouched."""
    before = list(scalar_model.cue_values)
    cue_columns = scalar_model.D.local_cue_matrix.shape[2]
    p = scalar_model.predictive_cue_p_value(99.0, 0.5)
    assert scalar_model.cue_values == before, "an unseen cue was registered"
    assert scalar_model.D.local_cue_matrix.shape[2] == cue_columns
    assert 0.0 <= p <= 1.0


def test_unseen_cue_lands_on_the_novel_cue_stick(scalar_model):
    """An unseen cue takes the trailing novel-cue column, not zero mass.

    ``D.local_cue_matrix`` always carries one more column than there are
    registered cue values - the novel-cue stick - and ``peek_cue_label`` returns
    exactly that column for an unseen value. So the p-value of an unseen cue is
    ``F(q-) + u * (novel-cue mass)``: it spans the top of ``[0, 1]`` and still
    depends on ``u``, contrary to the "carries zero mass" wording inherited from
    the stub docstring. This pins the MATLAB behaviour, which is identical.
    """
    n_cues = len(scalar_model.cue_values)
    assert scalar_model.D.local_cue_matrix.shape[2] == n_cues + 1

    at_zero = scalar_model.predictive_cue_p_value(99.0, 0.0)
    at_one = scalar_model.predictive_cue_p_value(99.0, 1.0)
    assert at_one == pytest.approx(1.0, abs=1e-9), (
        "the unseen cue is the last atom, so F(q) must be 1"
    )
    assert at_zero < at_one, "the novel-cue stick carries strictly positive mass"


def test_predictive_cue_p_value_is_nan_without_a_cue(scalar_model):
    """``q = None`` leaves the p-value undefined."""
    assert np.isnan(scalar_model.predictive_cue_p_value(None, 0.5))


def test_predictive_cue_p_value_draws_u_from_the_model_rng(scalar_model):
    """Omitting ``u`` consumes exactly one draw from ``model.rng``.

    Mirrors MATLAB, where ``u (1, 1) double = rand`` is evaluated in the
    ``arguments`` block, so the stream advances even on the ``nan`` return.
    """
    state = scalar_model.rng.bit_generator.state
    drawn = scalar_model.predictive_cue_p_value(2.0)
    scalar_model.rng.bit_generator.state = state
    expected_u = float(scalar_model.rng.random())
    scalar_model.rng.bit_generator.state = state
    assert drawn == pytest.approx(
        scalar_model.predictive_cue_p_value(2.0, expected_u)
    )

    scalar_model.rng.bit_generator.state = state
    scalar_model.predictive_cue_p_value(None)
    after_nan = scalar_model.rng.bit_generator.state
    scalar_model.rng.bit_generator.state = state
    scalar_model.rng.random()
    assert after_nan == scalar_model.rng.bit_generator.state

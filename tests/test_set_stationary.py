"""Stationarisation: ``set_stationary`` rewinds beliefs but keeps learning.

Python counterpart of ``tests/test_set_stationary.m``. The split the MATLAB
test draws, and which this file keeps, is between

* *what was learned* - transition and cue counts, sampled dynamics, biases, the
  cue registry - which must survive untouched, and
* *where in the experiment we are* - context probabilities, the sampled context,
  the state moments, the trial counter, the staged cue - which must be rewound
  to the stationary prior implied by what was learned.

The stationary values themselves are re-derived here from the learned
parameters (``pi T = pi``, ``m = d / (1 - a)``, ``v = sigma^2 / (1 - a^2)``, and
in the multi-dimensional case ``m = A m + d`` with the discrete Lyapunov
``V = A V A' + Q``) rather than read back from the same helper the
implementation calls, so a wrong formula cannot agree with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN
from realtimecoin.statics import stationary_distribution


def _train(state_dim=1, trials=10, **kwargs):
    """Build a seeded model and run some trials so there is something to keep."""
    kwargs.setdefault("num_particles", 25)
    kwargs.setdefault("max_contexts", 4)
    kwargs.setdefault("rng", 17)
    model = RealTimeCOIN(state_dim=state_dim, **kwargs)
    for t in range(1, trials + 1):
        model.observe_q(10 + (t % 2))
        if state_dim == 1:
            model.observe_y(0.03 * t)
        else:
            model.observe_y(np.array([0.04 * t, -0.02 * t] + [0.0] * (state_dim - 2)))
    return model


def _valid_mask(model, p):
    """The instantiated slots of particle ``p``, plus its novel slot if any."""
    k_active = int(model.D.n_active[p])
    valid = np.zeros(model.max_contexts + 1, dtype=bool)
    valid[:k_active] = True
    if k_active < model.max_contexts:
        valid[k_active] = True
    return valid


# ----------------------------------------------------------------------
# Scalar model
# ----------------------------------------------------------------------


def test_scalar_stationary_reset_keeps_counts_and_rewinds_the_trial():
    """Trial rewinds to zero while the learned transition counts stay put."""
    model = _train(infer_bias=True)
    count_mass_before = float(np.sum(model.D.n_context))
    cue_values_before = list(model.cue_values)
    n_cue_before = model.D.n_cue.copy()
    n_active_before = model.D.n_active.copy()
    assert count_mass_before > 0, "training should create transition counts"

    model.observe_q(11)                # leave a cue staged
    assert model.pending_q is not None
    model.set_stationary()

    assert model.Trial == 0
    assert model.pending_q is None
    assert float(np.sum(model.D.n_context)) == count_mass_before
    np.testing.assert_array_equal(model.D.n_cue, n_cue_before)
    np.testing.assert_array_equal(model.D.n_active, n_active_before)
    assert model.cue_values == cue_values_before
    assert model.alignment_cache is None


def test_scalar_predicted_probabilities_are_the_learned_stationary_distribution():
    """Each particle's context beliefs equal ``pi`` of its own learned ``T``."""
    model = _train(infer_bias=True)
    model.set_stationary()
    d = model.D

    for p in range(model.num_particles):
        valid = _valid_mask(model, p)
        idx = np.flatnonzero(valid)
        pi = stationary_distribution(d.local_transition_matrix[p][np.ix_(idx, idx)])

        for name in (
            "prior_probabilities",
            "predicted_probabilities",
            "responsibilities",
        ):
            row = getattr(d, name)[p]
            assert np.max(np.abs(row[idx] - pi)) < 1e-10, (
                "%s should equal the learned stationary distribution" % (name,)
            )
            assert np.all(row[~valid] == 0.0), (
                "invalid contexts should carry zero %s" % (name,)
            )
            assert abs(float(np.sum(row)) - 1.0) < 1e-10


def test_scalar_sampled_context_is_drawn_from_the_valid_slots():
    """The resampled context of every particle is one of its valid slots."""
    model = _train(infer_bias=True)
    model.set_stationary()

    np.testing.assert_array_equal(model.D.previous_context, model.D.context)
    np.testing.assert_array_equal(model.D.i_resampled, np.arange(model.num_particles))
    for p in range(model.num_particles):
        assert _valid_mask(model, p)[int(model.D.context[p])], (
            "particle %d landed on an invalid context slot" % (p,)
        )


@pytest.mark.parametrize("state_dim", [1, 2])
def test_the_context_draw_is_the_inverse_cdf_of_the_stationary_distribution(state_dim):
    """The sampled context is the exact inverse-CDF image of the drawn uniform.

    Checking only that the drawn slot is *valid* would pass for
    ``context[p] = valid_idx[0]``; so would comparing two identically-seeded
    runs. This replays the very uniforms ``set_stationary`` consumed - one per
    particle, in particle order - through ``cumsum(pi)`` and demands the exact
    slot back, which pins the draw itself.
    """
    model = _train(state_dim=state_dim, trials=6, num_particles=30, max_contexts=3)

    # Fork the stream at the point set_stationary is about to consume from.
    replay = np.random.default_rng()
    replay.bit_generator.state = model.rng.bit_generator.state

    model.set_stationary()
    d = model.D

    expected = np.zeros(model.num_particles, dtype=int)
    for p in range(model.num_particles):
        idx = np.flatnonzero(_valid_mask(model, p))
        pi = stationary_distribution(d.local_transition_matrix[p][np.ix_(idx, idx)])
        cumulative = np.cumsum(pi)
        cumulative[-1] = 1.0
        expected[p] = idx[int(np.searchsorted(cumulative, replay.random()))]

    np.testing.assert_array_equal(d.context, expected)
    # The draw must actually spread over more than one slot, or the assertion
    # above would be satisfied by any constant rule.
    assert len(set(d.context.tolist())) > 1, (
        "all particles drew the same context; the test cannot discriminate"
    )


def test_a_saturated_particle_gets_no_novel_context():
    """A particle that has used its whole budget has no novel slot to move to.

    Exercises the ``k_active == max_contexts`` branch of the valid mask, which a
    short training run never reaches on its own.
    """
    model = _train(trials=8, num_particles=12, max_contexts=3)
    model.D.n_active[:] = model.max_contexts        # saturate every particle
    model.set_stationary()
    d = model.D

    assert np.all(d.predicted_probabilities[:, model.max_contexts] == 0.0), (
        "a saturated particle must put zero mass on the novel slot"
    )
    assert np.all(d.context < model.max_contexts)
    np.testing.assert_allclose(
        np.sum(d.predicted_probabilities, axis=1), 1.0, atol=1e-10
    )
    for p in range(model.num_particles):
        idx = np.arange(model.max_contexts)
        pi = stationary_distribution(d.local_transition_matrix[p][np.ix_(idx, idx)])
        assert np.max(np.abs(d.predicted_probabilities[p, idx] - pi)) < 1e-10


def test_scalar_state_moments_are_stationary_under_the_learned_dynamics():
    """``mean = d / (1 - a)`` and ``var = sigma^2 / (1 - a^2)``, re-derived here."""
    model = _train(infer_bias=True)
    model.set_stationary()
    d = model.D
    eps = float(np.finfo(float).eps)

    denom_mean = 1.0 - d.retention
    good_mean = np.abs(denom_mean) > eps
    expected_mean = np.zeros_like(d.retention)
    expected_mean[good_mean] = d.drift[good_mean] / denom_mean[good_mean]

    denom_var = 1.0 - d.retention ** 2
    good_var = denom_var > eps
    expected_var = np.zeros_like(d.retention)
    expected_var[good_var] = model.sigma_process_noise ** 2 / denom_var[good_var]

    assert np.max(np.abs(d.state_filtered_mean - expected_mean)) < 1e-12
    assert np.max(np.abs(d.state_filtered_var - expected_var)) < 1e-12

    # The stationary equations themselves: m = a m + d and v = a^2 v + sigma^2,
    # checked only where the fixed point exists.
    residual_mean = d.state_filtered_mean - (
        d.retention * d.state_filtered_mean + d.drift
    )
    assert np.max(np.abs(residual_mean[good_mean])) < 1e-10
    residual_var = d.state_filtered_var - (
        d.retention ** 2 * d.state_filtered_var + model.sigma_process_noise ** 2
    )
    assert np.max(np.abs(residual_var[good_var])) < 1e-12


def test_scalar_predictive_and_feedback_moments_refresh():
    """Predicted and feedback moments are rebuilt from the stationary filter."""
    model = _train(infer_bias=True)
    model.set_stationary()
    d = model.D
    obs_var = model.sigma_sensory_noise ** 2 + model.sigma_motor_noise ** 2

    assert np.max(np.abs(d.state_mean - d.state_filtered_mean)) < 1e-12
    assert np.max(np.abs(d.state_var - d.state_filtered_var)) < 1e-12
    assert np.max(
        np.abs(d.previous_state_filtered_mean - d.state_filtered_mean)
    ) < 1e-12
    assert np.max(np.abs(d.previous_state_filtered_var - d.state_filtered_var)) < 1e-12
    assert np.max(np.abs(d.state_feedback_mean - (d.state_mean + d.bias))) < 1e-12
    assert np.max(np.abs(d.state_feedback_var - (d.state_var + obs_var))) < 1e-12
    assert np.all(d.probability_cue == 1.0)
    assert np.all(d.probability_state_feedback == 1.0)


@pytest.mark.parametrize("state_dim", [1, 2])
def test_state_arrays_are_not_aliased(state_dim):
    """Every state array is an independent buffer, not a view of a sibling.

    MATLAB gets this from struct value semantics; in Python a missing ``.copy()``
    would make the filtered, previous-filtered and predictive moments the SAME
    array, so the pipeline's next in-place write would silently corrupt all
    three. Checked pairwise, for both the mean/variance and the mean/covariance
    quartets, plus ``context`` / ``previous_context``.
    """
    model = _train(state_dim=state_dim, trials=6, num_particles=12, max_contexts=3,
                   infer_bias=True)
    model.set_stationary()
    d = model.D

    if state_dim == 1:
        groups = [
            ("state_filtered_mean", "previous_state_filtered_mean", "state_mean"),
            ("state_filtered_var", "previous_state_filtered_var", "state_var"),
        ]
    else:
        groups = [
            ("state_filtered_mean", "previous_state_filtered_mean", "state_mean"),
            ("state_filtered_cov", "previous_state_filtered_cov", "state_cov"),
        ]
    groups.append(("context", "previous_context"))

    for group in groups:
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                assert not np.shares_memory(
                    getattr(d, first), getattr(d, second)
                ), "%s and %s share memory" % (first, second)

    # And demonstrate the consequence directly.
    baseline = d.state_filtered_mean.copy()
    d.state_mean[...] = -99.0
    np.testing.assert_array_equal(d.state_filtered_mean, baseline)
    np.testing.assert_array_equal(d.previous_state_filtered_mean, baseline)


def test_stationarised_model_keeps_running():
    """A stationarised model is a usable model, not a wedged one."""
    model = _train(infer_bias=True)
    model.set_stationary()
    for t in range(1, 4):
        model.observe_q(10)
        model.observe_y(0.02 * t)
    assert model.Trial == 3
    assert np.isfinite(model.motor_output())


# ----------------------------------------------------------------------
# Multi-dimensional model
# ----------------------------------------------------------------------


def test_md_stationary_reset_keeps_counts_and_rewinds_the_trial():
    """MD: Trial rewinds, counts and sampled dynamics survive."""
    model = _train(state_dim=2, trials=6, num_particles=12, max_contexts=3)
    count_mass_before = float(np.sum(model.D.n_context))
    theta_before = model.D.Theta.copy()
    bias_before = model.D.bias.copy()
    assert count_mass_before > 0, "MD training should create transition counts"

    model.set_stationary()

    assert model.Trial == 0
    assert float(np.sum(model.D.n_context)) == count_mass_before
    np.testing.assert_array_equal(model.D.Theta, theta_before)
    np.testing.assert_array_equal(model.D.bias, bias_before)


def test_md_state_moments_solve_the_stationary_equations():
    """MD: ``m = A m + d`` and ``V = A V A' + Q`` hold for every (p, c)."""
    model = _train(state_dim=2, trials=6, num_particles=12, max_contexts=3)
    model.set_stationary()
    d = model.D
    n = model.state_dim
    q_cov = model.sigma_process_noise ** 2 * np.eye(n)

    worst_mean = 0.0
    worst_cov = 0.0
    for p in range(model.num_particles):
        for c in range(model.max_contexts + 1):
            a_matrix = d.Theta[p, c, :, :n]
            drift = d.Theta[p, c, :, n]
            m = d.state_filtered_mean[p, c]
            v = d.state_filtered_cov[p, c]
            worst_mean = max(
                worst_mean, float(np.max(np.abs(m - (a_matrix @ m + drift))))
            )
            worst_cov = max(
                worst_cov,
                float(np.linalg.norm(v - (a_matrix @ v @ a_matrix.T + q_cov), "fro")),
            )
    assert worst_mean < 1e-8, "MD stationary mean residual %.3g" % (worst_mean,)
    assert worst_cov < 1e-8, "MD stationary covariance residual %.3g" % (worst_cov,)


def test_md_stationary_covariance_honours_a_custom_process_noise():
    """A supplied ``Q`` (not the isotropic default) drives the Lyapunov solve."""
    q_cov = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    model = _train(
        state_dim=2,
        trials=5,
        num_particles=8,
        max_contexts=3,
        process_noise_covariance=q_cov,
    )
    model.set_stationary()
    d = model.D
    n = model.state_dim

    worst = 0.0
    for p in range(model.num_particles):
        for c in range(model.max_contexts + 1):
            a_matrix = d.Theta[p, c, :, :n]
            v = d.state_filtered_cov[p, c]
            worst = max(
                worst,
                float(np.linalg.norm(v - (a_matrix @ v @ a_matrix.T + q_cov), "fro")),
            )
    assert worst < 1e-8


def test_md_predicted_probabilities_are_the_learned_stationary_distribution():
    """The context branch is dimension-agnostic; check it in MD too."""
    model = _train(state_dim=2, trials=6, num_particles=12, max_contexts=3)
    model.set_stationary()
    d = model.D

    for p in range(model.num_particles):
        idx = np.flatnonzero(_valid_mask(model, p))
        pi = stationary_distribution(d.local_transition_matrix[p][np.ix_(idx, idx)])
        assert np.max(np.abs(d.predicted_probabilities[p, idx] - pi)) < 1e-10


def test_md_feedback_moments_refresh():
    """MD feedback moments are ``state + bias`` and ``cov + R``."""
    model = _train(state_dim=2, trials=6, num_particles=12, max_contexts=3,
                   infer_bias=True)
    model.set_stationary()
    d = model.D
    r_cov = (model.sigma_sensory_noise ** 2 + model.sigma_motor_noise ** 2) * np.eye(
        model.state_dim
    )

    np.testing.assert_allclose(d.state_mean, d.state_filtered_mean, atol=1e-12)
    np.testing.assert_allclose(d.state_cov, d.state_filtered_cov, atol=1e-12)
    np.testing.assert_allclose(
        d.state_feedback_mean, d.state_mean + d.bias, atol=1e-12
    )
    np.testing.assert_allclose(
        d.state_feedback_cov, d.state_cov + r_cov, atol=1e-12
    )


def test_md_stationarised_model_keeps_running():
    """An MD model still accepts trials after stationarisation."""
    model = _train(state_dim=2, trials=5, num_particles=10, max_contexts=3)
    model.set_stationary()
    for t in range(1, 4):
        model.observe_q(101)
        model.observe_y(np.array([0.01 * t, -0.01 * t]))
    assert model.Trial == 3
    assert np.all(np.isfinite(model.motor_output()))


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [1, 2])
def test_set_stationary_is_a_deterministic_function_of_the_rng(state_dim):
    """Two identically-seeded runs stationarise to the same state."""
    first = _train(state_dim=state_dim, trials=5, num_particles=10, max_contexts=3)
    second = _train(state_dim=state_dim, trials=5, num_particles=10, max_contexts=3)
    first.set_stationary()
    second.set_stationary()

    np.testing.assert_array_equal(first.D.context, second.D.context)
    np.testing.assert_array_equal(
        first.D.predicted_probabilities, second.D.predicted_probabilities
    )
    np.testing.assert_array_equal(
        first.D.state_filtered_mean, second.D.state_filtered_mean
    )


def test_set_stationary_on_a_fresh_model_is_a_no_op_on_the_counts():
    """A never-trained model stationarises without error and keeps zero counts."""
    model = RealTimeCOIN(num_particles=6, max_contexts=3, rng=1)
    model.set_stationary()
    assert model.Trial == 0
    assert float(np.sum(model.D.n_context)) == 0.0
    assert np.all(model.D.predicted_probabilities >= 0.0)
    np.testing.assert_allclose(
        np.sum(model.D.predicted_probabilities, axis=1), 1.0, atol=1e-12
    )

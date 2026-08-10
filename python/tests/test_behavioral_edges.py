"""Streaming edge cases: missing feedback, cue-only trials, caps, zero noise.

Python port of ``tests/test_behavioral_edges.m``. The MATLAB test reads several
of its facts out of ``diagnostics()`` (unit B3's alignment); those reads are
wrapped in :func:`_skipping_pending_units`, and the same facts are additionally
asserted directly against ``model.D``, which unit B1 owns outright. The MATLAB
"trial-counter bug" is not a bug: a missing observation is still a trial, so
three ``observe_y`` calls must leave ``Trial == 3`` (see ``CODE_REVIEW.md``, the
refuted item at the end of section 6).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN


@contextlib.contextmanager
def _skipping_pending_units(what):
    """Skip the enclosed assertions while their unit is still a stub.

    Parameters
    ----------
    what : str
        Short description of the pending dependency.

    Yields
    ------
    None
    """
    try:
        yield
    except NotImplementedError as exc:
        pytest.skip("pending %s: %s" % (what, exc))


def _edge_model():
    """Model driven through the three MATLAB edge-case trials.

    Returns
    -------
    RealTimeCOIN
        A 30-particle model after an uncued observation, a cue-only trial and
        an empty-feedback trial.
    """
    model = RealTimeCOIN(num_particles=30, max_contexts=3, rng=5)
    model.observe_y(0.1)         # observation without a cue
    model.observe_q(99)
    model.observe_y(float("nan"))   # cue-only trial (nan feedback)
    model.observe_q(42)
    model.observe_y([])          # cue-only trial (empty feedback)
    return model


def test_missing_feedback_still_advances_the_trial_counter():
    """[] and nan are missing observations, not skipped trials."""
    model = _edge_model()
    assert model.Trial == 3, "Trial counter mismatch for edge observations"


def test_responsibilities_stay_normalised_across_edge_trials():
    """Every particle keeps a proper context pmf through the edge cases."""
    model = _edge_model()
    np.testing.assert_allclose(model.D.responsibilities.sum(axis=1), 1.0, atol=1e-9)

    with _skipping_pending_units("unit B3 (alignment)"):
        vector = model.responsibilities_vector()
        assert abs(float(np.sum(vector)) - 1.0) < 1e-9, (
            "Responsibilities not normalized"
        )


def test_context_cap_and_matrix_shapes_survive_streaming():
    """The cap holds and the cue axis grew to cover both streamed cues."""
    model = _edge_model()
    assert np.all(model.D.n_active <= model.max_contexts), "Context cap exceeded"
    assert model.D.local_transition_matrix.shape[1] == model.max_contexts + 1, (
        "Transition matrix size mismatch"
    )
    # Two distinct raw cues were streamed, so at least two cue columns exist.
    assert model.D.local_cue_matrix.shape[2] >= 2, (
        "Streaming cue columns did not expand"
    )
    assert model.cue_values == [99.0, 42.0]

    with _skipping_pending_units("unit B3 (alignment)"):
        diag = model.diagnostics()
        assert np.all(diag["n_active"] <= model.max_contexts)


def test_observe_y_without_a_cue_leaves_the_cue_registry_empty():
    """An uncued trial neither registers a cue nor touches the cue counts."""
    model = RealTimeCOIN(num_particles=8, max_contexts=3, rng=5)
    before = model.D.n_cue.copy()
    model.observe_y(0.1)
    assert model.cue_values == []
    assert model.pending_q is None
    np.testing.assert_array_equal(model.D.n_cue, before)


def test_cue_columns_expand_on_a_new_raw_cue():
    """Each new raw cue value takes the next column of the cue arrays."""
    model = RealTimeCOIN(num_particles=8, max_contexts=3, rng=5)
    model.observe_q(7)
    model.observe_y(0.0)
    width_one = model.D.local_cue_matrix.shape[2]
    model.observe_q(8)
    model.observe_y(0.0)
    assert model.D.local_cue_matrix.shape[2] > width_one
    assert model.cue_values == [7.0, 8.0]
    # A repeat of a known cue must NOT grow the axis again.
    width_two = model.D.local_cue_matrix.shape[2]
    model.observe_q(7)
    model.observe_y(0.0)
    assert model.D.local_cue_matrix.shape[2] == width_two
    assert model.cue_values == [7.0, 8.0]


def test_context_cap_folds_the_novel_draw_back():
    """A long, strongly switching run never exceeds ``max_contexts``."""
    model = RealTimeCOIN(num_particles=20, max_contexts=2, rng=9)
    rng = np.random.default_rng(1)
    for t in range(60):
        # Alternate between two widely separated perturbations to push the
        # model to keep proposing novel contexts once the cap is reached.
        model.observe_y(float((-1.0) ** t) + 0.02 * rng.standard_normal())
        assert np.all(model.D.n_active <= model.max_contexts)
        assert np.all(model.D.context < model.D.n_active)
        assert np.isfinite(model.motor_output())


def test_capped_particle_folds_back_and_takes_no_stick_split():
    """The two cap rules of ``sampleContext.m``, forced with a built state.

    A particle AT ``max_contexts`` that draws the novel slot folds back onto its
    last context and does not stick-break. A particle whose increment lands
    exactly ON the cap also does not stick-break, because MATLAB selects the
    splitting particles with ``C > oldC & C < max_contexts``. Streaming rarely
    reaches either state, so the state is constructed directly.
    """
    from realtimecoin import pipeline_scalar

    max_contexts = 3
    model = RealTimeCOIN(num_particles=12, max_contexts=max_contexts, rng=42)
    d = model.D
    c_slots = max_contexts + 1
    # Four particles at the cap, four one below it, four with a single context.
    d.n_active = np.array([max_contexts] * 4 + [max_contexts - 1] * 4 + [1] * 4)
    d.context = np.zeros(model.num_particles, dtype=int)
    # Put almost all responsibility on each particle's novel slot.
    resp = np.zeros((model.num_particles, c_slots))
    for p in range(model.num_particles):
        k = int(d.n_active[p])
        resp[p, :k] = 0.02
        resp[p, k] = 1.0 - 0.02 * k
    d.responsibilities = resp / resp.sum(axis=1, keepdims=True)
    # Real stick mass in every slot so a split would be visible.
    d.global_transition_probabilities = np.tile(
        np.array([0.4, 0.3, 0.2, 0.1]), (model.num_particles, 1)
    )
    gtp_before = d.global_transition_probabilities.copy()
    filtered_before = d.state_filtered_mean.copy()

    pipeline_scalar.sample_context(model, None)

    # Capped particles: fold back onto the last context, count unchanged.
    assert np.all(model.D.n_active[:4] == max_contexts)
    assert np.all(model.D.context[:4] == max_contexts - 1)
    np.testing.assert_array_equal(
        model.D.global_transition_probabilities[:4], gtp_before[:4]
    )
    # Incremented exactly ONTO the cap: instantiated, but no split, no re-seed.
    assert np.all(model.D.n_active[4:8] == max_contexts)
    np.testing.assert_array_equal(
        model.D.global_transition_probabilities[4:8], gtp_before[4:8]
    )
    np.testing.assert_array_equal(
        model.D.state_filtered_mean[4:8], filtered_before[4:8]
    )
    # Still below the cap after the increment: split, conserving the stick mass.
    assert np.all(model.D.n_active[8:] == 2)
    for p in range(8, model.num_particles):
        c = int(model.D.n_active[p]) - 1
        kept = model.D.global_transition_probabilities[p, c]
        passed_on = model.D.global_transition_probabilities[p, c + 1]
        assert kept >= 0 and passed_on >= 0
        np.testing.assert_allclose(kept + passed_on, gtp_before[p, c], atol=1e-12)


def test_infer_bias_produces_finite_bias_samples():
    """With bias inference on, the sampled bias stays finite."""
    model = RealTimeCOIN(num_particles=20, max_contexts=2, infer_bias=True, rng=5)
    model.observe_q(1)
    model.observe_y(0.2)

    assert np.all(np.isfinite(model.D.bias)), "Bias samples must be finite"
    assert np.all(np.isfinite(model.D.bias_mean))
    assert np.all(model.D.bias_var >= 0)

    with _skipping_pending_units("unit B3 (alignment)"):
        diag = model.diagnostics()
        assert "bias" in diag, "Bias diagnostics missing"
        assert np.all(np.isfinite(np.asarray(diag["bias"])))


def test_bias_is_pinned_to_zero_when_not_inferred():
    """``infer_bias=False`` keeps the bias and its posterior variance at zero."""
    model = RealTimeCOIN(num_particles=10, max_contexts=2, rng=5)
    model.observe_y(0.2)
    np.testing.assert_array_equal(model.D.bias, 0.0)
    np.testing.assert_array_equal(model.D.bias_mean, 0.0)
    np.testing.assert_array_equal(model.D.bias_var, 0.0)


def _deterministic_model():
    """Zero-noise model with infinite-precision dynamics priors.

    Returns
    -------
    RealTimeCOIN
        The degenerate model of the MATLAB test: no process, sensory or motor
        noise, and retention/drift pinned by 1e12 prior precisions. This drives
        the ``qVar == 0`` / ``obsVar == 0`` branches of ``sample_states`` and the
        ``eps`` guards of ``sample_dynamics`` / ``sample_bias``.
    """
    return RealTimeCOIN(
        num_particles=10,
        max_contexts=1,
        sigma_process_noise=0.0,
        sigma_sensory_noise=0.0,
        sigma_motor_noise=0.0,
        prior_mean_retention=0.5,
        prior_precision_retention=1e12,
        prior_mean_drift=0.1,
        prior_precision_drift=1e12,
        rng=5,
    )


def test_zero_noise_model_stays_finite():
    """The fully deterministic model produces no inf/nan anywhere."""
    model = _deterministic_model()
    model.observe_y(0.2)

    d = model.D
    for name in (
        "state_mean",
        "state_var",
        "state_feedback_mean",
        "state_feedback_var",
        "state_filtered_mean",
        "state_filtered_var",
        "x_dynamics",
        "previous_x_dynamics",
        "x_bias",
        "retention",
        "drift",
        "bias",
        "dynamics_mean",
        "dynamics_covar",
        "responsibilities",
        "predicted_probabilities",
    ):
        value = getattr(d, name)
        assert value is not None, name
        assert np.all(np.isfinite(value)), name
    assert np.isfinite(model.motor_output())
    assert np.isfinite(model.state_moments()[0])


def test_zero_noise_pins_the_active_context_state_to_the_observation():
    """``qVar == 0`` and ``obsVar == 0`` pin the active context to ``y - bias``.

    This is the dedicated branch of ``sampleStates.m``; with no bias inference
    the sampled active-context state must equal the observation exactly.
    """
    model = _deterministic_model()
    model.observe_y(0.2)
    np.testing.assert_allclose(model.D.x_bias, 0.2, atol=1e-12)
    # The cached index is the sampled context, stored separately from D.context.
    np.testing.assert_array_equal(model.D.i_observed, model.D.context)


def test_zero_noise_density_query_is_not_all_nonfinite():
    """The degenerate posterior density still has finite entries."""
    model = _deterministic_model()
    model.observe_y(0.2)
    with _skipping_pending_units("unit C2 (densities)"):
        dens = model.state_probability(np.linspace(-1.0, 1.0, 101))
        assert np.any(np.isfinite(dens)), (
            "Deterministic density should not produce all nonfinite values"
        )

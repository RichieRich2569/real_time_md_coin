"""Basic sanity checks for the scalar RealTimeCOIN pipeline.

Python port of ``tests/test_basic.m``, plus a determinism check the MATLAB
suite has no counterpart for (MATLAB seeds the global stream; here every draw
goes through ``model.rng``, which is exactly what makes a same-seed replay
byte-identical and therefore worth asserting).

Checks that reach a module still owned by another unit - ``responsibilities_map``
and ``diagnostics`` (unit B3's alignment), ``state_probability`` (unit C2's
densities) - are wrapped in :func:`_skipping_pending_units`, so this file passes
today and tightens automatically as those units merge. Everything that depends
only on unit B1 is asserted unconditionally.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN

#: ``numpy.trapz`` was renamed ``numpy.trapezoid`` in numpy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@contextlib.contextmanager
def _skipping_pending_units(what):
    """Skip the enclosed assertions while their unit is still a stub.

    Parameters
    ----------
    what : str
        Short description of the pending dependency, used in the skip message.

    Yields
    ------
    None
    """
    try:
        yield
    except NotImplementedError as exc:
        pytest.skip("pending %s: %s" % (what, exc))


def test_initial_state_is_a_single_context(make_model):
    """A fresh model puts every particle in context 0 with weight one."""
    model = make_model(num_particles=20, max_contexts=3, rng=1)

    assert np.all(model.D.n_active == 1)
    assert np.all(model.D.context == 0)
    np.testing.assert_allclose(model.D.responsibilities[:, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(model.D.responsibilities[:, 1:], 0.0, atol=1e-12)


def test_initial_responsibilities_map_has_one_context(make_model):
    """Global-frame view of the same fact (needs the alignment unit)."""
    model = make_model(num_particles=20, max_contexts=3, rng=1)
    with _skipping_pending_units("unit B3 (alignment)"):
        probs = model.responsibilities_map()
        assert len(probs) == 1, "Initial context count mismatch"
        assert abs(probs[0] - 1.0) < 1e-12, "Initial context probability not 1"


def test_single_observation_keeps_everything_normalised(make_model):
    """One cued trial leaves proper pmfs and a finite motor output."""
    model = make_model(num_particles=20, max_contexts=3, rng=1)
    model.observe_q(1)
    model.observe_y(0.2)

    assert model.Trial == 1
    # Per-particle context distributions are proper pmfs.
    np.testing.assert_allclose(model.D.responsibilities.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        model.D.predicted_probabilities.sum(axis=1), 1.0, atol=1e-9
    )
    assert np.isfinite(model.motor_output()), "Motor output must be finite"


def test_global_context_summaries_normalise(make_model):
    """The aligned (global-frame) summaries are proper distributions."""
    model = make_model(num_particles=20, max_contexts=3, rng=1)
    model.observe_q(1)
    model.observe_y(0.2)

    with _skipping_pending_units("unit B3 (alignment)"):
        probs = model.responsibilities_map()
        assert abs(sum(probs.values()) - 1.0) < 1e-6, (
            "Context probabilities do not sum to 1"
        )

    with _skipping_pending_units("unit B3 (alignment)"):
        vector = model.predicted_context_probabilities_vector()
        assert abs(float(np.sum(vector)) - 1.0) < 1e-9, (
            "Predicted probabilities do not sum to 1"
        )


def test_state_probability_integrates_to_about_one(make_model):
    """The posterior state density integrates into a sane range."""
    model = make_model(num_particles=20, max_contexts=3, rng=1)
    model.observe_q(1)
    model.observe_y(0.2)

    grid = np.linspace(-3.0, 3.0, 601)
    with _skipping_pending_units("unit C2 (densities)"):
        dens = model.state_probability(grid)
        integral = float(_trapezoid(dens, grid))
        assert 0.0 < integral < 2.0, "State probability integral out of bounds"


def test_diagnostics_expose_the_aligned_modal_subset(make_model):
    """Diagnostics carry the predicted probabilities of the modal particles."""
    model = make_model(num_particles=20, max_contexts=3, rng=1)
    model.observe_q(1)
    model.observe_y(0.2)

    with _skipping_pending_units("unit B3 (alignment)"):
        diag = model.diagnostics()
        assert "predicted_probabilities" in diag
        assert "alignment" in diag
        assert diag["predicted_probabilities"].shape[0] == int(
            np.sum(diag["alignment"]["modal_particle_mask"])
        ), "Diagnostics should expose the aligned modal particle subset"


def test_motor_output_uses_predicted_not_posterior_weights(make_model):
    """``motor_output`` is the predicted-probability mixture mean.

    The asymmetry with ``state_probability`` (which uses the responsibilities)
    is intentional and documented in COIN; asserting the exact identity pins it.
    """
    model = make_model(num_particles=16, max_contexts=3, rng=3)
    model.observe_q(1)
    model.observe_y(0.4)

    d = model.D
    expected = float(
        np.sum(d.predicted_probabilities * d.state_feedback_mean)
        / model.num_particles
    )
    np.testing.assert_allclose(model.motor_output(), expected, rtol=0, atol=1e-15)


def test_implicit_component_is_zero_without_bias_inference(make_model):
    """With ``infer_bias`` off the motor output is exactly the state mean."""
    model = make_model(num_particles=16, max_contexts=3, rng=4)
    for t in range(5):
        model.observe_y(0.1 * t)
    np.testing.assert_allclose(model.implicit_component(), 0.0, atol=1e-12)


def test_streaming_stays_finite_and_within_the_context_cap(make_model):
    """A 40-trial run never goes non-finite and never exceeds the cap."""
    model = make_model(num_particles=25, max_contexts=4, rng=6)
    rng = np.random.default_rng(0)
    for t in range(40):
        model.observe_q(1 + (t // 10))
        model.observe_y((0.0 if t < 20 else 1.0) + 0.05 * rng.standard_normal())
        assert np.isfinite(model.motor_output())
        assert np.all(model.D.n_active <= model.max_contexts)
        assert np.all(np.isfinite(model.D.state_mean))
        assert np.all(model.D.state_var >= 0)
        assert np.all(model.D.retention >= 0) and np.all(model.D.retention < 1)


@pytest.mark.parametrize(
    "y, q", [(0.3, 1), (0.3, None), (None, 1), (None, None)]
)
def test_resample_particles_matches_a_column_major_transliteration(y, q):
    """The weighting step is bit-identical to a literal MATLAB transliteration.

    The reference below builds the ``(contexts, particles)`` arrays MATLAB works
    with, reduces over MATLAB's ``dim = 1`` and only then transposes back. It is
    therefore an independent check of every axis flip in ``resample_particles``:
    the log-sum-exp axis, the responsibility normalisation, the particle-weight
    normalisation and the identity-ancestry shortcut when both ``y`` and ``q``
    are missing.
    """
    from realtimecoin import context as _context
    from realtimecoin import pipeline_scalar
    from realtimecoin.numerics import normalize_columns, safe_log
    from realtimecoin.statics import log_sum_exp, normal_pdf, systematic_resampling

    model = RealTimeCOIN(num_particles=11, max_contexts=4, rng=7)
    noise = np.random.default_rng(3)
    for t in range(6):
        model.observe_q(1 + (t % 2))
        model.observe_y(float(noise.standard_normal()))

    # Re-run the three steps that precede the resampling for this trial.
    _context.predict_context(model, q)
    pipeline_scalar.predict_states(model)
    pipeline_scalar.predict_state_feedback(model)

    d = model.D
    c_slots = model.max_contexts + 1
    if y is None:
        py = np.ones((c_slots, model.num_particles))
    else:
        py = normal_pdf(y, d.state_feedback_mean.T, d.state_feedback_var.T)
    log_pc = safe_log(d.prior_probabilities.T)
    if q is not None:
        log_pc = log_pc + safe_log(d.probability_cue.T)
    if y is not None:
        log_pc = log_pc + safe_log(py)
    l_w = log_sum_exp(log_pc, axis=0)          # MATLAB's dim 1 == the rows
    resp = np.exp(log_pc - l_w[None, :])
    resp[~np.isfinite(resp)] = 0.0

    if y is None and q is None:
        expected_idx = np.arange(model.num_particles)
    else:
        saved = model.rng.bit_generator.state
        weights = np.exp(l_w - log_sum_exp(l_w, axis=-1))
        expected_idx = systematic_resampling(model.rng, weights)
        model.rng.bit_generator.state = saved   # replay the same draw below

    expected_resp = normalize_columns(resp.T[expected_idx])
    pipeline_scalar.resample_particles(model, y, q)

    np.testing.assert_array_equal(model.D.i_resampled, expected_idx)
    np.testing.assert_array_equal(
        model.D.probability_state_feedback, py.T[expected_idx]
    )
    np.testing.assert_array_equal(model.D.responsibilities, expected_resp)


def test_same_seed_gives_identical_motor_output_traces():
    """Two models built with the same seed produce identical traces."""
    y = np.concatenate([np.zeros(15), np.ones(15)])

    def _run():
        model = RealTimeCOIN(num_particles=30, max_contexts=4, rng=2024)
        trace = []
        for t, y_t in enumerate(y):
            model.observe_q(1 + (t % 2))
            model.observe_y(float(y_t))
            trace.append(model.motor_output())
        return np.asarray(trace)

    first = _run()
    second = _run()
    assert first.shape == (30,)
    assert np.all(np.isfinite(first))
    # Bit-for-bit: every draw goes through model.rng, nothing through the
    # global numpy state.
    np.testing.assert_array_equal(first, second)


@pytest.mark.slow
@pytest.mark.statistical
def test_spontaneous_recovery_after_a_counter_perturbation():
    """The scalar pipeline reproduces COIN's hallmark behavioural signature.

    Adapt to +1, briefly counter-perturb to -1 until the motor output crosses
    back through zero, then run error-clamp (missing-feedback) trials. Contextual
    inference predicts *spontaneous recovery*: the output drifts back towards the
    first perturbation instead of staying where the counter-perturbation left it.
    A single-context learner cannot produce this, so it is an end-to-end check
    that the context machinery, the resampling and the state updates all work
    together - not just that each is individually finite.
    """
    noise = np.random.default_rng(0)
    schedule = [1.0] * 60 + [-1.0] * 15 + [None] * 25
    model = RealTimeCOIN(num_particles=80, max_contexts=5, rng=11)

    trace = []
    for target in schedule:
        model.observe_y(
            None if target is None else target + 0.03 * noise.standard_normal()
        )
        trace.append(model.motor_output())
    trace = np.asarray(trace)

    assert np.all(np.isfinite(trace))
    adapted = trace[55:60].mean()
    counter = trace[70:75].mean()
    clamp_early = trace[75:80].mean()
    clamp_late = trace[95:100].mean()

    assert adapted > 0.5, "did not adapt to the first perturbation"
    assert counter < 0.0, "did not de-adapt under the counter-perturbation"
    assert clamp_late - clamp_early > 0.1, (
        "no spontaneous recovery during the error-clamp phase "
        "(early %.3f, late %.3f)" % (clamp_early, clamp_late)
    )
    # More than one context must have been inferred for this to be possible.
    assert model.D.n_active.max() >= 2


def test_a_different_seed_gives_a_different_trace():
    """The determinism above is seeding, not a degenerate constant trace."""
    y = np.concatenate([np.zeros(15), np.ones(15)])

    def _run(seed):
        model = RealTimeCOIN(num_particles=30, max_contexts=4, rng=seed)
        return np.asarray([
            (model.observe_y(float(y_t)), model.motor_output())[1] for y_t in y
        ])

    assert not np.array_equal(_run(1), _run(2))

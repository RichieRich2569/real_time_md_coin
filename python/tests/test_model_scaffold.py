"""Scaffold tests: constructor validation, reset invariants, cues, prediction.

These pin the package interface that every later translation unit builds on:
the constructor defaults and validators, the particle-state array shapes, the
cue registry's consume-vs-peek asymmetry, and the shared
:func:`realtimecoin.context.predict_context` step.

Expected values are read from the MATLAB sources (``RealTimeCOIN.m``,
``private/resetParticles.m``, ``private/resetParticlesMD.m``,
``private/validateMultiDimConfig.m``), not guessed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from realtimecoin import ParticleState, RealTimeCOIN
from realtimecoin.exceptions import NameValuePairsError
from realtimecoin import context as rc_context
from realtimecoin import state as rc_state

from helpers import assert_close, assert_in_range, must_error


# ----------------------------------------------------------------------
# Constructor: defaults
# ----------------------------------------------------------------------


def test_defaults_match_matlab(make_model):
    """Every default equals the literal in @RealTimeCOIN/RealTimeCOIN.m."""
    m = make_model()
    assert m.num_particles == 100
    assert m.max_contexts == 10
    assert m.state_dim == 1
    assert m.gamma_context == 0.1
    assert m.alpha_context == 8.955
    assert m.rho_context == 0.2501
    assert m.gamma_cue == 0.1
    assert m.alpha_cue == 25
    assert m.prior_mean_retention == 0.9425
    assert m.prior_mean_drift == 0.0
    assert m.prior_mean_bias == 0.0
    # Kept in the unevaluated (std)^2 form of the MATLAB source.
    assert m.prior_precision_retention == 837.1 ** 2
    assert m.prior_precision_drift == 1.2227e3 ** 2
    assert m.prior_precision_bias == 70 ** 2
    assert m.sigma_process_noise == 0.0089
    assert m.sigma_sensory_noise == 0.03
    assert m.sigma_motor_noise == 0.0
    assert m.process_noise_covariance is None
    assert m.observation_noise_covariance is None
    assert m.infer_bias is False


def test_trial_starts_at_zero_and_is_read_only(make_model):
    """Trial is a dependent, read-only view of the private counter."""
    m = make_model()
    assert m.Trial == 0
    with pytest.raises(AttributeError):
        m.Trial = 5


def test_rng_seed_makes_construction_reproducible():
    """The same seed gives byte-identical prior draws."""
    a = RealTimeCOIN(num_particles=7, max_contexts=3, rng=99)
    b = RealTimeCOIN(num_particles=7, max_contexts=3, rng=99)
    assert_close(a.D.retention, b.D.retention, 0.0, "retention differs")
    assert_close(a.D.drift, b.D.drift, 0.0, "drift differs")


def test_rng_accepts_a_generator():
    """A Generator instance is accepted as well as a seed."""
    m = RealTimeCOIN(num_particles=4, rng=np.random.default_rng(3))
    assert isinstance(m.rng, np.random.Generator)


# ----------------------------------------------------------------------
# Constructor: validation
# ----------------------------------------------------------------------


def test_unknown_kwarg_raises_name_value_pairs():
    """An unrecognised option maps onto RealTimeCOIN:NameValuePairs."""
    with pytest.raises(NameValuePairsError, match="RealTimeCOIN:NameValuePairs"):
        RealTimeCOIN(no_such_property=1)


def test_unknown_kwarg_names_the_offender():
    """The message lists the bad option so a typo is obvious."""
    exc = must_error(
        "unknown kwarg",
        lambda: RealTimeCOIN(num_partciles=10),
        "num_partciles",
    )
    assert isinstance(exc, NameValuePairsError)


@pytest.mark.parametrize(
    "kwargs, identifier",
    [
        ({"num_particles": -5}, "mustBePositive"),
        ({"num_particles": 0}, "mustBePositive"),
        ({"num_particles": 2.5}, "mustBeInteger"),
        ({"max_contexts": -1}, "mustBePositive"),
        ({"state_dim": 0}, "mustBePositive"),
        ({"alpha_context": 0}, "mustBePositive"),
        ({"alpha_cue": -1}, "mustBePositive"),
        ({"gamma_context": -0.1}, "mustBeNonnegative"),
        ({"gamma_cue": -0.1}, "mustBeNonnegative"),
        ({"sigma_process_noise": -1}, "mustBeNonnegative"),
        ({"prior_precision_bias": -1}, "mustBeNonnegative"),
        ({"rho_context": 1.0}, "mustBeLessThan"),
        ({"rho_context": 1.5}, "mustBeLessThan"),
        ({"rho_context": -0.1}, "mustBeNonnegative"),
    ],
)
def test_invalid_property_values_raise(kwargs, identifier):
    """Each MATLAB property validator has a Python counterpart."""
    must_error(str(kwargs), lambda: RealTimeCOIN(**kwargs), identifier)


def test_rho_context_just_below_one_is_accepted():
    """[0, 1) is half-open: 1 - eps is legal."""
    m = RealTimeCOIN(num_particles=2, rho_context=1.0 - 1e-12, rng=0)
    assert m.rho_context < 1


def test_covariance_size_mismatch_rejected():
    """A covariance must be state_dim-by-state_dim."""
    must_error(
        "bad size",
        lambda: RealTimeCOIN(
            state_dim=3, num_particles=2, process_noise_covariance=np.eye(2), rng=0
        ),
        "RealTimeCOIN:BadCovarianceSize",
    )


def test_non_symmetric_covariance_rejected():
    """Asymmetry beyond the 1e-9 relative tolerance is an error."""
    bad = np.array([[1.0, 0.5], [0.0, 1.0]])
    must_error(
        "asymmetric",
        lambda: RealTimeCOIN(
            state_dim=2, num_particles=2, observation_noise_covariance=bad, rng=0
        ),
        "RealTimeCOIN:CovarianceNotSymmetric",
    )


def test_non_psd_covariance_rejected():
    """A symmetric but indefinite covariance is an error."""
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues 3 and -1
    must_error(
        "not psd",
        lambda: RealTimeCOIN(
            state_dim=2, num_particles=2, process_noise_covariance=bad, rng=0
        ),
        "RealTimeCOIN:CovarianceNotPSD",
    )


def test_non_finite_covariance_rejected():
    """Non-finite entries are caught before they reach the filter."""
    bad = np.array([[1.0, 0.0], [0.0, np.inf]])
    must_error(
        "non-finite",
        lambda: RealTimeCOIN(
            state_dim=2, num_particles=2, process_noise_covariance=bad, rng=0
        ),
        "RealTimeCOIN:BadCovarianceValue",
    )


def test_scalar_model_warns_about_ignored_covariance():
    """validateMultiDimConfig WARNS (does not error) for state_dim == 1.

    The MATLAB source issues RealTimeCOIN:CovarianceIgnored and carries on with
    the scalar sigma_* properties, so the Python translation must warn too.
    """
    with pytest.warns(UserWarning, match="RealTimeCOIN:CovarianceIgnored"):
        m = RealTimeCOIN(
            num_particles=3, process_noise_covariance=np.eye(1) * 4.0, rng=0
        )
    # The override is retained on the object but not used by the scalar path.
    assert m.state_dim == 1
    assert m.process_noise_covariance is not None


def test_scalar_model_without_covariance_does_not_warn():
    """No warning when the overrides are left unset."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        RealTimeCOIN(num_particles=3, rng=0)


def test_valid_md_covariances_accepted():
    """A well-formed pair of SPD covariances constructs cleanly."""
    q = np.array([[2.0, 0.3], [0.3, 1.0]])
    r = np.array([[0.5, 0.0], [0.0, 0.4]])
    m = RealTimeCOIN(
        state_dim=2,
        num_particles=3,
        max_contexts=2,
        process_noise_covariance=q,
        observation_noise_covariance=r,
        rng=0,
    )
    assert_close(rc_state.process_noise_cov(m), q, 1e-12, "Q round-trip")
    assert_close(rc_state.observation_noise_cov(m), r, 1e-12, "R round-trip")


def test_default_noise_covariances_are_isotropic():
    """Unset overrides give the isotropic defaults derived from sigma_*."""
    m = RealTimeCOIN(
        state_dim=3,
        num_particles=2,
        max_contexts=2,
        sigma_process_noise=0.5,
        sigma_sensory_noise=0.2,
        sigma_motor_noise=0.1,
        rng=0,
    )
    assert_close(
        rc_state.process_noise_cov(m), 0.25 * np.eye(3), 1e-12, "default Q"
    )
    expected_r = (0.2 ** 2 + 0.1 ** 2) * np.eye(3)
    assert_close(rc_state.observation_noise_cov(m), expected_r, 1e-12, "default R")
    assert rc_state.observation_variance(m) == pytest.approx(0.2 ** 2 + 0.1 ** 2)


# ----------------------------------------------------------------------
# Reset invariants
# ----------------------------------------------------------------------


def test_reset_scalar_discrete_bookkeeping(make_model):
    """Scalar reset seeds the discrete per-particle fields (resetParticles.m)."""
    p_count, max_c = 6, 4
    m = make_model(num_particles=p_count, max_contexts=max_c)
    c_slots = max_c + 1
    d = m.D

    assert d.n_active.shape == (p_count,)
    assert np.all(d.n_active == 1)                    # one instantiated context
    assert np.all(d.context == 0)                     # 0-based: context 0
    assert np.all(d.previous_context == 0)
    assert d.n_cues == 0

    assert d.n_context.shape == (p_count, c_slots, c_slots)
    assert not d.n_context.any()                      # counts start at zero
    assert d.n_cue.shape == (p_count, c_slots, 1)
    assert not d.n_cue.any()

    assert d.dynamics_ss_1.shape == (p_count, c_slots, 2)
    assert d.dynamics_ss_2.shape == (p_count, c_slots, 2, 2)
    assert d.bias_ss_1.shape == (p_count, c_slots)
    assert d.bias_ss_2.shape == (p_count, c_slots)
    assert not d.dynamics_ss_1.any()
    assert not d.dynamics_ss_2.any()

    assert d.global_transition_probabilities.shape == (p_count, c_slots)
    # MATLAB seeds ROW 1 of a (Cmax, P) array; transposed, that is COLUMN 0.
    assert np.all(d.global_transition_probabilities[:, 0] == 1.0)
    assert not d.global_transition_probabilities[:, 1:].any()
    assert d.global_cue_probabilities.shape == (p_count, 1)
    assert np.all(d.global_cue_probabilities == 1.0)


def test_reset_scalar_per_context_shapes(make_model):
    """Every scalar per-context array is (P, C)."""
    p_count, max_c = 6, 4
    m = make_model(num_particles=p_count, max_contexts=max_c)
    c_slots = max_c + 1
    d = m.D

    for name in (
        "retention",
        "drift",
        "bias",
        "state_filtered_mean",
        "state_filtered_var",
        "previous_state_filtered_mean",
        "previous_state_filtered_var",
        "state_mean",
        "state_var",
        "state_feedback_mean",
        "state_feedback_var",
        "prior_probabilities",
        "predicted_probabilities",
        "responsibilities",
        "probability_cue",
        "probability_state_feedback",
    ):
        assert getattr(d, name).shape == (p_count, c_slots), name

    assert_in_range("retention", d.retention, 0.0, 1.0)
    assert not d.bias.any()                            # infer_bias defaults off
    assert np.all(d.probability_cue == 1.0)
    assert np.all(d.probability_state_feedback == 1.0)
    assert np.all(d.prior_probabilities[:, 0] == 1.0)
    assert not d.prior_probabilities[:, 1:].any()
    assert_close(d.predicted_probabilities, d.prior_probabilities, 0.0, "predicted")
    assert_close(d.responsibilities, d.prior_probabilities, 0.0, "responsibilities")
    assert np.array_equal(d.i_resampled, np.arange(p_count))

    # Derived local matrices are built by reset.
    assert d.local_transition_matrix.shape == (p_count, c_slots, c_slots)
    assert d.local_cue_matrix.shape == (p_count, c_slots, 1)


def test_reset_scalar_leaves_other_mode_fields_none(make_model):
    """MD-only and lazily-created fields are None after a scalar reset."""
    d = make_model(num_particles=6, max_contexts=4).D

    # MD-only fields stay None in the scalar model.
    for name in ("Theta", "Lambda_xx", "Lambda_yx", "state_cov", "state_filtered_cov"):
        assert getattr(d, name) is None, name

    # Lazily created fields stay None until their pipeline step runs, mirroring
    # MATLAB where the struct field does not exist until then.
    for name in ("x_dynamics", "x_bias", "i_observed", "dynamics_mean", "bias_mean"):
        assert getattr(d, name) is None, name


def test_reset_scalar_predicted_is_not_an_alias(make_model):
    """predicted_probabilities is a COPY of prior_probabilities, not an alias."""
    m = make_model(num_particles=3, max_contexts=2)
    m.D.prior_probabilities[0, 0] = 0.5
    assert m.D.predicted_probabilities[0, 0] == 1.0
    assert m.D.responsibilities[0, 0] == 1.0


def test_reset_scalar_state_moments_are_stationary(make_model):
    """Filtered moments start at the stationary AR(1) values of the priors."""
    m = make_model(num_particles=5, max_contexts=3)
    d = m.D
    expected_mean = d.drift / (1.0 - d.retention)
    expected_var = m.sigma_process_noise ** 2 / (1.0 - d.retention ** 2)
    assert_close(d.state_filtered_mean, expected_mean, 1e-9, "stationary mean")
    assert_close(d.state_filtered_var, expected_var, 1e-9, "stationary var")
    # Feedback moments add the bias and the observation variance.
    assert_close(d.state_feedback_mean, d.state_mean + d.bias, 0.0, "fb mean")
    assert_close(
        d.state_feedback_var,
        d.state_var + rc_state.observation_variance(m),
        0.0,
        "fb var",
    )


def test_reset_scalar_previous_moments_are_copies(make_model):
    """previous_* start equal to the current moments but are independent."""
    m = make_model(num_particles=3, max_contexts=2)
    d = m.D
    assert_close(d.previous_state_filtered_mean, d.state_filtered_mean, 0.0, "prev")
    d.state_filtered_mean[0, 0] = 12345.0
    assert d.previous_state_filtered_mean[0, 0] != 12345.0


def test_reset_scalar_infer_bias_draws_a_bias(make_model):
    """With infer_bias the bias is drawn from its prior rather than zeroed."""
    m = make_model(num_particles=8, max_contexts=3, infer_bias=True)
    assert m.D.bias.shape == (8, 4)
    assert m.D.bias.any()


def test_reset_md_shapes(make_model):
    """MD reset matches resetParticlesMD.m field-for-field."""
    p_count, max_c, n = 4, 3, 2
    m = make_model(num_particles=p_count, max_contexts=max_c, state_dim=n)
    c_slots = max_c + 1
    d = m.D

    assert np.all(d.n_active == 1)
    assert np.all(d.context == 0)
    assert d.n_cues == 0
    assert d.n_context.shape == (p_count, c_slots, c_slots)
    assert d.n_cue.shape == (p_count, c_slots, 1)
    assert d.global_transition_probabilities.shape == (p_count, c_slots)
    assert np.all(d.global_transition_probabilities[:, 0] == 1.0)
    assert d.global_cue_probabilities.shape == (p_count, 1)

    assert d.Lambda_xx.shape == (p_count, c_slots, n + 1, n + 1)
    assert d.Lambda_yx.shape == (p_count, c_slots, n, n + 1)
    assert d.bias_info_ss.shape == (p_count, c_slots, n)
    assert d.bias_precision_ss.shape == (p_count, c_slots, n, n)
    assert not d.Lambda_xx.any()
    assert not d.Lambda_yx.any()

    assert d.Theta.shape == (p_count, c_slots, n, n + 1)
    assert d.bias.shape == (p_count, c_slots, n)
    assert not d.bias.any()                       # infer_bias off by default

    assert d.state_filtered_mean.shape == (p_count, c_slots, n)
    assert d.state_filtered_cov.shape == (p_count, c_slots, n, n)
    assert d.state_mean.shape == (p_count, c_slots, n)
    assert d.state_cov.shape == (p_count, c_slots, n, n)
    assert d.state_feedback_mean.shape == (p_count, c_slots, n)
    assert d.state_feedback_cov.shape == (p_count, c_slots, n, n)

    # Context probabilities keep their scalar-model shapes and semantics.
    assert d.prior_probabilities.shape == (p_count, c_slots)
    assert np.all(d.prior_probabilities[:, 0] == 1.0)
    assert np.all(d.probability_cue == 1.0)
    assert np.array_equal(d.i_resampled, np.arange(p_count))

    # Scalar-only fields stay None in the MD model.
    for name in ("retention", "drift", "state_var", "dynamics_ss_1", "bias_ss_1"):
        assert getattr(d, name) is None, name


def test_reset_md_theta_is_stable(make_model):
    """Every sampled Theta has a dynamics block with spectral radius < 1."""
    m = make_model(num_particles=3, max_contexts=2, state_dim=2)
    n = m.state_dim
    for p in range(m.num_particles):
        for c in range(m.max_contexts + 1):
            a = m.D.Theta[p, c, :, :n]
            assert np.max(np.abs(np.linalg.eigvals(a))) < 1.0


def test_reset_md_feedback_covariance_adds_r(make_model):
    """state_feedback_cov = state_cov + R, broadcast over particles/contexts."""
    m = make_model(num_particles=3, max_contexts=2, state_dim=2)
    r = rc_state.observation_noise_cov(m)
    assert_close(
        m.D.state_feedback_cov, m.D.state_cov + r, 1e-12, "feedback covariance"
    )


def test_copy_state_is_deep(make_model):
    """copy_state shares no buffer with the original."""
    m = make_model(num_particles=3, max_contexts=2)
    other = rc_state.copy_state(m.D)
    other.retention[0, 0] = -99.0
    assert m.D.retention[0, 0] != -99.0
    assert set(other.field_names()) == set(m.D.field_names())


# ----------------------------------------------------------------------
# Cue registry
# ----------------------------------------------------------------------


def test_observe_q_stages_and_clears(make_model):
    """observe_q stages a raw value; None and nan clear it."""
    m = make_model(num_particles=2, max_contexts=2)
    assert m.pending_q is None
    m.observe_q(7)
    assert m.pending_q == 7
    m.observe_q(None)
    assert m.pending_q is None
    m.observe_q(7)
    m.observe_q(float("nan"))
    assert m.pending_q is None


def test_consume_registers_and_grows_arrays(make_model):
    """Consuming a fresh cue appends a label and grows the cue-indexed arrays."""
    m = make_model(num_particles=3, max_contexts=2)
    c_slots = m.max_contexts + 1
    assert m.D.n_cue.shape == (3, c_slots, 1)

    m.observe_q(5)
    q = rc_state.consume_pending_cue(m)
    assert q == 0                       # 0-based first label
    assert m.cue_values == [5.0]
    assert m.pending_q is None
    assert m.D.n_cue.shape == (3, c_slots, 1)   # label 0 already addressable

    m.observe_q(9)
    q = rc_state.consume_pending_cue(m)
    assert q == 1
    assert m.cue_values == [5.0, 9.0]
    assert m.D.n_cue.shape == (3, c_slots, 2)          # grew a column
    assert m.D.global_cue_probabilities.shape == (3, 2)
    assert m.D.local_cue_matrix.shape == (3, c_slots, 2)
    assert not m.D.n_cue[:, :, 1].any()                # new column is zero


def test_consume_reuses_the_label_of_a_repeated_cue(make_model):
    """A repeated raw value maps back to its existing label."""
    m = make_model(num_particles=2, max_contexts=2)
    m.observe_q(4)
    first = rc_state.consume_pending_cue(m)
    m.observe_q(4)
    second = rc_state.consume_pending_cue(m)
    assert first == second == 0
    assert m.cue_values == [4.0]


def test_consume_without_a_staged_cue_returns_none(make_model):
    """No staged cue means an un-cued trial."""
    m = make_model(num_particles=2, max_contexts=2)
    assert rc_state.consume_pending_cue(m) is None
    assert m.cue_values == []


def test_peek_does_not_register(make_model):
    """peek_cue_label is read-only: it registers nothing and grows nothing."""
    m = make_model(num_particles=3, max_contexts=2)
    before_shape = m.D.n_cue.shape
    assert rc_state.peek_cue_label(m, 42) == 0       # next unused label
    assert m.cue_values == []
    assert m.D.n_cue.shape == before_shape

    m.observe_q(42)
    rc_state.consume_pending_cue(m)
    assert rc_state.peek_cue_label(m, 42) == 0       # now an existing label
    assert rc_state.peek_cue_label(m, 43) == 1       # still just a preview
    assert m.cue_values == [42.0]
    assert rc_state.peek_cue_label(m, None) is None


def test_ensure_cue_column_is_idempotent_and_preserves_data(make_model):
    """Growing to a width already available is a no-op; data survives a grow."""
    m = make_model(num_particles=2, max_contexts=2)
    m.D.n_cue[0, 0, 0] = 3.0
    rc_state.ensure_cue_column(m, 1)
    assert m.D.n_cue.shape[2] == 1
    rc_state.ensure_cue_column(m, 4)
    assert m.D.n_cue.shape[2] == 4
    assert m.D.global_cue_probabilities.shape[1] == 4
    assert m.D.n_cue[0, 0, 0] == 3.0                 # preserved
    assert not m.D.n_cue[:, :, 1:].any()             # new columns zero
    rc_state.ensure_cue_column(m, None)              # no-op


def test_instantiate_cue_splits_the_stick(make_model):
    """instantiate_cue_if_needed stick-breaks mass onto the fresh category."""
    m = make_model(num_particles=5, max_contexts=2)
    rc_state.instantiate_cue_if_needed(m, 0)
    assert m.D.n_cues == 1
    assert m.D.global_cue_probabilities.shape[1] >= 2
    # The split conserves the mass that was on category 0.
    total = m.D.global_cue_probabilities[:, 0] + m.D.global_cue_probabilities[:, 1]
    assert_close(total, np.ones(5), 1e-12, "stick mass conserved")
    # Already instantiated: a second call is a no-op.
    before = m.D.global_cue_probabilities.copy()
    rc_state.instantiate_cue_if_needed(m, 0)
    assert_close(m.D.global_cue_probabilities, before, 0.0, "no-op")
    rc_state.instantiate_cue_if_needed(m, None)


# ----------------------------------------------------------------------
# predict_context
# ----------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [1, 2])
def test_predict_context_rows_are_distributions(make_model, state_dim):
    """Each particle's predicted row is a probability distribution."""
    m = make_model(num_particles=6, max_contexts=4, state_dim=state_dim)
    rc_context.predict_context(m, None)
    d = m.D
    assert d.prior_probabilities.shape == (6, 5)
    assert d.predicted_probabilities.shape == (6, 5)
    for p in range(6):
        assert d.predicted_probabilities[p, :].sum() == pytest.approx(1.0)
        assert d.prior_probabilities[p, :].sum() == pytest.approx(1.0)
    assert_in_range("predicted", d.predicted_probabilities, 0.0, 1.0)


@pytest.mark.parametrize("state_dim", [1, 2])
def test_predict_context_confines_mass_to_valid_slots(make_model, state_dim):
    """Slots beyond a particle's novel slot can never carry mass.

    On a FRESH model the global franchise weights are still ``e_0`` (see
    resetParticles.m), so the novel slot itself has zero mass too: the first
    trial must be context 0. The novel slot only opens once the global betas
    have been resampled - see the next test.
    """
    m = make_model(num_particles=6, max_contexts=4, state_dim=state_dim)
    rc_context.predict_context(m, None)
    d = m.D
    for p in range(6):
        novel = int(d.n_active[p])          # 0-based novel slot
        assert not d.predicted_probabilities[p, novel + 1:].any()
    assert_close(
        d.predicted_probabilities[:, 0], np.ones(6), 1e-12, "all mass on context 0"
    )


@pytest.mark.parametrize("state_dim", [1, 2])
def test_predict_context_novel_slot_opens_after_a_beta_resample(make_model, state_dim):
    """Once gamma_context reaches the franchise, the novel slot carries mass."""
    m = make_model(num_particles=6, max_contexts=4, state_dim=state_dim)
    rc_context.sample_global_transition_probabilities(m)
    d = m.D
    # gamma_context lands in the first unused slot; nothing beyond it.
    assert np.all(d.global_transition_probabilities[:, 1] > 0.0)
    assert not d.global_transition_probabilities[:, 2:].any()

    rc_context.predict_context(m, None)
    for p in range(6):
        novel = int(d.n_active[p])
        assert d.predicted_probabilities[p, novel] > 0.0
        assert d.predicted_probabilities[p, :].sum() == pytest.approx(1.0)
        assert not d.predicted_probabilities[p, novel + 1:].any()


def test_predict_context_uncued_predicted_equals_prior(make_model):
    """Without a cue the cue term is a no-op and predicted == prior."""
    m = make_model(num_particles=4, max_contexts=3)
    rc_context.predict_context(m, None)
    assert_close(
        m.D.predicted_probabilities, m.D.prior_probabilities, 0.0, "uncued"
    )
    assert np.all(m.D.probability_cue == 1.0)
    # And they are separate arrays, not aliases.
    m.D.prior_probabilities[0, 0] = 0.0
    assert m.D.predicted_probabilities[0, 0] != 0.0


def test_predict_context_with_a_cue_tilts_the_prior(make_model):
    """A cue multiplies in p(q|c) and renormalises."""
    m = make_model(num_particles=4, max_contexts=3)
    m.observe_q(1)
    q = rc_state.consume_pending_cue(m)
    rc_context.predict_context(m, q)
    d = m.D
    expected = d.prior_probabilities * d.probability_cue
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert_close(d.predicted_probabilities, expected, 1e-12, "cue tilt")
    for p in range(4):
        assert d.predicted_probabilities[p, :].sum() == pytest.approx(1.0)


def test_predict_context_probability_cue_is_not_a_view(make_model):
    """probability_cue must be an independent array, not a view.

    MATLAB's squeeze() copies; numpy integer indexing returns a strided view,
    so without an explicit copy a write to probability_cue would corrupt
    local_cue_matrix.
    """
    m = make_model(num_particles=4, max_contexts=3)
    m.observe_q(1)
    q = rc_state.consume_pending_cue(m)
    rc_context.predict_context(m, q)
    before = m.D.local_cue_matrix[:, :, q].copy()
    m.D.probability_cue[:] = -1.0
    assert_close(m.D.local_cue_matrix[:, :, q], before, 0.0, "cue matrix intact")


def test_predict_context_is_deterministic_under_a_fixed_seed():
    """Same seed, same result: predict_context draws no random numbers."""
    a = RealTimeCOIN(num_particles=5, max_contexts=3, rng=2024)
    b = RealTimeCOIN(num_particles=5, max_contexts=3, rng=2024)
    rc_context.predict_context(a, None)
    rc_context.predict_context(b, None)
    assert_close(
        a.D.predicted_probabilities, b.D.predicted_probabilities, 0.0, "determinism"
    )
    # Re-running it on the same model is idempotent (no RNG consumed).
    before = a.D.predicted_probabilities.copy()
    rc_context.predict_context(a, None)
    assert_close(a.D.predicted_probabilities, before, 0.0, "idempotent")


def test_local_transition_matrix_rows_are_stochastic_or_zero(make_model):
    """Valid source rows sum to one; invalid ones are exactly zero."""
    m = make_model(num_particles=4, max_contexts=3)
    t = m.D.local_transition_matrix
    for p in range(4):
        k = int(m.D.n_active[p])
        for r in range(m.max_contexts + 1):
            row_sum = t[p, r, :].sum()
            if r <= k:                    # instantiated rows plus the novel slot
                assert row_sum == pytest.approx(1.0)
            else:
                assert row_sum == 0.0


def test_kappa_matches_the_formula(make_model):
    """kappa = alpha * rho / (1 - rho)."""
    m = make_model(alpha_context=4.0, rho_context=0.25, num_particles=2)
    assert rc_context.kappa(m) == pytest.approx(4.0 * 0.25 / 0.75)


def test_current_transition_prior_matches_predict_context(make_model):
    """The read-only prior equals the one predict_context computes."""
    m = make_model(num_particles=4, max_contexts=3)
    rc_context.predict_context(m, None)
    assert_close(
        rc_context.current_transition_prior(m),
        m.D.prior_probabilities,
        1e-15,
        "current_transition_prior",
    )


def test_preview_cue_pmf_is_a_distribution(make_model):
    """The one-step cue pmf sums to one over the instantiated labels."""
    m = make_model(num_particles=4, max_contexts=3)
    pmf, labels = rc_context.preview_cue_pmf(m)
    assert pmf.shape == labels.shape
    assert np.array_equal(labels, np.arange(pmf.size))
    assert pmf.sum() == pytest.approx(1.0)


def test_next_trial_context_weights_without_a_cue(make_model):
    """With no cue the next-trial weights are just the transition prior."""
    m = make_model(num_particles=4, max_contexts=3)
    w = rc_context.next_trial_context_weights(m, None)
    assert_close(w, rc_context.current_transition_prior(m), 0.0, "no cue")


def test_next_trial_context_weights_clamps_an_out_of_range_label(make_model):
    """A label past the last instantiated column is clamped, not an error."""
    m = make_model(num_particles=3, max_contexts=2)
    w = rc_context.next_trial_context_weights(m, 99)
    assert w.shape == (3, 3)
    for p in range(3):
        assert w[p, :].sum() == pytest.approx(1.0)


def test_context_probability_vector_core_rejects_a_bad_kind():
    """An unrecognised kind is a clear error, not a silent empty result."""
    must_error(
        "bad kind",
        lambda: rc_context.context_probability_vector_core(
            "nope", lambda: None, lambda: None, lambda: None, [0.0, 1.0]
        ),
        "RealTimeCOIN:BadContextVectorKind",
    )


def test_context_probability_vector_core_counts(make_model):
    """The 'count' kind histograms 0-based labels into their own bins."""
    edges = np.arange(4) - 0.5      # bins for labels 0, 1, 2
    weights = rc_context.context_probability_vector_core(
        "count", lambda: None, lambda: None, lambda: np.array([0, 0, 2]), edges
    )
    assert_close(weights, np.array([2 / 3, 0.0, 1 / 3]), 1e-15, "count")


def test_context_probability_vector_core_means():
    """The 'predicted'/'responsibilities' kinds average over particles."""
    w = np.array([[0.2, 0.8], [0.4, 0.6]])
    assert_close(
        rc_context.context_probability_vector_core(
            "predicted", lambda: w, lambda: None, lambda: None, [0, 1, 2]
        ),
        np.array([0.3, 0.7]),
        1e-15,
        "predicted mean",
    )
    assert_close(
        rc_context.context_probability_vector_core(
            "responsibilities", lambda: None, lambda: w, lambda: None, [0, 1, 2]
        ),
        np.array([0.3, 0.7]),
        1e-15,
        "responsibilities mean",
    )


# ----------------------------------------------------------------------
# observe_y reaches the pipeline (implemented for the scalar path, stubbed
# for the multi-dimensional one)
# ----------------------------------------------------------------------


def test_observe_y_scalar_runs_the_b1_pipeline(make_model):
    """The scalar pipeline is unit B1's and now runs end to end."""
    m = make_model(num_particles=3, max_contexts=2)
    m.observe_y(0.5)
    # The whole pipeline ran, so the trial advanced.
    assert m.Trial == 1
    assert np.isfinite(m.motor_output())


def test_observe_y_md_runs_the_b2_pipeline(make_model):
    """The multi-dimensional pipeline is unit B2's - and it is implemented.

    Was ``test_observe_y_md_reaches_the_b2_stub``: it pinned the
    ``NotImplementedError`` while ``pipeline_md`` was a stub. Now that B2 has
    landed the same dispatch is asserted through its effect - the MD path runs
    end to end and advances the trial.
    """
    m = make_model(num_particles=3, max_contexts=2, state_dim=2)
    m.observe_y([0.5, -0.25])
    assert m.Trial == 1
    assert m.D.state_filtered_mean.shape == (3, 3, 2)
    assert np.all(np.isfinite(m.D.state_filtered_mean))


def test_observe_y_consumes_the_pending_cue(make_model):
    """The cue is resolved before the pipeline runs, so it is consumed."""
    m = make_model(num_particles=3, max_contexts=2)
    m.observe_q(11)
    m.observe_y(0.1)
    assert m.pending_q is None
    assert m.cue_values == [11.0]


def test_observe_y_rejects_a_wrong_length_y(make_model):
    """y must carry exactly state_dim elements."""
    m = make_model(num_particles=2, max_contexts=2, state_dim=3)
    must_error("wrong length", lambda: m.observe_y([1.0, 2.0]), "Size:notStateDim")
    m1 = make_model(num_particles=2, max_contexts=2)
    must_error("scalar model", lambda: m1.observe_y([1.0, 2.0]), "Size:notStateDim")


def test_observe_y_missing_observation_still_runs_the_pipeline(make_model):
    """None, nan and [] all mean 'missing', and the trial still runs.

    MATLAB's observe_y documents [] as the primary missing-observation form
    (isempty(y)), so an empty array must not be a size error.
    """
    m = make_model(num_particles=2, max_contexts=2)
    for i, missing in enumerate((None, float("nan"), [], np.array([]))):
        m.observe_y(missing)
        assert m.Trial == i + 1
    assert np.isfinite(m.motor_output())


def test_observe_y_md_accepts_an_empty_observation(make_model):
    """An empty y on the MD path is a fully-unobserved trial, not an error.

    Updated by unit B2: previously this pinned the ``pipeline_md`` stub error;
    now it checks the behaviour the stub stood in for - the trial runs, the
    counter advances, and with nothing observed the filtered moments are exactly
    the predicted ones.
    """
    m = make_model(num_particles=2, max_contexts=2, state_dim=3)
    for i, missing in enumerate((None, [], np.array([]))):
        m.observe_y(missing)
        assert m.Trial == i + 1
        np.testing.assert_array_equal(m.D.state_filtered_mean, m.D.state_mean)
        np.testing.assert_array_equal(m.D.state_filtered_cov, m.D.state_cov)


def test_query_stubs_name_their_owning_unit(make_model):
    """Each still-stubbed facade delegate reaches the module that will own it.

    Entries drop off this list as their unit lands; the pipeline and aligned
    context queries left it when units B1/B2/B3 were implemented and are
    covered by their own test files instead.
    """
    m = make_model(num_particles=2, max_contexts=2)
    for call, unit in [
        (lambda: m.state_probability([0.0, 1.0]), "unit C2"),
    ]:
        with pytest.raises(NotImplementedError, match=unit):
            call()


def test_b1_queries_are_implemented(make_model):
    """The unit B1 point-prediction queries no longer raise."""
    m = make_model(num_particles=2, max_contexts=2)
    assert np.isfinite(m.motor_output())
    mu, v = m.state_moments()
    assert np.isfinite(mu) and np.isfinite(v)


def test_deprecated_aliases_are_not_ported(make_model):
    """The four deprecated MATLAB aliases have no Python counterpart."""
    m = make_model(num_particles=2, max_contexts=2)
    for name in (
        "predicted_context_probabilities",
        "responsibilities",
        "context_predicted_probabilities",
        "context_responsibilities",
    ):
        assert not hasattr(m, name), name


# ----------------------------------------------------------------------
# from_state
# ----------------------------------------------------------------------


def _hand_built_state():
    """A small, fully specified scalar particle state (2 particles, 3 slots).

    The row prototypes below are hardcoded 3-vectors, so the shape is fixed;
    it is not parameterised on purpose.
    """
    p_count, c_slots = 2, 3
    return ParticleState(
        n_active=np.array([2, 1]),
        context=np.array([1, 0]),
        previous_context=np.array([0, 0]),
        n_cues=1,
        i_resampled=np.arange(p_count),
        n_context=np.zeros((p_count, c_slots, c_slots)),
        n_cue=np.zeros((p_count, c_slots, 1)),
        global_transition_probabilities=np.tile(
            np.array([0.6, 0.3, 0.1]), (p_count, 1)
        ),
        global_cue_probabilities=np.ones((p_count, 1)),
        prior_probabilities=np.tile(np.array([0.5, 0.4, 0.1]), (p_count, 1)),
        predicted_probabilities=np.tile(np.array([0.5, 0.4, 0.1]), (p_count, 1)),
        responsibilities=np.tile(np.array([0.7, 0.2, 0.1]), (p_count, 1)),
        probability_cue=np.ones((p_count, c_slots)),
        probability_state_feedback=np.ones((p_count, c_slots)),
        retention=np.full((p_count, c_slots), 0.9),
        drift=np.zeros((p_count, c_slots)),
        bias=np.zeros((p_count, c_slots)),
        state_mean=np.tile(np.array([1.0, -1.0, 0.0]), (p_count, 1)),
        state_var=np.full((p_count, c_slots), 0.01),
        state_filtered_mean=np.tile(np.array([1.0, -1.0, 0.0]), (p_count, 1)),
        state_filtered_var=np.full((p_count, c_slots), 0.01),
        state_feedback_mean=np.tile(np.array([1.0, -1.0, 0.0]), (p_count, 1)),
        state_feedback_var=np.full((p_count, c_slots), 0.02),
    )


def test_from_state_round_trip():
    """from_state installs the supplied state and bookkeeping verbatim."""
    p_count, c_slots = 2, 3
    d = _hand_built_state()
    properties = {"num_particles": p_count, "max_contexts": c_slots - 1}
    m = RealTimeCOIN.from_state(properties, d, trial=4, cue_values=(1, 7), rng=0)

    assert m.num_particles == p_count
    assert m.max_contexts == c_slots - 1
    assert m.Trial == 4
    assert m.cue_values == [1, 7]
    assert m.pending_q is None
    assert m.alignment_seed is None
    assert m.alignment_cache is None
    # invalidate_context_alignment bumps state_version past the restored trial.
    assert m.state_version == 5

    assert m.D is d
    assert np.array_equal(m.D.n_active, [2, 1])
    assert np.array_equal(m.D.context, [1, 0])
    assert_close(m.D.state_mean, d.state_mean, 0.0, "state_mean restored")
    assert_close(m.D.retention, np.full((p_count, c_slots), 0.9), 0.0, "retention")


def test_from_state_accepts_a_field_dict():
    """A partial {field: value} mapping is accepted; the rest stays None."""
    m = RealTimeCOIN.from_state(
        {"num_particles": 2, "max_contexts": 2},
        {
            "n_active": np.array([1, 1]),
            "context": np.array([0, 0]),
            "responsibilities": np.tile(np.array([1.0, 0.0, 0.0]), (2, 1)),
        },
        rng=0,
    )
    assert isinstance(m.D, ParticleState)
    assert np.array_equal(m.D.context, [0, 0])
    assert m.D.retention is None
    assert m.Trial == 1
    assert m.cue_values == [1]


def test_from_state_rejects_an_unknown_property():
    """Bad property keys surface as NameValuePairsError, as in the constructor."""
    with pytest.raises(NameValuePairsError):
        RealTimeCOIN.from_state({"nope": 1}, ParticleState(), rng=0)


def test_from_state_drives_predict_context():
    """A hand-built state is usable by the shared context step."""
    d = _hand_built_state()
    m = RealTimeCOIN.from_state(
        {"num_particles": 2, "max_contexts": 2}, d, rng=0
    )
    rc_context.predict_context(m, None)
    for p in range(2):
        assert m.D.predicted_probabilities[p, :].sum() == pytest.approx(1.0)

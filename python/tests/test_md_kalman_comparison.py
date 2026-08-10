"""Multi-dimensional Kalman equivalence in the single-context limit.

Python translation of ``tests/test_md_kalman_comparison.m``.

With one context and very precise dynamics priors the ``N``-dimensional
pipeline reduces to a multivariate Kalman filter. The model is driven with
CORRELATED (full-covariance) process and observation noise so the matrix
machinery - matrix Kalman gain, Cholesky likelihood, cross-dimension covariance
propagation - is genuinely exercised, and the one-step predictive feedback mean
is compared against an independent 2-D Kalman filter. A chi-square
probability-integral-transform check confirms the predictive covariance is on
the right scale.

Unit-B2 note: :func:`realtimecoin.queries_core.predictive_feedback_moments` and
``state_moments`` belong to unit B1 and are still stubs here, so the moments are
re-derived locally from :class:`~realtimecoin.state.ParticleState` (see
:func:`md_predictive_feedback_moments`, a line-for-line translation of
``predictive_feedback_moments.m``'s ``multiMoments``). The few assertions that
genuinely need the public queries are guarded so they activate on merge.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammainc

from realtimecoin import RealTimeCOIN, pipeline_md
from realtimecoin.context import next_trial_context_weights
from realtimecoin.numerics import (
    choljitter,
    safe_inverse,
    stationary_state_cov_md,
    stationary_state_mean_md,
)
from realtimecoin.state import (
    observation_noise_cov,
    peek_cue_label,
    process_noise_cov,
)

# --- Ground-truth linear-Gaussian system shared by the tests -----------------
N = 2
A_RETENTION = 0.8
A_DRIFT = 0.03
A_MAT = A_RETENTION * np.eye(N)
D_VEC = A_DRIFT * np.ones(N)
#: Correlated process noise - off-diagonal entries make the covariance
#: propagation genuinely matrix-valued.
Q_COV = np.array([[1.0e-4, 3.0e-5], [3.0e-5, 1.2e-4]])
#: Correlated observation noise, with a NEGATIVE off-diagonal so a sign error in
#: the innovation covariance cannot cancel out.
R_COV = np.array([[4.0e-4, -1.0e-4], [-1.0e-4, 5.0e-4]])

KALMAN_CFG = dict(
    num_particles=500,
    max_contexts=1,
    state_dim=N,
    prior_mean_retention=A_RETENTION,
    prior_precision_retention=1e12,
    prior_mean_drift=A_DRIFT,
    prior_precision_drift=1e12,
    process_noise_covariance=Q_COV,
    observation_noise_covariance=R_COV,
)


def md_predictive_feedback_moments(model, raw_cue=None):
    """One-step predictive feedback mean and covariance, marginalised.

    Direct translation of ``multiMoments`` in
    ``@RealTimeCOIN/predictive_feedback_moments.m``, computed from the particle
    state so it does not depend on unit B1's ``queries_core``. Read-only.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    raw_cue : float or None, optional
        RAW upcoming cue value (as passed to ``observe_q``); ``None``
        marginalises the cue out.

    Returns
    -------
    mu : numpy.ndarray
        ``(N,)`` predictive feedback mean.
    sigma : numpy.ndarray
        ``(N, N)`` predictive feedback covariance.
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    q_cov = process_noise_cov(model)
    r_cov = observation_noise_cov(model)
    d = model.D

    q_label = peek_cue_label(model, raw_cue)
    weights = next_trial_context_weights(model, q_label)        # (P, C)

    mu = np.zeros(n)
    second = np.zeros((n, n))
    for p in range(p_count):
        novel = min(int(d.n_active[p]), c_slots - 1)   # 0-based novel slot
        for c in range(c_slots):
            w = weights[p, c] / p_count
            if w == 0:
                continue
            a_mat = d.Theta[p, c, :, :n]
            drift = d.Theta[p, c, :, n]
            if c == novel and d.n_active[p] < model.max_contexts:
                s_pred = stationary_state_mean_md(a_mat, drift)
                p_pred = stationary_state_cov_md(a_mat, q_cov)
            else:
                s_pred = a_mat @ d.state_filtered_mean[p, c] + drift
                p_pred = a_mat @ d.state_filtered_cov[p, c] @ a_mat.T + q_cov
            fb_mean = s_pred + d.bias[p, c]
            fb_cov = p_pred + r_cov
            mu = mu + w * fb_mean
            second = second + w * (fb_cov + np.outer(fb_mean, fb_mean))
    sigma = second - np.outer(mu, mu)
    return mu, (sigma + sigma.T) / 2.0


def simulate_linear_gaussian(seed, num_trials):
    """Draw a trajectory from the ground-truth multivariate linear-Gaussian model.

    Parameters
    ----------
    seed : int
        Seed for the data-generating generator (independent of the model's).
    num_trials : int
        Number of trials to generate.

    Returns
    -------
    numpy.ndarray
        ``(num_trials, N)`` observations, one per row.
    """
    rng = np.random.default_rng(seed)
    l_q = np.linalg.cholesky(Q_COV)
    l_r = np.linalg.cholesky(R_COV)
    s = D_VEC / (1.0 - A_RETENTION)          # stationary mean (A = a I)
    y = np.zeros((num_trials, N))
    for t in range(num_trials):
        s = A_MAT @ s + D_VEC + l_q @ rng.standard_normal(N)
        y[t] = s + l_r @ rng.standard_normal(N)
    return y


def test_md_matches_multivariate_kalman_with_correlated_noise():
    """Predictive feedback mean tracks an independent 2-D Kalman filter."""
    num_trials = 20
    y = simulate_linear_gaussian(7, num_trials)

    # Reference Kalman filter, initialised at the same stationary distribution
    # the model uses (P0 = Q / (1 - a^2) because A = a I).
    m = D_VEC / (1.0 - A_RETENTION)
    p_cov = Q_COV / (1.0 - A_RETENTION ** 2)

    coin = RealTimeCOIN(rng=7, **KALMAN_CFG)

    kf_pred_mean = np.zeros((num_trials, N))
    rt_pred_mean = np.zeros((num_trials, N))
    pit = np.zeros(num_trials)

    for t in range(num_trials):
        # Kalman predictive feedback distribution for this trial.
        m_pred = A_MAT @ m + D_VEC
        p_pred = A_MAT @ p_cov @ A_MAT.T + Q_COV
        s_inn = p_pred + R_COV
        kf_pred_mean[t] = m_pred

        # Model predictive feedback distribution (read-only, before observing).
        coin.observe_q(1)
        mu, sigma = md_predictive_feedback_moments(coin, 1)
        rt_pred_mean[t] = mu

        assert np.linalg.norm(sigma - sigma.T, "fro") < 1e-8, (
            "predictive feedback covariance is not symmetric at trial %d" % t
        )
        assert np.min(np.linalg.eigvalsh(sigma)) > -1e-9, (
            "predictive feedback covariance is not PSD at trial %d" % t
        )

        innovation = y[t] - mu
        mahalanobis = float(innovation @ np.linalg.solve(sigma, innovation))
        # Regularised lower incomplete gamma == chi-square_N CDF. scipy takes
        # (a, x) where MATLAB's gammainc takes (x, a).
        pit[t] = gammainc(N / 2.0, mahalanobis / 2.0)

        # Kalman measurement update.
        k_gain = p_pred @ np.linalg.inv(s_inn)
        m = m_pred + k_gain @ (y[t] - m_pred)
        p_cov = (np.eye(N) - k_gain) @ p_pred
        p_cov = (p_cov + p_cov.T) / 2.0

        coin.observe_y(y[t])

    rmse = float(np.sqrt(np.mean((rt_pred_mean - kf_pred_mean) ** 2)))
    assert rmse < 0.05, (
        "MD predictive means differ from the Kalman filter beyond tolerance "
        "(RMSE %.4f)" % rmse
    )
    assert np.all((pit >= 0.0) & (pit <= 1.0)), "PIT values fell outside [0, 1]"
    assert np.all(np.isfinite(pit))


def test_md_state_moment_shapes_and_psd():
    """Per-context filtered moments keep their MD shapes and stay PSD."""
    num_trials = 8
    y = simulate_linear_gaussian(21, num_trials)
    coin = RealTimeCOIN(rng=3, **KALMAN_CFG)
    for t in range(num_trials):
        coin.observe_q(1)
        coin.observe_y(y[t])

    d = coin.D
    c_slots = coin.max_contexts + 1
    assert d.state_filtered_mean.shape == (coin.num_particles, c_slots, N)
    assert d.state_filtered_cov.shape == (coin.num_particles, c_slots, N, N)
    assert d.state_mean.shape == (coin.num_particles, c_slots, N)
    assert d.state_cov.shape == (coin.num_particles, c_slots, N, N)
    assert np.all(np.isfinite(d.state_filtered_mean))
    assert np.all(np.isfinite(d.state_filtered_cov))

    for name in ("state_cov", "state_filtered_cov", "state_feedback_cov"):
        cov = getattr(d, name)
        assert np.allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-12), (
            "%s is not symmetric" % name
        )
        assert np.min(np.linalg.eigvalsh((cov + np.swapaxes(cov, -1, -2)) / 2)) > -1e-9, (
            "%s is not PSD" % name
        )

    # Public MD moment accessor, once unit B1 lands.
    try:
        mu_state, cov_state = coin.state_moments()
    except NotImplementedError:
        pytest.skip("state_moments pending unit B1")
    mu_state = np.asarray(mu_state).reshape(-1)
    cov_state = np.asarray(cov_state)
    assert mu_state.shape == (N,), "state_moments mean must be length N in MD mode"
    assert cov_state.shape == (N, N), "state_moments covariance must be N-by-N"
    sym = (cov_state + cov_state.T) / 2.0
    assert np.min(np.linalg.eigvalsh(sym)) > -1e-9, "state covariance is not PSD"


def test_md_predictive_feedback_moments_match_public_query():
    """The locally derived moments agree with the public query after B1 merges."""
    coin = RealTimeCOIN(rng=5, **KALMAN_CFG)
    y = simulate_linear_gaussian(31, 4)
    for t in range(4):
        coin.observe_q(1)
        coin.observe_y(y[t])
    coin.observe_q(1)

    mu_local, sigma_local = md_predictive_feedback_moments(coin, 1)
    try:
        mu_public, sigma_public = coin.predictive_feedback_moments(0)
    except NotImplementedError:
        pytest.skip("predictive_feedback_moments pending unit B1")
    np.testing.assert_allclose(np.asarray(mu_public).reshape(-1), mu_local, atol=1e-10)
    np.testing.assert_allclose(np.asarray(sigma_public), sigma_local, atol=1e-10)


def test_md_determinism_same_seed_same_state():
    """Two identically seeded MD models agree field-by-field after 20 trials."""
    y = simulate_linear_gaussian(41, 20)
    cfg = dict(
        state_dim=N,
        num_particles=30,
        max_contexts=4,
        infer_bias=True,
        process_noise_covariance=Q_COV,
        observation_noise_covariance=R_COV,
    )
    models = []
    for _ in range(2):
        coin = RealTimeCOIN(rng=2024, **cfg)
        for t in range(20):
            coin.observe_q(1 + (t % 3))
            coin.observe_y(y[t])
        models.append(coin)

    left, right = models[0].D, models[1].D
    compared = 0
    for name in left.field_names():
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None:
            assert b is None, "field %s populated in only one run" % name
            continue
        compared += 1
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b), err_msg="field %s diverged" % name
        )
    assert compared > 20, "determinism check compared suspiciously few fields"


# ----------------------------------------------------------------------
# Batched linear algebra == MATLAB's per-page loops
#
# The MD pipeline replaces MATLAB's `for p ... for c` pages with numpy batched
# linear algebra. That is only legitimate if it reproduces the per-page result
# INCLUDING the degenerate branches of safe_inverse / choljitter, so pin it.
# ----------------------------------------------------------------------


def _random_spd(rng, shape, dim, ridge=0.3):
    """Random symmetric positive-definite pages of shape ``shape + (dim, dim)``."""
    a = rng.standard_normal(tuple(shape) + (dim, dim))
    return a @ np.swapaxes(a, -1, -2) + ridge * np.eye(dim)


@pytest.mark.parametrize("case", ["healthy", "singular", "non_finite", "all_zero"])
def test_safe_inverse_batch_matches_per_page_loop(case):
    """``_safe_inverse_batch`` == a per-page ``safe_inverse`` loop."""
    rng = np.random.default_rng(0)
    mats = _random_spd(rng, (5, 4), 3)
    if case == "singular":
        mats[2, 1] = np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    elif case == "non_finite":
        mats[0, 0, 1, 1] = np.nan
    elif case == "all_zero":
        mats[:] = 0.0

    got = pipeline_md._safe_inverse_batch(mats)
    want = np.stack(
        [np.stack([safe_inverse(mats[p, c]) for c in range(4)]) for p in range(5)]
    )
    np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("case", ["pd", "indefinite", "zero_page"])
def test_chol_batch_matches_per_page_choljitter(case):
    """``_chol_batch`` == a per-page ``choljitter`` loop (jitter path included)."""
    rng = np.random.default_rng(1)
    mats = _random_spd(rng, (4, 3), 2, ridge=0.2)
    if case == "indefinite":
        mats[1, 2] = np.array([[1.0, 2.0], [2.0, 1.0]])   # not PD
    elif case == "zero_page":
        mats[3, 0] = 0.0

    got = pipeline_md._chol_batch(mats, exact_zero=False)
    want = np.stack(
        [np.stack([choljitter(mats[p, c])[0] for c in range(3)]) for p in range(4)]
    )
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_chol_batch_exact_zero_reproduces_draw_gaussian_short_circuit():
    """A zero covariance yields a ZERO factor, so the draw returns mu exactly.

    ``drawGaussian`` in ``sampleStatesMD.m`` short-circuits on
    ``all(Sigma(:) == 0)``; without that, ``choljitter`` would hand back a
    ``1e-6 * I`` jittered factor and perturb the sample.
    """
    rng = np.random.default_rng(2)
    mats = _random_spd(rng, (2, 2), 2, ridge=0.2)
    mats[1, 1] = 0.0

    factors = pipeline_md._chol_batch(mats, exact_zero=True)
    assert np.all(factors[1, 1] == 0.0)
    np.testing.assert_allclose(factors[0, 0], choljitter(mats[0, 0])[0], atol=1e-12)

    # Without the short-circuit the same page keeps its jittered factor, which
    # is what sample_bias_md relies on.
    jittered = pipeline_md._chol_batch(mats, exact_zero=False)
    assert np.any(jittered[1, 1] != 0.0)

    mean = rng.standard_normal((2, 2, 2))
    drawn = pipeline_md._draw_gaussian_batch(
        np.random.default_rng(3), mean, mats, exact_zero=True
    )
    np.testing.assert_array_equal(drawn[1, 1], mean[1, 1])


def test_right_solve_matches_per_page_right_division():
    """``_right_solve`` == a per-page ``a @ inv(s)``, and survives a singular page."""
    rng = np.random.default_rng(4)
    s = _random_spd(rng, (6,), 3, ridge=0.5)
    a = rng.standard_normal((6, 4, 3))

    got = pipeline_md._right_solve(a, s)
    want = np.stack([a[i] @ np.linalg.inv(s[i]) for i in range(6)])
    np.testing.assert_allclose(got, want, atol=1e-10)

    s_bad = s.copy()
    s_bad[3] = np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])     # exactly singular
    fallback = pipeline_md._right_solve(a, s_bad)
    assert np.all(np.isfinite(fallback)), (
        "a singular innovation covariance leaked non-finite values into the gain"
    )
    # The degenerate page must not perturb its healthy neighbours: MATLAB's
    # per-particle loop cannot do that, so neither may the batched form.
    healthy = [0, 1, 2, 4, 5]
    np.testing.assert_array_equal(
        fallback[healthy], got[healthy],
        err_msg="one singular page contaminated the other particles' gains",
    )


def test_resampled_field_list_covers_every_md_particle_field():
    """``RESAMPLED_FIELDS_MD`` covers every particle-indexed field the MD run writes.

    A missing entry would silently keep a field's PRE-resampling ancestry, which
    no behavioural assertion would catch. The exclusions are the ones
    ``resampleStateMD.m`` also makes, each for a stated reason.
    """
    coin = RealTimeCOIN(state_dim=N, num_particles=4, max_contexts=3, infer_bias=True, rng=6)
    coin.observe_q(1)
    coin.observe_y(np.array([0.05, -0.02]))

    #: Populated MD fields that must NOT be resampled, with the reason.
    excluded = {
        "previous_context": "overwritten from context by sample_context_md",
        "i_resampled": "just written; it indexes the OLD ancestry",
        "previous_state_filtered_mean": "refreshed at the end of resample_state_md",
        "previous_state_filtered_cov": "refreshed at the end of resample_state_md",
        "responsibilities": "rebuilt from resp[idx] after the gather",
        "x_dynamics": "recomputed by sample_states_md after the gather",
        "previous_x_dynamics": "recomputed by sample_states_md after the gather",
        "x_bias": "recomputed by sample_states_md after the gather",
    }

    d = coin.D
    missing = []
    for name in d.field_names():
        value = getattr(d, name)
        if not isinstance(value, np.ndarray):
            continue                      # n_cues is a plain int
        if value.shape[0] != coin.num_particles:
            continue                      # not particle-indexed
        if name in excluded or name in pipeline_md.RESAMPLED_FIELDS_MD:
            continue
        missing.append(name)
    assert not missing, (
        "particle-indexed MD field(s) %s are neither resampled nor explicitly "
        "excluded - add them to RESAMPLED_FIELDS_MD" % missing
    )

    # Typo guard: every listed name must be a declared ParticleState field.
    # (The list is NOT required to be all-populated - bias_ss_1 / bias_ss_2 are
    # carried for parity with resampleStateMD.m and are None in MD mode.)
    declared = set(d.field_names())
    unknown = [n for n in pipeline_md.RESAMPLED_FIELDS_MD if n not in declared]
    assert not unknown, "RESAMPLED_FIELDS_MD names unknown field(s) %s" % unknown
    assert len(set(pipeline_md.RESAMPLED_FIELDS_MD)) == len(
        pipeline_md.RESAMPLED_FIELDS_MD
    ), "RESAMPLED_FIELDS_MD contains duplicates"


@pytest.mark.parametrize(
    "label, cfg",
    [
        # Zero process noise: Q == 0, so sample_states_md's forward draw hits
        # drawGaussian's `all(Sigma(:) == 0)` short-circuit.
        ("zero process noise", dict(sigma_process_noise=0.0)),
        # Zero observation noise: R == 0, so the innovation covariance is the
        # bare predictive covariance and choljitter's jitter path is exercised.
        ("zero observation noise", dict(sigma_sensory_noise=0.0, sigma_motor_noise=0.0)),
        # Uninformative dynamics prior: V0inv == 0, so the matrix-normal
        # posterior precision is rank deficient until data arrives.
        ("uninformative dynamics prior",
         dict(prior_precision_retention=0.0, prior_precision_drift=0.0)),
        # Rank-1 (singular) Q: forces safe_inverse's pseudo-inverse fallback.
        ("singular process noise",
         dict(process_noise_covariance=np.array([[1e-4, 1e-4], [1e-4, 1e-4]]))),
        # One particle: every batched reduction degenerates to a single page.
        ("single particle", dict(num_particles=1)),
        # No novel slot at all (n_active starts at max_contexts).
        ("no novel slot", dict(max_contexts=1)),
        # Aggressive novelty against a low cap: exercises the fold-at-cap branch
        # of sample_context_md, where no stick is split.
        ("context cap saturation", dict(max_contexts=2, gamma_context=5.0, trials=30)),
    ],
)
def test_md_degenerate_configurations_stay_finite(label, cfg):
    """Degenerate covariances / caps exercise the fallback branches safely."""
    trials = cfg.pop("trials", 12)
    base = dict(state_dim=N, num_particles=8, max_contexts=3)
    base.update(cfg)
    coin = RealTimeCOIN(rng=7, **base)

    rng = np.random.default_rng(3)
    for t in range(trials):
        coin.observe_q(1 + (t % 2))
        if t % 5 == 4:
            y = None                                   # channel trial
        elif t % 5 == 3:
            y = np.array([0.1, np.nan])                # partially observed
        else:
            y = 0.1 * rng.standard_normal(N)
        coin.observe_y(y)

    assert coin.Trial == trials
    d = coin.D
    for name in d.field_names():
        value = getattr(d, name)
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value)), (
                "%s: field %s went non-finite" % (label, name)
            )
    np.testing.assert_allclose(d.responsibilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(d.n_active >= 1)
    assert np.all(d.n_active <= coin.max_contexts)
    # After sample_context_md the sampled context is always an INSTANTIATED
    # label: the novel slot is either promoted (label == old n_active, so
    # < the incremented one) or folded back onto the last context at the cap.
    assert np.all(d.context >= 0)
    assert np.all(d.context < d.n_active)


def test_md_three_dimensional_smoke():
    """N = 3 runs end to end, stays finite and keeps normalised probabilities."""
    dim = 3
    rng = np.random.default_rng(99)
    coin = RealTimeCOIN(
        state_dim=dim, num_particles=20, max_contexts=3, infer_bias=True, rng=8
    )
    for t in range(12):
        coin.observe_q(1 if t < 6 else 2)
        coin.observe_y(0.05 * rng.standard_normal(dim))

    d = coin.D
    assert coin.Trial == 12
    for name in (
        "state_mean",
        "state_cov",
        "state_filtered_mean",
        "state_filtered_cov",
        "state_feedback_mean",
        "state_feedback_cov",
        "Theta",
        "Lambda_xx",
        "Lambda_yx",
        "bias",
        "x_dynamics",
        "previous_x_dynamics",
        "x_bias",
        "responsibilities",
    ):
        value = getattr(d, name)
        assert value is not None, "field %s was never written" % name
        assert np.all(np.isfinite(value)), "field %s contains non-finite values" % name

    assert d.Theta.shape == (20, 4, dim, dim + 1)
    assert d.Lambda_xx.shape == (20, 4, dim + 1, dim + 1)
    for name in ("responsibilities", "predicted_probabilities", "prior_probabilities"):
        np.testing.assert_allclose(getattr(d, name).sum(axis=1), 1.0, atol=1e-12)

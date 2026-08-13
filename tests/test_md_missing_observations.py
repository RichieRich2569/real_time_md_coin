"""Multi-dimensional ``nan`` feedback handling.

Python translation of ``tests/test_md_missing_observations.m``.

Three properties are pinned:

1. an all-``nan`` feedback vector is EXACTLY the same channel trial as ``None``
   under an identical seed (bit-for-bit over the whole particle state);
2. mixed missingness uses the observed coordinates, keeps every field finite and
   leaves the responsibilities normalised per particle;
3. the partial-observation Kalman update matches a masked Joseph-form oracle,
   and an unobserved dimension really does move through the cross-covariance
   when the noise is correlated (and essentially does not when it is not).

Unit-B2 note: the MATLAB original reads the particle state through
``diagnostics().raw``; ``diagnostics`` belongs to unit B3, so the state is read
straight off ``model.D`` here (the identical object ``raw`` aliases). The public
predictive-moment query is unit B1's, so the predictive mean is re-derived by
:func:`tests.test_md_kalman_comparison.md_predictive_feedback_moments`.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOIN
from test_md_kalman_comparison import md_predictive_feedback_moments

N = 2
A_RETENTION = 0.8
A_DRIFT = 0.03
A_MAT = A_RETENTION * np.eye(N)
D_VEC = A_DRIFT * np.ones(N)
Q_COV = np.array([[1.0e-4, 3.0e-5], [3.0e-5, 1.2e-4]])
R_COV = np.array([[4.0e-4, -1.0e-4], [-1.0e-4, 5.0e-4]])

#: Config shared by the equivalence / mixed-missingness cases.
MISSING_CFG = dict(
    state_dim=2,
    num_particles=40,
    max_contexts=3,
    infer_bias=True,
    process_noise_covariance=np.array([[1.0e-4, 2.0e-5], [2.0e-5, 1.1e-4]]),
    observation_noise_covariance=np.array([[4.0e-4, -1.0e-4], [-1.0e-4, 5.0e-4]]),
)

#: Fields the MATLAB helper ``assertFiniteMD`` checks.
FINITE_FIELDS = (
    "responsibilities",
    "state_filtered_mean",
    "state_filtered_cov",
    "state_mean",
    "state_cov",
    "state_feedback_mean",
    "state_feedback_cov",
    "bias",
    "bias_info_ss",
    "bias_precision_ss",
    "probability_state_feedback",
)


def assert_finite_md(state):
    """Assert every populated MD field of ``state`` is finite."""
    for name in FINITE_FIELDS:
        value = getattr(state, name, None)
        if value is None:
            continue
        assert np.all(np.isfinite(value)), (
            "field %s contains non-finite values" % name
        )


def test_all_nan_feedback_equals_empty_feedback():
    """``[nan, nan]`` is bit-for-bit the same trial as ``None``."""
    coin_empty = RealTimeCOIN(rng=12, **MISSING_CFG)
    coin_empty.observe_q(3)
    coin_empty.observe_y(None)

    coin_nan = RealTimeCOIN(rng=12, **MISSING_CFG)
    coin_nan.observe_q(3)
    coin_nan.observe_y(np.array([np.nan, np.nan]))

    assert coin_nan.Trial == 1, "all-nan trial did not advance the trial counter"
    assert coin_empty.Trial == 1

    left, right = coin_nan.D, coin_empty.D
    compared = 0
    for name in left.field_names():
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None:
            assert b is None, "field %s populated in only one run" % name
            continue
        compared += 1
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg="all-nan run diverged from the empty-feedback run on %s" % name,
        )
    assert compared > 20, "equivalence check compared suspiciously few fields"
    assert_finite_md(left)


def test_mixed_missingness_stays_finite_and_normalised():
    """Per-dimension missingness keeps every field finite and normalised."""
    coin = RealTimeCOIN(rng=13, **MISSING_CFG)
    coin.observe_y(np.array([0.1, np.nan]))
    coin.observe_y(np.array([np.nan, -0.2]))

    d = coin.D
    assert_finite_md(d)
    assert coin.Trial == 2
    total = float(np.sum(d.responsibilities))
    assert abs(total - coin.num_particles) < 1e-9, (
        "mixed-nan responsibilities are not normalised per particle (sum %.12f)"
        % total
    )
    # The bias statistics must have accumulated on the observed coordinate only.
    assert np.all(np.isfinite(d.bias_info_ss))
    assert np.all(np.isfinite(d.bias_precision_ss))
    assert np.any(d.bias_precision_ss != 0.0), (
        "no bias precision accumulated from the partially observed trials"
    )


def test_partial_observation_matches_masked_kalman_oracle():
    """Per-trial masks track a Joseph-form partial-observation Kalman filter."""
    num_trials = 18
    # MATLAB's `masks` matrix, one ROW per trial here.
    masks = np.array(
        [
            [1, 1], [1, 0], [0, 1], [0, 0], [1, 1], [1, 0],
            [0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [0, 0],
            [1, 1], [1, 0], [0, 1], [0, 0], [1, 1], [1, 0],
        ],
        dtype=bool,
    )
    assert masks.shape == (num_trials, N)

    rng = np.random.default_rng(14)
    l_q = np.linalg.cholesky(Q_COV)
    l_r = np.linalg.cholesky(R_COV)
    s = D_VEC / (1.0 - A_RETENTION)
    y_full = np.zeros((num_trials, N))
    for t in range(num_trials):
        s = A_MAT @ s + D_VEC + l_q @ rng.standard_normal(N)
        y_full[t] = s + l_r @ rng.standard_normal(N)

    m = D_VEC / (1.0 - A_RETENTION)
    p_cov = Q_COV / (1.0 - A_RETENTION ** 2)

    coin = RealTimeCOIN(
        num_particles=500,
        max_contexts=1,
        state_dim=N,
        prior_mean_retention=A_RETENTION,
        prior_precision_retention=1e12,
        prior_mean_drift=A_DRIFT,
        prior_precision_drift=1e12,
        process_noise_covariance=Q_COV,
        observation_noise_covariance=R_COV,
        rng=15,
    )

    kf_pred_mean = np.zeros((num_trials, N))
    rt_pred_mean = np.zeros((num_trials, N))

    for t in range(num_trials):
        m_pred = A_MAT @ m + D_VEC
        p_pred = A_MAT @ p_cov @ A_MAT.T + Q_COV
        kf_pred_mean[t] = m_pred

        coin.observe_q(1)
        mu, sigma = md_predictive_feedback_moments(coin, 1)
        rt_pred_mean[t] = mu
        sym = (sigma + sigma.T) / 2.0
        assert np.min(np.linalg.eigvalsh(sym)) > -1e-9, (
            "partial-observation predictive covariance is not PSD at trial %d" % t
        )

        obs_mask = masks[t]
        y_obs = y_full[t].copy()
        y_obs[~obs_mask] = np.nan
        if np.any(obs_mask):
            obs_idx = np.flatnonzero(obs_mask)
            s_inn = (
                p_pred[np.ix_(obs_idx, obs_idx)] + R_COV[np.ix_(obs_idx, obs_idx)]
            )
            k_gain = p_pred[:, obs_idx] @ np.linalg.inv(s_inn)
            innovation = y_full[t][obs_idx] - m_pred[obs_idx]
            kh = np.zeros((N, N))
            kh[:, obs_idx] = k_gain
            i_kh = np.eye(N) - kh
            m = m_pred + k_gain @ innovation
            # Joseph form, restricted to the observed sub-block.
            p_cov = (
                i_kh @ p_pred @ i_kh.T
                + k_gain @ R_COV[np.ix_(obs_idx, obs_idx)] @ k_gain.T
            )
            p_cov = (p_cov + p_cov.T) / 2.0
        else:
            m = m_pred
            p_cov = p_pred

        coin.observe_y(y_obs)

    rmse = float(np.sqrt(np.mean((rt_pred_mean - kf_pred_mean) ** 2)))
    assert rmse < 0.06, (
        "partial-observation MD Kalman predictive means differ too much "
        "(RMSE %.4f)" % rmse
    )


def _unobserved_delta(q_cov):
    """Shift of the UNOBSERVED dimension caused by observing dimension 0.

    Runs two single-particle models from the same seed: one channel trial
    (nothing observed) and one trial observing dimension 0 only. Returns how far
    apart their filtered means are on dimension 1 (context 0, particle 0).

    Parameters
    ----------
    q_cov : numpy.ndarray
        ``(N, N)`` process-noise covariance to run with.

    Returns
    -------
    float
        ``abs`` difference of the two filtered means on the unobserved
        dimension.
    """
    cfg = dict(
        num_particles=1,
        max_contexts=1,
        state_dim=N,
        prior_mean_retention=0.8,
        prior_precision_retention=1e12,
        prior_mean_drift=0.0,
        prior_precision_drift=1e12,
        process_noise_covariance=q_cov,
        observation_noise_covariance=R_COV,
    )
    coin_channel = RealTimeCOIN(rng=16, **cfg)
    coin_channel.observe_y(None)

    coin_partial = RealTimeCOIN(rng=16, **cfg)
    coin_partial.observe_y(np.array([0.25, np.nan]))

    # [particle 0, context 0, dimension 1] - MATLAB's raw(2, 1, 1).
    return abs(
        float(coin_partial.D.state_filtered_mean[0, 0, 1])
        - float(coin_channel.D.state_filtered_mean[0, 0, 1])
    )


def test_correlated_noise_moves_the_unobserved_dimension():
    """The unobserved dimension updates through the cross-covariance."""
    # Strongly correlated process noise: dimension 1 must move.
    correlated = _unobserved_delta(np.array([[1.0e-4, 7.0e-5], [7.0e-5, 1.2e-4]]))
    assert correlated > 1e-8, (
        "unobserved dimension did not update differently from a channel trial"
    )
    assert correlated > 1e-3, (
        "cross-covariance correction is implausibly small (%.3g)" % correlated
    )

    # Diagonal process noise (and the observation noise block that enters the
    # gain is a scalar here): the cross-covariance is ~0, so dimension 1 should
    # barely move at all.
    uncorrelated = _unobserved_delta(np.diag([1.0e-4, 1.2e-4]))
    assert uncorrelated < 1e-5, (
        "uncorrelated run leaked a large correction into the unobserved "
        "dimension (%.3g)" % uncorrelated
    )
    assert correlated > 1000.0 * uncorrelated, (
        "correlated (%.3g) and uncorrelated (%.3g) unobserved-dimension updates "
        "are not clearly separated" % (correlated, uncorrelated)
    )


def test_all_missing_trial_still_advances_and_predicts():
    """A fully missing MD trial predicts without correcting, and still counts."""
    coin = RealTimeCOIN(rng=17, **MISSING_CFG)
    coin.observe_y(np.array([0.05, 0.02]))
    before = coin.D.state_mean.copy()
    coin.observe_y(None)
    d = coin.D
    assert coin.Trial == 2
    # With no observation the filtered moments are exactly the predicted ones.
    np.testing.assert_array_equal(d.state_filtered_mean, d.state_mean)
    np.testing.assert_array_equal(d.state_filtered_cov, d.state_cov)
    assert np.all(np.isfinite(before))
    assert_finite_md(d)
    np.testing.assert_allclose(d.responsibilities.sum(axis=1), 1.0, atol=1e-12)

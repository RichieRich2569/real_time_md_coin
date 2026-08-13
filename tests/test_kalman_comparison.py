"""Scalar single-context limit versus a hand-written 1-D Kalman filter.

Python port of ``tests/test_kalman_comparison.m``. With ``max_contexts = 1`` and
priors tight enough to pin the dynamics, RealTimeCOIN degenerates to an ordinary
scalar Kalman filter, so its posterior state mean must track a directly coded
filter that knows the true parameters.

Deviation from the MATLAB source: that test recovers the posterior mean by
trapezoid-integrating ``state_probability`` over a grid.
``state_probability`` is documented as the Gaussian mixture over
``responsibilities`` and the FILTERED state moments, so the same mean is
computed here in closed form from those two arrays - and, as a second, weaker
check, from :meth:`RealTimeCOIN.state_moments` (which mixes the PREDICTED
moments and therefore lags the filtered mean by one Kalman correction). The
MATLAB tolerance of 0.15 is kept for both.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOIN

#: Tolerance from the MATLAB test; the latent state lives on a ~0.25 scale, so
#: this admits a Kalman-correction-sized lag but not a structural error.
TOLERANCE = 0.15

A_TRUE = 0.8
D_TRUE = 0.05
Q_STD = 0.01
R_STD = 0.02
NUM_TRIALS = 8


def _synthetic_run(rng):
    """Generate the AR(1)-plus-noise observation sequence of the MATLAB test.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source for the process and observation noise.

    Returns
    -------
    numpy.ndarray
        ``(NUM_TRIALS,)`` observed feedback.
    """
    s = 0.0
    y = np.zeros(NUM_TRIALS)
    for t in range(NUM_TRIALS):
        s = A_TRUE * s + D_TRUE + Q_STD * rng.standard_normal()
        y[t] = s + R_STD * rng.standard_normal()
    return y


def _reference_kalman(y):
    """Filtered means of a scalar Kalman filter that knows the true parameters.

    Parameters
    ----------
    y : array_like
        Observation sequence.

    Returns
    -------
    numpy.ndarray
        ``(len(y),)`` posterior (filtered) state means.
    """
    # Start at the stationary distribution of the true AR(1), as the model does.
    m = D_TRUE / (1.0 - A_TRUE)
    p = Q_STD ** 2 / (1.0 - A_TRUE ** 2)
    means = np.zeros(len(y))
    for t, y_t in enumerate(y):
        m_pred = A_TRUE * m + D_TRUE
        p_pred = A_TRUE ** 2 * p + Q_STD ** 2
        gain = p_pred / (p_pred + R_STD ** 2)
        m = m_pred + gain * (y_t - m_pred)
        p = (1.0 - gain) * p_pred
        means[t] = m
    return means


def _single_context_model():
    """Build the pinned single-context model of the MATLAB test.

    Returns
    -------
    RealTimeCOIN
        A scalar model with one context and near-deterministic dynamics.
    """
    return RealTimeCOIN(
        num_particles=100,
        max_contexts=1,
        prior_mean_retention=A_TRUE,
        prior_precision_retention=1e12,
        prior_mean_drift=D_TRUE,
        prior_precision_drift=1e12,
        sigma_process_noise=Q_STD,
        sigma_sensory_noise=R_STD,
        sigma_motor_noise=0.0,
        rng=2,
    )


def _filtered_mixture_mean(model):
    """Posterior state mean as ``state_probability`` would integrate it.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float
        ``sum_{p,c} responsibilities[p, c] * state_filtered_mean[p, c] / P``.
    """
    d = model.D
    return float(
        np.sum(d.responsibilities * d.state_filtered_mean) / model.num_particles
    )


def test_matches_scalar_kalman_filter():
    """The single-context posterior mean tracks a true-parameter Kalman filter."""
    y = _synthetic_run(np.random.default_rng(2))
    kf_means = _reference_kalman(y)

    model = _single_context_model()
    filtered = np.zeros(NUM_TRIALS)
    predicted = np.zeros(NUM_TRIALS)
    for t in range(NUM_TRIALS):
        model.observe_q(1)
        model.observe_y(y[t])
        filtered[t] = _filtered_mixture_mean(model)
        predicted[t] = model.state_moments()[0]

    assert np.all(np.isfinite(filtered))
    assert np.all(np.isfinite(predicted))
    np.testing.assert_allclose(filtered, kf_means, atol=TOLERANCE)
    np.testing.assert_allclose(predicted, kf_means, atol=TOLERANCE)


def test_single_context_never_instantiates_a_second_context():
    """With ``max_contexts = 1`` every particle stays in the single context."""
    y = _synthetic_run(np.random.default_rng(7))
    model = _single_context_model()
    for y_t in y:
        model.observe_q(1)
        model.observe_y(y_t)
    assert np.all(model.D.n_active == 1)
    assert np.all(model.D.context == 0)


def test_predictive_moments_bracket_the_filtered_mean():
    """The one-step predictive mean stays on the same scale as the filter."""
    y = _synthetic_run(np.random.default_rng(11))
    kf_means = _reference_kalman(y)
    model = _single_context_model()
    for y_t in y:
        model.observe_q(1)
        model.observe_y(y_t)

    mu, sigma = model.predictive_feedback_moments()
    assert np.isfinite(mu) and np.isfinite(sigma)
    # The predictive variance must be at least the observation noise.
    assert sigma >= R_STD ** 2 - 1e-12
    assert abs(mu - kf_means[-1]) < TOLERANCE
    # predictive_motor_output is documented to equal the predictive mean.
    np.testing.assert_allclose(model.predictive_motor_output(), mu, atol=1e-12)

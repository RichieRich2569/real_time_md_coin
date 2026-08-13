"""Analytic Kalman recursion used as an external mathematical reference.

Translated from ``validation/validation_kalman_reference.m``.

One predict/update step of the linear-Gaussian Kalman filter for the
one-context COIN state model::

    s_t = A s_{t-1} + d + w_t,   w_t ~ N(0, Q)
    y_t = s_t + v_t,             v_t ~ N(0, R)

The scalar single-context case is the ``N == 1`` specialisation of exactly this
recursion, so :mod:`validation.validate_single_context_kalman` and
:mod:`validation.validate_multidim_kalman` both call
:func:`kalman_reference_step`. Keeping the recursion in one place is deliberate:
it makes it impossible for the scalar and multivariate references to drift
apart.

Nothing in this module touches :mod:`realtimecoin` - it is a from-scratch
implementation of the textbook recursion, which is the whole point of using it
as ground truth.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kalman_reference_step"]


def kalman_reference_step(m, P, A, d, Q, R, y):
    """Run one predict-update step of the multivariate Kalman recursion.

    Scalar inputs (a 0-d ``m``) are detected and the four outputs are returned
    as Python floats; otherwise the state is treated as an ``(N,)`` vector and
    the outputs are ``numpy`` arrays. ``N`` is taken from ``m``, and every other
    argument is RESHAPED (not broadcast) to ``(N,)`` or ``(N, N)``: each must
    supply exactly that many elements, so passing a scalar ``A`` alongside an
    ``(N,)`` ``m`` raises :class:`ValueError` rather than being read as
    ``A * I_N``.

    Parameters
    ----------
    m : float or array_like
        Prior state mean: scalar, or ``(N,)``.
    P : float or array_like
        Prior state covariance: scalar, or ``(N, N)``.
    A : float or array_like
        State-transition (retention) matrix: scalar, or ``(N, N)``.
    d : float or array_like
        Drift: scalar, or ``(N,)``.
    Q : float or array_like
        Process-noise covariance: scalar, or ``(N, N)``.
    R : float or array_like
        Observation-noise covariance: scalar, or ``(N, N)``.
    y : float or array_like
        Observation: scalar, or ``(N,)``.

    Returns
    -------
    pred_mean : float or numpy.ndarray
        One-step predictive state mean ``A m + d``.
    pred_feedback_cov : float or numpy.ndarray
        Predictive FEEDBACK covariance ``A P A' + Q + R`` (the innovation
        covariance), i.e. the covariance of the next observation.
    post_mean : float or numpy.ndarray
        Posterior (filtered) mean after assimilating ``y``.
    post_cov : float or numpy.ndarray
        Posterior (filtered) STATE covariance, re-symmetrised.

    Examples
    --------
    >>> pred, s, m, p = kalman_reference_step(0.0, 1.0, 0.5, 0.0, 0.1, 0.2, 1.0)
    >>> round(pred, 6), round(s, 6)
    (0.0, 0.55)
    """
    scalar = np.ndim(m) == 0

    m_vec = np.atleast_1d(np.asarray(m, dtype=float)).reshape(-1)
    n = m_vec.size
    p_mat = np.asarray(P, dtype=float).reshape(n, n)
    a_mat = np.asarray(A, dtype=float).reshape(n, n)
    d_vec = np.atleast_1d(np.asarray(d, dtype=float)).reshape(-1)
    q_mat = np.asarray(Q, dtype=float).reshape(n, n)
    r_mat = np.asarray(R, dtype=float).reshape(n, n)
    y_vec = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1)

    # Predict: m^- = A m + d,  P^- = A P A' + Q.
    pred_mean = a_mat @ m_vec + d_vec
    pred_state_cov = a_mat @ p_mat @ a_mat.T + q_mat
    # Innovation (predictive feedback) covariance S = P^- + R.
    pred_feedback_cov = pred_state_cov + r_mat

    # Kalman gain K = P^- S^{-1}. MATLAB writes this as the right division
    # `predStateCov / predFeedbackCov`, which solves K * S = P^- without ever
    # forming an explicit inverse; `X = B / A` is `solve(A', B')'` in numpy.
    gain = np.linalg.solve(pred_feedback_cov.T, pred_state_cov.T).T

    # Update: m^+ = m^- + K (y - m^-),  P^+ = P^- - K P^- = (I - K) P^-.
    post_mean = pred_mean + gain @ (y_vec - pred_mean)
    post_cov = pred_state_cov - gain @ pred_state_cov
    post_cov = (post_cov + post_cov.T) / 2.0   # re-symmetrise (no-op if N == 1)

    if scalar:
        return (
            float(pred_mean[0]),
            float(pred_feedback_cov[0, 0]),
            float(post_mean[0]),
            float(post_cov[0, 0]),
        )
    return pred_mean, pred_feedback_cov, post_mean, post_cov

"""Gaussian-mixture moments and CDF used by the validation suite.

Translated from ``validation/validation_mixture_moments.m`` and
``validation/validation_mixture_cdf.m``.

Every RealTimeCOIN predictive distribution is a Gaussian mixture over contexts
and particles, so reducing one to a mean/variance pair or evaluating its CDF is
the single most reused operation in the scientific validators.

Both functions sanitise the weights the same way MATLAB does - non-finite or
negative weights are zeroed, and the remainder renormalised to sum to one - so
an unused context slot carrying a ``nan`` weight cannot poison the reduction.
"""

from __future__ import annotations

import numpy as np

from realtimecoin.statics import normal_cdf

__all__ = ["mixture_moments", "mixture_cdf", "sanitise_weights"]


def sanitise_weights(weights):
    """Zero out invalid weights and renormalise; ``None`` if nothing is left.

    MATLAB's ``w(~isfinite(w) | w < 0) = 0; w = w ./ sum(w, 'all')``.

    Parameters
    ----------
    weights : array_like
        Unnormalised mixture weights of any shape.

    Returns
    -------
    numpy.ndarray or None
        Weights of the same shape summing to one, or ``None`` when the total
        mass is non-positive (the caller then reports ``nan``).
    """
    w = np.asarray(weights, dtype=float)
    # `w >= 0` is False for nan, so this single mask covers both MATLAB tests.
    w = np.where(np.isfinite(w) & (w >= 0.0), w, 0.0)
    total = float(w.sum())
    if not total > 0.0:
        return None
    return w / total


def mixture_moments(weights, means, variances):
    """Mean and variance of a Gaussian mixture.

    If component ``k`` has weight ``w_k``, mean ``m_k`` and variance ``V_k``::

        E[X]   = sum_k w_k m_k
        Var[X] = sum_k w_k (V_k + m_k^2) - E[X]^2

    Parameters
    ----------
    weights : array_like
        Unnormalised mixture weights.
    means : array_like
        Component means, same shape as ``weights``.
    variances : array_like
        Component variances, same shape as ``weights``. Negative entries are
        floored at zero, as in the MATLAB original.

    Returns
    -------
    mu : float
        Mixture mean, or ``nan`` when the weights carry no mass.
    v : float
        Mixture variance (floored at zero), or ``nan``.
    """
    w = sanitise_weights(weights)
    if w is None:
        return float("nan"), float("nan")

    means = np.asarray(means, dtype=float)
    variances = np.fmax(np.asarray(variances, dtype=float), 0.0)

    mu = float(np.sum(w * means))
    second = float(np.sum(w * (variances + means ** 2)))
    # fmax, not maximum: MATLAB's max(x, 0) returns 0 when x is nan.
    return mu, float(np.fmax(second - mu ** 2, 0.0))


def mixture_cdf(x, weights, means, variances):
    """CDF of a Gaussian mixture evaluated at ``x``.

    ``F(x) = sum_k w_k Phi((x - m_k) / sqrt(V_k))``, clipped to ``[0, 1]``.
    Uses :func:`realtimecoin.statics.normal_cdf`, the exact function the MATLAB
    helper calls (``RealTimeCOIN.normal_cdf``), so the degenerate zero-variance
    branch behaves identically.

    Parameters
    ----------
    x : float
        Query point.
    weights : array_like
        Unnormalised mixture weights.
    means : array_like
        Component means, same shape as ``weights``.
    variances : array_like
        Component variances, same shape as ``weights``.

    Returns
    -------
    float
        Cumulative probability in ``[0, 1]``, or ``nan`` when the weights carry
        no mass.
    """
    w = sanitise_weights(weights)
    if w is None:
        return float("nan")

    means = np.asarray(means, dtype=float)
    variances = np.fmax(np.asarray(variances, dtype=float), 0.0)
    p = float(np.sum(w * normal_cdf(x, means, variances)))
    # fmin/fmax mirror MATLAB's min(max(p, 0), 1), which maps nan to 0.
    return float(np.fmin(np.fmax(p, 0.0), 1.0))

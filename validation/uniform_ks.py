"""One-sample Kolmogorov-Smirnov distance from Uniform(0, 1).

Translated from ``validation/validation_uniform_ks.m``.

Probability-integral-transform (PIT) values are Uniform(0, 1) when the
predictive distribution that produced them is calibrated, so the KS distance
between their empirical CDF and ``F(u) = u`` is the suite's primary calibration
statistic.
"""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_ks"]


def uniform_ks(values):
    """Largest vertical gap between the empirical CDF of ``values`` and ``u``.

    Non-finite entries are dropped, the rest clipped to ``[0, 1]`` and sorted.
    With the sorted sample ``p_(1) <= ... <= p_(n)`` the statistic is::

        D = max_i max( |i/n - p_(i)| , |(i-1)/n - p_(i)| )

    which is the exact one-sample KS distance (both the "above" and "below"
    step of the empirical CDF are examined at every order statistic).

    Parameters
    ----------
    values : array_like
        PIT values of any shape; flattened before use.

    Returns
    -------
    float
        The KS distance, or ``nan`` when no finite value survives filtering.
    """
    p = np.asarray(values, dtype=float).reshape(-1)
    p = p[np.isfinite(p)]
    p = np.sort(np.fmin(np.fmax(p, 0.0), 1.0))
    n = p.size
    if n == 0:
        return float("nan")

    upper = np.arange(1, n + 1, dtype=float) / n     # i / n
    lower = np.arange(0, n, dtype=float) / n         # (i - 1) / n
    return float(max(np.max(np.abs(upper - p)), np.max(np.abs(lower - p))))

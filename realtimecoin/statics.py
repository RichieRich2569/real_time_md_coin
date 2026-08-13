"""Public static helpers of the RealTimeCOIN model.

Direct translation of the ``methods (Static)`` block of
``@RealTimeCOIN/RealTimeCOIN.m``: systematic resampling, the guarded normal
pdf/cdf, log-sum-exp, the Markov-chain stationary distribution and the
Antoniak/CRP table-count sampler.

Every degenerate branch and magic constant of the MATLAB source is preserved
verbatim, because downstream particle-filter behaviour depends on them:

* ``eps``    -> ``numpy.finfo(float).eps``    (2.22e-16, MATLAB ``eps``)
* ``realmax``-> ``numpy.finfo(float).max``    (1.80e308, MATLAB ``realmax``)

MATLAB's two-argument ``max`` / ``min`` OMIT ``NaN`` (they return the other
operand) while ``numpy.maximum`` / ``numpy.minimum`` propagate it, so the
variance floors and probability clamps below use ``numpy.fmax`` / ``fmin``,
which omit ``NaN`` the same way.

Randomness is taken from an explicit :class:`numpy.random.Generator` passed as
the first argument; the global ``numpy.random`` state is never touched.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc

__all__ = [
    "systematic_resampling",
    "normal_pdf",
    "normal_cdf",
    "log_sum_exp",
    "stationary_distribution",
    "sample_num_tables",
]

EPS = float(np.finfo(float).eps)
REALMAX = float(np.finfo(float).max)


def systematic_resampling(rng, weights):
    """Systematic (low-variance) resampling of particle indices.

    Non-finite weights are zeroed; if the remaining mass is not strictly
    positive the weights fall back to uniform. Exactly ONE uniform variate is
    drawn (that single draw is what makes the scheme "systematic"), and the
    final cumulative entry is forced to 1 so round-off can never leave the
    last position unmatched.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator; consumes exactly one uniform variate.
    weights : array_like
        Particle weights (any shape; flattened). Need not be normalised.

    Returns
    -------
    numpy.ndarray
        Integer array of ``n`` resampled indices, ZERO-based (the MATLAB
        original returns one-based indices).
    """
    w = np.asarray(weights, dtype=float).ravel().copy()
    w[~np.isfinite(w)] = 0.0
    n = w.size
    if w.sum() <= 0:
        w = np.ones(n) / n
    else:
        w = w / w.sum()

    # positions = ((0:n-1) + u) / n  with a single uniform draw u ~ U(0, 1)
    positions = (np.arange(n) + rng.random()) / n
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0

    # MATLAB advances j while positions(i) > cumulative(j), i.e. it selects the
    # first j with cumulative(j) >= positions(i): a left-side binary search.
    indices = np.searchsorted(cumulative, positions, side="left")
    return np.minimum(indices, n - 1)


def normal_pdf(x, m, v):
    """Univariate normal density with guarded degenerate branches.

    Implements ``p(x) = exp(-0.5 (x - m)^2 / v) / sqrt(2 pi v)`` element-wise.
    A scalar, non-positive variance is treated as a Dirac spike of height
    ``1 / sqrt(eps)`` on ``|x - m| <= sqrt(eps)``; otherwise the variance is
    floored at ``eps`` and any non-finite result is replaced by ``realmax``.

    Parameters
    ----------
    x : array_like
        Query point(s).
    m : array_like
        Mean(s), broadcastable against ``x``.
    v : array_like
        Variance(s), broadcastable against ``x``.

    Returns
    -------
    numpy.ndarray
        Density evaluated at ``x`` (broadcast shape of the inputs).
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    if v.size == 1 and v.reshape(-1)[0] <= 0:
        # Degenerate (zero-variance) component: a numerical Dirac spike.
        return np.where(np.abs(x - m) <= math.sqrt(EPS), 1.0 / math.sqrt(EPS), 0.0)

    v = np.fmax(v, EPS)
    # p(x) = N(x; m, v) = exp(-0.5 (x - m)^2 / v) / sqrt(2 pi v)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2.0 * np.pi * v)
        p = np.where(np.isfinite(p), p, REALMAX)
    return p


def normal_cdf(x, m, v):
    """Univariate normal cumulative distribution function.

    Evaluated through ``erfc`` for tail accuracy and clipped to [0, 1]. A
    scalar, non-positive variance degenerates to the step function
    ``1{x >= m}``.

    Parameters
    ----------
    x : array_like
        Query point(s).
    m : array_like
        Mean(s), broadcastable against ``x``.
    v : array_like
        Variance(s), broadcastable against ``x``.

    Returns
    -------
    numpy.ndarray
        Cumulative probability at ``x`` (broadcast shape of the inputs).
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    if v.size == 1 and v.reshape(-1)[0] <= 0:
        return (x >= m).astype(float)

    v = np.fmax(v, EPS)
    # F(x) = 0.5 * erfc(-(x - m) / sqrt(2 v))
    p = 0.5 * erfc(-(x - m) / np.sqrt(2.0 * v))
    # fmin/fmax, not clip: MATLAB's min(max(p, 0), 1) maps a NaN to 0.
    return np.fmin(np.fmax(p, 0.0), 1.0)


def log_sum_exp(log_p, axis=-1):
    """Numerically stable ``log(sum(exp(log_p)))`` along one axis.

    The maximum is shifted out before exponentiating. Slices whose maximum is
    non-finite (all ``-inf``, or containing ``nan``) return ``-inf``, which
    mirrors the explicit ``l(~isfinite(m)) = -Inf`` branch of the MATLAB
    source.

    Parameters
    ----------
    log_p : array_like
        Array of log-values.
    axis : int, optional
        Axis to reduce over. The default is ``-1``, NOT the MATLAB default of
        ``dim = 1``: MATLAB reduces the leading dimension of a
        ``(contexts, particles)`` array, and under this package's
        ``(particles, contexts)`` layout the same context axis is the trailing
        one. Callers translating MATLAB that omitted ``dim`` therefore get the
        same reduction by also omitting ``axis``.

    Returns
    -------
    numpy.ndarray
        The reduced array with ``axis`` removed.
    """
    log_p = np.asarray(log_p, dtype=float)
    m_keep = np.max(log_p, axis=axis, keepdims=True)
    m = np.squeeze(m_keep, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        # log sum_i exp(x_i) = m + log sum_i exp(x_i - m)
        result = m + np.log(np.sum(np.exp(log_p - m_keep), axis=axis))
    return np.where(np.isfinite(m), result, -np.inf)


def stationary_distribution(transition):
    """Stationary distribution of a time-homogeneous Markov chain.

    Solves ``pi T = pi`` subject to ``sum(pi) = 1`` through the augmented
    system ``[T' - I; ones] x = [zeros; 1]``. MATLAB's backslash on this
    overdetermined system is a least-squares QR solve, so ``lstsq`` is used
    here. Negative entries (round-off) are clipped to zero and the result is
    renormalised; a zero-mass solution falls back to the uniform distribution.

    Parameters
    ----------
    transition : array_like
        ``(C, C)`` row-stochastic transition matrix.

    Returns
    -------
    numpy.ndarray
        Length-``C`` stationary probability vector summing to one.

    Notes
    -----
    When the augmented system is rank-deficient - a reducible chain with more
    than one closed communicating class, so the stationary distribution is not
    unique - MATLAB's pivoted-QR backslash returns a basic solution (free
    variables set to zero) whereas ``lstsq`` returns the minimum-norm solution.
    Both are valid stationary distributions; the minimum-norm one simply
    spreads the mass evenly over the tied classes instead of picking one by
    pivot order.
    """
    t = np.asarray(transition, dtype=float)
    c = t.shape[0]

    # Stack the balance equations (T' - I) x = 0 with the normalisation row.
    a = np.vstack([t.T - np.eye(c), np.ones((1, c))])   # (C + 1, C)
    b = np.concatenate([np.zeros(c), [1.0]])            # (C + 1,)

    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    x = np.where(x < 0, 0.0, x)   # literal x(x < 0) = 0: a nan is left alone
    total = x.sum()
    if total == 0:
        return np.ones(c) / c
    return x / total


def sample_num_tables(rng, base, counts):
    """Antoniak/CRP sampler for the number of tables serving each dish.

    For ``n`` customers and concentration ``b`` the number of occupied tables
    is the sum of independent Bernoulli draws, customer ``i`` (1-based) opening
    a new table with probability ``b / (b + i - 1)``.

    The explicit per-customer loop is kept deliberately: the draw count and
    ordering are part of the algorithm's RNG consumption pattern, so it must
    not be collapsed into a single vectorised call. Elements are visited in
    C (row-major) order, which is the natural Python traversal; the MATLAB
    original walks the array in column-major order. Only the RNG stream order
    differs, not the distribution of the result.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    base : array_like
        Concentration parameter(s), broadcastable to the shape of ``counts``.
    counts : array_like
        Number of customers per cell. Non-integral counts are truncated
        towards zero, matching MATLAB's ``for customer = 1:n``.

    Returns
    -------
    numpy.ndarray
        Table counts, same shape as ``counts``, with
        ``0 <= tables <= floor(counts)``.
    """
    counts = np.asarray(counts, dtype=float)
    base = np.broadcast_to(np.asarray(base, dtype=float), counts.shape)

    tables = np.zeros(counts.shape, dtype=float)
    flat_counts = counts.reshape(-1)
    flat_base = base.reshape(-1)
    flat_tables = tables.reshape(-1)

    for i in range(flat_counts.size):
        n = flat_counts[i]
        b = flat_base[i]
        if not (n > 0) or not (b > 0):
            continue
        opened = 0
        for customer in range(1, int(math.floor(n)) + 1):
            # P(customer i opens a new table) = b / (b + i - 1)
            if rng.random() < b / (b + customer - 1):
                opened += 1
        flat_tables[i] = opened

    return tables

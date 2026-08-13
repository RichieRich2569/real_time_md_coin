"""Random samplers used by the RealTimeCOIN particle filter.

Translated from ``@RealTimeCOIN/private/``: ``gammaSample``, ``betaSample``,
``dirichletSample``, ``binomialSample``, ``trandn``, ``sampleScalarNormal``,
``sampleBivariateTruncated``, ``sampleMatrixNormal`` and
``sampleStableTheta``.

Every sampler takes an explicit :class:`numpy.random.Generator` as its FIRST
argument; the global ``numpy.random`` state is never used. Degenerate
short-circuits are preserved exactly, both because they define the behaviour
on invalid inputs and because they determine how many variates are consumed
(``binomialSample`` in particular sums Bernoulli draws by design, so it must
not be replaced by ``rng.binomial``).

The truncated-normal routine follows Botev (2016), "The normal law under
linear restrictions: simulation and estimation via minimax tilting", JRSS-B,
doi:10.1111/rssb.12162.

MATLAB's two-argument ``max`` / ``min`` OMIT ``NaN`` (they return the other
operand) while ``numpy.maximum`` / ``numpy.minimum`` propagate it, so the
bound clamps below use ``numpy.fmax`` / ``fmin``. That is what makes the
degenerate fallbacks return a valid retention in [0, 1) rather than leaking a
``NaN`` into the particle state.

Provenance: the ``trandn`` / ``ntail`` / ``tn`` structure was cross-checked
against, and partly adapted from, the half-translated reference in the COINRL
project (``utils/distribution_utils.py``), which itself adapts Changmin Yu's
COIN_Python and is licensed GPL-3.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcinv

from .numerics import choljitter

__all__ = [
    "gamma_sample",
    "beta_sample",
    "dirichlet_sample",
    "binomial_sample",
    "trandn",
    "sample_scalar_normal",
    "sample_bivariate_truncated",
    "sample_matrix_normal",
    "sample_stable_theta",
]

EPS = float(np.finfo(float).eps)


def gamma_sample(rng, shape):
    """Draw gamma variates with unit scale, element-wise.

    ``g[i] ~ Gamma(shape[i], 1)``. Entries with a non-positive or ``nan``
    shape return exactly 0 (the degenerate gamma), so the result is always
    well defined for the counts and concentration parameters passed in by the
    beta / Dirichlet helpers.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    shape : array_like
        Gamma shape parameters (any shape of array).

    Returns
    -------
    numpy.ndarray
        Draws with the same shape as ``shape``; 0 where ``shape <= 0`` or
        ``shape`` is ``nan``.
    """
    shape = np.asarray(shape, dtype=float)
    g = np.zeros(shape.shape)
    good = shape > 0
    if np.any(good):
        g[good] = rng.standard_gamma(shape[good])
    return g


def beta_sample(rng, a, b):
    """Draw beta variates via the gamma-ratio construction.

    ``Beta(a, b) = X / (X + Y)`` with ``X ~ Gamma(a, 1)`` and
    ``Y ~ Gamma(b, 1)``.

    Degenerate fallback: when both gamma draws vanish (both shapes
    non-positive, so ``0 / 0`` is undefined) the entry is set to 1. This is
    deliberately asymmetric - it could equally be 0 - and is kept because the
    only callers draw stick-breaking weights ``Beta(1, gamma)`` where
    ``a = 1 > 0``, so the fallback never fires on the valid path. Returning 1
    keeps a degenerate stick weight from spuriously halting the break.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    a : array_like
        First (alpha) shape parameters.
    b : array_like
        Second (beta) shape parameters, broadcastable against ``a``.

    Returns
    -------
    numpy.ndarray
        Beta draws in [0, 1] with the broadcast shape of ``a`` and ``b``.
    """
    # Each gamma is drawn at its OWN size and the ratio broadcasts afterwards,
    # exactly as MATLAB does. That matters when one argument is scalar: a
    # single Y is then shared across all entries rather than one Y per entry.
    x = gamma_sample(rng, a)
    y = gamma_sample(rng, b)
    denom = x + y
    x = np.broadcast_to(x, denom.shape)
    good = denom > 0
    out = np.ones(denom.shape)   # asymmetric 0/0 fallback (see docstring)
    out[good] = x[good] / denom[good]
    return out


def dirichlet_sample(rng, alpha):
    """Draw a Dirichlet-distributed probability vector.

    Uses the gamma-normalisation construction: ``g[i] ~ Gamma(alpha[i], 1)``
    then ``x = g / sum(g)``.

    Degenerate fallback: if every gamma draw is zero, all probability mass is
    placed on the first entry with ``alpha > 0`` (or entry 0 if none is
    positive), so the result stays a valid probability vector instead of
    ``0 / 0 = nan``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    alpha : array_like
        Concentration parameters (any orientation; flattened).

    Returns
    -------
    numpy.ndarray
        Length-``numel(alpha)`` non-negative vector summing to one.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    draws = gamma_sample(rng, alpha)
    if draws.sum() <= 0:
        draws = np.zeros(alpha.shape)
        positive = np.flatnonzero(alpha > 0)
        first = positive[0] if positive.size else 0
        draws[first] = 1.0
    return draws / draws.sum()


def binomial_sample(rng, trials, prob):
    """Draw a single binomial count by explicit Bernoulli summation.

    ``n ~ Binomial(trials, prob)``, computed as
    ``sum(rng.random(trials) < prob)``. The Bernoulli summation is part of the
    algorithm (it fixes how many uniforms are consumed), so it must not be
    replaced by ``rng.binomial``. The degenerate cases are short-circuited:
    ``trials <= 0`` or ``prob <= 0`` gives 0 and ``prob >= 1`` gives
    ``trials``, consuming no random numbers on those paths.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    trials : float
        Number of Bernoulli trials (scalar).
    prob : float
        Success probability (scalar).

    Returns
    -------
    int
        Count in ``[0, trials]``.

    Notes
    -----
    The Bernoulli path is O(trials) in time and memory. The sole caller passes
    small table counts, so this is not a bottleneck.
    """
    trials = float(trials)
    prob = float(prob)
    if trials <= 0 or prob <= 0:
        return 0
    if prob >= 1:
        return int(trials)
    return int(np.sum(rng.random(int(trials)) < prob))


def trandn(rng, lower, upper):
    """Sample the standard normal truncated to ``[lower, upper]``, element-wise.

    Entry ``i`` is drawn from ``N(0, 1)`` conditioned on
    ``lower[i] < X < upper[i]``. Infinite limits are allowed (one-sided or
    unbounded truncation). To sample ``N(m, s^2)`` truncated to ``(a, b)``,
    draw ``x = trandn(rng, (a - m)/s, (b - m)/s)`` and set ``z = m + s x``.

    Algorithm (Botev 2016): each element is routed to one of three exact
    acceptance-rejection schemes according to where the interval sits relative
    to the tuned threshold ``a = 0.66``:

    * ``lower > a``  : right tail, sampled by ``ntail`` (Rayleigh proposal);
    * ``upper < -a`` : left tail, sampled by ``-ntail`` on mirrored limits;
    * otherwise      : interval straddling 0, sampled by ``tn``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    lower : array_like
        Lower truncation limits (``-inf`` allowed).
    upper : array_like
        Upper truncation limits (``+inf`` allowed), same length as ``lower``.

    Returns
    -------
    numpy.ndarray
        Length-``n`` vector of truncated standard-normal draws.
    """
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if lower.size != upper.size:
        raise ValueError(
            "RealTimeCOIN:trandn:limitLength: Truncation limits have to be "
            "vectors of the same length."
        )
    if np.any(lower > upper):
        raise ValueError(
            "RealTimeCOIN:trandn:emptyInterval: Each lower limit must not "
            "exceed its upper limit."
        )

    x = np.full(lower.shape, np.nan)
    a = 0.66   # Botev's tuned tail / central switch-over

    right = lower > a
    if np.any(right):
        x[right] = _ntail(rng, lower[right], upper[right])

    left = upper < -a
    if np.any(left):
        x[left] = -_ntail(rng, -upper[left], -lower[left])

    middle = ~(right | left)
    if np.any(middle):
        x[middle] = _tn(rng, lower[middle], upper[middle])

    return x


def _ntail(rng, lower, upper):
    """Standard normal truncated to ``[lower, upper]`` with ``lower > 0``.

    Acceptance-rejection from a Rayleigh proposal.
    """
    c = lower ** 2 / 2.0
    n = lower.size
    f = np.expm1(c - upper ** 2 / 2.0)
    x = c - np.log1p(rng.random(n) * f)   # Rayleigh proposal
    idx = np.flatnonzero(rng.random(n) ** 2 * x > c)
    while idx.size > 0:
        cy = c[idx]
        y = cy - np.log1p(rng.random(idx.size) * f[idx])
        accepted = rng.random(idx.size) ** 2 * y < cy
        x[idx[accepted]] = y[accepted]
        idx = idx[~accepted]
    return np.sqrt(2.0 * x)


def _tn(rng, lower, upper):
    """Standard normal truncated to ``[lower, upper]`` around the origin.

    Intervals wider than ``tol = 2`` (Botev's switch-over) use plain normal
    rejection, which then has a high acceptance rate; narrower ones use the
    numerically stable ``erfc`` / ``erfcinv`` inverse transform.
    """
    tol = 2
    wide = np.abs(upper - lower) > tol
    x = lower.copy()

    if np.any(wide):
        x[wide] = _trnd_reject(rng, lower[wide], upper[wide])

    narrow = ~wide
    if np.any(narrow):
        tl = lower[narrow]
        tu = upper[narrow]
        # Inverse transform through the normal tail probabilities
        # F(t) = erfc(t / sqrt(2)) / 2.
        pl = erfc(tl / np.sqrt(2.0)) / 2.0
        pu = erfc(tu / np.sqrt(2.0)) / 2.0
        x[narrow] = np.sqrt(2.0) * erfcinv(
            2.0 * (pl - (pl - pu) * rng.random(tl.size))
        )
    return x


def _trnd_reject(rng, lower, upper):
    """Rejection sampling from the untruncated standard normal."""
    x = rng.standard_normal(lower.size)
    idx = np.flatnonzero((x < lower) | (x > upper))
    while idx.size > 0:
        y = rng.standard_normal(idx.size)
        accepted = (y > lower[idx]) & (y < upper[idx])
        x[idx[accepted]] = y[accepted]
        idx = idx[~accepted]
    return x


def sample_scalar_normal(rng, mu, variance, shape, low, high):
    """Draw truncated univariate normals element-wise.

    Each stochastic entry is drawn as ``mu + sigma * trandn(l, u)`` with
    ``sigma = sqrt(variance)`` and ``(l, u)`` the standardised truncation
    limits. Entries with a non-positive or non-finite variance are treated as
    deterministic and returned at ``mu``, so degenerate priors and
    fully-observed states pass through unchanged.

    The result is finally clamped with ``x = min(max(x, low), high - eps)``.
    The ``-eps`` keeps every draw STRICTLY below the upper bound, which matters
    for the retention parameter ``a`` in [0, 1): a value of exactly 1 would
    make the latent AR(1) process non-stationary.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    mu : array_like
        Mean(s): scalar or an array of shape ``shape``.
    variance : array_like
        Variance(s): scalar or an array of shape ``shape``; ``<= 0`` gives a
        deterministic entry.
    shape : tuple of int
        Shape of the output array.
    low : array_like
        Lower truncation bound(s): scalar or an array of shape ``shape``.
    high : array_like
        Upper truncation bound(s): scalar or an array of shape ``shape``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``shape`` of truncated normal draws.
    """
    shape = tuple(np.atleast_1d(shape).astype(int))
    mu = np.broadcast_to(np.asarray(mu, dtype=float), shape)
    variance = np.broadcast_to(np.asarray(variance, dtype=float), shape)
    low = np.broadcast_to(np.asarray(low, dtype=float), shape)
    high = np.broadcast_to(np.asarray(high, dtype=float), shape)

    x = np.array(mu, dtype=float, copy=True)
    stochastic = (variance > 0) & np.isfinite(variance)
    if np.any(stochastic):
        sigma = np.sqrt(variance[stochastic])
        l = (low[stochastic] - mu[stochastic]) / sigma
        u = (high[stochastic] - mu[stochastic]) / sigma
        x[stochastic] = mu[stochastic] + sigma * trandn(rng, l, u)
    # fmin/fmax, not minimum/maximum: MATLAB's min(max(x, low), high - eps)
    # clamps a nan mean to low rather than letting the nan through.
    return np.fmin(np.fmax(x, low), high - EPS)


def sample_bivariate_truncated(rng, mu, covar):
    """Draw ``[a; d]`` from a bivariate normal with ``a`` truncated to [0, 1).

    This is the conjugate posterior draw of the scalar dynamics ``[a; d]``: the
    retention ``a`` is truncated to keep the latent AR(1) in its stationary
    range (the multi-dimensional analogue is the spectral-radius constraint of
    :func:`sample_stable_theta`), while the drift ``d`` is unconstrained.

    Method: with ``L`` the lower Cholesky factor of ``covar``, draw a
    standardised ``z ~ N(0, I)`` whose first coordinate is truncated to
    ``[(0 - mu[0]) / L[0, 0], (1 - mu[0]) / L[0, 0]]``, then ``x = mu + L z``.
    Because ``L`` is lower triangular, ``x[0] = mu[0] + L[0, 0] z[0]`` depends
    only on the truncated coordinate, so ``a`` is truncated exactly.

    Degenerate fallbacks: non-finite ``mu`` or ``covar``, or a ``covar`` that
    stays non-PD (Cholesky fails, or ``L[0, 0] <= 0``), return the
    deterministic mean with ``a`` clamped to ``[0, 1 - eps]``. ``covar`` is
    symmetrised and given a 1e-12 diagonal ridge before the Cholesky; that
    ridge is negligible relative to the parameter scales.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    mu : array_like
        Length-2 posterior mean ``[a, d]``.
    covar : array_like
        ``(2, 2)`` posterior covariance.

    Returns
    -------
    numpy.ndarray
        Length-2 sample ``[a, d]`` with ``a`` in [0, 1).
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    covar = np.asarray(covar, dtype=float)

    # fmin/fmax, not the builtins: this branch exists for non-finite input, and
    # MATLAB's min(max(a, 0), 1 - eps) clamps a nan retention to 0.
    fallback = np.array([np.fmin(np.fmax(mu[0], 0.0), 1.0 - EPS), mu[1]])
    if not np.all(np.isfinite(covar)) or not np.all(np.isfinite(mu)):
        return fallback

    # Symmetrise and add a tiny 1e-12 ridge so a (near-)singular posterior
    # covariance still yields a valid Cholesky factor.
    covar = (covar + covar.T) / 2.0 + 1e-12 * np.eye(2)
    try:
        factor = np.linalg.cholesky(covar)
    except (np.linalg.LinAlgError, ValueError):
        return fallback
    if factor[0, 0] <= 0:
        return fallback

    l = np.array([(0.0 - mu[0]) / factor[0, 0], -np.inf])
    u = np.array([(1.0 - mu[0]) / factor[0, 0], np.inf])
    z = trandn(rng, l, u)
    x = mu + factor @ z
    x[0] = np.fmin(np.fmax(x[0], 0.0), 1.0 - EPS)
    return x


def sample_matrix_normal(rng, mean, row_cov, col_cov):
    """Draw ``X ~ MN_{n,p}(mean, row_cov, col_cov)``.

    Sampling identity: if ``Z`` is ``n`` by ``p`` standard normal,
    ``L_U L_U' = U`` and ``R_V' R_V = V``, then ``X = M + L_U Z R_V`` has the
    required first two moments, because
    ``Cov(vec(X)) = (R_V' R_V) kron (L_U L_U') = V kron U``.

    ``L_U`` is the lower Cholesky factor of ``U`` (via :func:`choljitter`, so a
    near-singular row covariance such as zero process noise degrades
    gracefully) and ``R_V`` the upper Cholesky factor of ``V``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    mean : array_like
        ``(n, p)`` mean matrix.
    row_cov : array_like
        ``(n, n)`` among-row covariance.
    col_cov : array_like
        ``(p, p)`` among-column covariance.

    Returns
    -------
    numpy.ndarray
        ``(n, p)`` matrix-normal sample.
    """
    mean = np.atleast_2d(np.asarray(mean, dtype=float))
    row_cov = np.asarray(row_cov, dtype=float)
    col_cov = np.asarray(col_cov, dtype=float)
    n, p = mean.shape
    if row_cov.shape != (n, n):
        raise ValueError(
            "RealTimeCOIN:sampleMatrixNormal:invalidRowCov: row_cov must be "
            "%d-by-%d to match mean." % (n, n)
        )
    if col_cov.shape != (p, p):
        raise ValueError(
            "RealTimeCOIN:sampleMatrixNormal:invalidColCov: col_cov must be "
            "%d-by-%d to match mean." % (p, p)
        )

    z = rng.standard_normal((n, p))
    lu, _ = choljitter(row_cov)                          # lu @ lu.T = row_cov
    rv = _chol_upper_psd((col_cov + col_cov.T) / 2.0)    # rv.T @ rv = col_cov
    return mean + lu @ z @ rv


def _chol_upper_psd(v):
    """Upper Cholesky factor of a symmetric PSD matrix, with jitter.

    Returns ``R`` with ``R.T @ R = v``. On failure an escalating diagonal
    jitter starting at one part in 1e12 of the matrix scale (x10 per attempt,
    up to 8 tries) is applied; if all attempts fail the diagonal square root is
    returned.
    """
    factor = _cholesky_upper(v)
    if factor is not None:
        return factor

    scale = np.mean(np.diag(v)) if v.size else np.nan
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    jit = 1e-12 * scale
    for _ in range(8):
        factor = _cholesky_upper(v + jit * np.eye(v.shape[0]))
        if factor is not None:
            return factor
        jit *= 10.0
    # fmax keeps the fallback finite for non-finite input (MATLAB max omits NaN).
    return np.diag(np.sqrt(np.fmax(np.diag(v), EPS)))


def _cholesky_upper(v):
    """Upper Cholesky factor ``R`` with ``R.T @ R = v``, or ``None``.

    As in ``numerics._try_cholesky_lower``, both a raised ``ValueError`` on
    non-finite input and a silently non-finite factor count as failure, so the
    caller reaches its jitter / diagonal fallback either way.
    """
    try:
        factor = np.linalg.cholesky(v).T
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(factor)):
        return None
    return factor


def sample_stable_theta(rng, mean, row_cov, col_cov, state_dim=None):
    """Draw a stable augmented dynamics matrix ``Theta = [A | d]``.

    Draws ``Theta ~ MN(mean, row_cov, col_cov)`` and enforces bounded
    stability on the dynamics block ``A = Theta[:, :N]``: the spectral radius
    ``rho(A) = max(abs(eig(A)))`` must be ``< 1`` so the latent AR(1) process
    is stationary. This is the multi-dimensional analogue of the scalar
    ``a in [0, 1)`` truncation in :func:`sample_bivariate_truncated`. It also
    implies ``abs(det(A)) < 1``, the determinant magnitude being the product of
    the eigenvalue magnitudes.

    Strategy: rejection-sample up to ``max_iter = 10`` times; if no stable draw
    is found, force stability by spectral scaling ``A <- A / (rho + margin)``
    with ``margin = 0.01``, which shrinks every eigenvalue inside the unit disc
    while leaving the drift column ``d`` unchanged.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    mean : array_like
        ``(N, N + 1)`` mean of the augmented dynamics ``[A | d]``.
    row_cov : array_like
        ``(N, N)`` among-row covariance.
    col_cov : array_like
        ``(N + 1, N + 1)`` among-column covariance.
    state_dim : int, optional
        The model's latent state dimension ``N``. The MATLAB original reads it
        from ``obj.state_dim`` and validates ``mean`` against it; pass it here
        to get the same check. When omitted, ``N`` is inferred from ``mean``,
        which can only catch a wrong column count.

    Returns
    -------
    numpy.ndarray
        ``(N, N + 1)`` stable augmented dynamics with spectral radius < 1.
    """
    mean = np.atleast_2d(np.asarray(mean, dtype=float))
    n = mean.shape[0] if state_dim is None else int(state_dim)
    if mean.shape != (n, n + 1):
        raise ValueError(
            "RealTimeCOIN:sampleStableTheta:invalidMean: mean must be "
            "%d-by-%d ([A | d])." % (n, n + 1)
        )

    max_iter = 10
    margin = 0.01

    theta = np.array(mean, dtype=float, copy=True)   # default if sampling fails
    rho = np.inf
    for _ in range(max_iter):
        theta = sample_matrix_normal(rng, mean, row_cov, col_cov)
        rho = _spectral_radius(theta[:, :n])
        if np.isfinite(rho) and rho < 1:
            return theta

    if not np.isfinite(rho) or rho <= 0:
        # Degenerate draw: fall back to the prior/posterior mean, then apply
        # the same stability projection below if it is still needed.
        theta = np.array(mean, dtype=float, copy=True)
        rho = _spectral_radius(theta[:, :n])

    if not np.isfinite(rho) or rho <= 0:
        theta[:, :n] = np.zeros((n, n))
        return theta

    if rho < 1:
        return theta

    # rho(A / (rho + margin)) = rho / (rho + margin) < 1 strictly.
    theta[:, :n] = theta[:, :n] / (rho + margin)
    return theta


def _spectral_radius(a):
    """Largest eigenvalue magnitude of ``a``; ``inf`` if it cannot be found."""
    if a.size == 0:
        return 0.0
    try:
        return float(np.max(np.abs(np.linalg.eigvals(a))))
    except np.linalg.LinAlgError:
        return np.inf

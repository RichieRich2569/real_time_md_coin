"""Deterministic numerical helpers for the RealTimeCOIN particle filter.

Translated from ``@RealTimeCOIN/private/``: positive-definite repair, Cholesky
factorisation with escalating jitter, guarded division/log/inverse, Gaussian
log-likelihoods, probability normalisation, stationary moments of the latent
AR(1) process and the Jeffreys divergences.

Every threshold and fallback branch of the MATLAB sources is preserved. The
MATLAB constants map onto numpy as

* ``eps``     -> ``numpy.finfo(float).eps``   (2.22e-16)
* ``realmin`` -> ``numpy.finfo(float).tiny``  (2.23e-308)
* ``realmax`` -> ``numpy.finfo(float).max``   (1.80e308)

MATLAB's two-argument ``max(a, b)`` / ``min(a, b)`` OMIT ``NaN`` (they return
the other operand), whereas ``numpy.maximum`` / ``numpy.minimum`` propagate it.
Wherever a MATLAB source uses those to floor or clamp data that may legally be
``NaN``, the faithful translation is ``numpy.fmax`` / ``numpy.fmin``, which
omit ``NaN`` the same way. That is what keeps the graceful-degradation
fallbacks here finite instead of quietly poisoning particle weights.

Array-layout note: the MATLAB code is contexts-first / particles-last, this
package is particles-leading with matrix dimensions trailing. Functions that
reduce over the context axis therefore default to ``axis=-1`` rather than
MATLAB's leading dimension; the docstrings flag this where it matters.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .statics import normal_pdf

__all__ = [
    "ensure_pd",
    "choljitter",
    "regularize_covariance",
    "safe_inverse",
    "safe_divide",
    "safe_log",
    "gaussian_log_lik_chol",
    "gaussian_pdf_columns_md",
    "normalize_columns",
    "normalize_probability",
    "renormalize_global_weights",
    "precision_to_variance",
    "mixture_density_on_grid",
    "stationary_state_mean",
    "stationary_state_var",
    "stationary_state_mean_md",
    "stationary_state_cov_md",
    "categorical_jeffreys",
    "gaussian_jeffreys",
    "gaussian_jeffreys_multi",
    "jeffreys_finite_clip",
]

EPS = float(np.finfo(float).eps)
REALMIN = float(np.finfo(float).tiny)
REALMAX = float(np.finfo(float).max)


def _rcond(a):
    """Reciprocal 1-norm condition number, the analogue of MATLAB ``rcond``.

    Parameters
    ----------
    a : numpy.ndarray
        Square matrix.

    Returns
    -------
    float
        ``1 / cond(a, 1)``; ``0.0`` for a singular matrix and ``nan`` for a
        non-finite one, exactly as MATLAB's ``rcond`` behaves. The ``nan`` is
        deliberate: every caller tests ``rcond(a) < 1e-12`` and ``nan < x`` is
        false, so a non-finite matrix takes the direct-solve branch in Python
        just as it does in MATLAB, rather than being pushed into ``pinv``
        (which would raise instead of propagating ``nan``).
    """
    if a.size == 0:
        return 0.0
    if not np.all(np.isfinite(a)):
        return np.nan
    try:
        condition = np.linalg.cond(a, 1)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.isfinite(condition) or condition == 0:
        return 0.0
    return 1.0 / condition


def _try_cholesky_lower(a):
    """Lower Cholesky factor of ``a``, or ``None`` if it is not usable.

    A failure is reported as ``None`` so the caller can fall through to its
    jitter / diagonal fallback, which is how MATLAB's ``[L, flag] = chol(A)``
    two-output form behaves. Two numpy-specific hazards are folded into the
    same "failed" answer: it can raise ``ValueError`` on non-finite input, and
    on some platforms LAPACK returns a factor containing ``nan`` WITHOUT
    signalling failure - a silently poisoned factor that would blow up the
    triangular solves downstream.
    """
    try:
        factor = np.linalg.cholesky(a)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(factor)):
        return None
    return factor


def ensure_pd(mode, m):
    """Positive-definite repair backing the three particle-filter tactics.

    The three modes are numerically distinct and each is a verbatim copy of the
    body of its original caller, so no caller's output changes:

    ``"jitter"``
        Lower Cholesky factor with escalating diagonal jitter (backs
        :func:`choljitter`). Returns the factor and whether a genuine Cholesky
        succeeded.
    ``"load"``
        Zero non-finite entries, symmetrise and diagonally load to a workable
        condition number (backs :func:`regularize_covariance`).
    ``"eigclip"``
        Symmetrise and clip negative eigenvalues to zero, i.e. project onto
        the nearest PSD matrix (backs :func:`stationary_state_cov_md`).

    Parameters
    ----------
    mode : {'jitter', 'load', 'eigclip'}
        Which repair tactic to apply.
    m : array_like
        Square matrix to repair.

    Returns
    -------
    out : numpy.ndarray
        The repaired matrix (a Cholesky factor for ``"jitter"``).
    ok : bool
        Only meaningful for ``"jitter"``: ``False`` when the diagonal
        last-resort fallback was used. ``True`` otherwise.
    """
    m = np.asarray(m, dtype=float)

    if mode == "jitter":
        # --- verbatim choljitter body ---
        s = (m + m.T) / 2.0
        factor = _try_cholesky_lower(s)
        if factor is not None:
            return factor, True

        scale = np.mean(np.diag(s)) if s.size else np.nan
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        jit = 1e-12 * scale   # base jitter: 1e-12 relative to the mean diagonal
        for _ in range(8):
            factor = _try_cholesky_lower(s + jit * np.eye(s.shape[0]))
            if factor is not None:
                return factor, True
            jit *= 10.0

        # Last-resort diagonal fallback (treat the dimensions as independent).
        # fmax, not maximum: this branch is reached for non-finite input too,
        # and MATLAB's max(NaN, eps) is eps, which keeps the factor finite.
        d = np.fmax(np.diag(s), EPS)
        return np.diag(np.sqrt(d)), False

    if mode == "load":
        # --- verbatim regularizeCovariance body ---
        covar = np.array(m, dtype=float, copy=True)
        covar[~np.isfinite(covar)] = 0.0
        covar = (covar + covar.T) / 2.0   # enforce exact symmetry
        if covar.size == 0:
            # empty -> smallest positive scalar variance. MATLAB returns the
            # bare scalar eps; it is returned as a 1-by-1 matrix here so the
            # downstream matrix helpers (rcond, inv, chol) still accept it.
            return np.array([[EPS]]), True
        # Add eps on the diagonal so a zero / rank-deficient covariance becomes
        # strictly positive definite.
        covar = covar + EPS * np.eye(covar.shape[0])
        # rcond < 1e-12 flags a near-singular matrix; bump the diagonal by 1e-9
        # to restore a usable condition number.
        if _rcond(covar) < 1e-12:
            covar = covar + 1e-9 * np.eye(covar.shape[0])
        return covar, True

    if mode == "eigclip":
        # --- verbatim stationaryStateCovMD PSD-projection epilogue ---
        p = (m + m.T) / 2.0
        # Project onto the nearest PSD matrix by clipping negative eigenvalues.
        eigenvalues, eigenvectors = np.linalg.eigh(p)
        dvals = np.maximum(np.real(eigenvalues), 0.0)
        p = eigenvectors @ np.diag(dvals) @ eigenvectors.T
        return (p + p.T) / 2.0, True

    raise ValueError(
        "RealTimeCOIN:ensurePD:UnknownMode: Unknown PD-repair mode '%s'." % (mode,)
    )


def choljitter(s):
    """Lower-triangular Cholesky factor with a PSD-failure fallback.

    Particle-filter covariances occasionally drift very slightly non-positive
    definite through accumulated round-off. Rather than erroring, ``s`` is
    symmetrised and given escalating diagonal jitter until the factorisation
    succeeds; if even that fails the diagonal square root is returned so
    downstream code degrades gracefully.

    Parameters
    ----------
    s : array_like
        ``(N, N)`` symmetric (near-)positive-definite matrix.

    Returns
    -------
    l : numpy.ndarray
        Lower-triangular factor with ``l @ l.T ~= s``.
    ok : bool
        ``True`` if a genuine Cholesky succeeded, ``False`` for the diagonal
        fallback.
    """
    return ensure_pd("jitter", s)


def regularize_covariance(covar):
    """Symmetrise and condition a covariance matrix.

    Zeroes non-finite entries, symmetrises, and adds diagonal loading to
    guarantee positive definiteness and a workable condition number. Used
    before :func:`safe_inverse` in the Jeffreys / multi-dimensional Kalman
    paths.

    Parameters
    ----------
    covar : array_like
        ``(N, N)`` covariance matrix.

    Returns
    -------
    numpy.ndarray
        The conditioned covariance.
    """
    out, _ = ensure_pd("load", covar)
    return out


def safe_inverse(a):
    """Invert a square matrix, falling back to the pseudo-inverse.

    Returns ``inv(a)`` for well-conditioned ``a`` and ``pinv(a)`` when ``a`` is
    numerically singular (``rcond(a) < 1e-12``). This guards the
    multi-dimensional Kalman / precision updates against blow-up when a
    covariance degenerates.

    Parameters
    ----------
    a : array_like
        ``(N, N)`` matrix to invert.

    Returns
    -------
    numpy.ndarray
        The (pseudo-)inverse of ``a``.
    """
    a = np.asarray(a, dtype=float)
    if _rcond(a) < 1e-12:
        return np.linalg.pinv(a)
    return np.linalg.inv(a)


def safe_divide(a, b):
    """Element-wise division that returns zero where the divisor is ~0.

    Parameters
    ----------
    a : array_like
        Numerator.
    b : array_like
        Denominator; entries with ``abs(b) <= eps`` yield 0 instead of
        ``inf`` / ``nan``.

    Returns
    -------
    numpy.ndarray
        ``a / b`` with the guarded entries set to zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.abs(b) > EPS
    return np.where(good, a / np.where(good, b, 1.0), 0.0)


def safe_log(x):
    """Natural logarithm floored at ``realmin`` to avoid ``log(0) = -inf``.

    Parameters
    ----------
    x : array_like
        Values to take the logarithm of.

    Returns
    -------
    numpy.ndarray
        ``log(max(x, realmin))``, so exact zeros map to a large finite
        negative number rather than ``-inf``.
    """
    return np.log(np.fmax(np.asarray(x, dtype=float), REALMIN))


def gaussian_log_lik_chol(y_tilde, s):
    """Stable multivariate Gaussian log-likelihood of an innovation.

    Returns ``log N(y_tilde | 0, s)`` evaluated through the Cholesky factor of
    ``s`` (no explicit inverse). With ``s = L L'``,

    ``log|s| = 2 sum(log(diag(L)))`` and
    ``y' s^-1 y = (L \\ y)' (L \\ y)``, so

    ``log N(y | 0, s) = -0.5 (M log(2 pi) + log|s| + mahalanobis)``.

    At ``M == 1`` this reduces to ``log(normal_pdf(...))``.

    Parameters
    ----------
    y_tilde : array_like
        Length-``M`` innovation (measurement residual).
    s : array_like
        ``(M, M)`` innovation covariance.

    Returns
    -------
    float
        The Gaussian log-density of ``y_tilde``.
    """
    y_tilde = np.asarray(y_tilde, dtype=float).reshape(-1)
    m = y_tilde.size
    factor, ok = choljitter(s)
    if not ok:
        # choljitter fell back to a diagonal factor; treat the dimensions as
        # independent, which still yields a finite, well-ordered likelihood.
        factor = np.diag(np.fmax(np.diag(factor), np.sqrt(EPS)))
    log_det_s = 2.0 * np.sum(np.log(np.diag(factor)))
    # check_finite=False so a non-finite residual propagates as nan, the way
    # MATLAB's L \ yTilde does, instead of raising.
    foo = solve_triangular(factor, y_tilde, lower=True, check_finite=False)
    mahalanobis = float(foo @ foo)
    return -0.5 * (m * np.log(2.0 * np.pi) + log_det_s + mahalanobis)


def gaussian_pdf_columns_md(points, mean, cov):
    """Multivariate Gaussian pdf evaluated at many query points.

    Vectorised, grid-evaluating counterpart of :func:`gaussian_log_lik_chol`:
    ``cov`` is factored once (``cov = L L'``) and the factor is reused across
    all query points, so the per-point work is a triangular solve rather than a
    fresh O(N^3) factorisation.

    ``log N(x | m, S) = -0.5 (N log(2 pi) + log|S| + r' S^-1 r)`` with
    ``r = x - m``.

    Layout note: the MATLAB original takes an ``N`` by ``K`` array whose
    COLUMNS are the query points. Following this package's particles-leading
    convention, ``points`` here is ``(K, N)`` with query points along the
    leading axis.

    Parameters
    ----------
    points : array_like
        ``(K, N)`` query points (one point per row).
    mean : array_like
        Length-``N`` mean vector.
    cov : array_like
        ``(N, N)`` covariance matrix.

    Returns
    -------
    numpy.ndarray
        Length-``K`` densities; non-finite entries are replaced by
        ``realmax``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    mean = np.asarray(mean, dtype=float).reshape(-1)
    n = points.shape[1]

    factor, _ = choljitter(cov)              # diagonal fallback is PSD-safe
    log_det_s = 2.0 * np.sum(np.log(np.diag(factor)))
    residuals = points - mean                # (K, N)
    foo = solve_triangular(
        factor, residuals.T, lower=True, check_finite=False
    )                                        # (N, K)
    mahalanobis = np.sum(foo ** 2, axis=0)   # (K,)
    log_pdf = -0.5 * (n * np.log(2.0 * np.pi) + log_det_s + mahalanobis)
    with np.errstate(over="ignore"):
        d = np.exp(log_pdf)
    # sentinel: largest finite double on overflow
    return np.where(np.isfinite(d), d, REALMAX)


def normalize_columns(x, axis=-1):
    """Normalise slices of an array to sum to one.

    A slice whose sum is zero or non-finite collapses to the unit vector
    ``e_1`` (all mass on the first entry), which keeps context-weight slices
    valid even for degenerate particles.

    Layout note: the MATLAB original normalises the COLUMNS of a
    ``(contexts, particles)`` matrix. Under this package's ``(particles,
    contexts)`` layout the same reduction is over the last axis, hence the
    ``axis=-1`` default.

    Parameters
    ----------
    x : array_like
        Array of non-negative weights.
    axis : int, optional
        Axis to normalise over (default ``-1``, the context axis).

    Returns
    -------
    numpy.ndarray
        Array of the same shape whose slices along ``axis`` sum to one.
    """
    x = np.asarray(x, dtype=float)
    moved = np.moveaxis(x, axis, -1)
    moved_shape = moved.shape
    flat = moved.reshape(-1, moved_shape[-1])   # (slices, length)

    sums = flat.sum(axis=1)
    good = (sums > 0) & np.isfinite(sums)

    out = np.zeros_like(flat)
    out[good] = flat[good] / sums[good][:, None]
    bad = ~good
    if np.any(bad):
        out[bad, 0] = 1.0   # degenerate slice collapses to e_1
    return np.moveaxis(out.reshape(moved_shape), -1, axis)


def normalize_probability(p):
    """Normalise a vector into a strictly positive probability vector.

    Non-finite and negative entries are treated as zero; an all-zero (or empty)
    input falls back to the uniform distribution. Every entry is then floored
    at ``realmin`` and the vector renormalised, so downstream ``log`` and
    division operations never see an exact zero.

    Parameters
    ----------
    p : array_like
        Unnormalised weights (any shape; flattened).

    Returns
    -------
    numpy.ndarray
        Length-``N`` probability vector summing to one, all entries > 0.
    """
    p = np.asarray(p, dtype=float).reshape(-1).copy()
    p[~np.isfinite(p) | (p < 0)] = 0.0
    total = p.sum()
    if total <= 0:
        p = np.ones(p.shape) / max(p.size, 1)
    else:
        p = p / total
    p = np.fmax(p, REALMIN)
    return p / p.sum()


def renormalize_global_weights(w):
    """Zero non-finite entries and renormalise to sum to one.

    Unlike :func:`normalize_probability` there is no uniform fallback and no
    ``realmin`` flooring, so an all-zero (or all-non-finite) input is returned
    all-zero: an empty franchise must stay empty.

    Parameters
    ----------
    w : array_like
        Unnormalised global weights.

    Returns
    -------
    numpy.ndarray
        The weights, summing to one when their mass is strictly positive.
    """
    w = np.array(w, dtype=float, copy=True)
    w[~np.isfinite(w)] = 0.0
    total = w.sum()
    if total > 0:
        w = w / total
    return w


def precision_to_variance(precision):
    """Convert a precision to a variance, mapping precision 0 to ``inf``.

    A precision of exactly zero encodes a completely uninformative belief, so
    the variance is ``inf`` rather than a division by zero.

    Parameters
    ----------
    precision : array_like
        Precision(s). The MATLAB original is scalar-only; this version is
        element-wise, which is a strict superset of that behaviour.

    Returns
    -------
    numpy.ndarray
        ``1 / precision`` with zeros mapped to ``inf``.
    """
    precision = np.asarray(precision, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(precision == 0, np.inf, 1.0 / np.where(precision == 0, 1.0, precision))


def mixture_density_on_grid(
    values, weights, means, variances, normalizer, state_dim, method_name="mixture_density_on_grid"
):
    """Weighted Gaussian-mixture density evaluated on a query grid.

    Evaluates ``p(x) = (1 / normalizer) sum_p sum_c W[p, c] N(x | M[p, c],
    V[p, c])``, dispatching on ``state_dim``. The double loop runs ``p``
    (outer) then ``c`` (inner), skipping zero-weight components, so the
    accumulation order matches the MATLAB original. The division by
    ``normalizer`` is applied only when ``normalizer > 0``, so an empty mixture
    returns the zero density unchanged.

    Layout note: MATLAB stores ``W`` as ``(contexts, particles)`` and the MD
    grid as ``(N, K)``; here ``weights`` is ``(P, C)``, ``means`` is ``(P, C)``
    (scalar model) or ``(P, C, N)``, ``variances`` is ``(P, C)`` or
    ``(P, C, N, N)``, and the MD grid is ``(K, N)``.

    Parameters
    ----------
    values : array_like
        Query grid: length-``K`` for the scalar model, ``(K, N)`` for the
        multi-dimensional model.
    weights : array_like
        ``(P, C)`` mixing weights.
    means : array_like
        ``(P, C)`` or ``(P, C, N)`` component means.
    variances : array_like
        ``(P, C)`` variances or ``(P, C, N, N)`` covariances.
    normalizer : float
        Divisor applied to the accumulated density when strictly positive
        (normally the number of particles).
    state_dim : int
        Latent state dimension; ``1`` selects the scalar path.
    method_name : str, optional
        Caller name, used only in the grid-dimension-mismatch message.

    Returns
    -------
    numpy.ndarray
        Length-``K`` mixture density on the grid.
    """
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    num_particles, num_contexts = weights.shape

    if state_dim == 1:
        # MATLAB's values(:) is a column-major flatten; match it so a matrix
        # grid enumerates points in the same order as every density query.
        values = np.asarray(values, dtype=float).reshape(-1, order="F")
        densities = np.zeros(values.shape)
        for p in range(num_particles):
            for c in range(num_contexts):
                if weights[p, c] > 0:
                    densities = densities + weights[p, c] * normal_pdf(
                        values, means[p, c], variances[p, c]
                    )
        if normalizer > 0:
            densities = densities / normalizer
        return densities

    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != state_dim:
        raise ValueError(
            "RealTimeCOIN:GridDimensionMismatch: %s expects a K-by-%d grid "
            "(each row a query point) for state_dim == %d; received a "
            "%d-by-%d array." % (method_name, state_dim, state_dim, values.shape[0], values.shape[1])
        )
    densities = np.zeros(values.shape[0])
    for p in range(num_particles):
        for c in range(num_contexts):
            if weights[p, c] > 0:
                densities = densities + weights[p, c] * gaussian_pdf_columns_md(
                    values, means[p, c], variances[p, c]
                )
    if normalizer > 0:
        densities = densities / normalizer
    return densities


def stationary_state_mean(a, d):
    """Stationary mean of the scalar latent AR(1) process.

    For ``s_i = a s_{i-1} + d + w`` the stationary mean solves ``m = a m + d``,
    i.e. ``m = d / (1 - a)``. Only entries with ``abs(1 - a) > eps`` are
    divided: a retention at or above 1 has no finite stationary mean and is
    left at zero.

    Parameters
    ----------
    a : array_like
        Retention parameter(s).
    d : array_like
        Drift parameter(s), broadcastable against ``a``.

    Returns
    -------
    numpy.ndarray
        Element-wise stationary mean, zero where undefined.
    """
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    denom = 1.0 - a
    good = np.abs(denom) > EPS
    return np.where(good, d / np.where(good, denom, 1.0), 0.0)


def stationary_state_var(a, sigma_process_noise):
    """Stationary variance of the scalar latent AR(1) process.

    For ``s_i = a s_{i-1} + w`` with ``w ~ N(0, sigma^2)`` the stationary
    variance solves ``v = a^2 v + sigma^2``, i.e. ``v = sigma^2 / (1 - a^2)``.
    Only entries with ``1 - a^2 > eps`` are divided; ``abs(a) >= 1`` has no
    finite stationary variance and is left at zero.

    Parameters
    ----------
    a : array_like
        Retention parameter(s).
    sigma_process_noise : float
        Process-noise standard deviation.

    Returns
    -------
    numpy.ndarray
        Element-wise stationary variance, zero where undefined.
    """
    a = np.asarray(a, dtype=float)
    denom = 1.0 - a ** 2
    good = denom > EPS
    return np.where(
        good, sigma_process_noise ** 2 / np.where(good, denom, 1.0), 0.0
    )


def stationary_state_mean_md(a_matrix, d):
    """Stationary mean of the multi-dimensional latent AR(1) process.

    For ``s_i = A s_{i-1} + d + w_i`` the stationary mean solves ``m = A m +
    d``, i.e. ``m = (I - A) \\ d``. If ``I - A`` is near-singular (a unit
    eigenvalue, ``rcond < 1e-12``) the pseudo-inverse is used so a new context
    is still seeded with a finite mean instead of ``inf`` / ``nan``.

    Parameters
    ----------
    a_matrix : array_like
        ``(N, N)`` dynamics matrix.
    d : array_like
        Length-``N`` drift vector.

    Returns
    -------
    numpy.ndarray
        Length-``N`` stationary mean.
    """
    a_matrix = np.asarray(a_matrix, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    n = a_matrix.shape[0]
    i_minus_a = np.eye(n) - a_matrix
    if _rcond(i_minus_a) < 1e-12:
        return np.linalg.pinv(i_minus_a) @ d
    return np.linalg.solve(i_minus_a, d)


def stationary_state_cov_md(a_matrix, q):
    """Stationary covariance of the multi-dimensional latent AR(1) process.

    For ``s_i = A s_{i-1} + d + w_i`` with ``w_i ~ N(0, Q)`` the stationary
    covariance solves the discrete Lyapunov equation ``P = A P A' + Q``.
    Vectorising with ``vec(A X B) = (B' kron A) vec(X)`` gives
    ``vec(P) = (I - A kron A) \\ vec(Q)``, solved directly so no Control
    System Toolbox equivalent is needed. ``vec`` is column-major here (as in
    MATLAB), hence the Fortran-order flatten/reshape.

    If the system sits at or beyond the stability boundary the Kronecker system
    is singular (``rcond < 1e-12``); the pseudo-inverse is used and the result
    is symmetrised with negative eigenvalues clipped, so the output is always a
    valid covariance.

    Parameters
    ----------
    a_matrix : array_like
        ``(N, N)`` dynamics matrix.
    q : array_like
        ``(N, N)`` process-noise covariance.

    Returns
    -------
    numpy.ndarray
        ``(N, N)`` stationary state covariance.
    """
    a_matrix = np.asarray(a_matrix, dtype=float)
    q = np.asarray(q, dtype=float)
    n = a_matrix.shape[0]

    # vec(P) = (I - A kron A) \ vec(Q)
    kronecker = np.eye(n ** 2) - np.kron(a_matrix, a_matrix)
    qvec = q.flatten(order="F")
    if _rcond(kronecker) < 1e-12:
        pvec = np.linalg.pinv(kronecker) @ qvec
    else:
        pvec = np.linalg.solve(kronecker, qvec)
    p = pvec.reshape(n, n, order="F")

    out, _ = ensure_pd("eigclip", p)
    return out


def categorical_jeffreys(p, q):
    """Jeffreys divergence between two categorical distributions.

    ``d = sum((p - q) * log(p / q)) = KL(p||q) + KL(q||p) >= 0``. The shorter
    vector is zero-padded to the common support and both are renormalised
    (:func:`normalize_probability`) before comparison.

    Parameters
    ----------
    p, q : array_like
        Categorical probability vectors (need not be normalised or the same
        length).

    Returns
    -------
    float
        The symmetric divergence, clipped to be finite and non-negative.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    n = max(p.size, q.size)
    p = np.concatenate([p, np.zeros(n - p.size)])   # zero-pad to common support
    q = np.concatenate([q, np.zeros(n - q.size)])
    p = normalize_probability(p)
    q = normalize_probability(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sum((p - q) * np.log(p / q))
    return jeffreys_finite_clip(d)


def gaussian_jeffreys(m1, v1, m2, v2):
    """Jeffreys divergence between two scalar Gaussians.

    ``d = 0.5 (v1/v2 + v2/v1 + (m1 - m2)^2 (1/v1 + 1/v2) - 2) >= 0``, the
    symmetrised KL divergence ``KL(1||2) + KL(2||1)``. A non-finite (e.g.
    novel-context) variance is treated as maximally diffuse, ``1/eps``, and
    variances are floored at ``eps`` to keep the reciprocals finite.

    Parameters
    ----------
    m1, m2 : array_like
        Means.
    v1, v2 : array_like
        Variances.

    Returns
    -------
    numpy.ndarray
        Element-wise divergence, clipped to be finite and non-negative.
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    v1 = np.where(np.isfinite(v1), v1, 1.0 / EPS)
    v2 = np.where(np.isfinite(v2), v2, 1.0 / EPS)
    v1 = np.fmax(v1, EPS)
    v2 = np.fmax(v2, EPS)
    with np.errstate(over="ignore", invalid="ignore"):
        d = 0.5 * (v1 / v2 + v2 / v1 + (m1 - m2) ** 2 * (1.0 / v1 + 1.0 / v2) - 2.0)
    return jeffreys_finite_clip(d)


def gaussian_jeffreys_multi(m1, s1, m2, s2):
    """Jeffreys divergence between two multivariate Gaussians.

    ``d = 0.5 (trace(s2^-1 s1 + s1^-1 s2) + delta' (s1^-1 + s2^-1) delta -
    2k)`` with ``delta = m1 - m2``; it collapses to :func:`gaussian_jeffreys`
    at ``k == 1``. Covariances are regularised
    (:func:`regularize_covariance`) and inverted stably
    (:func:`safe_inverse`), so slightly non-PD particle covariances do not
    throw.

    Parameters
    ----------
    m1, m2 : array_like
        Length-``k`` mean vectors.
    s1, s2 : array_like
        ``(k, k)`` covariance matrices.

    Returns
    -------
    float
        The symmetric divergence, clipped to be finite and non-negative.
    """
    m1 = np.asarray(m1, dtype=float).reshape(-1)
    m2 = np.asarray(m2, dtype=float).reshape(-1)
    s1 = regularize_covariance(s1)
    s2 = regularize_covariance(s2)
    inv1 = safe_inverse(s1)
    inv2 = safe_inverse(s2)
    delta = m1 - m2
    k = m1.size
    d = 0.5 * (
        np.trace(inv2 @ s1 + inv1 @ s2) + delta @ (inv1 + inv2) @ delta - 2.0 * k
    )
    return jeffreys_finite_clip(d)


def jeffreys_finite_clip(d):
    """Finite-clip epilogue shared by the Jeffreys divergences.

    A non-finite divergence is replaced by the finite sentinel ``realmax``,
    then the result is clipped at zero because a divergence is non-negative.

    Parameters
    ----------
    d : array_like
        Raw divergence value(s).

    Returns
    -------
    float or numpy.ndarray
        Finite, non-negative divergence; a Python float for scalar input.
    """
    d = np.asarray(d, dtype=float)
    # sentinel: "infinitely divergent" but still finite
    d = np.where(np.isfinite(d), d, REALMAX)
    d = np.fmax(d, 0.0)      # divergence is non-negative; clip round-off
    if d.ndim == 0:
        return float(d)
    return d

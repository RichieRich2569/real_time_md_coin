"""Multi-dimensional (``state_dim > 1``) per-trial particle-filter pipeline.

Translated from ``@RealTimeCOIN/private/``: ``predictStatesMD``,
``predictStateFeedbackMD``, ``resampleParticlesMD``, ``resampleStateMD``,
``sampleContextMD``, ``updateBeliefAboutStatesMD``, ``sampleStatesMD``,
``updateSufficientStatisticsMD``, ``sampleParametersMD``, ``sampleDynamicsMD``
and ``sampleBiasMD``.

Step-for-step parallel to :mod:`realtimecoin.pipeline_scalar` - the per-trial
order is identical, only the algebra generalises: the latent state is an
``N``-vector with a full ``(N, N)`` covariance and the dynamics are the
augmented matrix ``Theta = [A | d]`` with a matrix-normal conjugate prior (see
:func:`realtimecoin.state.dynamics_prior_md`). The sufficient statistics are
kept in information form (``Lambda_xx``, ``Lambda_yx``, ``bias_info_ss``,
``bias_precision_ss``).

The MD steps additionally take ``obs_mask``, a ``(N,)`` boolean array marking
which entries of ``y`` were actually observed, so a partially observed
feedback vector updates only the dimensions it carries.

Array layout is particles-leading throughout; see :mod:`realtimecoin.state` for
the field-by-field shape table.

Batched linear algebra
----------------------
MATLAB walks the ``(N, N, Cmax, P)`` tensors page by page (or with
``pagemtimes``). Under the particles-leading layout every such page is the
trailing ``(N, N)`` block of a ``(P, C, N, N)`` array, so ``@``,
:func:`numpy.linalg.cholesky` and :func:`numpy.linalg.solve` broadcast over the
leading ``(P, C)`` axes natively and reproduce the per-page result exactly.
Every vectorised site below names the MATLAB loop it replaces. The three
helpers :func:`_safe_inverse_batch`, :func:`_chol_batch` and
:func:`_right_solve` keep the *degenerate* branches of the scalar helpers
(``safe_inverse``'s pinv fallback, ``choljitter``'s escalating jitter) by
falling back to a per-page loop whenever the fast path is not exactly
equivalent.

RNG-stream note
---------------
The Gaussian draws of :func:`sample_states_md` and :func:`sample_bias_md` are
taken as one block per stage rather than one ``randn(N, 1)`` per
``(particle, context)`` cell, and ``drawGaussian``'s zero-covariance
short-circuit consumes variates here where MATLAB consumes none. The
DISTRIBUTION is unchanged and a given ``model.rng`` seed is fully reproducible,
but the variate ORDER differs from MATLAB - the two implementations are not
stream-identical (they cannot be anyway, MATLAB and numpy using different
generators).
"""

from __future__ import annotations

import numpy as np

from .context import (
    sample_global_cue_probabilities,
    sample_global_transition_probabilities,
    update_local_cue_matrix,
    update_local_transition_matrix,
)
from .numerics import (
    choljitter,
    gaussian_log_lik_chol,
    normalize_columns,
    safe_inverse,
    safe_log,
    stationary_state_cov_md,
    stationary_state_mean_md,
)
from .samplers import beta_sample, sample_stable_theta
from .state import (
    dynamics_prior_md,
    ensure_cue_column,
    feedback_transform,
    instantiate_cue_if_needed,
    observation_noise_cov,
    process_noise_cov,
)
from .statics import log_sum_exp, systematic_resampling

__all__ = [
    "predict_states_md",
    "predict_state_feedback_md",
    "resample_particles_md",
    "resample_state_md",
    "sample_context_md",
    "update_belief_about_states_md",
    "sample_states_md",
    "update_sufficient_statistics_md",
    "sample_parameters_md",
    "sample_dynamics_md",
    "sample_bias_md",
]

#: ``safe_inverse``'s near-singularity threshold, repeated here so the batched
#: fast path routes exactly the same matrices to the pseudo-inverse.
RCOND_TOL = 1e-12


# --------------------------------------------------------------------------
# Batched linear-algebra helpers
# --------------------------------------------------------------------------


def _symmetrise(m):
    """Symmetrise the trailing ``(N, N)`` block of every page.

    ``(M + M') / 2`` applied over the leading axes, the batched form of the
    ``(Pf + Pf') ./ 2`` round-off guards that appear throughout the MATLAB MD
    sources.

    Parameters
    ----------
    m : numpy.ndarray
        ``(..., N, N)`` array of square matrices.

    Returns
    -------
    numpy.ndarray
        Same shape, symmetric in the trailing two axes.
    """
    return (m + np.swapaxes(m, -1, -2)) / 2.0


def _matrix_norm_1(m):
    """1-norm (max absolute column sum) of every trailing ``(N, N)`` page."""
    return np.max(np.sum(np.abs(m), axis=-2), axis=-1)


def _safe_inverse_batch(mats):
    """Page-wise :func:`realtimecoin.numerics.safe_inverse` over leading axes.

    Vectorises the ``safeInverse`` calls inside MATLAB's ``for p ... for c``
    loops (``sampleStatesMD``, ``sampleDynamicsMD``, ``sampleBiasMD``). The fast
    path is a single batched LU inverse; the reciprocal 1-norm condition number
    is then formed page-wise from that inverse (``rcond = 1 / (||M||_1
    ||M^-1||_1)``, exactly what ``safe_inverse`` tests) and any page below
    :data:`RCOND_TOL`, or any non-finite page, is redone through
    :func:`~realtimecoin.numerics.safe_inverse` so its pseudo-inverse fallback
    is preserved.

    Parameters
    ----------
    mats : numpy.ndarray
        ``(..., N, N)`` array of matrices to invert.

    Returns
    -------
    numpy.ndarray
        Same shape; page ``i`` is ``safe_inverse(mats[i])``.
    """
    mats = np.asarray(mats, dtype=float)
    n = mats.shape[-1]
    flat = mats.reshape(-1, n, n)
    out = np.empty_like(flat)

    inv_flat = None
    if np.all(np.isfinite(flat)):
        try:
            inv_flat = np.linalg.inv(flat)
        except np.linalg.LinAlgError:
            inv_flat = None

    if inv_flat is None or not np.all(np.isfinite(inv_flat)):
        # Slow path: any exactly singular / non-finite page. Reproduce
        # safe_inverse page by page rather than guessing which pages are bad.
        for i in range(flat.shape[0]):
            out[i] = safe_inverse(flat[i])
        return out.reshape(mats.shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        cond = _matrix_norm_1(flat) * _matrix_norm_1(inv_flat)
        rcond = np.where(np.isfinite(cond) & (cond > 0), 1.0 / cond, 0.0)
    good = rcond >= RCOND_TOL
    out[good] = inv_flat[good]
    for i in np.flatnonzero(~good):
        out[i] = safe_inverse(flat[i])       # pinv fallback, per safe_inverse
    return out.reshape(mats.shape)


def _chol_batch(cov, exact_zero=False):
    """Page-wise lower Cholesky factors, with the ``choljitter`` fallback.

    Vectorises the per-cell ``[L, ~] = choljitter(Sigma)`` of ``sampleStatesMD``
    and ``sampleBiasMD``. Every page is symmetrised first (as ``choljitter``
    does), then one batched :func:`numpy.linalg.cholesky` is attempted; if ANY
    page is not positive definite numpy raises for the whole batch, so the whole
    batch falls back to a per-page :func:`~realtimecoin.numerics.choljitter`
    loop and its escalating-jitter / diagonal behaviour is preserved exactly.

    Parameters
    ----------
    cov : numpy.ndarray
        ``(..., N, N)`` covariance pages.
    exact_zero : bool, optional
        When ``True``, a page whose covariance is EXACTLY zero gets a zero
        factor. That reproduces ``drawGaussian``'s ``all(Sigma(:) == 0)``
        short-circuit (``x = mu``): without it ``choljitter`` would return a
        ``1e-6 * I`` jittered factor and the draw would be perturbed. Default
        ``False``, which is what ``sampleBiasMD`` wants - it calls
        ``choljitter`` unconditionally, jitter included.

    Returns
    -------
    numpy.ndarray
        ``(..., N, N)`` lower-triangular factors with ``L @ L.T ~= cov``.
    """
    cov = _symmetrise(np.asarray(cov, dtype=float))
    n = cov.shape[-1]
    flat = cov.reshape(-1, n, n)

    zero = (
        np.all(flat == 0.0, axis=(1, 2))
        if exact_zero
        else np.zeros(flat.shape[0], dtype=bool)
    )
    work = flat
    if np.any(zero):
        # Substitute I on the short-circuited pages so they cannot fail the
        # batched factorisation; their factor is zeroed again below.
        work = flat.copy()
        work[zero] = np.eye(n)

    factors = None
    try:
        factors = np.linalg.cholesky(work)
    except (np.linalg.LinAlgError, ValueError):
        factors = None
    if factors is None or not np.all(np.isfinite(factors)):
        factors = np.empty_like(work)
        for i in range(work.shape[0]):
            factors[i], _ = choljitter(work[i])

    if np.any(zero):
        factors[zero] = 0.0
    return factors.reshape(cov.shape)


def _draw_gaussian_batch(rng, mean, cov, exact_zero=False):
    """Draw ``x = mu + L z`` for every leading cell in one block.

    Batched form of ``drawGaussian`` in ``sampleStatesMD``. See the module
    docstring for the RNG-stream caveat.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    mean : numpy.ndarray
        ``(..., N)`` means.
    cov : numpy.ndarray
        ``(..., N, N)`` covariances.
    exact_zero : bool, optional
        Forwarded to :func:`_chol_batch`; ``True`` preserves the zero-covariance
        short-circuit ``x == mu``.

    Returns
    -------
    numpy.ndarray
        ``(..., N)`` draws.
    """
    factors = _chol_batch(cov, exact_zero=exact_zero)          # (..., N, N)
    z = rng.standard_normal(mean.shape)                        # (..., N)
    return mean + (factors @ z[..., None])[..., 0]


def _right_solve(a, s):
    """Batched right matrix division ``a / s`` (i.e. ``a @ inv(s)``).

    MATLAB's ``Pp(:, obsIdx) / S`` in ``updateBeliefAboutStatesMD``, batched
    over particles as ``(S' \\ a')'``.

    Deviation: MATLAB's ``/`` on a singular ``S`` warns and yields ``inf`` /
    ``nan``; here a raised :class:`numpy.linalg.LinAlgError` (or a non-finite
    result) falls back to ``a @ safe_inverse(s)``, whose pseudo-inverse keeps
    the Kalman gain finite. Strictly more robust, never less accurate.

    The fallback is applied PER PAGE. Recomputing the whole batch would hand
    the healthy particles ``a @ inv(S)`` instead of a triangular solve - a
    ~1e-17 perturbation, but one that MATLAB's per-particle loop cannot
    produce, so a single degenerate particle would nudge all the others.

    Parameters
    ----------
    a : numpy.ndarray
        ``(..., N, K)`` numerator pages.
    s : numpy.ndarray
        ``(..., K, K)`` denominator pages.

    Returns
    -------
    numpy.ndarray
        ``(..., N, K)`` result.
    """
    a = np.asarray(a, dtype=float)
    s = np.asarray(s, dtype=float)
    try:
        out = np.swapaxes(
            np.linalg.solve(np.swapaxes(s, -1, -2), np.swapaxes(a, -1, -2)), -1, -2
        )
    except np.linalg.LinAlgError:
        # numpy raises for the WHOLE batch when any single page is singular, so
        # this tells us nothing about which page failed - fall through and redo
        # every page individually below.
        out = None
    if out is not None and np.all(np.isfinite(out)):
        return out

    n_rows, n_cols = a.shape[-2], s.shape[-1]
    flat_a = a.reshape(-1, n_rows, n_cols)
    flat_s = s.reshape(-1, n_cols, n_cols)
    if out is None:
        flat_out = np.empty((flat_a.shape[0], n_rows, n_cols))
        redo = np.arange(flat_a.shape[0])
    else:
        flat_out = out.reshape(-1, n_rows, n_cols).copy()
        redo = np.flatnonzero(~np.all(np.isfinite(flat_out), axis=(1, 2)))

    for i in redo:
        page = None
        try:
            page = np.linalg.solve(flat_s[i].T, flat_a[i].T).T
        except np.linalg.LinAlgError:
            page = None
        if page is None or not np.all(np.isfinite(page)):
            page = flat_a[i] @ safe_inverse(flat_s[i])
        flat_out[i] = page
    return flat_out.reshape(a.shape[:-2] + (n_rows, n_cols))


def _resolve_mask(model, y, obs_mask):
    """Resolve the per-dimension observation mask.

    Mirrors the ``if nargin < k || isempty(obsMask), obsMask = ~isnan(y(:))``
    preamble every MD step opens with.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``state_dim``.
    y : numpy.ndarray or None
        ``(N,)`` observed state feedback, or ``None``.
    obs_mask : array_like or None
        Explicit mask, or ``None`` to derive it from ``y``.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` boolean mask.
    """
    if obs_mask is not None:
        return np.asarray(obs_mask, dtype=bool).reshape(-1)
    if y is None:
        return np.zeros(model.state_dim, dtype=bool)
    return ~np.isnan(np.asarray(y, dtype=float).reshape(-1))


def _has_observation(y, obs_mask):
    """``~isempty(y) && any(obsMask)`` - is anything observed this trial?"""
    return y is not None and bool(np.any(obs_mask))


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------


def predict_states_md(model):
    """Propagate every context's latent state one step through ``Theta``.

    ``m <- A m + d`` and ``S <- A S A' + Q``, with each particle's novel slot
    re-seeded at the stationary moments of its freshly sampled ``Theta``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.state_mean`` ``(P, C, N)`` and
        ``D.state_cov`` ``(P, C, N, N)``.

    Returns
    -------
    None
    """
    n = model.state_dim
    p_count = model.num_particles
    q_cov = process_noise_cov(model)                     # (N, N)
    d = model.D

    # Batched Kalman time update. Vectorises MATLAB's pagemtimes over the
    # (N, N, Cmax, P) pages; here the (P, C) batch axes lead.
    a_mat = d.Theta[:, :, :, :n]                         # (P, C, N, N)
    drift = d.Theta[:, :, :, n]                          # (P, C, N)

    # s_{i|i-1} = A s_{i-1|i-1} + d
    d.state_mean = (
        a_mat @ d.state_filtered_mean[..., None]
    )[..., 0] + drift                                    # (P, C, N)
    # P_{i|i-1} = A P_{i-1|i-1} A' + Q   (Q broadcasts over the batch axes)
    p_pred = (
        a_mat @ d.state_filtered_cov @ np.swapaxes(a_mat, -1, -2) + q_cov
    )                                                    # (P, C, N, N)
    d.state_cov = _symmetrise(p_pred)                    # symmetrise vs round-off

    # Re-seed each particle's novel context slot to its stationary distribution
    # (overwrites the generic propagation above for that slot). Slot n_active[p]
    # is the novel one; MATLAB writes it as C(p) + 1 in 1-based labels.
    for p in range(p_count):
        k_active = int(d.n_active[p])
        if k_active < model.max_contexts:
            novel = k_active
            a_novel = d.Theta[p, novel, :, :n]           # (N, N)
            d_novel = d.Theta[p, novel, :, n]            # (N,)
            d.state_mean[p, novel] = stationary_state_mean_md(a_novel, d_novel)
            d.state_cov[p, novel] = stationary_state_cov_md(a_novel, q_cov)


def predict_state_feedback_md(model):
    """Map the predicted state moments into observation space.

    ``state_feedback_mean = state_mean + bias`` and
    ``state_feedback_cov = state_cov + R``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.state_feedback_mean`` ``(P, C, N)``
        and ``D.state_feedback_cov`` ``(P, C, N, N)``.

    Returns
    -------
    None
    """
    d = model.D
    r_cov = observation_noise_cov(model)                 # (N, N)
    # R is (N, N) and state_cov is (P, C, N, N), so R broadcasts across the
    # leading particle/context axes - identical values to a materialised
    # repmat, exactly as MATLAB's implicit expansion does.
    d.state_feedback_mean, d.state_feedback_cov = feedback_transform(
        d.state_mean, d.state_cov, d.bias, r_cov
    )


def resample_particles_md(model, y, q, obs_mask=None):
    """Weight particles by the trial likelihood and systematically resample.

    Multi-dimensional counterpart of
    :func:`realtimecoin.pipeline_scalar.resample_particles`: the per-context
    likelihood is the multivariate Gaussian density of the observed
    sub-vector of ``y``::

        log p(y, q, c) = log p(c | history) + log p(q | c) + log p(y | c)

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Uses ``model.rng``.
    y : numpy.ndarray or None
        ``(N,)`` observed state feedback, or ``None`` when nothing was
        observed.
    q : int or None
        0-based cue label for this trial, or ``None``.
    obs_mask : numpy.ndarray, optional
        ``(N,)`` boolean mask of the entries of ``y`` that are not ``nan``.
        ``None`` derives it from ``y``.

    Returns
    -------
    None
    """
    d = model.D
    p_count = model.num_particles
    c_slots = model.max_contexts + 1
    obs_mask = _resolve_mask(model, y, obs_mask)
    has_obs = _has_observation(y, obs_mask)

    if not has_obs:
        log_py = np.zeros((p_count, c_slots))            # (P, C)
        py = np.ones((p_count, c_slots))                 # (P, C)
    else:
        yv = np.asarray(y, dtype=float).reshape(-1)      # (N,)
        obs_idx = np.flatnonzero(obs_mask)               # (k,)
        # Observed sub-block only: rows/cols obs_idx of the predictive moments.
        mean_obs = d.state_feedback_mean[:, :, obs_idx]  # (P, C, k)
        cov_obs = d.state_feedback_cov[
            :, :, obs_idx[:, None], obs_idx[None, :]
        ]                                                # (P, C, k, k)
        innovation = yv[obs_idx] - mean_obs              # (P, C, k)
        log_py = np.zeros((p_count, c_slots))
        # Kept as a loop (not batched): gaussian_log_lik_chol owns the
        # jitter/diagonal degradation contract and is the shared A1 helper.
        for p in range(p_count):
            for c in range(c_slots):
                log_py[p, c] = gaussian_log_lik_chol(innovation[p, c], cov_obs[p, c])
        with np.errstate(over="ignore", under="ignore"):
            py = np.exp(log_py)
    d.probability_state_feedback = py                    # (P, C)

    log_pc = safe_log(d.prior_probabilities)             # (P, C)
    if q is not None:
        log_pc = log_pc + safe_log(d.probability_cue)
    if has_obs:
        # Add the log-likelihood directly (rather than log(exp(.))) for
        # numerical robustness when components underflow.
        log_pc = log_pc + log_py

    l_w = log_sum_exp(log_pc, axis=-1)                   # (P,) context axis
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        resp = np.exp(log_pc - l_w[:, None])             # (P, C)
    resp[~np.isfinite(resp)] = 0.0

    if not has_obs and q is None:
        # Nothing to weight by: keep the ancestry (and consume no variates).
        idx = np.arange(p_count)
    else:
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            weights = np.exp(l_w - log_sum_exp(l_w, axis=-1))
        idx = systematic_resampling(model.rng, weights)

    d.i_resampled = idx                                  # (P,) 0-based
    resample_state_md(model, idx)
    # Responsibilities are renormalised AFTER the ancestry gather, so each
    # resampled particle carries a proper distribution over its own contexts.
    d.responsibilities = normalize_columns(resp[idx, :])


#: Every ``ParticleState`` field carrying a particle axis in the MD model. The
#: layout puts the particle FIRST on every one of them, so a single ``X[idx]``
#: reorders them all - MATLAB needs three shape-grouped lists because the
#: particle sits on the 2nd, 3rd or 4th trailing axis there.
#:
#: WARNING: this list is EXPLICIT on purpose (the scalar MATLAB routine's
#: ``size(X, 2) == P`` heuristic is a known defect - see ``docs/CODE_REVIEW.md``
#: F-item on ``resampleState``). ADD ANY NEW PARTICLE-INDEXED FIELD HERE, or it
#: will silently keep its pre-resampling ancestry.
#:
#: Deliberately absent: ``previous_context`` (overwritten from ``context`` by
#: :func:`sample_context_md`, exactly as MATLAB leaves it out), ``i_resampled``
#: (just written, and it indexes the OLD ancestry), ``responsibilities``
#: (rebuilt from ``resp[idx]`` after the gather), the ``previous_state_*`` pair
#: (refreshed at the end of this function) and the ``x_*`` sampled trajectory
#: (recomputed by :func:`sample_states_md` later in the same trial).
#:
#: ``bias_ss_1`` / ``bias_ss_2`` are scalar-model fields and are always ``None``
#: in an MD run, so listing them is inert - but ``resampleStateMD.m`` lists them
#: (``isfield``-guarded) and matching it exactly costs nothing.
RESAMPLED_FIELDS_MD = (
    # per-particle vectors and (P, C) matrices
    "n_active",
    "context",
    "prior_probabilities",
    "predicted_probabilities",
    "probability_cue",
    "probability_state_feedback",
    "global_transition_probabilities",
    "global_cue_probabilities",
    "bias_ss_1",
    "bias_ss_2",
    # (P, C, N) means / bias / bias information statistic
    "state_mean",
    "state_filtered_mean",
    "state_feedback_mean",
    "bias",
    "bias_info_ss",
    # (P, C, N, N) covariances and (P, C, ., .) matrix statistics
    "state_cov",
    "state_filtered_cov",
    "state_feedback_cov",
    "Theta",
    "Lambda_xx",
    "Lambda_yx",
    "bias_precision_ss",
    # context-inference tensors (shared shapes with the scalar model)
    "n_context",
    "n_cue",
    "local_transition_matrix",
    "local_cue_matrix",
)


def resample_state_md(model, idx):
    """Reorder every particle-indexed field by a resampling index.

    Multi-dimensional counterpart of
    :func:`realtimecoin.pipeline_scalar.resample_state`. Particle ``k`` becomes
    the old particle ``idx[k]``. As in the scalar routine, the previous filtered
    estimate is refreshed at the end so the next-trial smoother in
    :func:`sample_states_md` has the correct lag state on the resampled
    ancestry.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.
    idx : numpy.ndarray
        ``(P,)`` 0-based resampling index; repeats allowed.

    Returns
    -------
    None
    """
    d = model.D
    idx = np.asarray(idx)
    for name in RESAMPLED_FIELDS_MD:
        value = getattr(d, name)
        if value is None:
            continue        # MATLAB's `if ~isfield(obj.D, f), continue; end`
        # Fancy indexing on axis 0 returns a fresh array, so no field aliases
        # its pre-resampling buffer.
        setattr(d, name, value[idx])

    # Refresh the lag (previous-trial) filtered estimate for the smoother.
    # Copies, not aliases: update_belief_about_states_md rebinds
    # state_filtered_* and sample_context_md writes into it in place.
    d.previous_state_filtered_mean = d.state_filtered_mean.copy()
    d.previous_state_filtered_cov = d.state_filtered_cov.copy()


def sample_context_md(model, q):
    """Sample each particle's context and instantiate new contexts.

    Identical in structure to
    :func:`realtimecoin.pipeline_scalar.sample_context`; the only difference is
    that a newly instantiated context's latent state is seeded to the MD
    stationary distribution (vector mean and ``(N, N)`` covariance).

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Uses ``model.rng``.
    q : int or None
        0-based cue label for this trial, or ``None``.

    Returns
    -------
    None
    """
    n = model.state_dim
    p_count = model.num_particles
    d = model.D
    old_n_active = d.n_active.copy()          # context counts before this draw
    d.previous_context = d.context.copy()     # source context for transition stats

    # Inverse-CDF categorical draw of the context from the responsibilities.
    # MATLAB's `sum(r > cumResp, 1) + 1` is a 1-based label; with 0-based labels
    # the `+ 1` disappears.
    cum_resp = np.cumsum(d.responsibilities, axis=1)      # (P, C)
    r = model.rng.random(p_count)                         # (P,)
    new_context = np.sum(r[:, None] > cum_resp, axis=1)   # (P,) 0-based

    for p in range(p_count):
        # A draw at or beyond the current count means the novel slot was chosen
        # (slot n_active[p] is the novel one).
        if new_context[p] >= d.n_active[p]:
            if d.n_active[p] < model.max_contexts:
                new_context[p] = d.n_active[p]        # instantiate: label == old count
                d.n_active[p] = d.n_active[p] + 1
            else:
                new_context[p] = d.n_active[p] - 1    # at cap: fold onto the last context
    d.context = new_context

    # Particles that gained a context (and are below the cap) split the novel
    # context's stick-breaking mass and seed the new context's MD state. At the
    # cap there is no novel slot left, hence no stick split.
    p_new = np.flatnonzero(
        (d.n_active > old_n_active) & (d.n_active < model.max_contexts)
    )
    if p_new.size:
        q_cov = process_noise_cov(model)
        b = beta_sample(
            model.rng, np.ones(p_new.size), model.gamma_context * np.ones(p_new.size)
        )                                                 # (len(p_new),)
        for k, p in enumerate(p_new):
            c = int(d.n_active[p]) - 1                    # 0-based new context label
            mass = d.global_transition_probabilities[p, c]
            # Stick-breaking: keep proportion b, pass the remainder to the novel
            # slot (c + 1 exists because n_active[p] < max_contexts).
            d.global_transition_probabilities[p, c + 1] = mass * (1.0 - b[k])
            d.global_transition_probabilities[p, c] = mass * b[k]

            # Seed the new context's state at its MD stationary distribution.
            a_matrix = d.Theta[p, c, :, :n]               # (N, N)
            drift = d.Theta[p, c, :, n]                   # (N,)
            # NOTE: these two writes are DEAD - update_belief_about_states_md
            # rebinds state_filtered_* from state_mean/state_cov on the very
            # next step, and previous_state_filtered_* was already snapshotted
            # in resample_state_md. They are kept because sampleContextMD.m has
            # exactly the same dead write, so deleting them here would break
            # line-for-line parity for no behavioural gain. (The new context is
            # still seeded correctly: predict_states_md re-seeds the novel slot
            # from the same stationary moments.)
            d.state_filtered_mean[p, c] = stationary_state_mean_md(a_matrix, drift)
            d.state_filtered_cov[p, c] = stationary_state_cov_md(a_matrix, q_cov)

    instantiate_cue_if_needed(model, q)


def update_belief_about_states_md(model, y, obs_mask=None):
    """Kalman-update every context's latent-state belief with the observation.

    With the identity observation model ``y = s + b + v``, ``v ~ N(0, R)``::

        y_tilde = y - (s_pred + b)          S = P_pred + R
        K       = P_pred S^-1
        s_post  = s_pred + K y_tilde
        P_post  = (I - KH) P_pred (I - KH)' + K R K'     (Joseph form)

    Only the sub-block selected by ``obs_mask`` enters the gain, so a partially
    observed trial corrects only the dimensions it carries - but the UNOBSERVED
    dimensions still move, through the cross-covariance rows of ``K``. Inactive
    contexts inherit the prediction unchanged.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.state_filtered_mean`` ``(P, C, N)``
        and ``D.state_filtered_cov`` ``(P, C, N, N)``.
    y : numpy.ndarray or None
        ``(N,)`` observed state feedback, or ``None``.
    obs_mask : numpy.ndarray, optional
        ``(N,)`` boolean observation mask; ``None`` derives it from ``y``.

    Returns
    -------
    None
    """
    n = model.state_dim
    p_count = model.num_particles
    d = model.D
    obs_mask = _resolve_mask(model, y, obs_mask)

    # Inactive contexts: posterior == prior prediction. Copies, not aliases -
    # MATLAB struct assignment has value semantics and state_mean/state_cov are
    # read again below.
    d.state_filtered_mean = d.state_mean.copy()          # (P, C, N)
    d.state_filtered_cov = d.state_cov.copy()            # (P, C, N, N)
    if not _has_observation(y, obs_mask):
        return

    r_cov = observation_noise_cov(model)                 # (N, N)
    yv = np.asarray(y, dtype=float).reshape(-1)          # (N,)
    obs_idx = np.flatnonzero(obs_mask)                   # (k,)
    r_obs = r_cov[np.ix_(obs_idx, obs_idx)]              # (k, k)

    # Gather the active context of every particle. Vectorises MATLAB's
    # `for p = 1:P` loop: the particle rows are distinct, so no cell is written
    # twice and the batched form is exactly the sequential one.
    rows = np.arange(p_count)
    active = d.context                                   # (P,)
    p_pred = d.state_cov[rows, active]                   # (P, N, N)
    s_pred = d.state_mean[rows, active]                  # (P, N)
    yhat = d.state_feedback_mean[rows, active][:, obs_idx]   # (P, k)  s_pred + b

    s_inn = p_pred[:, obs_idx[:, None], obs_idx[None, :]] + r_obs   # (P, k, k)
    k_gain = _right_solve(p_pred[:, :, obs_idx], s_inn)  # (P, N, k)  K = Pp(:, obs) / S
    innovation = yv[obs_idx] - yhat                      # (P, k)
    s_filt = s_pred + (k_gain @ innovation[..., None])[..., 0]      # (P, N)

    kh = np.zeros((p_count, n, n))                       # (P, N, N)
    kh[:, :, obs_idx] = k_gain                           # K scattered into H's columns
    i_kh = np.eye(n) - kh
    p_filt = (
        i_kh @ p_pred @ np.swapaxes(i_kh, -1, -2)
        + k_gain @ r_obs @ np.swapaxes(k_gain, -1, -2)
    )                                                    # (P, N, N) Joseph form
    p_filt = _symmetrise(p_filt)                         # enforce symmetry vs round-off

    d.state_filtered_mean[rows, active] = s_filt
    d.state_filtered_cov[rows, active] = p_filt


def sample_states_md(model, y, obs_mask=None):
    """Draw the latent-state trajectory for the dynamics and bias regressions.

    Two latent quantities are sampled per context per particle:

    1. the lag state ``s_{i-1}`` through the Rauch-Tung-Striebel smoother gain
       ``J = P_{i-1|i-1} A' P_{i|i-1}^-1``, giving
       ``mean = s_{i-1|i-1} + J (s_{i|i} - s_{i|i-1})`` and
       ``cov = P_{i-1|i-1} + J (P_{i|i} - P_{i|i-1}) J'``;
    2. the current state ``s_i``, drawn from the one-step dynamics prior
       ``N(A s_{i-1} + d, Q)`` for inactive contexts, and for the ACTIVE context
       from the information-form product with the observation::

           postPrec = Q^-1 + R_obs^-1 (scattered into the observed block)
           postMean = postCov (Q^-1 m_dyn + obsInfo)

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.previous_x_dynamics`` and
        ``D.x_dynamics`` ``(P, C, N)`` and ``D.x_bias`` ``(P, N)``. Uses
        ``model.rng``.
    y : numpy.ndarray or None
        ``(N,)`` observed state feedback, or ``None``.
    obs_mask : numpy.ndarray, optional
        ``(N,)`` boolean observation mask; ``None`` derives it from ``y``.

    Returns
    -------
    None

    Notes
    -----
    ``D.i_observed`` is deliberately NOT written: the MD path reads
    ``D.context`` directly (as ``sampleStatesMD.m`` does), and only the scalar
    ``sampleStates.m`` caches the linear index.
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    d = model.D
    q_cov = process_noise_cov(model)                     # (N, N)
    r_cov = observation_noise_cov(model)                 # (N, N)
    q_inv = safe_inverse(q_cov)                          # (N, N)
    obs_mask = _resolve_mask(model, y, obs_mask)
    has_obs = _has_observation(y, obs_mask)

    if has_obs:
        yv = np.asarray(y, dtype=float).reshape(-1)      # (N,)
        obs_idx = np.flatnonzero(obs_mask)               # (k,)
        r_inv_obs = safe_inverse(r_cov[np.ix_(obs_idx, obs_idx)])       # (k, k)
        obs_precision = np.zeros((n, n))
        obs_precision[np.ix_(obs_idx, obs_idx)] = r_inv_obs
        # The active-context posterior covariance is invariant across the
        # (context, particle) loop, so factor its inverse once (bit-identical).
        post_cov_active = safe_inverse(q_inv + obs_precision)           # (N, N)
        post_cov_active = (post_cov_active + post_cov_active.T) / 2.0

    a_mat = d.Theta[:, :, :, :n]                         # (P, C, N, N)
    drift = d.Theta[:, :, :, n]                          # (P, C, N)

    # --- 1. Smoother sample of the lag state s_{i-1} ---
    # Batched over MATLAB's `for p ... for c` loop:
    #   J = P_{i-1|i-1} A' P_{i|i-1}^-1
    p_pred_inv = _safe_inverse_batch(d.state_cov)                       # (P, C, N, N)
    j_gain = (
        d.previous_state_filtered_cov @ np.swapaxes(a_mat, -1, -2) @ p_pred_inv
    )                                                                   # (P, C, N, N)
    mean_s = d.previous_state_filtered_mean + (
        j_gain @ (d.state_filtered_mean - d.state_mean)[..., None]
    )[..., 0]                                                           # (P, C, N)
    cov_s = d.previous_state_filtered_cov + j_gain @ (
        d.state_filtered_cov - d.state_cov
    ) @ np.swapaxes(j_gain, -1, -2)                                     # (P, C, N, N)
    cov_s = _symmetrise(cov_s)
    s_prev = _draw_gaussian_batch(model.rng, mean_s, cov_s, exact_zero=True)
    d.previous_x_dynamics = s_prev                                      # (P, C, N)

    # --- 2. Forward sample of the current state s_i ---
    post_mean = (a_mat @ s_prev[..., None])[..., 0] + drift             # (P, C, N)
    post_cov = np.broadcast_to(q_cov, (p_count, c_slots, n, n)).copy()  # (P, C, N, N)
    if has_obs:
        rows = np.arange(p_count)
        active = d.context                                              # (P,)
        bias_active = d.bias[rows, active]                              # (P, N)
        obs_info = np.zeros((p_count, n))                               # (P, N)
        # obsInfo(obsIdx) = R_obs^-1 (y_obs - b_obs), per particle.
        obs_info[:, obs_idx] = (yv[obs_idx] - bias_active[:, obs_idx]) @ r_inv_obs.T
        dyn_mean_active = post_mean[rows, active]                       # (P, N)
        post_mean[rows, active] = (
            post_cov_active
            @ ((q_inv @ dyn_mean_active[..., None])[..., 0] + obs_info)[..., None]
        )[..., 0]
        post_cov[rows, active] = post_cov_active
    d.x_dynamics = _draw_gaussian_batch(
        model.rng, post_mean, post_cov, exact_zero=True
    )                                                                   # (P, C, N)

    # Active-context sampled state, used for the bias residual.
    d.x_bias = d.x_dynamics[np.arange(p_count), d.context]              # (P, N)


def update_sufficient_statistics_md(model, y, q, obs_mask=None):
    """Accumulate the conjugate sufficient statistics for this trial.

    The context-transition and cue counts are accumulated exactly as in the
    scalar model. The dynamics statistics are the matrix accumulators of the
    matrix-normal regression ``s_i = Theta x_{i-1} + w`` with augmented
    regressor ``x_a = [s_{i-1}; 1]``::

        Lambda_xx <- Lambda_xx + x_a x_a'       ((N+1) x (N+1))
        Lambda_yx <- Lambda_yx + s_i x_a'       (N x (N+1))

    Only contexts that have been visited accumulate, and accumulation starts
    after the first trial. The optional bias statistics collect the observation
    residual ``y - s_i`` weighted by the observed-block precision.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.
    y : numpy.ndarray or None
        ``(N,)`` observed state feedback, or ``None``.
    q : int or None
        0-based cue label for this trial, or ``None``.
    obs_mask : numpy.ndarray, optional
        ``(N,)`` boolean observation mask; ``None`` derives it from ``y``.

    Returns
    -------
    None
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    d = model.D
    obs_mask = _resolve_mask(model, y, obs_mask)
    has_obs = _has_observation(y, obs_mask)
    rows = np.arange(p_count)

    # --- Context-transition and cue counts (identical to the scalar model) ---
    # n_context[p, from, to]; MATLAB's sub2ind gather over 1:P becomes the
    # per-particle row index here. The rows are distinct, so `+= 1` cannot
    # collide the way np.add.at guards against.
    d.n_context[rows, d.previous_context, d.context] += 1.0

    if q is not None:
        ensure_cue_column(model, q + 1)      # COUNT, not a label (see state.py)
        d.n_cue[rows, d.context, q] += 1.0

    # --- Matrix dynamics sufficient statistics ---
    if model.trial > 0:
        # A context accumulates once it has been departed from at least once;
        # MATLAB sums n_context over its 2nd (destination) dimension.
        observed = d.n_context.sum(axis=2) > 0                     # (P, C)
        x_aug = np.concatenate(
            [d.previous_x_dynamics, np.ones((p_count, c_slots, 1))], axis=2
        )                                                          # (P, C, N+1)
        s_cur = d.x_dynamics                                       # (P, C, N)
        # Vectorises MATLAB's `for p ... for c` accumulation:
        #   Lambda_xx += x_a x_a' ,  Lambda_yx += s_i x_a'
        outer_xx = x_aug[..., :, None] * x_aug[..., None, :]       # (P, C, N+1, N+1)
        outer_yx = s_cur[..., :, None] * x_aug[..., None, :]       # (P, C, N, N+1)
        keep = observed[..., None, None]
        d.Lambda_xx = d.Lambda_xx + np.where(keep, outer_xx, 0.0)
        d.Lambda_yx = d.Lambda_yx + np.where(keep, outer_yx, 0.0)

    # --- Bias sufficient statistics ---
    if model.infer_bias and has_obs:
        yv = np.asarray(y, dtype=float).reshape(-1)                # (N,)
        obs_idx = np.flatnonzero(obs_mask)                         # (k,)
        r_cov = observation_noise_cov(model)
        r_inv_obs = safe_inverse(r_cov[np.ix_(obs_idx, obs_idx)])  # (k, k)
        precision_update = np.zeros((n, n))
        precision_update[np.ix_(obs_idx, obs_idx)] = r_inv_obs
        info_update = np.zeros((p_count, n))                       # (P, N)
        info_update[:, obs_idx] = (
            yv[obs_idx] - d.x_bias[:, obs_idx]
        ) @ r_inv_obs.T
        # Distinct particle rows again, so the indexed += is collision-free.
        d.bias_info_ss[rows, d.context] += info_update
        d.bias_precision_ss[rows, d.context] += precision_update


def sample_parameters_md(model):
    """Resample every model parameter and rebuild the derived local matrices.

    Same order as :func:`realtimecoin.pipeline_scalar.sample_parameters`, with
    :func:`sample_dynamics_md` and :func:`sample_bias_md` replacing their scalar
    counterparts. The two global HDP samplers and the two local-matrix rebuilds
    are shared, from :mod:`realtimecoin.context`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Uses ``model.rng``.

    Returns
    -------
    None
    """
    sample_global_transition_probabilities(model)   # sticky HDP-HMM betas (shared)
    sample_global_cue_probabilities(model)          # HDP cue-context betas (shared)
    sample_dynamics_md(model)                       # matrix-normal Theta = [A | d]
    sample_bias_md(model)                           # multivariate observation bias
    update_local_transition_matrix(model)           # rebuild local rows (shared)
    update_local_cue_matrix(model)                  # rebuild cue likelihoods (shared)


def sample_dynamics_md(model):
    """Sample the per-context augmented dynamics ``Theta = [A | d]``.

    Draws from the matrix-normal conjugate posterior of the regression
    ``s_i = Theta x_{i-1} + w``, ``w ~ N(0, Q)``, ``x_{i-1} = [s_{i-1}; 1]``,
    with the matrix-normal prior ``MN(M0, U = Q, V0)``::

        V_post = (V0inv + Lambda_xx)^-1
        M_post = (M0 * V0inv + Lambda_yx) * V_post

    The draw is constrained to a spectral radius below one (bounded stability),
    the multi-dimensional analogue of the scalar ``a in [0, 1)`` truncation.

    At ``N == 1`` this reduces ALGEBRAICALLY to the scalar posterior, because
    :func:`realtimecoin.state.dynamics_prior_md` builds
    ``V0inv = sigma^2 diag([prec_ret, prec_drift])`` against ``U = Q = sigma^2``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.Theta`` ``(P, C, N, N + 1)``. Uses
        ``model.rng``.

    Returns
    -------
    None
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    d = model.D

    m0, v0inv, _ = dynamics_prior_md(model)      # (N, N+1) and (N+1, N+1)
    q_cov = process_noise_cov(model)             # process-noise (row) covariance U

    # Batched over MATLAB's `for p ... for c` posterior formation; the
    # rejection sampler below stays a loop (its iteration count is data
    # dependent per cell).
    v_post = _safe_inverse_batch(v0inv + d.Lambda_xx)     # (P, C, N+1, N+1)
    v_post = _symmetrise(v_post)                          # re-symmetrise vs round-off
    m_post = (m0 @ v0inv + d.Lambda_yx) @ v_post          # (P, C, N, N+1)

    theta = np.empty_like(d.Theta)                        # (P, C, N, N+1)
    for p in range(p_count):
        for c in range(c_slots):
            # Draw from MN(Mpost, Q, Vpost), rejecting until rho(A) < 1.
            theta[p, c] = sample_stable_theta(
                model.rng, m_post[p, c], q_cov, v_post[p, c], n
            )
    d.Theta = theta


def sample_bias_md(model):
    """Sample the per-context observation bias vector.

    With ``y = s + b + v``, ``v ~ N(0, R)`` and the isotropic Gaussian prior
    ``b ~ N(prior_mean_bias * 1, prior_precision_bias^-1 I)``, the conjugate
    posterior per context per particle is::

        postPrec = prior_precision_bias * I + bias_precision_ss
        postMean = postCov (prior_precision_bias * prior_mean_bias * 1
                            + bias_info_ss)

    The sufficient statistics were accumulated over whichever coordinates were
    observed on each trial, so a fully observed run reduces to the usual
    ``n R^-1`` / ``R^-1 sum(y - s)`` formulae.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.bias`` ``(P, C, N)``. Uses
        ``model.rng``.

    Returns
    -------
    None
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    d = model.D

    if not model.infer_bias:
        d.bias = np.zeros((p_count, c_slots, n))          # (P, C, N)
        return

    prior_prec = model.prior_precision_bias * np.eye(n)               # (N, N)
    prior_term = model.prior_precision_bias * model.prior_mean_bias * np.ones(n)

    # Batched over MATLAB's `for p ... for c` conjugate draw. No zero-covariance
    # short-circuit here: sampleBiasMD calls choljitter unconditionally, so a
    # degenerate posterior keeps its jittered factor.
    post_prec = prior_prec + d.bias_precision_ss          # (P, C, N, N)
    post_cov = _symmetrise(_safe_inverse_batch(post_prec))
    post_mean = (post_cov @ (prior_term + d.bias_info_ss)[..., None])[..., 0]
    d.bias = _draw_gaussian_batch(model.rng, post_mean, post_cov)     # (P, C, N)

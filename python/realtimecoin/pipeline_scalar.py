"""Scalar (``state_dim == 1``) per-trial particle-filter pipeline.

Translated from ``@RealTimeCOIN/private/``: ``predictStates``,
``predictStateFeedback``, ``resampleParticles``, ``resampleState``,
``sampleContext``, ``updateBeliefAboutStates``, ``sampleStates``,
``updateSufficientStatistics``, ``sampleParameters``, ``sampleDynamics`` and
``sampleBias``.

Every function takes the model as its first argument and mutates ``model.D``
in place, exactly as the MATLAB private methods mutate ``obj.D``. They are
called, in this order, by :meth:`realtimecoin.model.RealTimeCOIN.observe_y`::

    predict_context            (shared, realtimecoin.context)
    predict_states
    predict_state_feedback
    resample_particles
    sample_context
    update_belief_about_states
    sample_states
    update_sufficient_statistics
    sample_parameters

The scalar path is the regression baseline: it must stay behaviourally
equivalent to ``COIN.m``. New behaviour belongs in
:mod:`realtimecoin.pipeline_md`.

Array layout is particles-leading throughout; see :mod:`realtimecoin.state` for
the field-by-field shape table. Under that layout MATLAB's ``sub2ind`` gathers
into a ``(Cmax, P)`` plane become the pair ``(numpy.arange(P), labels)``: the
linear index itself never survives the transposition and is always re-derived.
"""

from __future__ import annotations

import numpy as np

from .numerics import (
    normalize_columns,
    safe_divide,
    safe_inverse,
    safe_log,
    stationary_state_mean,
    stationary_state_var,
)
from .samplers import (
    beta_sample,
    sample_bivariate_truncated,
    sample_scalar_normal,
)
from .statics import log_sum_exp, normal_pdf, systematic_resampling
from .state import (
    EPS,
    ensure_cue_column,
    feedback_transform,
    instantiate_cue_if_needed,
    observation_variance,
)

__all__ = [
    "predict_states",
    "predict_state_feedback",
    "resample_particles",
    "resample_state",
    "sample_context",
    "update_belief_about_states",
    "sample_states",
    "update_sufficient_statistics",
    "sample_parameters",
    "sample_dynamics",
    "sample_bias",
]


# --------------------------------------------------------------------------
# Shared index helpers
# --------------------------------------------------------------------------


def _active_cells(model):
    """Row/column index pair addressing each particle's sampled context.

    The particles-leading replacement for MATLAB's
    ``sub2ind([Cmax, P], D.context, 1:P)``: ``X[rows, cols]`` gathers one cell
    per particle from a ``(P, C)`` array.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``D.context`` is read.

    Returns
    -------
    rows : numpy.ndarray
        ``(P,)`` particle indices ``0 .. P-1``.
    cols : numpy.ndarray
        ``(P,)`` 0-based sampled context labels.
    """
    return np.arange(model.num_particles), model.D.context


def _novel_cells(model):
    """Row/column index pair addressing each particle's novel-context slot.

    Only particles strictly below the context cap are returned: MATLAB guards
    every novel-slot re-seed with ``if D.C(p) < max_contexts``, so a particle at
    the cap has no novel slot to re-seed. For the remaining particles the novel
    slot is ``n_active[p]`` (0-based), which is MATLAB's ``C(p) + 1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``D.n_active`` is read.

    Returns
    -------
    rows : numpy.ndarray
        ``(k,)`` particle indices below the cap.
    cols : numpy.ndarray
        ``(k,)`` novel-slot column for each of those particles.
    """
    d = model.D
    rows = np.nonzero(d.n_active < model.max_contexts)[0]
    return rows, d.n_active[rows]


# --------------------------------------------------------------------------
# Per-trial pipeline
# --------------------------------------------------------------------------


def predict_states(model):
    """Propagate each context's latent state one step through its dynamics.

    Applies the AR(1) prediction ``s <- a s + d`` with variance
    ``v <- a^2 v + sigma_process_noise^2`` to every context of every particle,
    and re-seeds each particle's novel slot at the stationary distribution of
    its freshly sampled dynamics.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.state_mean`` and ``D.state_var``,
        each ``(P, C)``.

    Returns
    -------
    None
    """
    d = model.D
    qv = model.sigma_process_noise ** 2                 # process-noise variance
    # Vectorised Kalman predict across all contexts and particles at once.
    d.state_mean = d.retention * d.state_filtered_mean + d.drift        # (P, C)
    d.state_var = d.retention ** 2 * d.state_filtered_var + qv          # (P, C)

    rows, cols = _novel_cells(model)
    if rows.size:
        # Re-seed the novel context at its stationary (prior) distribution.
        d.state_mean[rows, cols] = stationary_state_mean(
            d.retention[rows, cols], d.drift[rows, cols]
        )                                                               # (k,)
        d.state_var[rows, cols] = stationary_state_var(
            d.retention[rows, cols], model.sigma_process_noise
        )                                                               # (k,)
    # fmax, not maximum: MATLAB's max(x, 0) drops a NaN in favour of 0.
    d.state_var = np.fmax(d.state_var, 0.0)             # guard round-off


def predict_state_feedback(model):
    """Map the predicted state moments into observation space.

    ``state_feedback_mean = state_mean + bias`` and
    ``state_feedback_var = state_var + observation_variance(model)``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.state_feedback_mean`` and
        ``D.state_feedback_var``, each ``(P, C)``.

    Returns
    -------
    None
    """
    d = model.D
    d.state_feedback_mean, d.state_feedback_var = feedback_transform(
        d.state_mean, d.state_var, d.bias, observation_variance(model)
    )                                                                   # (P, C)


def resample_particles(model, y, q):
    """Weight particles by the trial likelihood and systematically resample.

    Computes the per-particle marginal likelihood of the observation (and cue),
    stores it in ``D.probability_state_feedback``, draws the resampling index
    with :func:`realtimecoin.statics.systematic_resampling`, records it in
    ``D.i_resampled`` and applies it through :func:`resample_state`.

    The per-context log-joint combines the available factors::

        log p(y, q, c) = log p(c | history)     (prior_probabilities)
                       + log p(q | c)           (probability_cue, if q given)
                       + log p(y | c)           (Gaussian pdf,    if y given)

    Marginalising over contexts gives each particle's log weight. When BOTH
    ``y`` and ``q`` are missing the weights are uniform, so resampling is
    skipped entirely (identity ancestry) to avoid needless particle depletion -
    ``resample_state`` is still called, because it also refreshes the
    previous-trial filtered estimate the smoother needs.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Uses ``model.rng``.
    y : float or None
        Observed state feedback for this trial; ``None`` for a missing
        observation (the trial still runs, states are predicted not corrected).
    q : int or None
        0-based cue label for this trial, or ``None``.

    Returns
    -------
    None
    """
    d = model.D
    p_count = model.num_particles
    c_slots = model.max_contexts + 1

    if y is None:
        py = np.ones((p_count, c_slots))                                # (P, C)
    else:
        py = normal_pdf(y, d.state_feedback_mean, d.state_feedback_var)  # (P, C)
    d.probability_state_feedback = py

    log_pc = safe_log(d.prior_probabilities)                            # (P, C)
    if q is not None:
        log_pc = log_pc + safe_log(d.probability_cue)
    if y is not None:
        log_pc = log_pc + safe_log(py)

    # Marginalise over the context axis (MATLAB's dim 1 is our trailing axis).
    l_w = log_sum_exp(log_pc, axis=-1)                                  # (P,)
    with np.errstate(invalid="ignore"):
        resp = np.exp(log_pc - l_w[:, None])                            # (P, C)
    resp = np.where(np.isfinite(resp), resp, 0.0)

    if y is None and q is None:
        # Uniform weights: identity ancestry, no resampling noise injected.
        idx = np.arange(p_count)                                        # (P,)
    else:
        weights = np.exp(l_w - log_sum_exp(l_w, axis=-1))               # (P,)
        idx = systematic_resampling(model.rng, weights)                 # (P,)

    d.i_resampled = idx
    resample_state(model, idx)
    # Responsibilities are re-normalised AFTER the ancestry gather, so each
    # resampled column is a proper pmf over contexts.
    d.responsibilities = normalize_columns(resp[idx])                   # (P, C)


#: Particle-indexed fields of the scalar state, grouped by tensor rank exactly
#: as ``resampleState.m`` groups them. Under the particles-leading layout the
#: particle axis is axis 0 for every one of them, but the explicit lists are
#: kept (rather than a shape heuristic) so a future field cannot be silently
#: mis-resampled - the fix ``CODE_REVIEW.md`` F6 asks for.
_RESAMPLE_VECTOR_FIELDS = ("n_active", "context")

_RESAMPLE_MATRIX_FIELDS = (
    "prior_probabilities",
    "predicted_probabilities",
    "probability_cue",
    "probability_state_feedback",
    "global_transition_probabilities",
    "retention",
    "drift",
    "bias",
    "state_mean",
    "state_var",
    "state_feedback_mean",
    "state_feedback_var",
    "state_filtered_mean",
    "state_filtered_var",
    "bias_ss_1",
    "bias_ss_2",
)

_RESAMPLE_TENSOR_FIELDS = (
    "n_context",             # (P, C, C)
    "n_cue",                 # (P, C, Q)
    "dynamics_ss_1",         # (P, C, 2)
    "dynamics_ss_2",         # (P, C, 2, 2)
    "global_cue_probabilities",   # (P, Q)
    "local_transition_matrix",    # (P, C, C)
    "local_cue_matrix",           # (P, C, Q)
)


def resample_state(model, idx):
    """Reorder every particle-indexed field by a resampling index.

    Particle ``k`` becomes the old particle ``idx[k]``. Fields are reordered by
    explicit name lists grouped by tensor rank (never by guessing which axis has
    length ``P``). Afterwards the previous-trial filtered estimate is refreshed
    so the next-trial smoother in :func:`sample_states` operates on the
    resampled ancestry.

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
    idx = np.asarray(idx).reshape(-1)

    for group in (
        _RESAMPLE_VECTOR_FIELDS,
        _RESAMPLE_MATRIX_FIELDS,
        _RESAMPLE_TENSOR_FIELDS,
    ):
        for name in group:
            # Fancy indexing on axis 0 already returns a fresh array, so the
            # gather can never alias the source (MATLAB gets that from value
            # semantics).
            setattr(d, name, getattr(d, name)[idx])

    # Refresh the lag (previous-trial) filtered estimate for the smoother.
    d.previous_state_filtered_mean = d.state_filtered_mean.copy()        # (P, C)
    d.previous_state_filtered_var = d.state_filtered_var.copy()          # (P, C)


def sample_context(model, q):
    """Sample each particle's context and instantiate new contexts.

    Draws ``D.context`` from ``D.responsibilities``, promotes a particle's novel
    slot to a real context when it is selected - incrementing ``D.n_active`` -
    and stick-breaks a new global cue category if the cue label is new.

    A particle already at ``max_contexts`` that draws the novel slot folds back
    onto its LAST context and takes no ``Beta(1, gamma_context)`` stick split.
    So does a particle whose increment lands exactly ON the cap: MATLAB selects
    the splitting particles with ``C > oldC & C < max_contexts``, so the last
    context a particle can create never splits the stick and never has its
    latent state re-seeded.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.responsibilities`` ``(P, C)``,
        ``D.previous_context`` and ``D.context`` ``(P,)`` and ``D.n_active``.
        Uses ``model.rng``.
    q : int or None
        0-based cue label for this trial, or ``None``.

    Returns
    -------
    None
    """
    d = model.D
    p_count = model.num_particles
    old_n_active = d.n_active.copy()             # counts before this draw
    # Copy, not alias: MATLAB struct assignment has value semantics.
    d.previous_context = d.context.copy()        # source context for the stats

    # Inverse-CDF categorical draw. MATLAB's `sum(r > cumResp, 1) + 1` yields a
    # 1-based label; dropping the `+ 1` yields the 0-based label directly. The
    # comparison stays STRICTLY `>` against the cumulative.
    cum_resp = np.cumsum(d.responsibilities, axis=-1)                   # (P, C)
    r = model.rng.random(p_count)                                       # (P,)
    new_context = np.sum(r[:, None] > cum_resp, axis=-1)                # (P,)

    # A draw at or beyond the current count selected the novel slot.
    novel_drawn = new_context >= d.n_active
    below_cap = d.n_active < model.max_contexts
    # Below the cap the novel slot is instantiated; at the cap the draw folds
    # back onto the last context. Both cases end at label n_active - 1, so the
    # increment is applied first and the label read off afterwards.
    d.n_active = np.where(novel_drawn & below_cap, d.n_active + 1, d.n_active)
    new_context = np.where(novel_drawn, d.n_active - 1, new_context)    # (P,)
    d.context = new_context

    # Particles that gained a context AND are still below the cap split the
    # novel context's stick-breaking mass.
    p_new = np.nonzero((d.n_active > old_n_active) & (d.n_active < model.max_contexts))[0]
    if p_new.size:
        b = beta_sample(
            model.rng,
            np.ones(p_new.size),
            model.gamma_context * np.ones(p_new.size),
        )                                                               # (k,)
        c_new = d.n_active[p_new] - 1        # 0-based label of the new context
        mass = d.global_transition_probabilities[p_new, c_new].copy()   # (k,)
        # Stick-breaking: keep proportion b for the new context, pass the
        # remainder (1 - b) to the next (still novel) slot.
        d.global_transition_probabilities[p_new, c_new + 1] = mass * (1.0 - b)
        d.global_transition_probabilities[p_new, c_new] = mass * b
        # Seed the new context's state at its stationary distribution.
        d.state_filtered_mean[p_new, c_new] = stationary_state_mean(
            d.retention[p_new, c_new], d.drift[p_new, c_new]
        )
        d.state_filtered_var[p_new, c_new] = stationary_state_var(
            d.retention[p_new, c_new], model.sigma_process_noise
        )

    instantiate_cue_if_needed(model, q)


def update_belief_about_states(model, y):
    """Kalman-update every context's latent-state belief with the observation.

    Fuses the predicted moments with the observation in information form,
    producing the filtered (posterior) moments for each particle's ACTIVE
    context only; every inactive context keeps its prediction. A missing
    observation leaves all predicted moments in place.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.state_filtered_mean`` and
        ``D.state_filtered_var``, each ``(P, C)``.
    y : float or None
        Observed state feedback, or ``None`` when missing.

    Returns
    -------
    None
    """
    d = model.D
    # Copies, not aliases: MATLAB struct assignment is a value copy, and the
    # active cells are overwritten below.
    d.state_filtered_mean = d.state_mean.copy()                         # (P, C)
    d.state_filtered_var = d.state_var.copy()                           # (P, C)
    if y is None:
        return                                   # missing observation: no update

    obs_var = observation_variance(model)
    rows, cols = _active_cells(model)
    pred_var = d.state_var[rows, cols]                                  # (P,)
    total_var = pred_var + obs_var                                      # (P,)
    # A degenerate total variance means "trust the prior": gain 0. The test is
    # `<= 0`, not `> 0`, so a NaN variance propagates as NaN exactly as MATLAB's
    # `if totalVar <= 0 ... else K = predVar ./ totalVar` does.
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(total_var <= 0, 0.0, pred_var / total_var)      # (P,)
    innovation = y - d.state_feedback_mean[rows, cols]                   # (P,)
    d.state_filtered_mean[rows, cols] = d.state_mean[rows, cols] + gain * innovation
    d.state_filtered_var[rows, cols] = np.fmax((1.0 - gain) * pred_var, 0.0)


def sample_states(model, y):
    """Draw the latent-state trajectory for the dynamics and bias regressions.

    Backward-samples the previous state from the RTS smoothing distribution,
    then forward-samples the current state conditioned on it and (for the active
    context only) on the observation. Also caches the active-context state
    ``D.x_bias`` and its column ``D.i_observed`` for the bias update.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.previous_x_dynamics`` and
        ``D.x_dynamics`` ``(P, C)``, ``D.x_bias`` ``(P,)`` and ``D.i_observed``
        ``(P,)``. Uses ``model.rng``.
    y : float or None
        Observed state feedback, or ``None`` when missing.

    Returns
    -------
    None
    """
    d = model.D
    q_var = model.sigma_process_noise ** 2       # process (state) noise variance
    obs_var = observation_variance(model)        # observation noise variance
    p_count = model.num_particles
    c_slots = model.max_contexts + 1
    shape = (p_count, c_slots)
    rng = model.rng

    # --- Stage 1: RTS smoother draw of the previous state s_{i-1} -----------
    # Smoother gain J = a * P_{i-1|i-1} / P_{i|i-1}; safe_divide guards a zero
    # predicted variance.
    g = d.retention * safe_divide(
        d.previous_state_filtered_var, d.state_var
    )                                                                   # (P, C)
    smooth_mean = d.previous_state_filtered_mean + g * (
        d.state_filtered_mean - d.state_mean
    )                                                                   # (P, C)
    smooth_var = d.previous_state_filtered_var + g ** 2 * (
        d.state_filtered_var - d.state_var
    )                                                                   # (P, C)
    # fmax, not maximum: clamp tiny negative round-off (and drop NaN) to zero.
    smooth_var = np.fmax(smooth_var, 0.0)
    d.previous_x_dynamics = sample_scalar_normal(
        rng, smooth_mean, smooth_var, shape, -np.inf, np.inf
    )                                                                   # (P, C)

    # --- Stage 2: forward draw of the current state s_i ---------------------
    dyn_mean = d.retention * d.previous_x_dynamics + d.drift            # (P, C)
    if y is None:
        # Missing observation: sample straight from the dynamics prior.
        d.x_dynamics = sample_scalar_normal(
            rng, dyn_mean, q_var, shape, -np.inf, np.inf
        )                                                               # (P, C)
    else:
        rows, cols = _active_cells(model)
        active = np.zeros(shape)                                        # (P, C)
        active[rows, cols] = 1.0
        if q_var == 0:
            # Deterministic dynamics: the prior is a point mass at dyn_mean.
            post_mean = dyn_mean.copy()                                 # (P, C)
            post_var = np.zeros(shape)                                  # (P, C)
            if obs_var == 0:
                # Noiseless observation too: pin the active context exactly.
                post_mean[rows, cols] = y - d.bias[rows, cols]
        elif obs_var == 0:
            # Noiseless observation: the active context collapses onto y - bias.
            post_mean = dyn_mean.copy()                                 # (P, C)
            post_var = q_var * np.ones(shape)                           # (P, C)
            post_mean[rows, cols] = y - d.bias[rows, cols]
            post_var[rows, cols] = 0.0
        else:
            # General Gaussian fusion in precision (information) form.
            post_var = 1.0 / (1.0 / q_var + active / obs_var)           # (P, C)
            post_mean = post_var * (
                dyn_mean / q_var + active * (y - d.bias) / obs_var
            )                                                           # (P, C)
        d.x_dynamics = sample_scalar_normal(
            rng, post_mean, post_var, shape, -np.inf, np.inf
        )                                                               # (P, C)

    # Cache the active-context sample and its column for the bias update. The
    # column is stored separately (rather than re-read from D.context) exactly
    # as MATLAB caches the linear index, so a later context resample cannot
    # silently redirect the bias statistics.
    rows, cols = _active_cells(model)
    d.x_bias = d.x_dynamics[rows, cols]                                 # (P,)
    d.i_observed = cols.copy()                                          # (P,)


def update_sufficient_statistics(model, y, q):
    """Accumulate the conjugate sufficient statistics for this trial.

    Increments the context-transition counts ``D.n_context``, the cue counts
    ``D.n_cue``, the dynamics regression accumulators ``D.dynamics_ss_1`` /
    ``D.dynamics_ss_2`` and, when ``infer_bias`` is set, the bias accumulators
    ``D.bias_ss_1`` / ``D.bias_ss_2`` at the cells cached in ``D.i_observed``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.
    y : float or None
        Observed state feedback, or ``None`` when missing (the bias statistics
        are then left untouched).
    q : int or None
        0-based cue label for this trial, or ``None``.

    Returns
    -------
    None
    """
    d = model.D
    p_count = model.num_particles
    particles = np.arange(p_count)

    # --- Context-transition counts ------------------------------------------
    # One count per particle at (previous_context -> context); the (p, i, j)
    # triples are distinct across p, so a plain scatter-add is exact.
    d.n_context[particles, d.previous_context, d.context] += 1.0

    # --- Cue-emission counts -------------------------------------------------
    if q is not None:
        ensure_cue_column(model, q + 1)     # grow n_cue if this cue is new
        d.n_cue[particles, d.context, q] += 1.0

    # --- Dynamics regression sufficient statistics --------------------------
    # Skipped on trial 0 (no previous state yet). model.trial is still the
    # PRE-increment value here: observe_y advances it only after the pipeline.
    if model.trial > 0:
        c_slots = model.max_contexts + 1
        # Augmented regressor [s_{i-1}; 1] on the TRAILING axis (MATLAB stacks
        # it on the leading one; the layout flip moves it to the end).
        x_aug = np.ones((p_count, c_slots, 2))                       # (P, C, 2)
        x_aug[:, :, 0] = d.previous_x_dynamics
        # Only contexts with any transition count contribute.
        obs_mask = (d.n_context.sum(axis=2) > 0).astype(float)       # (P, C)
        # dynamics_ss_1 accumulates s_i * [s_{i-1}; 1].
        ss1 = d.x_dynamics[:, :, None] * x_aug                       # (P, C, 2)
        d.dynamics_ss_1 = d.dynamics_ss_1 + ss1 * obs_mask[:, :, None]
        # dynamics_ss_2 accumulates the 2x2 Gram matrix [s_{i-1};1][s_{i-1};1]'.
        gram = x_aug[:, :, :, None] * x_aug[:, :, None, :]        # (P, C, 2, 2)
        d.dynamics_ss_2 = d.dynamics_ss_2 + gram * obs_mask[:, :, None, None]

    # --- Bias sufficient statistics -----------------------------------------
    if model.infer_bias and y is not None:
        d.bias_ss_1[particles, d.i_observed] += y - d.x_bias
        d.bias_ss_2[particles, d.i_observed] += 1.0


def sample_parameters(model):
    """Resample every model parameter and rebuild the derived local matrices.

    Sampling order (each step conditions on the current sufficient
    statistics)::

        1. sample_global_transition_probabilities   (sticky HDP-HMM betas)
        2. sample_global_cue_probabilities          (HDP cue betas)
        3. sample_dynamics                          (per-context [a; d])
        4. sample_bias                              (per-context bias)
        5. update_local_transition_matrix
        6. update_local_cue_matrix

    Steps 1, 2, 5 and 6 live in :mod:`realtimecoin.context` and are shared with
    the multi-dimensional pipeline.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Uses ``model.rng``.

    Returns
    -------
    None
    """
    from . import context as _context   # local import: breaks the import cycle

    _context.sample_global_transition_probabilities(model)
    _context.sample_global_cue_probabilities(model)
    sample_dynamics(model)
    sample_bias(model)
    _context.update_local_transition_matrix(model)
    _context.update_local_cue_matrix(model)


def sample_dynamics(model):
    """Sample the per-context linear dynamics ``[a; d]``.

    Draws from the conjugate bivariate-normal posterior of the regression
    ``s_i = a s_{i-1} + d + w_i``::

        covar = (prior_precision + ss2 / qVar)^-1
        mu    = covar * (prior_precision * prior_mean + ss1 / qVar)

    truncated to the stable, causal region ``a in [0, 1)``. A zero process
    variance divides by ``eps`` instead of zero so the posterior collapses onto
    the least-squares fit without producing ``inf`` / ``nan``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.retention`` and ``D.drift``
        ``(P, C)``, ``D.dynamics_mean`` ``(P, C, 2)`` and ``D.dynamics_covar``
        ``(P, C, 2, 2)``. Uses ``model.rng``.

    Returns
    -------
    None
    """
    d = model.D
    p_count = model.num_particles
    c_slots = model.max_contexts + 1
    rng = model.rng

    prior_prec = np.diag(
        [model.prior_precision_retention, model.prior_precision_drift]
    )                                                                   # (2, 2)
    prior_mean = np.array(
        [model.prior_mean_retention, model.prior_mean_drift]
    )                                                                   # (2,)
    q_var = model.sigma_process_noise ** 2
    # Zero process noise means infinite data precision; dividing by eps keeps
    # the posterior finite while collapsing it onto the least-squares fit.
    scale = q_var if q_var != 0 else EPS

    # Every cell is written below, so a fresh allocation matches MATLAB's
    # element-wise overwrite of the persistent field.
    d.dynamics_mean = np.zeros((p_count, c_slots, 2))            # (P, C, 2)
    d.dynamics_covar = np.zeros((p_count, c_slots, 2, 2))        # (P, C, 2, 2)

    prior_info = prior_prec @ prior_mean                                # (2,)
    for p in range(p_count):
        for c in range(c_slots):
            ss2 = d.dynamics_ss_2[p, c]                                 # (2, 2)
            ss1 = d.dynamics_ss_1[p, c]                                 # (2,)
            covar = safe_inverse(prior_prec + ss2 / scale)              # (2, 2)
            mu = covar @ (prior_info + ss1 / scale)                     # (2,)
            # Draw [a; d] truncated to the stable region a in [0, 1).
            sample = sample_bivariate_truncated(rng, mu, covar)         # (2,)
            d.retention[p, c] = sample[0]
            d.drift[p, c] = sample[1]
            d.dynamics_mean[p, c] = mu
            d.dynamics_covar[p, c] = covar


def sample_bias(model):
    """Sample the per-context observation bias.

    With ``infer_bias`` disabled the bias stays zero and its posterior moments
    degenerate to ``mean = bias``, ``var = 0``. Otherwise the conjugate normal
    posterior formed from ``D.bias_ss_1`` / ``D.bias_ss_2`` and the observation
    variance is drawn from.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.bias``, ``D.bias_mean`` and
        ``D.bias_var``, each ``(P, C)``. Uses ``model.rng``.

    Returns
    -------
    None
    """
    d = model.D
    shape = (model.num_particles, model.max_contexts + 1)

    if not model.infer_bias:
        # Bias disabled: force b = 0 with zero posterior variance.
        d.bias = np.zeros(shape)                                        # (P, C)
        d.bias_mean = d.bias.copy()                                     # (P, C)
        d.bias_var = np.zeros(shape)                                    # (P, C)
        return

    obs_var = observation_variance(model)
    if obs_var == 0:
        # Zero observation noise implies infinite data precision; floor to eps
        # so bias_ss / obs_var stays finite (cf. the q_var == 0 guard above).
        obs_var = EPS
    # A completely uninformative prior (prior_precision_bias == 0) with no data
    # yet gives var_b = inf and mu_b = nan; sample_scalar_normal then treats the
    # entry as deterministic, exactly as MATLAB's min(max(NaN, -Inf), Inf - eps)
    # does. errstate only silences the warning - the values are unchanged.
    with np.errstate(divide="ignore", invalid="ignore"):
        var_b = 1.0 / (model.prior_precision_bias + d.bias_ss_2 / obs_var)  # (P, C)
        mu_b = var_b * (
            model.prior_precision_bias * model.prior_mean_bias
            + d.bias_ss_1 / obs_var
        )                                                               # (P, C)
    d.bias_mean = mu_b
    d.bias_var = var_b
    # Untruncated Gaussian draw (the bias is unconstrained on the real line).
    d.bias = sample_scalar_normal(
        model.rng, mu_b, var_b, shape, -np.inf, np.inf
    )                                                                   # (P, C)

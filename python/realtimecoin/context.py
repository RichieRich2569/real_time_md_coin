"""Shared context machinery: transition/cue matrices and the HDP samplers.

Direct translation of the dimension-agnostic half of the COIN pipeline from
``@RealTimeCOIN/private/``: ``predictContext``, ``updateLocalTransitionMatrix``,
``updateLocalCueMatrix``, ``currentTransitionPrior``, ``nContextSlice``,
``kappa``, ``sampleGlobalTransitionProbabilities``,
``sampleGlobalCueProbabilities``, ``contextProbabilityVectorCore``,
``localContextProbabilityVector``, ``nextTrialContextWeights`` and
``previewCuePmf``.

Everything here operates on discrete context labels only, so the scalar and
multi-dimensional pipelines share it verbatim - exactly as in MATLAB, where
``predictContext`` is called from both branches of ``observe_y``.

Model
-----
Contexts follow a sticky hierarchical Dirichlet process (Teh et al. 2006;
Fox et al. 2011, "A Sticky HDP-HMM with Application to Speaker Diarization",
Annals of Applied Statistics 5(2)), the prior used by the COIN model of Heald,
Lengyel & Wolpert (2021)::

    beta   ~ GEM(gamma_context)                       global franchise weights
    pi_j   ~ DP(alpha + kappa, (alpha*beta + kappa*delta_j) / (alpha + kappa))
    kappa  = alpha_context * rho_context / (1 - rho_context)

with ``kappa`` the extra self-transition ("stickiness") mass. The cue level is
the same hierarchy WITHOUT the sticky term::

    beta_cue ~ GEM(gamma_cue)
    l_c      ~ DP(alpha_cue, beta_cue)

Layout: all arrays are particles-leading, see :mod:`realtimecoin.state`.
Context labels are 0-based; slot ``n_active[p]`` is particle ``p``'s novel
context.
"""

from __future__ import annotations

import numpy as np

from .numerics import normalize_columns, renormalize_global_weights
from .samplers import binomial_sample, dirichlet_sample
from .state import ensure_cue_column
from .statics import sample_num_tables

__all__ = [
    "kappa",
    "n_context_slice",
    "update_local_transition_matrix",
    "update_local_cue_matrix",
    "predict_context",
    "current_transition_prior",
    "next_trial_context_weights",
    "preview_cue_pmf",
    "sample_global_transition_probabilities",
    "sample_global_cue_probabilities",
    "context_probability_vector_core",
    "local_context_probability_vector",
]

REALMIN = float(np.finfo(float).tiny)


def kappa(model):
    """Self-transition (stickiness) concentration of the sticky HDP prior.

    ``kappa = alpha_context * rho_context / (1 - rho_context)`` - the extra
    prior weight placed on staying in the current context. The denominator is
    floored at ``realmin`` so ``rho_context -> 1`` yields a large finite kappa
    instead of a divide-by-zero.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``alpha_context`` and ``rho_context``.

    Returns
    -------
    float
        The self-transition concentration.
    """
    # fmax, not max: MATLAB's two-argument max omits NaN.
    return float(
        model.alpha_context
        * model.rho_context
        / np.fmax(1.0 - model.rho_context, REALMIN)
    )


def _valid_context_mask(model, k_active):
    """Boolean mask of the context slots a particle may currently occupy.

    A context is "valid" if it is already instantiated (slots ``0 .. k-1``) or
    is the single next novel slot (``k``), the latter only while the particle is
    below the ``max_contexts`` cap. MATLAB spells this rule out separately in
    ``updateLocalTransitionMatrix``, ``updateLocalCueMatrix`` and
    ``sampleGlobalTransitionProbabilities``; it is the subtlest invariant in the
    file, so it lives in one place here.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``max_contexts``.
    k_active : int
        Number of contexts the particle has instantiated (``D.n_active[p]``).

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` boolean mask.
    """
    c_slots = model.max_contexts + 1
    valid = np.zeros(c_slots, dtype=bool)
    valid[:k_active] = True
    if k_active < model.max_contexts:
        valid[k_active] = True
    return valid


def _row_normalise_valid(raw):
    """Row-normalise a matrix, leaving all-zero rows exactly zero.

    This is NOT :func:`realtimecoin.numerics.normalize_columns`: that helper
    collapses a degenerate slice onto ``e_0``, whereas here a row belonging to a
    context that cannot occur yet must stay exactly zero.

    Parameters
    ----------
    raw : numpy.ndarray
        ``(rows, cols)`` matrix of non-negative unnormalised weights.

    Returns
    -------
    numpy.ndarray
        ``raw`` with every strictly-positive-sum row divided by its sum.
    """
    out = np.zeros(raw.shape)
    row_sums = raw.sum(axis=1)
    nz = row_sums > 0             # valid rows; all-zero rows stay zero
    out[nz, :] = raw[nz, :] / row_sums[nz][:, None]
    return out


def n_context_slice(model, p):
    """Context-transition count matrix of one particle.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    p : int
        Particle index.

    Returns
    -------
    numpy.ndarray
        ``(C, C)`` matrix of observed from-context (row) to-context (column)
        transition counts, i.e. ``model.D.n_context[p]``.
    """
    return model.D.n_context[p]


def update_local_transition_matrix(model):
    """Rebuild the per-particle context transition matrices.

    Writes ``model.D.local_transition_matrix``, a ``(P, C, C)`` tensor whose
    ``[p, r, :]`` row is the posterior-mean distribution over the next context
    given current context ``r``. Each row is the expectation of a Dirichlet
    whose parameters are the observed counts plus the sticky HDP-HMM prior::

        raw[r, c] = n_context[p, r, c]                  (observed counts)
                  + alpha_context * beta[p, c]          (HDP base measure)
                  + kappa * (r == c)                    (self-transition mass)

    normalised over ``c``. Rows and columns of contexts that do not yet exist
    (beyond the instantiated count plus one novel slot) are zeroed before
    normalisation, so an unreachable context can never draw mass.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.

    Returns
    -------
    None
    """
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    kappa_self = kappa(model)
    d = model.D
    transition = np.zeros((p_count, c_slots, c_slots))          # (P, C, C)
    identity = np.eye(c_slots)
    for p in range(p_count):
        # Unnormalised Dirichlet parameters. beta is broadcast along the
        # DESTINATION axis (the trailing one), matching MATLAB's row-vector
        # `alpha_context .* beta'` added to a (from, to) matrix.
        raw = (
            n_context_slice(model, p)
            + model.alpha_context * d.global_transition_probabilities[p][None, :]
            + kappa_self * identity
        )                                                        # (C, C)
        valid = _valid_context_mask(model, int(d.n_active[p]))   # (C,)
        # Zero out rows and columns of contexts that cannot occur yet.
        raw[:, ~valid] = 0.0
        raw[~valid, :] = 0.0
        transition[p] = _row_normalise_valid(raw)                # (C, C)
    d.local_transition_matrix = transition                       # (P, C, C)


def update_local_cue_matrix(model):
    """Rebuild the per-particle context-to-cue emission matrices.

    Writes ``model.D.local_cue_matrix``, a ``(P, C, Q)`` tensor whose
    ``[p, c, :]`` row is the posterior-mean distribution over cue emissions for
    context ``c``. Each row is the mean of a Dirichlet combining the observed
    cue counts with the global HDP cue base measure::

        raw[c, k] = n_cue[p, c, k] + alpha_cue * beta_cue[p, k]

    normalised over ``k``. Rows of contexts that do not yet exist are zeroed
    before normalisation. Structurally mirrors
    :func:`update_local_transition_matrix` but has NO sticky self-transition
    term - cues have no self-loop.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.

    Returns
    -------
    None
    """
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    d = model.D
    n_q = max(1, d.global_cue_probabilities.shape[1])            # cue columns
    emission = np.zeros((p_count, c_slots, n_q))                 # (P, C, Q)
    # Loop-invariant: how many cue columns n_cue actually carries. The counts
    # are trimmed to the known cue values and right-padded with zeros when
    # fewer columns have been instantiated.
    available = min(n_q, d.n_cue.shape[2])
    for p in range(p_count):
        counts = np.zeros((c_slots, n_q))                        # (C, Q)
        counts[:, :available] = d.n_cue[p, :, :available]
        # Unnormalised Dirichlet parameters: counts + HDP cue base measure.
        raw = counts + model.alpha_cue * d.global_cue_probabilities[p, :n_q][None, :]
        valid = _valid_context_mask(model, int(d.n_active[p]))   # (C,)
        raw[~valid, :] = 0.0        # zero out non-existent context rows
        emission[p] = _row_normalise_valid(raw)                  # (C, Q)
    d.local_cue_matrix = emission                                # (P, C, Q)


def predict_context(model, q):
    """Predict the per-particle context distribution for this trial.

    Forms the prior over context assignments for the upcoming observation and,
    when a cue is present, tilts it by the cue likelihood ``p(q | c)``. This is
    the first step of the per-trial pipeline and is shared by the scalar and
    multi-dimensional paths.

    The prior for particle ``p`` is the transition row out of that particle's
    current context, taken from the local transition matrix::

        prior[p, :] = T[p, context[p], :]         (then normalised)

    When a cue is observed the predicted distribution multiplies in the cue
    emission probabilities and re-normalises::

        predicted[p, :] = normalise( prior[p, :] * p(q | c, p) )

    This step owns its dependencies: it refreshes the local transition matrix on
    entry, and the local cue matrix ONLY when a cue is present, so it has no
    hidden ordering dependency on an external caller. Neither refresh is hoisted
    out of the branch - the un-cued path deliberately does not touch the cue
    matrix.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.prior_probabilities``,
        ``D.probability_cue`` and ``D.predicted_probabilities``, each
        ``(P, C)``; the trailing context slot is the novel-context slot.
    q : int or None
        0-based cue label for this trial, or ``None`` for an un-cued trial. A
        non-``None`` ``q`` indexes the trailing axis of ``D.local_cue_matrix``.

    Returns
    -------
    None
    """
    update_local_transition_matrix(model)
    p_count = model.num_particles
    c_slots = model.max_contexts + 1
    d = model.D
    # prior[p, :] = transition row out of particle p's current context. MATLAB
    # spells this out twice (predictContext.m uses a linear-index gather,
    # currentTransitionPrior.m a loop); neither form survives the
    # particles-leading transposition, so both share the one helper here.
    d.prior_probabilities = current_transition_prior(model)               # (P, C)

    if q is None:
        # Un-cued trial: the cue term is a no-op, so predicted == prior.
        d.probability_cue = np.ones((p_count, c_slots))                   # (P, C)
        # Copy, not alias: MATLAB struct assignment has value semantics.
        d.predicted_probabilities = d.prior_probabilities.copy()          # (P, C)
    else:
        update_local_cue_matrix(model)
        # Gather p(q | c, p) for the observed cue across all contexts/particles.
        # .copy() is load-bearing: integer indexing returns a strided VIEW into
        # local_cue_matrix, whereas MATLAB's squeeze() yields an independent
        # array. Without it, a later in-place write to probability_cue would
        # silently corrupt the cue matrix (and pin the whole (P, C, Q) tensor).
        p_cue = d.local_cue_matrix[:, :, q].copy()                        # (P, C)
        d.probability_cue = p_cue
        d.predicted_probabilities = normalize_columns(
            d.prior_probabilities * p_cue
        )                                                                 # (P, C)


def current_transition_prior(model):
    """Normalised one-step context transition prior (read-only).

    Returns the ``(P, C)`` matrix whose ``p``-th row is particle ``p``'s
    currently sampled context row of its local transition matrix, normalised to
    sum to one. This is the read-only context-weight term shared by the preview
    helpers and :func:`preview_cue_pmf`. Does NOT mutate model state (in
    particular it does not refresh the local transition matrix).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(P, C)`` row-normalised transition prior.
    """
    d = model.D
    p_count = model.num_particles
    prior = d.local_transition_matrix[np.arange(p_count), d.context, :]   # (P, C)
    return normalize_columns(prior)


def next_trial_context_weights(model, q_label):
    """One-step-ahead predicted context weights (read-only).

    Returns the ``(P, C)`` normalised context weights the model would predict on
    the NEXT trial given the optional upcoming cue label, by propagating each
    particle's currently sampled context through its expected local transition
    matrix and (when a cue is supplied) multiplying by the local cue likelihood.
    Exactly the weight term of :func:`predict_context`, without mutating any
    state. Dimension-agnostic. Used by the ``c*2`` query methods.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q_label : int or None
        0-based cue label of the upcoming trial; ``None`` marginalises the cue
        out. A label beyond the instantiated columns is clamped to the last
        column (mirroring MATLAB's ``min(qLabel, size(L, 2))``).

    Returns
    -------
    numpy.ndarray
        ``(P, C)`` normalised next-trial context weights.
    """
    prior = current_transition_prior(model)                    # (P, C)
    if q_label is None:
        return prior
    d = model.D
    q_col = min(int(q_label), d.local_cue_matrix.shape[2] - 1)
    p_cue = d.local_cue_matrix[:, :, q_col]                    # (P, C)
    return normalize_columns(prior * p_cue)                    # (P, C)


def preview_cue_pmf(model):
    """Predicted distribution over cue labels for the next trial (read-only).

    For each particle the sampled context is propagated through its local
    transition matrix, the per-context cue likelihoods are weighted by the
    resulting context distribution, and the result is averaged across particles
    and renormalised.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read. Not mutated.

    Returns
    -------
    pmf : numpy.ndarray
        ``(Q,)`` predicted cue probabilities, summing to one unless degenerate.
    labels : numpy.ndarray
        ``(Q,)`` 0-based integer cue labels ``0 .. Q-1`` indexing
        ``model.cue_values``.
    """
    d = model.D
    p_count = model.num_particles
    n_q = d.local_cue_matrix.shape[2]
    labels = np.arange(n_q)
    pmf = np.zeros(n_q)
    prior = current_transition_prior(model)                    # (P, C)
    for p in range(p_count):
        # (C,) @ (C, Q) -> (Q,). No zero-padding branch is needed here: unlike
        # MATLAB, local_cue_matrix always carries the full C context rows.
        pmf = pmf + prior[p] @ d.local_cue_matrix[p]
    pmf = pmf / p_count
    total = pmf.sum()
    if total > 0:
        pmf = pmf / total
    return pmf, labels


def sample_global_transition_probabilities(model):
    """Resample the global transition distribution (sticky HDP-HMM).

    Updates, per particle, the global ("beta") distribution over contexts that
    anchors the HDP prior on each context's transition row. Implements the
    sticky HDP-HMM posterior of Fox et al. (2011):

    1. Draw Antoniak/CRP table counts ``m`` for the observed transition counts
       ``n_context``, with base measure ``alpha_context * beta`` plus the sticky
       self-transition mass ``kappa`` on the diagonal::

           base[j, c] = alpha_context * beta[c] + kappa * (j == c)

    2. Remove the "override" self-transition tables attributed to the sticky
       term via ``Binomial(m_jj, rho / (rho + beta_j (1 - rho)))``, so only the
       shared (non-sticky) mass propagates upward.
    3. Sum tables across source contexts, add the novelty mass
       ``gamma_context`` at the first unused slot, zero everything beyond it,
       and draw the refreshed beta from that Dirichlet.

    Two guards are load-bearing and preserved verbatim: the ``realmin`` floor on
    the Binomial probability, and the floor that forces ``m[0, 0] = 1`` when it
    would otherwise be zero (context 0 is always present, so it must contribute
    at least one table; without this a particle whose self-transition tables were
    all stripped would get a degenerate all-zero Dirichlet parameter). When a
    particle has hit ``max_contexts`` there is NO novel slot at all.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.global_transition_probabilities``
        ``(P, C)``. Uses ``model.rng``.

    Returns
    -------
    None

    See Also
    --------
    sample_global_cue_probabilities : the non-sticky cue counterpart.
    """
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    sticky_kappa = kappa(model)
    rng = model.rng
    d = model.D
    sticky_diag = sticky_kappa * np.eye(c_slots)   # loop-invariant (C, C)
    for p in range(p_count):
        beta = d.global_transition_probabilities[p]                    # (C,)
        # Antoniak base measure: shared alpha*beta plus the sticky diagonal.
        base = model.alpha_context * beta[None, :] + sticky_diag
        m = sample_num_tables(rng, base, d.n_context[p])                # (C, C)
        if model.rho_context > 0:
            # Strip the sticky "override" tables from each diagonal so only the
            # shared mass propagates to the top-level Dirichlet (Fox et al.).
            # fmax, not max: MATLAB's two-argument max omits NaN. Computed for
            # every j at once; the binomial draws below stay in MATLAB's order.
            override_prob = model.rho_context / np.fmax(
                model.rho_context + beta * (1.0 - model.rho_context), REALMIN
            )                                                          # (C,)
            for j in range(c_slots):
                if m[j, j] > 0:
                    m[j, j] = m[j, j] - binomial_sample(
                        rng, m[j, j], float(override_prob[j])
                    )
        # Numerical floor: context 0 is always present (see docstring).
        if m[0, 0] == 0:
            m[0, 0] = 1
        alpha = m.sum(axis=0)                    # (C,) tables summed over sources
        k_active = int(d.n_active[p])
        if k_active < model.max_contexts:
            alpha[k_active] = model.gamma_context   # novelty mass, first unused
            alpha[k_active + 1:] = 0.0              # EXACT zeros beyond it
        else:
            alpha[k_active:] = 0.0                  # at the cap: no novel slot
        d.global_transition_probabilities[p] = dirichlet_sample(rng, alpha)


def sample_global_cue_probabilities(model):
    """Resample the global cue distribution (HDP top level).

    Updates, per particle, the global ("beta") distribution over cues that
    anchors the HDP prior on each context's cue emission distribution, following
    the standard Teh et al. posterior for the top-level measure:

    1. Draw Antoniak/CRP table counts ``m`` for the observed cue counts
       ``n_cue``, with base measure ``alpha_cue * beta_cue`` broadcast over
       contexts.
    2. Sum tables across contexts, add the novelty mass ``gamma_cue`` at the
       first unused ("novel cue") slot, and zero any slot beyond it.
    3. Draw the refreshed global cue probabilities from that Dirichlet.

    Unlike the transition sampler there is NO self-transition (sticky) term.
    Returns immediately until at least one cue column exists.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Writes ``D.global_cue_probabilities``
        ``(P, Q)`` and may grow the cue axis. Uses ``model.rng``.

    Returns
    -------
    None
    """
    d = model.D
    # Nothing to do until at least one cue has been registered.
    if d.global_cue_probabilities.shape[1] == 0:
        return
    p_count = model.num_particles
    rng = model.rng
    # Column budget: at least one slot past the highest seen cue label.
    n_q = max(d.n_cues + 1, d.global_cue_probabilities.shape[1])
    ensure_cue_column(model, n_q)
    for p in range(p_count):
        counts = d.n_cue[p, :, :n_q]                              # (C, Q)
        # Antoniak base measure alpha_cue * beta. MATLAB repmat's this over the
        # context rows; sample_num_tables broadcasts base to the shape of
        # counts itself, so the (Q,) row is passed straight through.
        base = model.alpha_cue * d.global_cue_probabilities[p, :n_q]   # (Q,)
        m = sample_num_tables(rng, base, counts)                  # (C, Q)
        alpha = m.sum(axis=0)                                     # (Q,)
        alpha[d.n_cues] = model.gamma_cue    # novelty mass at first unused slot
        if d.n_cues + 2 <= n_q:
            alpha[d.n_cues + 1:] = 0.0       # zero slots beyond the novel cue
        d.global_cue_probabilities[p, :n_q] = dirichlet_sample(rng, alpha)


def context_probability_vector_core(
    kind, predicted_fn, responsibilities_fn, count_fn, edges
):
    """Shared kind-dispatch for the context probability vectors.

    Computes the raw (unnormalised) context probability vector selected by
    ``kind``. It is the common core shared by the global-frame
    ``context_probability_vector`` (in :mod:`realtimecoin.alignment`) and the
    local-frame :func:`local_context_probability_vector`; each caller supplies
    frame-specific sources and applies its own trailing-slot handling and final
    renormalisation.

    Only the source for the selected ``kind`` is evaluated, hence the
    zero-argument callables.

    Parameters
    ----------
    kind : {'predicted', 'responsibilities', 'count'}
        Quantity to summarise: the mean of the predicted context probabilities,
        the mean of the context responsibilities, or the empirical frequency of
        the sampled contexts.
    predicted_fn : callable
        Returns the ``(P_sel, C)`` per-particle predicted-probability matrix.
    responsibilities_fn : callable
        Returns the ``(P_sel, C)`` per-particle responsibilities matrix.
    count_fn : callable
        Returns the ``(P_sel,)`` array of sampled context labels.
    edges : array_like
        Integer-centred histogram edges used by the ``"count"`` kind. With
        0-based labels these are ``-0.5, 0.5, ..., C - 0.5``, so label ``k``
        lands in bin ``k``.

    Returns
    -------
    numpy.ndarray
        ``(len(edges) - 1,)`` unnormalised weights. Callers handle slot zeroing
        and the guarded renormalisation.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three recognised values.
    """
    if kind == "predicted":
        return np.mean(predicted_fn(), axis=0)
    if kind == "responsibilities":
        return np.mean(responsibilities_fn(), axis=0)
    if kind == "count":
        c = np.asarray(count_fn()).reshape(-1)
        counts, _ = np.histogram(c, bins=np.asarray(edges, dtype=float))
        return counts / max(c.size, 1)
    raise ValueError(
        "RealTimeCOIN:BadContextVectorKind: kind must be one of 'predicted', "
        "'responsibilities' or 'count'; got %r." % (kind,)
    )


def local_context_probability_vector(model, kind):
    """Local-frame context probability vector.

    Returns a normalised probability over the LOCAL context slots, averaged over
    the modal particles, WITHOUT aligning labels across particles. It is the
    raw, per-particle-label counterpart of the global-frame
    ``context_probability_vector``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    kind : {'predicted', 'responsibilities', 'count'}
        Quantity to summarise; see :func:`context_probability_vector_core`.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` row summing to one, or all zeros when no modal
        particle exists. Slots beyond the modal cardinality are zeroed: at the
        cap there is no novel slot, otherwise the single novel slot is kept.
    """
    from . import alignment as _alignment   # local import: avoids a cycle

    c_slots = model.max_contexts + 1
    km, _modal_mask, modal_idx, _weights = _alignment.select_modal_contexts(model)
    modal_idx = np.asarray(modal_idx).reshape(-1)
    if modal_idx.size == 0:
        return np.zeros(c_slots)

    # Integer-centred bin edges -0.5, 0.5, ..., C - 0.5 so that each 0-based
    # context label k lands in its own bin k.
    edges = np.arange(c_slots + 1) - 0.5
    d = model.D
    weights = context_probability_vector_core(
        kind,
        lambda: d.predicted_probabilities[modal_idx, :],
        lambda: d.responsibilities[modal_idx, :],
        lambda: d.context[modal_idx],
        edges,
    )

    if km >= model.max_contexts:
        # All contexts instantiated: no novel slot beyond km.
        weights[km:] = 0.0
    elif km + 2 <= c_slots:
        # Keep the km used slots plus the single novel slot (index km).
        # The guard is vacuous - the branch above already gives
        # km < max_contexts, and c_slots == max_contexts + 1 - but it is kept
        # to stay line-for-line with localContextProbabilityVector.m.
        weights[km + 1:] = 0.0

    return renormalize_global_weights(weights)

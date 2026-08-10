"""Context-facing queries that require the global context alignment.

STUB MODULE - implemented by unit B3.

Translated from the public methods ``predicted_context_probabilities_vector``,
``predicted_context_probabilities_map``, ``responsibilities_vector``,
``responsibilities_map``, ``sampled_context_count``,
``stationary_context_probabilities``, ``global_transition_probabilities``,
``global_cue_probabilities``, ``local_transition_probabilities``,
``local_cue_probabilities``, ``context_alignment``, ``diagnostics``,
``state_given_context_probability`` and
``state_feedback_given_context_probability``.

Every function here calls ``ensure_context_alignment``, so each one TRIGGERS
AND CACHES the lazy alignment (see :mod:`realtimecoin.alignment`). That is the
only state they mutate; the particle arrays are read-only to them.

The two density queries live here rather than in
:mod:`realtimecoin.queries_density` precisely because they are per-GLOBAL-context
and therefore need the alignment; the marginal densities do not.

Maps are returned as plain Python ``dict`` objects keyed by the integer global
context label, standing in for MATLAB's ``containers.Map``. Keys are the
0-based global labels, and only contexts with strictly positive weight appear.
"""

from __future__ import annotations

__all__ = [
    "predicted_context_probabilities_vector",
    "predicted_context_probabilities_map",
    "responsibilities_vector",
    "responsibilities_map",
    "sampled_context_count",
    "stationary_context_probabilities",
    "global_transition_probabilities",
    "global_cue_probabilities",
    "local_transition_probabilities",
    "local_cue_probabilities",
    "context_alignment",
    "diagnostics",
    "diagnostics_md",
    "state_given_context_probability",
    "state_feedback_given_context_probability",
]

_UNIT = "implemented by unit B3"


def predicted_context_probabilities_vector(model):
    """Predicted (pre-observation) per-context probabilities, as a vector.

    Averaged over the modal particles and mapped into the aligned global-context
    frame; the trailing entry is the novel-context probability.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` probabilities.
    """
    raise NotImplementedError(_UNIT)


def predicted_context_probabilities_map(model):
    """Predicted per-context probabilities, as a label-keyed mapping.

    Same weights as :func:`predicted_context_probabilities_vector`, but only
    contexts with strictly positive probability appear as keys.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        ``{global_context_label: probability}``.
    """
    raise NotImplementedError(_UNIT)


def responsibilities_vector(model):
    """Posterior (post-observation) per-context responsibilities, as a vector.

    The posterior counterpart of
    :func:`predicted_context_probabilities_vector`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` responsibilities.
    """
    raise NotImplementedError(_UNIT)


def responsibilities_map(model):
    """Posterior context responsibilities, as a label-keyed mapping.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        ``{global_context_label: responsibility}``, positive entries only.
    """
    raise NotImplementedError(_UNIT)


def sampled_context_count(model):
    """Sampled-context occupancy across particles, normalised to sum to one.

    The fraction of particles whose sampled context for the current trial equals
    each aligned global context; the trailing entry is the novel context.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` frequencies.
    """
    raise NotImplementedError(_UNIT)


def stationary_context_probabilities(model):
    """Stationary distribution of the aligned context chain.

    Mirrors COIN's ``plot_stationary_probabilities``: the novel-context column
    is dropped and each row renormalised to form a stochastic matrix over the
    instantiated contexts before solving for its stationary distribution.
    Dimension-independent.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` stationary probabilities, ``K`` being the aligned context
        count.
    """
    raise NotImplementedError(_UNIT)


def global_transition_probabilities(model):
    """Expected global (franchise) transition distribution.

    The expected global context distribution of the hierarchical Dirichlet
    process, averaged over the modal particles and mapped into the aligned
    frame. Entries beyond the number of active contexts are zero; the last
    active entry is the novel-context stick.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` franchise weights.
    """
    raise NotImplementedError(_UNIT)


def global_cue_probabilities(model):
    """Expected global (franchise) cue distribution.

    Averaged over particles. Cue labels are numbered by order of presentation,
    so they need no alignment; the trailing entry is the novel-cue stick.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(Q,)`` franchise cue weights.

    Raises
    ------
    NoCuesError
        If no sensory cue has been observed yet.
    """
    raise NotImplementedError(_UNIT)


def local_transition_probabilities(model):
    """Expected local context-transition matrix in the aligned frame.

    Row ``i`` is the transition distribution out of global context ``i``;
    columns ``0 .. K-1`` are the known contexts and column ``K`` is the novel
    context. Dimension-independent.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(K, K + 1)`` transition probabilities.
    """
    raise NotImplementedError(_UNIT)


def local_cue_probabilities(model):
    """Expected local cue-emission matrix in the aligned frame.

    Row ``i`` is the cue-emission distribution of global context ``i``, over the
    observed cue labels plus a trailing novel-cue column.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(K, Q)`` cue-emission probabilities.

    Raises
    ------
    NoCuesError
        If no sensory cue has been observed yet.
    """
    raise NotImplementedError(_UNIT)


def context_alignment(model):
    """Global context alignment across particles (cached).

    Public accessor for the structure that maps each particle's arbitrary local
    context labels onto one globally consistent labelling for the current
    trial. Computed lazily and cached; the cache is invalidated by each
    ``observe_y``, so repeated context-facing queries within a trial share one
    alignment.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose alignment cache is read and possibly written.

    Returns
    -------
    dict
        The alignment structure; see
        :func:`realtimecoin.alignment.compute_context_alignment` for the keys.
    """
    raise NotImplementedError(_UNIT)


def diagnostics(model):
    """Full globally-aligned snapshot of the current particle state.

    A structure summarising every per-context quantity of the current trial,
    relabelled into the aligned global-context frame. Delegates to
    :func:`diagnostics_md` when ``state_dim > 1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        Scalar-model keys (``K`` = aligned context count): ``trial``, ``C``,
        ``context``, ``predicted_probabilities``, ``responsibilities``,
        ``state_mean``, ``state_var``, ``state_feedback_mean``,
        ``state_feedback_var``, ``retention``, ``drift``, ``bias``,
        ``global_transition_probabilities``, ``local_transition_matrix``,
        ``global_cue_probabilities``, ``local_cue_matrix``, ``alignment`` and
        ``raw`` (a reference to the raw particle state).
    """
    raise NotImplementedError(_UNIT)


def diagnostics_md(model):
    """Multi-dimensional counterpart of :func:`diagnostics`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        The MD diagnostic structure (matrix-valued state and dynamics fields in
        place of their scalar counterparts).
    """
    raise NotImplementedError(_UNIT)


def state_given_context_probability(model, values):
    """Per-context posterior latent-state density on a grid.

    Uses the aligned global-context prototype moments, so the keys are stable
    global labels.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        Query grid: ``(K_pts,)`` for the scalar model, ``(K_pts, N)`` (one query
        point per ROW) for the multi-dimensional model.

    Returns
    -------
    dict
        ``{global_context_label: numpy.ndarray}``, each value a
        ``(K_pts,)`` density.
    """
    raise NotImplementedError(_UNIT)


def state_feedback_given_context_probability(model, values):
    """Per-context predictive feedback density on a grid.

    Observation-space counterpart of
    :func:`state_given_context_probability`: the mean is shifted by the learned
    observation bias and the covariance inflated by the observation noise ``R``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        Query grid: ``(K_pts,)`` scalar, ``(K_pts, N)`` multi-dimensional.

    Returns
    -------
    dict
        ``{global_context_label: numpy.ndarray}``, each value a
        ``(K_pts,)`` density.
    """
    raise NotImplementedError(_UNIT)

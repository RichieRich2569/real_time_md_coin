"""Context-facing queries that require the global context alignment.

Translated from the public methods ``predicted_context_probabilities_vector``,
``predicted_context_probabilities_map``, ``responsibilities_vector``,
``responsibilities_map``, ``sampled_context_count``,
``stationary_context_probabilities``, ``global_transition_probabilities``,
``global_cue_probabilities``, ``local_transition_probabilities``,
``local_cue_probabilities``, ``context_alignment``, ``diagnostics``
(+ ``private/diagnosticsMD``), ``state_given_context_probability`` and
``state_feedback_given_context_probability``.

Every function here calls ``ensure_context_alignment``, so each one TRIGGERS
AND CACHES the lazy alignment (see :mod:`realtimecoin.alignment`). That is the
only state they mutate; the particle arrays are read-only to them.

The two density queries live here rather than in
:mod:`realtimecoin.queries_density` precisely because they are per-GLOBAL-context
and therefore need the alignment; the marginal densities do not.

Maps are returned as plain Python ``dict`` objects keyed by the integer global
context label, standing in for MATLAB's ``containers.Map``. Keys are the
0-BASED global labels - one less than the MATLAB key for the same context -
so that ``some_map[c]`` and ``some_vector[c]`` always name the same context.
Only contexts with strictly positive weight appear.
"""

from __future__ import annotations

import numpy as np

from .alignment import (
    clamp_active_summary_contexts,
    context_probability_vector,
    ensure_context_alignment,
    global_context_matrix,
    global_context_weights,
    global_cue_tensor,
    global_sampled_contexts,
    global_transition_tensor,
)
from .exceptions import NoCuesError
from .numerics import gaussian_pdf_columns_md, renormalize_global_weights
from .state import feedback_transform, observation_noise_cov
from .statics import normal_pdf, stationary_distribution

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

#: Per-context fields of the scalar diagnostics that share one relabelling.
#: Each name matches its raw ``ParticleState`` field, so one table drives them
#: all (as in ``diagnostics.m``).
_SCALAR_MATRIX_FIELDS = (
    "state_mean",
    "state_var",
    "state_feedback_mean",
    "state_feedback_var",
    "retention",
    "drift",
    "bias",
    "global_transition_probabilities",
)


def _positive_entry_map(weights):
    """Turn a weight vector into a ``{label: weight}`` dict of positive entries.

    The Python stand-in for MATLAB's ``containers.Map`` construction loop in
    ``predicted_context_probabilities_map`` / ``responsibilities_map``.

    Parameters
    ----------
    weights : array_like
        Per-global-context weights.

    Returns
    -------
    dict
        ``{0-based global context label: weight}``, strictly positive entries
        only.
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    return {
        int(c): float(w) for c, w in enumerate(weights) if w > 0
    }


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
    return context_probability_vector(model, "predicted")


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
        ``{0-based global context label: probability}``.
    """
    return _positive_entry_map(context_probability_vector(model, "predicted"))


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
    return context_probability_vector(model, "responsibilities")


def responsibilities_map(model):
    """Posterior context responsibilities, as a label-keyed mapping.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        ``{0-based global context label: responsibility}``, positive entries
        only.
    """
    return _positive_entry_map(context_probability_vector(model, "responsibilities"))


def sampled_context_count(model):
    """Sampled-context occupancy across particles, normalised to sum to one.

    The fraction of MODAL particles whose sampled context for the current trial
    equals each aligned global context; the trailing entry is the novel context.
    Particles whose sampled context has no global label contribute to the
    denominator but to no bin, so the raw counts can sum to less than one before
    the final renormalisation.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` frequencies.
    """
    return context_probability_vector(model, "count")


def stationary_context_probabilities(model):
    """Stationary distribution of the aligned context chain.

    Mirrors COIN's ``plot_stationary_probabilities``: the novel-context column
    is dropped and each row renormalised to form a stochastic matrix over the
    instantiated contexts before solving for its stationary distribution. A row
    left with no mass falls back to uniform. Dimension-independent.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` stationary probabilities, ``K`` being the aligned context
        count; empty when ``K == 0``.
    """
    alignment = ensure_context_alignment(model)
    k = int(alignment["K"])
    if k == 0:
        return np.zeros(0)
    # Copy: the alignment prototypes are cached and must not be renormalised.
    t = np.array(
        alignment["global_contexts"]["transition_prob"][:k, :k], dtype=float
    )                                                    # (K, K)
    row_sum = t.sum(axis=1)                              # (K,)
    zero_rows = row_sum <= 0
    t[~zero_rows] = t[~zero_rows] / row_sum[~zero_rows][:, None]
    t[zero_rows] = 1.0 / k                               # uniform fallback
    return stationary_distribution(t)


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
    alignment = ensure_context_alignment(model)
    wg = global_context_weights(
        model, model.D.global_transition_probabilities, alignment
    )                                       # (P_modal, C)
    return renormalize_global_weights(np.mean(wg, axis=0))


def global_cue_probabilities(model):
    """Expected global (franchise) cue distribution.

    Averaged over ALL particles (not just the modal ones). Cue labels are
    numbered by order of presentation, so they need no alignment; the trailing
    entry is the novel-cue stick.

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
    if not model.cue_values:
        raise NoCuesError(
            "global_cue_probabilities requires the model to have observed "
            "sensory cues."
        )
    return renormalize_global_weights(
        np.mean(model.D.global_cue_probabilities, axis=0)
    )


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
        ``(K, K + 1)`` transition probabilities (a copy; the cached prototypes
        are not exposed for mutation).
    """
    alignment = ensure_context_alignment(model)
    k = int(alignment["K"])
    return np.array(
        alignment["global_contexts"]["transition_prob"][:k, : k + 1], dtype=float
    )


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
        ``(K, Q)`` cue-emission probabilities (a copy).

    Raises
    ------
    NoCuesError
        If no sensory cue has been observed yet.
    """
    if not model.cue_values:
        raise NoCuesError(
            "local_cue_probabilities requires the model to have observed "
            "sensory cues."
        )
    alignment = ensure_context_alignment(model)
    k = int(alignment["K"])
    return np.array(
        alignment["global_contexts"]["cue_prob"][:k, :], dtype=float
    )


def context_alignment(model):
    """Global context alignment across particles (cached).

    Public accessor for the structure that maps each particle's arbitrary local
    context labels onto one globally consistent labelling for the current
    trial. Computed lazily and cached; the cache is invalidated by each
    ``observe_y``, so repeated context-facing queries within a trial share one
    alignment (and get back the very same object).

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
    return ensure_context_alignment(model)


def diagnostics(model):
    """Full globally-aligned snapshot of the current particle state.

    A structure summarising every per-context quantity of the current trial,
    relabelled into the aligned global-context frame. Delegates to
    :func:`diagnostics_md` when ``state_dim > 1``.

    Note the layout transposition relative to MATLAB: the per-particle arrays
    here have the MODAL PARTICLE axis LEADING, so ``S["responsibilities"]`` is
    ``(P_modal, C)`` where MATLAB's is ``(Cmax, nModal)``.

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
    if model.state_dim > 1:
        return diagnostics_md(model)

    alignment = ensure_context_alignment(model)
    d = model.D
    modal_idx = np.asarray(
        alignment["modal_particle_indices"], dtype=int
    ).reshape(-1)

    s = {
        "trial": model.trial,
        "C": int(alignment["K"]),
        "context": global_sampled_contexts(model, alignment),
        "predicted_probabilities": global_context_weights(
            model, d.predicted_probabilities, alignment
        ),
        "responsibilities": global_context_weights(
            model, d.responsibilities, alignment
        ),
    }
    for name in _SCALAR_MATRIX_FIELDS:
        s[name] = global_context_matrix(model, getattr(d, name), alignment)

    s["local_transition_matrix"] = global_transition_tensor(
        model, d.local_transition_matrix, alignment
    )
    s["global_cue_probabilities"] = d.global_cue_probabilities[modal_idx, :]
    s["local_cue_matrix"] = global_cue_tensor(model, d.local_cue_matrix, alignment)
    s["alignment"] = alignment
    s["raw"] = model.D
    return s


def diagnostics_md(model):
    """Multi-dimensional counterpart of :func:`diagnostics`.

    Per-context dynamics are reported as the augmented matrix
    ``Theta = [A | d]`` split into the retention matrices ``A`` and the drift
    vectors, alongside the filtered state mean/covariance and the (optional)
    observation bias. Context probabilities reuse the dimension-agnostic
    global-weight machinery.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        ``trial``, ``K``, ``A`` ``(K, N, N)``, ``drift`` ``(K, N)``, ``bias``
        ``(K, N)``, ``state_mean`` ``(K, N)``, ``state_cov`` ``(K, N, N)``,
        ``predicted_probabilities`` / ``responsibilities`` ``(K,)``,
        ``predicted_probabilities_particles`` /
        ``responsibilities_particles`` ``(P_modal, C)``, ``context``
        ``(P_modal,)``, ``transition_prob`` ``(K, K + 1)``, ``cue_prob``
        ``(K, Q)``, ``alignment`` and ``raw``.
    """
    n = model.state_dim
    alignment = ensure_context_alignment(model)
    proto = alignment["global_contexts"]
    km = int(alignment["K"])
    d = model.D

    theta = np.asarray(proto["theta_mean"], dtype=float)   # (K, N, N + 1)
    w_pred = global_context_weights(model, d.predicted_probabilities, alignment)
    w_resp = global_context_weights(model, d.responsibilities, alignment)

    return {
        "trial": model.trial,
        "K": km,
        # Per-context dynamics: split Theta = [A | d].
        "A": theta[:, :, :n].copy(),                       # (K, N, N)
        "drift": theta[:, :, n].copy(),                    # (K, N)
        "bias": np.array(proto["bias_mean"], dtype=float),
        "state_mean": np.array(proto["state_mean"], dtype=float),
        "state_cov": np.array(proto["state_cov"], dtype=float),
        # Context probabilities (aligned across the modal particles).
        "predicted_probabilities_particles": w_pred,       # (P_modal, C)
        "responsibilities_particles": w_resp,
        "predicted_probabilities": np.mean(w_pred[:, :km], axis=0),   # (K,)
        "responsibilities": np.mean(w_resp[:, :km], axis=0),          # (K,)
        "context": global_sampled_contexts(model, alignment),
        "transition_prob": np.array(proto["transition_prob"], dtype=float),
        "cue_prob": np.array(proto["cue_prob"], dtype=float),
        "alignment": alignment,
        "raw": model.D,
    }


def _validate_grid(model, values, name):
    """Validate and normalise a per-context density query grid.

    Layout note: the MATLAB originals take an ``N``-by-``K_pts`` array whose
    COLUMNS are the query points; following this package's convention the grid
    is ``(K_pts,)`` for the scalar model and ``(K_pts, N)`` - one query point
    per ROW - for the multi-dimensional model.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``state_dim``.
    values : array_like
        The caller's grid.
    name : str
        Query name, quoted in the error message.

    Returns
    -------
    numpy.ndarray
        ``(K_pts,)`` for the scalar model, ``(K_pts, N)`` otherwise.

    Raises
    ------
    ValueError
        If the grid is not finite/real (``RealTimeCOIN:GridNotFinite``) or has
        the wrong trailing dimension
        (``RealTimeCOIN:GridDimensionMismatch``).
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "RealTimeCOIN:GridNotFinite: %s expects a finite, real grid." % name
        )

    n = model.state_dim
    if n == 1:
        if values.ndim == 2 and 1 in values.shape:
            return values.reshape(-1)
        if values.ndim > 1:
            raise ValueError(
                "RealTimeCOIN:GridDimensionMismatch: %s expects a (K,) grid "
                "for state_dim == 1; received an array of shape %r."
                % (name, values.shape)
            )
        return values.reshape(-1)

    values = np.atleast_2d(values)
    if values.ndim != 2 or values.shape[1] != n:
        raise ValueError(
            "RealTimeCOIN:GridDimensionMismatch: %s expects a (K, %d) grid "
            "(one query point per row) for state_dim == %d; received an array "
            "of shape %r." % (name, n, n, values.shape)
        )
    return values


def state_given_context_probability(model, values):
    """Per-context posterior latent-state density on a grid.

    Uses the aligned global-context prototype moments, so the keys are stable
    global labels. Only the ACTIVE summary contexts (those with strictly
    positive predicted probability, clamped to the aligned label set) appear.

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
        ``{0-based global context label: numpy.ndarray}``, each value a
        ``(K_pts,)`` density.
    """
    grid = _validate_grid(model, values, "state_given_context_probability")
    alignment = ensure_context_alignment(model)
    proto = alignment["global_contexts"]
    active = clamp_active_summary_contexts(model, alignment)

    densities = {}
    if model.state_dim == 1:
        mean = proto["state_mean"]           # (K,)
        var = proto["state_var"]             # (K,)
        for c in active:
            densities[int(c)] = normal_pdf(grid, mean[c], var[c])
        return densities

    mean = proto["state_mean"]               # (K, N)
    cov = proto["state_cov"]                 # (K, N, N)
    for c in active:
        densities[int(c)] = gaussian_pdf_columns_md(grid, mean[c], cov[c])
    return densities


def state_feedback_given_context_probability(model, values):
    """Per-context predictive feedback density on a grid.

    Observation-space counterpart of
    :func:`state_given_context_probability`: the mean is shifted by the learned
    observation bias and the covariance inflated by the observation noise.

    .. note::
       The scalar branch inflates by ``sigma_sensory_noise ** 2`` ALONE, not by
       the full ``observation_variance`` (which also carries the motor-noise
       term), whereas the multi-dimensional branch uses the full
       ``observation_noise_cov``. That asymmetry is carried over verbatim from
       ``state_feedback_given_context_probability.m``; the two agree whenever
       ``sigma_motor_noise == 0`` (the default).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        Query grid: ``(K_pts,)`` scalar, ``(K_pts, N)`` multi-dimensional.

    Returns
    -------
    dict
        ``{0-based global context label: numpy.ndarray}``, each value a
        ``(K_pts,)`` density.
    """
    grid = _validate_grid(
        model, values, "state_feedback_given_context_probability"
    )
    alignment = ensure_context_alignment(model)
    proto = alignment["global_contexts"]
    active = clamp_active_summary_contexts(model, alignment)

    densities = {}
    if model.state_dim == 1:
        mean = proto["state_mean"]           # (K,)
        var = proto["state_var"]             # (K,)
        bias = proto["bias_mean"]            # (K,)
        r = model.sigma_sensory_noise ** 2
        for c in active:
            m, v = feedback_transform(mean[c], var[c], bias[c], r)
            densities[int(c)] = normal_pdf(grid, m, v)
        return densities

    mean = proto["state_mean"]               # (K, N)
    cov = proto["state_cov"]                 # (K, N, N)
    bias = proto["bias_mean"]                # (K, N)
    r = observation_noise_cov(model)         # (N, N)
    for c in active:
        m, v = feedback_transform(mean[c], cov[c], bias[c], r)
        densities[int(c)] = gaussian_pdf_columns_md(grid, m, v)
    return densities

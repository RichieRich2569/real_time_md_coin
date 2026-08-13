"""Independent re-derivation of the one-step predictive feedback mixture.

Translated from ``validation/validation_predictive_feedback_moments.m``.

This module re-derives the predictive feedback distribution of the SCALAR model
from :meth:`realtimecoin.RealTimeCOIN.diagnostics` alone (its ``"raw"`` particle
state plus the model's public hyperparameters). It deliberately does NOT call
:mod:`realtimecoin.queries_core`: the point is to have a second, independent
implementation of the same quantity so that the validators cross-check the
class's predictive path rather than merely re-running it. If the two ever
disagree, one of them is wrong - which is exactly the signal a scientific
validator should give.

The mixture is, for particle ``p`` and context ``c``::

    weight[p, c]   proportional to  T[p, context[p], c] * L[p, c, q]
    mean[p, c]     = a[p, c] * m_filt[p, c] + d[p, c] + bias[p, c]
    var[p, c]      = a[p, c]^2 * V_filt[p, c] + sigma_process^2
                     + sigma_sensory^2 + sigma_motor^2

with ``T`` the expected local transition matrix, ``L`` the expected local cue
likelihood and ``a`` / ``d`` the sampled retention / drift. The NOVEL context
slot of each particle has no filtering history, so its state moments are
replaced by the stationary moments of its sampled dynamics::

    mean = d / (1 - a)        var = sigma_process^2 / (1 - a^2)

both denominators floored at ``eps``, verbatim as MATLAB writes them.

Layout note
-----------
MATLAB is contexts-first / particles-last; this package is particles-leading.
Everywhere the MATLAB helper writes ``X(c, p)`` this module writes ``X[p, c]``,
and ``local_transition_matrix(i, :, p)`` becomes
``local_transition_matrix[p, i, :]``.
"""

from __future__ import annotations

import numpy as np

from .mixture_utils import mixture_moments

__all__ = ["predictive_feedback_mixture", "predictive_feedback_moments"]

EPS = float(np.finfo(float).eps)


def _normalize_rows(x):
    """Row-normalise a ``(P, C)`` weight array, uniform where a row has no mass.

    The transposed counterpart of the MATLAB helper's ``normalize_columns``
    (MATLAB normalises over the leading CONTEXT axis; here the context axis is
    trailing).

    Parameters
    ----------
    x : numpy.ndarray
        ``(P, C)`` unnormalised weights.

    Returns
    -------
    numpy.ndarray
        ``(P, C)`` weights whose rows each sum to one.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x >= 0.0), x, 0.0)
    den = x.sum(axis=1)                              # (P,)
    bad = den <= 0.0
    if np.any(bad):
        x[bad, :] = 1.0 / x.shape[1]
        den = np.where(bad, 1.0, den)
    return x / den[:, None]


def predictive_feedback_mixture(model, q=None):
    """Per-particle, per-context components of the predictive feedback mixture.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Scalar model (``state_dim == 1``) whose ``diagnostics()`` is read. The
        model is NOT mutated.
    q : int or None, optional
        0-based cue LABEL of the upcoming trial, clamped to the last
        instantiated cue column (MATLAB's ``min(q, size(L, 2))``). ``None``
        marginalises the cue out, leaving the transition prior alone.

    Returns
    -------
    weights : numpy.ndarray
        ``(P, C)`` mixture weights; each row sums to one.
    means : numpy.ndarray
        ``(P, C)`` component means.
    variances : numpy.ndarray
        ``(P, C)`` component variances.

    Raises
    ------
    ValueError
        If ``model.state_dim != 1``; the MATLAB helper reads scalar-only fields
        (``retention``, ``state_filtered_var``) and has no MD counterpart.
    """
    if model.state_dim != 1:
        raise ValueError(
            "predictive_feedback_mixture is only defined for the scalar model "
            "(state_dim == 1); state_dim == %d." % model.state_dim
        )

    raw = model.diagnostics()["raw"]
    c_max = model.max_contexts + 1
    p_count = model.num_particles

    # --- context weights: this particle's row of its own transition matrix ---
    particles = np.arange(p_count)
    prior = raw.local_transition_matrix[particles, raw.context, :]     # (P, C)
    prior = _normalize_rows(np.array(prior, dtype=float))

    if q is None:
        weights = prior
    else:
        q_col = min(int(q), raw.local_cue_matrix.shape[2] - 1)
        cue_likelihood = raw.local_cue_matrix[:, :, q_col]             # (P, C)
        weights = _normalize_rows(prior * cue_likelihood)

    # --- one-step state moments: s' = a s + d + w,  w ~ N(0, sigma_Q^2) ---
    state_mean = raw.retention * raw.state_filtered_mean + raw.drift   # (P, C)
    state_var = (
        raw.retention ** 2 * raw.state_filtered_var
        + model.sigma_process_noise ** 2
    )                                                                  # (P, C)

    # --- novel slot: stationary moments of its freshly sampled dynamics ---
    rows = np.nonzero(raw.n_active < model.max_contexts)[0]
    if rows.size:
        cols = np.minimum(raw.n_active[rows], c_max - 1)
        a = raw.retention[rows, cols]
        d = raw.drift[rows, cols]
        state_mean[rows, cols] = d / np.fmax(1.0 - a, EPS)
        state_var[rows, cols] = model.sigma_process_noise ** 2 / np.fmax(
            1.0 - a ** 2, EPS
        )

    # --- observation model: y = s + bias + sensory + motor noise ---
    means = state_mean + raw.bias
    variances = (
        state_var + model.sigma_sensory_noise ** 2 + model.sigma_motor_noise ** 2
    )
    return weights, means, variances


def predictive_feedback_moments(model, q=None):
    """One-step predictive feedback mean and variance, from diagnostics alone.

    The Gaussian-mixture reduction of :func:`predictive_feedback_mixture`.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Scalar model whose ``diagnostics()`` is read.
    q : int or None, optional
        0-based cue label of the upcoming trial; ``None`` marginalises it out.

    Returns
    -------
    mu : float
        Predictive feedback mean.
    v : float
        Predictive feedback variance.
    """
    weights, means, variances = predictive_feedback_mixture(model, q)
    return mixture_moments(weights, means, variances)

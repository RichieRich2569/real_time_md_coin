"""Marginal density and CDF queries (no context alignment needed).

Translated from the public methods ``state_probability``,
``state_feedback_probability``, ``novel_state_probability``,
``novel_state_feedback_probability``, ``retention_given_context_probability``,
``drift_given_context_probability``, ``bias_given_context_probability``,
``bias_probability``, ``predictive_state_feedback_cdf`` and
``predictive_cue_p_value``.

These are the density read-outs that marginalise over contexts (or address the
single novel context), so - unlike the per-context densities in
:mod:`realtimecoin.queries_aligned` - most of them do NOT need the global
context alignment. The three ``*_given_context`` densities kept here are the
scalar-only retention / drift / bias ones, which read the aligned prototype
moments and therefore do trigger the alignment; they live here because they
share the scalar-only density machinery with :func:`bias_probability`.

Grid convention: for the scalar model ``values`` is a ``(K,)`` vector of query
points and the returned density is ``(K,)``. For the multi-dimensional model
``values`` is ``(K, N)`` with ONE QUERY POINT PER ROW - the transpose of the
MATLAB ``N``-by-``K`` column convention, matching this package's
particles-leading / points-leading layout - and the density is still ``(K,)``.

Weight-source asymmetry (carried over verbatim from the MATLAB sources, and the
single most important thing to preserve here):

* :func:`state_probability` weights by the RESPONSIBILITIES (post-feedback) and
  uses the FILTERED state moments;
* :func:`state_feedback_probability` weights by the PREDICTED context
  probabilities and uses the feedback moments (state moments already inflated by
  the observation noise);
* the novel-context densities use each particle's STATIONARY state distribution
  with equal weights, because the novel slot has no filtered belief yet.

Every function here is read-only and consumes no randomness, with the single
documented exception of :func:`predictive_cue_p_value`, which draws its uniform
variate from ``model.rng`` when ``u`` is not supplied.
"""

from __future__ import annotations

import numpy as np

from .alignment import (
    clamp_active_summary_contexts,
    ensure_context_alignment,
    local_bias_distribution,
)
from .context import preview_cue_pmf
from .exceptions import BiasNotInferredError
from .numerics import (
    mixture_density_on_grid,
    stationary_state_cov_md,
    stationary_state_mean,
    stationary_state_mean_md,
    stationary_state_var,
)
# ``_must_be_scalar_model`` is the translation of the shared MATLAB private
# validator ``mustBeScalarModel``; imported rather than re-spelled so the error
# text of every scalar-only query stays single-sourced.
from .queries_core import (
    _must_be_scalar_model,
    preview_predictive_feedback,
    preview_predictive_feedback_md,
)
from .state import (
    feedback_transform,
    observation_noise_cov,
    peek_cue_label,
    process_noise_cov,
)
from .statics import normal_cdf, normal_pdf

__all__ = [
    "state_probability",
    "state_feedback_probability",
    "novel_state_probability",
    "novel_state_feedback_probability",
    "retention_given_context_probability",
    "drift_given_context_probability",
    "bias_given_context_probability",
    "bias_probability",
    "predictive_state_feedback_cdf",
    "predictive_cue_p_value",
]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _finite_grid(values, name):
    """Validate a density query grid and normalise its array form.

    Stands in for the ``values (:, :) double {mustBeFinite, mustBeReal}``
    argument validator every MATLAB density method carries. Only the
    finite/real check and the array conversion live here: the MD
    dimension check is applied downstream by
    :func:`realtimecoin.numerics.mixture_density_on_grid`, exactly as MATLAB
    applies it inside ``mixtureDensityOnGrid``.

    Parameters
    ----------
    values : array_like
        The caller's grid.
    name : str
        Query name, quoted in the error message.

    Returns
    -------
    numpy.ndarray
        ``values`` as a float array, unchanged in shape.

    Raises
    ------
    ValueError
        If any entry is non-finite or complex
        (``RealTimeCOIN:GridNotFinite``).
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "RealTimeCOIN:GridNotFinite: %s expects a finite, real grid." % name
        )
    return values


def _scalar_grid(values, name):
    """Validate a scalar-model grid and flatten it to ``(K,)``.

    The Python form of MATLAB's ``values = values(:)'`` reshape used by the
    scalar-only density queries (``bias_probability`` and the three
    ``*_given_context`` densities).

    Parameters
    ----------
    values : array_like
        The caller's grid.
    name : str
        Query name, quoted in the error message.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` query points.
    """
    return _finite_grid(values, name).reshape(-1)


def _novel_slot_particles(model):
    """Particles that still have a free novel-context slot.

    Mirrors the ``if obj.D.C(p) >= obj.max_contexts, continue; end`` guard the
    novel-context densities share: a particle that has instantiated its full
    budget has no novel slot to describe and contributes nothing - neither a
    mixture component nor a unit of the normaliser.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    rows : numpy.ndarray
        ``(used,)`` indices of the contributing particles.
    novel : numpy.ndarray
        ``(used,)`` 0-based novel-slot index of each of them (``n_active[p]``).
    """
    rows = np.flatnonzero(model.D.n_active < model.max_contexts)     # (used,)
    return rows, model.D.n_active[rows]                              # (used,)


def _novel_moments(model, rows, novel):
    """Stationary state moments of each particle's novel context slot.

    The same re-seeding the prediction step applies to a freshly adopted
    context: mean ``d / (1 - a)`` and variance ``Q / (1 - a^2)`` in the scalar
    case, the multivariate stationary moments (Lyapunov solve) otherwise.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    rows : numpy.ndarray
        ``(used,)`` particle indices, from :func:`_novel_slot_particles`.
    novel : numpy.ndarray
        ``(used,)`` novel-slot indices for those particles.

    Returns
    -------
    means : numpy.ndarray
        ``(used, 1)`` scalar means, or ``(used, 1, N)`` vectors. The trailing
        singleton context axis is what
        :func:`realtimecoin.numerics.mixture_density_on_grid` expects.
    covs : numpy.ndarray
        ``(used, 1)`` variances, or ``(used, 1, N, N)`` covariances.
    """
    d = model.D
    used = rows.size

    if model.state_dim == 1:
        means = stationary_state_mean(
            d.retention[rows, novel], d.drift[rows, novel]
        ).reshape(used, 1)                                            # (used, 1)
        covs = stationary_state_var(
            d.retention[rows, novel], model.sigma_process_noise
        ).reshape(used, 1)                                            # (used, 1)
        return means, covs

    n = model.state_dim
    q_cov = process_noise_cov(model)                                  # (N, N)
    means = np.zeros((used, 1, n))                                 # (used, 1, N)
    covs = np.zeros((used, 1, n, n))                            # (used, 1, N, N)
    for i, (p, c) in enumerate(zip(rows, novel)):
        a_matrix = d.Theta[p, c, :, :n]                               # (N, N)
        drift = d.Theta[p, c, :, n]                                   # (N,)
        means[i, 0] = stationary_state_mean_md(a_matrix, drift)
        covs[i, 0] = stationary_state_cov_md(a_matrix, q_cov)
    return means, covs


def _per_context_normal_map(model, values, moments):
    """Build the ``{global context: (K,) density}`` map of a scalar query.

    The shared tail of ``retention_given_context_probability``,
    ``drift_given_context_probability`` and
    ``bias_given_context_probability``: trigger the alignment, pull the two
    prototype moment vectors, clamp the active summary contexts to the aligned
    label set and evaluate one univariate normal per surviving context.

    Keys are the 0-BASED global context labels, matching the convention of the
    per-context densities in :mod:`realtimecoin.queries_aligned`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : numpy.ndarray
        ``(K,)`` validated query grid.
    moments : callable
        Maps the ``global_contexts`` prototype dict to the pair of
        ``(K_ctx,)`` per-global-context means and variances this query reads.

    Returns
    -------
    dict
        ``{0-based global context label: numpy.ndarray}``, each value ``(K,)``.
    """
    alignment = ensure_context_alignment(model)
    mean, var = moments(alignment["global_contexts"])
    active = clamp_active_summary_contexts(model, alignment)
    return {int(c): normal_pdf(values, mean[c], var[c]) for c in active}


# --------------------------------------------------------------------------
# Marginal state / feedback densities
# --------------------------------------------------------------------------


def state_probability(model, values):
    """Posterior latent-state density on a grid of query points.

    The posterior Gaussian mixture over particles and contexts::

        p(x) = (1 / P) sum_p sum_c W[p, c] N(x | m[p, c], V[p, c])

    using the RESPONSIBILITIES and the filtered (posterior) state moments.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` densities.
    """
    name = "state_probability"
    grid = _finite_grid(values, name)
    d = model.D
    covs = d.state_filtered_var if model.state_dim == 1 else d.state_filtered_cov
    return mixture_density_on_grid(
        grid,
        d.responsibilities,                 # (P, C) posterior weights
        d.state_filtered_mean,
        covs,
        model.num_particles,
        model.state_dim,
        name,
    )


def state_feedback_probability(model, values):
    """Predictive feedback (observation) density on a grid.

    The predictive Gaussian mixture using the PREDICTED context probabilities
    and the predictive feedback moments (state moments inflated by the
    observation noise).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` densities.
    """
    name = "state_feedback_probability"
    grid = _finite_grid(values, name)
    d = model.D
    covs = d.state_feedback_var if model.state_dim == 1 else d.state_feedback_cov
    return mixture_density_on_grid(
        grid,
        d.predicted_probabilities,          # (P, C) pre-observation weights
        d.state_feedback_mean,
        covs,
        model.num_particles,
        model.state_dim,
        name,
    )


def novel_state_probability(model, values):
    """Posterior state density of the novel (not yet instantiated) context.

    Each particle contributes its novel slot's stationary state distribution -
    the same re-seeding the prediction step uses: mean ``d / (1 - a)`` and
    variance ``Q / (1 - a^2)`` in the scalar case, the multivariate stationary
    moments otherwise. The result is the equal-weight mixture over the particles
    that still have an available novel slot; if every particle has saturated its
    context budget the density is all zeros.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` densities.
    """
    name = "novel_state_probability"
    grid = _finite_grid(values, name)
    rows, novel = _novel_slot_particles(model)
    means, covs = _novel_moments(model, rows, novel)
    # One equally weighted component per contributing particle; the normaliser
    # is `used`, not num_particles, so the mixture stays a proper density (and
    # is left as the zero density when nothing contributes).
    weights = np.ones((rows.size, 1))                              # (used, 1)
    return mixture_density_on_grid(
        grid, weights, means, covs, rows.size, model.state_dim, name
    )


def novel_state_feedback_probability(model, values):
    """Feedback density of the novel (not yet instantiated) context.

    Observation-space counterpart of :func:`novel_state_probability`: each
    particle's novel-slot stationary distribution shifted by that slot's sampled
    bias and inflated by the observation noise ``R``.

    .. note::
       As in ``novel_state_feedback_probability.m`` - and matching
       ``state_feedback_given_context_probability`` - the scalar branch inflates
       by ``sigma_sensory_noise ** 2`` ALONE, while the multi-dimensional branch
       uses the full :func:`realtimecoin.state.observation_noise_cov` (which also
       carries the motor-noise term). The two agree whenever
       ``sigma_motor_noise == 0`` (the default).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` densities.
    """
    name = "novel_state_feedback_probability"
    grid = _finite_grid(values, name)
    rows, novel = _novel_slot_particles(model)
    means, covs = _novel_moments(model, rows, novel)

    if model.state_dim == 1:
        bias = model.D.bias[rows, novel].reshape(-1, 1)            # (used, 1)
        noise = model.sigma_sensory_noise ** 2
    else:
        bias = model.D.bias[rows, novel, :].reshape(
            rows.size, 1, model.state_dim
        )                                                       # (used, 1, N)
        noise = observation_noise_cov(model)                       # (N, N)
    means, covs = feedback_transform(means, covs, bias, noise)

    weights = np.ones((rows.size, 1))                              # (used, 1)
    return mixture_density_on_grid(
        grid, weights, means, covs, rows.size, model.state_dim, name
    )


# --------------------------------------------------------------------------
# Scalar-only parameter densities
# --------------------------------------------------------------------------


def retention_given_context_probability(model, values):
    """Per-context retention density on a grid.

    Reads the aligned global-context dynamics moments (the retention entry of
    ``dynamics_mean`` / ``dynamics_covar``). Mirrors COIN's
    ``plot_retention_given_context``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` grid of query points.

    Returns
    -------
    dict
        ``{global_context_label: numpy.ndarray}``, each value ``(K,)``.

    Raises
    ------
    ScalarModelOnlyError
        If ``state_dim > 1``: retention is a scalar-dynamics quantity with no
        multi-dimensional counterpart (the MD model stores a dynamics matrix).
    """
    name = "retention_given_context_probability"
    _must_be_scalar_model(model, name)
    grid = _scalar_grid(values, name)
    # dynamics = [retention; drift]; entry 0 is the retention factor.
    return _per_context_normal_map(
        model,
        grid,
        lambda proto: (
            proto["dynamics_mean"][:, 0],                          # (K_ctx,)
            proto["dynamics_covar"][:, 0, 0],                      # (K_ctx,)
        ),
    )


def drift_given_context_probability(model, values):
    """Per-context drift density on a grid.

    Reads the drift entry of the aligned global-context dynamics moments.
    Mirrors COIN's ``plot_drift_given_context``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` grid of query points.

    Returns
    -------
    dict
        ``{global_context_label: numpy.ndarray}``, each value ``(K,)``.

    Raises
    ------
    ScalarModelOnlyError
        If ``state_dim > 1``.
    """
    name = "drift_given_context_probability"
    _must_be_scalar_model(model, name)
    grid = _scalar_grid(values, name)
    # dynamics = [retention; drift]; entry 1 is the drift.
    return _per_context_normal_map(
        model,
        grid,
        lambda proto: (
            proto["dynamics_mean"][:, 1],                          # (K_ctx,)
            proto["dynamics_covar"][:, 1, 1],                      # (K_ctx,)
        ),
    )


def bias_given_context_probability(model, values):
    """Per-context measurement-bias density on a grid.

    Reads the aligned global-context bias moments. Mirrors COIN's
    ``plot_bias_given_context``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` grid of query points.

    Returns
    -------
    dict
        ``{global_context_label: numpy.ndarray}``, each value ``(K,)``.

    Raises
    ------
    BiasNotInferredError
        If ``infer_bias`` is false.
    ScalarModelOnlyError
        If ``state_dim > 1``: the MD prototypes track a bias mean but no bias
        covariance.
    """
    name = "bias_given_context_probability"
    _must_be_scalar_model(model, name)
    if not model.infer_bias:
        raise BiasNotInferredError("%s requires infer_bias == True." % name)
    grid = _scalar_grid(values, name)
    return _per_context_normal_map(
        model,
        grid,
        lambda proto: (
            proto["bias_mean"],                                    # (K_ctx,)
            proto["bias_var"],                                     # (K_ctx,)
        ),
    )


def bias_probability(model, values):
    """Marginal (across-context) measurement-bias density on a grid.

    Marginalised over contexts and particles with the predicted context
    probabilities as mixing weights::

        p(b) = (1 / P) sum_p sum_c W[p, c] N(b | bias_mean[p, c], bias_var[p, c])

    Mirrors COIN's ``plot_bias`` / ``compute_marginal_distribution``. Unlike the
    state densities this reads the per-particle bias BELIEF through
    :func:`realtimecoin.alignment.local_bias_distribution` (posterior sufficient
    statistics when they exist, the prior otherwise), so it cannot go through
    ``mixture_density_on_grid``, which takes ready-made moment arrays.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    values : array_like
        ``(K,)`` grid of query points.

    Returns
    -------
    numpy.ndarray
        ``(K,)`` densities.

    Raises
    ------
    BiasNotInferredError
        If ``infer_bias`` is false.
    ScalarModelOnlyError
        If ``state_dim > 1``.
    """
    name = "bias_probability"
    _must_be_scalar_model(model, name)
    if not model.infer_bias:
        raise BiasNotInferredError("%s requires infer_bias == True." % name)
    grid = _scalar_grid(values, name)

    weights = model.D.predicted_probabilities                        # (P, C)
    densities = np.zeros(grid.shape)                                 # (K,)
    for p in range(model.num_particles):
        for c in range(model.max_contexts + 1):
            w = weights[p, c]
            if w > 0:
                mu, var = local_bias_distribution(model, c, p)
                densities = densities + w * normal_pdf(grid, mu, var)
    return densities / model.num_particles


# --------------------------------------------------------------------------
# Predictive CDF and cue p-value
# --------------------------------------------------------------------------


def predictive_state_feedback_cdf(model, y, q=None):
    """Predictive CDF of the next feedback evaluated at ``y``.

    For the scalar model returns the scalar predictive probability
    ``P(Y <= y)`` given the optional upcoming cue. For the multi-dimensional
    model ``y`` is an ``(N,)`` vector and the return is the ``(N,)`` vector of
    MARGINAL predictive CDFs ``p_j = P(Y_j <= y_j)``, the standard per-dimension
    probability-integral transform used for calibration. Each marginal of the
    predictive Gaussian mixture is itself a 1-D Gaussian mixture, so this reuses
    the scalar normal CDF and reduces exactly to the scalar result at ``N == 1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    y : float or array_like
        Query point: a scalar, or an ``(N,)`` vector.
    q : float or None, optional
        Raw cue value of the upcoming trial; ``None`` uses the pending cue.

    Returns
    -------
    float or numpy.ndarray
        Scalar CDF value, or an ``(N,)`` vector of marginal CDF values.

    Raises
    ------
    ValueError
        If ``y`` does not have ``state_dim`` elements
        (``RealTimeCOIN:FeedbackDimensionMismatch``).
    """
    # The preview helpers take a cue LABEL and never consult pending_q, so the
    # pending raw cue is resolved here (predictive_state_feedback_cdf.m:18-21).
    if q is None:
        q = model.pending_q
    q_label = peek_cue_label(model, q)          # read-only: registers nothing

    n = model.state_dim
    y = np.asarray(y, dtype=float).reshape(-1)                       # (N,)
    if y.size != n:
        raise ValueError(
            "RealTimeCOIN:FeedbackDimensionMismatch: predictive_state_feedback_cdf "
            "expects y to have %d elements for state_dim == %d; received %d."
            % (n, n, y.size)
        )

    if n == 1:
        w, m, v = preview_predictive_feedback(model, q_label)        # (P, C) x3
        p = float(np.sum(w * normal_cdf(y[0], m, v)) / model.num_particles)
        # fmin/fmax, not clip: MATLAB's min(max(p, 0), 1) maps a NaN to 0.
        return float(np.fmin(np.fmax(p, 0.0), 1.0))

    w, m, cov = preview_predictive_feedback_md(model, q_label)
    #                              w (P, C), m (P, C, N), cov (P, C, N, N)
    p = np.zeros(n)                                                  # (N,)
    for j in range(n):
        # Marginal j of each component: mean m[:, :, j], variance cov[:, :, j, j].
        p[j] = np.sum(
            w * normal_cdf(y[j], m[:, :, j], cov[:, :, j, j])
        ) / model.num_particles
    return np.fmin(np.fmax(p, 0.0), 1.0)


def predictive_cue_p_value(model, q, u=None):
    """Randomised predictive p-value for a cue label.

    The randomised probability-integral transform of ``q`` under the current
    predictive cue pmf::

        p = F(q-) + u * f(q)

    where ``f`` is the predictive pmf over cue labels and ``F(q-)`` the
    cumulative mass of the labels below ``q``. The ``u * f(q)`` term spreads the
    atom into a continuous ``[F(q-), F(q)]``, so under a correct model ``p`` is
    uniform.

    A cue value never observed is treated as the next novel label (see
    :func:`realtimecoin.state.peek_cue_label`), WITHOUT registering it.

    .. note::
       The stub docstring this replaced - and the facade docstring on
       :meth:`realtimecoin.model.RealTimeCOIN.predictive_cue_p_value` - describe
       that novel label as carrying "zero mass", so that its p-value reduces to
       ``F(q-)``. That is not what either implementation does. ``D.local_cue_matrix``
       always carries ONE MORE column than there are registered cue values (the
       novel-cue stick of the hierarchical Dirichlet process), and
       ``peek_cue_label`` returns exactly that trailing column for an unseen
       value, so the novel label carries the novel-cue mass. The zero-mass path
       is only the defensive padding below, which MATLAB spells the same way
       (``predictive_cue_p_value.m:28-30``) and which is likewise unreachable
       through the public API. The behaviour here is a faithful translation of
       the MATLAB source; only the prose was wrong.

    .. note::
       The uniform variate is drawn BEFORE the ``q is None`` short-circuit,
       mirroring MATLAB, where ``u (1, 1) double = rand`` is evaluated in the
       ``arguments`` block before the body runs. The ``rng`` stream therefore
       advances by one draw even on the ``nan`` return.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read; the cue registry is NOT mutated.
    q : float or None
        Raw cue value. ``None`` returns ``nan`` (the p-value is undefined).
    u : float or None, optional
        Uniform variate in [0, 1]. ``None`` draws one from ``model.rng``, which
        makes the call non-reproducible - pass ``u`` explicitly in tests.

    Returns
    -------
    float
        The p-value in [0, 1], or ``nan``.
    """
    if u is None:
        u = float(model.rng.random())
    q_label = peek_cue_label(model, q)          # read-only: registers nothing
    if q_label is None:
        return float("nan")

    pmf, labels = preview_cue_pmf(model)                             # (Q,) x2
    if q_label >= pmf.size:
        # Defensive padding for a label past the last pmf entry (zero mass, so
        # p collapses to F(q-)). Unreachable through the public API - the pmf
        # always spans the novel-cue column - but MATLAB spells it out too.
        pmf = np.concatenate([pmf, np.zeros(q_label + 1 - pmf.size)])
        labels = np.arange(pmf.size)

    f = float(pmf[q_label])
    f_minus = float(np.sum(pmf[labels < q_label]))
    return float(np.fmin(np.fmax(f_minus + u * f, 0.0), 1.0))

"""Marginal density and CDF queries (no context alignment needed).

STUB MODULE - implemented by unit C2.

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
"""

from __future__ import annotations

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

_UNIT = "implemented by unit C2"


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


def novel_state_feedback_probability(model, values):
    """Feedback density of the novel (not yet instantiated) context.

    Observation-space counterpart of :func:`novel_state_probability`: each
    particle's novel-slot stationary distribution shifted by that slot's sampled
    bias and inflated by the observation noise ``R``.

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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


def bias_probability(model, values):
    """Marginal (across-context) measurement-bias density on a grid.

    Marginalised over contexts and particles with the predicted context
    probabilities as mixing weights::

        p(b) = (1 / P) sum_p sum_c W[p, c] N(b | bias_mean[p, c], bias_var[p, c])

    Mirrors COIN's ``plot_bias`` / ``compute_marginal_distribution``.

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
    raise NotImplementedError(_UNIT)


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
    """
    raise NotImplementedError(_UNIT)


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
    :func:`realtimecoin.state.peek_cue_label`), which carries zero mass, so its
    p-value reduces to ``F(q-)``.

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
    raise NotImplementedError(_UNIT)

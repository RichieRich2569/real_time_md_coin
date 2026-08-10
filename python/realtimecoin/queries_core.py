"""Point-prediction and moment queries, plus the fast local-label summaries.

Translated from the public methods ``motor_output``,
``predictive_motor_output``, ``state_moments``, ``predictive_feedback_moments``,
``explicit_component``, ``implicit_component``, ``state_cstar1/2/3``,
``predicted_probability_cstar1/3``, ``kalman_gain_cstar1/2``,
``predicted_context_probabilities_local``, ``context_responsibilities_local``
and ``sampled_context_count_local``, together with the private helpers
``previewPredictiveFeedback[MD]``, ``selectContextStateMean`` and
``scalarKalmanGains``.

Every function here is READ-ONLY: it draws no random numbers and does not
mutate particle state (and, unlike the aligned queries, does not touch the
alignment cache either). All of them work for both the scalar and the
multi-dimensional model unless the docstring says otherwise; the Kalman-gain
queries are scalar-only, because the MD gain is a matrix with no scalar
counterpart.

Two ``q`` conventions coexist here, inherited from the MATLAB sources and
deliberately preserved:

* ``predictive_motor_output``, ``state_cstar2`` and ``kalman_gain_cstar2`` take
  a RAW cue value and resolve it through
  :func:`realtimecoin.state.peek_cue_label`;
* ``predictive_feedback_moments``, :func:`preview_predictive_feedback` and
  :func:`preview_predictive_feedback_md` take a 0-based cue LABEL, which they
  use to index ``D.local_cue_matrix`` directly (clamped to the last column).

In BOTH conventions ``None`` marginalises the cue out at the point of use; the
raw-value entry points additionally substitute ``model.pending_q`` first, so a
cue staged by ``observe_q`` is honoured.
"""

from __future__ import annotations

import numpy as np

from .context import next_trial_context_weights
from .exceptions import ScalarModelOnlyError
from .numerics import (
    stationary_state_cov_md,
    stationary_state_mean,
    stationary_state_mean_md,
    stationary_state_var,
)
from .state import (
    EPS,
    observation_noise_cov,
    observation_variance,
    peek_cue_label,
    process_noise_cov,
)

__all__ = [
    "motor_output",
    "predictive_motor_output",
    "state_moments",
    "predictive_feedback_moments",
    "explicit_component",
    "implicit_component",
    "state_cstar1",
    "state_cstar2",
    "state_cstar3",
    "predicted_probability_cstar1",
    "predicted_probability_cstar3",
    "kalman_gain_cstar1",
    "kalman_gain_cstar2",
    "predicted_context_probabilities_local",
    "context_responsibilities_local",
    "sampled_context_count_local",
    "preview_predictive_feedback",
    "preview_predictive_feedback_md",
    "scalar_kalman_gains",
]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _must_be_scalar_model(model, method_name):
    """Raise unless the model is scalar (``state_dim == 1``).

    Translation of ``private/mustBeScalarModel.m``. The message keeps the
    MATLAB wording so a caller matching on the identifier or on the text sees
    the same string.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to check.
    method_name : str
        Name reported in the error message.

    Returns
    -------
    None

    Raises
    ------
    ScalarModelOnlyError
        If ``model.state_dim != 1``.
    """
    if model.state_dim != 1:
        raise ScalarModelOnlyError(
            "%s is only defined for the scalar model (state_dim == 1); "
            "state_dim == %d." % (method_name, model.state_dim)
        )


def _resolve_raw_cue(model, q):
    """0-based cue label for a RAW cue value, defaulting to the pending cue.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``pending_q`` and cue registry are read (never mutated).
    q : float or None
        Raw cue value; ``None`` falls back to ``model.pending_q``.

    Returns
    -------
    int or None
        0-based cue label, or ``None`` when there is no cue to resolve.
    """
    if q is None:
        q = model.pending_q
    return peek_cue_label(model, q)


def _drop_zero_weight(weights, terms):
    """Zero out mixture terms whose weight is exactly zero.

    The multi-dimensional reductions in ``motor_output.m``, ``state_moments.m``
    and ``predictive_feedback_moments.m`` all guard their accumulation with
    ``if w ~= 0 ... end``. That guard is not cosmetic: an unused context slot can
    legitimately carry a non-finite mean or covariance, and ``0 * inf`` would
    poison the sum. The vectorised form reproduces it by masking rather than by
    multiplying.

    Parameters
    ----------
    weights : numpy.ndarray
        ``(P, C)`` mixture weights.
    terms : numpy.ndarray
        Array whose two leading axes are ``(P, C)``.

    Returns
    -------
    numpy.ndarray
        ``terms`` with every zero-weight component replaced by zero.
    """
    keep = weights != 0.0                                               # (P, C)
    return np.where(
        keep.reshape(keep.shape + (1,) * (terms.ndim - keep.ndim)), terms, 0.0
    )


def _select_context_state_mean(model, idx):
    """Particle-average of ``D.state_mean`` at a per-particle context choice.

    Translation of ``private/selectContextStateMean.m``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    idx : array_like
        ``(P,)`` 0-based context label selected for each particle.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    d = model.D
    particles = np.arange(model.num_particles)
    idx = np.asarray(idx).reshape(-1)
    if model.state_dim == 1:
        return float(np.mean(d.state_mean[particles, idx]))
    # (P, N) gather, then average across particles.
    return np.mean(d.state_mean[particles, idx, :], axis=0)              # (N,)


def _weights_for_label(model, q_label):
    """Next-trial context weights, optionally tilted by a cue LABEL.

    Thin alias for :func:`realtimecoin.context.next_trial_context_weights`; kept
    so the preview helpers read like their MATLAB counterparts, which spell the
    same expression out inline.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q_label : int or None
        0-based cue label, clamped to the last instantiated column; ``None``
        marginalises the cue out.

    Returns
    -------
    numpy.ndarray
        ``(P, C)`` normalised context weights.
    """
    return next_trial_context_weights(model, q_label)


# --------------------------------------------------------------------------
# Point predictions and moments
# --------------------------------------------------------------------------


def motor_output(model):
    """Expected state feedback, marginalised over contexts and particles.

    The mixture mean ``u = (sum_{p,c} w[p, c] * m[p, c]) / num_particles`` with
    ``w = D.predicted_probabilities`` and ``m = D.state_feedback_mean``.

    Note this uses the PREDICTED (pre-observation) context weights, matching
    COIN's motor-output definition, not the posterior responsibilities.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    d = model.D
    w = d.predicted_probabilities                                       # (P, C)
    if model.state_dim == 1:
        return float(np.sum(w * d.state_feedback_mean) / model.num_particles)
    # (P, C, 1) * (P, C, N) summed over particles and contexts, with the
    # zero-weight components dropped exactly as the MATLAB loop drops them.
    means = _drop_zero_weight(w, d.state_feedback_mean)               # (P, C, N)
    total = np.sum(w[:, :, None] * means, axis=(0, 1))                  # (N,)
    return total / model.num_particles


def predictive_motor_output(model, q=None):
    """Expected next observation given the upcoming cue.

    A read-only one-step prediction from the current posterior, so it may be
    called between ``observe_q(q)`` and ``observe_y(y)``. Equal to the mean
    returned by :func:`predictive_feedback_moments`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : float or None, optional
        RAW cue value of the upcoming trial (resolved through
        :func:`realtimecoin.state.peek_cue_label`, so an unseen value is treated
        as the next novel label without registering it). ``None`` (the default)
        uses the cue staged by ``observe_q``.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    q_label = _resolve_raw_cue(model, q)
    if model.state_dim > 1:
        mu, _sigma = predictive_feedback_moments(model, q_label)
        return mu
    w, m, _v = preview_predictive_feedback(model, q_label)
    return float(np.sum(w * m) / model.num_particles)


def state_moments(model):
    """Predictive latent-state mean and (co)variance.

    Marginalised over contexts and particles with the predicted context
    probabilities, using the Gaussian-mixture moments::

        E[s]   = sum_k w_k m_k
        Cov[s] = sum_k w_k (V_k + m_k m_k') - E[s] E[s]'

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    mu : float or numpy.ndarray
        Scalar mean, or an ``(N,)`` mean vector.
    v : float or numpy.ndarray
        Scalar variance, or an ``(N, N)`` covariance matrix.
    """
    d = model.D
    p_count = model.num_particles
    w = d.predicted_probabilities                                       # (P, C)

    if model.state_dim == 1:
        mu = float(np.sum(w * d.state_mean) / p_count)
        second = float(
            np.sum(w * (d.state_var + d.state_mean ** 2)) / p_count
        )
        # fmax, not maximum: MATLAB's max(x, 0) drops a NaN in favour of 0.
        return mu, float(np.fmax(second - mu ** 2, 0.0))

    # state_moments.m:38 divides by P BEFORE its `if w == 0, continue` test, so
    # the mask is built from the scaled weights (motor_output.m:34 tests the
    # unscaled weight - the difference is unreachable, but it is transcribed).
    weights = w / p_count                                               # (P, C)
    means = _drop_zero_weight(weights, d.state_mean)                 # (P, C, N)
    covs = _drop_zero_weight(weights, d.state_cov)                # (P, C, N, N)
    mu = np.sum(weights[:, :, None] * means, axis=(0, 1))               # (N,)
    # Per-component second moment V_k + m_k m_k', weighted and summed.
    outer = means[:, :, :, None] * means[:, :, None, :]                 # (P,C,N,N)
    second = np.sum(
        weights[:, :, None, None] * (covs + outer), axis=(0, 1)
    )                                                                   # (N, N)
    v = second - np.outer(mu, mu)                                       # (N, N)
    return mu, (v + v.T) / 2.0


def predictive_feedback_moments(model, q=None):
    """One-step predictive observation moments.

    Mean and covariance of the NEXT observation given the optional upcoming
    cue, marginalised over contexts and particles. Read-only, so it can be
    called between ``observe_q(q)`` and ``observe_y(y)`` to obtain the model's
    belief about the imminent feedback.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : int or None, optional
        0-based cue LABEL of the upcoming trial - NOT a raw cue value: this
        function indexes ``D.local_cue_matrix`` directly (clamped to the last
        column), matching the MATLAB source. ``None`` marginalises the cue out.

    Returns
    -------
    mu : float or numpy.ndarray
        Scalar mean, or an ``(N,)`` mean vector.
    sigma : float or numpy.ndarray
        Scalar variance, or an ``(N, N)`` covariance matrix.
    """
    weights = _weights_for_label(model, q)                              # (P, C)
    if model.state_dim == 1:
        return _scalar_moments(model, weights)
    return _multi_moments(model, weights)


def _scalar_moments(model, weights):
    """One-step predictive feedback moments for the scalar model.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    weights : numpy.ndarray
        ``(P, C)`` context mixture weights.

    Returns
    -------
    mu : float
        Predictive feedback mean.
    v : float
        Predictive feedback variance.

    Notes
    -----
    The novel-slot fallback is written exactly as ``predictive_feedback_moments.m``
    writes it - ``d / max(1 - a, eps)`` and ``sigma^2 / max(1 - a^2, eps)`` -
    rather than through :func:`realtimecoin.numerics.stationary_state_mean`,
    which returns 0 instead of a huge value when ``a`` is within ``eps`` of 1.
    The two agree everywhere except in that (unreachable for a truncated
    ``a in [0, 1 - eps]``) corner; keeping the MATLAB form makes this function a
    literal transliteration.
    """
    d = model.D
    p_count = model.num_particles
    state_mean = d.retention * d.state_filtered_mean + d.drift          # (P, C)
    state_var = (
        d.retention ** 2 * d.state_filtered_var + model.sigma_process_noise ** 2
    )                                                                   # (P, C)

    rows = np.nonzero(d.n_active < model.max_contexts)[0]
    if rows.size:
        cols = d.n_active[rows]
        a = d.retention[rows, cols]
        drift = d.drift[rows, cols]
        state_mean[rows, cols] = drift / np.fmax(1.0 - a, EPS)
        state_var[rows, cols] = model.sigma_process_noise ** 2 / np.fmax(
            1.0 - a ** 2, EPS
        )

    feedback_mean = state_mean + d.bias                                 # (P, C)
    feedback_var = state_var + observation_variance(model)              # (P, C)

    mu = float(np.sum(weights * feedback_mean) / p_count)
    second = float(
        np.sum(weights * (feedback_var + feedback_mean ** 2)) / p_count
    )
    return mu, float(np.fmax(second - mu ** 2, 0.0))


def _multi_moments(model, weights):
    """One-step predictive feedback moments for the multi-dimensional model.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    weights : numpy.ndarray
        ``(P, C)`` context mixture weights.

    Returns
    -------
    mu : numpy.ndarray
        ``(N,)`` predictive feedback mean.
    sigma : numpy.ndarray
        ``(N, N)`` predictive feedback covariance.
    """
    p_count = model.num_particles
    w, m, cov = preview_predictive_feedback_md(model, None, weights=weights)
    # As in state_moments: predictive_feedback_moments.m:86 tests w / P == 0.
    scaled = w / p_count                                                # (P, C)
    m = _drop_zero_weight(scaled, m)                                 # (P, C, N)
    cov = _drop_zero_weight(scaled, cov)                          # (P, C, N, N)
    mu = np.sum(scaled[:, :, None] * m, axis=(0, 1))                    # (N,)
    outer = m[:, :, :, None] * m[:, :, None, :]                      # (P, C, N, N)
    second = np.sum(
        scaled[:, :, None, None] * (cov + outer), axis=(0, 1)
    )                                                                   # (N, N)
    sigma = second - np.outer(mu, mu)                                    # (N, N)
    return mu, (sigma + sigma.T) / 2.0


def explicit_component(model):
    """Explicit component of adaptation.

    The predictive latent-state mean of the highest-responsibility context,
    averaged over particles (COIN's ``plot_explicit_component``). On the first
    trial, when responsibilities are not yet informative, context 0 is used.
    This is the ``c*1`` state; see :func:`state_cstar1`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    if model.trial <= 1:
        # Before/at the first feedback the responsibilities are uninformative,
        # so their argmax is arbitrary; COIN falls back to context 0.
        idx = np.zeros(model.num_particles, dtype=int)                  # (P,)
    else:
        idx = np.argmax(model.D.responsibilities, axis=-1)              # (P,)
    return _select_context_state_mean(model, idx)


def implicit_component(model):
    """Implicit component of adaptation.

    The motor output minus the average predicted state, i.e. the part of the
    motor output attributable to the across-context marginal bias (COIN's
    ``plot_implicit_component`` / ``plot_average_bias``).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    mu, _v = state_moments(model)
    return motor_output(model) - mu


def state_cstar1(model):
    """Expected latent state of the highest-responsibility context (``c*1``).

    For each particle the context maximising the current responsibilities is
    selected and its state mean read off; the selection is then averaged across
    particles. Mirrors COIN's ``plot_state_given_cstar1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    idx = np.argmax(model.D.responsibilities, axis=-1)                  # (P,)
    return _select_context_state_mean(model, idx)


def state_cstar2(model, q=None):
    """Expected state of the highest NEXT-trial predicted-prob context (``c*2``).

    Selects the argmax of the next trial's predicted probabilities (conditioned
    on the upcoming cue) but reads the state mean of the CURRENT trial, exactly
    as COIN's ``plot_state_given_cstar2``. The one-step look-ahead is computed
    from the sampled transition and cue matrices without mutating model state.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : float or None, optional
        Raw cue value of the upcoming trial; ``None`` uses the pending cue.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    weights = _weights_for_label(model, _resolve_raw_cue(model, q))     # (P, C)
    idx = np.argmax(weights, axis=-1)                                   # (P,)
    return _select_context_state_mean(model, idx)


def state_cstar3(model):
    """Expected latent state of the highest predicted-probability context.

    ``c*3`` selects the argmax of the CURRENT trial's predicted probabilities,
    which is the only difference from :func:`state_cstar2`. Mirrors COIN's
    ``plot_state_given_cstar3``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float or numpy.ndarray
        A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
    """
    idx = np.argmax(model.D.predicted_probabilities, axis=-1)           # (P,)
    return _select_context_state_mean(model, idx)


def predicted_probability_cstar1(model):
    """Predicted probability of the highest-responsibility context.

    Selects, per particle, the context with the highest responsibility and reads
    off its predicted probability, then averages over particles. Mirrors COIN's
    ``plot_predicted_probability_cstar1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float
        A scalar in [0, 1], for both the scalar and MD models (context
        probabilities are dimension-independent).
    """
    d = model.D
    particles = np.arange(model.num_particles)
    idx = np.argmax(d.responsibilities, axis=-1)                        # (P,)
    return float(np.mean(d.predicted_probabilities[particles, idx]))


def predicted_probability_cstar3(model):
    """Highest predicted context probability on the current trial.

    The particle-average of the maximum predicted context probability. Mirrors
    COIN's ``plot_predicted_probability_cstar3``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float
        A scalar in [0, 1].
    """
    return float(np.mean(np.max(model.D.predicted_probabilities, axis=-1)))


def kalman_gain_cstar1(model):
    """Kalman gain of the highest-responsibility context.

    Selects, per particle, the context with the highest responsibility and reads
    off its scalar Kalman gain ``state_var / state_feedback_var``, then averages
    over particles. Mirrors COIN's ``plot_Kalman_gain_given_cstar1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float
        The particle-averaged gain.

    Raises
    ------
    ScalarModelOnlyError
        If ``state_dim > 1``: the MD Kalman gain is a matrix and has no scalar
        counterpart.
    """
    _must_be_scalar_model(model, "kalman_gain_cstar1")
    gains = scalar_kalman_gains(model)                                  # (P, C)
    particles = np.arange(model.num_particles)
    idx = np.argmax(model.D.responsibilities, axis=-1)                  # (P,)
    return float(np.mean(gains[particles, idx]))


def kalman_gain_cstar2(model, q=None):
    """Kalman gain of the highest NEXT-trial predicted-prob context.

    The selector uses the one-step-ahead prediction (given the optional upcoming
    cue) while the gain is that of the current trial. Mirrors COIN's
    ``plot_Kalman_gain_given_cstar2``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : float or None, optional
        Raw cue value of the upcoming trial; ``None`` uses the pending cue.

    Returns
    -------
    float
        The particle-averaged gain.

    Raises
    ------
    ScalarModelOnlyError
        If ``state_dim > 1``.
    """
    _must_be_scalar_model(model, "kalman_gain_cstar2")
    weights = _weights_for_label(model, _resolve_raw_cue(model, q))     # (P, C)
    gains = scalar_kalman_gains(model)                                  # (P, C)
    particles = np.arange(model.num_particles)
    idx = np.argmax(weights, axis=-1)                                   # (P,)
    return float(np.mean(gains[particles, idx]))


# --------------------------------------------------------------------------
# Local-label (unaligned) context summaries
# --------------------------------------------------------------------------


def predicted_context_probabilities_local(model):
    """Fast local-label predicted context probabilities.

    A row vector in the MODAL particles' local label frame; it deliberately
    avoids the global relabelling, so it is cheap enough for live plots and
    logging but its entries are NOT comparable across trials.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` weights summing to one, or all zeros.
    """
    from .context import local_context_probability_vector

    return local_context_probability_vector(model, "predicted")


def context_responsibilities_local(model):
    """Fast local-label posterior context weights.

    Local-frame counterpart of the aligned ``responsibilities_vector``; see
    :func:`predicted_context_probabilities_local`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` weights summing to one, or all zeros.
    """
    from .context import local_context_probability_vector

    return local_context_probability_vector(model, "responsibilities")


def sampled_context_count_local(model):
    """Fast local-label sampled-context occupancy.

    Local-frame counterpart of the aligned ``sampled_context_count``; see
    :func:`predicted_context_probabilities_local`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` frequencies summing to one, or all zeros.
    """
    from .context import local_context_probability_vector

    return local_context_probability_vector(model, "count")


# --------------------------------------------------------------------------
# One-step-ahead previews
# --------------------------------------------------------------------------


def preview_predictive_feedback(model, q=None):
    """One-step-ahead predictive feedback mixture (scalar model, read-only).

    Returns the mixture components of the next observation without mutating
    model state: the next-trial context weights together with the corresponding
    per-context feedback means and variances.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : int or None, optional
        0-based cue LABEL of the upcoming trial; ``None`` (the default)
        marginalises the cue out. Matching ``previewPredictiveFeedback.m``, this
        helper does NOT fall back to ``model.pending_q`` - its callers
        (``predictive_motor_output``, the predictive CDF) resolve the pending
        raw cue into a label themselves before calling.

    Returns
    -------
    w : numpy.ndarray
        ``(P, C)`` next-trial context weights.
    m : numpy.ndarray
        ``(P, C)`` per-component predictive feedback means.
    v : numpy.ndarray
        ``(P, C)`` per-component predictive feedback variances.
    """
    d = model.D
    w = _weights_for_label(model, q)                                    # (P, C)

    state_mean = d.retention * d.state_filtered_mean + d.drift          # (P, C)
    state_var = (
        d.retention ** 2 * d.state_filtered_var + model.sigma_process_noise ** 2
    )                                                                   # (P, C)
    # The novel slot has no filtered state yet: predict it from the stationary
    # distribution of its AR(1) dynamics. Particles at the cap have no novel
    # slot and are skipped.
    rows = np.nonzero(d.n_active < model.max_contexts)[0]
    if rows.size:
        cols = d.n_active[rows]
        state_mean[rows, cols] = stationary_state_mean(
            d.retention[rows, cols], d.drift[rows, cols]
        )
        state_var[rows, cols] = stationary_state_var(
            d.retention[rows, cols], model.sigma_process_noise
        )

    m = state_mean + d.bias                                             # (P, C)
    v = state_var + observation_variance(model)                         # (P, C)
    return w, m, v


def preview_predictive_feedback_md(model, q=None, *, weights=None):
    """One-step-ahead predictive feedback mixture (MD model, read-only).

    Multi-dimensional counterpart of :func:`preview_predictive_feedback`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : int or None, optional
        0-based cue LABEL of the upcoming trial; ``None`` marginalises the cue
        out. As in the scalar helper, no fallback to ``model.pending_q``.
    weights : numpy.ndarray or None, optional
        Pre-computed ``(P, C)`` context weights, used instead of deriving them
        from ``q``. Internal shortcut for :func:`predictive_feedback_moments`,
        which has already built them; not part of the MATLAB signature.

    Returns
    -------
    w : numpy.ndarray
        ``(P, C)`` next-trial context weights.
    m : numpy.ndarray
        ``(P, C, N)`` per-component predictive feedback means.
    cov : numpy.ndarray
        ``(P, C, N, N)`` per-component predictive feedback covariances.
    """
    d = model.D
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    q_cov = process_noise_cov(model)                                    # (N, N)
    r_cov = observation_noise_cov(model)                                # (N, N)

    w = _weights_for_label(model, q) if weights is None else weights    # (P, C)

    m = np.zeros((p_count, c_slots, n))                              # (P, C, N)
    cov = np.zeros((p_count, c_slots, n, n))                      # (P, C, N, N)
    for p in range(p_count):
        # Novel slot (0-based); at the cap there is none, hence the guard below.
        novel = min(int(d.n_active[p]), model.max_contexts)
        for c in range(c_slots):
            a_matrix = d.Theta[p, c, :, :n]                             # (N, N)
            drift = d.Theta[p, c, :, n]                                 # (N,)
            if c == novel and d.n_active[p] < model.max_contexts:
                s_pred = stationary_state_mean_md(a_matrix, drift)      # (N,)
                p_pred = stationary_state_cov_md(a_matrix, q_cov)       # (N, N)
            else:
                s_pred = a_matrix @ d.state_filtered_mean[p, c] + drift
                p_pred = (
                    a_matrix @ d.state_filtered_cov[p, c] @ a_matrix.T + q_cov
                )
            m[p, c] = s_pred + d.bias[p, c]
            cov[p, c] = p_pred + r_cov
    return w, m, cov


def scalar_kalman_gains(model):
    """Per-context scalar Kalman gains.

    ``gain = state_var / state_feedback_var`` for every context and particle,
    the quantity the ``kalman_gain_cstar*`` queries select from.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        ``(P, C)`` Kalman gains.

    Raises
    ------
    ScalarModelOnlyError
        If ``state_dim > 1``.
    """
    _must_be_scalar_model(model, "scalar_kalman_gains")
    d = model.D
    # Literal transliteration of `state_var ./ state_feedback_var`: MATLAB
    # yields Inf/NaN on a zero denominator rather than guarding, and the callers
    # average the result, so no guard is added here either.
    with np.errstate(divide="ignore", invalid="ignore"):
        return d.state_var / d.state_feedback_var                       # (P, C)

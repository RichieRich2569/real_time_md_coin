"""Point-prediction and moment queries, plus the fast local-label summaries.

STUB MODULE - implemented by unit B1.

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
"""

from __future__ import annotations

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

_UNIT = "implemented by unit B1"


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)


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
        0-based cue label of the upcoming trial; ``None`` uses the pending cue.

    Returns
    -------
    w : numpy.ndarray
        ``(P, C)`` next-trial context weights.
    m : numpy.ndarray
        ``(P, C)`` per-component predictive feedback means.
    v : numpy.ndarray
        ``(P, C)`` per-component predictive feedback variances.
    """
    raise NotImplementedError(_UNIT)


def preview_predictive_feedback_md(model, q=None):
    """One-step-ahead predictive feedback mixture (MD model, read-only).

    Multi-dimensional counterpart of :func:`preview_predictive_feedback`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    q : int or None, optional
        0-based cue label of the upcoming trial; ``None`` uses the pending cue.

    Returns
    -------
    w : numpy.ndarray
        ``(P, C)`` next-trial context weights.
    m : numpy.ndarray
        ``(P, C, N)`` per-component predictive feedback means.
    cov : numpy.ndarray
        ``(P, C, N, N)`` per-component predictive feedback covariances.
    """
    raise NotImplementedError(_UNIT)


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
    raise NotImplementedError(_UNIT)

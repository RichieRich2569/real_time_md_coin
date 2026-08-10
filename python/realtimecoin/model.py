"""The :class:`RealTimeCOIN` facade: state machine, validation and delegation.

Python translation of the MATLAB class folder ``@RealTimeCOIN``. The MATLAB
implementation puts each public method in its own file inside the class folder;
here the class is a thin facade whose methods are one-line delegates to the
module that owns the algorithm, so the file layout still maps one-to-one onto
the MATLAB one:

======================================= ====================================
module                                  MATLAB counterpart
======================================= ====================================
:mod:`realtimecoin.context`             the dimension-agnostic context steps
:mod:`realtimecoin.pipeline_scalar`     the ``state_dim == 1`` pipeline
:mod:`realtimecoin.pipeline_md`         the ``*MD`` pipeline
:mod:`realtimecoin.queries_core`        point predictions, moments, ``c*`` traces
:mod:`realtimecoin.queries_density`     densities and CDFs
:mod:`realtimecoin.queries_aligned`     context summaries needing alignment
:mod:`realtimecoin.alignment`           the cross-particle context alignment
:mod:`realtimecoin.persist`             snapshot / save / load / stationary
======================================= ====================================

State-machine API, one cycle per trial::

    model.observe_q(q)    # stage the cue for the upcoming trial (optional)
    model.observe_y(y)    # process the feedback; ADVANCES the trial counter
    model.motor_output()  # ... and any other query

Randomness goes exclusively through ``model.rng``, a
:class:`numpy.random.Generator`; the global ``numpy.random`` state is never
used. Seeding the constructor therefore makes a whole run reproducible.

Deprecated MATLAB aliases (``predicted_context_probabilities``,
``responsibilities``, ``context_predicted_probabilities``,
``context_responsibilities``) are deliberately NOT ported; use the explicit
``*_vector`` / ``*_map`` forms.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import (
    alignment as _alignment,
    context as _context,
    persist as _persist,
    pipeline_md as _pipeline_md,
    pipeline_scalar as _pipeline_scalar,
    queries_aligned as _queries_aligned,
    queries_core as _queries_core,
    queries_density as _queries_density,
    state as _state,
)
from .exceptions import NameValuePairsError
from .state import ParticleState

__all__ = ["RealTimeCOIN", "PROPERTY_NAMES"]

#: Public property names, in MATLAB declaration order. This is the list
#: ``serializable_state`` walks (MATLAB's ``properties(obj)`` minus the
#: dependent ``Trial``).
PROPERTY_NAMES = (
    "num_particles",
    "max_contexts",
    "state_dim",
    "gamma_context",
    "alpha_context",
    "rho_context",
    "gamma_cue",
    "alpha_cue",
    "prior_mean_retention",
    "prior_mean_drift",
    "prior_mean_bias",
    "prior_precision_retention",
    "prior_precision_drift",
    "prior_precision_bias",
    "sigma_process_noise",
    "sigma_sensory_noise",
    "sigma_motor_noise",
    "process_noise_covariance",
    "observation_noise_covariance",
    "infer_bias",
)


class RealTimeCOIN:
    """Sequential (real-time) COIN particle filter.

    The COntextual INference model of Heald, Lengyel & Wolpert (2021),
    "Contextual inference underlies the learning of sensorimotor repertoires",
    Nature 600:489-493, run one observation at a time instead of over a
    pre-generated block.

    All constructor arguments are keyword-only and mirror the MATLAB properties
    exactly, including their default values (which are the fitted group-average
    parameters reported in that paper - do not round them).

    Parameters
    ----------
    num_particles : int, optional
        Particle count ``P``. Default 100.
    max_contexts : int, optional
        Maximum contexts per particle. The context axis has
        ``max_contexts + 1`` slots, the extra one being the novel context.
        Default 10.
    state_dim : int, optional
        Latent (and, since the observation map is the identity, observation)
        dimension ``N``. ``1`` selects the scalar pipeline verbatim; ``> 1``
        selects the multi-dimensional pipeline. Default 1.
    gamma_context : float, optional
        DP novelty concentration for contexts. Non-negative. Default 0.1.
    alpha_context : float, optional
        DP transition concentration for contexts. Positive. Default 8.955.
    rho_context : float, optional
        Sticky self-transition weight, in [0, 1). Default 0.2501.
    gamma_cue : float, optional
        DP novelty concentration for cues. Non-negative. Default 0.1.
    alpha_cue : float, optional
        DP cue-context concentration. Positive. Default 25.
    prior_mean_retention : float, optional
        Prior mean of the retention factor ``a``. Default 0.9425.
    prior_mean_drift : float, optional
        Prior mean of the drift ``d``. Default 0.0.
    prior_mean_bias : float, optional
        Prior mean of the observation bias. Default 0.0.
    prior_precision_retention : float, optional
        Prior precision of ``a``. Non-negative. Default ``837.1 ** 2``.
    prior_precision_drift : float, optional
        Prior precision of ``d``. Non-negative. Default ``1.2227e3 ** 2``.
    prior_precision_bias : float, optional
        Prior precision of the bias. Non-negative. Default ``70 ** 2``.
    sigma_process_noise : float, optional
        Process-noise standard deviation. Non-negative. Default 0.0089.
    sigma_sensory_noise : float, optional
        Sensory-noise standard deviation. Non-negative. Default 0.03.
    sigma_motor_noise : float, optional
        Motor-noise standard deviation. Non-negative. Default 0.0.
    process_noise_covariance : array_like or None, optional
        Explicit ``(N, N)`` process-noise covariance ``Q`` for the
        multi-dimensional model. ``None`` selects the isotropic default
        ``sigma_process_noise ** 2 * I``. Ignored (with a warning) when
        ``state_dim == 1``.
    observation_noise_covariance : array_like or None, optional
        Explicit ``(N, N)`` observation-noise covariance ``R``. ``None``
        selects ``(sigma_sensory_noise ** 2 + sigma_motor_noise ** 2) * I``.
        Ignored (with a warning) when ``state_dim == 1``.
    infer_bias : bool, optional
        Infer a per-context observation bias. Default ``False``.
    rng : numpy.random.Generator or int or None, optional
        Random source, passed through :func:`numpy.random.default_rng`. An
        ``int`` seeds a fresh generator, making the run reproducible.

    Attributes
    ----------
    Trial : int
        Read-only trial counter; starts at 0 and is advanced by
        :meth:`observe_y`.
    rng : numpy.random.Generator
        The model's only source of randomness.
    D : realtimecoin.state.ParticleState
        The vectorised particle state. See :mod:`realtimecoin.state` for the
        array-shape contract.

    Notes
    -----
    The property validators run ONCE, at construction. MATLAB's ``arguments``
    blocks are property validators that re-fire on every assignment for the
    life of the handle object; here the attributes are plain afterwards, so
    assigning e.g. ``model.rho_context = 1.5`` post-construction is NOT checked
    and will produce nonsense deep in the filter. Rebuild the model (or go
    through the persistence layer) rather than mutating hyperparameters in
    place.

    Raises
    ------
    NameValuePairsError
        If an unrecognised keyword argument is supplied.
    ValueError
        If a property fails its validator. The message carries the MATLAB
        error identifier as a prefix (for example
        ``"RealTimeCOIN:CovarianceNotPSD: ..."``).

    Examples
    --------
    >>> from realtimecoin import RealTimeCOIN
    >>> model = RealTimeCOIN(num_particles=50, rng=0)
    >>> model.Trial
    0
    """

    def __init__(
        self,
        *,
        num_particles=100,
        max_contexts=10,
        state_dim=1,
        gamma_context=0.1,
        alpha_context=8.955,
        rho_context=0.2501,
        gamma_cue=0.1,
        alpha_cue=25,
        prior_mean_retention=0.9425,
        prior_mean_drift=0.0,
        prior_mean_bias=0.0,
        prior_precision_retention=837.1 ** 2,
        prior_precision_drift=1.2227e3 ** 2,
        prior_precision_bias=70 ** 2,
        sigma_process_noise=0.0089,
        sigma_sensory_noise=0.03,
        sigma_motor_noise=0.0,
        process_noise_covariance=None,
        observation_noise_covariance=None,
        infer_bias=False,
        rng=None,
        **unknown,
    ):
        if unknown:
            # MATLAB raises RealTimeCOIN:NameValuePairs for a bad option list.
            raise NameValuePairsError(
                "Unrecognised option(s): %s." % ", ".join(sorted(unknown))
            )

        self.num_particles = _positive_integer(num_particles, "num_particles")
        self.max_contexts = _positive_integer(max_contexts, "max_contexts")
        self.state_dim = _positive_integer(state_dim, "state_dim")
        self.gamma_context = _nonnegative(gamma_context, "gamma_context")
        self.alpha_context = _positive(alpha_context, "alpha_context")
        self.rho_context = _unit_interval_exclusive(rho_context, "rho_context")
        self.gamma_cue = _nonnegative(gamma_cue, "gamma_cue")
        self.alpha_cue = _positive(alpha_cue, "alpha_cue")
        self.prior_mean_retention = float(prior_mean_retention)
        self.prior_mean_drift = float(prior_mean_drift)
        self.prior_mean_bias = float(prior_mean_bias)
        self.prior_precision_retention = _nonnegative(
            prior_precision_retention, "prior_precision_retention"
        )
        self.prior_precision_drift = _nonnegative(
            prior_precision_drift, "prior_precision_drift"
        )
        self.prior_precision_bias = _nonnegative(
            prior_precision_bias, "prior_precision_bias"
        )
        self.sigma_process_noise = _nonnegative(
            sigma_process_noise, "sigma_process_noise"
        )
        self.sigma_sensory_noise = _nonnegative(
            sigma_sensory_noise, "sigma_sensory_noise"
        )
        self.sigma_motor_noise = _nonnegative(sigma_motor_noise, "sigma_motor_noise")
        self.process_noise_covariance = _as_optional_matrix(process_noise_covariance)
        self.observation_noise_covariance = _as_optional_matrix(
            observation_noise_covariance
        )
        self.infer_bias = bool(infer_bias)

        self.rng = np.random.default_rng(rng)

        # --- private state (public attributes by Python convention) ---
        self.D = None
        self.pending_q = None
        self.trial = 0
        self.cue_values = []
        self.alignment_cache = None
        self.alignment_seed = None
        self.state_version = 0

        validate_multi_dim_config(self)
        _state.reset_particles(self)

    # ------------------------------------------------------------------
    # Read-only dependent property
    # ------------------------------------------------------------------

    @property
    def Trial(self):
        """int: Number of trials processed so far (read-only)."""
        return self.trial

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def observe_q(self, q=None):
        """Stage the sensory cue for the upcoming trial.

        Records the raw cue value for the next trial WITHOUT advancing the
        trial counter. The staged cue is consumed by the following
        :meth:`observe_y`, which runs the inference pipeline. Call this before
        each :meth:`observe_y` when the paradigm provides an explicit cue.

        Parameters
        ----------
        q : float or None, optional
            Raw scalar cue identifier; distinct values are mapped to
            consecutive internal cue labels. ``None`` or ``nan`` CLEARS any
            pending cue, so the next trial is treated as cue-free.

        Returns
        -------
        None

        See Also
        --------
        observe_y : consumes the staged cue.
        """
        if q is None:
            self.pending_q = None
            return
        value = float(q)
        if np.isnan(value):
            self.pending_q = None
            return
        self.pending_q = value

    def observe_y(self, y=None):
        """Process one trial's feedback and advance the inference pipeline.

        Consumes the cue staged by :meth:`observe_q`, runs the full COIN
        particle-filter update for the trial, advances the trial counter and
        invalidates the cached context alignment. Call it exactly once per
        trial.

        The nine pipeline steps run in this order in BOTH the scalar and the
        multi-dimensional branch::

            predict_context -> predict_states -> predict_state_feedback
            -> resample_particles -> sample_context
            -> update_belief_about_states -> sample_states
            -> update_sufficient_statistics -> sample_parameters

        Parameters
        ----------
        y : float or array_like or None, optional
            Observed state feedback: a scalar when ``state_dim == 1``,
            otherwise a length-``state_dim`` vector. ``None`` or ``nan`` marks
            a missing observation - the trial still runs, with states predicted
            rather than corrected. In the multi-dimensional case individual
            ``nan`` entries mark partially-observed dimensions via an
            observation mask.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``y`` does not have ``state_dim`` elements.

        See Also
        --------
        observe_q : stages the cue.
        """
        # Normalise y into y_val plus, for the MD path, a per-dimension mask of
        # the entries that are actually observed (non-nan).
        obs_mask = None
        if self.state_dim > 1:
            # An empty y is MATLAB's isempty(y): nothing observed this trial.
            if y is None or np.asarray(y, dtype=float).size == 0:
                y_val = None
                obs_mask = np.zeros(self.state_dim, dtype=bool)
            else:
                y_val = _as_state_vector(y, self.state_dim)
                obs_mask = ~np.isnan(y_val)
        elif y is None:
            y_val = None
        else:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            # MATLAB treats both [] and a scalar NaN as a missing observation;
            # anything else must have exactly state_dim (== 1) elements.
            if y_arr.size == 0 or (y_arr.size == 1 and np.isnan(y_arr[0])):
                y_val = None
            else:
                y_val = float(_as_state_vector(y_arr, self.state_dim)[0])

        # Resolve the cue staged by observe_q into a 0-based cue label for this
        # trial (this also clears pending_q).
        q_val = _state.consume_pending_cue(self)

        if self.state_dim > 1:
            # The context-inference step is dimension-agnostic and reused; the
            # state and parameter steps have dedicated *_md implementations.
            _context.predict_context(self, q_val)
            _pipeline_md.predict_states_md(self)
            _pipeline_md.predict_state_feedback_md(self)
            _pipeline_md.resample_particles_md(self, y_val, q_val, obs_mask)
            _pipeline_md.sample_context_md(self, q_val)
            _pipeline_md.update_belief_about_states_md(self, y_val, obs_mask)
            _pipeline_md.sample_states_md(self, y_val, obs_mask)
            _pipeline_md.update_sufficient_statistics_md(
                self, y_val, q_val, obs_mask
            )
            _pipeline_md.sample_parameters_md(self)
        else:
            _context.predict_context(self, q_val)
            _pipeline_scalar.predict_states(self)
            _pipeline_scalar.predict_state_feedback(self)
            _pipeline_scalar.resample_particles(self, y_val, q_val)
            _pipeline_scalar.sample_context(self, q_val)
            _pipeline_scalar.update_belief_about_states(self, y_val)
            _pipeline_scalar.sample_states(self, y_val)
            _pipeline_scalar.update_sufficient_statistics(self, y_val, q_val)
            _pipeline_scalar.sample_parameters(self)

        # Advance the trial and drop the cached alignment, now stale relative to
        # the updated particle set.
        self.trial = self.trial + 1
        _alignment.invalidate_context_alignment(self)

    # ------------------------------------------------------------------
    # Construction from a prepared state
    # ------------------------------------------------------------------

    @classmethod
    def from_state(
        cls,
        properties,
        state,
        *,
        trial=1,
        cue_values=(1,),
        alignment_seed=None,
        rng=None,
    ):
        """Build a model from a properties mapping and a prepared particle state.

        The Python counterpart of the MATLAB test helper
        ``testutil.loadFixtureModel``: it yields a model whose internal state is
        fully controlled by the caller, without going through a file. Used to
        drive queries from hand-crafted particle configurations.

        Parameters
        ----------
        properties : dict
            Public property values; keys must be a subset of
            :data:`PROPERTY_NAMES`. Missing properties take their defaults.
        state : realtimecoin.state.ParticleState or dict
            The particle state to install. A ``dict`` is interpreted as
            ``{field_name: value}`` and may be partial - unspecified fields stay
            ``None``.
        trial : int, optional
            Trial counter to report; also the initial ``state_version``.
            Default 1.
        cue_values : sequence, optional
            Raw cue values already registered, in label order. Default
            ``(1,)``, matching the MATLAB fixture helper.
        alignment_seed : object, optional
            Warm-start seed for the alignment optimiser. Default ``None``.
        rng : numpy.random.Generator or int or None, optional
            Random source for the new model.

        Returns
        -------
        RealTimeCOIN
            A model carrying the supplied state.

        Raises
        ------
        NameValuePairsError
            If ``properties`` contains an unrecognised key.
        """
        model = cls(rng=rng, **dict(properties))
        if not isinstance(state, ParticleState):
            state = ParticleState(**dict(state))
        model._restore_state(
            state,
            trial=trial,
            cue_values=cue_values,
            alignment_seed=alignment_seed,
        )
        return model

    def _restore_state(self, state, *, trial, cue_values, alignment_seed):
        """Install a particle state and its bookkeeping in place.

        Minimal internal restore path used by :meth:`from_state` for a
        hand-built particle state. The full save/load persistence layer -
        schema versioning, validation and the model's own snapshot round-trip -
        lives in :mod:`realtimecoin.persist`.

        Parameters
        ----------
        state : realtimecoin.state.ParticleState
            Particle state to install (by reference, not copied).
        trial : int
            Trial counter; also becomes ``state_version``.
        cue_values : sequence
            Raw cue values already registered, in label order.
        alignment_seed : object
            Warm-start seed for the alignment optimiser.

        Returns
        -------
        None
        """
        self.D = state
        self.pending_q = None
        self.trial = int(trial)
        self.cue_values = list(cue_values)
        self.state_version = int(trial)
        self.alignment_seed = alignment_seed
        _alignment.invalidate_context_alignment(self)

    # ------------------------------------------------------------------
    # Point predictions and moments  ->  queries_core
    # ------------------------------------------------------------------

    def motor_output(self):
        """Expected state feedback, marginalised over contexts and particles.

        The mixture mean of the per-context feedback means weighted by the
        PREDICTED (pre-observation) context probabilities - COIN's definition of
        the motor output. Read-only.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.motor_output(self)

    def predictive_motor_output(self, q=None):
        """Expected next observation given the upcoming cue.

        A read-only one-step prediction from the current posterior, so it may be
        called between :meth:`observe_q` and :meth:`observe_y`.

        Parameters
        ----------
        q : float or None, optional
            RAW cue value of the upcoming trial. ``None`` uses the cue staged by
            :meth:`observe_q`. An unseen value is treated as the next novel cue
            label without registering it.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.predictive_motor_output(self, q)

    def state_moments(self):
        """Predictive latent-state mean and (co)variance.

        Marginalised over contexts and particles with the predicted context
        probabilities. Read-only.

        Returns
        -------
        mu : float or numpy.ndarray
            Scalar mean, or an ``(N,)`` mean vector.
        v : float or numpy.ndarray
            Scalar variance, or an ``(N, N)`` covariance matrix.
        """
        return _queries_core.state_moments(self)

    def predictive_feedback_moments(self, q=None):
        """One-step predictive observation mean and covariance.

        Read-only, so it can be called between :meth:`observe_q` and
        :meth:`observe_y` to obtain the model's belief about the imminent
        feedback.

        Parameters
        ----------
        q : int or None, optional
            0-based cue LABEL of the upcoming trial (not a raw cue value - this
            method indexes the cue matrix directly, matching the MATLAB source).
            ``None`` marginalises the cue out.

        Returns
        -------
        mu : float or numpy.ndarray
            Scalar mean, or an ``(N,)`` mean vector.
        sigma : float or numpy.ndarray
            Scalar variance, or an ``(N, N)`` covariance matrix.
        """
        return _queries_core.predictive_feedback_moments(self, q)

    def explicit_component(self):
        """Explicit component of adaptation.

        The predictive latent-state mean of the highest-responsibility context,
        averaged over particles. This is the ``c*1`` state; the implicit
        component is the residual of the motor output beyond it.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.explicit_component(self)

    def implicit_component(self):
        """Implicit component of adaptation.

        The motor output minus the average predicted state, i.e. the part of the
        motor output attributable to the across-context marginal bias.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.implicit_component(self)

    def state_cstar1(self):
        """Expected latent state of the highest-responsibility context.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.state_cstar1(self)

    def state_cstar2(self, q=None):
        """Expected state of the highest NEXT-trial predicted-prob context.

        Selects the argmax of the next trial's predicted probabilities but reads
        the state mean of the CURRENT trial. Computed without mutating the
        model.

        Parameters
        ----------
        q : float or None, optional
            RAW cue value of the upcoming trial; ``None`` uses the pending cue.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.state_cstar2(self, q)

    def state_cstar3(self):
        """Expected latent state of the highest predicted-probability context.

        Differs from :meth:`state_cstar2` only in which trial's predicted
        probabilities the argmax is taken over: here, the current one.

        Returns
        -------
        float or numpy.ndarray
            A scalar for ``state_dim == 1``, otherwise an ``(N,)`` vector.
        """
        return _queries_core.state_cstar3(self)

    def predicted_probability_cstar1(self):
        """Predicted probability of the highest-responsibility context.

        Returns
        -------
        float
            A scalar in [0, 1], for both the scalar and MD models.
        """
        return _queries_core.predicted_probability_cstar1(self)

    def predicted_probability_cstar3(self):
        """Highest predicted context probability on the current trial.

        Returns
        -------
        float
            A scalar in [0, 1].
        """
        return _queries_core.predicted_probability_cstar3(self)

    def kalman_gain_cstar1(self):
        """Kalman gain of the highest-responsibility context.

        ``state_var / state_feedback_var`` of the selected context, averaged
        over particles.

        Returns
        -------
        float
            The particle-averaged gain.

        Raises
        ------
        ScalarModelOnlyError
            If ``state_dim > 1``.
        """
        return _queries_core.kalman_gain_cstar1(self)

    def kalman_gain_cstar2(self, q=None):
        """Kalman gain of the highest NEXT-trial predicted-prob context.

        The selector uses the one-step-ahead prediction while the gain is that
        of the current trial.

        Parameters
        ----------
        q : float or None, optional
            RAW cue value of the upcoming trial; ``None`` uses the pending cue.

        Returns
        -------
        float
            The particle-averaged gain.

        Raises
        ------
        ScalarModelOnlyError
            If ``state_dim > 1``.
        """
        return _queries_core.kalman_gain_cstar2(self, q)

    # ------------------------------------------------------------------
    # Densities and CDFs  ->  queries_density
    # ------------------------------------------------------------------

    def state_probability(self, values):
        """Posterior latent-state density on a grid of query points.

        The posterior Gaussian mixture over particles and contexts, using the
        responsibilities and the filtered state moments.

        Parameters
        ----------
        values : array_like
            ``(K,)`` grid for the scalar model, or ``(K, N)`` with one query
            point per ROW for the multi-dimensional model.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        return _queries_density.state_probability(self, values)

    def state_feedback_probability(self, values):
        """Predictive feedback (observation) density on a grid.

        The predictive Gaussian mixture using the predicted context
        probabilities and the observation-noise-inflated moments.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        return _queries_density.state_feedback_probability(self, values)

    def novel_state_probability(self, values):
        """Posterior state density of the novel (not yet instantiated) context.

        The equal-weight mixture, over the particles that still have a free
        novel slot, of that slot's stationary state distribution. All zeros when
        every particle has saturated its context budget.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        return _queries_density.novel_state_probability(self, values)

    def novel_state_feedback_probability(self, values):
        """Feedback density of the novel (not yet instantiated) context.

        As :meth:`novel_state_probability`, with the mean shifted by the novel
        slot's sampled bias and the covariance inflated by the observation
        noise.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        return _queries_density.novel_state_feedback_probability(self, values)

    def retention_given_context_probability(self, values):
        """Per-context retention density on a grid.

        Parameters
        ----------
        values : array_like
            ``(K,)`` grid of query points.

        Returns
        -------
        dict
            ``{global_context_label: (K,) density}``.

        Raises
        ------
        ScalarModelOnlyError
            If ``state_dim > 1``.
        """
        return _queries_density.retention_given_context_probability(self, values)

    def drift_given_context_probability(self, values):
        """Per-context drift density on a grid.

        Parameters
        ----------
        values : array_like
            ``(K,)`` grid of query points.

        Returns
        -------
        dict
            ``{global_context_label: (K,) density}``.

        Raises
        ------
        ScalarModelOnlyError
            If ``state_dim > 1``.
        """
        return _queries_density.drift_given_context_probability(self, values)

    def bias_given_context_probability(self, values):
        """Per-context measurement-bias density on a grid.

        Parameters
        ----------
        values : array_like
            ``(K,)`` grid of query points.

        Returns
        -------
        dict
            ``{global_context_label: (K,) density}``.

        Raises
        ------
        BiasNotInferredError
            If ``infer_bias`` is false.
        ScalarModelOnlyError
            If ``state_dim > 1``.
        """
        return _queries_density.bias_given_context_probability(self, values)

    def bias_probability(self, values):
        """Marginal (across-context) measurement-bias density on a grid.

        Parameters
        ----------
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
        return _queries_density.bias_probability(self, values)

    def state_given_context_probability(self, values):
        """Per-context posterior latent-state density on a grid.

        Keyed by aligned global context label, so this triggers (and caches) the
        lazy context alignment.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        dict
            ``{global_context_label: (K,) density}``.
        """
        return _queries_aligned.state_given_context_probability(self, values)

    def state_feedback_given_context_probability(self, values):
        """Per-context predictive feedback density on a grid.

        Observation-space counterpart of
        :meth:`state_given_context_probability`; also triggers the alignment.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        dict
            ``{global_context_label: (K,) density}``.
        """
        return _queries_aligned.state_feedback_given_context_probability(self, values)

    def predictive_state_feedback_cdf(self, y, q=None):
        """Predictive CDF of the next feedback evaluated at ``y``.

        For the multi-dimensional model this is the vector of per-dimension
        MARGINAL CDFs, the standard probability-integral transform used for
        calibration checks.

        Parameters
        ----------
        y : float or array_like
            Query point: scalar, or an ``(N,)`` vector.
        q : float or None, optional
            RAW cue value of the upcoming trial; ``None`` uses the pending cue.

        Returns
        -------
        float or numpy.ndarray
            Scalar CDF value, or an ``(N,)`` vector of marginal CDF values.
        """
        return _queries_density.predictive_state_feedback_cdf(self, y, q)

    def predictive_cue_p_value(self, q, u=None):
        """Randomised predictive p-value for a cue value.

        ``p = F(q-) + u f(q)`` under the current predictive cue pmf. Under a
        correct model ``p`` is uniform on [0, 1].

        Parameters
        ----------
        q : float or None
            RAW cue value. A value never observed is treated as the next novel
            label - the trailing novel-cue column, which carries the novel-cue
            stick's mass, NOT zero - WITHOUT registering it. ``None`` returns
            ``nan``.
        u : float or None, optional
            Uniform variate in [0, 1]. ``None`` draws one from ``self.rng``,
            which makes the call non-reproducible - pass ``u`` in tests.

        Returns
        -------
        float
            The p-value in [0, 1], or ``nan``.
        """
        return _queries_density.predictive_cue_p_value(self, q, u)

    # ------------------------------------------------------------------
    # Context summaries: global (aligned)  ->  queries_aligned
    # ------------------------------------------------------------------

    def predicted_context_probabilities_vector(self):
        """Predicted (pre-observation) per-context probabilities, as a vector.

        Averaged over the modal particles in the aligned global frame; the
        trailing entry is the novel-context probability. Triggers (and caches)
        the lazy context alignment.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` probabilities.
        """
        return _queries_aligned.predicted_context_probabilities_vector(self)

    def predicted_context_probabilities_map(self):
        """Predicted per-context probabilities, keyed by global context label.

        Same weights as
        :meth:`predicted_context_probabilities_vector`; only strictly positive
        entries appear.

        Returns
        -------
        dict
            ``{global_context_label: probability}``.
        """
        return _queries_aligned.predicted_context_probabilities_map(self)

    def responsibilities_vector(self):
        """Posterior per-context responsibilities, as a vector.

        The post-observation counterpart of
        :meth:`predicted_context_probabilities_vector`.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` responsibilities.
        """
        return _queries_aligned.responsibilities_vector(self)

    def responsibilities_map(self):
        """Posterior context responsibilities, keyed by global context label.

        Returns
        -------
        dict
            ``{global_context_label: responsibility}``.
        """
        return _queries_aligned.responsibilities_map(self)

    def sampled_context_count(self):
        """Sampled-context occupancy across particles, normalised to sum to one.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` frequencies.
        """
        return _queries_aligned.sampled_context_count(self)

    def stationary_context_probabilities(self):
        """Stationary distribution of the aligned context transition matrix.

        The novel-context column is dropped and each row renormalised before
        solving, mirroring COIN's ``plot_stationary_probabilities``.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` stationary probabilities.
        """
        return _queries_aligned.stationary_context_probabilities(self)

    def global_transition_probabilities(self):
        """Expected global (franchise) transition distribution.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` franchise weights; the last active entry is
            the novel-context stick.
        """
        return _queries_aligned.global_transition_probabilities(self)

    def global_cue_probabilities(self):
        """Expected global (franchise) cue distribution.

        Cue labels are numbered by order of presentation, so no alignment is
        needed; the trailing entry is the novel-cue stick.

        Returns
        -------
        numpy.ndarray
            ``(Q,)`` franchise cue weights.

        Raises
        ------
        NoCuesError
            If no sensory cue has been observed yet.
        """
        return _queries_aligned.global_cue_probabilities(self)

    def local_transition_probabilities(self):
        """Expected local context-transition matrix in the aligned frame.

        Returns
        -------
        numpy.ndarray
            ``(K, K + 1)``; row ``i`` is the transition distribution out of
            global context ``i`` and column ``K`` is the novel context.
        """
        return _queries_aligned.local_transition_probabilities(self)

    def local_cue_probabilities(self):
        """Expected local cue-emission matrix in the aligned frame.

        Returns
        -------
        numpy.ndarray
            ``(K, Q)``; row ``i`` is the cue-emission distribution of global
            context ``i``.

        Raises
        ------
        NoCuesError
            If no sensory cue has been observed yet.
        """
        return _queries_aligned.local_cue_probabilities(self)

    def context_alignment(self):
        """Global context alignment across particles (cached).

        Maps each particle's arbitrary local context labels onto one globally
        consistent labelling for the current trial. Computed lazily and cached
        until the next :meth:`observe_y`.

        Returns
        -------
        dict
            Keys include ``K``, ``assignment``, ``modal_particle_mask``,
            ``modal_particle_indices``, ``global_contexts``, ``converged`` and
            ``iterations``.
        """
        return _queries_aligned.context_alignment(self)

    def diagnostics(self):
        """Full globally-aligned snapshot of the current particle state.

        Summarises every per-context quantity of the current trial in the
        aligned global frame. Triggers (and caches) the lazy alignment.

        Returns
        -------
        dict
            See :func:`realtimecoin.queries_aligned.diagnostics` for the key
            list (the multi-dimensional model returns the MD field set).
        """
        return _queries_aligned.diagnostics(self)

    # ------------------------------------------------------------------
    # Context summaries: local (fast, unaligned)  ->  queries_core
    # ------------------------------------------------------------------

    def predicted_context_probabilities_local(self):
        """Fast local-label predicted context probabilities.

        In the modal particles' LOCAL label frame, deliberately skipping the
        global relabelling. Intended for live plots and logging; the entries are
        not comparable across trials.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` weights.
        """
        return _queries_core.predicted_context_probabilities_local(self)

    def context_responsibilities_local(self):
        """Fast local-label posterior context weights.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` weights.
        """
        return _queries_core.context_responsibilities_local(self)

    def sampled_context_count_local(self):
        """Fast local-label sampled-context occupancy.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` frequencies.
        """
        return _queries_core.sampled_context_count_local(self)

    # ------------------------------------------------------------------
    # Persistence  ->  persist
    # ------------------------------------------------------------------

    def snapshot(self):
        """Capture the full model state as a plain, serialisable mapping.

        A true deep copy: after ``other.load_snapshot(model.snapshot())``,
        ``other`` reproduces every query output of ``model``.

        Returns
        -------
        dict
            The snapshot payload.
        """
        return _persist.snapshot(self)

    def load_snapshot(self, s):
        """Restore full model state in place from a snapshot mapping.

        Parameters
        ----------
        s : dict
            Mapping produced by :meth:`snapshot`.

        Returns
        -------
        None
        """
        return _persist.load_snapshot(self, s)

    def save_model(self, filename, set_stationary=True):
        """Serialise the model state to a file.

        Parameters
        ----------
        filename : str or os.PathLike
            Destination path.
        set_stationary : bool, optional
            When true (the default) the model is first placed at its stationary
            prior, so the saved snapshot is contingency-independent; the live
            object is restored afterwards even if serialisation fails.

        Returns
        -------
        None
        """
        return _persist.save_model(self, filename, set_stationary)

    def load_model(self, filename):
        """Restore model state in place from a file written by :meth:`save_model`.

        Parameters
        ----------
        filename : str or os.PathLike
            Source path.

        Returns
        -------
        None

        Raises
        ------
        ModelFormatError
            If the file does not hold a recognisable model snapshot.
        """
        return _persist.load_model(self, filename)

    def set_stationary(self):
        """Reset the particle filter to its stationary prior.

        Re-initialises every particle's context and state beliefs to the
        stationary distribution implied by the current hyperparameters, rewinds
        the trial counter to 0 and clears any pending cue.

        Returns
        -------
        None
        """
        return _persist.set_stationary(self)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def validate_multi_dim_config(model):
    """Validate the multi-dimensional configuration.

    Checks any user-supplied process / observation noise covariance against
    ``state_dim`` and the symmetric positive-semidefinite requirement, so
    misconfigurations fail fast with a clear identifier rather than surfacing as
    an obscure linear-algebra error deep inside the particle filter.

    When ``state_dim == 1`` the scalar pipeline is used and the covariance
    overrides are IGNORED - a :class:`UserWarning` is issued if either is set,
    matching the MATLAB ``RealTimeCOIN:CovarianceIgnored`` warning. It is
    deliberately not an error.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        With one of the identifiers ``RealTimeCOIN:BadCovarianceSize``,
        ``RealTimeCOIN:BadCovarianceValue``,
        ``RealTimeCOIN:CovarianceNotSymmetric`` or
        ``RealTimeCOIN:CovarianceNotPSD``.
    """
    n = model.state_dim

    if n == 1:
        if (
            model.process_noise_covariance is not None
            or model.observation_noise_covariance is not None
        ):
            warnings.warn(
                "RealTimeCOIN:CovarianceIgnored: process_noise_covariance/"
                "observation_noise_covariance are ignored when state_dim == 1; "
                "the scalar sigma_* properties are used instead.",
                UserWarning,
                # 3 frames points at the RealTimeCOIN(...) call site, the only
                # caller in practice; a direct call attributes it one frame out.
                stacklevel=3,
            )
        return

    _validate_noise_covariance(
        model.process_noise_covariance, n, "process_noise_covariance"
    )
    _validate_noise_covariance(
        model.observation_noise_covariance, n, "observation_noise_covariance"
    )


def _validate_noise_covariance(m, n, name):
    """Assert that ``m`` is ``None`` or a valid ``(n, n)`` symmetric PSD matrix."""
    if m is None:
        return   # None selects the isotropic default elsewhere.
    if m.shape != (n, n):
        raise ValueError(
            "RealTimeCOIN:BadCovarianceSize: %s must be a %d-by-%d matrix to "
            "match state_dim, but is %s." % (name, n, n, "-by-".join(map(str, m.shape)))
        )
    if not np.all(np.isfinite(m)):
        raise ValueError(
            "RealTimeCOIN:BadCovarianceValue: %s must contain only finite "
            "values." % (name,)
        )
    # The shape check above guarantees a non-empty (n, n) matrix with n >= 2,
    # so no empty-matrix guards are needed below.
    scale = max(1.0, float(np.max(np.abs(m))))
    asym = float(np.max(np.abs(m - m.T)))
    if asym > 1e-9 * scale:
        raise ValueError(
            "RealTimeCOIN:CovarianceNotSymmetric: %s must be symmetric (max "
            "asymmetry %.3g)." % (name, asym)
        )
    # PSD check on the symmetrised matrix; a small negative tolerance absorbs
    # round-off in user-supplied matrices.
    m_sym = (m + m.T) / 2.0
    min_eig = float(np.min(np.linalg.eigvalsh(m_sym))) if m.size else 0.0
    if min_eig < -1e-9 * scale:
        raise ValueError(
            "RealTimeCOIN:CovarianceNotPSD: %s must be positive semidefinite "
            "(min eigenvalue %.3g)." % (name, min_eig)
        )


def _positive_integer(value, name):
    """Coerce to ``int`` and require it to be a positive whole number."""
    number = float(value)
    if not np.isfinite(number) or number != int(number):
        raise ValueError(
            "MATLAB:validators:mustBeInteger: %s must be an integer." % (name,)
        )
    if number <= 0:
        raise ValueError(
            "MATLAB:validators:mustBePositive: %s must be positive." % (name,)
        )
    return int(number)


def _positive(value, name):
    """Coerce to ``float`` and require it to be strictly positive."""
    number = float(value)
    if not (number > 0):
        raise ValueError(
            "MATLAB:validators:mustBePositive: %s must be positive." % (name,)
        )
    return number


def _nonnegative(value, name):
    """Coerce to ``float`` and require it to be non-negative."""
    number = float(value)
    if not (number >= 0):
        raise ValueError(
            "MATLAB:validators:mustBeNonnegative: %s must be non-negative."
            % (name,)
        )
    return number


def _unit_interval_exclusive(value, name):
    """Coerce to ``float`` and require it to lie in [0, 1)."""
    number = _nonnegative(value, name)
    if not (number < 1):
        raise ValueError(
            "MATLAB:validators:mustBeLessThan: %s must be less than 1." % (name,)
        )
    return number


def _as_optional_matrix(value):
    """Coerce an optional covariance override to a float array (or ``None``).

    An empty array is treated as ``None``, matching MATLAB's ``[]`` sentinel.
    """
    if value is None:
        return None
    arr = np.atleast_2d(np.asarray(value, dtype=float))
    if arr.size == 0:
        return None
    return arr


def _as_state_vector(y, state_dim):
    """Coerce ``y`` to a length-``state_dim`` float vector, or raise."""
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size != state_dim:
        raise ValueError(
            "Size:notStateDim: Size of observed y must match the state "
            "dimension (expected %d, received %d)." % (state_dim, arr.size)
        )
    return arr

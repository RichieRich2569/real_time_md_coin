"""Particle state container, initialisation, cue registry and config helpers.

Translated from ``@RealTimeCOIN/private/``: ``resetParticles``,
``resetParticlesMD``, ``ensureCueColumn``, ``instantiateCueIfNeeded``,
``consumePendingCue``, ``peekCueLabel``, ``processNoiseCov``,
``observationNoiseCov``, ``observationVariance``, ``feedbackTransform`` and
``dynamicsPriorMD``.

The MATLAB implementation keeps every particle-filter array in one private
struct ``obj.D``. Here that struct is the :class:`ParticleState` dataclass; one
class serves both the scalar and the multi-dimensional model, with the fields
that belong to the other mode left at ``None``. ``dataclasses.fields()`` drives
the deep copy in :func:`copy_state`, so a new field is picked up automatically.


Array layout (THE contract every other module is written against)
-----------------------------------------------------------------
MATLAB is contexts-first / particles-LAST; this package is **particles-LEADING
with the matrix dimensions trailing**. The transposition is total: wherever
MATLAB writes ``X(c, p)`` the Python code writes ``X[p, c]``, and wherever
MATLAB writes ``X(:, :, c, p)`` (a matrix per context per particle) the Python
code writes ``X[p, c, :, :]``. That ordering makes every per-particle matrix a
contiguous C-order block, which is what numpy's batched linear algebra wants.

Symbols used in the table::

    P = num_particles
    C = max_contexts + 1        (the +1 is the "novel context" slot)
    N = state_dim
    Q = number of cue columns instantiated so far (grows at run time)

============================== ==================== ====================
field (MATLAB name)            scalar shape         multi-dim shape
============================== ==================== ====================
n_active   (MATLAB ``D.C``)    (P,) int             (P,) int
context                        (P,) int             (P,) int
previous_context               (P,) int             (P,) int
n_cues     (MATLAB ``D.Q``)    int (scalar)         int (scalar)
i_resampled                    (P,) int             (P,) int
i_observed                     (P,) int             (P,) int
n_context                      (P, C, C)            (P, C, C)
n_cue                          (P, C, Q)            (P, C, Q)
global_transition_probabilit.  (P, C)               (P, C)
global_cue_probabilities       (P, Q)               (P, Q)
local_transition_matrix        (P, C, C)            (P, C, C)
local_cue_matrix               (P, C, Q)            (P, C, Q)
prior_probabilities            (P, C)               (P, C)
predicted_probabilities        (P, C)               (P, C)
responsibilities               (P, C)               (P, C)
probability_cue                (P, C)               (P, C)
probability_state_feedback     (P, C)               (P, C)
state_filtered_mean            (P, C)               (P, C, N)
previous_state_filtered_mean   (P, C)               (P, C, N)
state_mean                     (P, C)               (P, C, N)
state_feedback_mean            (P, C)               (P, C, N)
bias                           (P, C)               (P, C, N)
x_dynamics                     (P, C)               (P, C, N)
previous_x_dynamics            (P, C)               (P, C, N)
x_bias                         (P,)                 (P, N)
retention                      (P, C)               --
drift                          (P, C)               --
state_filtered_var             (P, C)               --
previous_state_filtered_var    (P, C)               --
state_var                      (P, C)               --
state_feedback_var             (P, C)               --
dynamics_ss_1                  (P, C, 2)            --
dynamics_ss_2                  (P, C, 2, 2)         --
dynamics_mean                  (P, C, 2)            --
dynamics_covar                 (P, C, 2, 2)         --
bias_ss_1                      (P, C)               --
bias_ss_2                      (P, C)               --
bias_mean                      (P, C)               --
bias_var                       (P, C)               --
state_filtered_cov             --                   (P, C, N, N)
previous_state_filtered_cov    --                   (P, C, N, N)
state_cov                      --                   (P, C, N, N)
state_feedback_cov             --                   (P, C, N, N)
Theta                          --                   (P, C, N, N+1)
Lambda_xx                      --                   (P, C, N+1, N+1)
Lambda_yx                      --                   (P, C, N, N+1)
bias_info_ss                   --                   (P, C, N)
bias_precision_ss              --                   (P, C, N, N)
============================== ==================== ====================

Naming decisions worth memorising:

* ``n_active`` is MATLAB's ``D.C`` - the number of contexts particle ``p`` has
  instantiated. ``C`` was too easy to confuse with the context-axis length, so
  it was renamed; nothing else about it changed (it is still a COUNT, not an
  index).
* ``n_cues`` is MATLAB's ``D.Q`` - the number of instantiated cue categories.
  Numerically identical to the MATLAB value, because MATLAB's ``D.Q`` holds the
  highest 1-based cue label seen, which equals the count.
* ``Theta``, ``Lambda_xx`` and ``Lambda_yx`` keep their MATLAB capitalisation:
  they are mathematical symbols from the matrix-normal dynamics model, and
  keeping the names identical makes the MD pipeline diffable line by line.
* ``i_observed`` is the ONE field whose representation genuinely changed.
  MATLAB stores ``sub2ind([Cmax, P], context, 1:P)``, i.e. LINEAR indices into
  the ``(Cmax, P)`` plane. Linear indices do not survive the transposition, so
  here ``i_observed`` is simply the ``(P,)`` array of 0-based active-context
  column indices; the MATLAB gather ``X(i_observed)`` becomes
  ``X[np.arange(P), state.i_observed]``. (After ``sample_states`` it equals
  ``state.context``, but it is stored separately, exactly as MATLAB does, so a
  later context resample cannot silently redirect the bias update.)

Context labels are 0-BASED internally. For a particle with ``k = n_active[p]``
instantiated contexts, slots ``0 .. k-1`` are real contexts and slot ``k`` is
the novel context (MATLAB's ``C(p) + 1``). Cue labels are 0-based too, so
``cue_values[q]`` is the raw value of internal label ``q`` and ``q`` indexes the
trailing axis of ``n_cue`` / ``local_cue_matrix``.

Fields created lazily by the pipeline (``x_dynamics``, ``previous_x_dynamics``,
``x_bias``, ``i_observed``, ``dynamics_mean``, ``dynamics_covar``,
``bias_mean``, ``bias_var``) stay ``None`` after a reset, mirroring MATLAB where
the struct field simply does not exist until the step that writes it has run.
Reading one before its first write is therefore a loud ``None`` failure rather
than a silently plausible zero.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from .numerics import (
    precision_to_variance,
    stationary_state_cov_md,
    stationary_state_mean,
    stationary_state_mean_md,
    stationary_state_var,
)
from .samplers import beta_sample, sample_scalar_normal, sample_stable_theta

__all__ = [
    "ParticleState",
    "copy_state",
    "reset_particles",
    "reset_particles_md",
    "ensure_cue_column",
    "instantiate_cue_if_needed",
    "consume_pending_cue",
    "peek_cue_label",
    "process_noise_cov",
    "observation_noise_cov",
    "observation_variance",
    "feedback_transform",
    "dynamics_prior_md",
]

EPS = float(np.finfo(float).eps)


@dataclass
class ParticleState:
    """Vectorised particle-filter state (the Python form of MATLAB ``obj.D``).

    Every field is documented with both its scalar and its multi-dimensional
    shape in the module docstring table; that table is the authoritative
    contract. All fields default to ``None`` so a partially populated state can
    be hand-built in tests and fed to
    :meth:`realtimecoin.model.RealTimeCOIN.from_state`.

    Attributes
    ----------
    n_active : numpy.ndarray
        ``(P,)`` integer count of contexts instantiated by each particle
        (MATLAB ``D.C``). Slot ``n_active[p]`` is that particle's novel slot.
    context : numpy.ndarray
        ``(P,)`` 0-based sampled context label of the current trial.
    previous_context : numpy.ndarray
        ``(P,)`` 0-based sampled context label of the previous trial.
    n_cues : int
        Number of instantiated cue categories (MATLAB ``D.Q``).
    """

    # --- discrete per-particle bookkeeping ---
    n_active: Optional[np.ndarray] = None            # (P,)   MATLAB D.C
    context: Optional[np.ndarray] = None             # (P,)
    previous_context: Optional[np.ndarray] = None    # (P,)
    n_cues: int = 0                                  # scalar MATLAB D.Q
    i_resampled: Optional[np.ndarray] = None         # (P,)
    i_observed: Optional[np.ndarray] = None          # (P,)  see module docstring

    # --- context-inference sufficient statistics (shared scalar / MD) ---
    n_context: Optional[np.ndarray] = None                        # (P, C, C)
    n_cue: Optional[np.ndarray] = None                            # (P, C, Q)
    global_transition_probabilities: Optional[np.ndarray] = None  # (P, C)
    global_cue_probabilities: Optional[np.ndarray] = None         # (P, Q)
    local_transition_matrix: Optional[np.ndarray] = None          # (P, C, C)
    local_cue_matrix: Optional[np.ndarray] = None                 # (P, C, Q)

    # --- context probabilities (shared scalar / MD) ---
    prior_probabilities: Optional[np.ndarray] = None          # (P, C)
    predicted_probabilities: Optional[np.ndarray] = None      # (P, C)
    responsibilities: Optional[np.ndarray] = None             # (P, C)
    probability_cue: Optional[np.ndarray] = None              # (P, C)
    probability_state_feedback: Optional[np.ndarray] = None   # (P, C)

    # --- state moments: (P, C) scalar, (P, C, N) multi-dimensional ---
    state_filtered_mean: Optional[np.ndarray] = None
    previous_state_filtered_mean: Optional[np.ndarray] = None
    state_mean: Optional[np.ndarray] = None
    state_feedback_mean: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None

    # --- sampled latent trajectory: (P, C) scalar, (P, C, N) MD ---
    x_dynamics: Optional[np.ndarray] = None
    previous_x_dynamics: Optional[np.ndarray] = None
    x_bias: Optional[np.ndarray] = None               # (P,) scalar, (P, N) MD

    # --- scalar-model-only fields ---
    retention: Optional[np.ndarray] = None                    # (P, C)
    drift: Optional[np.ndarray] = None                        # (P, C)
    state_filtered_var: Optional[np.ndarray] = None           # (P, C)
    previous_state_filtered_var: Optional[np.ndarray] = None  # (P, C)
    state_var: Optional[np.ndarray] = None                    # (P, C)
    state_feedback_var: Optional[np.ndarray] = None           # (P, C)
    dynamics_ss_1: Optional[np.ndarray] = None                # (P, C, 2)
    dynamics_ss_2: Optional[np.ndarray] = None                # (P, C, 2, 2)
    dynamics_mean: Optional[np.ndarray] = None                # (P, C, 2)
    dynamics_covar: Optional[np.ndarray] = None               # (P, C, 2, 2)
    bias_ss_1: Optional[np.ndarray] = None                    # (P, C)
    bias_ss_2: Optional[np.ndarray] = None                    # (P, C)
    bias_mean: Optional[np.ndarray] = None                    # (P, C)
    bias_var: Optional[np.ndarray] = None                     # (P, C)

    # --- multi-dimensional-model-only fields ---
    state_filtered_cov: Optional[np.ndarray] = None           # (P, C, N, N)
    previous_state_filtered_cov: Optional[np.ndarray] = None  # (P, C, N, N)
    state_cov: Optional[np.ndarray] = None                    # (P, C, N, N)
    state_feedback_cov: Optional[np.ndarray] = None           # (P, C, N, N)
    Theta: Optional[np.ndarray] = None                        # (P, C, N, N+1)
    Lambda_xx: Optional[np.ndarray] = None                    # (P, C, N+1, N+1)
    Lambda_yx: Optional[np.ndarray] = None                    # (P, C, N, N+1)
    bias_info_ss: Optional[np.ndarray] = None                 # (P, C, N)
    bias_precision_ss: Optional[np.ndarray] = None            # (P, C, N, N)

    def field_names(self):
        """Return the tuple of declared field names.

        Returns
        -------
        tuple of str
            Names in declaration order, the order :func:`copy_state` walks.
        """
        return tuple(f.name for f in fields(self))

    def as_dict(self):
        """Return a shallow ``{name: value}`` view of every field.

        Returns
        -------
        dict
            Field name to value. The arrays are NOT copied; use
            :func:`copy_state` when an independent state is required.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


def copy_state(state):
    """Deep-copy a particle state.

    Every ``numpy`` field is copied with :func:`numpy.copy` and every other
    field with :func:`copy.deepcopy`, so the result shares no buffer with
    ``state``. This is what makes ``snapshot`` a true value copy (MATLAB gets
    that for free from struct value semantics).

    Parameters
    ----------
    state : ParticleState
        State to duplicate.

    Returns
    -------
    ParticleState
        An independent copy.
    """
    out = ParticleState()
    for f in fields(state):
        value = getattr(state, f.name)
        if isinstance(value, np.ndarray):
            setattr(out, f.name, value.copy())
        elif value is None or isinstance(value, (bool, int, float, str)):
            # Immutable: sharing it IS an independent copy, and skipping the
            # deepcopy machinery matters because roughly half the fields are
            # None in any one mode.
            setattr(out, f.name, value)
        else:
            setattr(out, f.name, _copy.deepcopy(value))
    return out


# --------------------------------------------------------------------------
# Configuration helpers (translated from the private config accessors)
# --------------------------------------------------------------------------


def observation_variance(model):
    """Scalar observation (measurement) noise variance.

    ``v = sigma_sensory_noise^2 + sigma_motor_noise^2``, the sum of the two
    independent noise sources. :func:`observation_noise_cov` scales this into
    the isotropic ``R = v I_N`` of the multi-dimensional model.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``sigma_sensory_noise`` / ``sigma_motor_noise`` are read.

    Returns
    -------
    float
        The total observation variance.
    """
    return model.sigma_sensory_noise ** 2 + model.sigma_motor_noise ** 2


def process_noise_cov(model):
    """State (process) noise covariance ``Q``.

    Returns the explicitly supplied ``process_noise_covariance`` (symmetrised)
    when set, otherwise the isotropic default
    ``sigma_process_noise^2 * I_N``. At ``N == 1`` the default collapses to the
    scalar process variance used by the scalar pipeline, so the two branches
    agree by construction.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``state_dim``, ``sigma_process_noise`` and the optional
        override.

    Returns
    -------
    numpy.ndarray
        ``(N, N)`` process-noise covariance.
    """
    n = model.state_dim
    if model.process_noise_covariance is None:
        return model.sigma_process_noise ** 2 * np.eye(n)
    q = np.asarray(model.process_noise_covariance, dtype=float)
    return (q + q.T) / 2.0


def observation_noise_cov(model):
    """Observation (measurement) noise covariance ``R``.

    Returns the explicitly supplied ``observation_noise_covariance``
    (symmetrised) when set, otherwise the isotropic default
    ``observation_variance(model) * I_N``. Collapses to the scalar observation
    variance at ``N == 1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``state_dim``, the sigma properties and the optional
        override.

    Returns
    -------
    numpy.ndarray
        ``(N, N)`` observation-noise covariance.
    """
    n = model.state_dim
    if model.observation_noise_covariance is None:
        return observation_variance(model) * np.eye(n)
    r = np.asarray(model.observation_noise_covariance, dtype=float)
    return (r + r.T) / 2.0


def feedback_transform(mean_in, cov_in, bias, noise_cov):
    """Map state moments to feedback (observation-space) moments.

    The observation model that turns a latent-state Gaussian into the
    predictive feedback Gaussian::

        feedback mean = state mean + bias
        feedback cov  = state cov  + R

    Shape-agnostic: scalars for the scalar model, ``(N,)`` means with
    ``(N, N)`` covariances for the multi-dimensional model. Pure arithmetic; it
    reads and mutates no model state.

    Parameters
    ----------
    mean_in : array_like
        Latent-state mean.
    cov_in : array_like
        Latent-state variance / covariance.
    bias : array_like
        Sampled observation bias, broadcastable against ``mean_in``.
    noise_cov : array_like
        Observation noise variance / covariance, broadcastable against
        ``cov_in``.

    Returns
    -------
    mean_out : numpy.ndarray
        ``mean_in + bias``.
    cov_out : numpy.ndarray
        ``cov_in + noise_cov``.
    """
    return mean_in + bias, cov_in + noise_cov


def dynamics_prior_md(model):
    """Matrix-normal prior on the augmented dynamics ``Theta = [A | d]``.

    Returns the prior mean ``M0`` and the prior column-covariance precision
    ``V0inv`` and covariance ``V0`` of ``Theta ~ MN(M0, U = Q, V0)``::

        M0    = [prior_mean_retention * I_N , prior_mean_drift * 1_N]
        V0inv = sigma_process_noise^2 * diag([prec_ret * ones(N), prec_drift])

    With ``U = Q`` the posterior column covariance and mean reduce
    ALGEBRAICALLY at ``N == 1`` to the scalar update of ``sample_dynamics``.
    ``sigma_process_noise^2`` is the reference scale even when a custom ``Q`` is
    supplied (so the isotropic default reproduces the scalar prior); a zero
    scale is floored at ``eps``, mirroring the ``qVar == 0`` guard of the scalar
    sampler.

    Parameters
    ----------
    model : RealTimeCOIN
        Model supplying ``state_dim``, the prior means/precisions and
        ``sigma_process_noise``.

    Returns
    -------
    m0 : numpy.ndarray
        ``(N, N + 1)`` prior mean of ``[A | d]``.
    v0inv : numpy.ndarray
        ``(N + 1, N + 1)`` prior column precision (diagonal).
    v0 : numpy.ndarray
        ``(N + 1, N + 1)`` prior column covariance (diagonal, the exact
        inverse of ``v0inv``).
    """
    n = model.state_dim

    m0 = np.hstack(
        [
            model.prior_mean_retention * np.eye(n),
            model.prior_mean_drift * np.ones((n, 1)),
        ]
    )                                                     # (N, N + 1)

    s2 = model.sigma_process_noise ** 2
    if s2 == 0:
        s2 = EPS
    prec_diag = np.concatenate(
        [model.prior_precision_retention * np.ones(n),
         [model.prior_precision_drift]]
    )                                                     # (N + 1,)
    v0inv = s2 * np.diag(prec_diag)                       # (N + 1, N + 1)
    # v0inv is diagonal, so the inverse is exact and well conditioned. Going
    # through precision_to_variance maps a zero prior precision (a legal,
    # completely uninformative prior) to inf instead of raising a divide-by-zero
    # warning.
    v0 = np.diag(precision_to_variance(np.diag(v0inv)))   # (N + 1, N + 1)
    return m0, v0inv, v0


# --------------------------------------------------------------------------
# Particle initialisation
# --------------------------------------------------------------------------


def _finish_reset(model, d):
    """Install a freshly seeded state and rebuild everything derived from it.

    The common tail of :func:`reset_particles` and :func:`reset_particles_md`
    (identical in ``resetParticles.m`` and ``resetParticlesMD.m``): publish the
    new state, rebuild the local transition and cue matrices from it, drop the
    alignment warm-start seed and invalidate the alignment cache.

    Kept as one helper so a change to the reset contract cannot be applied to
    only one of the two dimensions.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.
    d : ParticleState
        The newly seeded particle state to install.

    Returns
    -------
    None
    """
    from . import context as _context       # local imports: break the cycle
    from . import alignment as _alignment   # state <-> context/alignment

    model.D = d
    _context.update_local_transition_matrix(model)
    _context.update_local_cue_matrix(model)
    model.alignment_seed = None
    _alignment.invalidate_context_alignment(model)


def reset_particles(model):
    """Initialise particle storage for the scalar (1-D) model.

    Allocates and seeds ``model.D`` for a fresh run, rebuilds the derived local
    transition / cue matrices and invalidates any cached context alignment.
    When ``state_dim > 1`` the call is forwarded to :func:`reset_particles_md`.

    Every particle starts with exactly one instantiated context (slot 0); the
    remaining slots hold prior draws so the novel slot is ready to be adopted.
    ``global_transition_probabilities`` puts all franchise mass on context 0.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to (re-)initialise in place. Uses ``model.rng`` for the prior
        draws.

    Returns
    -------
    None
    """
    # Dispatch to the multi-dimensional initialiser; the scalar branch below is
    # kept equivalent to the original COIN initialisation.
    if model.state_dim > 1:
        reset_particles_md(model)
        return

    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    rng = model.rng
    d = ParticleState()

    d.n_active = np.ones(p_count, dtype=int)              # (P,)
    d.context = np.zeros(p_count, dtype=int)              # (P,) 0-based label 0
    d.previous_context = np.zeros(p_count, dtype=int)     # (P,)
    d.n_cues = 0

    d.n_context = np.zeros((p_count, c_slots, c_slots))   # (P, C, C)
    d.n_cue = np.zeros((p_count, c_slots, 1))             # (P, C, Q=1)
    # Dynamics accumulators: the regressor is the augmented [s_prev; 1], hence
    # the trailing 2 (and 2-by-2 Gram matrix) per context per particle.
    d.dynamics_ss_1 = np.zeros((p_count, c_slots, 2))         # (P, C, 2)
    d.dynamics_ss_2 = np.zeros((p_count, c_slots, 2, 2))      # (P, C, 2, 2)
    d.bias_ss_1 = np.zeros((p_count, c_slots))                # (P, C)
    d.bias_ss_2 = np.zeros((p_count, c_slots))                # (P, C)

    d.global_transition_probabilities = np.zeros((p_count, c_slots))   # (P, C)
    d.global_transition_probabilities[:, 0] = 1.0
    d.global_cue_probabilities = np.ones((p_count, 1))                 # (P, 1)

    d.retention = sample_scalar_normal(
        rng,
        model.prior_mean_retention,
        float(precision_to_variance(model.prior_precision_retention)),
        (p_count, c_slots),
        0.0,
        1.0,
    )                                                     # (P, C)
    d.drift = sample_scalar_normal(
        rng,
        model.prior_mean_drift,
        float(precision_to_variance(model.prior_precision_drift)),
        (p_count, c_slots),
        -np.inf,
        np.inf,
    )                                                     # (P, C)
    if model.infer_bias:
        d.bias = sample_scalar_normal(
            rng,
            model.prior_mean_bias,
            float(precision_to_variance(model.prior_precision_bias)),
            (p_count, c_slots),
            -np.inf,
            np.inf,
        )                                                 # (P, C)
    else:
        d.bias = np.zeros((p_count, c_slots))             # (P, C)

    d.state_filtered_mean = stationary_state_mean(d.retention, d.drift)   # (P, C)
    d.state_filtered_var = stationary_state_var(
        d.retention, model.sigma_process_noise
    )                                                                    # (P, C)
    d.previous_state_filtered_mean = d.state_filtered_mean.copy()        # (P, C)
    d.previous_state_filtered_var = d.state_filtered_var.copy()          # (P, C)
    d.state_mean = d.state_filtered_mean.copy()                          # (P, C)
    d.state_var = d.state_filtered_var.copy()                            # (P, C)
    d.state_feedback_mean, d.state_feedback_var = feedback_transform(
        d.state_mean, d.state_var, d.bias, observation_variance(model)
    )                                                                    # (P, C)

    d.prior_probabilities = np.zeros((p_count, c_slots))   # (P, C)
    d.prior_probabilities[:, 0] = 1.0
    # Copies, not aliases: MATLAB struct assignment is a value copy.
    d.predicted_probabilities = d.prior_probabilities.copy()   # (P, C)
    d.responsibilities = d.prior_probabilities.copy()          # (P, C)
    d.probability_cue = np.ones((p_count, c_slots))            # (P, C)
    d.probability_state_feedback = np.ones((p_count, c_slots))  # (P, C)
    d.i_resampled = np.arange(p_count)                         # (P,) 0-based

    _finish_reset(model, d)


def reset_particles_md(model):
    """Initialise particle storage for the N-dimensional model.

    Multi-dimensional counterpart of :func:`reset_particles`. The latent state
    is an ``N``-vector per context per particle with a full ``(N, N)``
    covariance, and the dynamics are the augmented matrix ``Theta = [A | d]``
    drawn from the matrix-normal prior of :func:`dynamics_prior_md`. The
    context-inference fields keep exactly the shapes and semantics of the
    scalar model, so the shared context routines work unchanged.

    Each context is seeded at the stationary distribution implied by its own
    sampled dynamics.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to (re-)initialise in place. Uses ``model.rng``.

    Returns
    -------
    None
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    rng = model.rng

    d = ParticleState()
    d.n_active = np.ones(p_count, dtype=int)              # (P,)
    d.context = np.zeros(p_count, dtype=int)              # (P,)
    d.previous_context = np.zeros(p_count, dtype=int)     # (P,)
    d.n_cues = 0

    # --- Context-inference sufficient statistics (shared with scalar model) ---
    d.n_context = np.zeros((p_count, c_slots, c_slots))   # (P, C, C)
    d.n_cue = np.zeros((p_count, c_slots, 1))             # (P, C, Q=1)
    d.global_transition_probabilities = np.zeros((p_count, c_slots))   # (P, C)
    d.global_transition_probabilities[:, 0] = 1.0
    d.global_cue_probabilities = np.ones((p_count, 1))                 # (P, 1)

    # --- Dynamics parameter sufficient statistics (matrix accumulators) ---
    d.Lambda_xx = np.zeros((p_count, c_slots, n + 1, n + 1))   # (P, C, N+1, N+1)
    d.Lambda_yx = np.zeros((p_count, c_slots, n, n + 1))       # (P, C, N, N+1)
    d.bias_info_ss = np.zeros((p_count, c_slots, n))           # (P, C, N)
    d.bias_precision_ss = np.zeros((p_count, c_slots, n, n))   # (P, C, N, N)

    # --- Sample Theta = [A | d] from the matrix-normal prior ---
    m0, _, v0 = dynamics_prior_md(model)
    q_cov = process_noise_cov(model)
    d.Theta = np.zeros((p_count, c_slots, n, n + 1))           # (P, C, N, N+1)
    for p in range(p_count):
        for c in range(c_slots):
            d.Theta[p, c] = sample_stable_theta(rng, m0, q_cov, v0, n)

    if model.infer_bias:
        bias_var = float(precision_to_variance(model.prior_precision_bias))
        d.bias = model.prior_mean_bias + np.sqrt(bias_var) * rng.standard_normal(
            (p_count, c_slots, n)
        )                                                      # (P, C, N)
    else:
        d.bias = np.zeros((p_count, c_slots, n))               # (P, C, N)

    # --- Seed every context to its stationary state distribution ---
    d.state_filtered_mean = np.zeros((p_count, c_slots, n))        # (P, C, N)
    d.state_filtered_cov = np.zeros((p_count, c_slots, n, n))      # (P, C, N, N)
    for p in range(p_count):
        for c in range(c_slots):
            a_matrix = d.Theta[p, c, :, :n]     # (N, N)
            drift = d.Theta[p, c, :, n]         # (N,)
            d.state_filtered_mean[p, c] = stationary_state_mean_md(a_matrix, drift)
            d.state_filtered_cov[p, c] = stationary_state_cov_md(a_matrix, q_cov)

    d.previous_state_filtered_mean = d.state_filtered_mean.copy()   # (P, C, N)
    d.previous_state_filtered_cov = d.state_filtered_cov.copy()     # (P, C, N, N)
    d.state_mean = d.state_filtered_mean.copy()                     # (P, C, N)
    d.state_cov = d.state_filtered_cov.copy()                       # (P, C, N, N)

    r_cov = observation_noise_cov(model)                            # (N, N)
    # (P, C, N) mean and (P, C, N, N) covariance; r_cov broadcasts over both
    # leading axes.
    d.state_feedback_mean, d.state_feedback_cov = feedback_transform(
        d.state_mean, d.state_cov, d.bias, r_cov
    )

    # --- Context probabilities (shared shapes/semantics with scalar model) ---
    d.prior_probabilities = np.zeros((p_count, c_slots))            # (P, C)
    d.prior_probabilities[:, 0] = 1.0
    d.predicted_probabilities = d.prior_probabilities.copy()        # (P, C)
    d.responsibilities = d.prior_probabilities.copy()               # (P, C)
    d.probability_cue = np.ones((p_count, c_slots))                 # (P, C)
    d.probability_state_feedback = np.ones((p_count, c_slots))      # (P, C)
    d.i_resampled = np.arange(p_count)                              # (P,)

    _finish_reset(model, d)


# --------------------------------------------------------------------------
# Cue registry
# --------------------------------------------------------------------------


def ensure_cue_column(model, num_cues):
    """Grow the cue-indexed particle arrays to hold at least ``num_cues`` cues.

    Zero-pads the trailing cue axis of ``n_cue`` ``(P, C, Q)``,
    ``global_cue_probabilities`` ``(P, Q)`` and ``local_cue_matrix``
    ``(P, C, Q)`` so that every cue column below ``num_cues`` is addressable.
    Existing entries are preserved and new columns are zero.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is grown in place.
    num_cues : int or None
        Required number of cue COLUMNS. Note the difference from MATLAB's
        ``ensureCueColumn(q)``, which takes a 1-based cue LABEL: with 0-based
        labels here, making column ``q`` addressable means passing
        ``num_cues = q + 1``. ``None`` is a no-op.

    Returns
    -------
    None
    """
    if num_cues is None:
        return
    num_cues = int(num_cues)
    d = model.D

    # (field, cue axis): n_cue and local_cue_matrix are (P, C, Q) so the cue
    # axis is 2; global_cue_probabilities is (P, Q) so it is 1. A field still
    # None (not yet built by the reset) is skipped.
    for name, axis in (("n_cue", 2), ("global_cue_probabilities", 1),
                       ("local_cue_matrix", 2)):
        array = getattr(d, name)
        if array is None or array.shape[axis] >= num_cues:
            continue
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, num_cues - array.shape[axis])
        setattr(d, name, np.pad(array, pad_width))   # zero-pads by default


def instantiate_cue_if_needed(model, q):
    """Split off a new global cue category via one stick-breaking step.

    Grows the global cue distribution to cover a newly observed cue label ``q``
    when it exceeds the current count. One ``Beta(1, gamma_cue)`` draw per
    particle splits the mass currently on category ``q`` into a retained
    fraction ``b`` (stays on ``q``) and ``1 - b`` (moves to the fresh category
    ``q + 1``), matching the DP-based cue prior of the COIN model. No-op when
    ``q`` is ``None`` or already instantiated.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is grown in place. Uses ``model.rng``.
    q : int or None
        0-based cue label.

    Returns
    -------
    None
    """
    d = model.D
    if q is None or q + 1 <= d.n_cues:
        return
    ensure_cue_column(model, q + 2)
    p_count = model.num_particles
    b = beta_sample(
        model.rng, np.ones(p_count), model.gamma_cue * np.ones(p_count)
    )                                                                # (P,)
    mass = d.global_cue_probabilities[:, q].copy()                   # (P,)
    d.global_cue_probabilities[:, q + 1] = mass * (1.0 - b)
    d.global_cue_probabilities[:, q] = mass * b
    d.n_cues = q + 1


def consume_pending_cue(model):
    """Resolve and clear the staged cue into a 0-based integer column label.

    Reads the raw cue staged by ``observe_q`` from ``model.pending_q``, clears
    it, and maps the raw value to a 0-based integer label indexing
    ``model.cue_values``. A previously unseen raw value is appended to
    ``cue_values`` (taking the next label) and the backing cue columns are grown
    via :func:`ensure_cue_column`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose staged cue is consumed. Mutates ``pending_q``,
        ``cue_values`` and the cue-indexed particle arrays.

    Returns
    -------
    int or None
        The 0-based cue label, or ``None`` when no cue was staged.
    """
    raw = model.pending_q
    model.pending_q = None
    if raw is None:
        return None
    q = _find_cue_label(model.cue_values, raw)
    if q is None:
        model.cue_values.append(raw)
        q = len(model.cue_values) - 1
    ensure_cue_column(model, q + 1)
    return q


def peek_cue_label(model, raw):
    """0-based label a raw cue would take, WITHOUT mutating any state.

    The read-only counterpart of :func:`consume_pending_cue`, used by the
    preview queries. If ``raw`` is already registered its existing label is
    returned; if it is new, the next unused label ``len(cue_values)`` is
    returned but ``cue_values`` is NOT modified and no array is grown.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose cue registry is inspected.
    raw : float or None
        Raw cue value.

    Returns
    -------
    int or None
        The 0-based label, or ``None`` for a ``None`` input.
    """
    if raw is None:
        return None
    q = _find_cue_label(model.cue_values, raw)
    if q is None:
        return len(model.cue_values)
    return q


def _find_cue_label(cue_values, raw):
    """Index of ``raw`` in ``cue_values`` by value equality, or ``None``.

    Mirrors MATLAB's ``find(arrayfun(@(x) isequal(x, raw), cue_values), 1)``:
    the FIRST exact match wins. The mirror is exact except for ``nan``, which
    ``isequal`` treats as equal to itself and ``==`` does not; that case is
    unreachable because ``observe_q`` rejects ``nan`` before it is ever staged.
    """
    for i, value in enumerate(cue_values):
        if value == raw:
            return i
    return None

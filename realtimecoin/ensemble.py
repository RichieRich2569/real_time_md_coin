"""Multi-run averaging wrapper over :class:`~realtimecoin.model.RealTimeCOIN`.

Translated from the MATLAB class folder ``@RealTimeCOINEnsemble/`` (classdef,
``observe_q``, ``observe_y``, ``simulate``, the run-averaged queries, and the
private helpers ``averageAcrossRuns``, ``poolMoments``, ``makeMemberStream`` and
``replayMemberSequence``). The authoritative behavioural contract is
``docs/SPEC_ensemble.md``.

An ensemble orchestrates ``R`` independent ``RealTimeCOIN`` filters ("members" /
"runs") that all consume the IDENTICAL observation stream, fed one trial at a
time. Every query returns the equal-weight average across runs of the
corresponding single-model quantity - the readout of the pooled mixture that
gives each run weight ``1 / R``. This is the real-time analogue of ``COIN.m``'s
offline ``runs``: Monte-Carlo variance reduction ("probability averaging").

The wrapper does NOT change any per-trial behaviour; each member is an ordinary
:class:`~realtimecoin.model.RealTimeCOIN`.

The CONTEXT-INDEXED queries cannot be averaged slot by slot, because every
member labels its contexts in its own arbitrary frame; they first map all
members onto one reference frame through
:mod:`realtimecoin.ensemble_alignment` (SPEC Part 10), then average there.


Random-stream contract (replaces SPEC section 3's MATLAB ``RandStream`` design)
-------------------------------------------------------------------------------
MATLAB gives member ``k`` a Threefry substream and swaps it onto the GLOBAL
stream around each member step. This package draws every random number from an
explicit ``model.rng``, so no global-stream juggling is needed or wanted::

    children = numpy.random.SeedSequence(seed).spawn(runs)
    member_k = RealTimeCOIN(rng=numpy.random.default_rng(children[k]), ...)

Member ``k`` is CONSTRUCTED with its own generator, so the construction
randomness (``reset_particles``) already belongs to member ``k``'s stream, and
the member consumes only that stream ever after. The normative consequences,
which replace the MATLAB guarantees one for one, are:

a. **Reproducibility.** The same ``(seed, runs, member params)`` gives a
   bit-identical ensemble (SPEC 3.2).
b. **Executor invariance.** Results are independent of execution order, of
   ``max_cores`` and of ``segment_length``, because the stream state lives in
   the member rather than in a shared global (SPEC 3.3).
c. **Independence.** Distinct spawned children are statistically independent, so
   distinct members follow distinct trajectories, and distinct seeds give
   distinct ensembles (SPEC 3.4).
d. **No global-stream side effects.** No code path here touches the legacy
   global ``numpy.random`` state (SPEC 3.5) - it is untouched by construction,
   not by save/restore.

The single-member oracle for any averaging test is therefore ``R`` plain
``RealTimeCOIN`` instances built with
``rng=numpy.random.default_rng(numpy.random.SeedSequence(seed).spawn(runs)[k])``,
which reproduce the ensemble's members exactly.


Deliberate departures from the MATLAB original
----------------------------------------------
1. **Scalar-model shapes drop the singleton dimension**, as everywhere else in
   this package: :meth:`RealTimeCOINEnsemble.motor_output` returns a ``float``
   rather than a ``1 x 1``, densities are ``(K,)`` rather than ``1 x K``, and
   :meth:`RealTimeCOINEnsemble.simulate` returns ``(T,)`` traces rather than
   ``1 x T``. The multi-dimensional shapes are MATLAB's: ``(N, T)`` motor /
   mean traces and an ``(N, N, T)`` covariance trace.
2. **The NaN-aware mean follows the SPEC, not the MATLAB helper.**
   ``averageAcrossRuns.m`` computes ``sum(stacked .* finiteMask)``, and in
   MATLAB ``Inf * 0`` is ``NaN``, so a single infinite run poisons an entry that
   other runs report finitely. SPEC 5.4 (and both blind test suites' oracles)
   call for an omit-non-finite mean; :func:`_average_across_runs` masks with
   ``numpy.where`` instead of multiplying, so a finite run always contributes.
3. **A single contributing run is passed through verbatim** by
   :func:`_pool_moments`. Evaluating ``(v + mu ** 2) - mu ** 2`` for one run is
   algebraically the identity but not the identity in floating point, and SPEC
   5.5 requires ``runs == 1`` to equal the single member EXACTLY. The SPEC 5.2
   post-conditions (variance floored at 0, covariance symmetrised) are still
   applied, and are exact on a member's own moments.
3a. **Moment validity is per RUN, not per entry.** SPEC 5.4 phrases the
   NaN rule entrywise, but ``poolMoments.m`` (line 17) admits a run only when
   its mean AND its (co)variance are ENTIRELY finite, and this port keeps that
   behaviour so the pooled covariance stays a covariance: mixing entries from
   different subsets of runs can produce a ``second - mu mu'`` that is not
   positive semi-definite. The two rules differ only when a run's mean is
   finite while its covariance is not - a state no member can currently reach.
   The blind suites' oracles implement the entrywise reading, so a future
   change here must reconcile all three.
4. **The parallel executor is a process pool over runs**, chunked into segments
   of ``segment_length`` trials, with members crossing the process boundary as
   :meth:`~realtimecoin.model.RealTimeCOIN.snapshot` payloads (the purpose SPEC
   section 7 gives them). As in MATLAB, only :meth:`RealTimeCOINEnsemble.simulate`
   is parallelised; the live ``observe_y`` path always fans out serially, since
   pickling ``R`` members per trial costs far more than the per-trial update.
   A caller using ``max_cores > 0`` must guard its script entry point with
   ``if __name__ == "__main__":`` - see
   :meth:`RealTimeCOINEnsemble.simulate`.
"""

from __future__ import annotations

import concurrent.futures as _futures
import copy as _copy
import warnings
from dataclasses import dataclass

import numpy as np

from .ensemble_alignment import (
    ensemble_context_alignment as _ensemble_context_alignment,
    ensemble_context_density as _ensemble_context_density,
    ensemble_context_vector as _ensemble_context_vector,
)
from .exceptions import ModelFormatError, RealTimeCOINError
from .model import RealTimeCOIN

__all__ = [
    "RealTimeCOINEnsemble",
    "SimulationTraces",
    "EnsembleLockstepError",
    "ENSEMBLE_FORMAT_VERSION",
]

#: Schema name written into (and demanded by) every ensemble snapshot.
ENSEMBLE_FORMAT_VERSION = "rtcoin-ensemble-v1"


class EnsembleLockstepError(RealTimeCOINError):
    """Raised when the members are no longer on a common trial.

    SPEC 2.2 guarantees ``ens.Trial == member_k.Trial`` for every ``k``; the
    only way to break it is an exception escaping part-way through a fan-out,
    which leaves some members stepped and some not. The ensemble refuses to
    keep operating on a desynchronised member set rather than silently
    averaging quantities from different trials.
    """


@dataclass
class SimulationTraces:
    """Per-trial run-averaged traces returned by :meth:`RealTimeCOINEnsemble.simulate`.

    The MATLAB counterpart is the ``traces`` struct; attribute access reads the
    same (``traces.motor_output``).

    Attributes
    ----------
    motor_output : numpy.ndarray
        ``(T,)`` for the scalar model, ``(N, T)`` otherwise. Column ``t`` is
        :meth:`RealTimeCOINEnsemble.motor_output` after trial ``t``.
    state_mean : numpy.ndarray
        ``(T,)`` / ``(N, T)``; the ``mu`` of
        :meth:`RealTimeCOINEnsemble.state_moments` after trial ``t``.
    state_var : numpy.ndarray
        ``(T,)`` for the scalar model, ``(N, N, T)`` otherwise; the ``v`` of
        :meth:`RealTimeCOINEnsemble.state_moments` after trial ``t``.
    Trial : numpy.ndarray
        ``(T,)`` integer trial numbers, equal to ``1 .. T``.
    """

    motor_output: np.ndarray
    state_mean: np.ndarray
    state_var: np.ndarray
    Trial: np.ndarray


# --------------------------------------------------------------------------
# Averaging primitives (private/averageAcrossRuns.m, private/poolMoments.m)
# --------------------------------------------------------------------------


def _average_across_runs(values):
    """NaN-aware equal-weight mean of per-run arrays (SPEC 5.4).

    Each output entry is the mean over the runs for which that entry is
    FINITE; an entry that is non-finite (``nan`` or ``+-inf``) in every run
    comes back as ``nan``. This is ``COIN.weighted_sum_along_dimension``'s
    ``omitnan``-with-all-NaN-gives-NaN rule.

    Parameters
    ----------
    values : sequence of array_like
        One identically-shaped array (or scalar) per run.

    Returns
    -------
    float or numpy.ndarray
        The elementwise mean, as a ``float`` when the per-run values are
        scalars.

    Notes
    -----
    Non-finite entries are dropped with :func:`numpy.where`, NOT by multiplying
    by a 0/1 mask as ``averageAcrossRuns.m`` does: ``inf * 0`` is ``nan``, so
    the MATLAB form propagates a single infinite run into an entry every other
    run reports finitely.
    """
    if len(values) == 0:
        return np.asarray([], dtype=float)

    stacked = np.stack(
        [np.asarray(v, dtype=float) for v in values], axis=-1
    )                                                          # (..., R)
    finite = np.isfinite(stacked)                              # (..., R)
    count = finite.sum(axis=-1)                                # (...)
    total = np.where(finite, stacked, 0.0).sum(axis=-1)        # (...)

    out = np.full(np.shape(count), np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return float(out) if out.ndim == 0 else out


def _pool_moments(mus, vs, state_dim):
    """Moments of the equal-weight pooled mixture over runs (SPEC 5.2).

    The law of total (co)variance, NOT a naive mean of the per-run
    covariances::

        mu = (1 / R) sum_k mu_k
        v  = (1 / R) sum_k ( v_k + mu_k mu_k' )  -  mu mu'

    A run contributes only if BOTH its mean and its covariance are entirely
    finite (run-granular NaN awareness, matching ``poolMoments.m``); if no run
    is valid the result is all ``nan``.

    Parameters
    ----------
    mus : sequence
        Per-run means: scalars when ``state_dim == 1``, else ``(N,)`` vectors.
    vs : sequence
        Per-run (co)variances: scalars when ``state_dim == 1``, else ``(N, N)``.
    state_dim : int
        Latent dimension ``N``.

    Returns
    -------
    mu : float or numpy.ndarray
        Pooled mean; ``float`` for the scalar model, else ``(N,)``.
    v : float or numpy.ndarray
        Pooled variance (floored at 0) or symmetric ``(N, N)`` covariance.

    Notes
    -----
    With exactly ONE contributing run the pair is returned verbatim rather than
    round-tripped through the formula. The two agree algebraically, but
    ``(v + mu ** 2) - mu ** 2`` loses low bits whenever ``mu ** 2`` dominates
    ``v``, and SPEC 5.5 requires the ``runs == 1`` ensemble to equal its single
    member exactly.
    """
    n = int(state_dim)
    mu_arrays = [np.asarray(m, dtype=float).reshape(-1) for m in mus]   # R x (N,)
    v_arrays = [np.asarray(v, dtype=float) for v in vs]                 # R x (N,N)
    valid = [
        k
        for k in range(len(mu_arrays))
        if np.all(np.isfinite(mu_arrays[k])) and np.all(np.isfinite(v_arrays[k]))
    ]

    if not valid:
        if n == 1:
            return float("nan"), float("nan")
        return np.full(n, np.nan), np.full((n, n), np.nan)

    if len(valid) == 1:
        # Exact pass-through: the pooled mixture of one component IS that
        # component (SPEC 5.5). See the note above. The SPEC 5.2
        # post-conditions still apply - and both are exact identities on a
        # member's own moments (a member variance is already non-negative, a
        # member covariance already symmetric), so the reduction stays
        # bit-exact while the invariants hold for R == 1 as well as R > 1.
        k = valid[0]
        if n == 1:
            return float(mu_arrays[k][0]), float(np.fmax(vs[k], 0.0))
        v_k = np.reshape(v_arrays[k], (n, n))                           # (N, N)
        return mu_arrays[k].copy(), (v_k + v_k.T) / 2.0

    mu = np.zeros(n)                                                    # (N,)
    second = np.zeros((n, n))                                           # (N, N)
    for k in valid:
        m_k = mu_arrays[k]                                              # (N,)
        mu += m_k
        second += np.reshape(v_arrays[k], (n, n)) + np.outer(m_k, m_k)
    mu /= len(valid)
    second /= len(valid)
    v = second - np.outer(mu, mu)                                       # (N, N)

    if n == 1:
        # fmax, not maximum: MATLAB's max(x, 0) drops a NaN in favour of 0.
        return float(mu[0]), float(np.fmax(v[0, 0], 0.0))
    return mu, (v + v.T) / 2.0


# --------------------------------------------------------------------------
# Sequence / cue normalisation
# --------------------------------------------------------------------------


def _normalise_cue(q):
    """Coerce a cue argument to ``float`` or ``None`` (cue-free).

    MATLAB's ``[]`` and ``NaN`` both mark a cue-free trial; so do Python's
    ``None``, an empty array and ``nan``.

    Parameters
    ----------
    q : float or array_like or None
        Raw scalar cue identifier.

    Returns
    -------
    float or None
        The cue value, or ``None`` for a cue-free trial.

    Raises
    ------
    ValueError
        If ``q`` holds more than one element.
    """
    if q is None:
        return None
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size != 1:
        raise ValueError(
            "RealTimeCOINEnsemble:observe_q: q must be a scalar cue value or "
            "empty, got %d elements." % (arr.size,)
        )
    value = float(arr[0])
    return None if np.isnan(value) else value


def _normalise_sequences(q_seq, y_seq, state_dim):
    """Validate a batch observation sequence and resolve its length.

    Mirrors ``simulate.m``'s local ``sequenceLength``, including its two error
    conditions.

    Parameters
    ----------
    q_seq : array_like or None
        ``(T,)`` cue row (``nan`` for cue-free trials), or ``None`` / empty for
        an all-cue-free sequence.
    y_seq : array_like or None
        ``(N, T)`` feedback with ONE TRIAL PER COLUMN (``(T,)`` is accepted for
        the scalar model), or ``None`` / empty for all-missing feedback.
        Individual ``nan`` entries mark missing / partially-observed feedback.
    state_dim : int
        Latent dimension ``N``, which ``y_seq`` must have rows for.

    Returns
    -------
    cues : numpy.ndarray or None
        ``(T,)`` float cue row, or ``None``.
    obs : numpy.ndarray or None
        ``(N, T)`` float feedback, or ``None``.
    length : int
        Number of trials ``T``.

    Raises
    ------
    ValueError
        If ``y_seq`` does not have ``state_dim`` rows, or if ``q_seq`` and
        ``y_seq`` imply different lengths.
    """
    cues = None
    if q_seq is not None:
        cues = np.asarray(q_seq, dtype=float)
        # Rejected rather than flattened: MATLAB's numel() flattens a matrix
        # column-major and numpy row-major, so the same 2-D input would order
        # the trials differently in the two languages.
        if cues.ndim > 1:
            raise ValueError(
                "RealTimeCOINEnsemble:simulate:qSize: q_seq must be a 1-D cue "
                "row, got %d dimensions." % (cues.ndim,)
            )
        if cues.size == 0:
            cues = None

    obs = None
    if y_seq is not None:
        obs = np.asarray(y_seq, dtype=float)
        if obs.size == 0:
            obs = None
        else:
            if obs.ndim == 1:
                # A bare row is a scalar-model sequence; for N > 1 it is
                # ambiguous (N values of one trial, or T values of one
                # dimension), so it is rejected below by the row check.
                obs = obs.reshape(1, -1)                                # (1, T)
            elif obs.ndim != 2:
                raise ValueError(
                    "RealTimeCOINEnsemble:simulate:ySize: y_seq must be "
                    "1- or 2-dimensional, got %d dimensions." % (obs.ndim,)
                )
            if obs.shape[0] != state_dim:
                raise ValueError(
                    "RealTimeCOINEnsemble:simulate:ySize: y_seq must have "
                    "state_dim (%d) rows, got %d." % (state_dim, obs.shape[0])
                )

    if cues is not None and obs is not None and cues.size != obs.shape[1]:
        raise ValueError(
            "RealTimeCOINEnsemble:simulate:lengthMismatch: q_seq (%d) and "
            "y_seq (%d) imply different sequence lengths."
            % (cues.size, obs.shape[1])
        )

    if obs is not None:
        length = int(obs.shape[1])
    elif cues is not None:
        length = int(cues.size)
    else:
        length = 0
    return cues, obs, length


def _cue_at(cues, t):
    """Cue for trial ``t``: ``None`` when the sequence is cue-free."""
    if cues is None:
        return None
    return _normalise_cue(cues[t])


def _obs_at(obs, t):
    """Feedback column for trial ``t``: ``None`` when the sequence is empty."""
    if obs is None:
        return None
    return obs[:, t]                                                    # (N,)


# --------------------------------------------------------------------------
# Replay machinery, shared by the serial and the process-pool executors
# --------------------------------------------------------------------------


def _record_segment(model, cues, obs, offset, length, state_dim):
    """Step one member through ``length`` trials, recording the queries.

    Parameters
    ----------
    model : RealTimeCOIN
        Member, stepped in place.
    cues : numpy.ndarray or None
        Full ``(T,)`` cue row.
    obs : numpy.ndarray or None
        Full ``(N, T)`` feedback.
    offset : int
        Index of the segment's first trial within the full sequence.
    length : int
        Number of trials in this segment.
    state_dim : int
        Latent dimension ``N``.

    Returns
    -------
    motor : numpy.ndarray
        ``(N, length)`` per-trial ``motor_output``.
    mean : numpy.ndarray
        ``(N, length)`` per-trial state mean.
    var : numpy.ndarray
        ``(length,)`` for ``N == 1``, else ``(N, N, length)`` per-trial
        (co)variance.
    """
    motor = np.full((state_dim, length), np.nan)                   # (N, L)
    mean = np.full((state_dim, length), np.nan)                    # (N, L)
    if state_dim == 1:
        var = np.full(length, np.nan)                              # (L,)
    else:
        var = np.full((state_dim, state_dim, length), np.nan)      # (N, N, L)

    for i in range(length):
        t = offset + i
        model.observe_q(_cue_at(cues, t))
        model.observe_y(_obs_at(obs, t))
        motor[:, i] = np.asarray(model.motor_output(), dtype=float).reshape(-1)
        mu, v = model.state_moments()
        mean[:, i] = np.asarray(mu, dtype=float).reshape(-1)
        if state_dim == 1:
            var[i] = float(v)
        else:
            var[:, :, i] = v
    return motor, mean, var


def _replay_segment_worker(payload):
    """Process-pool entry point: rebuild a member, replay one segment.

    Module level (not a closure or a bound method) so it survives pickling
    under the ``spawn`` start method Windows uses. The member crosses the
    process boundary as a snapshot - the use SPEC section 7 introduces
    ``snapshot`` / ``load_snapshot`` for - which restores the particle state
    AND the member's random-stream position exactly, so a segment replayed in a
    worker is bit-identical to the same segment replayed in-process.

    Parameters
    ----------
    payload : tuple
        ``(member_kwargs, snapshot, cues, obs, offset, length, state_dim)``.

    Returns
    -------
    tuple
        ``(snapshot_out, motor, mean, var)`` - the member state at the end of
        the segment plus the traces from :func:`_record_segment`.
    """
    member_kwargs, snap, cues, obs, offset, length, state_dim = payload
    model = RealTimeCOIN(rng=0, **member_kwargs)
    model.load_snapshot(snap)
    traces = _record_segment(model, cues, obs, offset, length, state_dim)
    return (model.snapshot(),) + traces


def _pool_probe():
    """Trivial task proving a pool worker can start (see ``_start_pool``)."""
    return True


def _segment_bounds(length, segment_length):
    """Yield ``(offset, size)`` pairs partitioning ``length`` trials."""
    step = max(int(segment_length), 1)
    for offset in range(0, length, step):
        yield offset, min(step, length - offset)


# --------------------------------------------------------------------------
# The ensemble
# --------------------------------------------------------------------------


class RealTimeCOINEnsemble:
    """Equal-weight ensemble of ``R`` independent :class:`RealTimeCOIN` runs.

    All members consume the identical observation stream, one trial at a time;
    every query returns the average across runs of the corresponding
    single-model quantity. See the module docstring for the random-stream
    contract and ``docs/SPEC_ensemble.md`` for the full behavioural contract.

    Parameters
    ----------
    runs : int, optional
        Number ``R`` of independent member filters. Positive. Default 1.
    seed : int, optional
        Base seed for the whole ensemble; member ``k`` uses spawned child ``k``
        of ``numpy.random.SeedSequence(seed)``. Non-negative. Default 0.
    max_cores : int, optional
        Worker cap for :meth:`simulate`: ``0`` selects the serial executor,
        ``> 0`` a process pool with that many workers. Non-negative. Default 0.
        It is a scheduling choice only - results are identical either way.
    segment_length : int, optional
        Number of trials replayed per parallel dispatch. Positive. Default 1.
        Scheduling only; it cannot change a result.
    **member_kwargs
        Every other keyword is forwarded VERBATIM and IDENTICALLY to each
        member's :class:`~realtimecoin.model.RealTimeCOIN` constructor
        (``state_dim``, ``num_particles``, ``max_contexts``, ...). An
        unrecognised name raises the member constructor's
        :class:`~realtimecoin.exceptions.NameValuePairsError`.

    Attributes
    ----------
    Trial : int
        Read-only common trial counter; equals every member's ``Trial``.

    Raises
    ------
    ValueError
        If an ensemble parameter fails its validator, or if ``rng`` is passed
        as a member keyword (``seed`` governs all randomness here).

    Examples
    --------
    >>> from realtimecoin import RealTimeCOINEnsemble
    >>> ens = RealTimeCOINEnsemble(runs=4, seed=7, num_particles=20)
    >>> ens.observe_q(1)
    >>> ens.observe_y(0.1)
    >>> ens.Trial
    1
    """

    def __init__(self, *, runs=1, seed=0, max_cores=0, segment_length=1,
                 **member_kwargs):
        if "rng" in member_kwargs:
            raise ValueError(
                "RealTimeCOINEnsemble:RngNotAccepted: member randomness is "
                "governed by 'seed'; pass seed=<int> instead of rng=... so "
                "every member gets its own reproducible sub-stream."
            )

        self._runs = _positive_integer(runs, "runs")
        self._seed = _nonnegative_integer(seed, "seed")
        self._max_cores = _nonnegative_integer(max_cores, "max_cores")
        self._segment_length = _positive_integer(segment_length, "segment_length")
        self._member_kwargs = dict(member_kwargs)

        self._weights = np.full(self._runs, 1.0 / self._runs)           # (R,)
        self._weights.flags.writeable = False   # read-only property, not a copy

        self._members = self._build_members()
        self._state_dim = int(self._members[0].state_dim)
        self._trial = 0
        self._desynced = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _member_seeds(self):
        """Spawn the per-member seed sequences (a pure function of the seed).

        Returns
        -------
        list of numpy.random.SeedSequence
            One child per member. ``SeedSequence(seed).spawn(runs)`` is
            deterministic, so a fresh call reproduces the same children - which
            is what lets :meth:`simulate` rebuild members identical to the live
            ones.
        """
        return np.random.SeedSequence(self._seed).spawn(self._runs)

    def _build_members(self):
        """Construct ``R`` members, each under its own generator.

        Returns
        -------
        list of RealTimeCOIN
            Fresh members. Because the generator is handed to the CONSTRUCTOR,
            the construction randomness (``reset_particles``) is already drawn
            from member ``k``'s stream.
        """
        return [
            RealTimeCOIN(rng=np.random.default_rng(child), **self._member_kwargs)
            for child in self._member_seeds()
        ]

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def runs(self):
        """int: Number ``R`` of member filters (read-only)."""
        return self._runs

    @property
    def seed(self):
        """int: Base RNG seed of the ensemble (read-only)."""
        return self._seed

    @property
    def max_cores(self):
        """int: Worker cap for :meth:`simulate`; 0 is serial (read-only)."""
        return self._max_cores

    @property
    def segment_length(self):
        """int: Trials per parallel dispatch (read-only)."""
        return self._segment_length

    @property
    def weights(self):
        """numpy.ndarray: ``(R,)`` uniform run weights ``1 / R`` (read-only)."""
        return self._weights

    @property
    def state_dim(self):
        """int: Member latent dimension ``N`` (read-only)."""
        return self._state_dim

    @property
    def members(self):
        """tuple: The member models, as a read-only view (they are live objects)."""
        return tuple(self._members)

    @property
    def Trial(self):
        """int: Common trial counter; equals every member's ``Trial``."""
        return self._trial

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _check_lockstep(self):
        """Raise if the members are not all on the ensemble's trial.

        Returns
        -------
        None

        Raises
        ------
        EnsembleLockstepError
            If a previous fan-out failed part-way, or a member has been stepped
            behind the ensemble's back.
        """
        if self._desynced:
            raise EnsembleLockstepError(
                "RealTimeCOINEnsemble:Desynchronised: a previous observe_y "
                "failed part-way through the members, so they are no longer on "
                "a common trial. Rebuild the ensemble (or restore a snapshot) "
                "before continuing."
            )
        stray = [
            k for k, m in enumerate(self._members) if m.Trial != self._trial
        ]
        if stray:
            self._desynced = True
            raise EnsembleLockstepError(
                "RealTimeCOINEnsemble:Desynchronised: member(s) %s are not on "
                "the ensemble trial %d (SPEC 2.2 requires ens.Trial == "
                "member_k.Trial for every k)." % (stray, self._trial)
            )

    def observe_q(self, q=None):
        """Stage the cue ``q`` for the upcoming trial on every member.

        Forwards the IDENTICAL cue to each member (SPEC 4.1). Draws no
        randomness and does not advance the trial.

        Parameters
        ----------
        q : float or array_like or None, optional
            Raw scalar cue identifier. ``None``, an empty array and ``nan`` all
            mark a cue-free trial.

        Returns
        -------
        None
        """
        self._check_lockstep()
        value = _normalise_cue(q)
        for member in self._members:
            member.observe_q(value)

    def observe_y(self, y=None):
        """Feed the feedback ``y`` to every member, advancing one trial.

        Forwards the IDENTICAL feedback to each member in index order, each
        drawing from its own stream, then advances the common trial counter
        (SPEC 4.2). The fan-out is always serial: ``max_cores`` schedules
        :meth:`simulate`, not the live path.

        Parameters
        ----------
        y : float or array_like or None, optional
            Feedback: a scalar for the scalar model, else a length-``N``
            vector. ``None``, an empty array and ``nan`` mark a missing
            observation (the trial still advances); individual ``nan`` entries
            of a vector mark partially-observed dimensions.

        Returns
        -------
        None

        Raises
        ------
        EnsembleLockstepError
            If the members are not on a common trial.
        """
        self._check_lockstep()
        stepped = 0
        try:
            for member in self._members:
                member.observe_y(y)
                stepped += 1
        except Exception:
            # Some members advanced and some did not: every averaged query
            # from here on would mix trials, so refuse rather than mislead.
            if 0 < stepped < self._runs:
                self._desynced = True
            raise
        self._trial += 1

    # ------------------------------------------------------------------
    # Averaged queries (Phase 1)
    # ------------------------------------------------------------------

    def motor_output(self):
        """Run-averaged predicted state feedback.

        ``u = (1 / R) sum_k motor_output(member_k)``, NaN-aware (SPEC 5.1).

        Returns
        -------
        float or numpy.ndarray
            A ``float`` for ``state_dim == 1``, otherwise ``(N,)``.
        """
        self._check_lockstep()
        return _average_across_runs([m.motor_output() for m in self._members])

    def state_moments(self):
        """Moments of the run-pooled predictive-state mixture.

        The law-of-total-(co)variance combination of the per-run moments, NOT a
        mean of the per-run covariances (SPEC 5.2)::

            mu = (1 / R) sum_k mu_k
            v  = (1 / R) sum_k ( v_k + mu_k mu_k' )  -  mu mu'

        Returns
        -------
        mu : float or numpy.ndarray
            Scalar mean, or an ``(N,)`` mean vector.
        v : float or numpy.ndarray
            Scalar variance (floored at 0), or the symmetric ``(N, N)``
            covariance.
        """
        self._check_lockstep()
        moments = [m.state_moments() for m in self._members]
        return _pool_moments(
            [mu for mu, _ in moments], [v for _, v in moments], self._state_dim
        )

    def state_probability(self, values):
        """Run-averaged posterior latent-state density on a grid.

        The mean of the per-run densities, which IS the density of the pooled
        ``1 / R``-weighted mixture (SPEC 5.3).

        Parameters
        ----------
        values : array_like
            ``(K,)`` grid for the scalar model, or ``(K, N)`` with one query
            point per ROW - the single-model shape rules exactly.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        self._check_lockstep()
        return _average_across_runs(
            [m.state_probability(values) for m in self._members]
        )

    def state_feedback_probability(self, values):
        """Run-averaged predictive feedback density on a grid (SPEC 5.3).

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        self._check_lockstep()
        return _average_across_runs(
            [m.state_feedback_probability(values) for m in self._members]
        )

    def novel_state_probability(self, values):
        """Run-averaged novel-context state density on a grid (SPEC 5.3).

        A member that has exhausted its context budget has an all-zero novel
        density; those zeros are a valid finite contribution to the average,
        not a missing value (SPEC section 8).

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        self._check_lockstep()
        return _average_across_runs(
            [m.novel_state_probability(values) for m in self._members]
        )

    def novel_state_feedback_probability(self, values):
        """Run-averaged novel-context feedback density on a grid (SPEC 5.3).

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        numpy.ndarray
            ``(K,)`` densities.
        """
        self._check_lockstep()
        return _average_across_runs(
            [m.novel_state_feedback_probability(values) for m in self._members]
        )

    # ------------------------------------------------------------------
    # Batch replay
    # ------------------------------------------------------------------

    def simulate(self, q_seq=None, y_seq=None):
        """Batch-replay a full observation sequence across the runs.

        The offline analogue of ``COIN.simulate_COIN`` and the parallel
        throughput path. Numerically identical to constructing a fresh ensemble
        with the same ``(seed, runs, member params)`` and stepping it through
        the sequence, reading the queries after each trial (SPEC section 6) -
        and identical for any ``max_cores`` / ``segment_length`` (SPEC 3.3).

        A call is a one-shot batch on a FRESH member set seeded from this
        ensemble: it does not disturb the live stepping state, and repeated
        calls return identical traces.

        Parameters
        ----------
        q_seq : array_like or None, optional
            ``(T,)`` cue row (``nan`` for cue-free trials); ``None`` or empty
            makes every trial cue-free.
        y_seq : array_like or None, optional
            ``(N, T)`` feedback with ONE TRIAL PER COLUMN (``(T,)`` accepted
            for the scalar model). ``nan`` entries mark missing /
            partially-observed feedback, exactly as for :meth:`observe_y`.

        Returns
        -------
        SimulationTraces
            Per-trial run-averaged traces; see that class for the shapes.

        Raises
        ------
        ValueError
            If ``y_seq`` does not have ``state_dim`` rows, or ``q_seq`` and
            ``y_seq`` imply different lengths.

        Notes
        -----
        On platforms whose default process start method is ``spawn`` (Windows
        and macOS), ``max_cores > 0`` means each worker RE-IMPORTS the module
        that started the program. Guard the entry point of any script that
        calls this with ``max_cores > 0``::

            if __name__ == "__main__":
                traces = ens.simulate(q_seq, y_seq)

        Without the guard every worker re-runs the whole script before it
        serves a single task - results stay correct, but the "parallel" run is
        far slower than the serial one. This is the standard
        :mod:`multiprocessing` requirement, not a property of this class;
        importing :mod:`realtimecoin` has no side effects.
        """
        cues, obs, length = _normalise_sequences(q_seq, y_seq, self._state_dim)
        n = self._state_dim

        if self._max_cores > 0 and length > 0:
            per_run = self._replay_parallel(cues, obs, length)
        else:
            per_run = self._replay_serial(cues, obs, length)

        motor = _average_across_runs([r[0] for r in per_run])           # (N, T)
        mean, var = self._pool_moment_traces(per_run, length)

        if n == 1:
            motor = np.asarray(motor, dtype=float).reshape(-1)          # (T,)
            mean = mean.reshape(-1)                                     # (T,)
        return SimulationTraces(
            motor_output=motor,
            state_mean=mean,
            state_var=var,
            Trial=np.arange(1, length + 1),
        )

    def _replay_serial(self, cues, obs, length):
        """Replay every member over the whole sequence, in this process.

        Parameters
        ----------
        cues : numpy.ndarray or None
            ``(T,)`` cue row.
        obs : numpy.ndarray or None
            ``(N, T)`` feedback.
        length : int
            Trial count ``T``.

        Returns
        -------
        list of tuple
            One ``(motor, mean, var)`` triple per member, as returned by
            :func:`_record_segment`.
        """
        return [
            _record_segment(member, cues, obs, 0, length, self._state_dim)
            for member in self._build_members()
        ]

    def _start_pool(self):
        """Start a process pool and prove a worker can actually run in it.

        Creating a :class:`~concurrent.futures.ProcessPoolExecutor` starts no
        worker, so the failures that matter surface only on the first submit:
        under the ``spawn`` start method an unguarded ``__main__`` raises
        ``RuntimeError`` from ``multiprocessing.spawn``, and a sandbox that
        forbids subprocesses raises ``OSError`` or ``BrokenProcessPool``
        (itself a ``RuntimeError``). A trivial probe task therefore runs here,
        inside the fallback's reach, so that the real replay can run outside it
        with its errors intact.

        Returns
        -------
        concurrent.futures.ProcessPoolExecutor or None
            A working pool, or ``None`` if one could not be started (a warning
            has then been issued and the caller should replay serially).
        """
        pool = None
        try:
            pool = _futures.ProcessPoolExecutor(max_workers=self._max_cores)
            pool.submit(_pool_probe).result()
        except (OSError, RuntimeError, NotImplementedError, ImportError) as exc:
            if pool is not None:
                pool.shutdown(wait=False)
            warnings.warn(
                "RealTimeCOINEnsemble: could not start a process pool (%s); "
                "falling back to the serial executor. Results are unchanged "
                "(SPEC 3.3); only throughput is affected. On Windows and macOS "
                "a script calling simulate() with max_cores > 0 must guard its "
                "entry point with `if __name__ == \"__main__\":`."
                % (exc,),
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return pool

    def _replay_parallel(self, cues, obs, length):
        """Replay every member in a process pool, segment by segment.

        Runs are the unit of parallelism (as in ``simulate.m``'s ``parfor``);
        ``segment_length`` sets how many trials each dispatch covers, and the
        member state crosses the boundary as a snapshot. Because each member
        carries its own generator, the result is bit-identical to
        :meth:`_replay_serial` whatever the worker count or chunking.

        Parameters
        ----------
        cues : numpy.ndarray or None
            ``(T,)`` cue row.
        obs : numpy.ndarray or None
            ``(N, T)`` feedback.
        length : int
            Trial count ``T``.

        Returns
        -------
        list of tuple
            One ``(motor, mean, var)`` triple per member.

        Warns
        -----
        RuntimeWarning
            If a process pool cannot be STARTED in this environment; the replay
            then falls back to the serial executor, which returns the SAME
            numbers (SPEC 3.3), so only throughput is affected. Once the pool
            has proved itself, any later error propagates - a failure inside
            the replay is never downgraded to a warning.
        """
        n = self._state_dim
        pool = self._start_pool()
        if pool is None:
            return self._replay_serial(cues, obs, length)

        snapshots = [member.snapshot() for member in self._build_members()]
        chunks = {k: [] for k in range(self._runs)}

        # Deliberately OUTSIDE the fallback's try: from here on an exception is
        # a real failure of the replay, and must surface as itself rather than
        # be silently re-run serially.
        with pool:
            for offset, size in _segment_bounds(length, self._segment_length):
                futures = {
                    pool.submit(
                        _replay_segment_worker,
                        (self._member_kwargs, snapshots[k], cues, obs,
                         offset, size, n),
                    ): k
                    for k in range(self._runs)
                }
                # Collected by member index, never by completion order, so the
                # assembled traces do not depend on the scheduler.
                for future, k in futures.items():
                    snapshots[k], motor, mean, var = future.result()
                    chunks[k].append((motor, mean, var))

        per_run = []
        for k in range(self._runs):
            motors = np.concatenate([c[0] for c in chunks[k]], axis=1)  # (N, T)
            means = np.concatenate([c[1] for c in chunks[k]], axis=1)   # (N, T)
            var_axis = 0 if n == 1 else 2
            variances = np.concatenate([c[2] for c in chunks[k]], axis=var_axis)
            per_run.append((motors, means, variances))
        return per_run

    def _pool_moment_traces(self, per_run, length):
        """Pool the per-run moment traces trial by trial (``poolMomentTraces``).

        Parameters
        ----------
        per_run : list of tuple
            One ``(motor, mean, var)`` triple per member.
        length : int
            Trial count ``T``.

        Returns
        -------
        mean : numpy.ndarray
            ``(N, T)`` pooled mean trace.
        var : numpy.ndarray
            ``(T,)`` for ``N == 1``, else ``(N, N, T)`` pooled covariance trace.
        """
        n = self._state_dim
        mean = np.full((n, length), np.nan)                             # (N, T)
        var = (
            np.full(length, np.nan) if n == 1
            else np.full((n, n, length), np.nan)
        )
        for t in range(length):
            mus = [run[1][:, t] for run in per_run]                     # R x (N,)
            if n == 1:
                vs = [run[2][t] for run in per_run]
            else:
                vs = [run[2][:, :, t] for run in per_run]
            mu_t, v_t = _pool_moments(mus, vs, n)
            mean[:, t] = np.asarray(mu_t, dtype=float).reshape(-1)
            if n == 1:
                var[t] = v_t
            else:
                var[:, :, t] = v_t
        return mean, var

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot(self):
        """Capture the full ensemble state as a plain, serialisable mapping.

        Holds the ensemble parameters, the trial counter, the forwarded member
        parameters and one member snapshot per run (each of which carries that
        member's random-stream position). Restoring it with
        :meth:`load_snapshot` reproduces every query output exactly and
        continues the run identically.

        Returns
        -------
        dict
            Deep copy of the ensemble state; shares no buffer with this object.
        """
        return {
            "format": ENSEMBLE_FORMAT_VERSION,
            "runs": self._runs,
            "seed": self._seed,
            "max_cores": self._max_cores,
            "segment_length": self._segment_length,
            "trial": self._trial,
            "desynced": self._desynced,
            "member_kwargs": _copy.deepcopy(self._member_kwargs),
            "members": [member.snapshot() for member in self._members],
        }

    def load_snapshot(self, s):
        """Restore full ensemble state in place from a snapshot mapping.

        Rebuilds the member list when the snapshot holds a different number of
        runs, so an ensemble of any size can load any snapshot.

        Parameters
        ----------
        s : dict
            Mapping produced by :meth:`snapshot`.

        Returns
        -------
        None

        Raises
        ------
        ModelFormatError
            If ``s`` is not a mapping, declares an unknown format, or is
            missing a required key.
        """
        if not isinstance(s, dict):
            raise ModelFormatError(
                "Ensemble snapshot must be a mapping, got %r."
                % (type(s).__name__,)
            )
        if s.get("format") != ENSEMBLE_FORMAT_VERSION:
            raise ModelFormatError(
                "Ensemble snapshot has format %r; this build reads %r."
                % (s.get("format"), ENSEMBLE_FORMAT_VERSION)
            )
        for required in ("runs", "seed", "trial", "member_kwargs", "members"):
            if required not in s:
                raise ModelFormatError(
                    "Ensemble snapshot is missing the required key %r."
                    % (required,)
                )
        member_states = s["members"]
        if not isinstance(member_states, list) or len(member_states) != s["runs"]:
            raise ModelFormatError(
                "Ensemble snapshot promises %r runs but carries %r member "
                "states." % (s["runs"], len(member_states)
                             if isinstance(member_states, list) else None)
            )

        self._runs = _positive_integer(s["runs"], "runs")
        self._seed = _nonnegative_integer(s["seed"], "seed")
        self._max_cores = _nonnegative_integer(
            s.get("max_cores", self._max_cores), "max_cores"
        )
        self._segment_length = _positive_integer(
            s.get("segment_length", self._segment_length), "segment_length"
        )
        self._member_kwargs = _copy.deepcopy(s["member_kwargs"])
        self._weights = np.full(self._runs, 1.0 / self._runs)           # (R,)
        self._weights.flags.writeable = False

        # Fresh members are built with rng=0 and then fully overwritten by the
        # snapshot (which restores each member's stream position), so the
        # construction generator has no influence on anything.
        self._members = [
            RealTimeCOIN(rng=0, **self._member_kwargs)
            for _ in range(self._runs)
        ]
        for member, member_state in zip(self._members, member_states):
            member.load_snapshot(member_state)

        self._state_dim = int(self._members[0].state_dim)
        self._trial = int(s["trial"])
        self._desynced = bool(s.get("desynced", False))

    # ------------------------------------------------------------------
    # Phase 2 - context-indexed queries, in the cross-run reference frame
    # (see realtimecoin.ensemble_alignment and SPEC_ensemble Part 10)
    # ------------------------------------------------------------------

    def responsibilities_vector(self):
        """Cross-run averaged posterior context probabilities (SPEC 10.4).

        Zero-fill average in the common reference frame: real mass in slots
        ``0 .. Kref - 1``, novel mass in slot ``Kref``, the rest zero.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` row in the common reference frame, summing
            to 1.
        """
        self._check_lockstep()
        return _ensemble_context_vector(
            self, lambda m: m.responsibilities_vector()
        )

    def predicted_context_probabilities_vector(self):
        """Cross-run averaged predicted context probabilities (SPEC 10.4).

        The pre-observation counterpart of :meth:`responsibilities_vector`.

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` row in the common reference frame.
        """
        self._check_lockstep()
        return _ensemble_context_vector(
            self, lambda m: m.predicted_context_probabilities_vector()
        )

    def sampled_context_count(self):
        """Cross-run averaged sampled-context proportions (SPEC 10.4).

        Returns
        -------
        numpy.ndarray
            ``(max_contexts + 1,)`` row in the common reference frame.
        """
        self._check_lockstep()
        return _ensemble_context_vector(
            self, lambda m: m.sampled_context_count()
        )

    def stationary_context_probabilities(self):
        """Cross-run averaged stationary context distribution (SPEC 10.4).

        Each member's ``(K_r,)`` stationary row is zero-filled onto the
        reference contexts through the cross-run matching, summed over runs and
        renormalised to sum to 1 (uniform if the accumulator is empty).

        Returns
        -------
        numpy.ndarray
            ``(Kref,)`` over the reference contexts; an EMPTY ``(0,)`` array
            when no member holds a context - the single-member convention (the
            MATLAB original returns ``[]`` there), so the ``runs == 1``
            reduction of SPEC 10.5.1 holds for that case too.

        Notes
        -----
        DELIBERATE FORK from ``stationary_context_probabilities.m``, which
        always returns ``acc / sum(acc)``: with exactly ONE contributing run the
        scattered accumulator is returned VERBATIM instead. That run necessarily
        holds ``Kref`` contexts (it is the only member with any), so it IS the
        reference member and - unless two of its prototypes coincide exactly,
        making the self-matching non-unique - the accumulator already equals its
        stationary row. That row sums to ``1 +- 1 ulp`` rather than exactly 1
        (``stationary_distribution`` ends in a division), so the extra
        normalisation would perturb the low bits and break the bit-exact
        ``runs == 1`` reduction SPEC 10.5.1 demands - which is exactly what the
        blind Phase-2 suites assert. The fork therefore resolves a conflict
        between the MATLAB source and the MATLAB tests in favour of the SPEC.
        It is the same single-contributor pass-through :func:`_pool_moments`
        applies for SPEC 5.5, and it costs nothing: the result still sums to 1
        to within a few eps, as SPEC 10.5.2 requires.
        """
        self._check_lockstep()
        alignment = _ensemble_context_alignment(self)
        k_ref = int(alignment["k_ref"])
        if k_ref == 0:
            return np.zeros(0)

        acc = np.zeros(k_ref)                                    # (Kref,)
        contributors = 0
        for r, member in enumerate(self._members):
            s = np.asarray(
                member.stationary_context_probabilities(), dtype=float
            ).reshape(-1)                                        # (K_r,)
            if s.size == 0:
                continue
            contributors += 1
            for i in range(min(int(alignment["k"][r]), s.size)):
                acc[alignment["perm"][r][i]] += s[i]

        if contributors == 1:
            return acc                                # exact pass-through
        total = float(acc.sum())
        if total > 0:
            return acc / total
        return np.full(k_ref, 1.0 / k_ref)

    def state_given_context_probability(self, values):
        """Cross-run averaged per-context state density (SPEC 10.4).

        NaN-omit average in the reference frame: reference context ``j``'s
        density is the mean over only the runs holding a context matched to
        ``j``; a label no run holds is absent from the result.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid - the
            single-model shape rules exactly.

        Returns
        -------
        dict
            ``{0-based reference label: (K,) density}``.
        """
        self._check_lockstep()
        return _ensemble_context_density(
            self, lambda m: m.state_given_context_probability(values)
        )

    def state_feedback_given_context_probability(self, values):
        """Cross-run averaged per-context feedback density (SPEC 10.4).

        Observation-space counterpart of
        :meth:`state_given_context_probability`.

        Parameters
        ----------
        values : array_like
            ``(K,)`` scalar grid, or ``(K, N)`` multi-dimensional grid.

        Returns
        -------
        dict
            ``{0-based reference label: (K,) density}``.
        """
        self._check_lockstep()
        return _ensemble_context_density(
            self, lambda m: m.state_feedback_given_context_probability(values)
        )


# --------------------------------------------------------------------------
# Parameter validators
# --------------------------------------------------------------------------


def _positive_integer(value, name):
    """Validate a positive integer ensemble parameter.

    Parameters
    ----------
    value : object
        Supplied value.
    name : str
        Parameter name, used in the error message.

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    ValueError
        If ``value`` is not an integer >= 1.
    """
    number = _as_integer(value, name)
    if number < 1:
        raise ValueError(
            "RealTimeCOINEnsemble:InvalidParameter: %s must be a positive "
            "integer, got %r." % (name, value)
        )
    return number


def _nonnegative_integer(value, name):
    """Validate a non-negative integer ensemble parameter.

    Parameters
    ----------
    value : object
        Supplied value.
    name : str
        Parameter name, used in the error message.

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    ValueError
        If ``value`` is not an integer >= 0.
    """
    number = _as_integer(value, name)
    if number < 0:
        raise ValueError(
            "RealTimeCOINEnsemble:InvalidParameter: %s must be a non-negative "
            "integer, got %r." % (name, value)
        )
    return number


def _as_integer(value, name):
    """Coerce ``value`` to ``int``, rejecting non-integral and non-scalar input.

    Parameters
    ----------
    value : object
        Supplied value.
    name : str
        Parameter name, used in the error message.

    Returns
    -------
    int
        The value as an ``int``.

    Raises
    ------
    ValueError
        If ``value`` is not a real scalar with an integral value. ``2.0`` is
        accepted (MATLAB has no integer literals, so the ported call sites pass
        doubles); ``2.5`` and ``nan`` are not.
    """
    if isinstance(value, bool):
        raise ValueError(
            "RealTimeCOINEnsemble:InvalidParameter: %s must be an integer, got "
            "%r." % (name, value)
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "RealTimeCOINEnsemble:InvalidParameter: %s must be a real scalar, "
            "got %r." % (name, value)
        ) from exc
    if not np.isfinite(number) or number != int(number):
        raise ValueError(
            "RealTimeCOINEnsemble:InvalidParameter: %s must be an integer, got "
            "%r." % (name, value)
        )
    return int(number)

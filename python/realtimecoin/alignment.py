"""Cross-particle context alignment (global relabelling for reporting).

Translated from ``@RealTimeCOIN/private/``: ``invalidateContextAlignment``,
``ensureContextAlignment``, ``computeContextAlignment``,
``initializeContextAlignment``, ``optimizeContextAlignment``,
``linearAssignment``, ``assignmentCostMatrix[MD]``,
``prepareAssignmentPrototypes``, ``updateGlobalContexts[MD]``,
``selectModalContexts``, ``modalCardinality``, ``globalContextWeights``,
``globalSampledContexts``, ``globalContextMatrix``, ``globalTransitionRow``,
``globalTransitionTensor``, ``globalCueTensor``, ``scatterToGlobal``,
``activeSummaryContexts``, ``clampActiveSummaryContexts``,
``contextProbabilityVector`` and the three per-context "local" belief readers
``localDynamicsDistribution`` / ``localBiasDistribution`` /
``localCueProbability``.

Why this exists
---------------
Context labels are per-particle and arbitrary: particle 3's "context 2" has
nothing to do with particle 7's "context 2". Any summary that reports
per-context quantities across particles must therefore first solve an
assignment problem that maps each particle's local labels onto one global
labelling. That alignment is REPORTING ONLY - it never feeds back into
inference, and NOTHING in this module ever mutates ``model.D`` - and it is
computed lazily and cached, keyed on ``model.state_version``.

Caching contract
----------------
``model.state_version`` is a monotone counter bumped by every mutation;
``model.alignment_cache`` holds the alignment computed at some version and
carries the version it was built for. ``model.alignment_seed`` is a warm-start
seed for the optimiser and is deliberately NOT cleared on invalidation, so a
recompute can reuse the previous solution.

Label conventions (re-derived from MATLAB, read this before editing)
--------------------------------------------------------------------
* Arrays are particles-leading, per :mod:`realtimecoin.state`. Where MATLAB
  writes ``assignment(local, p)`` this module writes ``assignment[p, local]``.
* Global context labels are 0-BASED here (MATLAB's are 1-based) and the
  "unassigned" sentinel is ``-1`` (MATLAB uses ``0``). Every MATLAB guard of
  the form ``target > 0 && target <= Cmax`` therefore becomes
  ``0 <= target < c_slots``, and ``target <= Km + 1`` becomes ``target <= km``.
* ``alignment["assignment"]`` is ``(P, C)`` int; only the modal particles carry
  meaningful entries. Slots ``0 .. km-1`` hold the matched global labels and
  slot ``km`` holds the novel-context label ``km`` when ``km < max_contexts``.
* Prototypes are dicts of numpy arrays with the global-context axis LEADING:
  scalar ``state_mean (Km,)``, ``dynamics_mean (Km, 2)``,
  ``dynamics_covar (Km, 2, 2)``, ``cue_prob (Km, Q)``,
  ``transition_prob (Km, Km + 1)``; MD ``state_mean (Km, N)``,
  ``state_cov (Km, N, N)``, ``theta_mean (Km, N, N + 1)``,
  ``bias_mean (Km, N)``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

from .numerics import (
    categorical_jeffreys,
    gaussian_jeffreys,
    gaussian_jeffreys_multi,
    jeffreys_finite_clip,
    normalize_probability,
    precision_to_variance,
    regularize_covariance,
    renormalize_global_weights,
    safe_inverse,
)

__all__ = [
    "invalidate_context_alignment",
    "ensure_context_alignment",
    "compute_context_alignment",
    "initialize_context_alignment",
    "optimize_context_alignment",
    "linear_assignment",
    "assignment_cost_matrix",
    "assignment_cost_matrix_md",
    "prepare_assignment_prototypes",
    "update_global_contexts",
    "update_global_contexts_md",
    "select_modal_contexts",
    "modal_cardinality",
    "global_context_weights",
    "global_sampled_contexts",
    "global_context_matrix",
    "global_transition_row",
    "global_transition_tensor",
    "global_cue_tensor",
    "scatter_to_global",
    "active_summary_contexts",
    "clamp_active_summary_contexts",
    "context_probability_vector",
    # Per-context local belief readers. Not in the MATLAB "alignment" family,
    # but they have no other home in the Python layout and the cost matrices /
    # prototype updates are their only in-tree callers so far.
    "local_dynamics_distribution",
    "local_bias_distribution",
    "local_cue_probability",
]

EPS = float(np.finfo(float).eps)

#: Sentinel stored in ``assignment`` for a local slot with no global label.
#: MATLAB uses 0 for this; 0 is a valid 0-based label here, hence -1.
UNASSIGNED = -1

#: Safety cap on assignment/prototype sweeps in
#: :func:`optimize_context_alignment`. Convergence normally takes a few
#: iterations; changing this re-baselines the alignment output.
MAX_ALIGNMENT_ITERATIONS = 20


def invalidate_context_alignment(model):
    """Mark the cached global alignment stale.

    Bumps the monotone ``model.state_version`` counter and clears
    ``model.alignment_cache``. Called at the end of every ``observe_y`` (and by
    the particle reset) so the next context-facing query recomputes the
    alignment against the updated particle state.

    The warm-start seed in ``model.alignment_seed`` is deliberately left intact,
    so the recompute can reuse it.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.

    Returns
    -------
    None
    """
    # Bump the version so any existing cache is detected as stale, then drop it.
    model.state_version = model.state_version + 1
    model.alignment_cache = None


def ensure_context_alignment(model):
    """Return the cached global alignment, recomputing it if stale.

    The caching front-end for :func:`compute_context_alignment`. Returns
    ``model.alignment_cache`` when it was built for the current
    ``model.state_version``; otherwise recomputes and stores the fresh
    alignment. Called by every context-facing query method, so a single trial's
    queries share one alignment.

    A cache hit returns the SAME dict object (MATLAB's struct copy is
    indistinguishable from this because its structs are values), so repeated
    calls within a trial are free and their ``iterations`` / ``used_seed``
    diagnostics describe the one computation that actually ran.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose alignment cache is read and possibly written.

    Returns
    -------
    dict
        The alignment structure; see :func:`compute_context_alignment`. Its
        arrays are READ-ONLY - the returned object IS the cache and the
        warm-start seed, so writing to them would corrupt both.
    """
    cache = model.alignment_cache
    if cache is not None and cache.get("cache_state_version") == model.state_version:
        return cache
    alignment = compute_context_alignment(model)
    model.alignment_cache = alignment
    return alignment


def compute_context_alignment(model):
    """Solve the cross-particle context alignment for the current trial.

    Pipeline (identical to ``computeContextAlignment.m``):

    1. :func:`select_modal_contexts` - pick the modal cardinality ``K`` and the
       particles carrying exactly ``K`` contexts, with their uniform weights.
    2. :func:`initialize_context_alignment` - seed the assignment and
       prototypes, warm-starting from ``model.alignment_seed`` when compatible.
    3. :func:`optimize_context_alignment` - alternate min-cost matching and
       prototype recomputation until the labels stabilise.

    The result is stored back into ``model.alignment_seed`` so the next
    computation can warm-start from it (this is the ONLY model state written
    here; the particle arrays are read-only throughout).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    dict
        Alignment structure with these keys:

        ``K`` : int
            Number of instantiated (aligned) global contexts.
        ``assignment`` : numpy.ndarray
            ``(P, C)`` per-particle local-to-global label mapping;
            :data:`UNASSIGNED` where a local slot has no global label.
        ``modal_particle_mask`` : numpy.ndarray
            ``(P,)`` boolean mask of particles whose context count equals the
            mode.
        ``modal_particle_indices`` : numpy.ndarray
            Indices of those modal particles.
        ``modal_particle_weights`` : numpy.ndarray
            Their (uniform) weights, summing to one.
        ``global_contexts`` : dict
            Per-global-context prototype parameters.
        ``converged`` : bool
        ``iterations`` : int
            Alignment-solver diagnostics.
        ``used_seed`` : bool
            Whether the warm start was taken.
        ``cache_state_version`` : int
            The ``model.state_version`` this alignment was computed for.
        ``computed_at_trial`` : int
            ``model.trial`` at computation time.
    """
    # Modal particle subset and per-particle weights.
    km, modal_mask, modal_idx, weights = select_modal_contexts(model)
    # Seed assignment/prototypes (warm-started from alignment_seed when valid).
    assignment, prototypes, used_seed = initialize_context_alignment(
        model, km, modal_idx
    )
    # Iterate assignment <-> prototype recomputation to a fixed point.
    assignment, prototypes, converged, iterations = optimize_context_alignment(
        model, km, modal_idx, assignment, prototypes, weights=weights
    )

    alignment = {
        "K": int(km),
        "assignment": _freeze(assignment),
        "modal_particle_mask": _freeze(modal_mask),
        "modal_particle_indices": _freeze(modal_idx),
        "modal_particle_weights": _freeze(weights),
        "global_contexts": {
            name: _freeze(value) for name, value in prototypes.items()
        },
        "converged": bool(converged),
        "iterations": int(iterations),
        "used_seed": bool(used_seed),
        "cache_state_version": model.state_version,
        "computed_at_trial": model.trial,
    }

    # Retain this alignment as the warm-start seed for the next computation.
    model.alignment_seed = alignment
    return alignment


def _freeze(array):
    """Return ``array`` marked read-only.

    MATLAB structs are VALUES: ``ensureContextAlignment`` and
    ``computeContextAlignment`` each hand out an independent copy, so a caller
    who scribbles on ``alignment.assignment`` cannot corrupt
    ``obj.alignment_cache`` - nor ``obj.alignment_seed``, which would poison the
    NEXT trial's warm start. Python has no value semantics, and the cache, the
    seed and the object returned to the caller are deliberately one and the same
    (so a cache hit is free). Freezing the arrays restores the safety half of
    MATLAB's guarantee without paying for a copy per query: an accidental write
    raises ``ValueError`` instead of silently corrupting two trials' worth of
    state.

    Parameters
    ----------
    array : numpy.ndarray
        Array to publish. Mutated in place (its writeable flag is cleared), so
        pass a buffer that is finished being built.

    Returns
    -------
    numpy.ndarray
        The same array, no longer writeable.
    """
    array.flags.writeable = False
    return array


def initialize_context_alignment(model, km, modal_idx):
    """Seed the alignment optimiser with an initial assignment and prototypes.

    Two paths, exactly as in ``initializeContextAlignment.m``:

    * Warm start (``used_seed = True``): when ``model.alignment_seed`` targets
      the same cardinality ``km`` and carries prototypes of the right shape,
      reuse its prototypes and per-particle assignment. This is what keeps
      global labels stable across trials. Any modal particle the seed left
      unlabelled gets the identity map, and the novel slot ``km`` is labelled
      when there is room below ``max_contexts``.
    * Cold start (``used_seed = False``): anchor the identity map on the FIRST
      modal particle and build the prototypes from that particle alone.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    km : int
        Modal context cardinality.
    modal_idx : numpy.ndarray
        Indices of the modal particles.

    Returns
    -------
    assignment : numpy.ndarray
        ``(P, C)`` initial per-particle local-to-global mapping.
    prototypes : dict
        Initial global-context prototypes.
    used_seed : bool
        Whether ``model.alignment_seed`` was reused as the warm start.
    """
    c_slots = model.max_contexts + 1
    km = int(km)
    modal_idx = np.asarray(modal_idx, dtype=int).reshape(-1)
    assignment = np.full((model.num_particles, c_slots), UNASSIGNED, dtype=int)

    if _is_compatible_seed(model, km):
        # --- Warm start from the previous alignment seed ---
        seed = model.alignment_seed
        prototypes = seed["global_contexts"]
        seed_assignment = seed.get("assignment")
        if (
            seed_assignment is not None
            and np.shape(seed_assignment) == assignment.shape
        ):
            assignment[modal_idx, :] = np.asarray(seed_assignment)[modal_idx, :]
        for p in modal_idx:
            # Fill particles the seed did not cover with the identity map.
            if np.all(assignment[p, :km] == UNASSIGNED):
                assignment[p, :km] = np.arange(km)
            # Label the novel-context slot when there is room for it.
            if km < model.max_contexts and assignment[p, km] == UNASSIGNED:
                assignment[p, km] = km
        return assignment, prototypes, True

    # --- Cold start: anchor identity map on the first modal particle ---
    anchor = int(modal_idx[0])
    assignment[anchor, :km] = np.arange(km)
    if km < model.max_contexts:
        assignment[anchor, km] = km
    prototypes = update_global_contexts(
        model, km, np.array([anchor]), np.array([1.0]), assignment
    )
    return assignment, prototypes, False


def _is_compatible_seed(model, km):
    """True when ``model.alignment_seed`` can be reused for cardinality ``km``.

    Mirrors the nested ``isCompatibleSeed`` of ``initializeContextAlignment.m``:
    the seed must exist, target the same ``km``, and carry prototypes whose
    fields and context-axis length match the current model (scalar vs MD).

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``alignment_seed`` is inspected.
    km : int
        Modal context cardinality the seed would be reused for.

    Returns
    -------
    bool
        Whether the warm start is admissible.
    """
    seed = model.alignment_seed
    if not isinstance(seed, dict):
        return False
    if seed.get("K") != km:
        return False
    proto = seed.get("global_contexts")
    if not isinstance(proto, dict):
        return False
    if model.state_dim > 1:
        if not {"state_mean", "state_cov", "theta_mean"} <= set(proto):
            return False
        # (Km, N): MATLAB checks size(state_mean, 2) on its N-by-Km layout.
        return np.shape(proto["state_mean"])[0] == km
    if not {"state_mean", "state_var", "dynamics_mean"} <= set(proto):
        return False
    return np.size(proto["state_mean"]) == km


def optimize_context_alignment(
    model, km, modal_idx, assignment, prototypes, *, weights=None
):
    """Iterate assignment and prototype updates to a fixed point.

    Hard-assignment, EM-style block coordinate descent
    (``optimizeContextAlignment.m``): each sweep relabels every modal particle
    by solving its local-to-global min-cost matching, then rebuilds the global
    prototypes from the fresh labels. Reporting-only; inference is untouched.

    The transition-divergence term is DISABLED on the first sweep (the
    transition prototypes are not yet meaningful before any relabelling) and
    enabled from the second sweep onward. Preserved verbatim - it changes the
    fixed point, not just the path to it.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    km : int
        Modal context cardinality.
    modal_idx : numpy.ndarray
        Indices of the modal particles.
    assignment : numpy.ndarray
        ``(P, C)`` initial per-particle local-to-global mapping. Mutated in
        place (and also returned), exactly as MATLAB's copy-in/copy-out is used.
    prototypes : dict
        Initial global-context prototypes.
    weights : numpy.ndarray or None, keyword-only, optional
        Per-modal-particle weights for the prototype update. ``None`` (the
        default) means the uniform weights that :func:`select_modal_contexts`
        always produces.

        MATLAB's ``optimizeContextAlignment`` takes ``weights`` POSITIONALLY,
        between ``modalIdx`` and ``assignment``. Here it is trailing and
        keyword-only: trailing so the positional signature matches the rest of
        this module, keyword-only so a literal port of a MATLAB call site
        cannot silently pass ``weights`` where ``assignment`` is expected.

    Returns
    -------
    assignment : numpy.ndarray
        Converged mapping.
    prototypes : dict
        Converged prototypes.
    converged : bool
        Whether the modal-particle label block stopped changing.
    iterations : int
        Number of sweeps performed.
    """
    km = int(km)
    modal_idx = np.asarray(modal_idx, dtype=int).reshape(-1)
    if weights is None:
        weights = np.ones(modal_idx.size) / max(modal_idx.size, 1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    c_slots = model.max_contexts + 1
    # MATLAB seeds this with zeros, but 0 is its unassigned sentinel and a VALID
    # label here, so seed with UNASSIGNED instead. The value is provably unused
    # (the comparison below is guarded by iteration > 1, and the buffer is
    # overwritten at the end of sweep 1) - this just removes the trap.
    old_assignment = np.full((model.num_particles, c_slots), UNASSIGNED, dtype=int)
    converged = False
    include_transition = False   # transition term disabled on the first sweep
    iterations = 0

    for iteration in range(1, MAX_ALIGNMENT_ITERATIONS + 1):
        iterations = iteration
        # Prototype quantities shared by every particle within this sweep.
        prepared = prepare_assignment_prototypes(model, km, prototypes)

        # --- Assignment step: relabel each modal particle by min-cost matching.
        for p in modal_idx:
            cost = assignment_cost_matrix(
                model, int(p), km, prototypes, assignment, include_transition, prepared
            )
            perm = linear_assignment(cost)
            assignment[p, :] = UNASSIGNED
            assignment[p, :km] = perm
            if km < model.max_contexts:
                assignment[p, km] = km   # label the novel-context slot

        # --- Prototype step: rebuild global prototypes from the new labels.
        prototypes = update_global_contexts(
            model, km, modal_idx, weights, assignment
        )

        # Converged once the modal-particle assignment stops changing.
        if iteration > 1 and np.array_equal(
            assignment[modal_idx, :km], old_assignment[modal_idx, :km]
        ):
            converged = True
            break
        old_assignment = assignment.copy()
        include_transition = True   # enable the transition term from sweep 2

    return assignment, prototypes, converged, iterations


def linear_assignment(cost):
    """Solve a square minimum-cost assignment problem.

    Thin wrapper over :func:`scipy.optimize.linear_sum_assignment`, replacing
    the hand-written Jonker-Volgenant solver of ``linearAssignment.m``. Both
    return a minimum-cost perfect matching, so the TOTAL cost is identical.

    .. warning::
       TIE-BREAKING MAY DIFFER FROM MATLAB. When several permutations achieve
       the same optimal cost - a degenerate cost matrix, e.g. contexts with
       identical prototypes - SciPy and the MATLAB solver may return different
       (equally optimal) permutations. Assert cost optimality and label
       consistency, not an exact permutation, unless the optimum is unique.

    Non-finite and oversized costs are replaced by a large finite sentinel so
    infeasible pairings are effectively forbidden while the solver still runs;
    this mirrors the MATLAB guard (which needed it because ``inf`` poisons the
    dual updates, whereas SciPy simply rejects non-finite input).

    Parameters
    ----------
    cost : numpy.ndarray
        ``(n, n)`` cost matrix.

    Returns
    -------
    numpy.ndarray
        Length-``n`` array giving the column assigned to each row.

    Raises
    ------
    ValueError
        If ``cost`` is not square (identifier
        ``RealTimeCOIN:AssignmentMatrixNotSquare``).
    """
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2:
        c = np.atleast_2d(c)
    n = c.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if c.shape[1] != n:
        raise ValueError(
            "RealTimeCOIN:AssignmentMatrixNotSquare: "
            "Assignment cost matrix must be square."
        )

    # large: finite sentinel standing in for +inf / forbidden pairings.
    large = 1e100
    c = np.where(np.isfinite(c), c, large)
    c = np.fmin(c, large)

    rows, cols = _linear_sum_assignment(c)
    out = np.zeros(n, dtype=int)
    out[rows] = cols
    return out


def assignment_cost_matrix(
    model, p, km, prototypes, assignment, include_transition, prepared=None
):
    """Local-to-global assignment cost for one particle (scalar model).

    ``cost[local, g]`` is the sum of symmetric (Jeffreys) divergences between
    local context ``local`` of particle ``p`` and global prototype ``g``:

    * state - Gaussian-Jeffreys on the filtered state (mean, variance);
    * dynamics - BIVARIATE Gaussian-Jeffreys on ``[a; d]`` (mean + covariance);
    * bias - Gaussian-Jeffreys on the observation bias, only when
      ``infer_bias``;
    * cue - categorical Jeffreys between cue-emission rows;
    * transition - categorical Jeffreys between transition rows, only when
      ``include_transition``.

    Delegates to :func:`assignment_cost_matrix_md` when ``state_dim > 1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    p : int
        Particle index.
    km : int
        Modal context cardinality.
    prototypes : dict
        Current global-context prototypes.
    assignment : numpy.ndarray
        ``(P, C)`` current per-particle mapping.
    include_transition : bool
        Whether to include the transition-row divergence term.
    prepared : dict or None, optional
        Precomputed prototype quantities from
        :func:`prepare_assignment_prototypes`; unused on the scalar path.

    Returns
    -------
    numpy.ndarray
        ``(km, km)`` cost over (local label, global label) pairs.
    """
    if model.state_dim > 1:
        return assignment_cost_matrix_md(
            model, p, km, prototypes, assignment, include_transition, prepared
        )

    km = int(km)
    p = int(p)
    d = model.D
    cost = np.zeros((km, km))
    for local in range(km):
        # Local-context distributions, computed once per row.
        dyn_mean, dyn_covar = local_dynamics_distribution(model, local, p)
        cue_prob = local_cue_probability(model, local, p)
        transition_prob = None
        if include_transition:
            transition_prob = global_transition_row(model, local, p, km, assignment)
        bias_mean = bias_var = None
        if model.infer_bias:
            bias_mean, bias_var = local_bias_distribution(model, local, p)

        state_mean = d.state_filtered_mean[p, local]
        state_var = d.state_filtered_var[p, local]
        for g in range(km):
            # Accumulate the per-block symmetric divergences.
            total = gaussian_jeffreys(
                state_mean,
                state_var,
                prototypes["state_mean"][g],
                prototypes["state_var"][g],
            )
            total = total + gaussian_jeffreys_multi(
                dyn_mean,
                dyn_covar,
                prototypes["dynamics_mean"][g],
                prototypes["dynamics_covar"][g],
            )
            if model.infer_bias:
                total = total + gaussian_jeffreys(
                    bias_mean,
                    bias_var,
                    prototypes["bias_mean"][g],
                    prototypes["bias_var"][g],
                )
            total = total + categorical_jeffreys(cue_prob, prototypes["cue_prob"][g])
            if include_transition:
                total = total + categorical_jeffreys(
                    transition_prob, prototypes["transition_prob"][g]
                )
            cost[local, g] = total
    return cost


def assignment_cost_matrix_md(
    model, p, km, prototypes, assignment, include_transition, prepared=None
):
    """Local-to-global assignment cost for one particle (MD model).

    Multi-dimensional counterpart of :func:`assignment_cost_matrix`; see there
    for the parameters and return value. The blocks are the same, generalised
    to vectors/matrices, with ONE deliberate asymmetry carried over from
    ``assignmentCostMatrixMD.m``: the dynamics term is the plain squared
    Euclidean distance between the vectorised ``Theta = [A | d]`` matrices,
    i.e. a Gaussian-Jeffreys with an isotropic IDENTITY reference covariance.
    It therefore ignores the scale and uncertainty of the dynamics, unlike the
    scalar path's full bivariate Jeffreys. That is disclosed behaviour, not a
    bug - do not "improve" it here.

    The vectorisation order of ``Theta`` differs from MATLAB (C-order here,
    Fortran-order there); irrelevant, because only the squared norm of the
    difference is used and that is invariant to a shared permutation.
    """
    if prepared is None or not prepared:
        prepared = prepare_assignment_prototypes(model, km, prototypes)

    km = int(km)
    p = int(p)
    n = model.state_dim
    d = model.D
    cost = np.zeros((km, km))

    # Precompute the local-context terms once per row.
    local_state_cov = np.zeros((km, n, n))
    local_state_inv = np.zeros((km, n, n))
    local_theta = np.zeros((km, n * (n + 1)))
    local_cue = [None] * km
    local_transition = [None] * km
    for local in range(km):
        covar = regularize_covariance(d.state_filtered_cov[p, local])
        local_state_cov[local] = covar
        local_state_inv[local] = safe_inverse(covar)
        local_theta[local] = d.Theta[p, local].reshape(-1)
        local_cue[local] = local_cue_probability(model, local, p)
        if include_transition:
            local_transition[local] = global_transition_row(
                model, local, p, km, assignment
            )

    for local in range(km):
        s_mean = d.state_filtered_mean[p, local]      # (N,)
        s_cov = local_state_cov[local]                # (N, N)
        s_inv = local_state_inv[local]                # (N, N)
        theta_vec = local_theta[local]                # (N * (N + 1),)
        for g in range(km):
            # State: symmetric Gaussian-Jeffreys (mean + covariance).
            total = _prepared_gaussian_jeffreys(
                s_mean,
                s_cov,
                s_inv,
                prototypes["state_mean"][g],
                prepared["state_cov"][g],
                prepared["state_inv"][g],
            )
            # Dynamics: squared Euclidean distance on vectorised [A | d].
            theta_delta = theta_vec - prepared["theta_vec"][g]
            total = total + float(theta_delta @ theta_delta)
            # Cue (always) and transition (only past the first sweep).
            total = total + categorical_jeffreys(
                local_cue[local], prototypes["cue_prob"][g]
            )
            if include_transition:
                total = total + categorical_jeffreys(
                    local_transition[local], prototypes["transition_prob"][g]
                )
            cost[local, g] = total
    return cost


def _prepared_gaussian_jeffreys(m1, s1, inv1, m2, s2, inv2):
    """Jeffreys divergence between two Gaussians, given cached inverses.

    ``d = 0.5 (tr(inv2 s1 + inv1 s2) + delta' (inv1 + inv2) delta - 2k)`` with
    ``delta = m1 - m2`` and ``k`` the dimension. Duplicates
    :func:`realtimecoin.numerics.gaussian_jeffreys_multi` but takes the already
    computed inverses, so the pairwise cost loop does not re-invert every
    prototype covariance for every local context.

    Parameters
    ----------
    m1, m2 : array_like
        Length-``k`` mean vectors.
    s1, s2 : array_like
        ``(k, k)`` covariances.
    inv1, inv2 : array_like
        Their inverses.

    Returns
    -------
    float
        Finite, non-negative divergence.
    """
    delta = np.asarray(m1, dtype=float).reshape(-1) - np.asarray(
        m2, dtype=float
    ).reshape(-1)
    k = delta.size
    d = 0.5 * (
        np.trace(inv2 @ s1 + inv1 @ s2) + delta @ (inv1 + inv2) @ delta - 2.0 * k
    )
    return jeffreys_finite_clip(d)


def prepare_assignment_prototypes(model, km, prototypes):
    """Precompute the prototype quantities reused across particles.

    Caches, once per optimiser sweep, what :func:`assignment_cost_matrix_md`
    would otherwise recompute for every (particle, context) pair:
    ``state_cov`` (regularised), ``state_inv`` (their inverses) and
    ``theta_vec`` (vectorised ``[A | d]``). The scalar cost path needs nothing,
    so it gets an empty dict.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``state_dim`` selects the branch.
    km : int
        Modal context cardinality.
    prototypes : dict
        Current global-context prototypes.

    Returns
    -------
    dict
        ``{}`` for the scalar model; otherwise ``state_cov (Km, N, N)``,
        ``state_inv (Km, N, N)`` and ``theta_vec (Km, N * (N + 1))``.
    """
    prepared = {}
    if model.state_dim <= 1:
        return prepared   # scalar cost path uses no prototype precompute

    km = int(km)
    n = model.state_dim
    state_cov = np.zeros((km, n, n))
    state_inv = np.zeros((km, n, n))
    theta_vec = np.zeros((km, n * (n + 1)))
    for g in range(km):
        covar = regularize_covariance(prototypes["state_cov"][g])
        state_cov[g] = covar
        state_inv[g] = safe_inverse(covar)
        theta_vec[g] = np.asarray(prototypes["theta_mean"][g], dtype=float).reshape(-1)
    prepared["state_cov"] = state_cov
    prepared["state_inv"] = state_inv
    prepared["theta_vec"] = theta_vec
    return prepared


def _normalised_modal_weights(modal_idx, weights):
    """Flatten ``(modal_idx, weights)`` and renormalise the weights to sum one.

    Shared preamble of the two prototype builders. ``modal_idx`` may be a bare
    scalar (the cold-start anchor call passes a single particle), hence the
    ``atleast_1d``. A zero or non-finite total is left alone rather than
    producing ``nan`` weights.

    Parameters
    ----------
    modal_idx : array_like
        Particle index/indices.
    weights : array_like
        Matching weights.

    Returns
    -------
    modal_idx : numpy.ndarray
        1-D integer indices.
    weights : numpy.ndarray
        1-D weights summing to one (unless their total was non-positive).
    """
    modal_idx = np.atleast_1d(np.asarray(modal_idx, dtype=int)).reshape(-1)
    weights = np.atleast_1d(np.asarray(weights, dtype=float)).reshape(-1)
    total = weights.sum()
    if weights.size and np.isfinite(total) and total > 0:
        weights = weights / total
    return modal_idx, weights


def _assigned_local(assignment, p, km, g):
    """First local slot of particle ``p`` carrying global label ``g``.

    MATLAB: ``find(assignment(1:Km, p) == globalIdx, 1)``.

    Parameters
    ----------
    assignment : numpy.ndarray
        ``(P, C)`` mapping.
    p : int
        Particle index.
    km : int
        Modal cardinality (only slots below it are searched).
    g : int
        Global label to look for.

    Returns
    -------
    int or None
        The local slot, or ``None`` when this particle has no slot on ``g``.
    """
    hits = np.flatnonzero(assignment[p, :km] == g)
    if hits.size == 0:
        return None
    return int(hits[0])


def update_global_contexts(model, km, modal_idx, weights, assignment):
    """Recompute the global-context prototypes (scalar model).

    For each global context, a WEIGHTED summary of the local contexts currently
    assigned to it across the modal particles. These prototypes are the
    reference distributions the assignment cost is measured against, so they are
    rebuilt after every assignment sweep.

    Weighted first and second moments are accumulated and converted to
    means/(co)variances (``var = E[x^2] - E[x]^2``, floored at zero). A global
    context that no particle claims keeps zero moments, a regularised zero
    covariance and - because ``normalize_probability`` falls back to uniform on
    an all-zero input - UNIFORM cue and transition rows. That fallback is load
    bearing: it stops an unclaimed prototype from attracting everything through
    a degenerate (zero) categorical divergence.

    Delegates to :func:`update_global_contexts_md` when ``state_dim > 1``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    km : int
        Modal context cardinality.
    modal_idx : array_like
        Indices of the modal particles (a scalar is accepted).
    weights : array_like
        Their weights; renormalised internally.
    assignment : numpy.ndarray
        ``(P, C)`` current per-particle local-to-global mapping.

    Returns
    -------
    dict
        ``state_mean (Km,)``, ``state_var (Km,)``, ``dynamics_mean (Km, 2)``,
        ``dynamics_covar (Km, 2, 2)``, ``bias_mean (Km,)``, ``bias_var (Km,)``,
        ``cue_prob (Km, Q)`` and ``transition_prob (Km, Km + 1)``.
    """
    if model.state_dim > 1:
        return update_global_contexts_md(model, km, modal_idx, weights, assignment)

    km = int(km)
    modal_idx, weights = _normalised_modal_weights(modal_idx, weights)
    d = model.D
    q_cols = max(1, d.local_cue_matrix.shape[2])   # number of cue columns

    prototypes = {
        "state_mean": np.zeros(km),
        "state_var": np.zeros(km),
        "dynamics_mean": np.zeros((km, 2)),
        "dynamics_covar": np.zeros((km, 2, 2)),
        "bias_mean": np.zeros(km),
        "bias_var": np.zeros(km),
        "cue_prob": np.zeros((km, q_cols)),
        "transition_prob": np.zeros((km, km + 1)),
    }

    for g in range(km):
        total_weight = 0.0
        state_mean = 0.0
        state_second = 0.0
        dyn_mean = np.zeros(2)
        dyn_second = np.zeros((2, 2))
        bias_mean = 0.0
        bias_second = 0.0
        cue_accum = np.zeros(q_cols)
        transition_accum = np.zeros(km + 1)

        for idx, p in enumerate(modal_idx):
            # Local context (if any) that particle p maps to this global label.
            local = _assigned_local(assignment, p, km, g)
            if local is None:
                continue
            w = weights[idx]
            total_weight += w

            # Weighted first and second moments of the filtered state.
            m = float(d.state_filtered_mean[p, local])
            v = max(float(d.state_filtered_var[p, local]), 0.0)
            state_mean += w * m
            state_second += w * (v + m * m)

            dm, dc = local_dynamics_distribution(model, local, p)
            dyn_mean += w * dm
            dyn_second += w * (dc + np.outer(dm, dm))

            bm, bv = local_bias_distribution(model, local, p)
            bias_mean += w * bm
            bias_second += w * (bv + bm * bm)

            cue_accum += w * local_cue_probability(model, local, p)
            transition_accum += w * global_transition_row(
                model, local, p, km, assignment
            )

        # Normalise the weighted sums into expectations for this global context.
        if total_weight > 0:
            state_mean /= total_weight
            state_second /= total_weight
            dyn_mean = dyn_mean / total_weight
            dyn_second = dyn_second / total_weight
            bias_mean /= total_weight
            bias_second /= total_weight
            cue_accum = cue_accum / total_weight
            transition_accum = transition_accum / total_weight

        prototypes["state_mean"][g] = state_mean
        prototypes["state_var"][g] = max(state_second - state_mean ** 2, 0.0)
        prototypes["dynamics_mean"][g] = dyn_mean
        prototypes["dynamics_covar"][g] = regularize_covariance(
            dyn_second - np.outer(dyn_mean, dyn_mean)
        )
        prototypes["bias_mean"][g] = bias_mean
        prototypes["bias_var"][g] = max(bias_second - bias_mean ** 2, 0.0)
        prototypes["cue_prob"][g] = normalize_probability(cue_accum)
        prototypes["transition_prob"][g] = normalize_probability(transition_accum)

    return prototypes


def update_global_contexts_md(model, km, modal_idx, weights, assignment):
    """Recompute the global-context prototypes (MD model).

    Multi-dimensional counterpart of :func:`update_global_contexts`. The scalar
    state/dynamics/bias summaries become vectors and matrices; the cue and
    transition prototypes are identical in form. Note that the dynamics
    prototype is the MEAN augmented matrix ``Theta`` only - there is no
    second-moment/covariance counterpart, matching the isotropic dynamics term
    of :func:`assignment_cost_matrix_md`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    km : int
        Modal context cardinality.
    modal_idx : array_like
        Indices of the modal particles.
    weights : array_like
        Their weights; renormalised internally.
    assignment : numpy.ndarray
        ``(P, C)`` current per-particle local-to-global mapping.

    Returns
    -------
    dict
        ``state_mean (Km, N)``, ``state_cov (Km, N, N)``,
        ``theta_mean (Km, N, N + 1)``, ``bias_mean (Km, N)``,
        ``cue_prob (Km, Q)`` and ``transition_prob (Km, Km + 1)``.
    """
    km = int(km)
    n = model.state_dim
    modal_idx, weights = _normalised_modal_weights(modal_idx, weights)
    d = model.D
    q_cols = max(1, d.local_cue_matrix.shape[2])

    prototypes = {
        "state_mean": np.zeros((km, n)),
        "state_cov": np.zeros((km, n, n)),
        "theta_mean": np.zeros((km, n, n + 1)),
        "bias_mean": np.zeros((km, n)),
        "cue_prob": np.zeros((km, q_cols)),
        "transition_prob": np.zeros((km, km + 1)),
    }

    for g in range(km):
        total_weight = 0.0
        state_mean = np.zeros(n)
        state_second = np.zeros((n, n))
        theta_mean = np.zeros((n, n + 1))
        bias_mean = np.zeros(n)
        cue_accum = np.zeros(q_cols)
        transition_accum = np.zeros(km + 1)

        for idx, p in enumerate(modal_idx):
            local = _assigned_local(assignment, p, km, g)
            if local is None:
                continue
            w = weights[idx]
            total_weight += w

            m = np.asarray(d.state_filtered_mean[p, local], dtype=float)   # (N,)
            v = np.asarray(d.state_filtered_cov[p, local], dtype=float)    # (N, N)
            state_mean += w * m
            state_second += w * (v + np.outer(m, m))

            theta_mean += w * np.asarray(d.Theta[p, local], dtype=float)
            bias_mean += w * np.asarray(d.bias[p, local], dtype=float)

            cue_accum += w * local_cue_probability(model, local, p)
            transition_accum += w * global_transition_row(
                model, local, p, km, assignment
            )

        if total_weight > 0:
            state_mean = state_mean / total_weight
            state_second = state_second / total_weight
            theta_mean = theta_mean / total_weight
            bias_mean = bias_mean / total_weight
            cue_accum = cue_accum / total_weight
            transition_accum = transition_accum / total_weight

        prototypes["state_mean"][g] = state_mean
        prototypes["state_cov"][g] = regularize_covariance(
            state_second - np.outer(state_mean, state_mean)
        )
        prototypes["theta_mean"][g] = theta_mean
        prototypes["bias_mean"][g] = bias_mean
        prototypes["cue_prob"][g] = normalize_probability(cue_accum)
        prototypes["transition_prob"][g] = normalize_probability(transition_accum)

    return prototypes


def select_modal_contexts(model):
    """Select the particles whose context count equals the modal cardinality.

    Summaries are formed over these particles only, so every particle
    contributing to a summary has the same number of contexts and the labels can
    be put into one-to-one correspondence (which keeps the matching square).

    Fallback (should not normally trigger): if NO particle carries the modal
    cardinality, every particle is treated as modal and ``km`` falls back to the
    largest observed cardinality, clipped to ``max_contexts``.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    km : int
        Modal context cardinality.
    modal_mask : numpy.ndarray
        ``(P,)`` boolean mask of the modal particles.
    modal_idx : numpy.ndarray
        Indices of the modal particles.
    weights : numpy.ndarray
        Uniform per-modal-particle weights summing to one.
    """
    p_count = model.num_particles
    cards = np.asarray(model.D.n_active).reshape(-1)   # per-particle counts
    km = modal_cardinality(cards)
    modal_mask = cards == km
    modal_idx = np.flatnonzero(modal_mask)
    if modal_idx.size == 0:
        # Degenerate case: fall back to using all particles.
        modal_idx = np.arange(p_count)
        modal_mask = np.ones(p_count, dtype=bool)
        km = int(min(int(cards.max()), model.max_contexts)) if cards.size else 0
    weights = np.ones(modal_idx.size) / max(modal_idx.size, 1)
    return int(km), modal_mask, modal_idx, weights


def modal_cardinality(cards):
    """Modal value of a set of per-particle context counts, SMALLEST on ties.

    The number of contexts held by the largest number of particles. When two
    counts are equally common the SMALLER one wins, giving a stable,
    conservative point estimate - this tie rule is part of the numerical
    contract, not an implementation detail.

    Parameters
    ----------
    cards : array_like
        ``(P,)`` per-particle context counts.

    Returns
    -------
    int
        The modal count (``0`` for an empty input).
    """
    cards = np.asarray(cards).reshape(-1)
    if cards.size == 0:
        return 0
    vals, counts = np.unique(cards, return_counts=True)   # vals ascending
    best = counts == counts.max()
    return int(vals[best].min())


def global_context_weights(model, w, alignment=None):
    """Relabel a per-particle context-weight matrix into the global frame.

    Weights are ADDITIVE: when several local slots of one particle map to the
    same global slot their mass is summed, so probability mass is conserved.
    (Contrast :func:`global_context_matrix`, which overwrites.)

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    w : numpy.ndarray
        ``(P, C)`` per-particle weights in local labels.
    alignment : dict or None, optional
        Alignment structure; recomputed lazily when omitted.

    Returns
    -------
    numpy.ndarray
        ``(P_modal, C)`` weights in global labels. When the modal cardinality
        has reached ``max_contexts`` every particle has instantiated all
        contexts, so the novel slot carries no mass and slots from ``km`` on are
        zeroed.
    """
    if alignment is None:
        alignment = ensure_context_alignment(model)
    return scatter_to_global(model, w, alignment, "add")


def global_sampled_contexts(model, alignment=None):
    """Sampled context labels of the modal particles, in the global frame.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    alignment : dict or None, optional
        Alignment structure; recomputed lazily when omitted.

    Returns
    -------
    numpy.ndarray
        ``(P_modal,)`` float array of 0-based global context labels, ``nan``
        where the sampled local context has no global assignment.
    """
    if alignment is None:
        alignment = ensure_context_alignment(model)
    return scatter_to_global(model, None, alignment, "labels")


def global_context_matrix(model, x, alignment=None):
    """Relabel a per-particle per-context quantity into the global frame.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    x : numpy.ndarray
        ``(P, C)`` array whose context axis carries local labels.
    alignment : dict or None, optional
        Alignment structure; recomputed lazily when omitted.

    Returns
    -------
    numpy.ndarray
        ``(P_modal, C)`` in global labels. Unmatched local slots are dropped and
        the assignment OVERWRITES (the last local slot mapped to a global slot
        wins); use :func:`global_context_weights` for probability mass.
    """
    if alignment is None:
        alignment = ensure_context_alignment(model)
    return scatter_to_global(model, x, alignment, "overwrite")


def global_transition_row(model, local, p, km, assignment):
    """Relabel one particle's transition row into the global frame.

    Takes the outgoing transition probabilities of local context ``local`` in
    particle ``p`` and scatters them into the global labels given by
    ``assignment[p]``, then renormalises. Used both to build the transition
    prototypes and as a cost term during assignment.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    local : int
        Local context label of the source row.
    p : int
        Particle index.
    km : int
        Modal context cardinality.
    assignment : numpy.ndarray
        ``(P, C)`` per-particle local-to-global mapping.

    Returns
    -------
    numpy.ndarray
        ``(km + 1,)`` row summing to one: ``km`` aligned destinations plus the
        novel-context slot.
    """
    km = int(km)
    row = np.zeros(km + 1)
    # Consider the km aligned destinations, plus the novel slot when it exists.
    max_local = km + 1 if km < model.max_contexts else km
    transition = model.D.local_transition_matrix
    for dest in range(max_local):
        target = assignment[p, dest]   # global label of this local destination
        if 0 <= target <= km:
            # Accumulate mass onto the destination's global label.
            row[target] += transition[p, local, dest]
    return normalize_probability(row)


def global_transition_tensor(model, t, alignment=None):
    """Relabel the per-particle transition tensor into the global frame.

    BOTH context axes are realigned and the mass of local ``(from, to)`` pairs
    mapping to the same global pair is summed.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    t : numpy.ndarray
        ``(P, C, C)`` transition tensor in local labels.
    alignment : dict or None, optional
        Alignment structure; recomputed lazily when omitted.

    Returns
    -------
    numpy.ndarray
        ``(P_modal, C, C)`` transition tensor in global labels.
    """
    if alignment is None:
        alignment = ensure_context_alignment(model)
    return scatter_to_global(model, t, alignment, "transition")


def global_cue_tensor(model, l, alignment=None):
    """Relabel the per-particle cue tensor into the global frame.

    The context-by-cue analogue of :func:`global_context_weights`: the cue axis
    is carried through unchanged while the context axis is realigned and summed.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    l : numpy.ndarray
        ``(P, C, Q)`` cue-emission tensor in local labels.
    alignment : dict or None, optional
        Alignment structure; recomputed lazily when omitted.

    Returns
    -------
    numpy.ndarray
        ``(P_modal, C, Q)`` cue tensor in global labels.
    """
    if alignment is None:
        alignment = ensure_context_alignment(model)
    return scatter_to_global(model, l, alignment, "cue")


def scatter_to_global(model, x, alignment, mode):
    """Scatter per-particle local context slots into the global frame.

    The shared skeleton behind the global-aggregation helpers. It walks the
    MODAL particles, maps each local context slot to its global slot via
    ``alignment["assignment"]`` (:data:`UNASSIGNED` = dropped), guards the
    target to ``[0, C)``, and accumulates into the global-frame output. One
    output row/page per modal particle.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    x : numpy.ndarray or None
        Per-particle array indexed by local context slot; ``(P, C)``,
        ``(P, C, Q)`` or ``(P, C, C)`` depending on ``mode``. Ignored (and may
        be ``None``) for ``"labels"``.
    alignment : dict
        Alignment structure.
    mode : {'overwrite', 'add', 'cue', 'transition', 'labels'}
        Accumulation rule, tensor rank and post-processing:

        ``'overwrite'``
            ``(P_modal, C)``; the last local slot mapped to a global slot wins.
        ``'add'``
            ``(P_modal, C)``; local slots are summed into shared global slots.
            When the modal cardinality has reached ``max_contexts`` there is no
            novel slot, so slots from ``km`` on are zeroed.
        ``'cue'``
            ``(P_modal, C, Q)``; the cue axis passes through unchanged.
        ``'transition'``
            ``(P_modal, C, C)``; both context axes are realigned.
        ``'labels'``
            ``(P_modal,)`` global label of each particle's sampled context,
            ``nan`` when unmatched.

    Returns
    -------
    numpy.ndarray
        The global-frame aggregate; shape and fill value depend on ``mode``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the five recognised values.
    """
    c_slots = model.max_contexts + 1              # context slots incl. novel
    km = int(alignment["K"])                      # modal context cardinality
    modal_idx = np.asarray(
        alignment["modal_particle_indices"], dtype=int
    ).reshape(-1)                                 # particles at that cardinality
    assignment = alignment["assignment"]
    n_modal = modal_idx.size

    if mode == "labels":
        out = np.full(n_modal, np.nan)
        context = model.D.context
        for idx, p in enumerate(modal_idx):
            local = int(context[p])               # sampled local slot
            if 0 <= local < assignment.shape[1]:
                target = assignment[p, local]     # corresponding global slot
                if target >= 0:
                    out[idx] = target
        return out

    x = np.asarray(x, dtype=float)
    if mode == "cue":
        out = np.zeros((n_modal, c_slots, x.shape[2]))
    elif mode == "transition":
        out = np.zeros((n_modal, c_slots, c_slots))
    elif mode in ("overwrite", "add"):
        out = np.zeros((n_modal, c_slots))
    else:
        raise ValueError(
            "RealTimeCOIN:BadScatterMode: mode must be one of 'overwrite', "
            "'add', 'cue', 'transition' or 'labels'; got %r." % (mode,)
        )

    for idx, p in enumerate(modal_idx):
        targets = assignment[p, :c_slots]
        keep = (targets >= 0) & (targets < c_slots)
        local = np.flatnonzero(keep)              # local slots with a label
        target = targets[local]                   # their global slots
        if mode == "overwrite":
            # Explicit loop: fancy-index assignment with repeated indices has no
            # documented last-wins guarantee, and last-wins IS the contract.
            for lo, tg in zip(local, target):
                out[idx, tg] = x[p, lo]
        elif mode == "add":
            np.add.at(out, (idx, target), x[p, local])
            if km >= model.max_contexts:
                # All contexts instantiated: no novel slot, drop trailing slots.
                out[idx, km:] = 0.0
        elif mode == "cue":
            np.add.at(out, (idx, target), x[p, local, :])
        else:   # "transition": realign both axes
            np.add.at(
                out,
                (idx, target[:, None], target[None, :]),
                x[p][np.ix_(local, local)],
            )
    return out


def active_summary_contexts(model):
    """Global context slots worth reporting in a summary.

    The slots whose PREDICTED probability is strictly positive. When no slot
    carries mass it falls back to slot 0, so callers always receive at least one
    context.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    numpy.ndarray
        0-based global context slot indices; never empty. (MATLAB's
        ``activeSummaryContexts`` returns the same list 1-based.)
    """
    weights = context_probability_vector(model, "predicted")
    active = np.flatnonzero(weights > 0)
    if active.size == 0:
        return np.array([0])
    return active


def clamp_active_summary_contexts(model, alignment):
    """Active summary contexts restricted to the aligned label set.

    :func:`active_summary_contexts` filtered to slots that carry a valid global
    label in ``alignment`` (index ``< K``). If the restriction empties the list
    but the alignment does hold contexts, it falls back to context 0 - COIN's
    convention of always summarising at least the first global context.

    Shared by the per-global-context density queries so the clamp/fallback
    boilerplate lives in one place. Pure read; mutates nothing.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    alignment : dict
        Alignment structure.

    Returns
    -------
    numpy.ndarray
        0-based global context slot indices, all ``< alignment["K"]``.
    """
    active = active_summary_contexts(model)
    active = active[active < int(alignment["K"])]
    if active.size == 0 and int(alignment["K"]) > 0:
        return np.array([0])
    return active


def context_probability_vector(model, kind="predicted"):
    """Global-frame context probability vector.

    The globally aligned counterpart of
    :func:`realtimecoin.context.local_context_probability_vector`: a normalised
    probability over the GLOBAL context slots, averaged over the modal
    particles. Triggers (and caches) the lazy context alignment.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    kind : {'predicted', 'responsibilities', 'count'}, optional
        Quantity to summarise: the mean aligned predicted probabilities (before
        the observation), the mean aligned responsibilities (after it), or the
        empirical frequency of the sampled global contexts. ``'predicted'`` by
        default.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` weights summing to one, or all zeros when no
        mass exists. Non-finite entries are treated as zero.
    """
    from .context import context_probability_vector_core   # local: avoid a cycle

    alignment = ensure_context_alignment(model)
    d = model.D
    # Integer-centred bin edges -0.5, 0.5, ..., C - 0.5, so each 0-based global
    # label k lands in its own bin k (MATLAB uses 0.5 .. Cmax+0.5 for 1-based).
    edges = np.arange(model.max_contexts + 2) - 0.5
    weights = context_probability_vector_core(
        kind,
        lambda: global_context_weights(model, d.predicted_probabilities, alignment),
        lambda: global_context_weights(model, d.responsibilities, alignment),
        lambda: global_sampled_contexts(model, alignment),
        edges,
    )
    return renormalize_global_weights(weights)


# --------------------------------------------------------------------------
# Per-context local belief readers
# --------------------------------------------------------------------------


def local_dynamics_distribution(model, local, p):
    """Gaussian dynamics belief for one local context of one particle (scalar).

    Returns the mean and covariance of the joint ``[retention; drift]``
    posterior. The posterior sufficient statistics (``dynamics_mean`` /
    ``dynamics_covar``) are used when they exist for this slot; otherwise the
    PRIOR is used - the stored ``[retention; drift]`` sample with a diagonal
    covariance from the prior precisions. Both fields stay ``None`` until
    ``sample_parameters`` has run once, which is exactly the MATLAB
    ``isfield``/size guard.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    local : int
        Local context slot.
    p : int
        Particle index.

    Returns
    -------
    mu : numpy.ndarray
        ``(2,)`` mean ``[retention, drift]``.
    covar : numpy.ndarray
        ``(2, 2)`` regularised covariance.
    """
    d = model.D
    dynamics_mean = d.dynamics_mean
    if (
        dynamics_mean is not None
        and dynamics_mean.shape[0] > p
        and dynamics_mean.shape[1] > local
    ):
        mu = np.asarray(dynamics_mean[p, local], dtype=float).reshape(-1)   # (2,)
    else:
        mu = np.array(
            [float(d.retention[p, local]), float(d.drift[p, local])]
        )

    dynamics_covar = d.dynamics_covar
    if (
        dynamics_covar is not None
        and dynamics_covar.shape[0] > p
        and dynamics_covar.shape[1] > local
    ):
        covar = np.asarray(dynamics_covar[p, local], dtype=float)           # (2, 2)
    else:
        covar = np.diag(
            [
                float(precision_to_variance(model.prior_precision_retention)),
                float(precision_to_variance(model.prior_precision_drift)),
            ]
        )
    return mu, regularize_covariance(covar)


def local_bias_distribution(model, local, p):
    """Gaussian observation-bias belief for one local context of one particle.

    Posterior sufficient statistics (``bias_mean`` / ``bias_var``) when they
    exist for this slot, else the prior (the stored bias sample with the prior
    bias precision as a variance). With bias inference disabled the bias is
    deterministically zero.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    local : int
        Local context slot.
    p : int
        Particle index.

    Returns
    -------
    mu : float
        Bias mean (0 when ``infer_bias`` is false).
    variance : float
        Bias variance, finite and non-negative. A non-finite (completely
        uninformative) prior precision becomes ``1 / eps``.
    """
    d = model.D
    bias_mean = d.bias_mean
    if (
        model.infer_bias
        and bias_mean is not None
        and bias_mean.shape[0] > p
        and bias_mean.shape[1] > local
    ):
        mu = float(bias_mean[p, local])
        variance = float(d.bias_var[p, local])
    elif model.infer_bias:
        mu = float(d.bias[p, local])
        variance = float(precision_to_variance(model.prior_precision_bias))
    else:
        mu = 0.0
        variance = 0.0
    if not np.isfinite(variance):
        variance = 1.0 / EPS   # unbounded variance, kept finite
    return mu, max(variance, 0.0)


def local_cue_probability(model, local, p):
    """Cue-emission distribution of one local context of one particle.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose particle state is read.
    local : int
        Local context slot.
    p : int
        Particle index.

    Returns
    -------
    numpy.ndarray
        ``(Q,)`` cue probabilities summing to one.
    """
    return normalize_probability(model.D.local_cue_matrix[p, local, :])

"""Cross-run context alignment for the ensemble (SPEC_ensemble Part 10).

Translation of the MATLAB private helpers
``@RealTimeCOINEnsemble/private/ensembleContextAlignment.m``,
``ensembleContextVector.m`` and ``ensembleContextDensity.m``. They back the
context-indexed queries of
:class:`~realtimecoin.ensemble.RealTimeCOINEnsemble`:
:func:`ensemble_context_vector` serves ``responsibilities_vector``,
``predicted_context_probabilities_vector`` and ``sampled_context_count``;
:func:`ensemble_context_density` serves ``state_given_context_probability`` and
``state_feedback_given_context_probability``; and
``stationary_context_probabilities`` consumes
:func:`ensemble_context_alignment` directly (its zero-fill accumulator is
renormalised rather than divided by ``R``), exactly as the MATLAB is split.

Why an alignment is needed
--------------------------
Context labels are per-member and arbitrary: each member already aligns its own
particles onto a member-local global frame
(:func:`realtimecoin.alignment.context_alignment`), but member 2's context 1 need
not be member 1's context 1. Context-indexed readouts therefore cannot be
averaged slot by slot. Phase 2 maps every member's contexts onto ONE common
reference frame and averages there. This is the real-time analogue of
``COIN.m``'s ``find_optimal_context_labels`` + ``integrate_over_runs``.

The contract (``docs/SPEC_ensemble.md`` Part 10)
------------------------------------------------
* **Reference frame** (10.2). ``Kref = max_r K_r``, ties broken by the LOWEST
  member index; that member's contexts ``0 .. Kref - 1`` are the reference
  labels.
* **Matching** (10.2). For each member ``r``, a minimum-total-cost assignment of
  its ``K_r`` contexts onto distinct reference labels, the cost between member
  context ``i`` and reference context ``j`` being the Euclidean distance between
  their prototype state means (``global_contexts["state_mean"]``). Since
  ``K_r <= Kref`` every member context is matched and ``Kref - K_r`` reference
  labels are left unmatched for that member. The reference member matches
  itself by the identity (its own cost matrix has a zero diagonal, which is the
  unique optimum unless two of its prototypes coincide exactly). The NOVEL
  context of every member always maps to the reference novel slot.
* **Probability vectors** (10.3): ZERO-FILL. A run lacking reference context
  ``j`` contributes 0 to slot ``j``; each member's novel mass (and any residual
  beyond it) goes to the novel slot. Divide by ``R``, never omit - which is what
  keeps the aligned average summing to exactly 1.
* **Per-context densities** (10.3): NaN-OMIT. A run lacking reference context
  ``j`` has no density for ``j`` (undefined, not zero), so ``j``'s density is the
  mean over the contributing runs only; a label no run holds is absent from the
  output.
* The alignment is a deterministic function of the members' current states, so
  it draws no randomness and inherits reproducibility and executor invariance.
  It is READ-ONLY on the members' PARTICLE state and generators: it calls their
  public ``context_alignment`` and query methods, which touch only each member's
  own alignment cache and its ``alignment_seed`` (the warm start the member's
  NEXT alignment may reuse - ``computeContextAlignment.m`` writes it too).

Order dependence (a caveat on SPEC 10.5.3)
------------------------------------------
The reference LABELS are the reference member's own labels, so the readout is
invariant to member order only up to a relabelling: if two members tie for the
most contexts, reordering them can hand the frame to the other one and permute
the output. Both this port and the MATLAB original have that property (both
pick the first maximiser). With ``argmax_r K_r`` unique - and in practice
whenever the members are driven by one shared stream, so they order their
contexts alike - the readout is invariant outright.

Implementation notes
--------------------
* MATLAB solves the rectangular assignment with ``matchpairs`` plus a
  "cost of non-assignment" large enough to force every row to match; SciPy's
  :func:`scipy.optimize.linear_sum_assignment` matches every row of a wide
  (``K_r x Kref``, ``K_r <= Kref``) matrix by construction, so no such trick is
  needed. As always with a linear assignment, TIE-BREAKING may differ from
  MATLAB when several matchings are equally optimal (identical prototypes);
  the total cost is the same.
* Labels here are the 0-based member-local global context labels used by
  :mod:`realtimecoin.queries_aligned`, and the reference labels use the same
  convention, so ``runs == 1`` reduces to the single member exactly
  (SPEC 10.5.1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

__all__ = [
    "ensemble_context_alignment",
    "ensemble_context_vector",
    "ensemble_context_density",
]


def _member_reference_cost(member_mean, ref_mean, k_r, k_ref):
    """Prototype-distance cost between one member's contexts and the reference.

    Translated from the local function ``memberRefCost`` of
    ``ensembleContextAlignment.m``. The scalar branch is the absolute
    difference of the per-context means; the multi-dimensional branch is the
    Euclidean distance between the per-context mean VECTORS.

    Parameters
    ----------
    member_mean : numpy.ndarray
        Member prototype state means: ``(K_r,)`` for the scalar model,
        ``(K_r, N)`` (one context per ROW) otherwise. May carry trailing
        columns/rows beyond ``k_r``; they are ignored.
    ref_mean : numpy.ndarray
        Reference member's prototype state means, same layout.
    k_r : int
        Member context count.
    k_ref : int
        Reference context count.

    Returns
    -------
    numpy.ndarray
        ``(K_r, Kref)`` non-negative cost matrix.
    """
    member_mean = np.asarray(member_mean, dtype=float)
    ref_mean = np.asarray(ref_mean, dtype=float)

    if member_mean.ndim == 1:
        # Scalar model: |mean_i - mean_j|.        # (K_r, 1) - (1, Kref)
        return np.abs(
            member_mean[:k_r, None] - ref_mean[None, :k_ref]
        )                                                        # (K_r, Kref)

    # MD model: ||mean_i - mean_j||_2 between state-mean vectors.
    diff = (
        member_mean[:k_r, None, :] - ref_mean[None, :k_ref, :]
    )                                                            # (K_r, Kref, N)
    return np.sqrt(np.sum(diff ** 2, axis=-1))                   # (K_r, Kref)


def _assign_member_contexts(cost):
    """Match each member context (row) to a distinct reference label (column).

    Translated from the local function ``assignMemberContexts`` of
    ``ensembleContextAlignment.m``, minus its "force every row to match"
    machinery: on a wide cost matrix SciPy already matches every row.

    Parameters
    ----------
    cost : numpy.ndarray
        ``(K_r, Kref)`` cost matrix with ``K_r <= Kref``.

    Returns
    -------
    numpy.ndarray
        ``(K_r,)`` int array; entry ``i`` is the reference label assigned to
        member context ``i``.
    """
    c = np.asarray(cost, dtype=float)
    if c.size == 0:
        return np.zeros(c.shape[0], dtype=int)

    # Defensive, mirroring realtimecoin.alignment.linear_assignment (and its
    # MATLAB source linearAssignment.m) rather than assignMemberContexts.m,
    # which has no such guard: SciPy REJECTS a non-finite cost outright, so map
    # one to a large finite sentinel instead. Prototype means are finite, so
    # this never fires in practice.
    large = 1e100
    c = np.where(np.isfinite(c), c, large)
    c = np.fmin(c, large)

    rows, cols = _linear_sum_assignment(c)
    perm = np.zeros(c.shape[0], dtype=int)
    perm[rows] = cols
    return perm


def ensemble_context_alignment(ensemble):
    """Match every member's contexts onto one common reference frame.

    Translated from ``private/ensembleContextAlignment.m`` (SPEC 10.2). The
    reference member is the one holding the most contexts (ties broken by the
    lowest member index); every member's contexts are then matched onto
    distinct reference labels by minimum total prototype distance.

    Parameters
    ----------
    ensemble : realtimecoin.ensemble.RealTimeCOINEnsemble
        Ensemble whose members are aligned. Read-only; draws no randomness.

    Returns
    -------
    dict
        ``{"k_ref": int, "ref_index": int, "k": (R,) int array,
        "perm": list of (K_r,) int arrays}``, where ``perm[r][i]`` is the
        0-based reference label of member ``r``'s context ``i`` (an empty array
        when ``k[r] == 0``).
    """
    members = ensemble.members
    r_count = len(members)

    k = np.zeros(r_count, dtype=int)                             # (R,)
    prototypes = [None] * r_count
    for r, member in enumerate(members):
        alignment = member.context_alignment()
        k[r] = int(alignment["K"])
        prototypes[r] = alignment["global_contexts"]

    # argmax returns the FIRST maximiser, i.e. MATLAB's tie-break by lowest
    # index (the arrays here are 0-based, the MATLAB ones 1-based).
    ref_index = int(np.argmax(k)) if r_count else 0
    k_ref = int(k[ref_index]) if r_count else 0

    if k_ref == 0:
        return {
            "k_ref": 0,
            "ref_index": ref_index,
            "k": k,
            "perm": [np.zeros(0, dtype=int) for _ in range(r_count)],
        }

    ref_mean = prototypes[ref_index]["state_mean"]   # (Kref,) or (Kref, N)
    perm = []
    for r in range(r_count):
        k_r = int(k[r])
        if k_r == 0:
            perm.append(np.zeros(0, dtype=int))
            continue
        cost = _member_reference_cost(
            prototypes[r]["state_mean"], ref_mean, k_r, k_ref
        )                                                        # (K_r, Kref)
        perm.append(_assign_member_contexts(cost))               # (K_r,)

    return {"k_ref": k_ref, "ref_index": ref_index, "k": k, "perm": perm}


def ensemble_context_vector(ensemble, member_fn):
    """Zero-fill cross-run average of a context probability row (SPEC 10.3).

    Translated from ``private/ensembleContextVector.m``. Real mass goes to the
    matched reference slots; each member's novel mass (its slot ``K_r``, plus
    any residual beyond) goes to the reference novel slot ``Kref``; the
    accumulator is divided by ``R``, so the result sums to 1.

    Parameters
    ----------
    ensemble : realtimecoin.ensemble.RealTimeCOINEnsemble
        Ensemble to read.
    member_fn : callable
        ``member -> (max_contexts + 1,)`` probability row in that member's own
        frame, e.g. ``lambda m: m.responsibilities_vector()``.

    Returns
    -------
    numpy.ndarray
        ``(max_contexts + 1,)`` row in the reference frame.
    """
    alignment = ensemble_context_alignment(ensemble)
    members = ensemble.members
    r_count = len(members)
    c_max = int(members[0].max_contexts) + 1
    novel_slot = int(alignment["k_ref"])                 # 0-based novel index

    acc = np.zeros(c_max)                                        # (Cmax,)
    for r, member in enumerate(members):
        v = np.asarray(member_fn(member), dtype=float).reshape(-1)   # (Cmax,)
        k_r = int(alignment["k"][r])
        # Real contexts -> matched reference slots.
        for i in range(k_r):
            acc[alignment["perm"][r][i]] += v[i]
        # Member novel slot (index K_r) and any residual beyond -> novel slot.
        if k_r < c_max:
            acc[novel_slot] += np.sum(v[k_r:])
    return acc / r_count


def ensemble_context_density(ensemble, member_fn):
    """NaN-omit cross-run average of a per-context density map (SPEC 10.3).

    Translated from ``private/ensembleContextDensity.m``. The density of
    reference context ``j`` is the mean over only the runs that have a context
    matched to ``j``; if no run has ``j``, ``j`` is absent from the result.

    Parameters
    ----------
    ensemble : realtimecoin.ensemble.RealTimeCOINEnsemble
        Ensemble to read.
    member_fn : callable
        ``member -> {label: (K,) density}`` in that member's own frame, e.g.
        ``lambda m: m.state_given_context_probability(values)``.

    Returns
    -------
    dict
        ``{0-based reference label: (K,) density}``.
    """
    alignment = ensemble_context_alignment(ensemble)
    k_ref = int(alignment["k_ref"])
    if k_ref == 0:
        return {}

    sums = [None] * k_ref
    counts = np.zeros(k_ref, dtype=int)                          # (Kref,)
    for r, member in enumerate(ensemble.members):
        densities = member_fn(member)   # keyed by member r's labels 0..K_r-1
        k_r = int(alignment["k"][r])
        for i in range(k_r):
            if i not in densities:
                continue
            j = int(alignment["perm"][r][i])
            d = np.asarray(densities[i], dtype=float)
            sums[j] = d.copy() if sums[j] is None else sums[j] + d
            counts[j] += 1

    return {
        j: sums[j] / counts[j] for j in range(k_ref) if counts[j] > 0
    }

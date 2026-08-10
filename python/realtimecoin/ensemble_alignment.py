"""Cross-run context alignment for the ensemble - PHASE 2 STUB (unit E1).

Signature-only translation of the MATLAB private helpers
``@RealTimeCOINEnsemble/private/ensembleContextAlignment.m``,
``ensembleContextVector.m`` and ``ensembleContextDensity.m``. Every function
here raises :class:`NotImplementedError`; unit E1 fills them in and wires the
six context-indexed queries of
:class:`~realtimecoin.ensemble.RealTimeCOINEnsemble` (``responsibilities_vector``,
``predicted_context_probabilities_vector``, ``sampled_context_count``,
``stationary_context_probabilities``, ``state_given_context_probability``,
``state_feedback_given_context_probability``) onto them.

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
  member index; that member's contexts ``1 .. Kref`` are the reference labels.
* **Matching** (10.2). For each member ``r``, a minimum-total-cost assignment of
  its ``K_r`` contexts onto distinct reference labels, the cost between member
  context ``i`` and reference context ``j`` being the Euclidean distance between
  their prototype state means (``global_contexts.state_mean``). Since
  ``K_r <= Kref`` every member context is matched and ``Kref - K_r`` reference
  labels are left unmatched for that member. The reference member matches
  itself by the identity. The NOVEL context of every member always maps to the
  reference novel slot.
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

Implementation notes for unit E1
--------------------------------
* MATLAB solves the rectangular assignment with ``matchpairs``; the Python
  equivalent is :func:`scipy.optimize.linear_sum_assignment` on the
  ``K_r x Kref`` cost matrix (already a dependency; it needs no
  "cost of non-assignment" trick, since it matches every row of a wide matrix).
* Labels in this package are the member-local global context labels used by
  :mod:`realtimecoin.queries_aligned`; keep the ensemble's reference labels in
  the same convention so ``runs == 1`` reduces to the single member exactly
  (SPEC 10.5.1).
"""

from __future__ import annotations

__all__ = [
    "ensemble_context_alignment",
    "ensemble_context_vector",
    "ensemble_context_density",
]

#: Message shared by every stub, so the owning unit is obvious from a traceback.
_PHASE2 = "cross-run context alignment is Phase 2; implemented by unit E1."


def ensemble_context_alignment(ensemble):
    """Match every member's contexts onto one common reference frame.

    Translated from ``private/ensembleContextAlignment.m`` (SPEC 10.2).

    Parameters
    ----------
    ensemble : realtimecoin.ensemble.RealTimeCOINEnsemble
        Ensemble whose members are aligned. Read-only; draws no randomness.

    Returns
    -------
    dict
        ``{"k_ref": int, "ref_index": int, "k": (R,) int array,
        "perm": list of (K_r,) int arrays}``, where ``perm[r][i]`` is the
        reference label of member ``r``'s context ``i`` (empty when
        ``k[r] == 0``).

    Raises
    ------
    NotImplementedError
        Always, in this Phase-1 build.
    """
    raise NotImplementedError("ensemble_context_alignment: " + _PHASE2)


def ensemble_context_vector(ensemble, member_fn):
    """Zero-fill cross-run average of a context probability row (SPEC 10.3).

    Translated from ``private/ensembleContextVector.m``. Real mass goes to the
    matched reference slots; each member's novel mass (its slot ``K_r + 1``,
    plus any residual beyond) goes to the reference novel slot; the accumulator
    is divided by ``R``, so the result sums to 1.

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

    Raises
    ------
    NotImplementedError
        Always, in this Phase-1 build.
    """
    raise NotImplementedError("ensemble_context_vector: " + _PHASE2)


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
        ``{reference_label: (K,) density}``.

    Raises
    ------
    NotImplementedError
        Always, in this Phase-1 build.
    """
    raise NotImplementedError("ensemble_context_density: " + _PHASE2)

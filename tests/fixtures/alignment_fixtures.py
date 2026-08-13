"""Particle-state fixtures with a KNOWN local-to-global context permutation.

Translated from ``tests/+testutil/alignmentFixture.m`` and
``tests/+testutil/largeAssignmentFixture.m``. Every number matches the MATLAB
originals; only the layout and the label base change:

* arrays are particles-leading (``(P, C)`` instead of ``(C, P)``, and
  ``(P, C, 2)`` instead of ``(2, C, P)``);
* context labels are 0-based, so MATLAB's ``D.context = [1 2 1]`` becomes
  ``[0 1 0]`` and MATLAB's ``localToGlobal = [2 1]`` becomes ``[1 0]``.

VALUES that happen to be derived from a 1-based context number in MATLAB (the
state means, retentions and drifts of the ten-context fixture) keep their
MATLAB numbering, because they are data, not indices - hence the explicit
``label = g + 1`` in :func:`large_assignment_fixture`.

Both fixtures are constructed so the optimal assignment is UNIQUE (the
per-context state means are well separated), which is what lets the tests assert
an exact permutation despite the solver's tie-breaking being unspecified.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOIN
from realtimecoin.state import ParticleState

__all__ = [
    "alignment_fixture",
    "large_assignment_fixture",
    "md_alignment_fixture",
    "MD_GLOBAL_STATE_MEAN",
    "MD_GLOBAL_RETENTION",
    "MD_GLOBAL_DRIFT",
    "ALIGNMENT_GLOBAL_STATE_MEAN",
    "ALIGNMENT_GLOBAL_RETENTION",
    "ALIGNMENT_GLOBAL_DRIFT",
    "ALIGNMENT_GLOBAL_CUE",
    "ALIGNMENT_GLOBAL_TRANSITION",
    "ALIGNMENT_GLOBAL_PREDICTED",
    "ALIGNMENT_GLOBAL_RESPONSIBILITIES",
]

#: Ground truth of :func:`alignment_fixture`, indexed by 0-based GLOBAL label.
ALIGNMENT_GLOBAL_STATE_MEAN = np.array([-1.0, 1.0])
ALIGNMENT_GLOBAL_RETENTION = np.array([0.8, 0.6])
ALIGNMENT_GLOBAL_DRIFT = np.array([-0.1, 0.1])
ALIGNMENT_GLOBAL_CUE = np.array([[0.9, 0.1], [0.1, 0.9]])
#: Rows are the two real global contexts; the third column is the novel context.
ALIGNMENT_GLOBAL_TRANSITION = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
#: Franchise / predicted / posterior weights over [context 0, context 1, novel].
ALIGNMENT_GLOBAL_BETA = np.array([0.55, 0.35, 0.1])
ALIGNMENT_GLOBAL_PREDICTED = np.array([0.7, 0.2, 0.1])
ALIGNMENT_GLOBAL_RESPONSIBILITIES = np.array([0.6, 0.3, 0.1])

#: Ground truth of :func:`md_alignment_fixture` (N = 2), by 0-based label.
MD_GLOBAL_STATE_MEAN = np.array([[-1.0, -1.0], [1.0, 1.0]])
MD_GLOBAL_RETENTION = np.array([0.8, 0.6])      # A = retention * I_N
MD_GLOBAL_DRIFT = np.array([-0.1, 0.1])         # d = drift * 1_N


def alignment_fixture(include_non_modal=False, *, rng=0):
    """Two-context particle fixture with a deterministic label permutation.

    Three particles each carry two contexts; particle 1 (MATLAB's particle 2)
    labels them in the SWAPPED order, so the alignment has a known ground truth:
    particles 0 and 2 must keep the identity map and particle 1 must map back to
    ``[1, 0]``. Slot 2 is the novel context.

    Parameters
    ----------
    include_non_modal : bool, optional
        When true a fourth particle with a single, far-off-prototype context is
        appended. It is NOT at the modal cardinality, so every modal summary
        must ignore it - which is what exercises the modal-subset logic.
    rng : int or numpy.random.Generator or None, optional
        Random source for the model. The fixture state is fully specified, so
        the seed only matters if a query draws.

    Returns
    -------
    RealTimeCOIN
        A model whose particle state is the fixture, at ``trial = 1`` with cue
        value ``1`` registered.
    """
    p_count = 3 + int(bool(include_non_modal))
    max_contexts = 3
    c_slots = max_contexts + 1     # 4: three real slots plus the novel slot
    q_cols = 2                     # two cue columns, as in the MATLAB fixture

    zeros_pc = lambda: np.zeros((p_count, c_slots))          # noqa: E731

    d = ParticleState(
        n_active=2 * np.ones(p_count, dtype=int),
        # MATLAB [1 2 1 1] with 1-based labels.
        context=np.array([0, 1, 0] + [0] * (p_count - 3), dtype=int),
        n_cues=1,
        i_resampled=np.arange(p_count),
        n_context=np.zeros((p_count, c_slots, c_slots)),
        n_cue=np.zeros((p_count, c_slots, q_cols)),
        global_cue_probabilities=0.5 * np.ones((p_count, q_cols)),
        state_filtered_mean=zeros_pc(),
        state_filtered_var=np.ones((p_count, c_slots)),
        state_mean=zeros_pc(),
        state_var=np.ones((p_count, c_slots)),
        state_feedback_mean=zeros_pc(),
        state_feedback_var=np.ones((p_count, c_slots)),
        retention=zeros_pc(),
        drift=zeros_pc(),
        bias=zeros_pc(),
        bias_mean=zeros_pc(),
        bias_var=zeros_pc(),
        dynamics_mean=np.zeros((p_count, c_slots, 2)),
        dynamics_covar=np.zeros((p_count, c_slots, 2, 2)),
        local_cue_matrix=np.zeros((p_count, c_slots, q_cols)),
        local_transition_matrix=np.zeros((p_count, c_slots, c_slots)),
        global_transition_probabilities=zeros_pc(),
        predicted_probabilities=zeros_pc(),
        responsibilities=zeros_pc(),
        prior_probabilities=zeros_pc(),
        probability_cue=np.ones((p_count, c_slots)),
        probability_state_feedback=np.ones((p_count, c_slots)),
    )
    d.previous_context = d.context.copy()

    for p in range(3):
        # Particle 1 labels the same two contexts the other way round.
        local_to_global = np.array([1, 0]) if p == 1 else np.array([0, 1])

        for local in range(2):
            g = local_to_global[local]
            d.state_filtered_mean[p, local] = ALIGNMENT_GLOBAL_STATE_MEAN[g]
            d.state_filtered_var[p, local] = 0.02
            d.state_mean[p, local] = ALIGNMENT_GLOBAL_STATE_MEAN[g]
            d.state_var[p, local] = 0.02
            d.state_feedback_mean[p, local] = ALIGNMENT_GLOBAL_STATE_MEAN[g]
            d.state_feedback_var[p, local] = 0.02
            d.retention[p, local] = ALIGNMENT_GLOBAL_RETENTION[g]
            d.drift[p, local] = ALIGNMENT_GLOBAL_DRIFT[g]
            d.dynamics_mean[p, local] = [
                ALIGNMENT_GLOBAL_RETENTION[g],
                ALIGNMENT_GLOBAL_DRIFT[g],
            ]
            d.dynamics_covar[p, local] = np.diag([0.01, 0.01])
            d.local_cue_matrix[p, local, :] = ALIGNMENT_GLOBAL_CUE[g, :]
            d.global_transition_probabilities[p, local] = ALIGNMENT_GLOBAL_BETA[g]
            d.predicted_probabilities[p, local] = ALIGNMENT_GLOBAL_PREDICTED[g]
            d.responsibilities[p, local] = ALIGNMENT_GLOBAL_RESPONSIBILITIES[g]
            d.prior_probabilities[p, local] = ALIGNMENT_GLOBAL_PREDICTED[g]

        # Transition rows, written in LOCAL labels but carrying global values.
        for local in range(2):
            global_from = local_to_global[local]
            for dest_local in range(2):
                global_to = local_to_global[dest_local]
                d.local_transition_matrix[p, local, dest_local] = (
                    ALIGNMENT_GLOBAL_TRANSITION[global_from, global_to]
                )
            # Column 2 is the novel context, identical in both label orders.
            d.local_transition_matrix[p, local, 2] = ALIGNMENT_GLOBAL_TRANSITION[
                global_from, 2
            ]

        # The novel slot itself: uninformative cue row, absorbing transition.
        d.local_cue_matrix[p, 2, :] = [0.5, 0.5]
        d.local_transition_matrix[p, 2, 2] = 1.0
        d.global_transition_probabilities[p, 2] = ALIGNMENT_GLOBAL_BETA[2]
        d.predicted_probabilities[p, 2] = ALIGNMENT_GLOBAL_PREDICTED[2]
        d.responsibilities[p, 2] = ALIGNMENT_GLOBAL_RESPONSIBILITIES[2]
        d.prior_probabilities[p, 2] = ALIGNMENT_GLOBAL_PREDICTED[2]

    if include_non_modal:
        p = 3
        d.n_active[p] = 1
        d.context[p] = 0
        d.previous_context[p] = 0
        d.state_filtered_mean[p, 0] = 10.0
        d.state_filtered_var[p, 0] = 0.02
        d.state_mean[p, 0] = 10.0
        d.state_var[p, 0] = 0.02
        d.state_feedback_mean[p, 0] = 10.0
        d.state_feedback_var[p, 0] = 0.02
        d.retention[p, 0] = 0.1
        d.drift[p, 0] = 10.0
        d.dynamics_mean[p, 0] = [0.1, 10.0]
        d.dynamics_covar[p, 0] = np.diag([0.01, 0.01])
        d.local_cue_matrix[p, 0, :] = [0.5, 0.5]
        d.local_transition_matrix[p, 0, 0] = 1.0
        d.global_transition_probabilities[p, 0] = 1.0
        d.predicted_probabilities[p, 0] = 1.0
        d.responsibilities[p, 0] = 1.0
        d.prior_probabilities[p, 0] = 1.0

    return RealTimeCOIN.from_state(
        {"num_particles": p_count, "max_contexts": max_contexts},
        d,
        trial=1,
        cue_values=(1,),
        rng=rng,
    )


def large_assignment_fixture(*, rng=0):
    """Ten-context, two-particle fixture whose labels are exactly reversed.

    Particle 0 labels ten contexts in canonical order and particle 1 in reverse
    order, so the alignment must recover the reverse permutation. The modal
    cardinality equals ``max_contexts``, so there is NO novel slot to label -
    which also exercises the ``km >= max_contexts`` trailing-slot rule of
    ``scatter_to_global``.

    Parameters
    ----------
    rng : int or numpy.random.Generator or None, optional
        Random source for the model.

    Returns
    -------
    RealTimeCOIN
        A model whose particle state is the fixture, at ``trial = 1``.
    """
    p_count = 2
    max_contexts = 10
    c_slots = max_contexts + 1     # 11
    n_ctx = 10                     # contexts actually instantiated per particle

    d = ParticleState(
        n_active=max_contexts * np.ones(p_count, dtype=int),
        context=np.zeros(p_count, dtype=int),
        n_cues=1,
        i_resampled=np.arange(p_count),
        n_context=np.zeros((p_count, c_slots, c_slots)),
        n_cue=np.zeros((p_count, c_slots, 1)),
        global_cue_probabilities=np.ones((p_count, 1)),
        state_filtered_mean=np.zeros((p_count, c_slots)),
        state_filtered_var=0.01 * np.ones((p_count, c_slots)),
        retention=np.zeros((p_count, c_slots)),
        drift=np.zeros((p_count, c_slots)),
        bias=np.zeros((p_count, c_slots)),
        bias_mean=np.zeros((p_count, c_slots)),
        bias_var=np.ones((p_count, c_slots)),
        dynamics_mean=np.zeros((p_count, c_slots, 2)),
        dynamics_covar=np.tile(np.eye(2), (p_count, c_slots, 1, 1)),
        local_cue_matrix=np.ones((p_count, c_slots, 1)),
        local_transition_matrix=np.zeros((p_count, c_slots, c_slots)),
        global_transition_probabilities=np.zeros((p_count, c_slots)),
        predicted_probabilities=np.zeros((p_count, c_slots)),
        responsibilities=np.zeros((p_count, c_slots)),
        prior_probabilities=np.zeros((p_count, c_slots)),
        probability_cue=np.ones((p_count, c_slots)),
        probability_state_feedback=np.ones((p_count, c_slots)),
    )
    d.previous_context = d.context.copy()

    for p in range(p_count):
        local_to_global = (
            np.arange(n_ctx) if p == 0 else np.arange(n_ctx - 1, -1, -1)
        )
        for local in range(n_ctx):
            g = local_to_global[local]
            # MATLAB derives the VALUES from the 1-based global index.
            label = float(g) + 1.0
            d.state_filtered_mean[p, local] = label
            d.retention[p, local] = 0.5 + 0.01 * label
            d.drift[p, local] = 0.1 * label
            d.dynamics_mean[p, local] = [
                d.retention[p, local],
                d.drift[p, local],
            ]
            d.local_transition_matrix[p, local, local] = 1.0
            d.global_transition_probabilities[p, local] = 0.1
            d.predicted_probabilities[p, local] = 0.1
            d.responsibilities[p, local] = 0.1
            d.prior_probabilities[p, local] = 0.1

    # MATLAB aliases these onto the filtered moments BEFORE the loop and then
    # re-writes the means inside it; copying afterwards gives the identical
    # result (the variances are never touched in the loop).
    d.state_mean = d.state_filtered_mean.copy()
    d.state_var = d.state_filtered_var.copy()
    d.state_feedback_mean = d.state_filtered_mean.copy()
    d.state_feedback_var = d.state_filtered_var.copy()

    return RealTimeCOIN.from_state(
        {"num_particles": p_count, "max_contexts": max_contexts},
        d,
        trial=1,
        cue_values=(1,),
        rng=rng,
    )


def md_alignment_fixture(*, state_dim=2, rng=0):
    """Multi-dimensional counterpart of :func:`alignment_fixture`.

    Two particles, two contexts, particle 1's labels swapped - the same design
    as the scalar fixture, but with an ``N``-vector state, a full ``(N, N)``
    covariance and the augmented dynamics ``Theta = [A | d]``. It has no MATLAB
    original: ``alignmentFixture.m`` is scalar only, so this exists to cover
    ``update_global_contexts_md`` / ``assignment_cost_matrix_md`` /
    ``diagnostics_md``, which the MATLAB alignment test never reaches.

    Parameters
    ----------
    state_dim : int, optional
        Latent-state dimension ``N``. Default 2.
    rng : int or numpy.random.Generator or None, optional
        Random source for the model.

    Returns
    -------
    RealTimeCOIN
        A model whose particle state is the fixture, at ``trial = 1``.
    """
    p_count = 2
    max_contexts = 3
    c_slots = max_contexts + 1
    n = state_dim
    q_cols = 2

    state_mean = np.repeat(
        MD_GLOBAL_STATE_MEAN[:, :1], n, axis=1
    )                                                  # (2, N): [-1...], [1...]
    theta = np.stack(
        [
            np.hstack(
                [
                    MD_GLOBAL_RETENTION[g] * np.eye(n),
                    MD_GLOBAL_DRIFT[g] * np.ones((n, 1)),
                ]
            )
            for g in range(2)
        ]
    )                                                  # (2, N, N + 1)

    d = ParticleState(
        n_active=2 * np.ones(p_count, dtype=int),
        context=np.array([0, 1], dtype=int),
        n_cues=1,
        i_resampled=np.arange(p_count),
        n_context=np.zeros((p_count, c_slots, c_slots)),
        n_cue=np.zeros((p_count, c_slots, q_cols)),
        global_cue_probabilities=0.5 * np.ones((p_count, q_cols)),
        state_filtered_mean=np.zeros((p_count, c_slots, n)),
        state_filtered_cov=np.tile(0.02 * np.eye(n), (p_count, c_slots, 1, 1)),
        Theta=np.zeros((p_count, c_slots, n, n + 1)),
        bias=np.zeros((p_count, c_slots, n)),
        local_cue_matrix=np.zeros((p_count, c_slots, q_cols)),
        local_transition_matrix=np.zeros((p_count, c_slots, c_slots)),
        global_transition_probabilities=np.zeros((p_count, c_slots)),
        predicted_probabilities=np.zeros((p_count, c_slots)),
        responsibilities=np.zeros((p_count, c_slots)),
        prior_probabilities=np.zeros((p_count, c_slots)),
        probability_cue=np.ones((p_count, c_slots)),
        probability_state_feedback=np.ones((p_count, c_slots)),
    )
    d.previous_context = d.context.copy()

    for p in range(p_count):
        local_to_global = np.array([1, 0]) if p == 1 else np.array([0, 1])
        for local in range(2):
            g = local_to_global[local]
            d.state_filtered_mean[p, local] = state_mean[g]
            d.Theta[p, local] = theta[g]
            d.local_cue_matrix[p, local, :] = ALIGNMENT_GLOBAL_CUE[g, :]
            d.global_transition_probabilities[p, local] = ALIGNMENT_GLOBAL_BETA[g]
            d.predicted_probabilities[p, local] = ALIGNMENT_GLOBAL_PREDICTED[g]
            d.responsibilities[p, local] = ALIGNMENT_GLOBAL_RESPONSIBILITIES[g]
            d.prior_probabilities[p, local] = ALIGNMENT_GLOBAL_PREDICTED[g]

        for local in range(2):
            global_from = local_to_global[local]
            for dest_local in range(2):
                global_to = local_to_global[dest_local]
                d.local_transition_matrix[p, local, dest_local] = (
                    ALIGNMENT_GLOBAL_TRANSITION[global_from, global_to]
                )
            d.local_transition_matrix[p, local, 2] = ALIGNMENT_GLOBAL_TRANSITION[
                global_from, 2
            ]

        d.local_cue_matrix[p, 2, :] = [0.5, 0.5]
        d.local_transition_matrix[p, 2, 2] = 1.0
        d.global_transition_probabilities[p, 2] = ALIGNMENT_GLOBAL_BETA[2]
        d.predicted_probabilities[p, 2] = ALIGNMENT_GLOBAL_PREDICTED[2]
        d.responsibilities[p, 2] = ALIGNMENT_GLOBAL_RESPONSIBILITIES[2]
        d.prior_probabilities[p, 2] = ALIGNMENT_GLOBAL_PREDICTED[2]

    d.state_mean = d.state_filtered_mean.copy()
    d.state_cov = d.state_filtered_cov.copy()
    d.state_feedback_mean = d.state_filtered_mean.copy()
    d.state_feedback_cov = d.state_filtered_cov.copy()

    return RealTimeCOIN.from_state(
        {
            "num_particles": p_count,
            "max_contexts": max_contexts,
            "state_dim": n,
        },
        d,
        trial=1,
        cue_values=(1,),
        rng=rng,
    )

"""Lazy global context alignment and the context-facing APIs built on it.

Translated from ``tests/test_global_alignment.m``, plus the extra invariants the
Python port needs: that the alignment never mutates the particle state, that the
map and vector read-outs agree, and that the stationary distribution really is a
fixed point of the aligned transition matrix.

Almost everything is driven through ``RealTimeCOIN.from_state`` fixtures rather
than ``observe_y``, so these tests are largely independent of the inference
pipelines. The one exception is the lazy-cache section at the end, which the
MATLAB original drives with real observations and which is reproduced that way
here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN
from realtimecoin import alignment as rc_alignment
from realtimecoin.state import copy_state

from fixtures import (
    ALIGNMENT_GLOBAL_CUE,
    ALIGNMENT_GLOBAL_PREDICTED,
    ALIGNMENT_GLOBAL_RESPONSIBILITIES,
    ALIGNMENT_GLOBAL_STATE_MEAN,
    ALIGNMENT_GLOBAL_TRANSITION,
    MD_GLOBAL_DRIFT,
    MD_GLOBAL_RETENTION,
    alignment_fixture,
    large_assignment_fixture,
    md_alignment_fixture,
)
from helpers import assert_close, assert_density_peaks, must_error

TOL = 1e-12
LOOSE = 1e-6


# ----------------------------------------------------------------------
# modal cardinality and the modal particle subset
# ----------------------------------------------------------------------


def test_modal_cardinality_breaks_ties_towards_the_smaller_count():
    """The mode is the most common count, and ties go to the SMALLER one."""
    assert rc_alignment.modal_cardinality([2, 2, 3]) == 2
    assert rc_alignment.modal_cardinality([3, 3, 2]) == 3
    # 1 and 4 are equally common: the smaller wins.
    assert rc_alignment.modal_cardinality([1, 4]) == 1
    assert rc_alignment.modal_cardinality([4, 1, 4, 1, 2]) == 1
    assert rc_alignment.modal_cardinality([]) == 0


def test_select_modal_contexts_picks_the_modal_particles():
    """Only particles at the modal cardinality take part, with equal weight."""
    model = alignment_fixture(include_non_modal=True)
    km, mask, idx, weights = rc_alignment.select_modal_contexts(model)

    assert km == 2
    assert list(mask) == [True, True, True, False]
    assert list(idx) == [0, 1, 2]
    assert_close(weights, np.full(3, 1 / 3), TOL, "uniform modal weights")


def test_select_modal_contexts_falls_back_to_every_particle():
    """With no particle at the mode, all particles are used and Km is clipped.

    Unreachable through the real pipeline (the mode always has a member), so it
    is provoked by monkeypatching ``modal_cardinality`` to return a count no
    particle carries - which is exactly the state the MATLAB fallback guards.
    """
    model = alignment_fixture()
    original = rc_alignment.modal_cardinality
    try:
        rc_alignment.modal_cardinality = lambda cards: 99
        km, mask, idx, weights = rc_alignment.select_modal_contexts(model)
    finally:
        rc_alignment.modal_cardinality = original

    assert km == min(2, model.max_contexts)   # max(cards) clipped to max_contexts
    assert mask.all()
    assert list(idx) == [0, 1, 2]
    assert_close(weights.sum(), 1.0, TOL, "fallback weights normalised")


def test_select_modal_contexts_fallback_clips_to_max_contexts():
    """The fallback cardinality is clipped, not just taken from max(cards).

    Separate from the test above because there ``max(cards) == 2`` is already
    below ``max_contexts``, so the clip itself never fires.
    """
    model = alignment_fixture()
    model.D.n_active[:] = 9                   # above max_contexts == 3
    original = rc_alignment.modal_cardinality
    try:
        rc_alignment.modal_cardinality = lambda cards: 99
        km, mask, idx, _ = rc_alignment.select_modal_contexts(model)
    finally:
        rc_alignment.modal_cardinality = original

    assert km == model.max_contexts, "fallback cardinality must be clipped"
    assert mask.all()
    assert list(idx) == [0, 1, 2]


# ----------------------------------------------------------------------
# the alignment itself
# ----------------------------------------------------------------------


def test_alignment_recovers_the_known_two_context_permutation():
    """Anchor keeps canonical labels; the swapped particle maps back."""
    model = alignment_fixture()
    alignment = model.context_alignment()

    assert alignment["K"] == 2, "alignment should use the modal cardinality"
    assert alignment["modal_particle_mask"].all(), "all particles are modal"
    assert list(alignment["assignment"][0, :2]) == [0, 1], "anchor is canonical"
    assert list(alignment["assignment"][1, :2]) == [1, 0], "particle 1 maps back"
    assert list(alignment["assignment"][2, :2]) == [0, 1]
    # Km < max_contexts, so the novel slot carries its own global label.
    assert list(alignment["assignment"][:3, 2]) == [2, 2, 2]
    assert alignment["converged"]


def test_alignment_prototype_means_match_the_ground_truth():
    """The global prototypes recover the per-context state moments."""
    alignment = alignment_fixture().context_alignment()
    proto = alignment["global_contexts"]

    assert_close(
        proto["state_mean"], ALIGNMENT_GLOBAL_STATE_MEAN, LOOSE, "prototype means"
    )
    # Every particle contributes the same mean and variance, so the mixture
    # variance is just the shared component variance.
    assert_close(proto["state_var"], np.full(2, 0.02), LOOSE, "prototype vars")
    assert_close(proto["cue_prob"], ALIGNMENT_GLOBAL_CUE, LOOSE, "prototype cues")
    assert_close(
        proto["transition_prob"],
        ALIGNMENT_GLOBAL_TRANSITION,
        LOOSE,
        "prototype transitions",
    )


def test_global_responsibilities_and_predicted_probabilities():
    """Aligned summaries reproduce the fixture's global weights exactly."""
    model = alignment_fixture()

    resp = model.responsibilities_map()
    assert set(resp) == {0, 1, 2}
    assert_close(resp[0], 0.6, TOL, "global responsibility, context 0")
    assert_close(resp[1], 0.3, TOL, "global responsibility, context 1")
    assert_close(resp[2], 0.1, TOL, "novel responsibility bucket")

    pred = model.predicted_context_probabilities_vector()
    assert pred.shape == (model.max_contexts + 1,)
    assert_close(pred[:3], ALIGNMENT_GLOBAL_PREDICTED, TOL, "global predicted")
    assert_close(pred[3], 0.0, TOL, "unused slot stays empty")


def test_sampled_context_count_uses_global_labels():
    """All three particles sample the SAME physical context, i.e. global 0."""
    counts = alignment_fixture().sampled_context_count()
    assert_close(counts, [1.0, 0.0, 0.0, 0.0], TOL, "sampled context occupancy")


def test_modal_subset_filtering_ignores_the_non_modal_particle():
    """The off-prototype single-context particle must not dilute summaries."""
    model = alignment_fixture(include_non_modal=True)
    alignment = model.context_alignment()

    assert alignment["K"] == 2, "modal subset keeps the two-context cardinality"
    assert alignment["modal_particle_mask"].sum() == 3
    assert_close(
        model.responsibilities_map()[0], 0.6, TOL, "non-modal particle excluded"
    )

    diag = model.diagnostics()
    # Particles lead in this port, so the modal-particle axis is axis 0.
    assert diag["responsibilities"].shape[0] == 3, "diagnostics: modal only"
    assert diag["raw"].responsibilities.shape[0] == 4, "raw: all particles"


def test_ten_context_reverse_permutation_is_recovered():
    """The maximum-cardinality fixture recovers the known reverse permutation."""
    model = large_assignment_fixture()
    alignment = model.context_alignment()

    assert alignment["K"] == 10, "large fixture should align ten contexts"
    assert list(alignment["assignment"][0, :10]) == list(range(10))
    assert list(alignment["assignment"][1, :10]) == list(range(9, -1, -1))
    # Km == max_contexts, so there is no novel slot to label.
    assert alignment["assignment"][0, 10] == rc_alignment.UNASSIGNED


# ----------------------------------------------------------------------
# assignment solver
# ----------------------------------------------------------------------


def test_linear_assignment_is_cost_optimal():
    """The wrapper returns a permutation of minimum total cost."""
    rng = np.random.default_rng(7)
    for n in (1, 2, 3, 5):
        cost = rng.normal(size=(n, n))
        perm = rc_alignment.linear_assignment(cost)
        assert sorted(perm) == list(range(n)), "output must be a permutation"
        best = min(
            sum(cost[i, p[i]] for i in range(n))
            for p in itertools.permutations(range(n))
        )
        assert_close(
            sum(cost[i, perm[i]] for i in range(n)), best, 1e-12, "optimal cost"
        )


def test_linear_assignment_handles_degenerate_input():
    """Empty, non-finite and non-square inputs behave as documented."""
    assert rc_alignment.linear_assignment(np.zeros((0, 0))).size == 0
    # inf / nan become a finite sentinel, so the solver still returns a
    # permutation rather than throwing. With the sentinel substituted the
    # optimum is UNIQUE (1e100 for [1, 0] against 1e100 + 1 for [0, 1]), so
    # assert the permutation itself - a solver that ignored the sentinel would
    # pick the other one.
    cost = np.array([[np.inf, 0.0], [np.nan, 1.0]])
    assert list(rc_alignment.linear_assignment(cost)) == [1, 0]
    must_error(
        "non-square cost",
        lambda: rc_alignment.linear_assignment(np.zeros((2, 3))),
        "RealTimeCOIN:AssignmentMatrixNotSquare",
    )


def test_converged_assignment_minimises_the_cost_matrix():
    """The fixed point is optimal for the cost matrix it converged against.

    Asserted as COST optimality rather than an exact permutation, because
    tie-breaking on a degenerate cost matrix is solver-specific.
    """
    for model in (alignment_fixture(), large_assignment_fixture()):
        alignment = model.context_alignment()
        km = alignment["K"]
        proto = alignment["global_contexts"]
        prepared = rc_alignment.prepare_assignment_prototypes(model, km, proto)
        for p in alignment["modal_particle_indices"]:
            cost = rc_alignment.assignment_cost_matrix(
                model, int(p), km, proto, alignment["assignment"], True, prepared
            )
            perm = alignment["assignment"][p, :km]
            achieved = sum(cost[i, perm[i]] for i in range(km))
            optimal = sum(
                cost[i, j]
                for i, j in zip(*[range(km), rc_alignment.linear_assignment(cost)])
            )
            assert achieved <= optimal + 1e-9, "converged labels are optimal"


# ----------------------------------------------------------------------
# prototype-update and summary edge cases
# ----------------------------------------------------------------------


def test_a_global_context_no_particle_claims_falls_back_to_uniform():
    """An unclaimed prototype must not become a divergence-free attractor.

    ``normalize_probability`` maps the all-zero accumulator to UNIFORM rather
    than leaving it zero. That fallback is load bearing: a zero categorical row
    would give a degenerate (tiny) Jeffreys divergence against every local
    context and pull the whole matching onto the empty prototype.
    """
    model = alignment_fixture()
    c_slots = model.max_contexts + 1
    # Only local slot 0 is labelled, and only global 0 is used, so global 1 is
    # claimed by nobody. Particles 0 and 2 (not 1, whose labels are swapped)
    # keep global 0 an unmixed copy of the real context 0.
    assignment = np.full((model.num_particles, c_slots), rc_alignment.UNASSIGNED)
    assignment[:, 0] = 0

    proto = rc_alignment.update_global_contexts(
        model, 2, np.array([0, 2]), np.full(2, 0.5), assignment
    )

    assert_close(proto["state_mean"][1], 0.0, TOL, "unclaimed mean stays zero")
    assert_close(proto["state_var"][1], 0.0, TOL, "unclaimed variance stays zero")
    assert_close(proto["cue_prob"][1], [0.5, 0.5], 1e-9, "uniform cue fallback")
    assert_close(
        proto["transition_prob"][1], np.full(3, 1 / 3), 1e-9, "uniform transitions"
    )
    # regularize_covariance turns the zero matrix into something invertible.
    covar = proto["dynamics_covar"][1]
    assert np.all(np.isfinite(covar))
    assert np.all(np.linalg.eigvalsh(covar) > 0), "unclaimed covariance is PD"
    # Global 0 is still built from the real particles.
    assert_close(proto["state_mean"][0], -1.0, LOOSE, "claimed prototype intact")


def test_zero_cardinality_degrades_gracefully():
    """Km == 0 produces empty prototypes and an empty matching, not an error."""
    model = alignment_fixture()
    c_slots = model.max_contexts + 1
    assignment = np.full((model.num_particles, c_slots), rc_alignment.UNASSIGNED)

    proto = rc_alignment.update_global_contexts(
        model, 0, np.arange(3), np.full(3, 1 / 3), assignment
    )
    assert proto["state_mean"].shape == (0,)
    assert proto["transition_prob"].shape == (0, 1)
    assert proto["dynamics_covar"].shape == (0, 2, 2)

    cost = rc_alignment.assignment_cost_matrix(
        model, 0, 0, proto, assignment, True, {}
    )
    assert cost.shape == (0, 0)
    assert rc_alignment.linear_assignment(cost).size == 0

    # One destination slot only (the novel context), and it must still normalise.
    row = rc_alignment.global_transition_row(model, 0, 0, 0, assignment)
    assert row.shape == (1,)
    assert_close(row.sum(), 1.0, TOL, "degenerate transition row normalised")


def test_active_summary_contexts_falls_back_when_no_mass_exists():
    """With no predicted mass anywhere, slot 0 is still reported."""
    model = alignment_fixture()
    model.D.predicted_probabilities[:] = 0.0
    rc_alignment.invalidate_context_alignment(model)

    weights = rc_alignment.context_probability_vector(model, "predicted")
    assert_close(weights, np.zeros(4), TOL, "no mass anywhere")
    assert list(rc_alignment.active_summary_contexts(model)) == [0]

    alignment = model.context_alignment()
    assert list(rc_alignment.clamp_active_summary_contexts(model, alignment)) == [0]


def test_clamp_active_summary_contexts_drops_labels_beyond_K():
    """Slots without a global label in this alignment are filtered out."""
    model = alignment_fixture()
    alignment = model.context_alignment()

    # The novel slot (2) carries predicted mass but K == 2, so it must go.
    assert list(rc_alignment.active_summary_contexts(model)) == [0, 1, 2]
    assert list(rc_alignment.clamp_active_summary_contexts(model, alignment)) == [0, 1]

    empty = dict(alignment)
    empty["K"] = 0
    assert rc_alignment.clamp_active_summary_contexts(model, empty).size == 0


# ----------------------------------------------------------------------
# scatter_to_global, checked against a literal transcription of the MATLAB
# ----------------------------------------------------------------------


def _literal_scatter(model, x, alignment, mode):
    """Line-for-line transcription of ``scatterToGlobal.m`` (loops only).

    The shipped implementation vectorises the inner loops with ``np.add.at``;
    this deliberately naive version exists so the two can be differentially
    tested against each other on randomised, adversarial assignments.
    """
    c_slots = model.max_contexts + 1
    km = alignment["K"]
    modal_idx = alignment["modal_particle_indices"]
    asg = alignment["assignment"]
    n = len(modal_idx)
    if mode == "cue":
        g = np.zeros((n, c_slots, x.shape[2]))
    elif mode == "transition":
        g = np.zeros((n, c_slots, c_slots))
    elif mode == "labels":
        g = np.full(n, np.nan)
    else:
        g = np.zeros((n, c_slots))

    for idx, p in enumerate(modal_idx):
        if mode == "overwrite":
            for lo in range(c_slots):
                t = asg[p, lo]
                if 0 <= t < c_slots:
                    g[idx, t] = x[p, lo]
        elif mode == "add":
            for lo in range(c_slots):
                t = asg[p, lo]
                if 0 <= t < c_slots:
                    g[idx, t] += x[p, lo]
            if km >= model.max_contexts:
                for t in range(km, c_slots):
                    g[idx, t] = 0.0
        elif mode == "cue":
            for lo in range(c_slots):
                t = asg[p, lo]
                if 0 <= t < c_slots:
                    g[idx, t, :] += x[p, lo, :]
        elif mode == "transition":
            for local_from in range(c_slots):
                global_from = asg[p, local_from]
                if global_from < 0 or global_from >= c_slots:
                    continue
                for local_to in range(c_slots):
                    global_to = asg[p, local_to]
                    if 0 <= global_to < c_slots:
                        g[idx, global_from, global_to] += x[p, local_from, local_to]
        else:
            lo = model.D.context[p]
            if 0 <= lo < asg.shape[1]:
                t = asg[p, lo]
                if t >= 0:
                    g[idx] = t
    return g


def test_scatter_to_global_matches_the_literal_matlab_loops():
    """Randomised differential test of all five scatter modes.

    The assignments are drawn adversarially - duplicate targets (so the
    overwrite/accumulate distinction bites), unassigned slots, and a modal
    cardinality both below and at ``max_contexts`` - because those are exactly
    the cases where a vectorised re-derivation goes wrong silently.
    """
    rng = np.random.default_rng(0)
    model = alignment_fixture(include_non_modal=True)
    base = model.context_alignment()
    c_slots = model.max_contexts + 1
    p_count = model.num_particles

    for _ in range(100):
        alignment = dict(base)
        alignment["assignment"] = rng.integers(-1, c_slots, size=(p_count, c_slots))
        n_modal = int(rng.integers(1, p_count + 1))
        alignment["modal_particle_indices"] = np.sort(
            rng.choice(p_count, size=n_modal, replace=False)
        )
        alignment["K"] = int(rng.integers(0, model.max_contexts + 1))

        for mode, x in (
            ("overwrite", rng.normal(size=(p_count, c_slots))),
            ("add", rng.normal(size=(p_count, c_slots))),
            ("cue", rng.normal(size=(p_count, c_slots, 2))),
            ("transition", rng.normal(size=(p_count, c_slots, c_slots))),
            ("labels", None),
        ):
            got = rc_alignment.scatter_to_global(model, x, alignment, mode)
            want = _literal_scatter(model, x, alignment, mode)
            assert np.allclose(got, want, equal_nan=True), mode


def test_scatter_to_global_rejects_an_unknown_mode():
    """A typo'd mode must fail loudly rather than silently return zeros."""
    model = alignment_fixture()
    alignment = model.context_alignment()
    must_error(
        "bad scatter mode",
        lambda: rc_alignment.scatter_to_global(
            model, model.D.responsibilities, alignment, "nope"
        ),
        "RealTimeCOIN:BadScatterMode",
    )


# ----------------------------------------------------------------------
# caching / warm start
# ----------------------------------------------------------------------


def test_repeated_calls_reuse_the_cached_alignment():
    """Within one state version the very same object comes back."""
    model = alignment_fixture()
    a1 = model.context_alignment()
    a2 = model.context_alignment()

    assert a1 is a2, "a cache hit must not recompute"
    assert a1["cache_state_version"] == a2["cache_state_version"]
    assert a1["cache_state_version"] == model.state_version


def test_the_returned_alignment_cannot_corrupt_the_cache():
    """The published arrays are read-only, standing in for MATLAB's value copy.

    The returned dict IS the cache AND the warm-start seed, so a stray write
    would corrupt not just this trial's read-outs but the NEXT trial's warm
    start. MATLAB gets that safety free from struct value semantics.
    """
    model = alignment_fixture()
    alignment = model.context_alignment()

    with pytest.raises(ValueError):
        alignment["assignment"][0, 0] = 99
    with pytest.raises(ValueError):
        alignment["global_contexts"]["state_mean"][0] = 99.0
    with pytest.raises(ValueError):
        alignment["modal_particle_indices"][0] = 99

    assert list(model.context_alignment()["assignment"][0, :2]) == [0, 1]
    assert model.alignment_seed is alignment, "cache and seed are one object"


def test_diagnostics_exposes_the_same_protected_alignment():
    """diagnostics()["alignment"] is the cache too, so it is protected as well."""
    model = alignment_fixture()
    diag = model.diagnostics()
    with pytest.raises(ValueError):
        diag["alignment"]["assignment"][0, 0] = 99


def test_a_state_version_bump_invalidates_the_cache():
    """Simulates what observe_y does, without running the pipeline."""
    model = alignment_fixture()
    a1 = model.context_alignment()

    rc_alignment.invalidate_context_alignment(model)   # what observe_y calls
    assert model.alignment_cache is None
    assert model.alignment_seed is not None, "the warm-start seed survives"

    a3 = model.context_alignment()
    assert a3 is not a1
    assert a3["cache_state_version"] != a1["cache_state_version"]
    assert a3["cache_state_version"] == model.state_version
    # The particle state is unchanged, so the labels must be too.
    assert np.array_equal(a3["assignment"], a1["assignment"])


def test_warm_start_reuses_the_seed_and_preserves_label_order():
    """A fresh model seeded with a previous alignment keeps its labels."""
    warm = alignment_fixture()
    w1 = warm.context_alignment()

    seeded = RealTimeCOIN.from_state(
        {"num_particles": 3, "max_contexts": 3},
        copy_state(warm.D),
        trial=1,
        cue_values=(1,),
        alignment_seed=w1,
        rng=0,
    )
    w2 = seeded.context_alignment()

    assert w2["used_seed"], "post-update alignment should warm-start"
    assert np.array_equal(w2["assignment"][:3, :2], w1["assignment"][:3, :2])


def test_warm_start_preserves_a_PERMUTED_label_order():
    """The seed's labelling wins over the one a cold start would pick.

    The test above is the MATLAB one, and on its own it is weak: seeding with
    the alignment a cold start would have produced anyway means it passes even
    if the warm-start path is deleted. Seeding a RELABELLED alignment is what
    actually proves the seed is honoured - the cold start would anchor particle
    0 on the identity map, so recovering ``[1, 0]`` can only come from the seed.
    """
    warm = alignment_fixture()
    w1 = warm.context_alignment()

    seed = dict(w1)
    seed["assignment"] = np.array(w1["assignment"], copy=True)
    seed["assignment"][:3, :2] = np.array([[1, 0], [0, 1], [1, 0]])
    # Swap the two global prototypes to match the swapped labels.
    seed["global_contexts"] = {
        name: np.array(value, copy=True)[::-1]
        for name, value in w1["global_contexts"].items()
    }

    seeded = RealTimeCOIN.from_state(
        {"num_particles": 3, "max_contexts": 3},
        copy_state(warm.D),
        trial=1,
        cue_values=(1,),
        alignment_seed=seed,
        rng=0,
    )
    w2 = seeded.context_alignment()

    assert w2["used_seed"]
    assert list(w2["assignment"][0, :2]) == [1, 0], "seeded order must survive"
    assert list(w2["assignment"][1, :2]) == [0, 1]
    assert_close(
        w2["global_contexts"]["state_mean"],
        ALIGNMENT_GLOBAL_STATE_MEAN[::-1],
        LOOSE,
        "prototypes follow the seeded labelling",
    )


def test_an_incompatible_seed_is_rejected():
    """A seed built for a different cardinality must not be reused."""
    warm = alignment_fixture()
    seed = dict(warm.context_alignment())
    seed["K"] = 7                                    # no longer matches Km == 2

    seeded = RealTimeCOIN.from_state(
        {"num_particles": 3, "max_contexts": 3},
        copy_state(warm.D),
        trial=1,
        cue_values=(1,),
        alignment_seed=seed,
        rng=0,
    )
    assert not seeded.context_alignment()["used_seed"]


# ----------------------------------------------------------------------
# reporting-only guarantee
# ----------------------------------------------------------------------


def _state_fingerprint(state):
    """Return a comparable snapshot of every field of a particle state."""
    out = {}
    for name, value in state.as_dict().items():
        out[name] = None if value is None else np.array(value, copy=True)
    return out


def test_alignment_never_mutates_the_particle_state():
    """The alignment is reporting only: D must survive every query untouched."""
    model = alignment_fixture(include_non_modal=True)
    before = _state_fingerprint(model.D)

    grid = np.linspace(-2.0, 2.0, 21)
    model.context_alignment()
    model.diagnostics()
    model.predicted_context_probabilities_vector()
    model.predicted_context_probabilities_map()
    model.responsibilities_vector()
    model.responsibilities_map()
    model.sampled_context_count()
    model.stationary_context_probabilities()
    model.global_transition_probabilities()
    model.global_cue_probabilities()
    model.local_transition_probabilities()
    model.local_cue_probabilities()
    model.state_given_context_probability(grid)
    model.state_feedback_given_context_probability(grid)

    after = _state_fingerprint(model.D)
    assert set(before) == set(after)
    for name in before:
        if before[name] is None:
            assert after[name] is None, name
            continue
        assert np.array_equal(before[name], after[name]), name


def test_local_transition_probabilities_returns_a_copy():
    """Mutating a read-out must not corrupt the cached prototypes."""
    model = alignment_fixture()
    first = model.local_transition_probabilities()
    first[:] = -1.0
    assert_close(
        model.local_transition_probabilities(),
        ALIGNMENT_GLOBAL_TRANSITION,
        LOOSE,
        "cached prototypes survive a mutated read-out",
    )


# ----------------------------------------------------------------------
# aligned read-outs
# ----------------------------------------------------------------------


def test_map_and_vector_read_outs_agree():
    """Every map entry is the corresponding vector entry, positives only."""
    model = alignment_fixture()
    for vector_fn, map_fn in (
        (model.predicted_context_probabilities_vector,
         model.predicted_context_probabilities_map),
        (model.responsibilities_vector, model.responsibilities_map),
    ):
        vector = vector_fn()
        mapping = map_fn()
        assert set(mapping) == set(np.flatnonzero(vector > 0).tolist())
        for c, value in mapping.items():
            assert_close(value, vector[c], TOL, "map/vector agree at %d" % c)


def test_local_transition_and_cue_probabilities():
    """Aligned per-context rows match the fixture's global parameters."""
    model = alignment_fixture()
    k = model.context_alignment()["K"]

    transition = model.local_transition_probabilities()
    assert transition.shape == (k, k + 1)
    assert_close(transition, ALIGNMENT_GLOBAL_TRANSITION, LOOSE, "aligned rows")
    assert_close(transition.sum(axis=1), np.ones(k), LOOSE, "rows normalised")

    cue = model.local_cue_probabilities()
    assert cue.shape == (k, 2)
    assert_close(cue, ALIGNMENT_GLOBAL_CUE, LOOSE, "aligned cue rows")


def test_global_franchise_read_outs():
    """The franchise weights are relabelled (transitions) or raw (cues)."""
    model = alignment_fixture()
    beta = model.global_transition_probabilities()
    assert_close(beta, [0.55, 0.35, 0.1, 0.0], LOOSE, "franchise transitions")
    assert_close(
        model.global_cue_probabilities(), [0.5, 0.5], LOOSE, "franchise cues"
    )


def test_global_cue_read_outs_require_a_cue():
    """Both cue read-outs mirror MATLAB's RealTimeCOIN:NoCues guard."""
    model = alignment_fixture()
    model.cue_values = []
    must_error(
        "global cues", model.global_cue_probabilities, "RealTimeCOIN:NoCues"
    )
    must_error(
        "local cues", model.local_cue_probabilities, "RealTimeCOIN:NoCues"
    )


def test_stationary_probabilities_are_a_fixed_point():
    """pi T = pi for the row-renormalised, novel-context-free transition matrix."""
    model = alignment_fixture()
    pi = model.stationary_context_probabilities()
    k = model.context_alignment()["K"]

    assert pi.shape == (k,)
    assert_close(pi.sum(), 1.0, 1e-10, "stationary distribution normalised")
    assert (pi >= 0).all()

    t = model.local_transition_probabilities()[:, :k]
    t = t / t.sum(axis=1, keepdims=True)
    assert_close(pi @ t, pi, 1e-10, "stationary is a fixed point")
    # [[0.8, 0.1], [0.2, 0.7]] renormalised has the closed-form solution below.
    assert_close(pi, [2 / 3, 1 / 3], 1e-9, "closed-form stationary")


def test_diagnostics_relabels_every_per_context_field():
    """Diagnostics puts each physical context in the same slot for all particles."""
    model = alignment_fixture()
    diag = model.diagnostics()

    assert diag["trial"] == 1
    assert diag["C"] == 2
    assert diag["raw"] is model.D
    assert_close(diag["context"], [0.0, 0.0, 0.0], TOL, "sampled global context")

    # Every particle now reports the same aligned per-context state mean.
    expected = np.tile([-1.0, 1.0, 0.0, 0.0], (3, 1))
    assert_close(diag["state_mean"], expected, LOOSE, "aligned state means")
    assert_close(
        diag["responsibilities"],
        np.tile(list(ALIGNMENT_GLOBAL_RESPONSIBILITIES) + [0.0], (3, 1)),
        TOL,
        "aligned responsibilities",
    )
    # Transition tensor: (P_modal, C, C) with both axes realigned.
    assert diag["local_transition_matrix"].shape == (3, 4, 4)
    for p in range(3):
        assert_close(
            diag["local_transition_matrix"][p, :2, :3],
            ALIGNMENT_GLOBAL_TRANSITION,
            LOOSE,
            "aligned transition tensor, particle %d" % p,
        )
    # Cue tensor: (P_modal, C, Q), context axis realigned only.
    assert diag["local_cue_matrix"].shape == (3, 4, 2)
    for p in range(3):
        assert_close(
            diag["local_cue_matrix"][p, :2, :],
            ALIGNMENT_GLOBAL_CUE,
            LOOSE,
            "aligned cue tensor, particle %d" % p,
        )


# ----------------------------------------------------------------------
# per-global-context densities
# ----------------------------------------------------------------------


def test_state_given_context_densities_peak_at_the_prototype_means():
    """Keys are aligned global labels; each density peaks at its own mean."""
    model = alignment_fixture()
    grid = np.linspace(-2.0, 2.0, 401)
    densities = model.state_given_context_probability(grid)

    assert set(densities) == {0, 1}, "only the active, aligned contexts"
    for value in densities.values():
        assert value.shape == grid.shape
        assert np.all(np.isfinite(value))
    assert_density_peaks(
        "state|context", densities, grid, dict(enumerate(ALIGNMENT_GLOBAL_STATE_MEAN))
    )


def test_state_feedback_given_context_densities_are_wider():
    """Feedback densities share the means but carry the extra sensory noise."""
    model = alignment_fixture()
    grid = np.linspace(-2.0, 2.0, 401)
    state = model.state_given_context_probability(grid)
    feedback = model.state_feedback_given_context_probability(grid)

    assert set(feedback) == set(state)
    assert_density_peaks(
        "feedback|context",
        feedback,
        grid,
        dict(enumerate(ALIGNMENT_GLOBAL_STATE_MEAN)),
    )
    for c in state:
        # Same mass, larger variance => strictly lower peak.
        assert feedback[c].max() < state[c].max()


def test_density_grid_validation():
    """A grid whose trailing dimension is wrong is rejected, as in MATLAB."""
    model = alignment_fixture()
    must_error(
        "2-D grid on a scalar model",
        lambda: model.state_given_context_probability(np.zeros((3, 2))),
        "RealTimeCOIN:GridDimensionMismatch",
    )
    must_error(
        "non-finite grid",
        lambda: model.state_given_context_probability([0.0, np.inf]),
        "RealTimeCOIN:GridNotFinite",
    )


# ----------------------------------------------------------------------
# end-to-end smoke test over every aligned query
# ----------------------------------------------------------------------


def test_every_aligned_query_returns_finite_normalised_output():
    """from_state smoke test: shapes, normalisation and finiteness."""
    model = alignment_fixture(include_non_modal=True)
    c_slots = model.max_contexts + 1
    k = model.context_alignment()["K"]

    for name in (
        "predicted_context_probabilities_vector",
        "responsibilities_vector",
        "sampled_context_count",
        "global_transition_probabilities",
    ):
        value = getattr(model, name)()
        assert value.shape == (c_slots,), name
        assert np.all(np.isfinite(value)), name
        assert np.all(value >= 0), name
        assert_close(value.sum(), 1.0, 1e-12, "%s normalised" % name)

    for name in ("predicted_context_probabilities_map", "responsibilities_map"):
        mapping = getattr(model, name)()
        assert mapping, name
        assert all(isinstance(key, int) for key in mapping), name
        assert_close(sum(mapping.values()), 1.0, 1e-12, "%s normalised" % name)

    cues = model.global_cue_probabilities()
    assert cues.shape == (2,)
    assert_close(cues.sum(), 1.0, 1e-12, "franchise cues normalised")

    transition = model.local_transition_probabilities()
    assert transition.shape == (k, k + 1)
    assert_close(transition.sum(axis=1), np.ones(k), 1e-12, "rows normalised")

    cue_rows = model.local_cue_probabilities()
    assert cue_rows.shape == (k, 2)
    assert_close(cue_rows.sum(axis=1), np.ones(k), 1e-12, "cue rows normalised")

    pi = model.stationary_context_probabilities()
    assert pi.shape == (k,)
    assert_close(pi.sum(), 1.0, 1e-10, "stationary normalised")

    diag = model.diagnostics()
    assert diag["C"] == k
    for key in (
        "state_mean",
        "state_var",
        "state_feedback_mean",
        "state_feedback_var",
        "retention",
        "drift",
        "bias",
        "global_transition_probabilities",
    ):
        assert diag[key].shape == (3, c_slots), key
        assert np.all(np.isfinite(diag[key])), key

    grid = np.linspace(-2.0, 2.0, 51)
    for query in (
        model.state_given_context_probability,
        model.state_feedback_given_context_probability,
    ):
        densities = query(grid)
        assert densities
        for value in densities.values():
            assert value.shape == grid.shape
            assert np.all(np.isfinite(value))
            assert np.all(value >= 0)


# ----------------------------------------------------------------------
# multi-dimensional model
# ----------------------------------------------------------------------


def test_md_alignment_recovers_the_swapped_labels():
    """The MD cost matrix separates the two contexts just as the scalar one."""
    model = md_alignment_fixture()
    alignment = model.context_alignment()

    assert alignment["K"] == 2
    assert list(alignment["assignment"][0, :2]) == [0, 1]
    assert list(alignment["assignment"][1, :2]) == [1, 0]

    proto = alignment["global_contexts"]
    assert proto["state_mean"].shape == (2, 2)
    assert proto["state_cov"].shape == (2, 2, 2)
    assert proto["theta_mean"].shape == (2, 2, 3)
    assert_close(
        proto["state_mean"], [[-1.0, -1.0], [1.0, 1.0]], LOOSE, "MD prototypes"
    )
    for g in range(2):
        assert_close(
            proto["theta_mean"][g, :, :2],
            MD_GLOBAL_RETENTION[g] * np.eye(2),
            LOOSE,
            "MD retention matrix %d" % g,
        )
        assert_close(
            proto["theta_mean"][g, :, 2],
            np.full(2, MD_GLOBAL_DRIFT[g]),
            LOOSE,
            "MD drift %d" % g,
        )


def test_md_diagnostics_splits_theta_and_averages_weights():
    """diagnostics_md reports A / drift separately and K-length weights."""
    model = md_alignment_fixture()
    diag = model.diagnostics()      # dispatches to diagnostics_md

    assert diag["K"] == 2
    assert diag["A"].shape == (2, 2, 2)
    assert diag["drift"].shape == (2, 2)
    assert diag["bias"].shape == (2, 2)
    assert diag["state_cov"].shape == (2, 2, 2)
    assert diag["predicted_probabilities"].shape == (2,)
    assert diag["responsibilities_particles"].shape == (2, 4)
    assert_close(
        diag["predicted_probabilities"],
        ALIGNMENT_GLOBAL_PREDICTED[:2],
        TOL,
        "MD aligned predicted weights",
    )
    assert_close(
        diag["responsibilities"],
        ALIGNMENT_GLOBAL_RESPONSIBILITIES[:2],
        TOL,
        "MD aligned responsibilities",
    )
    assert diag["raw"] is model.D


def test_md_per_context_densities_use_a_row_grid():
    """MD grids are (K_pts, N) - one query point per ROW - as the port defines."""
    model = md_alignment_fixture()
    grid = np.array([[-1.0, -1.0], [1.0, 1.0], [5.0, 5.0]])
    densities = model.state_given_context_probability(grid)

    assert set(densities) == {0, 1}
    for c, value in densities.items():
        assert value.shape == (3,)
        assert np.all(np.isfinite(value))
        # Each density peaks at its own prototype mean, not at the far point.
        assert int(np.argmax(value)) == c
        assert value[2] < value[c]

    feedback = model.state_feedback_given_context_probability(grid)
    assert set(feedback) == {0, 1}
    for c in densities:
        assert feedback[c].max() < densities[c].max()

    must_error(
        "wrong trailing dimension",
        lambda: model.state_given_context_probability(np.zeros((3, 5))),
        "RealTimeCOIN:GridDimensionMismatch",
    )


def test_md_alignment_never_mutates_the_particle_state():
    """The reporting-only guarantee holds on the multi-dimensional path too."""
    model = md_alignment_fixture()
    before = _state_fingerprint(model.D)

    model.diagnostics()
    model.responsibilities_vector()
    model.stationary_context_probabilities()
    model.local_transition_probabilities()
    model.state_given_context_probability(np.zeros((4, 2)))

    after = _state_fingerprint(model.D)
    for name in before:
        if before[name] is None:
            assert after[name] is None, name
            continue
        assert np.array_equal(before[name], after[name]), name


# ----------------------------------------------------------------------
# integration: the lazy cache driven by the real inference pipeline
# ----------------------------------------------------------------------


def test_alignment_cache_across_observations():
    """The MATLAB lazy-cache section driven with real observations.

    Port of ``tests/test_global_alignment.m`` lines 34-41: repeated
    ``context_alignment`` calls within a trial reuse one cache entry, and each
    ``observe_y`` invalidates it.
    """
    model = RealTimeCOIN(num_particles=12, max_contexts=3, rng=0)
    model.observe_y(0.1)

    a1 = model.context_alignment()
    a2 = model.context_alignment()
    assert a1["cache_state_version"] == a2["cache_state_version"], (
        "repeated alignment calls should reuse the cached state version"
    )
    # A cache hit hands back the very same object (Python has no value structs).
    assert a2 is a1

    model.observe_y(0.2)
    a3 = model.context_alignment()
    assert a3["cache_state_version"] != a1["cache_state_version"], (
        "a new observation should invalidate the alignment cache"
    )
    assert a3["computed_at_trial"] == model.Trial

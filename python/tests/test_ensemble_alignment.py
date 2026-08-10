"""White-box checks on the cross-run ensemble context alignment.

Companion to the two blind Phase-2 suites
(``test_ensemble_phase2_blindA`` / ``_blindB``), which are written against
``docs/SPEC_ensemble.md`` Part 10 only. This module is allowed to look at
:mod:`realtimecoin.ensemble_alignment` directly and pins the properties the
blind suites cannot observe:

* the reference frame is ``Kref = max_r K_r`` with ties broken by the LOWEST
  member index, and the reference member matches itself by the identity;
* every member's contexts map onto DISTINCT reference labels;
* the alignment and the six Phase-2 queries are strictly READ-ONLY on the
  members - no particle array changes, no random numbers consumed - so an
  aligned readout can never perturb the filter it is reporting on;
* the two averaging rules are not interchangeable: the zero-fill rule keeps a
  probability row summing to 1, and the NaN-omit rule divides a density by the
  number of CONTRIBUTING runs.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOINEnsemble
from realtimecoin.ensemble_alignment import (
    ensemble_context_alignment,
    ensemble_context_density,
    ensemble_context_vector,
)

#: Phase-2 queries taking no argument.
VECTOR_QUERIES = (
    "responsibilities_vector",
    "predicted_context_probabilities_vector",
    "sampled_context_count",
    "stationary_context_probabilities",
)

#: Phase-2 queries taking a grid.
DENSITY_QUERIES = (
    "state_given_context_probability",
    "state_feedback_given_context_probability",
)


def driven_ensemble(runs=4, seed=77, **params):
    """Ensemble stepped through a two-block cued stream.

    Parameters
    ----------
    runs : int, optional
        Member count. Default 4.
    seed : int, optional
        Ensemble seed. Default 77.
    **params
        Member constructor keywords; sensible defaults are filled in.

    Returns
    -------
    RealTimeCOINEnsemble
    """
    params.setdefault("num_particles", 40)
    params.setdefault("max_contexts", 4)
    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    cues = [1, 1, 1, 2, 2, 2, 1, 1]
    obs = [0.30, 0.28, 0.32, -0.30, -0.28, -0.31, 0.29, 0.30]
    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
    return ens


def member_fingerprint(member):
    """Every particle array of a member, plus its generator state.

    Returns
    -------
    dict
        ``{name: numpy.ndarray}`` copies, plus ``"__rng__"`` and
        ``"__trial__"``.
    """
    snapshot = {
        name: np.array(value, copy=True)
        for name, value in member.D.as_dict().items()
        if isinstance(value, np.ndarray)
    }
    snapshot["__rng__"] = member.rng.bit_generator.state
    snapshot["__trial__"] = member.Trial
    return snapshot


# ----------------------------------------------------------------------
# Reference frame
# ----------------------------------------------------------------------


def test_reference_frame_is_the_member_with_the_most_contexts():
    """``Kref = max_r K_r``, reference index = the first maximiser."""
    ens = driven_ensemble()
    alignment = ensemble_context_alignment(ens)

    k = np.array([int(m.context_alignment()["K"]) for m in ens.members])   # (R,)
    np.testing.assert_array_equal(alignment["k"], k)
    assert alignment["k_ref"] == int(k.max())
    # Ties -> lowest member index: the first index attaining the maximum.
    assert alignment["ref_index"] == int(np.flatnonzero(k == k.max())[0])


def test_reference_member_matches_itself_by_the_identity():
    """The reference member's own cost matrix has a zero diagonal (SPEC 10.2)."""
    ens = driven_ensemble()
    alignment = ensemble_context_alignment(ens)
    ref = alignment["ref_index"]
    k_ref = alignment["k_ref"]
    assert k_ref >= 1, "the scenario must instantiate at least one context"
    np.testing.assert_array_equal(alignment["perm"][ref], np.arange(k_ref))


def test_every_member_maps_onto_distinct_reference_labels():
    """A matching, not a many-to-one collapse: labels are unique and in range."""
    ens = driven_ensemble(runs=6, seed=99)
    alignment = ensemble_context_alignment(ens)
    k_ref = alignment["k_ref"]

    for r, perm in enumerate(alignment["perm"]):
        assert perm.size == int(alignment["k"][r]), "member %d perm length" % r
        assert len(set(perm.tolist())) == perm.size, "member %d reuses a label" % r
        assert np.all((perm >= 0) & (perm < k_ref)), "member %d label out of range" % r


class _StubMember:
    """Member stand-in exposing only what the cross-run aligner reads.

    Real members cannot be steered into the configurations the aligner exists
    to handle (unequal context counts, a context-free run, a reference label
    only some runs hold), because every member sees the same stream and so
    tends to agree. These stubs pin the prototype means directly.

    Parameters
    ----------
    state_mean : array_like
        ``(K,)`` scalar or ``(K, N)`` multi-dimensional prototype means; ``K``
        is read from its leading axis.
    max_contexts : int, optional
        Member context cap. Default 3.
    """

    def __init__(self, state_mean, max_contexts=3):
        self.state_mean = np.asarray(state_mean, dtype=float)
        self.max_contexts = max_contexts

    def context_alignment(self):
        """The subset of the member alignment the cross-run aligner uses."""
        return {
            "K": int(self.state_mean.shape[0]),
            "global_contexts": {"state_mean": self.state_mean},
        }


class _StubEnsemble:
    """Minimal duck-typed ensemble over :class:`_StubMember` members."""

    def __init__(self, members):
        self.members = tuple(members)
        self.runs = len(self.members)


def test_alignment_with_no_instantiated_context_is_empty():
    """The ``Kref == 0`` branch, which a real ensemble cannot reach.

    A freshly constructed member already carries one context (every particle
    starts in context 0), so ``K == 0`` is only reachable through the defensive
    guard the MATLAB original also carries. It is exercised here with a stub so
    the branch is not dead code: an empty matching, all novel mass in reference
    slot 0, and an empty density map.
    """
    ens = _StubEnsemble([_StubMember(np.zeros(0)) for _ in range(3)])
    alignment = ensemble_context_alignment(ens)

    assert alignment["k_ref"] == 0
    assert alignment["ref_index"] == 0
    assert all(perm.size == 0 for perm in alignment["perm"])

    # Novel slot is index Kref == 0, so every member's whole row lands there.
    row = np.zeros(4)
    row[0] = 1.0
    aligned = ensemble_context_vector(ens, lambda m: row)
    np.testing.assert_allclose(aligned[0], 1.0, atol=1e-15)
    np.testing.assert_allclose(aligned[1:], 0.0, atol=1e-15)

    assert ensemble_context_density(ens, lambda m: {0: np.zeros(3)}) == {}


def test_a_fresh_ensemble_already_holds_one_context():
    """Sanity anchor for the guard above: construction instantiates context 0."""
    ens = RealTimeCOINEnsemble(runs=3, seed=5, num_particles=20, max_contexts=3)
    alignment = ensemble_context_alignment(ens)

    assert alignment["k_ref"] == 1
    row = ens.responsibilities_vector()
    np.testing.assert_allclose(row.sum(), 1.0, atol=1e-12)
    assert ens.stationary_context_probabilities().shape == (1,)


# ----------------------------------------------------------------------
# Reporting-only: no mutation, no randomness
# ----------------------------------------------------------------------


def test_phase2_queries_do_not_mutate_any_member():
    """Alignment is reporting-only (SPEC 10.4): members come back untouched."""
    ens = driven_ensemble()
    grid = np.linspace(-1, 1, 21)
    before = [member_fingerprint(m) for m in ens.members]

    for _ in range(2):        # twice: a cached alignment must not drift either
        for name in VECTOR_QUERIES:
            getattr(ens, name)()
        for name in DENSITY_QUERIES:
            getattr(ens, name)(grid)

    for r, (member, snapshot) in enumerate(zip(ens.members, before)):
        after = member_fingerprint(member)
        for name, value in snapshot.items():
            if name.startswith("__"):
                assert after[name] == value, "member %d %s changed" % (r, name)
                continue
            np.testing.assert_array_equal(
                after[name], value, err_msg="member %d %s" % (r, name)
            )
    assert ens.Trial == 8


def test_phase2_queries_are_pure_functions_of_the_state():
    """Repeating every query returns bit-identical values (no randomness)."""
    ens = driven_ensemble()
    grid = np.linspace(-1, 1, 21)

    for name in VECTOR_QUERIES:
        first = np.asarray(getattr(ens, name)(), dtype=float)
        second = np.asarray(getattr(ens, name)(), dtype=float)
        assert np.array_equal(first, second, equal_nan=True), name
    for name in DENSITY_QUERIES:
        first = getattr(ens, name)(grid)
        second = getattr(ens, name)(grid)
        assert sorted(first) == sorted(second), name
        for key in first:
            assert np.array_equal(first[key], second[key], equal_nan=True), (
                "%s[%s]" % (name, key)
            )


# ----------------------------------------------------------------------
# The two averaging rules are genuinely different
# ----------------------------------------------------------------------


def test_vector_rule_zero_fills_and_divides_by_the_run_count():
    """A synthetic member row is scattered, zero-filled and divided by R."""
    ens = driven_ensemble()
    alignment = ensemble_context_alignment(ens)
    k_ref = alignment["k_ref"]
    c_max = ens.members[0].max_contexts + 1

    # Give every member all its mass on its FIRST context, so the aligned row
    # must place 1/R at each of the reference labels those contexts map to.
    def one_hot(member):
        row = np.zeros(c_max)
        row[0] = 1.0
        return row

    aligned = ensemble_context_vector(ens, one_hot)
    expected = np.zeros(c_max)
    for r, perm in enumerate(alignment["perm"]):
        # A member with no context has its novel slot at index 0.
        expected[perm[0] if perm.size else k_ref] += 1.0
    expected /= ens.runs

    np.testing.assert_allclose(aligned, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(aligned.sum(), 1.0, rtol=0, atol=1e-12)


def test_density_rule_divides_by_the_contributing_runs_not_by_R():
    """NaN-omit: only run 0 supplies a density, so the mean IS run 0's."""
    ens = driven_ensemble()
    alignment = ensemble_context_alignment(ens)
    assert alignment["k_ref"] >= 1

    row = np.array([1.0, 2.0, 3.0, 4.0])
    members = list(ens.members)

    def only_first(member):
        # Label 0 exists in every member with a context; supply it for member 0
        # alone, so the contributing-run count for its reference label is 1.
        if member is members[0] and alignment["k"][0] > 0:
            return {0: row}
        return {}

    averaged = ensemble_context_density(ens, only_first)
    key = int(alignment["perm"][0][0])
    assert sorted(averaged) == [key]
    # Divided by 1 (the contributing runs), NOT by R - the zero-fill rule would
    # have produced row / R here.
    np.testing.assert_allclose(averaged[key], row, rtol=0, atol=1e-15)


def test_density_rule_averages_over_the_contributing_runs():
    """Two runs supplying the same reference label give their mean."""
    ens = driven_ensemble()
    alignment = ensemble_context_alignment(ens)
    assert alignment["k_ref"] >= 1
    assert int(np.sum(alignment["k"] > 0)) >= 2, "need two members with contexts"

    holders = [r for r in range(ens.runs) if alignment["k"][r] > 0][:2]
    # Both members' context 0 need not map to the same reference label; pick a
    # label they share by construction instead: give each its OWN context 0 and
    # only average where the reference labels coincide.
    values = {holders[0]: 2.0, holders[1]: 6.0}
    members = list(ens.members)

    def per_member(member):
        r = members.index(member)
        if r in values and alignment["k"][r] > 0:
            return {0: np.full(3, values[r])}
        return {}

    averaged = ensemble_context_density(ens, per_member)
    label_a = int(alignment["perm"][holders[0]][0])
    label_b = int(alignment["perm"][holders[1]][0])
    if label_a == label_b:
        np.testing.assert_allclose(averaged[label_a], np.full(3, 4.0))
    else:
        np.testing.assert_allclose(averaged[label_a], np.full(3, values[holders[0]]))
        np.testing.assert_allclose(averaged[label_b], np.full(3, values[holders[1]]))


# ----------------------------------------------------------------------
# The matching itself, on synthetic members
# ----------------------------------------------------------------------


def test_matching_follows_prototype_distance_not_slot_order():
    """A member whose contexts are listed in the OPPOSITE order is re-labelled.

    The load-bearing check on the cost matrix: a constant cost, or a transposed
    one, would leave the identity matching in place and still satisfy every
    sum-to-1 property, so the labels are pinned directly.
    """
    reference = _StubMember([1.0, -1.0, 0.25])       # labels 0, 1, 2
    flipped = _StubMember([-0.95, 1.05, 0.3])        # same contexts, reordered
    ens = _StubEnsemble([reference, flipped])
    alignment = ensemble_context_alignment(ens)

    assert alignment["k_ref"] == 3
    assert alignment["ref_index"] == 0
    np.testing.assert_array_equal(alignment["perm"][0], [0, 1, 2])
    np.testing.assert_array_equal(alignment["perm"][1], [1, 0, 2])


def test_matching_is_euclidean_on_the_multi_dimensional_path():
    """MD prototypes are matched by distance between VECTORS, not by axis 0."""
    # Axis 0 alone would pair (0.0, 1.0) with (0.1, -0.9); the full vector
    # distance pairs it with (-0.2, 0.9) instead.
    reference = _StubMember([[0.0, 1.0], [0.1, -1.0]])
    other = _StubMember([[0.1, -0.9], [-0.2, 0.9]])
    ens = _StubEnsemble([reference, other])
    alignment = ensemble_context_alignment(ens)

    assert alignment["k_ref"] == 2
    np.testing.assert_array_equal(alignment["perm"][1], [1, 0])


def test_unequal_context_counts_zero_fill_the_unmatched_labels():
    """A short member contributes 0 to the reference labels it does not hold."""
    max_contexts = 3
    c_max = max_contexts + 1
    ens = _StubEnsemble([
        _StubMember([1.0, -1.0, 0.25], max_contexts),   # K = 3  (reference)
        _StubMember([-0.9], max_contexts),              # K = 1, matches label 1
        _StubMember(np.zeros(0), max_contexts),         # K = 0, all novel
    ])
    alignment = ensemble_context_alignment(ens)
    assert list(alignment["k"]) == [3, 1, 0]
    assert alignment["k_ref"] == 3
    np.testing.assert_array_equal(alignment["perm"][1], [1])

    rows = {
        0: np.array([0.5, 0.3, 0.1, 0.1]),   # 3 real + novel
        1: np.array([0.8, 0.2, 0.0, 0.0]),   # 1 real + novel at index 1
        2: np.array([1.0, 0.0, 0.0, 0.0]),   # novel at index 0
    }
    members = list(ens.members)
    aligned = ensemble_context_vector(ens, lambda m: rows[members.index(m)])

    expected = np.zeros(c_max)
    expected[0] = 0.5                        # only run 0 holds label 0
    expected[1] = 0.3 + 0.8                  # runs 0 and 1
    expected[2] = 0.1                        # only run 0
    expected[3] = 0.1 + 0.2 + 1.0            # every run's novel mass
    expected /= 3.0

    np.testing.assert_allclose(aligned, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(aligned.sum(), 1.0, rtol=0, atol=1e-15)


def test_density_omits_the_runs_that_lack_a_reference_label():
    """NaN-omit with a genuinely absent key: label 2 is held by one run only."""
    ens = _StubEnsemble([
        _StubMember([1.0, -1.0, 0.25]),     # K = 3 (reference)
        _StubMember([-0.9, 1.1]),           # K = 2 -> labels 1 and 0
    ])
    alignment = ensemble_context_alignment(ens)
    np.testing.assert_array_equal(alignment["perm"][1], [1, 0])

    members = list(ens.members)
    densities = {
        0: {0: np.full(2, 2.0), 1: np.full(2, 4.0), 2: np.full(2, 6.0)},
        1: {0: np.full(2, 8.0), 1: np.full(2, 10.0)},
    }
    averaged = ensemble_context_density(
        ens, lambda m: densities[members.index(m)]
    )

    assert sorted(averaged) == [0, 1, 2]
    # Label 0: run 0's context 0 and run 1's context 1 -> mean(2, 10).
    np.testing.assert_allclose(averaged[0], np.full(2, 6.0))
    # Label 1: run 0's context 1 and run 1's context 0 -> mean(4, 8).
    np.testing.assert_allclose(averaged[1], np.full(2, 6.0))
    # Label 2: run 0 only, so its own density - NOT halved by R.
    np.testing.assert_allclose(averaged[2], np.full(2, 6.0))
    assert averaged[2] is not densities[0][2], "the member's array is not aliased"


def test_a_label_no_run_holds_is_absent_from_the_density_map():
    """A reference label whose density every run omits produces no key."""
    ens = _StubEnsemble([_StubMember([1.0, -1.0]), _StubMember([0.9, -1.1])])
    assert ensemble_context_alignment(ens)["k_ref"] == 2
    # Every member reports a density for context 0 only.
    averaged = ensemble_context_density(ens, lambda m: {0: np.ones(4)})
    assert sorted(averaged) == [0]

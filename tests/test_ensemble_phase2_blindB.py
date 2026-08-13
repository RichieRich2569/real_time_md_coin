"""Blind behavioural tests for ``RealTimeCOINEnsemble`` Phase 2 - suite B.

Port of ``tests/test_ensemble_phase2_blindB.m``. Authored strictly against
``docs/SPEC_ensemble.md`` Part 10, independently of suite A and of any
implementation.

Phase 2 adds cross-run context alignment onto a common reference frame (the
member with the most contexts) and two averaging rules:

* ZERO-FILL / divide-by-``R`` for probability vectors
  (``responsibilities_vector``, ``predicted_context_probabilities_vector``,
  ``sampled_context_count``, ``stationary_context_probabilities``), so they sum
  to 1;
* NaN-OMIT mean for per-context densities
  (``state_given_context_probability``,
  ``state_feedback_given_context_probability``).

Coverage (mapped to SPEC 10.5), chosen so the aligner need NOT be
re-implemented:

1. ``runs == 1`` reduction (scalar + MD): each Phase-2 query equals its single
   member's query exactly (identity matching, ``Kref = K_1``).
2. Probability conservation: the four probability quantities sum to 1, are
   non-negative, and have the right length.
3. Member-permutation invariance (order-independent naive oracle in the trivial
   regime, plus reproducibility).
4. Reproducibility and executor invariance (``max_cores`` / ``segment_length``).
5. Trivial-alignment regime: a single shared context, so the aligned average
   equals the naive slot-by-slot / key-by-key member average.
6. Density normalisation: each per-context density integrates to ~1 (trapezoid).
7. Shape / key correctness.

Interpretations, documented for the reviewer
--------------------------------------------
* **RNG contract.** The MATLAB suite had to detect whether the ensemble used a
  Threefry or a Philox substream (SPEC 3.1 permits either). This package pins
  the stream: member ``k`` is
  ``RealTimeCOIN(rng=numpy.random.default_rng(SeedSequence(seed).spawn(runs)[k]))``,
  so :func:`build_members` reproduces the ensemble's members directly.
* **"Sums to 1 within a few eps" (10.5.2)** is implemented as
  ``abs(sum - 1) <= 1e-12``, comfortably above the ``O(R * eps)`` accumulation
  of a zero-fill/divide-by-``R`` average yet far tighter than any modelling
  tolerance.
* **Trivial-alignment regime** is enforced as a SINGLE shared context (``K == 1``
  for every member): the global labels are then unambiguous AND the novel slot
  sits at the same index in every member, so the naive slot-by-slot / key-by-key
  member average provably equals the reference-frame aligned average. If a
  member splits, the ``K == 1`` assertion flags the scenario rather than
  silently passing.
* **Member-permutation invariance (10.5.3).** Unlike MATLAB, the member list is
  reachable here, so it is reversed literally - inside the trivial regime,
  where the reference frame is order-independent (in general the "most
  contexts, ties to the lowest index" rule means a reorder can pick a different
  but equally valid reference on a tie, permuting the output labels). The
  order-independence of the naive equal-weight average is asserted as well.
* **Shapes and labels** follow the package convention: ``(L,)`` rather than
  ``1 x L``, ``(0,)`` rather than ``[]``, an MD grid of ``(K, N)`` with one
  point per ROW, and 0-based context labels (reference labels ``0 .. Kref - 1``,
  novel slot ``Kref``).
* **The global ``numpy.random`` state** is never touched by any of this, which
  replaces the MATLAB suite's save/restore guards; the check itself lives in
  ``test_ensemble_blindB.test_global_rng_untouched``.
"""

from __future__ import annotations

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN, RealTimeCOINEnsemble

#: ``numpy.trapz`` was renamed ``numpy.trapezoid`` in numpy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: The four probability quantities of SPEC 10.4.
VECTOR_METHODS = (
    "responsibilities_vector",
    "predicted_context_probabilities_vector",
    "sampled_context_count",
)

#: The two per-context density queries of SPEC 10.4.
DENSITY_METHODS = (
    "state_given_context_probability",
    "state_feedback_given_context_probability",
)

#: "Sums to 1 within a few eps" (SPEC 10.5.2).
SUM_TOL = 1e-12


# ========================================================================= #
# ---------------------------- oracle helpers ----------------------------- #
# ========================================================================= #


def build_members(runs, seed, params):
    """Construct ``R`` members under the ensemble's pinned per-member streams.

    Parameters
    ----------
    runs : int
        Number of members ``R``.
    seed : int
        Ensemble base seed.
    params : dict
        Member constructor keywords.

    Returns
    -------
    list of RealTimeCOIN
    """
    children = np.random.SeedSequence(seed).spawn(runs)
    return [RealTimeCOIN(rng=np.random.default_rng(c), **params) for c in children]


def drive_ens(ens, cues, obs, has_q):
    """Step an ensemble over a ``(q, y)`` stream honouring the ``has_q`` pattern."""
    for t, y in enumerate(obs):
        if has_q[t]:
            ens.observe_q(cues[t])
        ens.observe_y(y)


def drive_members(members, cues, obs, has_q):
    """Step every reference member over the identical stream."""
    for t, y in enumerate(obs):
        for member in members:
            if has_q[t]:
                member.observe_q(cues[t])
            member.observe_y(y)


def member_context_counts(members):
    """Per-member instantiated-context count ``K`` (single-member API)."""
    return np.array(
        [int(m.context_alignment()["K"]) for m in members], dtype=int
    )                                                            # (R,)


def naive_vector_mean(members, method):
    """Plain equal-weight mean of a ``(max_contexts + 1,)`` probability row.

    A valid stand-in for the reference-frame aligned average ONLY when every
    member shares the same context count, so their novel slots coincide - i.e.
    only inside the trivial-alignment regime.
    """
    rows = [
        np.asarray(getattr(m, method)(), dtype=float).reshape(-1) for m in members
    ]
    return np.mean(np.stack(rows, axis=0), axis=0)


# ========================================================================= #
# ---------------------------- assert helpers ----------------------------- #
# ========================================================================= #


def assert_equaln(name, actual, expected):
    """Bit-identical equality, ``nan`` matching ``nan`` (MATLAB ``isequaln``)."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    assert a.shape == b.shape, "%s: shape %s != %s" % (name, a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=True), "%s: not bit-identical" % name


def assert_map_equal(name, ma, mb):
    """Same keys and bit-identical values."""
    assert isinstance(ma, dict), "%s: A is not a mapping" % name
    assert isinstance(mb, dict), "%s: B is not a mapping" % name
    assert sorted(ma) == sorted(mb), "%s: keys %s vs %s" % (
        name, sorted(ma), sorted(mb),
    )
    for key in sorted(ma):
        assert_equaln("%s value (key=%s)" % (name, key), ma[key], mb[key])


def assert_density_map_shape(name, densities, k_ref, n_points):
    """Keys are a subset of ``0 .. Kref - 1``; values are finite ``(n_points,)``."""
    assert isinstance(densities, dict), "%s: not a mapping" % name
    for key in sorted(densities):
        assert isinstance(key, (int, np.integer)), "%s: key %r is not an int" % (
            name, key,
        )
        assert 0 <= int(key) < k_ref, "%s: key %s outside 0 .. Kref - 1" % (name, key)
        row = np.asarray(densities[key], dtype=float)
        assert row.shape == (n_points,), "%s: value shape (key=%s) %s" % (
            name, key, row.shape,
        )
        assert np.all(np.isfinite(row)), "%s: value not finite (key=%s)" % (name, key)


def assert_phase2_bit_equal(name, e1, e2, grid):
    """Bit-identical across every Phase-2 query."""
    for method in VECTOR_METHODS:
        assert_equaln(
            "%s %s" % (name, method), getattr(e1, method)(), getattr(e2, method)()
        )
    assert_equaln(
        "%s stationary_context_probabilities" % name,
        e1.stationary_context_probabilities(),
        e2.stationary_context_probabilities(),
    )
    for method in DENSITY_METHODS:
        assert_map_equal(
            "%s %s" % (name, method),
            getattr(e1, method)(grid),
            getattr(e2, method)(grid),
        )


def assert_all_phase2_equal_member(tag, ens, member, grid):
    """Every Phase-2 query of a ``runs == 1`` ensemble equals its member's."""
    for method in VECTOR_METHODS:
        assert_equaln(
            "%s %s" % (tag, method), getattr(ens, method)(), getattr(member, method)()
        )
    assert_equaln(
        "%s stationary_context_probabilities" % tag,
        ens.stationary_context_probabilities(),
        member.stationary_context_probabilities(),
    )
    for method in DENSITY_METHODS:
        assert_map_equal(
            "%s %s" % (tag, method),
            getattr(ens, method)(grid),
            getattr(member, method)(grid),
        )


# ========================================================================= #
# Guarantee 1 (scalar): runs==1 reduces exactly to the single seeded member.
# ========================================================================= #


def test_runs_one_reduction_scalar():
    """``runs == 1`` equals the single member, including on missing trials."""
    params = dict(num_particles=40, max_contexts=4)
    seed = 1010
    cues = [1, 2, 1, np.nan, 2]
    has_q = [True] * 5
    obs = [0.2, -0.1, 0.05, np.nan, 0.3]
    grid = np.linspace(-1, 1, 17)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    member = build_members(1, seed, params)[0]

    for t, y in enumerate(obs):
        if has_q[t]:
            ens.observe_q(cues[t])
            member.observe_q(cues[t])
        ens.observe_y(y)
        member.observe_y(y)
        assert_all_phase2_equal_member(
            "runs==1 scalar (t=%d)" % (t + 1), ens, member, grid
        )


# ========================================================================= #
# Guarantee 1 (multi-dimensional, state_dim==2): runs==1 reduction.
# ========================================================================= #


def test_runs_one_reduction_md():
    """The multi-dimensional ``runs == 1`` reduction.

    SPEC 10.5.1 permits "a few eps" on the MD path for the probability vectors
    (the aligned average consolidates each member's tiny residual mass in unused
    slots into the novel slot), so those are compared with a tolerance; the
    density maps are still bit-identical.
    """
    params = dict(num_particles=40, max_contexts=4, state_dim=2)
    seed = 2020
    cues = [1, 1, 2, 2, 1]
    has_q = [True] * 5
    obs = [
        np.array([0.10, 0.05]),
        np.array([0.20, -0.10]),
        np.array([np.nan, 0.10]),
        np.array([0.15, 0.20]),
        None,
    ]
    grid = np.column_stack(
        [np.linspace(-0.4, 0.4, 7), np.linspace(-0.3, 0.3, 7)]
    )                                                            # (7, 2)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    member = build_members(1, seed, params)[0]

    for t, y in enumerate(obs):
        if has_q[t]:
            ens.observe_q(cues[t])
            member.observe_q(cues[t])
        ens.observe_y(y)
        member.observe_y(y)
        tag = "runs==1 MD (t=%d)" % (t + 1)

        for method in VECTOR_METHODS + ("stationary_context_probabilities",):
            actual = np.asarray(getattr(ens, method)(), dtype=float)
            expected = np.asarray(getattr(member, method)(), dtype=float)
            assert actual.shape == expected.shape, "%s %s shape" % (tag, method)
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=1e-12,
                err_msg="%s %s" % (tag, method),
            )
        for method in DENSITY_METHODS:
            assert_map_equal(
                "%s %s" % (tag, method),
                getattr(ens, method)(grid),
                getattr(member, method)(grid),
            )


# ========================================================================= #
# Guarantee 2 & 7: probability conservation + shape/key correctness (R>1).
# ========================================================================= #


def test_probability_conservation_and_shapes():
    """The four probability rows sum to 1; density maps have legal keys/rows."""
    max_contexts = 4
    params = dict(num_particles=60, max_contexts=max_contexts)
    runs, seed = 4, 3030
    # A two-phase perturbation (positive then negative) instantiates >= 1 context.
    cues = [1, 1, 2, 2, 1, 2, 1]
    has_q = [True] * len(cues)
    obs = [0.30, 0.35, -0.30, -0.35, 0.32, -0.28, 0.30]
    grid = np.linspace(-2, 2, 41)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    drive_ens(ens, cues, obs, has_q)

    # Independent oracle members (same stream contract) supply Kref for the
    # shape assertions, through the single-member API only.
    members = build_members(runs, seed, params)
    drive_members(members, cues, obs, has_q)
    k_ref = int(member_context_counts(members).max())
    assert k_ref >= 1, "conservation: at least one context instantiated"

    for method in VECTOR_METHODS:
        v = np.asarray(getattr(ens, method)(), dtype=float)
        assert v.shape == (max_contexts + 1,), "conservation %s length" % method
        assert np.all(np.isfinite(v)), "conservation %s finite" % method
        assert np.all(v >= -1e-12), "conservation %s non-negative" % method
        assert abs(float(v.sum()) - 1.0) <= SUM_TOL, (
            "conservation %s sums to 1 (got %.17g)" % (method, float(v.sum()))
        )

    sp = np.asarray(ens.stationary_context_probabilities(), dtype=float)
    assert sp.shape == (k_ref,), "conservation stationary length"
    assert np.all(np.isfinite(sp)), "conservation stationary finite"
    assert np.all(sp >= -1e-12), "conservation stationary non-negative"
    assert abs(float(sp.sum()) - 1.0) <= SUM_TOL, (
        "conservation stationary sums to 1 (got %.17g)" % float(sp.sum())
    )

    for method in DENSITY_METHODS:
        assert_density_map_shape(
            "conservation %s" % method, getattr(ens, method)(grid), k_ref, grid.size
        )


# ========================================================================= #
# Guarantee 4a: reproducibility -- same (seed, runs) => bit-identical Phase-2.
# ========================================================================= #


def test_reproducibility():
    """Two identically configured ensembles agree bit for bit, every trial."""
    params = dict(num_particles=40, max_contexts=4)
    runs, seed = 4, 4040
    cues = [1, 2, 1, np.nan, 2, 1]
    has_q = [True] * len(cues)
    obs = [0.2, -0.1, 0.3, 0.05, -0.2, 0.15]
    grid = np.linspace(-1, 1, 15)

    a = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    b = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for t, y in enumerate(obs):
        if has_q[t]:
            a.observe_q(cues[t])
            b.observe_q(cues[t])
        a.observe_y(y)
        b.observe_y(y)
        assert_phase2_bit_equal("reproducibility (t=%d)" % (t + 1), a, b, grid)


# ========================================================================= #
# Guarantee 4b: executor invariance -- max_cores / segment_length are scheduling
# knobs only, read off the live stepping state (simulate() traces carry no
# context-indexed queries).
# ========================================================================= #


def test_executor_invariance():
    """Serial and parallel configurations give bit-identical Phase-2 outputs."""
    params = dict(num_particles=40, max_contexts=4)
    runs, seed = 4, 5050
    cues = [1, 2, 2, 1, np.nan, 2]
    has_q = [True] * len(cues)
    obs = [0.15, -0.05, 0.25, 0.2, 0.1, -0.15]
    grid = np.linspace(-1, 1, 15)

    serial = RealTimeCOINEnsemble(
        runs=runs, seed=seed, max_cores=0, segment_length=1, **params
    )
    parallel = RealTimeCOINEnsemble(
        runs=runs, seed=seed, max_cores=2, segment_length=3, **params
    )
    for t, y in enumerate(obs):
        if has_q[t]:
            serial.observe_q(cues[t])
            parallel.observe_q(cues[t])
        serial.observe_y(y)
        parallel.observe_y(y)
        assert_phase2_bit_equal(
            "executor invariance (t=%d)" % (t + 1), serial, parallel, grid
        )


# ========================================================================= #
# Guarantee 5 & 3: trivial-alignment regime + member-permutation invariance.
# ========================================================================= #


def _trivial_setup():
    """Ensemble plus reference members in the single-shared-context regime."""
    params = dict(num_particles=80, max_contexts=4)
    runs, seed, n_trials = 4, 6060, 5
    cues = [1] * n_trials
    has_q = [True] * n_trials
    obs = [0.3] * n_trials

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    drive_ens(ens, cues, obs, has_q)
    members = build_members(runs, seed, params)
    drive_members(members, cues, obs, has_q)

    counts = member_context_counts(members)
    assert np.all(counts == 1), (
        "trivial: every member must have exactly one context, got %s" % counts
    )
    return ens, members, params["max_contexts"]


def test_trivial_alignment_vectors():
    """Aligned vectors equal the naive member mean in the trivial regime."""
    ens, members, max_contexts = _trivial_setup()
    tol = 1e-9

    for method in VECTOR_METHODS:
        v_ens = np.asarray(getattr(ens, method)(), dtype=float)
        v_naive = naive_vector_mean(members, method)
        assert v_ens.shape == (max_contexts + 1,), "trivial %s length" % method
        np.testing.assert_allclose(
            v_ens, v_naive, rtol=0, atol=tol,
            err_msg="trivial %s == naive member mean" % method,
        )
        # Permutation invariance of the equal-weight average itself.
        v_flip = naive_vector_mean(members[::-1], method)
        np.testing.assert_allclose(
            v_naive, v_flip, rtol=0, atol=tol,
            err_msg="permutation %s invariant to member order" % method,
        )

    sp = np.asarray(ens.stationary_context_probabilities(), dtype=float)
    assert sp.shape == (1,), "trivial stationary length"
    np.testing.assert_allclose(sp, [1.0], rtol=0, atol=tol)


@pytest.mark.parametrize("method", DENSITY_METHODS)
def test_trivial_alignment_densities(method):
    """Aligned densities equal the naive key-by-key member mean.

    With ``K == 1`` for every member, reference label 0 is present in all of
    them, so the NaN-omit mean reduces to a plain mean over all ``R``.
    """
    ens, members, _ = _trivial_setup()
    grid = np.linspace(-2, 2, 41)

    aligned = getattr(ens, method)(grid)
    assert sorted(aligned) == [0], "%s key set is {0}, got %s" % (
        method, sorted(aligned),
    )

    rows = []
    for member in members:
        member_map = getattr(member, method)(grid)
        assert 0 in member_map, "%s: every member contributes context 0" % method
        rows.append(np.asarray(member_map[0], dtype=float))
    naive = np.mean(np.stack(rows, axis=0), axis=0)

    np.testing.assert_allclose(
        np.asarray(aligned[0], dtype=float), naive, rtol=0, atol=1e-9,
        err_msg="%s == naive member mean" % method,
    )


def test_member_permutation_invariance():
    """Reversing the ensemble's stored members leaves Phase-2 outputs put.

    SPEC 10.5.3, checked literally (MATLAB could not reach its private member
    list). Restricted to the trivial regime, where the reference frame is
    order-independent - see this module's docstring.
    """
    ens, _members, _ = _trivial_setup()
    grid = np.linspace(-2, 2, 41)

    before_vectors = {
        method: np.asarray(getattr(ens, method)(), dtype=float)
        for method in VECTOR_METHODS + ("stationary_context_probabilities",)
    }
    before_densities = {
        method: getattr(ens, method)(grid) for method in DENSITY_METHODS
    }

    ens._members = list(reversed(ens._members))

    for method, expected in before_vectors.items():
        np.testing.assert_allclose(
            np.asarray(getattr(ens, method)(), dtype=float), expected,
            rtol=0, atol=1e-12, err_msg="permutation %s" % method,
        )
    for method, expected in before_densities.items():
        actual = getattr(ens, method)(grid)
        assert sorted(actual) == sorted(expected), "permutation %s keys" % method
        for key in sorted(expected):
            np.testing.assert_allclose(
                np.asarray(actual[key], dtype=float),
                np.asarray(expected[key], dtype=float),
                rtol=0, atol=1e-12,
                err_msg="permutation %s (key=%s)" % (method, key),
            )


# ========================================================================= #
# Guarantee 6: per-context densities integrate to ~1 over a wide, fine grid.
# ========================================================================= #


def test_density_normalisation():
    """Each aligned per-context state density integrates to ~1 (trapezoid)."""
    params = dict(num_particles=80, max_contexts=4)
    runs, seed = 3, 7070
    cues = [1, 1, 2, 2, 1, 2]
    has_q = [True] * len(cues)
    obs = [0.30, 0.35, -0.30, -0.35, 0.32, -0.28]
    grid = np.linspace(-3, 3, 601)
    integration_tol = 0.05        # discretisation slack, as in the MD queries

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    drive_ens(ens, cues, obs, has_q)

    densities = ens.state_given_context_probability(grid)
    assert densities, "density normalisation: at least one context density"
    for key in sorted(densities):
        row = np.asarray(densities[key], dtype=float)
        assert row.shape == (grid.size,), "density row shape (key=%s)" % key
        assert np.all(row >= -1e-12), "density non-negative (key=%s)" % key
        integral = float(_trapezoid(row, grid))
        assert abs(integral - 1.0) <= integration_tol, (
            "per-context density integrates to ~1 (key=%s, got %.4f)"
            % (key, integral)
        )

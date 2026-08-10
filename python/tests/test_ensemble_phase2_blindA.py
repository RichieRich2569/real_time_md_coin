"""Blind behavioural tests for ``RealTimeCOINEnsemble`` Phase 2 - suite A.

Port of ``tests/test_ensemble_phase2_blindA.m``. Authored strictly against
``docs/SPEC_ensemble.md`` Part 10, NOT against any implementation. Phase 2 adds
cross-run context alignment for six context-indexed ensemble readouts:

    responsibilities_vector
    predicted_context_probabilities_vector
    sampled_context_count
    stationary_context_probabilities
    state_given_context_probability
    state_feedback_given_context_probability

The checks are chosen (per SPEC 10.5) so that the cross-run prototype-Hungarian
aligner does NOT have to be re-implemented as an oracle. The independent oracle
builds ``R`` ordinary :class:`~realtimecoin.RealTimeCOIN` members under the same
random-stream contract the Phase-1 blind suites use, so the reference members
line up bit for bit with the ensemble's. ``RealTimeCOIN`` itself is not the code
under test and is used fully.

Coverage maps to SPEC 10.5:

1. ``runs == 1`` reduction to the single member (scalar and MD; vectors and
   density maps identical, including novel-slot placement)
2. probability conservation (the four probability quantities sum to 1)
3. member-permutation invariance
4. reproducibility and executor invariance (``max_cores``, ``segment_length``)
5. trivial-alignment regime: aligned average == naive slot/key average
6. per-context density normalisation (integrates to ~1)
7. shape / key correctness

Adaptations from the MATLAB suite (the Phase-1 ports' conventions)
------------------------------------------------------------------
* **No generator probing.** The MATLAB suite had to auto-detect whether the
  ensemble used a Threefry or a Philox substream, because SPEC 3.1 permits
  either. The Python contract pins the stream exactly - member ``k`` is
  ``RealTimeCOIN(rng=numpy.random.default_rng(SeedSequence(seed).spawn(runs)[k]))``
  - so :func:`oracle_members` builds the reference directly. Strictly stronger
  than detection.
* **Global-stream juggling becomes global-stream non-use.** Nothing here touches
  the legacy global ``numpy.random`` state, so the MATLAB save/restore guards
  have no counterpart; ``test_ensemble_blindA`` asserts that non-use directly.
* **Shapes and labels.** Scalar quantities are ``(L,)`` rather than ``1 x L``;
  an empty stationary distribution is ``(0,)`` rather than ``[]``; an MD query
  grid is ``(K, N)`` with one point per ROW rather than ``N x K``; context
  labels are 0-based, so reference labels run ``0 .. Kref - 1`` and the novel
  slot is index ``Kref``.
* **Member-permutation invariance is checked literally**, by reversing the
  ensemble's stored member list - which MATLAB could not do (its members are
  private) and which SPEC 10.5.3 asks for. It is checked only inside the
  trivial-alignment regime, where every member holds ``K == 1``: there the
  reference frame is order-independent by construction. In general the
  reference member is "the member with the most contexts, ties to the lowest
  index", so on a TIE a reorder may pick a different (equally valid) reference
  and permute the output labels; the SPEC's invariance claim is therefore read
  as up-to-relabelling outside that regime, and the order-independence of the
  equal-weight naive average is asserted as the MATLAB suite does.
"""

from __future__ import annotations

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN, RealTimeCOINEnsemble

from helpers import integrate_2d

#: ``numpy.trapz`` was renamed ``numpy.trapezoid`` in numpy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: Vector-valued Phase-2 queries taking no arguments.
VECTOR_QUERIES = (
    "responsibilities_vector",
    "predicted_context_probabilities_vector",
    "sampled_context_count",
)

#: Density-valued Phase-2 queries taking a grid.
DENSITY_QUERIES = (
    "state_given_context_probability",
    "state_feedback_given_context_probability",
)


# ==========================================================================
# Oracle helpers - the pinned RNG contract and the reference frame
# ==========================================================================


def oracle_members(seed, runs, params):
    """Build ``R`` plain members under the pinned per-member streams.

    Member ``k`` is CONSTRUCTED with child ``k`` of
    ``SeedSequence(seed).spawn(runs)``, exactly as the ensemble's members are,
    so its construction randomness belongs to its own stream.

    Parameters
    ----------
    seed : int
        Ensemble base seed.
    runs : int
        Number of members ``R``.
    params : dict
        Member constructor keywords.

    Returns
    -------
    list of RealTimeCOIN
        The reference members.
    """
    children = np.random.SeedSequence(seed).spawn(runs)
    return [RealTimeCOIN(rng=np.random.default_rng(c), **params) for c in children]


def oracle_step(members, q, y):
    """Drive every reference member with the identical ``(q, y)``."""
    for member in members:
        member.observe_q(q)
        member.observe_y(y)


def drive(ens, members, cues, obs):
    """Step an ensemble and its reference members over a whole stream."""
    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
        if members is not None:
            oracle_step(members, q, y)


def reference_frame(members):
    """Reference member index, ``Kref`` and the per-member context counts.

    The reference member is the one holding the most contexts, ties broken by
    the lowest index (SPEC 10.2). ``numpy.argmax`` returns the first maximiser,
    which is that rule.

    Parameters
    ----------
    members : sequence of RealTimeCOIN
        Reference members.

    Returns
    -------
    ref_index : int
    k_ref : int
    k : numpy.ndarray
        ``(R,)`` per-member context counts.
    """
    k = np.array(
        [int(m.context_alignment()["K"]) for m in members], dtype=int
    )                                                            # (R,)
    ref_index = int(np.argmax(k))
    return ref_index, int(k[ref_index]), k


def assert_trivial_frame(name, members):
    """Precondition for the trivial-alignment regime: every member has K == 1.

    With a single instantiated context per member the cross-run matching is a
    forced identity (its one real context maps to reference label 0, novel to
    the reference novel slot) whatever the prototypes are, so the aligned
    average must equal the naive slot-by-slot / key-by-key member average
    (SPEC 10.5.5).
    """
    _, _, k = reference_frame(members)
    assert np.all(k == 1), "%s: every member must have K == 1, got %s" % (name, k)


# ==========================================================================
# Naive (identity-frame) averaging oracles - valid ONLY in a trivial frame
# ==========================================================================


def naive_vector_average(members, method):
    """Equal-weight slot-by-slot mean of the member vectors.

    In a shared identity frame no run lacks a reference context and the slots
    beyond ``K`` are already zero, so the SPEC 10.3 zero-fill average reduces to
    the plain element-wise mean. Handles the empty stationary row.
    """
    rows = [
        np.asarray(getattr(m, method)(), dtype=float).reshape(-1) for m in members
    ]
    if rows[0].size == 0:
        return np.zeros(0)
    return np.mean(np.stack(rows, axis=0), axis=0)


def naive_density_average(members, method, values):
    """Key-by-key NaN-omit mean of the member density maps.

    Valid in a shared identity frame (member-local labels equal the reference
    labels): each key is averaged over the members that HAVE it (SPEC 10.3).
    """
    maps = [getattr(m, method)(values) for m in members]
    keys = sorted({key for mp in maps for key in mp})
    out = {}
    for key in keys:
        rows = np.stack(
            [np.asarray(mp[key], dtype=float) for mp in maps if key in mp], axis=0
        )                                                        # (n_contrib, K)
        finite = np.isfinite(rows)
        count = finite.sum(axis=0)
        total = np.where(finite, rows, 0.0).sum(axis=0)
        mean = np.full(rows.shape[1], np.nan)
        np.divide(total, count, out=mean, where=count > 0)
        out[key] = mean
    return out


# ==========================================================================
# Assertion helpers
# ==========================================================================


def assert_vec_equal(name, a, b, tol):
    """Closeness with matching shapes; nan aligns exactly; handles empties."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        assert a.size == b.size, "%s: one operand empty, the other not" % name
        return
    assert a.shape == b.shape, "%s: shape %s != %s" % (name, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "%s: nan pattern" % name
    finite = ~np.isnan(a)
    if finite.any():
        worst = float(np.max(np.abs(a[finite] - b[finite])))
        assert worst <= tol, "%s: max diff %g > tol %g" % (name, worst, tol)


def assert_bit(name, a, b):
    """Bit-identical equality (``nan`` counts as equal to ``nan``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert np.array_equal(a, b, equal_nan=True), "FAILED bit-identical: %s" % name


def assert_map_close(name, ma, mb, tol, k_pts):
    """Same keys and elementwise-close values; each value ``(k_pts,)``."""
    assert isinstance(ma, dict), "%s: not a mapping" % name
    assert set(ma) == set(mb), "%s: key sets differ %s vs %s" % (
        name, sorted(ma), sorted(mb),
    )
    for key in sorted(ma):
        row = np.asarray(ma[key], dtype=float)
        assert row.shape == (k_pts,), "%s: value row (key %s) has shape %s" % (
            name, key, row.shape,
        )
        assert_vec_equal("%s value (key %s)" % (name, key), row, mb[key], tol)


def assert_map_bit(name, ma, mb):
    """Same keys and bit-identical values."""
    assert set(ma) == set(mb), "%s: key sets differ %s vs %s" % (
        name, sorted(ma), sorted(mb),
    )
    for key in sorted(ma):
        assert_bit("%s (key %s)" % (name, key), ma[key], mb[key])


def assert_phase2_bit(name, a, b, grid):
    """Bit-identical Phase-2 outputs between two ensembles."""
    for method in VECTOR_QUERIES:
        assert_bit("%s %s" % (name, method), getattr(a, method)(), getattr(b, method)())
    assert_bit(
        "%s stationary" % name,
        a.stationary_context_probabilities(),
        b.stationary_context_probabilities(),
    )
    for method in DENSITY_QUERIES:
        assert_map_bit(
            "%s %s" % (name, method), getattr(a, method)(grid), getattr(b, method)(grid)
        )


# ==========================================================================
# Observation streams
# ==========================================================================


def ctx_scalar_stream():
    """Alternating blocks that instantiate at least one context."""
    return (
        [1, 1, 1, 2, 2, 2, 1, 1],
        [0.30, 0.28, 0.32, -0.30, -0.28, -0.31, 0.29, 0.30],
    )


def ctx_md_stream():
    """Multi-dimensional counterpart of :func:`ctx_scalar_stream`."""
    cues = [1, 1, 1, 2, 2, 2, 1, 1]
    obs = np.array(
        [[0.30, 0.28, 0.32, -0.30, -0.28, -0.31, 0.29, 0.30],
         [0.20, 0.18, 0.22, -0.20, -0.18, -0.21, 0.19, 0.20]]
    ).T                                                          # (T, 2)
    return cues, obs


def trivial_scalar_stream():
    """Constant small perturbation, so every member ends with K == 1."""
    return [1, 1, 1, 1, 1], [0.05, 0.05, 0.05, 0.05, 0.05]


def trivial_md_stream():
    """Multi-dimensional counterpart of :func:`trivial_scalar_stream`."""
    obs = np.array(
        [[0.05, 0.05, 0.05, 0.05, 0.05],
         [0.03, 0.03, 0.03, 0.03, 0.03]]
    ).T                                                          # (T, 2)
    return [1, 1, 1, 1, 1], obs


# ==========================================================================
# (1) runs == 1 reduction
# ==========================================================================


def test_runs_one_reduction_scalar():
    """Every Phase-2 query equals the single member's, trial by trial."""
    max_contexts = 4
    params = dict(num_particles=60, max_contexts=max_contexts, infer_bias=True)
    seed = 55
    cues, obs = ctx_scalar_stream()
    grid = np.linspace(-1.2, 1.2, 9)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    members = oracle_members(seed, 1, params)

    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)
        member = members[0]

        for method in VECTOR_QUERIES:
            value = getattr(ens, method)()
            assert value.shape == (max_contexts + 1,), (
                "runs1 scalar %s length" % method
            )
            assert_bit("runs1 scalar %s" % method, value, getattr(member, method)())

        assert_bit(
            "runs1 scalar stationary",
            ens.stationary_context_probabilities(),
            member.stationary_context_probabilities(),
        )

        for method in DENSITY_QUERIES:
            assert_map_close(
                "runs1 scalar %s" % method,
                getattr(ens, method)(grid),
                getattr(member, method)(grid),
                1e-12,
                grid.size,
            )


def test_runs_one_reduction_md():
    """The ``runs == 1`` reduction on the multi-dimensional path.

    SPEC 10.5.1 allows a few eps here (the zero-fill average consolidates each
    member's tiny residual mass in unused slots into the novel slot), so the
    vectors are compared with a tolerance rather than bit for bit.
    """
    max_contexts = 4
    params = dict(state_dim=2, num_particles=60, max_contexts=max_contexts)
    seed = 66
    cues, obs = ctx_md_stream()
    grid = np.column_stack(
        [np.linspace(-1, 1, 6), np.linspace(-0.5, 0.5, 6)]
    )                                                            # (6, 2)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    members = oracle_members(seed, 1, params)

    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)
        member = members[0]

        for method in VECTOR_QUERIES:
            value = getattr(ens, method)()
            assert value.shape == (max_contexts + 1,), "runs1 MD %s length" % method
            assert_vec_equal(
                "runs1 MD %s" % method, value, getattr(member, method)(), 1e-12
            )

        assert_vec_equal(
            "runs1 MD stationary",
            ens.stationary_context_probabilities(),
            member.stationary_context_probabilities(),
            1e-12,
        )

        for method in DENSITY_QUERIES:
            assert_map_close(
                "runs1 MD %s" % method,
                getattr(ens, method)(grid),
                getattr(member, method)(grid),
                1e-12,
                grid.shape[0],
            )


# ==========================================================================
# (2) Probability conservation with R > 1
# ==========================================================================


def conservation_checks(tag, ens, length, k_ref, tol):
    """Sum-to-1, non-negativity and length for the four probability rows.

    SPEC 10.3's zero-fill rule makes the conservation exact up to summation
    round-off.
    """
    for method in VECTOR_QUERIES:
        v = np.asarray(getattr(ens, method)(), dtype=float)
        assert v.shape == (length,), "%s %s length" % (tag, method)
        assert np.all(v >= -tol), "%s %s non-negative" % (tag, method)
        assert abs(float(v.sum()) - 1.0) <= tol, "%s %s sums to 1" % (tag, method)

    s = np.asarray(ens.stationary_context_probabilities(), dtype=float)
    assert s.shape == (k_ref,), "%s stationary length" % tag
    assert np.all(s >= -tol), "%s stationary non-negative" % tag
    assert abs(float(s.sum()) - 1.0) <= tol, "%s stationary sums to 1" % tag


def test_probability_conservation_scalar():
    """The four scalar-model probability rows conserve probability."""
    max_contexts = 4
    params = dict(num_particles=60, max_contexts=max_contexts, infer_bias=True)
    seed, runs = 123, 5
    cues, obs = ctx_scalar_stream()

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    drive(ens, members, cues, obs)

    _, k_ref, _ = reference_frame(members)
    assert k_ref >= 1, "conservation scalar Kref >= 1"
    conservation_checks("scalar", ens, max_contexts + 1, k_ref, 1e-9)


def test_probability_conservation_md():
    """The multi-dimensional counterpart of the conservation check."""
    max_contexts = 4
    params = dict(state_dim=2, num_particles=60, max_contexts=max_contexts)
    seed, runs = 123, 4
    cues, obs = ctx_md_stream()

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    drive(ens, members, cues, obs)

    _, k_ref, _ = reference_frame(members)
    assert k_ref >= 1, "conservation MD Kref >= 1"
    conservation_checks("MD", ens, max_contexts + 1, k_ref, 1e-9)


# ==========================================================================
# (4) Reproducibility and executor invariance
# ==========================================================================


def test_reproducibility_and_executor_invariance():
    """Same ``(seed, runs)`` gives bit-identical Phase-2 outputs.

    ``max_cores`` and ``segment_length`` schedule :meth:`simulate`, not the live
    stepping path, so they cannot move a Phase-2 readout either (SPEC 10.5.4).
    """
    params = dict(num_particles=60, max_contexts=4, infer_bias=True)
    seed = 202
    cues, obs = ctx_scalar_stream()
    grid = np.linspace(-1, 1, 9)

    ens_a = RealTimeCOINEnsemble(runs=5, seed=seed, **params)
    ens_b = RealTimeCOINEnsemble(runs=5, seed=seed, **params)
    ens_serial = RealTimeCOINEnsemble(
        runs=5, seed=seed, max_cores=0, segment_length=1, **params
    )
    ens_parallel = RealTimeCOINEnsemble(
        runs=5, seed=seed, max_cores=2, segment_length=4, **params
    )

    for q, y in zip(cues, obs):
        for ens in (ens_a, ens_b, ens_serial, ens_parallel):
            ens.observe_q(q)
            ens.observe_y(y)
        assert_phase2_bit("reproducible", ens_a, ens_b, grid)
        assert_phase2_bit("executor", ens_serial, ens_parallel, grid)


# ==========================================================================
# (5) + (3) Trivial-alignment regime and permutation invariance
# ==========================================================================


def test_trivial_alignment_and_permutation_scalar():
    """Aligned average == naive member average when every member has K == 1."""
    params = dict(num_particles=100, max_contexts=4, gamma_context=1e-4)
    seed, runs = 314, 4
    cues, obs = trivial_scalar_stream()
    grid = np.linspace(-1.0, 1.0, 11)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    drive(ens, members, cues, obs)

    assert_trivial_frame("trivial scalar", members)

    for method in VECTOR_QUERIES:
        naive = naive_vector_average(members, method)
        naive_reversed = naive_vector_average(members[::-1], method)
        assert_vec_equal("perm-order %s" % method, naive, naive_reversed, 1e-12)
        assert_vec_equal("trivial scalar %s" % method, getattr(ens, method)(), naive, 1e-9)

    assert_vec_equal(
        "trivial scalar stationary",
        ens.stationary_context_probabilities(),
        naive_vector_average(members, "stationary_context_probabilities"),
        1e-9,
    )

    for method in DENSITY_QUERIES:
        assert_map_close(
            "trivial scalar %s" % method,
            getattr(ens, method)(grid),
            naive_density_average(members, method, grid),
            1e-9,
            grid.size,
        )


def test_member_permutation_invariance_scalar():
    """Reversing the stored member list does not move a Phase-2 readout.

    SPEC 10.5.3, checked literally (the MATLAB suite could only check the
    order-independence of its naive oracle). The trivial regime is used because
    there the reference frame itself is order-independent - see this module's
    docstring on ties.
    """
    params = dict(num_particles=100, max_contexts=4, gamma_context=1e-4)
    seed, runs = 314, 4
    cues, obs = trivial_scalar_stream()
    grid = np.linspace(-1.0, 1.0, 11)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    drive(ens, members, cues, obs)
    assert_trivial_frame("permutation scalar", members)

    forward = {m: np.asarray(getattr(ens, m)()) for m in VECTOR_QUERIES}
    forward["stationary_context_probabilities"] = ens.stationary_context_probabilities()
    forward_density = {m: getattr(ens, m)(grid) for m in DENSITY_QUERIES}

    ens._members = list(reversed(ens._members))

    for method, expected in forward.items():
        assert_vec_equal(
            "permutation %s" % method, getattr(ens, method)(), expected, 1e-12
        )
    for method, expected in forward_density.items():
        assert_map_close(
            "permutation %s" % method, getattr(ens, method)(grid), expected,
            1e-12, grid.size,
        )


def test_trivial_alignment_and_density_normalisation_md():
    """Trivial-regime equality plus MD per-context density normalisation.

    The integration window is centred on the reference member's context
    prototype (``Kref == 1``, so reference label 0 is that member's context 0);
    the prototype only positions the window, it is not the oracle answer.
    """
    params = dict(
        state_dim=2, num_particles=100, max_contexts=4, gamma_context=1e-4
    )
    seed, runs = 271, 4
    cues, obs = trivial_md_stream()
    grid = np.column_stack(
        [np.linspace(-0.6, 0.6, 7), np.linspace(-0.6, 0.6, 7)]
    )                                                            # (7, 2)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    drive(ens, members, cues, obs)

    assert_trivial_frame("trivial MD", members)

    for method in VECTOR_QUERIES:
        assert_vec_equal(
            "trivial MD %s" % method,
            getattr(ens, method)(),
            naive_vector_average(members, method),
            1e-9,
        )
    assert_vec_equal(
        "trivial MD stationary",
        ens.stationary_context_probabilities(),
        naive_vector_average(members, "stationary_context_probabilities"),
        1e-9,
    )
    assert_map_close(
        "trivial MD state_given_context_probability",
        ens.state_given_context_probability(grid),
        naive_density_average(members, "state_given_context_probability", grid),
        1e-9,
        grid.shape[0],
    )

    # (6) MD per-context density normalisation.
    ref_index, k_ref, _ = reference_frame(members)
    assert k_ref == 1, "trivial MD Kref == 1"
    centre = np.asarray(
        members[ref_index].context_alignment()["global_contexts"]["state_mean"][0],
        dtype=float,
    )                                                            # (2,)
    _, pooled_cov = ens.state_moments()
    half_std = np.sqrt(np.fmax(np.diag(np.asarray(pooled_cov, dtype=float)), 1e-6))

    densities = ens.state_given_context_probability(grid)
    assert densities, "trivial MD: at least one aligned context density"
    for key in sorted(densities):
        integral, values = integrate_2d(
            lambda pts, k=key: ens.state_given_context_probability(pts)[k],
            centre,
            half_std,
        )
        assert np.all(values >= -1e-12), "trivial MD ctx %s density non-negative" % key
        assert abs(integral - 1.0) < 0.1, (
            "trivial MD ctx %s integrates to %.4f, expected ~1" % (key, integral)
        )


# ==========================================================================
# (6) Per-context density normalisation - scalar
# ==========================================================================


@pytest.mark.parametrize("method", DENSITY_QUERIES)
def test_density_normalisation_scalar(method):
    """Each aligned per-context density integrates to ~1 on a wide fine grid."""
    params = dict(num_particles=80, max_contexts=4, infer_bias=True)
    seed, runs = 909, 4
    cues, obs = ctx_scalar_stream()

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    drive(ens, None, cues, obs)

    # Spacing 1e-3 resolves the ~1e-1 density widths and spans well beyond any
    # context mean.
    grid = np.linspace(-4, 4, 8001)
    densities = getattr(ens, method)(grid)
    assert isinstance(densities, dict), "density-norm scalar %s is a mapping" % method
    assert densities, "density-norm scalar %s has a context" % method
    for key in sorted(densities):
        row = np.asarray(densities[key], dtype=float)
        assert row.shape == (grid.size,), "density-norm scalar %s row" % method
        assert np.all(row >= -1e-12), "density-norm scalar %s non-negative" % method
        integral = float(_trapezoid(row, grid))
        assert abs(integral - 1.0) < 0.05, (
            "density-norm scalar %s ctx %s integrates to %.4f, expected ~1"
            % (method, key, integral)
        )


# ==========================================================================
# (7) Shape / key correctness
# ==========================================================================


def test_shape_and_keys():
    """Vector lengths, stationary length, and density map keys / rows."""
    max_contexts = 5
    params = dict(num_particles=60, max_contexts=max_contexts, infer_bias=True)
    seed, runs = 4242, 4
    cues, obs = ctx_scalar_stream()
    grid = np.linspace(-1.5, 1.5, 13)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    drive(ens, members, cues, obs)
    _, k_ref, _ = reference_frame(members)

    for method in VECTOR_QUERIES:
        assert np.asarray(getattr(ens, method)()).shape == (max_contexts + 1,), (
            "shape %s" % method
        )

    s = np.asarray(ens.stationary_context_probabilities(), dtype=float)
    assert s.shape == (k_ref,), "shape stationary"

    for method in DENSITY_QUERIES:
        densities = getattr(ens, method)(grid)
        assert isinstance(densities, dict), "shape %s is a mapping" % method
        for key in densities:
            assert isinstance(key, (int, np.integer)), "shape %s key type" % method
            assert 0 <= int(key) < k_ref, (
                "shape %s key %s outside 0 .. Kref - 1" % (method, key)
            )
            assert np.asarray(densities[key]).shape == (grid.size,), (
                "shape %s row" % method
            )

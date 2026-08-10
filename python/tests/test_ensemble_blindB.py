"""Behavioural tests for ``RealTimeCOINEnsemble`` - suite B (blind, author B).

Port of ``tests/test_ensemble_blindB.m``. Authored strictly against
``docs/SPEC_ensemble.md`` (the contract), NOT against any implementation, and
deliberately redundant with suite A: it builds its own oracle by driving ``R``
seeded :class:`~realtimecoin.RealTimeCOIN` members with the identical ``(q, y)``
stream and averaging them by hand per SPEC section 5, with its own helper style
throughout.

Spec points resolved here (documented for the reviewer)
-------------------------------------------------------
* **The random stream is pinned, not probed.** The MATLAB suite had to detect
  whether the ensemble used a Threefry or a Philox substream (SPEC 3.1 permits
  either). The Python contract fixes it: member ``k`` is built as
  ``RealTimeCOIN(rng=numpy.random.default_rng(SeedSequence(seed).spawn(runs)[k]))``.
  :func:`build_members` constructs exactly that, so the oracle reproduces the
  ensemble bit for bit and every comparison below is exact rather than
  detected.
* **SPEC 5.4 "non-finite" is read as NaN OR Inf**; :func:`nan_aware_mean`
  therefore maps every non-finite entry to ``nan`` before taking an
  omit-``nan`` mean, so an all-non-finite column yields ``nan`` (matching
  ``COIN.weighted_sum_along_dimension``).
* **Cue-free trials** are ``None``/``nan``, or simply not calling
  ``observe_q``; the ensemble and the oracle honour an identical ``has_q``
  pattern.
* **Scalar shapes drop the singleton dimension** (``float`` not ``1 x 1``,
  ``(K,)`` not ``1 x K``, ``(T,)`` not ``1 x T``), this package's convention.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN, RealTimeCOINEnsemble

ATOL = 1e-9
RTOL = 1e-9


# ========================================================================= #
# Guarantee 1: fan-out and lockstep trial advance.
# ========================================================================= #


def test_fanout_and_lockstep():
    """Every ``observe_y`` advances exactly one trial on every member."""
    params = dict(num_particles=30, max_contexts=4)
    runs = 4
    ens = RealTimeCOINEnsemble(runs=runs, seed=11, **params)
    assert ens.Trial == 0, "initial Trial is 0"

    cues = [1, 2, float("nan"), 1]
    obs = [0.2, -0.1, 0.05, float("nan")]
    for t, (q, y) in enumerate(zip(cues, obs), start=1):
        ens.observe_q(q)
        ens.observe_y(y)
        assert ens.Trial == t, "Trial advances by 1 (t=%d)" % t
        assert [m.Trial for m in ens.members] == [t] * runs

    # runs / seed / weights accessors reflect construction (SPEC 2.2).
    assert ens.runs == runs, "runs accessor"
    assert ens.seed == 11, "seed accessor"
    np.testing.assert_array_equal(ens.weights, np.full(runs, 1 / runs))

    # A missing observation still advances a trial (SPEC section 8).
    ens.observe_y(None)
    assert ens.Trial == len(obs) + 1, "missing feedback advances trial"


# ========================================================================= #
# Guarantee 2: averaging correctness (scalar model). Core oracle.
# ========================================================================= #


def test_averaging_scalar():
    """Mean motor output, pooled moments and mean densities, trial by trial.

    ``max_contexts`` is deliberately small so the novel-context density
    saturates on some members (SPEC section 8): a saturated member must
    contribute finite zeros, not ``nan``, to the averaged density.
    """
    params = dict(num_particles=40, max_contexts=3)
    runs, seed, state_dim = 5, 202, 1
    grid = np.linspace(-1, 1, 21)                                  # (21,)

    cues = [1, 1, 2, float("nan"), 2, 1, 2]
    # has_q[3] is False, exercising the "never call observe_q" cue-free path.
    has_q = [True, True, True, False, True, True, True]
    obs = [0.10, 0.25, float("nan"), 0.15, None, 0.30, 0.20]

    oracle = oracle_traces(runs, seed, params, cues, obs, has_q, state_dim, grid)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for t, y in enumerate(obs):
        if has_q[t]:
            ens.observe_q(cues[t])
        ens.observe_y(y)

        assert ens.Trial == t + 1, "scalar Trial (t=%d)" % t

        u = ens.motor_output()
        assert np.shape(u) == (), "scalar motor_output shape (t=%d)" % t
        assert_num_close("scalar motor_output (t=%d)" % t, u, oracle["motor"][t])

        mu, v = ens.state_moments()
        assert_num_close("scalar state mean (t=%d)" % t, mu, oracle["mu"][t])
        assert_num_close("scalar state var  (t=%d)" % t, v, oracle["v"][t])
        assert v >= 0, "scalar var >= 0 (t=%d)" % t

        for key, method in DENSITY_KEYS.items():
            assert_num_close(
                "scalar %s (t=%d)" % (method, t),
                getattr(ens, method)(grid), oracle[key][t],
            )

        # Densities stay entirely finite even where a member saturated its
        # context budget (zeros, not NaN) -- SPEC section 8.
        assert np.all(np.isfinite(oracle["dnovel"][t])), (
            "scalar novel density finite (t=%d)" % t
        )


# ========================================================================= #
# Guarantee 2 (cont): multi-dimensional averaging (state_dim == 2).
# Exercises the pooled-mixture covariance (SPEC 5.2), which differs from a
# naive mean of per-run covariances whenever the per-run means differ.
# ========================================================================= #


def test_averaging_multidim():
    """The pooled covariance uses the law of total covariance."""
    params = dict(num_particles=40, max_contexts=4, state_dim=2)
    runs, seed, state_dim = 4, 303, 2
    grid = np.column_stack(
        [np.linspace(-0.4, 0.4, 11), np.linspace(-0.3, 0.3, 11)]
    )                                                              # (11, 2)

    cues = [1, 1, 2, 2, 1, 2]
    has_q = [True] * 6
    obs = [np.array([0.10, 0.05]), np.array([0.20, -0.10]),
           np.array([np.nan, 0.10]), np.array([0.15, 0.20]), None,
           np.array([0.05, 0.05])]

    oracle = oracle_traces(runs, seed, params, cues, obs, has_q, state_dim, grid)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for t, y in enumerate(obs):
        if has_q[t]:
            ens.observe_q(cues[t])
        ens.observe_y(y)

        u = ens.motor_output()
        assert np.shape(u) == (state_dim,), "MD motor_output shape (t=%d)" % t
        assert_num_close("MD motor_output (t=%d)" % t, u, oracle["motor"][t])

        mu, v = ens.state_moments()
        assert np.shape(mu) == (state_dim,), "MD mu shape (t=%d)" % t
        assert np.shape(v) == (state_dim, state_dim), "MD v shape (t=%d)" % t
        assert_num_close("MD state mean (t=%d)" % t, mu, oracle["mu"][t])
        assert_num_close("MD pooled covariance (t=%d)" % t, v, oracle["v"][t])
        assert_num_close("MD covariance symmetric (t=%d)" % t, v, np.asarray(v).T,
                         atol=1e-12, rtol=0.0)

        for key, method in DENSITY_KEYS.items():
            assert_num_close(
                "MD %s (t=%d)" % (method, t),
                getattr(ens, method)(grid), oracle[key][t],
            )


# ========================================================================= #
# Guarantee 3: reproducibility -- same (seed, runs) + same stream.
# ========================================================================= #


def test_reproducibility():
    """Two identically-seeded ensembles agree on every query, every trial."""
    params = dict(num_particles=32, max_contexts=4)
    runs, seed = 4, 77
    cues = [1, 2, 1, float("nan"), 2, 1]
    obs = [0.1, -0.2, float("nan"), 0.05, None, 0.3]

    a = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    b = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for t, (q, y) in enumerate(zip(cues, obs)):
        for ens in (a, b):
            ens.observe_q(q)
            ens.observe_y(y)
        assert_all_queries_bit_equal("reproducibility (t=%d)" % t, a, b, 1,
                                     np.linspace(-1, 1, 15))


# ========================================================================= #
# Guarantee 4: executor invariance (stepping) -- serial vs parallel and
# different segment_length must be bit-identical.
# ========================================================================= #


def test_executor_invariance_stepping():
    """``max_cores`` and ``segment_length`` do not change stepping results."""
    params = dict(num_particles=32, max_contexts=4)
    runs, seed = 4, 91
    cues = [1, 2, 2, 1, float("nan"), 2]
    obs = [0.15, -0.05, float("nan"), 0.2, 0.1, None]

    serial = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=0,
                                  segment_length=1, **params)
    parallel = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=2,
                                    segment_length=3, **params)
    for t, (q, y) in enumerate(zip(cues, obs)):
        for ens in (serial, parallel):
            ens.observe_q(q)
            ens.observe_y(y)
        assert_all_queries_bit_equal("executor invariance step (t=%d)" % t,
                                     serial, parallel, 1, np.linspace(-1, 1, 15))


# ========================================================================= #
# Guarantee 4 (cont): executor invariance for simulate().
# ========================================================================= #


def test_executor_invariance_simulate():
    """Serial and process-pool ``simulate`` return identical traces."""
    params = dict(num_particles=32, max_contexts=4)
    runs, seed, length = 4, 91, 6
    cue_seq = np.array([1, 2, 2, 1, np.nan, 2])
    obs_seq = np.array([0.15, -0.05, np.nan, 0.2, 0.1, -0.1])

    a = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=0,
                             segment_length=1, **params)
    b = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=2,
                             segment_length=4, **params)
    tra = a.simulate(cue_seq, obs_seq)
    # Guard against a vacuous pass: if the pool cannot start, simulate() warns
    # and replays serially, which would compare serial against serial.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trb = b.simulate(cue_seq, obs_seq)
    assert not [w for w in caught if "process pool" in str(w.message)], (
        "the process pool did not start; the comparison would be vacuous"
    )

    assert bit_equal(tra.motor_output, trb.motor_output), (
        "simulate motor_output executor-invariant"
    )
    assert bit_equal(tra.state_mean, trb.state_mean), (
        "simulate state_mean executor-invariant"
    )
    assert bit_equal(tra.state_var, trb.state_var), (
        "simulate state_var executor-invariant"
    )
    assert np.array_equal(tra.Trial, np.arange(1, length + 1))

    # Output shape contract (SPEC 6), scalar model.
    assert tra.motor_output.shape == (length,)
    assert tra.state_mean.shape == (length,)
    assert tra.state_var.shape == (length,)


# ========================================================================= #
# Guarantee 5: independence / seed sensitivity.
# ========================================================================= #


def test_seed_sensitivity_and_member_independence():
    """Distinct seeds and distinct members follow distinct trajectories."""
    params = dict(num_particles=40, max_contexts=4)
    runs = 5
    cues = [1, 2, 1, 2]
    obs = [0.2, -0.1, 0.15, 0.05]

    e0 = RealTimeCOINEnsemble(runs=runs, seed=0, **params)
    e1 = RealTimeCOINEnsemble(runs=runs, seed=123, **params)
    for q, y in zip(cues, obs):
        for ens in (e0, e1):
            ens.observe_q(q)
            ens.observe_y(y)
    assert abs(e0.motor_output() - e1.motor_output()) > 1e-6, (
        "different seeds give different motor output"
    )

    # Members within one ensemble use different spawned children, so they
    # diverge almost surely.
    members = build_members(runs, 55, params)
    for q, y in zip(cues, obs):
        for member in members:
            member.observe_q(q)
            member.observe_y(y)
    assert abs(members[0].motor_output() - members[1].motor_output()) > 1e-9, (
        "distinct members diverge"
    )


# ========================================================================= #
# Guarantee 6: simulate() equals the stepping loop; repeatable; and does not
# disturb the live stepping state.
# ========================================================================= #


def test_simulate_equivalence():
    """Batch replay, stepping loop, and repeat calls all agree (SPEC 6)."""
    params = dict(num_particles=32, max_contexts=4)
    runs, seed, length = 4, 404, 7
    cue_seq = np.array([1, 1, 2, np.nan, 2, 1, 2])
    obs_seq = np.array([0.10, 0.25, np.nan, 0.15, 0.0, 0.30, 0.20])

    # Reference: trial-by-trial stepping, recording queries after each trial.
    step = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    rec_motor = np.zeros(length)
    rec_mean = np.zeros(length)
    rec_var = np.zeros(length)
    for t in range(length):
        step.observe_q(cue_seq[t])
        step.observe_y(obs_seq[t])
        rec_motor[t] = step.motor_output()
        mu, v = step.state_moments()
        rec_mean[t] = mu
        rec_var[t] = v

    # simulate() on a fresh, identically-seeded ensemble must reproduce them.
    sim = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    tr = sim.simulate(cue_seq, obs_seq)
    assert bit_equal(tr.motor_output, rec_motor), "simulate motor == stepping"
    assert bit_equal(tr.state_mean, rec_mean), "simulate state_mean == stepping"
    assert bit_equal(tr.state_var, rec_var), "simulate state_var == stepping"
    assert np.array_equal(tr.Trial, np.arange(1, length + 1))

    # simulate() does not disturb the live stepping state of sim (SPEC 6).
    assert sim.Trial == 0, "simulate leaves live Trial untouched"

    # simulate() is repeatable.
    tr2 = sim.simulate(cue_seq, obs_seq)
    assert bit_equal(tr2.motor_output, tr.motor_output)
    assert bit_equal(tr2.state_mean, tr.state_mean)
    assert bit_equal(tr2.state_var, tr.state_var)


def test_simulate_rejects_inconsistent_sequences():
    """Shape and length errors are reported, not silently reinterpreted."""
    ens = RealTimeCOINEnsemble(runs=2, seed=1, num_particles=15)
    with pytest.raises(ValueError, match="lengthMismatch"):
        ens.simulate(np.array([1, 2, 1]), np.array([0.1, 0.2]))
    md = RealTimeCOINEnsemble(runs=2, seed=1, num_particles=15, state_dim=2)
    with pytest.raises(ValueError, match="ySize"):
        md.simulate(None, np.array([[0.1, 0.2, 0.3]]))


# ========================================================================= #
# Guarantee 7: runs == 1 reduces exactly to the single seeded member.
# ========================================================================= #


def test_runs_one_reduction():
    """A one-run ensemble equals its member EXACTLY, query for query."""
    params = dict(num_particles=40, max_contexts=4)
    seed = 606
    cues = [1, 2, float("nan"), 1, 2]
    obs = [0.2, -0.1, 0.05, float("nan"), 0.3]
    grid = np.linspace(-1, 1, 17)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    member = build_members(1, seed, params)[0]

    for t, (q, y) in enumerate(zip(cues, obs)):
        ens.observe_q(q)     # nan cue forwarded verbatim as a cue-free trial
        ens.observe_y(y)
        member.observe_q(q)
        member.observe_y(y)

        assert bit_equal(ens.motor_output(), member.motor_output()), (
            "runs==1 motor (t=%d)" % t
        )
        e_mu, e_v = ens.state_moments()
        s_mu, s_v = member.state_moments()
        assert bit_equal(e_mu, s_mu), "runs==1 mu (t=%d)" % t
        assert bit_equal(e_v, s_v), "runs==1 var (t=%d)" % t
        assert bit_equal(ens.state_probability(grid),
                         member.state_probability(grid)), (
            "runs==1 state_probability (t=%d)" % t
        )
        assert bit_equal(ens.novel_state_feedback_probability(grid),
                         member.novel_state_feedback_probability(grid)), (
            "runs==1 novel_state_feedback (t=%d)" % t
        )


# ========================================================================= #
# Guarantee 8: snapshot / load_snapshot round-trip (SPEC section 7).
# ========================================================================= #


def test_snapshot_roundtrip():
    """A restored model reproduces every query and the subsequent run."""
    params = dict(num_particles=40, max_contexts=4, infer_bias=True)

    b = RealTimeCOIN(rng=np.random.default_rng(7), **params)
    for q, y in zip([1, 2, 1, 2], [0.2, -0.1, 0.15, 0.05]):
        b.observe_q(q)
        b.observe_y(y)

    s = b.snapshot()
    a = RealTimeCOIN(rng=np.random.default_rng(11), **params)
    a.load_snapshot(s)

    grid = np.linspace(-1, 1, 21)
    assert a.Trial == b.Trial, "snapshot Trial"
    assert bit_equal(a.motor_output(), b.motor_output()), "snapshot motor"
    a_mu, a_v = a.state_moments()
    b_mu, b_v = b.state_moments()
    assert bit_equal(a_mu, b_mu), "snapshot mu"
    assert bit_equal(a_v, b_v), "snapshot var"
    assert bit_equal(a.state_probability(grid), b.state_probability(grid))
    assert bit_equal(a.state_feedback_probability(grid),
                     b.state_feedback_probability(grid))

    # The stream position travels with the snapshot, so a matched future step
    # needs no external RNG juggling (MATLAB had to rewind the global stream).
    for model in (b, a):
        model.observe_q(1)
        model.observe_y(0.12)
    assert bit_equal(a.motor_output(), b.motor_output()), (
        "snapshot post-step motor"
    )
    assert bit_equal(a.state_probability(grid), b.state_probability(grid)), (
        "snapshot post-step density"
    )


def test_ensemble_snapshot_roundtrip():
    """The ensemble snapshot restores members, trial counter and parameters."""
    params = dict(num_particles=30, max_contexts=4)
    runs, seed = 3, 66
    cues = [1, 2, 1, 2]
    obs = [0.2, -0.1, 0.15, 0.05]

    a = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for q, y in zip(cues, obs):
        a.observe_q(q)
        a.observe_y(y)

    b = RealTimeCOINEnsemble(runs=2, seed=1, num_particles=8)
    b.load_snapshot(a.snapshot())
    assert (b.runs, b.seed, b.Trial) == (a.runs, a.seed, a.Trial)
    assert bit_equal(b.motor_output(), a.motor_output())

    for q, y in zip([2, 1], [0.05, -0.2]):
        for ens in (a, b):
            ens.observe_q(q)
            ens.observe_y(y)
    assert bit_equal(b.motor_output(), a.motor_output())
    a_mu, a_v = a.state_moments()
    b_mu, b_v = b.state_moments()
    assert bit_equal(b_mu, a_mu) and bit_equal(b_v, a_v)


# ========================================================================= #
# Guarantee 9: the ensemble never touches the caller's global RNG (SPEC 3.5).
# ========================================================================= #


def test_global_stream_untouched():
    """The legacy global ``numpy.random`` sequence continues unbroken.

    Stronger than comparing states: the draws taken after the ensemble work
    must be the very draws that would have come next.
    """
    params = dict(num_particles=30, max_contexts=4)

    np.random.seed(12345)
    np.random.random(4)
    expected = np.random.random(4)

    np.random.seed(12345)
    np.random.random(4)
    ens = RealTimeCOINEnsemble(runs=3, seed=5, **params)
    ens.observe_q(1)
    ens.observe_y(0.1)
    ens.simulate(np.array([1, 2, 1]), np.array([0.1, -0.1, 0.05]))
    observed = np.random.random(4)

    np.testing.assert_array_equal(observed, expected)


def test_nan_aware_averaging_rule():
    """A non-finite run is ABSENT from the mean, not poisonous (SPEC 5.4).

    The member streams never produce a non-finite query in practice, so the
    rule is exercised by overriding the members' readouts directly: an entry
    finite in at least one run must be the mean over those runs, and an entry
    non-finite in every run must be ``nan``.
    """
    ens = RealTimeCOINEnsemble(runs=3, seed=2, num_particles=15)
    ens.observe_q(1)
    ens.observe_y(0.1)

    values = [float("nan"), 0.2, float("inf")]
    for member, value in zip(ens.members, values):
        member.motor_output = (lambda v: (lambda: v))(value)
    assert ens.motor_output() == pytest.approx(0.2), (
        "one finite run must carry the average"
    )

    for member in ens.members:
        member.motor_output = lambda: float("inf")
    assert np.isnan(ens.motor_output()), "all-non-finite gives NaN"

    # Densities follow the same rule entry by entry.
    grid = np.linspace(-1, 1, 5)
    rows = [np.array([np.nan, 0.0, 1.0, np.inf, np.nan]),
            np.array([2.0, 4.0, 3.0, np.inf, np.nan]),
            np.array([np.inf, 2.0, 2.0, np.inf, np.nan])]
    for member, row in zip(ens.members, rows):
        member.state_probability = (lambda r: (lambda _values: r))(row)
    averaged = ens.state_probability(grid)
    np.testing.assert_allclose(averaged[:3], [2.0, 2.0, 2.0])
    assert np.all(np.isnan(averaged[3:]))


def test_pooled_moments_skip_non_finite_runs():
    """A run with a non-finite moment does not contribute at all (SPEC 5.4)."""
    ens = RealTimeCOINEnsemble(runs=3, seed=2, num_particles=15)
    ens.observe_q(1)
    ens.observe_y(0.1)

    moments = [(float("nan"), 1.0), (0.5, 0.25), (1.5, 0.25)]
    for member, pair in zip(ens.members, moments):
        member.state_moments = (lambda p: (lambda: p))(pair)
    mu, v = ens.state_moments()
    # Pooled over runs 2 and 3 only: mu = 1, v = mean(0.25 + m^2) - 1 = 0.5.
    assert mu == pytest.approx(1.0)
    assert v == pytest.approx(0.5)

    for member in ens.members:
        member.state_moments = lambda: (float("nan"), float("nan"))
    mu, v = ens.state_moments()
    assert np.isnan(mu) and np.isnan(v)


def test_desynchronised_members_are_refused():
    """Stepping a member behind the ensemble's back is detected, not averaged.

    Every query is covered, not just the state machine: averaging members that
    sit on DIFFERENT trials would silently return a quantity that corresponds
    to no trial at all (SPEC 2.2).
    """
    from realtimecoin.ensemble import EnsembleLockstepError

    grid = np.linspace(-1, 1, 5)
    queries = [
        lambda e: e.observe_y(0.3),
        lambda e: e.observe_q(1),
        lambda e: e.motor_output(),
        lambda e: e.state_moments(),
        lambda e: e.state_probability(grid),
        lambda e: e.state_feedback_probability(grid),
        lambda e: e.novel_state_probability(grid),
        lambda e: e.novel_state_feedback_probability(grid),
    ]
    for query in queries:
        ens = RealTimeCOINEnsemble(runs=3, seed=2, num_particles=15)
        ens.observe_q(1)
        ens.observe_y(0.1)
        ens.members[1].observe_y(0.2)     # one member now a trial ahead
        with pytest.raises(EnsembleLockstepError):
            query(ens)


def test_moment_postconditions_hold_for_every_run_count():
    """The variance floor and covariance symmetry are not R-dependent.

    SPEC 5.2 states both unconditionally. ``runs == 1`` takes a different code
    path (the exact pass-through required by SPEC 5.5), so it needs its own
    check - and it is the run count most likely to be used.
    """
    for runs in (1, 2, 4):
        ens = RealTimeCOINEnsemble(runs=runs, seed=9, num_particles=20)
        ens.observe_q(1)
        ens.observe_y(0.1)

        # A member reporting a tiny negative variance (round-off territory)
        # must still leave the pooled variance floored at 0.
        for member in ens.members:
            member.state_moments = lambda: (0.5, -1e-18)
        _, v = ens.state_moments()
        assert v >= 0.0, "runs=%d: variance not floored" % runs

        md = RealTimeCOINEnsemble(runs=runs, seed=9, num_particles=20,
                                  state_dim=2)
        md.observe_q(1)
        md.observe_y(np.array([0.1, 0.0]))
        asymmetric = np.array([[1.0, 0.25], [0.75, 1.0]])
        for member in md.members:
            member.state_moments = (
                lambda: (np.array([0.5, 0.25]), asymmetric)
            )
        _, cov = md.state_moments()
        np.testing.assert_allclose(cov, cov.T, rtol=0, atol=0)


def test_members_do_not_share_a_generator():
    """Each member owns a distinct generator, so ordering cannot matter."""
    ens = RealTimeCOINEnsemble(runs=4, seed=3, num_particles=15)
    identities = {id(m.rng) for m in ens.members}
    assert len(identities) == ens.runs
    states = [m.rng.bit_generator.state["state"]["state"]
              for m in ens.members]
    assert len(set(states)) == ens.runs


# ========================================================================= #
# Phase 2 (unit E1): the context-indexed queries exist but are not yet live.
# ========================================================================= #


@pytest.mark.parametrize(
    "method",
    ["responsibilities_vector", "predicted_context_probabilities_vector",
     "sampled_context_count", "stationary_context_probabilities"],
)
def test_phase2_vector_queries_are_aligned(method):
    """Aligned context vectors are member-order invariant and sum to 1."""
    params = dict(num_particles=20, max_contexts=3)
    ens = RealTimeCOINEnsemble(runs=3, seed=31, **params)
    for q, y in zip([1, 1, 2, 2], [0.1, 0.2, -0.1, 0.3]):
        ens.observe_q(q)
        ens.observe_y(y)
    try:
        value = getattr(ens, method)()
    except NotImplementedError:
        pytest.skip("phase 2: unit E1")
    # None is admissible only for the stationary distribution with no contexts,
    # which four observed trials rule out; anything else is a silent failure.
    assert value is not None, "%s returned None with contexts instantiated" % method
    value = np.asarray(value, dtype=float)
    assert np.all(value >= -1e-12)
    assert abs(float(value.sum()) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "method",
    ["responsibilities_vector", "predicted_context_probabilities_vector",
     "sampled_context_count"],
)
def test_phase2_vectors_are_member_order_invariant(method):
    """Equal weights make the aligned average independent of member order.

    SPEC 10.5.3. Reversing the stored member list must not move the result.
    """
    params = dict(num_particles=20, max_contexts=3)
    ens = RealTimeCOINEnsemble(runs=3, seed=41, **params)
    for q, y in zip([1, 1, 2, 2], [0.1, 0.2, -0.1, 0.3]):
        ens.observe_q(q)
        ens.observe_y(y)
    try:
        forward = np.asarray(getattr(ens, method)(), dtype=float)
    except NotImplementedError:
        pytest.skip("phase 2: unit E1")

    ens._members = list(reversed(ens._members))
    reversed_order = np.asarray(getattr(ens, method)(), dtype=float)
    np.testing.assert_allclose(reversed_order, forward, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "method",
    ["state_given_context_probability",
     "state_feedback_given_context_probability"],
)
def test_phase2_density_queries_are_aligned(method):
    """Aligned per-context densities are non-negative rows on the grid."""
    params = dict(num_particles=20, max_contexts=3)
    ens = RealTimeCOINEnsemble(runs=3, seed=31, **params)
    grid = np.linspace(-1, 1, 11)
    for q, y in zip([1, 1, 2, 2], [0.1, 0.2, -0.1, 0.3]):
        ens.observe_q(q)
        ens.observe_y(y)
    try:
        value = getattr(ens, method)(grid)
    except NotImplementedError:
        pytest.skip("phase 2: unit E1")
    for density in value.values():
        density = np.asarray(density, dtype=float)
        assert density.shape == (grid.size,)
        assert np.all(density >= -1e-12)


# ========================================================================= #
# ---------------------------- oracle helpers ----------------------------- #
# ========================================================================= #


#: Trace key -> ensemble/member density method name.
DENSITY_KEYS = {
    "dstate": "state_probability",
    "dsf": "state_feedback_probability",
    "dnovel": "novel_state_probability",
    "dnovelf": "novel_state_feedback_probability",
}


def build_members(runs, seed, member_params):
    """Construct ``R`` members under the pinned per-member streams (SPEC 3.1).

    Member ``k`` is built with child ``k`` of ``SeedSequence(seed).spawn(runs)``
    as its generator, so its construction randomness belongs to its own stream.

    Parameters
    ----------
    runs : int
        Number of members ``R``.
    seed : int
        Ensemble base seed.
    member_params : dict
        Member constructor keywords.

    Returns
    -------
    list of RealTimeCOIN
        The reference members.
    """
    children = np.random.SeedSequence(seed).spawn(runs)
    return [
        RealTimeCOIN(rng=np.random.default_rng(child), **member_params)
        for child in children
    ]


def oracle_traces(runs, seed, member_params, cues, obs, has_q, state_dim, grid):
    """Per-trial averaged queries built independently from ``R`` members.

    Reproduces the random-stream contract (SPEC section 3) exactly and averages
    per SPEC section 5: mean motor output, pooled-mixture moments, mean
    density.

    Parameters
    ----------
    runs : int
        Number of members ``R``.
    seed : int
        Ensemble base seed.
    member_params : dict
        Member constructor keywords.
    cues : list
        Per-trial raw cue values.
    obs : list
        Per-trial feedback (``None`` / ``nan`` for missing).
    has_q : list of bool
        Whether ``observe_q`` is called at all on that trial.
    state_dim : int
        Latent dimension ``N``.
    grid : numpy.ndarray
        Density query grid.

    Returns
    -------
    dict
        ``{"motor": [...], "mu": [...], "v": [...], "dstate": [...],
        "dsf": [...], "dnovel": [...], "dnovelf": [...]}``, each a per-trial
        list.
    """
    members = build_members(runs, seed, member_params)
    traces = {key: [] for key in ("motor", "mu", "v", *DENSITY_KEYS)}

    for t, y in enumerate(obs):
        for member in members:
            if has_q[t]:
                member.observe_q(cues[t])
            member.observe_y(y)

        traces["motor"].append(avg_motor(members, state_dim))
        mu, v = avg_moments(members, state_dim)
        traces["mu"].append(mu)
        traces["v"].append(v)
        for key, method in DENSITY_KEYS.items():
            traces[key].append(avg_density(members, method, grid))
    return traces


def avg_motor(members, state_dim):
    """Equal-weight NaN-aware mean of the per-member ``motor_output``."""
    stacked = np.array(
        [np.reshape(np.asarray(m.motor_output(), dtype=float), state_dim)
         for m in members]
    )                                                              # (R, N)
    out = nan_aware_mean(stacked, axis=0)                          # (N,)
    return float(out[0]) if state_dim == 1 else out


def avg_moments(members, state_dim):
    """Equal-weight pooled-mixture moments (SPEC 5.2)."""
    means = np.zeros((len(members), state_dim))                    # (R, N)
    seconds = np.zeros((len(members), state_dim, state_dim))       # (R, N, N)
    for k, member in enumerate(members):
        mu_k, v_k = member.state_moments()
        mu_k = np.reshape(np.asarray(mu_k, dtype=float), state_dim)
        means[k] = mu_k
        seconds[k] = np.reshape(v_k, (state_dim, state_dim)) + np.outer(mu_k, mu_k)

    mu = nan_aware_mean(means, axis=0)                             # (N,)
    second = nan_aware_mean(seconds, axis=0)                       # (N, N)
    v = second - np.outer(mu, mu)
    if state_dim == 1:
        return float(mu[0]), float(max(v[0, 0], 0.0))
    return mu, (v + v.T) / 2.0


def avg_density(members, method, values):
    """Equal-weight NaN-aware mean of the per-member densities (SPEC 5.3)."""
    stacked = np.array(
        [np.asarray(getattr(m, method)(values), dtype=float).reshape(-1)
         for m in members]
    )                                                              # (R, K)
    return nan_aware_mean(stacked, axis=0)


def nan_aware_mean(x, axis):
    """Omit-``nan`` mean with all-non-finite giving ``nan`` (SPEC 5.4).

    Non-finite entries (``nan`` and ``+-inf``) are mapped to ``nan`` first, so
    an infinite run is treated as absent rather than poisoning the mean.

    Parameters
    ----------
    x : numpy.ndarray
        Values, with the runs along ``axis``.
    axis : int
        Axis to reduce.

    Returns
    -------
    numpy.ndarray
        The mean, with ``nan`` wherever every run was non-finite.
    """
    masked = np.where(np.isfinite(x), x, np.nan)
    with warnings.catch_warnings():
        # An all-NaN slice legitimately yields NaN here (SPEC 5.4).
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(masked, axis=axis)


# ========================================================================= #
# ---------------------------- assert helpers ----------------------------- #
# ========================================================================= #


def bit_equal(actual, expected):
    """Bit-identical equality, with ``nan`` equal to ``nan``."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def assert_num_close(name, actual, expected, atol=ATOL, rtol=RTOL):
    """Elementwise closeness with a matching ``nan`` pattern."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    assert a.shape == b.shape, "FAILED: %s shape %s != %s" % (name, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), (
        "FAILED: %s NaN pattern mismatch" % (name,)
    )
    finite = ~np.isnan(a)
    diff = np.abs(a[finite] - b[finite])
    tol = atol + rtol * np.maximum(np.abs(a[finite]), np.abs(b[finite]))
    assert np.all(diff <= tol), (
        "FAILED: %s max abs err %.3g exceeds tolerance"
        % (name, diff.max() if diff.size else 0.0)
    )


def assert_all_queries_bit_equal(name, e1, e2, state_dim, grid):
    """Bit-identical outputs across every Phase-1 query method."""
    assert e1.Trial == e2.Trial, name + " Trial"
    assert bit_equal(e1.motor_output(), e2.motor_output()), name + " motor_output"
    m1, v1 = e1.state_moments()
    m2, v2 = e2.state_moments()
    assert bit_equal(m1, m2), name + " state mean"
    assert bit_equal(v1, v2), name + " state var"
    g = grid if state_dim == 1 else np.column_stack([grid] * state_dim)
    for method in DENSITY_KEYS.values():
        assert bit_equal(getattr(e1, method)(g), getattr(e2, method)(g)), (
            "%s %s" % (name, method)
        )

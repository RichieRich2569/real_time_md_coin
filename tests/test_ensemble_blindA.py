"""Blind behavioural tests for ``RealTimeCOINEnsemble`` - suite A.

Port of ``tests/test_ensemble_blindA.m``. Authored strictly against
``docs/SPEC_ensemble.md`` (the contract), NOT against any implementation. The
independent oracle is built here by constructing ``R`` ordinary
:class:`~realtimecoin.RealTimeCOIN` members, driving them with the identical
``(q, y)`` stream, and averaging per SPEC section 5 - while replicating the
random-stream contract so the reference lines up bit for bit with the ensemble.

Coverage maps to SPEC section 9 "testable guarantees":

1. fan-out and lockstep trial counting
2. averaging correctness (motor_output, pooled state_moments, densities)
3. reproducibility (same seed/runs/stream gives bit-identical output)
4. executor invariance (``max_cores`` / ``segment_length`` independence)
5. independence / seed sensitivity
6. ``simulate`` equals the stepping loop; repeatable and non-disturbing
7. ``runs == 1`` reduction to the single member
8. ``snapshot`` / ``load_snapshot`` round-trip
9. the global RNG is left unchanged; edge cases (missing / cue-free trials)

Two deliberate departures from the MATLAB suite
-----------------------------------------------
* **No generator probing.** The MATLAB tests had to auto-detect whether the
  ensemble used a Threefry or a Philox substream, because SPEC 3.1 permits
  either. The Python contract pins the stream exactly - member ``k`` is
  ``RealTimeCOIN(rng=numpy.random.default_rng(SeedSequence(seed).spawn(runs)[k]))``
  - so :func:`oracle_members` builds the reference directly, and
  :func:`test_rng_contract_pins_member_streams` asserts the pinning itself.
  That is a strictly stronger check than detection.
* **Global-stream restoration becomes global-stream non-use.** MATLAB had to
  save and restore the global stream around every member step; nothing here
  touches the legacy global ``numpy.random`` state at all, which
  :func:`test_global_rng_unchanged` checks both empirically and by source
  inspection.

Scalar-model shapes follow this package's convention throughout: a scalar
``motor_output`` is a ``float`` (not ``1 x 1``), a density is ``(K,)`` (not
``1 x K``) and a ``simulate`` trace is ``(T,)`` (not ``1 x T``).
"""

from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN, RealTimeCOINEnsemble

#: RNG-matched averaging: allow only summation-order slack.
TOL = 1e-9


# ==========================================================================
# Oracle helpers - the RNG contract plus SPEC section 5 averaging
# ==========================================================================


def scalar_params():
    """Member parameters for the scalar-model checks."""
    return dict(num_particles=40, max_contexts=3, infer_bias=True)


def md_params():
    """Member parameters for the multi-dimensional checks."""
    return dict(state_dim=2, num_particles=40, max_contexts=3)


def scalar_stream():
    """Cue / feedback stream for the scalar-model checks."""
    return [1, 1, 2, 2, 1, 2], [0.10, 0.25, -0.10, 0.30, 0.05, -0.20]


def md_stream():
    """Cue / feedback stream for the multi-dimensional checks."""
    cues = [1, 1, 2, 2, 1, 2]
    obs = np.array(
        [[0.10, 0.25, -0.10, 0.30, 0.05, -0.20],
         [-0.05, 0.15, 0.20, -0.10, 0.00, 0.25]]
    )                                                              # (2, 6)
    return cues, obs


def oracle_members(seed, runs, params):
    """Build ``R`` plain members under the pinned per-member streams.

    Member ``k`` is CONSTRUCTED with child ``k`` of
    ``SeedSequence(seed).spawn(runs)``, so its construction randomness
    (``reset_particles``) belongs to its own stream, exactly as the ensemble's
    members do.

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
    """Drive every member with the identical ``(q, y)``."""
    for member in members:
        member.observe_q(q)     # draws no randomness (SPEC 4.1)
        member.observe_y(y)


def nanmean_finite(values):
    """Mean over the finite entries; ``nan`` when every entry is non-finite.

    Implements the SPEC 5.4 rule directly (no masking arithmetic), so it is an
    independent statement of the requirement.

    Parameters
    ----------
    values : sequence of float
        One value per run.

    Returns
    -------
    float
        The mean over the finite values, or ``nan``.
    """
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def oracle_motor(members):
    """Equal-weight, NaN-aware mean of the per-member ``motor_output``."""
    per_run = [np.atleast_1d(np.asarray(m.motor_output(), dtype=float))
               for m in members]
    out = np.array([nanmean_finite([r[i] for r in per_run])
                    for i in range(per_run[0].size)])
    return float(out[0]) if out.size == 1 else out


def oracle_moments(members, state_dim):
    """Pooled equal-weight mixture moments (SPEC 5.2), NaN-aware."""
    means = []
    seconds = []
    for member in members:
        mu, v = member.state_moments()
        mu = np.atleast_1d(np.asarray(mu, dtype=float))            # (N,)
        means.append(mu)
        seconds.append(np.reshape(v, (state_dim, state_dim)) + np.outer(mu, mu))

    mu_ref = np.array(
        [nanmean_finite([m[i] for m in means]) for i in range(state_dim)]
    )                                                              # (N,)
    second_ref = np.array(
        [[nanmean_finite([s[i, j] for s in seconds]) for j in range(state_dim)]
         for i in range(state_dim)]
    )                                                              # (N, N)
    v_ref = second_ref - np.outer(mu_ref, mu_ref)
    if state_dim == 1:
        return float(mu_ref[0]), float(max(v_ref[0, 0], 0.0))
    return mu_ref, (v_ref + v_ref.T) / 2.0


def oracle_density(members, method, values):
    """Equal-weight, NaN-aware mean of the per-member densities (SPEC 5.3)."""
    per_run = [np.asarray(getattr(m, method)(values), dtype=float)
               for m in members]
    return np.array(
        [nanmean_finite([r[j] for r in per_run]) for j in range(per_run[0].size)]
    )


# ==========================================================================
# Assertion helpers
# ==========================================================================


def assert_near(name, actual, expected, tol=TOL):
    """Elementwise closeness with matching shapes; ``nan`` must align exactly."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    assert a.shape == b.shape, "FAILED %s: shape %s != %s" % (name, a.shape, b.shape)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    assert np.array_equal(nan_a, nan_b), "FAILED %s: NaN pattern mismatch" % (name,)
    diff = np.abs(a[~nan_a] - b[~nan_b])
    if diff.size:
        assert np.all(np.isfinite(diff)) and diff.max() <= tol, (
            "FAILED %s (max diff %g, tol %g)" % (name, diff.max(), tol)
        )


def assert_bit(name, actual, expected):
    """Bit-identical equality (``nan`` counts as equal to ``nan``)."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    assert np.array_equal(a, b, equal_nan=True), (
        "FAILED bit-identical: %s" % (name,)
    )


DENSITY_METHODS = (
    "state_probability",
    "state_feedback_probability",
    "novel_state_probability",
    "novel_state_feedback_probability",
)


# ==========================================================================
# (0) The pinned RNG contract itself
# ==========================================================================


def test_rng_contract_pins_member_streams():
    """Ensemble member ``k`` equals a plain model on spawned child ``k``.

    This replaces the MATLAB suite's Threefry/Philox probing: the Python
    contract names the stream, so it can simply be asserted.
    """
    params = scalar_params()
    runs, seed = 3, 4242
    cues, obs = scalar_stream()

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)
    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)

    for k, member in enumerate(members):
        assert_bit("member %d motor" % k, ens.members[k].motor_output(),
                   member.motor_output())


# ==========================================================================
# (1) Fan-out and lockstep
# ==========================================================================


def test_fanout_and_lockstep():
    """Identical ``(q, y)`` reaches every member; one trial per ``observe_y``."""
    params = scalar_params()
    runs, seed = 4, 11
    cues, obs = scalar_stream()

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)

    assert ens.Trial == 0, "Trial starts at 0"
    for t, (q, y) in enumerate(zip(cues, obs), start=1):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)
        assert ens.Trial == t, "Trial==%d" % t
        assert all(m.Trial == t for m in ens.members), "members in lockstep"
        assert_near("fanout motor t=%d" % t, ens.motor_output(),
                    oracle_motor(members))


# ==========================================================================
# (2) Averaging correctness - scalar model
# ==========================================================================


def test_averaging_scalar():
    """Mean motor output, pooled moments and mean densities (SPEC 5.1-5.4)."""
    params = scalar_params()
    runs, seed = 4, 23
    cues, obs = scalar_stream()
    grid = np.linspace(-1.2, 1.2, 9)                               # (9,)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)

    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)

        assert_near("scalar motor_output", ens.motor_output(),
                    oracle_motor(members))

        mu, v = ens.state_moments()
        mu_ref, v_ref = oracle_moments(members, 1)
        assert_near("scalar state_moments mu", mu, mu_ref)
        assert_near("scalar state_moments v", v, v_ref)
        assert v >= 0, "scalar variance nonnegative"

        for method in DENSITY_METHODS:
            d = getattr(ens, method)(grid)
            assert np.shape(d) == (grid.size,), "scalar %s shape" % method
            assert_near("scalar " + method, d,
                        oracle_density(members, method, grid))


# ==========================================================================
# (2) Averaging correctness - multi-dimensional model (state_dim == 2)
# ==========================================================================


def test_averaging_multidim():
    """The pooled covariance is the law of total covariance, not a mean."""
    params = md_params()
    runs, seed = 4, 37
    cues, obs = md_stream()
    grid = np.column_stack(
        [np.linspace(-1, 1, 6), np.linspace(-0.5, 0.5, 6)]
    )                                                              # (6, 2)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)

    for t in range(obs.shape[1]):
        q, y = cues[t], obs[:, t]
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)

        u = ens.motor_output()
        assert np.shape(u) == (2,), "MD motor_output shape"
        assert_near("MD motor_output", u, oracle_motor(members))

        mu, v = ens.state_moments()
        mu_ref, v_ref = oracle_moments(members, 2)
        assert np.shape(mu) == (2,), "MD mu shape"
        assert np.shape(v) == (2, 2), "MD v shape"
        assert_near("MD state_moments mu", mu, mu_ref)
        assert_near("MD state_moments v", v, v_ref)
        assert_near("MD covariance symmetric", v, np.asarray(v).T)

        for method in DENSITY_METHODS:
            d = getattr(ens, method)(grid)
            assert np.shape(d) == (grid.shape[0],), "MD %s shape" % method
            assert_near("MD " + method, d, oracle_density(members, method, grid))


# ==========================================================================
# (3) Reproducibility: same (seed, runs) + same stream gives bit-identical
# ==========================================================================


def test_reproducibility():
    """Two identically-constructed ensembles agree bit for bit."""
    params = scalar_params()
    runs, seed = 5, 101
    cues, obs = scalar_stream()
    grid = np.linspace(-1, 1, 7)

    ens_a = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    ens_b = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for q, y in zip(cues, obs):
        for ens in (ens_a, ens_b):
            ens.observe_q(q)
            ens.observe_y(y)
        assert_bit("reproducible motor_output", ens_a.motor_output(),
                   ens_b.motor_output())
        mu_a, v_a = ens_a.state_moments()
        mu_b, v_b = ens_b.state_moments()
        assert_bit("reproducible mu", mu_a, mu_b)
        assert_bit("reproducible v", v_a, v_b)
        assert_bit("reproducible density", ens_a.state_probability(grid),
                   ens_b.state_probability(grid))


# ==========================================================================
# (4) Executor invariance: serial vs parallel, any segment_length
# ==========================================================================


def test_executor_invariance_stepping():
    """``max_cores`` / ``segment_length`` never change the stepping results."""
    params = scalar_params()
    runs, seed = 5, 202
    cues, obs = scalar_stream()
    grid = np.linspace(-1, 1, 7)

    serial = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=0,
                                  segment_length=1, **params)
    parallel = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=2,
                                    segment_length=4, **params)
    for q, y in zip(cues, obs):
        for ens in (serial, parallel):
            ens.observe_q(q)
            ens.observe_y(y)
        assert_bit("executor motor_output", serial.motor_output(),
                   parallel.motor_output())
        mu_s, v_s = serial.state_moments()
        mu_p, v_p = parallel.state_moments()
        assert_bit("executor mu", mu_s, mu_p)
        assert_bit("executor v", v_s, v_p)
        assert_bit("executor density", serial.state_probability(grid),
                   parallel.state_probability(grid))


def simulate_in_pool(ens, cues, obs):
    """Run ``simulate`` and assert the process pool was really used.

    Without this guard the executor-invariance checks would pass vacuously in
    any environment where the pool cannot start: ``_replay_parallel`` warns and
    falls back to the serial executor, so the test would compare serial against
    serial.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traces = ens.simulate(cues, obs)
    fallbacks = [w for w in caught if "process pool" in str(w.message)]
    assert not fallbacks, (
        "the process pool did not start, so this comparison would be vacuous: %s"
        % (fallbacks[0].message,)
    )
    return traces


def test_executor_invariance_simulate():
    """``simulate`` obeys executor invariance too (SPEC 3.3 + 6)."""
    params = scalar_params()
    runs, seed = 5, 202
    cues, obs = scalar_stream()

    serial = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=0,
                                  segment_length=1, **params)
    parallel = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=3,
                                    segment_length=5, **params)
    traces_s = serial.simulate(cues, obs)
    traces_p = simulate_in_pool(parallel, cues, obs)
    assert_bit("executor simulate motor", traces_s.motor_output,
               traces_p.motor_output)
    assert_bit("executor simulate state_mean", traces_s.state_mean,
               traces_p.state_mean)
    assert_bit("executor simulate state_var", traces_s.state_var,
               traces_p.state_var)


def test_executor_invariance_simulate_multidim():
    """The same invariance for the MD traces, whose covariance trace is 3-D.

    The ``(N, N, T)`` segment concatenation is a different code path from the
    scalar ``(T,)`` one, so it needs its own parallel check.
    """
    params = md_params()
    runs, seed = 3, 212
    cues, obs = md_stream()

    serial = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=0,
                                  segment_length=1, **params)
    parallel = RealTimeCOINEnsemble(runs=runs, seed=seed, max_cores=2,
                                    segment_length=4, **params)
    traces_s = serial.simulate(cues, obs)
    traces_p = simulate_in_pool(parallel, cues, obs)
    assert traces_s.state_var.shape == (2, 2, obs.shape[1])
    assert_bit("MD executor simulate motor", traces_s.motor_output,
               traces_p.motor_output)
    assert_bit("MD executor simulate state_mean", traces_s.state_mean,
               traces_p.state_mean)
    assert_bit("MD executor simulate state_var", traces_s.state_var,
               traces_p.state_var)


# ==========================================================================
# (5) Independence / seed sensitivity
# ==========================================================================


def test_seed_sensitivity():
    """Different seeds diverge; members within one ensemble differ."""
    params = scalar_params()
    runs = 4
    cues, obs = scalar_stream()

    ens_1 = RealTimeCOINEnsemble(runs=runs, seed=1, **params)
    ens_2 = RealTimeCOINEnsemble(runs=runs, seed=999, **params)
    for q, y in zip(cues, obs):
        for ens in (ens_1, ens_2):
            ens.observe_q(q)
            ens.observe_y(y)
    assert abs(ens_1.motor_output() - ens_2.motor_output()) > 1e-6, (
        "different seeds diverge"
    )

    members = oracle_members(1, runs, params)
    for q, y in zip(cues, obs):
        oracle_step(members, q, y)
    assert abs(members[0].motor_output() - members[1].motor_output()) > 1e-9, (
        "members within ensemble differ"
    )


# ==========================================================================
# (6a) simulate == the trial-by-trial stepping loop
# ==========================================================================


def test_simulate_matches_stepping():
    """Batch replay reproduces the stepping loop's queries (SPEC 6)."""
    params = scalar_params()
    runs, seed = 4, 303
    cues, obs = scalar_stream()
    length = len(obs)

    traces = RealTimeCOINEnsemble(runs=runs, seed=seed, **params).simulate(
        cues, obs
    )
    assert traces.motor_output.shape == (length,), "simulate motor shape"
    assert traces.state_mean.shape == (length,), "simulate state_mean shape"
    assert traces.state_var.shape == (length,), "simulate state_var shape"
    assert np.array_equal(traces.Trial, np.arange(1, length + 1))

    stepper = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for t, (q, y) in enumerate(zip(cues, obs)):
        stepper.observe_q(q)
        stepper.observe_y(y)
        mu, v = stepper.state_moments()
        assert_bit("simulate==step motor t=%d" % t, traces.motor_output[t],
                   stepper.motor_output())
        assert_bit("simulate==step mean t=%d" % t, traces.state_mean[t], mu)
        assert_bit("simulate==step var t=%d" % t, traces.state_var[t], v)


def test_simulate_matches_stepping_multidim():
    """The same equivalence for the multi-dimensional model."""
    params = md_params()
    runs, seed = 3, 313
    cues, obs = md_stream()
    length = obs.shape[1]

    traces = RealTimeCOINEnsemble(runs=runs, seed=seed, **params).simulate(
        cues, obs
    )
    assert traces.motor_output.shape == (2, length)
    assert traces.state_mean.shape == (2, length)
    assert traces.state_var.shape == (2, 2, length)

    stepper = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for t in range(length):
        stepper.observe_q(cues[t])
        stepper.observe_y(obs[:, t])
        mu, v = stepper.state_moments()
        assert_bit("MD simulate==step motor t=%d" % t,
                   traces.motor_output[:, t], stepper.motor_output())
        assert_bit("MD simulate==step mean t=%d" % t, traces.state_mean[:, t], mu)
        assert_bit("MD simulate==step var t=%d" % t, traces.state_var[:, :, t], v)


# ==========================================================================
# (6b) simulate repeatable; does not disturb the live stepping state
# ==========================================================================


def test_simulate_repeatable_and_nondisturbing():
    """Repeated calls agree, and the live state is untouched (SPEC 6)."""
    params = scalar_params()
    runs, seed = 4, 404
    cues, obs = scalar_stream()

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    first = ens.simulate(cues, obs)
    second = ens.simulate(cues, obs)
    assert_bit("simulate repeatable motor", first.motor_output,
               second.motor_output)
    assert_bit("simulate repeatable mean", first.state_mean, second.state_mean)
    assert_bit("simulate repeatable var", first.state_var, second.state_var)

    live = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for q, y in list(zip(cues, obs))[:3]:
        live.observe_q(q)
        live.observe_y(y)
    motor_before = live.motor_output()
    mu_before, v_before = live.state_moments()
    trial_before = live.Trial

    live.simulate(cues, obs)     # one-shot batch on a fresh member set
    assert live.Trial == trial_before, "simulate leaves Trial"
    mu_after, v_after = live.state_moments()
    assert_bit("simulate leaves motor", live.motor_output(), motor_before)
    assert_bit("simulate leaves mu", mu_after, mu_before)
    assert_bit("simulate leaves v", v_after, v_before)


# ==========================================================================
# (7) runs == 1 reduces to the single member exactly
# ==========================================================================


def test_runs_one_reduction():
    """A one-run ensemble is a transparent wrapper (SPEC 5.5)."""
    params = scalar_params()
    seed = 55
    cues, obs = scalar_stream()
    grid = np.linspace(-1, 1, 7)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    members = oracle_members(seed, 1, params)

    for q, y in zip(cues, obs):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)
        member = members[0]
        assert_bit("runs1 motor", ens.motor_output(), member.motor_output())
        mu, v = ens.state_moments()
        mu_m, v_m = member.state_moments()
        assert_bit("runs1 mu", mu, mu_m)
        assert_bit("runs1 v", v, v_m)
        assert_bit("runs1 density", ens.state_probability(grid),
                   member.state_probability(grid))


# ==========================================================================
# (8) snapshot / load_snapshot round-trip (SPEC section 7)
# ==========================================================================


def test_snapshot_roundtrip_member():
    """A restored member reproduces the original's queries and future."""
    params = scalar_params()
    grid = np.linspace(-1, 1, 7)

    a = RealTimeCOIN(rng=np.random.default_rng(7), **params)
    for q, y in zip([1, 1, 2, 2, 1], [0.1, 0.2, -0.1, 0.05, 0.15]):
        a.observe_q(q)
        a.observe_y(y)

    s = a.snapshot()
    b = RealTimeCOIN(rng=np.random.default_rng(999), **params)
    b.load_snapshot(s)

    assert b.Trial == a.Trial, "snapshot Trial"
    assert_bit("snapshot motor", b.motor_output(), a.motor_output())
    mu_a, v_a = a.state_moments()
    mu_b, v_b = b.state_moments()
    assert_bit("snapshot mu", mu_b, mu_a)
    assert_bit("snapshot v", v_b, v_a)
    assert_bit("snapshot density", b.state_probability(grid),
               a.state_probability(grid))

    # The snapshot carries the stream position, so no rewinding is needed for
    # the same subsequent input to give the same subsequent state.
    for model in (a, b):
        model.observe_q(2)
        model.observe_y(0.3)
    assert_bit("snapshot post-step motor", b.motor_output(), a.motor_output())
    mu_a2, v_a2 = a.state_moments()
    mu_b2, v_b2 = b.state_moments()
    assert_bit("snapshot post-step mu", mu_b2, mu_a2)
    assert_bit("snapshot post-step v", v_b2, v_a2)


def test_snapshot_roundtrip_ensemble():
    """An ensemble snapshot restores every member and the trial counter."""
    params = scalar_params()
    runs, seed = 3, 808
    cues, obs = scalar_stream()
    grid = np.linspace(-1, 1, 7)

    a = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    for q, y in list(zip(cues, obs))[:4]:
        a.observe_q(q)
        a.observe_y(y)

    b = RealTimeCOINEnsemble(runs=1, seed=0, num_particles=5)
    b.load_snapshot(a.snapshot())

    assert b.Trial == a.Trial and b.runs == a.runs
    assert_bit("ensemble snapshot motor", b.motor_output(), a.motor_output())
    assert_bit("ensemble snapshot density", b.state_probability(grid),
               a.state_probability(grid))

    for q, y in list(zip(cues, obs))[4:]:
        a.observe_q(q)
        a.observe_y(y)
        b.observe_q(q)
        b.observe_y(y)
    assert_bit("ensemble snapshot post-step motor", b.motor_output(),
               a.motor_output())


# ==========================================================================
# (9a) The ensemble leaves the caller's global RNG stream unchanged (SPEC 3.5)
# ==========================================================================


def _global_state():
    """Snapshot the legacy global ``numpy.random`` state as comparable parts."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return kind, keys.copy(), pos, has_gauss, cached


def _assert_global_unchanged(name, before):
    after = _global_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]), (
        "FAILED global RNG changed %s (SPEC 3.5)" % (name,)
    )
    assert before[2:] == after[2:], (
        "FAILED global RNG changed %s (SPEC 3.5)" % (name,)
    )


def test_global_rng_unchanged():
    """Construction, stepping and ``simulate`` never touch the global RNG."""
    params = scalar_params()
    cues, obs = scalar_stream()

    np.random.seed(123456)
    before = _global_state()

    ens = RealTimeCOINEnsemble(runs=4, seed=9, **params)
    _assert_global_unchanged("after construction", before)

    ens.observe_q(cues[0])
    ens.observe_y(obs[0])
    _assert_global_unchanged("after observe_y", before)

    ens.simulate(cues, obs)
    _assert_global_unchanged("after simulate", before)


def test_package_never_uses_the_legacy_global_rng():
    """No module calls a legacy global ``numpy.random`` routine.

    The empirical check above can only prove the paths it exercises; this one
    covers the package as a whole. ``default_rng`` (the seeded-Generator
    factory) is the sole permitted lower-case ``np.random`` name.
    """
    package = pathlib.Path(RealTimeCOIN.__module__.replace(".", "/")).parent
    root = pathlib.Path(__file__).resolve().parents[1] / package
    used = set()
    for source in sorted(root.glob("*.py")):
        used.update(re.findall(r"np\.random\.([a-z_]+)", source.read_text()))
    assert used <= {"default_rng"}, (
        "modules call legacy global numpy.random routines: %s"
        % (sorted(used - {"default_rng"}),)
    )


# ==========================================================================
# (9b) Edge cases: missing observation, cue-free trials (SPEC section 8)
# ==========================================================================


def test_edge_cases():
    """Missing feedback and cue-free trials stay well-defined and averaged."""
    params = scalar_params()
    runs, seed = 4, 71
    cues = [1, None, 2, 1]              # None: cue-free trial
    obs = [0.1, 0.2, float("nan"), 0.15]   # nan: missing observation
    grid = np.linspace(-1, 1, 7)

    ens = RealTimeCOINEnsemble(runs=runs, seed=seed, **params)
    members = oracle_members(seed, runs, params)

    for t, (q, y) in enumerate(zip(cues, obs), start=1):
        ens.observe_q(q)
        ens.observe_y(y)
        oracle_step(members, q, y)

        assert ens.Trial == t, "edge Trial==%d" % t
        u = ens.motor_output()
        assert_near("edge motor t=%d" % t, u, oracle_motor(members))
        assert np.all(np.isfinite(u)), "edge motor finite t=%d" % t

        d = ens.novel_state_probability(grid)
        # A saturated novel context contributes finite zeros, never NaN.
        assert np.all(np.isfinite(d)), "edge novel density finite t=%d" % t
        assert_near("edge novel density t=%d" % t, d,
                    oracle_density(members, "novel_state_probability", grid))


def test_missing_feedback_variants_advance_the_trial():
    """``None``, an empty array and ``nan`` all mark a missing observation."""
    ens = RealTimeCOINEnsemble(runs=2, seed=3, num_particles=20, max_contexts=3)
    for y in (None, np.array([]), float("nan")):
        trial = ens.Trial
        ens.observe_y(y)
        assert ens.Trial == trial + 1


# ==========================================================================
# Construction contract
# ==========================================================================


def test_constructor_validation_and_forwarding():
    """Ensemble parameters are validated; member parameters are forwarded."""
    ens = RealTimeCOINEnsemble(runs=3, seed=2, max_cores=1, segment_length=4,
                               num_particles=12, max_contexts=5)
    assert (ens.runs, ens.seed, ens.max_cores, ens.segment_length) == (3, 2, 1, 4)
    assert np.allclose(ens.weights, np.full(3, 1 / 3))
    assert all(m.num_particles == 12 and m.max_contexts == 5
               for m in ens.members)

    with pytest.raises(ValueError):
        RealTimeCOINEnsemble(runs=0)
    with pytest.raises(ValueError):
        RealTimeCOINEnsemble(runs=2.5)
    with pytest.raises(ValueError):
        RealTimeCOINEnsemble(seed=-1)
    with pytest.raises(ValueError):
        RealTimeCOINEnsemble(max_cores=-1)
    with pytest.raises(ValueError):
        RealTimeCOINEnsemble(segment_length=0)
    # rng is not a member parameter: the seed governs all randomness.
    with pytest.raises(ValueError):
        RealTimeCOINEnsemble(runs=2, rng=0)
    # An unknown member parameter surfaces the member constructor's error.
    with pytest.raises(Exception, match="NameValuePairs"):
        RealTimeCOINEnsemble(runs=2, not_a_parameter=1)


def test_readonly_properties():
    """``runs`` / ``seed`` / ``weights`` and friends cannot be reassigned."""
    ens = RealTimeCOINEnsemble(runs=2, seed=1, num_particles=10)
    for name in ("runs", "seed", "max_cores", "segment_length", "weights",
                 "Trial"):
        with pytest.raises(AttributeError):
            setattr(ens, name, 5)
    with pytest.raises(ValueError):
        ens.weights[0] = 1.0


# ==========================================================================
# Phase 2 - cross-run context alignment
# ==========================================================================


PHASE2_VECTOR_QUERIES = (
    "responsibilities_vector",
    "predicted_context_probabilities_vector",
    "sampled_context_count",
    "stationary_context_probabilities",
)
PHASE2_DENSITY_QUERIES = (
    "state_given_context_probability",
    "state_feedback_given_context_probability",
)


@pytest.mark.parametrize("method", PHASE2_VECTOR_QUERIES)
def test_phase2_context_vectors(method):
    """Aligned context vectors sum to 1 in the reference frame (SPEC 10.5.2)."""
    max_contexts = 3
    ens = RealTimeCOINEnsemble(runs=3, seed=17, num_particles=20,
                               max_contexts=max_contexts)
    for q, y in zip([1, 1, 2], [0.1, 0.2, -0.1]):
        ens.observe_q(q)
        ens.observe_y(y)
    value = getattr(ens, method)()
    # Only the stationary distribution may be empty, and only when no member
    # holds a context - which cannot be the case after three observed trials.
    assert value is not None, "%s returned None with contexts instantiated" % method
    value = np.asarray(value, dtype=float)
    if method != "stationary_context_probabilities":
        assert value.shape == (max_contexts + 1,), "%s shape" % method
    assert abs(float(np.sum(value)) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "method", PHASE2_VECTOR_QUERIES + PHASE2_DENSITY_QUERIES
)
def test_phase2_runs_one_reduction(method):
    """``runs == 1`` reduces to the single member (SPEC 10.5.1).

    The recommended Phase-2 oracle: with one run the reference frame IS the
    member's own frame, so the aligned query must reproduce the member's query
    exactly (scalar model).
    """
    params = dict(num_particles=20, max_contexts=3)
    seed = 19
    grid = np.linspace(-1, 1, 9)

    ens = RealTimeCOINEnsemble(runs=1, seed=seed, **params)
    member = oracle_members(seed, 1, params)[0]
    for q, y in zip([1, 1, 2, 2], [0.1, 0.2, -0.1, 0.3]):
        ens.observe_q(q)
        ens.observe_y(y)
        member.observe_q(q)
        member.observe_y(y)

    density = method in PHASE2_DENSITY_QUERIES
    value = getattr(ens, method)(grid) if density else getattr(ens, method)()
    expected = (
        getattr(member, method)(grid) if density else getattr(member, method)()
    )

    if density:
        assert set(value) == set(expected), "%s keys" % method
        for key, row in expected.items():
            assert_bit("%s[%s]" % (method, key), value[key], row)
    elif expected is None:
        assert value is None
    else:
        assert_bit(method, value, expected)


@pytest.mark.parametrize("method", PHASE2_DENSITY_QUERIES)
def test_phase2_context_densities(method):
    """Aligned per-context densities are keyed by reference label (SPEC 10.5.7)."""
    ens = RealTimeCOINEnsemble(runs=3, seed=17, num_particles=20, max_contexts=3)
    grid = np.linspace(-1, 1, 9)
    for q, y in zip([1, 1, 2], [0.1, 0.2, -0.1]):
        ens.observe_q(q)
        ens.observe_y(y)
    value = getattr(ens, method)(grid)
    for density in value.values():
        assert np.shape(density) == (grid.size,)

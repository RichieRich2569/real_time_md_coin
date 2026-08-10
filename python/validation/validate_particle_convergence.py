"""Monte Carlo convergence diagnostics across particle counts.

Translated from ``validation/validate_particle_convergence.m``.

Particle filters approximate integrals by empirical averages, so their random
error should shrink roughly like ``O(1 / sqrt(N_particles))`` while runtime
grows with the particle count. This validator checks that calibration (and,
when the oracle fixtures are present, agreement with the original offline COIN)
broadly improves as more particles are used, rather than behaving like a
structural bug.

What differs from the MATLAB original
-------------------------------------
1. **The PIT sub-run is inlined.** MATLAB delegates to
   ``validate_p_values_extended``; that validator belongs to a different
   translation unit here, so the same synthetic data-generating process is
   reproduced inside :func:`_pit_run` and the PIT values are computed
   independently (feedback PIT from
   :func:`validation.feedback_moments.predictive_feedback_mixture` plus
   :func:`validation.mixture_utils.mixture_cdf`; cue PIT as the randomized
   discrete transform ``F(q-) + u f(q)`` over
   :func:`realtimecoin.context.preview_cue_pmf`).

   Be aware of what this costs: MATLAB's PIT comes from the PRODUCTION queries
   ``predictive_state_feedback_cdf`` / ``predictive_cue_p_value``, both of
   which are unimplemented in this worktree, so ``best_feedback_ks`` currently
   gates the validation mirror rather than production code. Switch ``_pit_run``
   over to those queries once they land.
2. **The oracle RMSE is FIXTURE-GATED.** MATLAB calls ``compare_original_coin``,
   which runs the offline ``COIN.m`` to generate both the feedback stream and
   the reference motor output. There is no Python COIN, so the oracle traces
   must be exported from MATLAB once and dropped into
   ``python/tests/fixtures/oracle/``. When they are absent, every
   oracle-dependent metric is ``None``, ``checks["best_rmse"]`` is ``None``
   (skipped, not failed) and ``results["skipped_reason"]`` explains why.
3. **The timing therefore covers the PIT sub-run only** (plus the oracle replay
   when fixtures exist), where MATLAB times the PIT sub-run and the COIN
   comparison together.
4. **``calibration_or_rmse_improves`` gains a tolerance when - and only when -
   the oracle is missing.** It is a disjunction in MATLAB; dropping one arm
   would make it stricter than the original, so the surviving arm is widened by
   ``THRESHOLDS["calibration_ks_tolerance"]``. See that entry for the
   derivation.

Oracle fixture format
---------------------
Each fixture is a ``.npz`` archive in ``python/tests/fixtures/oracle/``, one per
particle count, named either::

    compare_original_coin_p{particles}_s{seed}.npz      (preferred, exact match)
    compare_original_coin_p{particles}.npz              (fallback, any seed)

Required arrays (all length ``T``, produced by one ``COIN.m`` run as in
``validation/compare_original_coin.m``):

``cues``
    Raw cue value presented on each trial (``old.cues``).
``feedback``
    State feedback COIN generated internally
    (``S.runs{1}.state_feedback``).
``motor_output``
    COIN's stored motor output (``S.runs{1}.stored.motor_output``), the
    quantity the RMSE is measured against.

Optional 0-d arrays overriding the RealTimeCOIN replay hyperparameters (each
defaults to the value ``compare_original_coin.m`` uses): ``max_contexts``,
``gamma_context``, ``alpha_context``, ``rho_context``, ``gamma_cue``,
``alpha_cue``, ``prior_mean_retention``, ``prior_precision_retention``,
``prior_mean_drift``, ``prior_precision_drift``, ``sigma_process_noise``,
``sigma_sensory_noise``, ``sigma_motor_noise``.

The replay itself mirrors ``compare_original_coin.m``: for each trial record
``predictive_motor_output(cue)``, then ``observe_q(cue)`` and
``observe_y(feedback)``; the RMSE is taken over the trials where both traces are
finite.

Run as a script for a MATLAB-style console summary::

    python -m validation.validate_particle_convergence --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from realtimecoin import RealTimeCOIN
from realtimecoin.context import preview_cue_pmf
from realtimecoin.state import peek_cue_label

from .feedback_moments import predictive_feedback_mixture
from .mixture_utils import mixture_cdf
from .sample_categorical import sample_categorical
from .uniform_ks import uniform_ks

__all__ = [
    "run",
    "THRESHOLDS",
    "DEFAULT_PARTICLES",
    "DEFAULT_TRIALS",
    "DEFAULT_DATASETS",
    "ORACLE_FIXTURE_DIR",
]

#: Compact-profile sweep from ``run_validation.m`` (``convergence_particles`` /
#: ``convergence_trials`` / ``convergence_datasets``).
DEFAULT_PARTICLES = (25, 50, 100)
DEFAULT_TRIALS = 45
DEFAULT_DATASETS = 4

#: Where the offline-COIN oracle traces are looked for. See the module
#: docstring for the archive format.
ORACLE_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "oracle"
)

#: Gate rationale, carried over verbatim: ``best_feedback_ks`` (0.12) is looser
#: than the standalone p-values feedback gate (0.08) because this validator
#: re-runs the PIT with fewer datasets/trials, so its KS statistic has higher
#: sampling variance. ``best_rmse`` (0.05) matches the Kalman validators'
#: RMSE gate. ``runtime_ratio_floor`` is only a SOFT trend floor - wall-clock
#: timing on a shared machine is noisy, and a strict per-step monotonicity test
#: flakes under ordinary jitter.
#:
#: ``calibration_ks_tolerance`` has no MATLAB counterpart. MATLAB's
#: ``calibration_or_rmse_improves`` is a DISJUNCTION -
#: ``ks[-1] <= ks[0] or rmse[-1] <= rmse[0]`` - and without the offline-COIN
#: oracle the second arm cannot be evaluated, which would leave a check
#: STRICTER than the one being transliterated. The tolerance restores the
#: intended leniency on a statistically principled scale: under the null the
#: pooled KS statistic has standard deviation about ``0.26 / sqrt(n)``, so at
#: the compact profile's ``n = datasets * trials = 180`` two runs differ by
#: about 0.03 (1 sd) by chance alone. 0.05 is roughly 2 sd - still small enough
#: to catch a genuinely diverging filter, but immune to the sub-1e-4 ties this
#: comparison otherwise turns on. It is used ONLY when the oracle arm is
#: missing; with fixtures present the MATLAB disjunction is applied verbatim.
THRESHOLDS = {
    "best_feedback_ks": 0.12,
    "best_rmse": 0.05,
    "runtime_ratio_floor": 0.75,
    "calibration_ks_tolerance": 0.05,
}

# --- synthetic DGP, identical to validate_p_values_extended.m's ---
_A = 0.9
_D = 0.02
_SIGMA_Q = 0.03
_SIGMA_R = 0.05
_CUE_PROB = np.array([[0.85, 0.15], [0.15, 0.85]])
_TRANSITION = np.array([[0.96, 0.04], [0.04, 0.96]])
_CONTEXT_OFFSET = np.array([-0.25, 0.25])

# --- defaults for the oracle replay, from compare_original_coin.m ---
_ORACLE_DEFAULTS = {
    "max_contexts": 4,
    "prior_mean_drift": 0.0,
    "sigma_motor_noise": 0.0,
}


def _pass_summary(checks):
    """Aggregate named checks into one overall pass flag.

    Local stand-in for ``validation_pass_summary.m``. ``None`` marks a SKIPPED
    check: it is preserved in the output and excluded from the verdict, so a
    missing oracle fixture never turns into a spurious failure. ``nan`` still
    counts as a failure.

    Parameters
    ----------
    checks : dict
        ``{name: bool or float or None}``.

    Returns
    -------
    passed : bool
        True when every non-skipped check holds.
    checks : dict
        The same mapping with numeric entries coerced to ``bool``.
    """
    out = {}
    passed = True
    for name, value in checks.items():
        if value is None:
            out[name] = None
            continue
        ok = bool(value) and bool(np.isfinite(float(value)))
        out[name] = ok
        passed = passed and ok
    return passed, out


def _cue_p_value(model, q_raw, u):
    """Randomized discrete PIT of a cue under the current predictive cue pmf.

    ``p = F(q-) + u f(q)``: the ``u f(q)`` term spreads the atom ``f(q)`` across
    ``[F(q-), F(q)]`` so that a correctly specified model gives a Uniform(0, 1)
    value despite the cue being discrete. A never-seen cue maps to the next
    novel label, which carries zero mass, so its p-value reduces to ``F(q-)``.

    An independent re-implementation of ``predictive_cue_p_value.m`` over the
    same ``preview_cue_pmf`` the class uses.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Model whose predictive cue pmf is read. Not mutated.
    q_raw : float
        RAW cue value about to be presented.
    u : float
        Uniform(0, 1) randomisation variate.

    Returns
    -------
    float
        The p-value in ``[0, 1]``, or ``nan`` when there is no cue.
    """
    q_label = peek_cue_label(model, q_raw)
    if q_label is None:
        return float("nan")

    pmf, _labels = preview_cue_pmf(model)
    if q_label >= pmf.size:
        pmf = np.concatenate([pmf, np.zeros(q_label + 1 - pmf.size)])
    f = float(pmf[q_label])
    f_minus = float(pmf[:q_label].sum())
    return float(np.fmin(np.fmax(f_minus + u * f, 0.0), 1.0))


def _pit_run(seed, datasets, trials, particles):
    """Prequential calibration run: pooled feedback and cue PIT values.

    Reproduces the data-generating process of
    ``validate_p_values_extended.m``: a two-context Markov chain emitting noisy
    cues, with the latent state following the shared linear-Gaussian random walk
    ``s_t = a s_{t-1} + d + N(0, sigma_Q^2)`` and feedback ``y_t = s_t +
    N(0, sigma_R^2)``. (As in MATLAB, the context offsets only set the initial
    state; the chain drives the CUE emissions.)

    Parameters
    ----------
    seed : int
        Seed for this sub-run; data, model and PIT-randomisation streams are
        spawned from it.
    datasets : int
        Number of independent datasets (each gets a fresh model).
    trials : int
        Trials per dataset.
    particles : int
        Particle count of the models.

    Returns
    -------
    feedback_ks : float
        KS distance of the pooled feedback PIT from Uniform(0, 1).
    cue_ks : float
        KS distance of the pooled randomized cue PIT.
    """
    data_seed, model_seed, u_seed = np.random.SeedSequence(seed).spawn(3)
    data_rng = np.random.default_rng(data_seed)
    model_seeds = model_seed.spawn(datasets)
    u_rng = np.random.default_rng(u_seed)

    feedback_p = []
    cue_p = []

    for dataset in range(datasets):
        model = RealTimeCOIN(
            num_particles=particles,
            max_contexts=4,
            prior_mean_retention=_A,
            prior_precision_retention=1e6,
            prior_mean_drift=_D,
            prior_precision_drift=1e6,
            sigma_process_noise=_SIGMA_Q,
            sigma_sensory_noise=_SIGMA_R,
            sigma_motor_noise=0.0,
            rng=np.random.default_rng(model_seeds[dataset]),
        )
        c = 0
        s = float(_CONTEXT_OFFSET[c])

        for _t in range(trials):
            c = sample_categorical(data_rng, _TRANSITION[c])
            # Raw cue values are 1 / 2, matching the MATLAB paradigm.
            q_raw = float(sample_categorical(data_rng, _CUE_PROB[c]) + 1)
            s = _A * s + _D + _SIGMA_Q * data_rng.standard_normal()
            y = s + _SIGMA_R * data_rng.standard_normal()

            q_label = peek_cue_label(model, q_raw)
            w, m, v = predictive_feedback_mixture(model, q_label)
            feedback_p.append(mixture_cdf(y, w, m, v))
            cue_p.append(_cue_p_value(model, q_raw, float(u_rng.random())))

            model.observe_q(q_raw)
            model.observe_y(y)

    return uniform_ks(feedback_p), uniform_ks(cue_p)


def _find_oracle_fixture(oracle_dir, particles, seed):
    """Locate the oracle archive for one particle count, if it exists.

    Parameters
    ----------
    oracle_dir : pathlib.Path
        Directory searched.
    particles : int
        Particle count the fixture must correspond to.
    seed : int
        Seed of the exact-match candidate; the seedless fallback is used when
        no exact match is present.

    Returns
    -------
    pathlib.Path or None
        The archive path, or ``None`` when neither candidate exists.
    """
    candidates = [
        oracle_dir / ("compare_original_coin_p%d_s%d.npz" % (particles, seed)),
        oracle_dir / ("compare_original_coin_p%d.npz" % particles),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _oracle_rmse(path, particles, seed):
    """Replay an oracle fixture through RealTimeCOIN and return the motor RMSE.

    Mirrors ``compare_original_coin.m``: the recorded feedback stream (generated
    by the offline COIN) is fed to RealTimeCOIN one trial at a time, and the
    predictive motor output is compared with COIN's own.

    Parameters
    ----------
    path : pathlib.Path
        ``.npz`` archive; see the module docstring for its contents.
    particles : int
        Particle count for the replay model.
    seed : int
        Seed for the replay model's generator.

    Returns
    -------
    rmse : float
        Root-mean-square difference of the two motor-output traces over the
        trials where both are finite, or ``nan`` when none are.
    correlation : float
        Pearson correlation of the same two traces, or ``nan``.
    """
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}

    cues = np.asarray(data["cues"], dtype=float).reshape(-1)
    feedback = np.asarray(data["feedback"], dtype=float).reshape(-1)
    old_motor = np.asarray(data["motor_output"], dtype=float).reshape(-1)
    n_trials = min(cues.size, feedback.size, old_motor.size)

    kwargs = dict(_ORACLE_DEFAULTS)
    for name in (
        "max_contexts",
        "gamma_context",
        "alpha_context",
        "rho_context",
        "gamma_cue",
        "alpha_cue",
        "prior_mean_retention",
        "prior_precision_retention",
        "prior_mean_drift",
        "prior_precision_drift",
        "sigma_process_noise",
        "sigma_sensory_noise",
        "sigma_motor_noise",
    ):
        if name in data:
            value = np.asarray(data[name]).reshape(-1)[0]
            kwargs[name] = int(value) if name == "max_contexts" else float(value)

    model = RealTimeCOIN(num_particles=particles, rng=seed, **kwargs)
    rt_motor = np.zeros(n_trials)
    for t in range(n_trials):
        rt_motor[t] = model.predictive_motor_output(cues[t])
        model.observe_q(cues[t])
        model.observe_y(feedback[t])

    valid = np.isfinite(old_motor[:n_trials]) & np.isfinite(rt_motor)
    if not np.any(valid):
        return float("nan"), float("nan")
    rmse = float(
        np.sqrt(np.mean((old_motor[:n_trials][valid] - rt_motor[valid]) ** 2))
    )
    correlation = float("nan")
    if int(valid.sum()) > 1:
        # A constant trace has zero variance, so the correlation is undefined;
        # MATLAB's corr returns NaN there too. errstate keeps numpy from
        # warning about the 0/0 it computes on the way to that NaN.
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = float(
                np.corrcoef(old_motor[:n_trials][valid], rt_motor[valid])[0, 1]
            )
    return rmse, correlation


def run(seed=0, strict=False, **overrides):
    """Run the particle-count convergence sweep.

    Parameters
    ----------
    seed : int, optional
        Base seed. Sub-run ``i`` (1-based, as in MATLAB) uses ``seed + i`` for
        the PIT run and ``seed + 100 + i`` for the oracle replay.
    strict : bool, optional
        When True, raise :class:`AssertionError` if any non-skipped gate fails.
    **overrides
        ``particles`` (default ``(25, 50, 100)``), ``trials`` (default 45),
        ``datasets`` (default 4) and ``oracle_dir`` (default
        :data:`ORACLE_FIXTURE_DIR`).

    Returns
    -------
    dict
        ``particles``, ``feedback_ks``, ``cue_ks``, ``original_coin_rmse``
        (``None`` entries when the fixture is missing), ``elapsed_seconds``,
        ``best_feedback_ks``, ``best_original_coin_rmse`` (aliased as
        ``best_rmse``), ``thresholds``, ``checks``, ``passed``, ``errored``,
        ``skipped_reason`` and ``config``.

    Raises
    ------
    TypeError
        On an unrecognised override.
    AssertionError
        If ``strict`` and any non-skipped gate fails.
    """
    requested = np.asarray(overrides.pop("particles", DEFAULT_PARTICLES))
    particles = tuple(int(p) for p in requested.reshape(-1))
    trials = int(overrides.pop("trials", DEFAULT_TRIALS))
    datasets = int(overrides.pop("datasets", DEFAULT_DATASETS))
    oracle_dir = Path(overrides.pop("oracle_dir", ORACLE_FIXTURE_DIR))
    if overrides:
        raise TypeError(
            "validate_particle_convergence.run got unexpected keyword "
            "argument(s): %s." % ", ".join(sorted(overrides))
        )

    n_steps = len(particles)
    feedback_ks = np.zeros(n_steps)
    cue_ks = np.zeros(n_steps)
    seconds = np.zeros(n_steps)
    rmse = [None] * n_steps
    correlation = [None] * n_steps
    missing = []

    for i, count in enumerate(particles):
        start = time.perf_counter()
        feedback_ks[i], cue_ks[i] = _pit_run(
            seed + i + 1, datasets, trials, count
        )
        fixture = _find_oracle_fixture(oracle_dir, count, seed + 100 + i + 1)
        if fixture is None:
            missing.append(count)
        else:
            rmse[i], correlation[i] = _oracle_rmse(
                fixture, count, seed + 100 + i + 1
            )
        seconds[i] = time.perf_counter() - start

    have_oracle = all(value is not None for value in rmse)
    skipped_reason = None
    if not have_oracle:
        skipped_reason = (
            "offline-COIN oracle fixtures missing for particle count(s) %s in "
            "%s; original_coin_rmse metrics and the best_rmse gate are skipped "
            "(see the module docstring for the expected archive format)."
            % (", ".join(str(c) for c in missing), oracle_dir)
        )

    best_feedback_ks = float(feedback_ks[-1])
    best_rmse = float(rmse[-1]) if have_oracle else None

    # Soft trend floor: the largest particle count must not run markedly faster
    # than the smallest. Not a per-step monotonicity test - see THRESHOLDS.
    runtime_nondecreasing = bool(
        seconds[-1] >= THRESHOLDS["runtime_ratio_floor"] * seconds[0]
    )
    if have_oracle:
        # MATLAB's disjunction, verbatim.
        calibration_improves = bool(feedback_ks[-1] <= feedback_ks[0]) or bool(
            rmse[-1] <= rmse[0]
        )
    else:
        # Oracle arm unavailable: widen the surviving arm by the KS sampling
        # tolerance so the degraded check is not stricter than MATLAB's.
        calibration_improves = bool(
            feedback_ks[-1]
            <= feedback_ks[0] + THRESHOLDS["calibration_ks_tolerance"]
        )

    passed, checks = _pass_summary(
        {
            "best_feedback_ks": best_feedback_ks < THRESHOLDS["best_feedback_ks"],
            "best_rmse": (
                (best_rmse < THRESHOLDS["best_rmse"]) if have_oracle else None
            ),
            "runtime_nondecreasing": runtime_nondecreasing,
            "calibration_or_rmse_improves": calibration_improves,
        }
    )

    results = {
        "particles": particles,
        "feedback_ks": feedback_ks,
        "cue_ks": cue_ks,
        "original_coin_rmse": rmse,
        "original_coin_correlation": correlation,
        "elapsed_seconds": seconds,
        "best_feedback_ks": best_feedback_ks,
        "best_original_coin_rmse": best_rmse,
        "best_rmse": best_rmse,
        "thresholds": dict(THRESHOLDS),
        # True when calibration_or_rmse_improves ran in its widened,
        # oracle-free form rather than MATLAB's exact disjunction, so a report
        # can tell which variant produced the verdict.
        "calibration_tolerance_applied": not have_oracle,
        "checks": checks,
        "passed": passed,
        "errored": False,
        "skipped_reason": skipped_reason,
        "config": {
            "seed": seed,
            "particles": particles,
            "trials": trials,
            "datasets": datasets,
            "oracle_dir": str(oracle_dir),
        },
    }

    rmse_first = "n/a" if rmse[0] is None else "%.4f" % rmse[0]
    rmse_last = "n/a" if rmse[-1] is None else "%.4f" % rmse[-1]
    print(
        "Particle convergence: feedback KS %.3f -> %.3f, original RMSE %s -> %s"
        % (feedback_ks[0], feedback_ks[-1], rmse_first, rmse_last)
    )
    if skipped_reason is not None:
        print("  skipped: %s" % skipped_reason)

    if strict and not passed:
        raise AssertionError(
            "Particle convergence validation failed: %r" % (checks,)
        )
    return results


def _main(argv=None):
    """Command-line entry point printing the MATLAB-style summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--particles", type=int, nargs="+", default=list(DEFAULT_PARTICLES)
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--datasets", type=int, default=DEFAULT_DATASETS)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    results = run(
        seed=args.seed,
        strict=args.strict,
        particles=tuple(args.particles),
        trials=args.trials,
        datasets=args.datasets,
    )
    for i, count in enumerate(results["particles"]):
        print(
            "  N=%-5d feedback KS %.3f  cue KS %.3f  %.2f s"
            % (
                count,
                results["feedback_ks"][i],
                results["cue_ks"][i],
                results["elapsed_seconds"][i],
            )
        )
    for name, ok in results["checks"].items():
        label = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print("  %-30s %s" % (name, label))
    print("passed: %s" % results["passed"])
    return 0 if results["passed"] else 1


if __name__ == "__main__":   # pragma: no cover - script entry point
    raise SystemExit(_main())

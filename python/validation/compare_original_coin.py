"""Probabilistic comparison with the original offline COIN, by fixture replay.

Translated from ``validation/compare_original_coin.m``.

How this differs from the ``.m``
--------------------------------
The MATLAB script RUNS the offline ``COIN.m`` to generate both the feedback
stream and the reference motor output, then replays that feedback through
``RealTimeCOIN``. There is no Python ``COIN``, so the generation half is
replaced by a FROZEN FIXTURE captured once from the live MATLAB session; the
replay half is transliterated exactly.

Everything the fixture pins down - the perturbation schedule, the cue schedule,
the generated feedback ``y``, COIN's motor output ``mo``, the hyperparameters
``hp``, the trial count ``T`` and the particle count ``P`` - was produced with
the ``.m``'s configuration verbatim (``max_contexts = 4``,
``prior_mean_drift = 0``, ``sigma_motor_noise = 0``, ``store = {'motor_output',
'state_feedback'}``).

Consequently ``trials`` and ``particles`` are CROSS-CHECKS here rather than
inputs: passing a value that disagrees with the fixture raises, because the
paradigm cannot be re-derived at a different size without re-running COIN.

Loop-order fidelity
-------------------
The replay loop is the ``.m``'s, in order, and the order is load-bearing::

    for t in trials:
        rt_motor[t] = predictive_motor_output(cues[t])   # BEFORE the update
        observe_q(cues[t])
        observe_y(y[t])

``predictive_motor_output`` is read FIRST, so it is the prior predictive for
trial ``t`` - the same quantity COIN stores as ``motor_output(t)``. Reading it
after ``observe_y`` would compare a posterior against a prior and shift the
traces by one trial.

Cue values are the fixture's raw 1-based MATLAB labels, passed straight through:
the package's cue registry maps distinct raw values to consecutive internal
labels, so only which values REPEAT matters.

Fixture format
--------------
``python/tests/fixtures/oracle/coin_trace_seed<seed>.mat``, readable with
``scipy.io.loadmat(..., squeeze_me=True)``. See
``python/tests/fixtures/oracle/README.md`` for the full variable list and the
MATLAB snippet that produced it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import scipy.io

if __package__ in (None, ""):        # pragma: no cover - direct-script launch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from realtimecoin import RealTimeCOIN

__all__ = [
    "ORACLE_FIXTURE_DIR",
    "DEFAULT_SEED",
    "available_seeds",
    "load_trace",
    "run",
    "export_oracle_npz",
]

#: Directory holding the frozen offline-COIN traces.
ORACLE_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "oracle"
)

#: Seed of the trace used when none is requested (the ``.m``'s default).
DEFAULT_SEED = 2001

#: Constructor keywords fixed by ``compare_original_coin.m`` rather than by the
#: stored hyperparameter struct. ``prior_mean_drift`` is set to 0 explicitly in
#: the ``.m``; ``max_contexts`` is its literal 4.
_FIXED_KWARGS = {"max_contexts": 4, "prior_mean_drift": 0.0}

#: Hyperparameters read out of the fixture's ``hp`` struct and forwarded to the
#: constructor under the same names.
_HP_KWARGS = (
    "gamma_context",
    "alpha_context",
    "rho_context",
    "gamma_cue",
    "alpha_cue",
    "prior_mean_retention",
    "prior_precision_retention",
    "prior_precision_drift",
    "sigma_process_noise",
    "sigma_sensory_noise",
    "sigma_motor_noise",
)


def _trace_path(seed, fixture_dir=None):
    """Path of one trace fixture.

    Parameters
    ----------
    seed : int
        Seed the MATLAB capture used.
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`ORACLE_FIXTURE_DIR`.

    Returns
    -------
    pathlib.Path
        Candidate path (not necessarily existing).
    """
    root = (
        pathlib.Path(fixture_dir)
        if fixture_dir is not None
        else ORACLE_FIXTURE_DIR
    )
    return root / ("coin_trace_seed%d.mat" % int(seed))


def available_seeds(fixture_dir=None):
    """Seeds for which a trace fixture exists.

    Parameters
    ----------
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`ORACLE_FIXTURE_DIR`.

    Returns
    -------
    list of int
        Sorted seeds parsed from ``coin_trace_seed<seed>.mat`` file names.
    """
    root = (
        pathlib.Path(fixture_dir)
        if fixture_dir is not None
        else ORACLE_FIXTURE_DIR
    )
    if not root.is_dir():
        return []
    seeds = []
    for path in root.glob("coin_trace_seed*.mat"):
        digits = path.stem[len("coin_trace_seed"):]
        if digits.isdigit():
            seeds.append(int(digits))
    return sorted(seeds)


def load_trace(seed=DEFAULT_SEED, fixture_dir=None):
    """Load one frozen offline-COIN trace.

    Parameters
    ----------
    seed : int, optional
        Seed of the trace. Default :data:`DEFAULT_SEED`.
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`ORACLE_FIXTURE_DIR`.

    Returns
    -------
    dict
        Keys ``seed``, ``trials``, ``particles``, ``cues`` (``(T,)`` float),
        ``feedback`` (``(T,)``), ``motor_output`` (``(T,)``),
        ``perturbations`` (``(T,)``) and ``hyperparameters`` (``dict``).

    Raises
    ------
    FileNotFoundError
        If the fixture is absent.

    Notes
    -----
    ``squeeze_me=True`` reduces MATLAB's ``1 x T`` row vectors to ``(T,)`` and
    its scalars to 0-d arrays, so every field is reshaped explicitly rather than
    indexed positionally.
    """
    path = _trace_path(seed, fixture_dir)
    if not path.is_file():
        raise FileNotFoundError(
            "missing offline-COIN trace fixture %s (see "
            "python/tests/fixtures/oracle/README.md)" % path
        )
    raw = scipy.io.loadmat(str(path), squeeze_me=True)

    hp_struct = raw["hp"]
    hyperparameters = {
        name: float(np.asarray(hp_struct[name]).reshape(-1)[0])
        for name in _HP_KWARGS
    }
    return {
        "seed": int(seed),
        "trials": int(np.asarray(raw["T"]).reshape(-1)[0]),
        "particles": int(np.asarray(raw["P"]).reshape(-1)[0]),
        "cues": np.asarray(raw["cues"], dtype=float).reshape(-1),
        "feedback": np.asarray(raw["y"], dtype=float).reshape(-1),
        "motor_output": np.asarray(raw["mo"], dtype=float).reshape(-1),
        "perturbations": np.asarray(raw["pert"], dtype=float).reshape(-1),
        "hyperparameters": hyperparameters,
    }


def _correlation(a, b):
    """Pearson correlation, or ``nan`` when it is undefined.

    Parameters
    ----------
    a, b : numpy.ndarray
        Equal-length 1-D samples.

    Returns
    -------
    float
        ``corr(a, b)``; ``nan`` if fewer than two points or either sample is
        constant (MATLAB's ``corr`` returns ``NaN`` there too, with a warning).
    """
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(seed=DEFAULT_SEED, trials=None, particles=None, fixture_dir=None):
    """Replay one frozen COIN trace through ``RealTimeCOIN`` and score it.

    Parameters
    ----------
    seed : int, optional
        Trace to replay; also the seed of the ``RealTimeCOIN`` generator, as in
        the ``.m`` (which calls ``rng(cfg.Seed)`` before constructing it).
        Default :data:`DEFAULT_SEED`.
    trials : int or None, optional
        Cross-check on the fixture's ``T``; ``None`` accepts whatever it stores.
    particles : int or None, optional
        Cross-check on the fixture's ``P``; ``None`` accepts whatever it stores.
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`ORACLE_FIXTURE_DIR`.

    Returns
    -------
    dict
        Field names verbatim from the ``.m``: ``rmse_motor_output``,
        ``correlation_motor_output``, ``original_motor_output``,
        ``realtime_motor_output``, ``feedback`` and ``config``.

    Raises
    ------
    FileNotFoundError
        If the fixture is absent.
    ValueError
        If ``trials`` or ``particles`` disagrees with the fixture.
    """
    trace = load_trace(seed, fixture_dir)
    if trials is not None and int(trials) != trace["trials"]:
        raise ValueError(
            "compare_original_coin: fixture for seed %d holds T = %d trials, "
            "but trials = %s was requested. The paradigm cannot be resized "
            "without re-running COIN.m; re-export the fixture instead."
            % (seed, trace["trials"], trials)
        )
    if particles is not None and int(particles) != trace["particles"]:
        raise ValueError(
            "compare_original_coin: fixture for seed %d was generated with "
            "P = %d particles, but particles = %s was requested. COIN's stored "
            "motor output depends on P; re-export the fixture instead."
            % (seed, trace["particles"], particles)
        )

    n_trials = trace["trials"]
    cues = trace["cues"]
    feedback = trace["feedback"]
    old_motor = trace["motor_output"]

    kwargs = dict(_FIXED_KWARGS)
    kwargs.update(trace["hyperparameters"])
    model = RealTimeCOIN(
        num_particles=trace["particles"], rng=int(seed), **kwargs
    )

    # Loop order transliterated from the .m: predict, stage the cue, observe.
    rt_motor = np.zeros(n_trials)
    for t in range(n_trials):
        rt_motor[t] = model.predictive_motor_output(cues[t])
        model.observe_q(cues[t])
        model.observe_y(feedback[t])

    valid = np.isfinite(old_motor) & np.isfinite(rt_motor)
    if not np.any(valid):
        rmse = float("nan")
        correlation = float("nan")
    else:
        rmse = float(
            np.sqrt(np.mean((old_motor[valid] - rt_motor[valid]) ** 2))
        )
        correlation = _correlation(old_motor[valid], rt_motor[valid])

    print(
        "Original/RT motor output RMSE: %.4f, correlation: %.3f"
        % (rmse, correlation)
    )

    return {
        "rmse_motor_output": rmse,
        "correlation_motor_output": correlation,
        "original_motor_output": old_motor,
        "realtime_motor_output": rt_motor,
        "feedback": feedback,
        "perturbations": trace["perturbations"],
        "cues": cues,
        "config": {
            "trials": n_trials,
            "particles": trace["particles"],
            "seed": int(seed),
            "fixture": str(_trace_path(seed, fixture_dir)),
        },
    }


def export_oracle_npz(seed=DEFAULT_SEED, fixture_dir=None, out_dir=None):
    """Convert one ``.mat`` trace into the ``.npz`` archive other tools expect.

    ``validation.validate_particle_convergence`` has a documented oracle hook
    that reads ``compare_original_coin_p{particles}[_s{seed}].npz`` archives
    holding ``cues``, ``feedback`` and ``motor_output``. This writes such an
    archive from a frozen ``.mat`` trace.

    Parameters
    ----------
    seed : int, optional
        Trace to convert. Default :data:`DEFAULT_SEED`.
    fixture_dir : path-like or None, optional
        Where the ``.mat`` lives; ``None`` uses :data:`ORACLE_FIXTURE_DIR`.
    out_dir : path-like or None, optional
        Where to write the ``.npz``; ``None`` uses the same directory.

    Returns
    -------
    pathlib.Path
        The archive written.

    Notes
    -----
    This does NOT by itself enable the convergence validator's oracle arm. That
    validator sweeps particle counts (25, 50, 100 in the compact profile) and
    needs a trace generated by ``COIN.m`` AT EACH COUNT - MATLAB's
    ``validate_particle_convergence.m`` calls ``compare_original_coin('Particles',
    particles(i))``, so COIN itself is re-run at every count. The frozen traces
    here are all ``P = 100``, so only the ``p100`` slot can be filled and the
    validator correctly keeps reporting its oracle arm as skipped. Reusing the
    ``P = 100`` reference for the ``P = 25`` and ``P = 50`` slots would measure a
    DIFFERENT quantity (RealTimeCOIN at N particles versus COIN at 100) and is
    deliberately not done.
    """
    trace = load_trace(seed, fixture_dir)
    root = (
        pathlib.Path(out_dir)
        if out_dir is not None
        else _trace_path(seed, fixture_dir).parent
    )
    path = root / (
        "compare_original_coin_p%d_s%d.npz" % (trace["particles"], int(seed))
    )
    np.savez(
        path,
        cues=trace["cues"],
        feedback=trace["feedback"],
        motor_output=trace["motor_output"],
        max_contexts=np.asarray(_FIXED_KWARGS["max_contexts"]),
        prior_mean_drift=np.asarray(_FIXED_KWARGS["prior_mean_drift"]),
        **{
            name: np.asarray(value)
            for name, value in trace["hyperparameters"].items()
        },
    )
    return path


def _main(argv=None):       # pragma: no cover - console entry point
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Always ``0``; this script reports metrics, it does not gate.
    """
    parser = argparse.ArgumentParser(
        description="Replay a frozen offline-COIN trace through RealTimeCOIN."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--export-npz",
        action="store_true",
        help="also write the .npz form consumed by other validators",
    )
    args = parser.parse_args(argv)
    run(seed=args.seed)
    if args.export_npz:
        print("wrote %s" % export_oracle_npz(seed=args.seed))
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(_main())

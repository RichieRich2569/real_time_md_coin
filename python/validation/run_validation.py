"""Compact RealTimeCOIN validation suite driver.

Translated from ``validation/run_validation.m``.

These stages report metrics and pass flags rather than acting like fast unit
tests; increase the per-stage sizes for publication-scale calibration runs.

Stage isolation
---------------
Every stage runs inside :func:`_run_stage`, so one failing validator records
``passed=False`` plus the captured error and the suite CONTINUES, rather than an
uncaught exception tearing down the independent validators that follow it.

Every sub-validator is called with ``strict=False`` (never the suite's
``strict``) on purpose: forwarding strictness makes a sub-validator raise
internally on failure, which the stage runner would collapse into a bare
``{passed, errored, error}`` dict, discarding the full metric struct we want to
inspect. With ``strict=False`` the sub-validators always return their complete
metrics (with ``passed=False`` on failure), and strictness is enforced ONCE at
the suite level via the ``results["passed"]`` gate.

Stage availability in this worktree
-----------------------------------
* ``single_context_kalman``, ``multidim_kalman`` and ``particle_convergence``
  belong to unit C3. Their imports are guarded, so until C3 merges they are
  reported as ``errored`` with ``error="pending unit C3 merge"`` and the rest of
  the suite still runs. ``particle_convergence`` is fixture-gated inside C3;
  whatever it reports is passed through unchanged.
* ``original_coin_monte_carlo`` compares against the offline ``COIN.m`` oracle,
  which needs frozen MATLAB fixtures (unit D2). It is reported as a SKIPPED
  stage: it carries ``skipped=True`` and a ``skipped_reason``, and it does not
  gate the suite (an unrun check is not evidence of failure, whereas an errored
  one is).
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

if __package__ in (None, ""):        # pragma: no cover - direct-script launch
    # ``python validation/run_validation.py`` puts ``python/validation`` on
    # sys.path, not ``python/``, so the imports below would fail. Prepending the
    # package root makes the direct invocation work as well as the canonical
    # ``python -m validation.run_validation``.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validation import (
    benchmark_realtimecoin,
    validate_context_recovery,
    validate_p_values_extended,
    validate_stress_cases,
)

__all__ = [
    "run",
    "compact_config",
    "format_stage_table",
    "DEFAULT_SEED",
    "STAGE_SEED_OFFSETS",
    "GATING_STAGES",
]

#: Suite seed. Every stage derives its own seed as
#: ``DEFAULT_SEED + STAGE_SEED_OFFSETS[stage]``, matching ``run_validation.m``.
#: Some stages (notably ``context_recovery``) are seed-sensitive, so tests that
#: assert a stage passes must use these values rather than an arbitrary seed.
DEFAULT_SEED = 1001

#: Per-stage seed offsets, verbatim from the ``.m``.
STAGE_SEED_OFFSETS = {
    "single_context_kalman": 10,
    "multidim_kalman": 15,
    "p_values_extended": 20,
    "original_coin_monte_carlo": 30,
    "particle_convergence": 40,
    "context_recovery": 50,
    "stress_cases": 60,
    "performance": 70,
}

#: Stages whose ``passed`` flag gates the suite (the ``.m``'s AND chain). The
#: ``performance`` benchmark is absent because it has no pass flag.
GATING_STAGES = (
    "single_context_kalman",
    "multidim_kalman",
    "p_values_extended",
    "original_coin_monte_carlo",
    "particle_convergence",
    "context_recovery",
    "stress_cases",
)

#: Reason recorded on the COIN.m-oracle stage that cannot run here.
SKIPPED_REASON = "requires frozen MATLAB oracle fixtures (see unit D2)"

#: Error recorded on a stage whose module has not been merged yet.
PENDING_C3 = "pending unit C3 merge"


def compact_config():
    """Sizes of the compact profile.

    Returns
    -------
    dict
        The per-stage trial/particle/dataset counts, verbatim from
        ``compact_config`` in the ``.m``. Pinned here (rather than left to each
        validator's own defaults) so the suite's runtime and its statistical
        power are reproducible.
    """
    return {
        "kalman_trials": 80,
        "kalman_particles": 180,
        "pit_datasets": 18,
        "pit_trials": 70,
        "pit_particles": 100,
        "original_trials": 60,
        "original_particles": 100,
        "convergence_particles": (25, 50, 100),
        "convergence_trials": 45,
        "convergence_datasets": 4,
        "context_trials": 100,
        "context_particles": 120,
        "stress_trials": 75,
        "stress_particles": 80,
        "benchmark_trials": 80,
        "benchmark_particles": (50, 100),
    }


def run(profile="compact", seed=DEFAULT_SEED, strict=False, make_plots=False):
    """Run the compact validation suite.

    Parameters
    ----------
    profile : str, optional
        Suite profile; only ``"compact"`` exists. Default ``"compact"``.
    seed : int, optional
        Suite seed; see :data:`DEFAULT_SEED`.
    strict : bool, optional
        Raise when the suite does not pass. Default ``False``.
    make_plots : bool, optional
        Forwarded to the sub-validators, which currently ignore it (this package
        has no plotting dependency). Default ``False``.

    Returns
    -------
    dict
        One entry per stage (each the sub-validator's results dict, or a
        ``{passed, errored, error}`` / skipped stub), plus the ``p_values`` and
        ``original_comparison`` compatibility aliases, ``config`` and
        ``passed``.

    Raises
    ------
    ValueError
        If ``profile`` is unknown.
    RuntimeError
        If ``strict`` and the suite does not pass.
    """
    if str(profile).lower() != "compact":
        raise ValueError(
            'run_validation:UnknownProfile: unknown validation profile "%s".'
            % profile
        )
    suite = compact_config()
    seed = int(seed)

    results = {
        "config": {
            "profile": profile,
            "make_plots": bool(make_plots),
            "seed": seed,
            "strict": bool(strict),
        }
    }

    results["single_context_kalman"] = _c3_stage(
        "single_context_kalman",
        "validate_single_context_kalman",
        trials=suite["kalman_trials"],
        particles=suite["kalman_particles"],
        seed=seed + STAGE_SEED_OFFSETS["single_context_kalman"],
        make_plots=make_plots,
        strict=False,
    )
    results["multidim_kalman"] = _c3_stage(
        "multidim_kalman",
        "validate_multidim_kalman",
        trials=suite["kalman_trials"],
        particles=suite["kalman_particles"],
        dim=2,
        seed=seed + STAGE_SEED_OFFSETS["multidim_kalman"],
        make_plots=make_plots,
        strict=False,
    )
    results["p_values_extended"] = _run_stage(
        "p_values_extended",
        validate_p_values_extended.run,
        num_datasets=suite["pit_datasets"],
        trials=suite["pit_trials"],
        particles=suite["pit_particles"],
        seed=seed + STAGE_SEED_OFFSETS["p_values_extended"],
        make_plots=make_plots,
        strict=False,
    )
    results["original_coin_monte_carlo"] = {
        "passed": True,
        "skipped": True,
        "skipped_reason": SKIPPED_REASON,
        # Recorded so the stage can be run verbatim once the fixtures exist.
        "seeds": [
            seed + STAGE_SEED_OFFSETS["original_coin_monte_carlo"] + i
            for i in range(5)
        ],
        "trials": suite["original_trials"],
        "particles": suite["original_particles"],
    }
    results["particle_convergence"] = _c3_stage(
        "particle_convergence",
        "validate_particle_convergence",
        particles=suite["convergence_particles"],
        trials=suite["convergence_trials"],
        num_datasets=suite["convergence_datasets"],
        seed=seed + STAGE_SEED_OFFSETS["particle_convergence"],
        strict=False,
    )
    results["context_recovery"] = _run_stage(
        "context_recovery",
        validate_context_recovery.run,
        trials=suite["context_trials"],
        particles=suite["context_particles"],
        seed=seed + STAGE_SEED_OFFSETS["context_recovery"],
        strict=False,
    )
    results["stress_cases"] = _run_stage(
        "stress_cases",
        validate_stress_cases.run,
        trials=suite["stress_trials"],
        particles=suite["stress_particles"],
        seed=seed + STAGE_SEED_OFFSETS["stress_cases"],
        strict=False,
    )
    results["performance"] = _run_stage(
        "performance",
        benchmark_realtimecoin.run,
        trials=suite["benchmark_trials"],
        particles=suite["benchmark_particles"],
        seed=seed + STAGE_SEED_OFFSETS["performance"],
    )

    # Compatibility aliases retained deliberately: existing scripts and
    # notebooks refer to these older names, so they are kept as views onto the
    # current stages rather than removed. They alias the SAME dict objects.
    results["p_values"] = results["p_values_extended"]
    results["original_comparison"] = results["original_coin_monte_carlo"]

    results["passed"] = all(
        bool(results[name].get("passed", False))
        for name in GATING_STAGES
        if not results[name].get("skipped", False)
    )

    print("Compact validation suite passed: %d" % int(results["passed"]))

    if strict and not results["passed"]:
        raise RuntimeError(
            "run_validation:Failed: one or more validation checks failed (%s)."
            % ", ".join(_failed_stages(results))
        )
    return results


def _run_stage(name, fn, **kwargs):
    """Execute one validator, isolating its failure from the rest of the suite.

    Parameters
    ----------
    name : str
        Stage name, used in the diagnostic written to stderr.
    fn : callable
        The validator's ``run`` function.
    **kwargs
        Forwarded to ``fn``.

    Returns
    -------
    dict
        The validator's results dict, or ``{"passed": False, "errored": True,
        "error": ..., "error_type": ...}`` when it raised.
    """
    try:
        return fn(**kwargs)
    except Exception as err:                            # noqa: BLE001
        print('Validator "%s" errored: %s' % (name, err), file=sys.stderr)
        return {
            "passed": False,
            "errored": True,
            "error": str(err),
            "error_type": type(err).__name__,
        }


def _c3_stage(name, module_name, **kwargs):
    """Run a stage whose module belongs to unit C3, tolerating its absence.

    Parameters
    ----------
    name : str
        Stage name.
    module_name : str
        Module under ``validation`` providing ``run``.
    **kwargs
        Forwarded to that ``run``.

    Returns
    -------
    dict
        The validator's results dict, an errored stub, or - when the module
        itself does not exist yet - a stub carrying ``error=PENDING_C3`` so this
        suite runs standalone before C3 merges.

    Notes
    -----
    Only a MISSING stage module counts as "pending C3". A module that exists but
    whose own imports fail is reported as an ordinary errored stage, so a real
    dependency break is never disguised as an unmerged unit.
    """
    target = "validation.%s" % module_name
    try:
        module = importlib.import_module(target)
    except ImportError as err:
        missing = getattr(err, "name", None)
        if missing == target:
            print(
                'Validator "%s" unavailable: %s (%s)' % (name, PENDING_C3, err),
                file=sys.stderr,
            )
            return {
                "passed": False,
                "errored": True,
                "error": PENDING_C3,
                "error_type": type(err).__name__,
                "import_error": str(err),
            }
        print('Validator "%s" failed to import: %s' % (name, err), file=sys.stderr)
        return {
            "passed": False,
            "errored": True,
            "error": str(err),
            "error_type": type(err).__name__,
        }
    return _run_stage(name, module.run, **kwargs)


def _failed_stages(results):
    """Names of the gating stages that did not pass."""
    return [
        name
        for name in GATING_STAGES
        if not results[name].get("skipped", False)
        and not results[name].get("passed", False)
    ] or ["none"]


def format_stage_table(results):
    """Render the suite results as a fixed-width status table.

    Parameters
    ----------
    results : dict
        The dict returned by :func:`run`.

    Returns
    -------
    str
        One line per stage: name, status and either the failing check names or a
        short reason.
    """
    lines = ["%-26s %-9s %s" % ("stage", "status", "detail"), "-" * 72]
    for name in GATING_STAGES + ("performance",):
        stage = results.get(name, {})
        if stage.get("skipped"):
            status, detail = "skipped", stage.get("skipped_reason", "")
        elif stage.get("errored"):
            status, detail = "errored", stage.get("error", "")
        elif "passed" not in stage:
            status, detail = "timed", _benchmark_detail(stage)
        else:
            status = "passed" if stage["passed"] else "FAILED"
            failing = [
                check for check, ok in stage.get("checks", {}).items() if not ok
            ]
            detail = "failing: %s" % ", ".join(failing) if failing else ""
        lines.append("%-26s %-9s %s" % (name, status, detail))
    lines.append("")
    lines.append("suite passed: %s" % results.get("passed"))
    return "\n".join(lines)


def _benchmark_detail(stage):
    """One-line summary of the benchmark stage (which has no pass flag)."""
    particles = stage.get("particles", [])
    seconds = stage.get("seconds_per_trial", [])
    return ", ".join(
        "%d particles: %.1f ms/trial" % (p, 1e3 * s)
        for p, s in zip(particles, seconds)
    )


def _main(argv=None):
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when the suite passed (or when it failed without ``--strict``,
        matching the ``.m``, which only errors under ``Strict``); ``1``
        otherwise.

    Notes
    -----
    ``run`` is always called with ``strict=False`` and the strict exit is
    applied afterwards, so the stage table is printed even on a failing run -
    which is the whole point of running the suite from a shell.
    """
    parser = argparse.ArgumentParser(
        description="Run the compact RealTimeCOIN validation suite."
    )
    parser.add_argument("--profile", default="compact")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args(argv)

    results = run(
        profile=args.profile,
        seed=args.seed,
        strict=False,
        make_plots=args.make_plots,
    )
    print()
    print(format_stage_table(results))
    if args.strict and not results["passed"]:
        print(
            "run_validation:Failed: one or more validation checks failed (%s)."
            % ", ".join(_failed_stages(results)),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(_main())

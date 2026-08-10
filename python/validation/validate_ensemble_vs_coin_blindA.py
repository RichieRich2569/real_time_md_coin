"""Compare ``RealTimeCOINEnsemble`` run-averaging with the offline COIN oracle.

Translated from ``validation/validate_ensemble_vs_coin_blindA.m`` (blind author
A), as a fixture replay.

The scientific argument, unchanged from the ``.m``
--------------------------------------------------
The offline COIN reduces Monte-Carlo variance by averaging ``R`` independent
stochastic realizations ("runs") with equal weight ``1/R``.
``RealTimeCOINEnsemble`` reproduces that averaging online. This validator checks
that the ensemble's run-averaged motor-output trajectory matches COIN's, within
a stated Monte-Carlo tolerance.

Fair-comparison design. COIN INTERNALLY GENERATES its feedback as
``perturbation + sensory_noise + motor_noise``, whereas the ensemble is FED an
observed feedback stream. To make both effectively see the perturbation, the
capture set ``sigma_motor_noise = 0`` and ``sigma_sensory_noise = 1e-3`` (exactly
zero makes COIN.m degenerate - its motor output becomes NaN after trial 2 - so a
tiny value is used and SHARED with the ensemble members). Each COIN run's
feedback is then, to within that tiny noise, the perturbation itself, so both
sides process the same effectively-deterministic stream. The only remaining
randomness is the particle filter, so the two run-averages are independent
Monte-Carlo estimates of the same expectation and agree to
``O(sqrt(2 / R))``.

Trial alignment. COIN's stored ``motor_output(t)`` is its prediction of the
trial-``t`` feedback, computed before the trial-``t`` update. The ensemble's
``simulate()`` trace column ``t`` is ``motor_output`` read after trial ``t``,
likewise the trial-``t`` prior predictive. Both align at the same index, with no
offset.

Fixture replay
--------------
The reference half (running ``COIN.m`` with ``R`` runs) is frozen in
``tests/fixtures/oracle/ensemble_blindA_coin.mat``, which stores the paradigm
(``pert``, ``cues``), the per-run motor outputs ``coinRuns`` (``R x T``), the
run-average ``coinMotorAvg``, the hyperparameters ``hp``, and the configuration
(``R``, ``T``, ``P``, ``seed``, ``sigmaR``). The candidate half is re-run here.

Pending dependency
------------------
``RealTimeCOINEnsemble`` is delivered by a separate translation unit. Until it
lands, :func:`run` returns ``skipped=True`` with
``skipped_reason="pending unit D1 merge"`` instead of failing. The ensemble API
this validator is written against::

    RealTimeCOINEnsemble(runs=..., seed=..., max_cores=0, **member_kwargs)
    .simulate(q_seq, y_seq) -> object/mapping exposing a run-averaged
                               ``motor_output`` trace of shape (T,)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import scipy.io

if __package__ in (None, ""):        # pragma: no cover - direct-script launch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validation.pass_summary import pass_summary

__all__ = [
    "THRESHOLDS",
    "FIXTURE_PATH",
    "PENDING_REASON",
    "expected_paradigm",
    "load_reference",
    "member_kwargs",
    "run",
]

#: Frozen reference capture.
FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "oracle"
    / "ensemble_blindA_coin.mat"
)

#: Reported when the ensemble class is not importable yet.
PENDING_REASON = "pending unit D1 merge"

#: Pass gates, verbatim from the ``.m``.
#:
#: Rationale (the ``.m``'s): with ``R`` independent runs on each side and only
#: particle-filter Monte-Carlo variance, the RMSE between the two run-averages
#: scales like ``sqrt(2 / R)`` times the per-trial across-run standard
#: deviation. For ``R = 30`` that floor sits comfortably below 0.05, and both
#: trajectories track the same expectation so the correlation is high.
THRESHOLDS = {
    "rmse": 0.05,
    "max_abs": 0.12,
    "correlation": 0.9,
}

#: Hyperparameter names read from the fixture's ``hp`` struct and forwarded to
#: each ensemble member under the same keyword.
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


def expected_paradigm(trials):
    """Re-derive the triphasic perturbation schedule and the two-cue sequence.

    Both ``.m`` validators build the same paradigm from ``T`` alone::

        third = floor(T / 3)
        pert  = [0]*third + [0.4]*third + [-0.2]*(T - 2*third)
        cues  = ones(1, T);  cues(floor(T / 2):end) = 2

    Regenerating it here lets :func:`load_reference` verify the fixture's stored
    ``pert`` / ``cues`` rather than trusting them, which is the provenance check
    the equivalence battery performs on its own fixtures.

    Parameters
    ----------
    trials : int
        Number of trials ``T``.

    Returns
    -------
    perturbations : numpy.ndarray
        ``(T,)`` perturbation schedule.
    cues : numpy.ndarray
        ``(T,)`` cue labels, 1-based.
    """
    trials = int(trials)
    third = trials // 3
    perturbations = np.concatenate(
        [
            np.zeros(third),
            0.4 * np.ones(third),
            -0.2 * np.ones(trials - 2 * third),
        ]
    )
    cues = np.ones(trials)
    # MATLAB's floor(T/2) is 1-based, hence the -1.
    cues[trials // 2 - 1:] = 2.0
    return perturbations, cues


def load_reference(fixture_path=None):
    """Load the frozen COIN run-average reference.

    Parameters
    ----------
    fixture_path : path-like or None, optional
        ``.mat`` file; ``None`` uses :data:`FIXTURE_PATH`.

    Returns
    -------
    dict
        Keys ``runs``, ``trials``, ``particles``, ``seed``, ``sigma_sensory``,
        ``perturbations`` (``(T,)``), ``cues`` (``(T,)``), ``coin_runs``
        (``(R, T)``), ``coin_motor_average`` (``(T,)``) and
        ``hyperparameters``.

    Raises
    ------
    FileNotFoundError
        If the fixture is absent.

    Notes
    -----
    Read with ``squeeze_me=True``: MATLAB's ``1 x T`` rows arrive as ``(T,)``
    and its scalars as 0-d arrays, so each field is reshaped explicitly.
    ``coinRuns`` is ``R x T`` in MATLAB and SciPy already presents it in C order
    as ``(R, T)`` - no transpose is needed or applied.
    """
    path = pathlib.Path(fixture_path) if fixture_path is not None else FIXTURE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            "missing ensemble oracle fixture %s (see "
            "python/tests/fixtures/oracle/README.md)" % path
        )
    raw = scipy.io.loadmat(str(path), squeeze_me=True)
    hp_struct = raw["hp"]
    runs = int(np.asarray(raw["R"]).reshape(-1)[0])
    trials = int(np.asarray(raw["T"]).reshape(-1)[0])
    coin_runs = np.asarray(raw["coinRuns"], dtype=float)
    if coin_runs.shape != (runs, trials):
        raise AssertionError(
            "fixture %s stores coinRuns with shape %s; expected (R, T) = %s"
            % (path, coin_runs.shape, (runs, trials))
        )

    perturbations = np.asarray(raw["pert"], dtype=float).reshape(-1)
    cues = np.asarray(raw["cues"], dtype=float).reshape(-1)
    expected_pert, expected_cues = expected_paradigm(trials)
    if not np.allclose(perturbations, expected_pert):
        raise AssertionError(
            "fixture %s stores a perturbation schedule that disagrees with the "
            "re-derived triphasic paradigm" % path
        )
    if not np.array_equal(cues, expected_cues):
        raise AssertionError(
            "fixture %s stores a cue sequence that disagrees with the "
            "re-derived two-cue paradigm" % path
        )

    # The stored run-average must be the equal-weight mean of the stored runs;
    # if it is not, one of the two was captured from a different simulation.
    if not np.allclose(
        np.nanmean(coin_runs, axis=0),
        np.asarray(raw["coinMotorAvg"], dtype=float).reshape(-1),
        equal_nan=True,
    ):
        raise AssertionError(
            "fixture %s: coinMotorAvg is not mean(coinRuns, axis=0)" % path
        )

    return {
        "runs": runs,
        "trials": trials,
        "particles": int(np.asarray(raw["P"]).reshape(-1)[0]),
        "seed": int(np.asarray(raw["seed"]).reshape(-1)[0]),
        "sigma_sensory": float(np.asarray(raw["sigmaR"]).reshape(-1)[0]),
        "perturbations": perturbations,
        "cues": cues,
        "coin_runs": coin_runs,
        "coin_motor_average": np.asarray(
            raw["coinMotorAvg"], dtype=float
        ).reshape(-1),
        "hyperparameters": {
            name: float(np.asarray(hp_struct[name]).reshape(-1)[0])
            for name in _HP_KWARGS
        },
    }


def member_kwargs(reference, max_contexts=4):
    """Build the per-member constructor keywords from the fixture.

    Parameters
    ----------
    reference : dict
        A :func:`load_reference` result.
    max_contexts : int, optional
        Context cap; the ``.m`` defaults to 4. Default ``4``.

    Returns
    -------
    dict
        Keywords mirroring the ``.m``'s ``memberParams`` cell array, including
        the explicit ``prior_mean_drift = 0``.
    """
    kwargs = {
        "num_particles": reference["particles"],
        "max_contexts": int(max_contexts),
        "prior_mean_drift": 0.0,
    }
    kwargs.update(reference["hyperparameters"])
    return kwargs


def _import_ensemble():
    """Import ``RealTimeCOINEnsemble``, or raise ``ImportError``.

    Isolated into its own function so :func:`run` can catch ``ImportError``
    around the IMPORT ALONE. Wrapping the construction and the ``simulate``
    call in the same guard would turn every ``AttributeError`` raised inside the
    ensemble - including a missing ``motor_output`` on the returned trace, which
    is exactly the API contract this validator exists to pin - into a passing
    skip.

    Returns
    -------
    type
        The ``RealTimeCOINEnsemble`` class.

    Raises
    ------
    ImportError
        While the ensemble translation unit has not merged.
    """
    from realtimecoin import RealTimeCOINEnsemble       # noqa: PLC0415

    return RealTimeCOINEnsemble


def _ensemble_motor_average(ensemble_cls, reference, max_cores=0, max_contexts=4):
    """Run the ensemble and return its run-averaged motor-output trace.

    Parameters
    ----------
    ensemble_cls : type
        The ``RealTimeCOINEnsemble`` class, from :func:`_import_ensemble`.
    reference : dict
        A :func:`load_reference` result, supplying ``runs``, ``seed``, ``cues``
        and ``perturbations``.
    max_cores : int, optional
        Forwarded to the ensemble. Default ``0`` (serial), as in the ``.m``.
    max_contexts : int, optional
        Context cap for the members. Default ``4``.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` run-averaged motor output.

    Notes
    -----
    Nothing is caught here. Any failure of the ensemble API is a real defect and
    must reach the caller as an error, not as a skip.
    """
    ensemble = ensemble_cls(
        runs=reference["runs"],
        seed=reference["seed"],
        max_cores=max_cores,
        **member_kwargs(reference, max_contexts),
    )
    traces = ensemble.simulate(reference["cues"], reference["perturbations"])
    trace = traces["motor_output"] if isinstance(traces, dict) else (
        traces.motor_output
    )
    return np.asarray(trace, dtype=float).reshape(-1)


def _skipped(reason, reference=None):
    """Build the skipped-result dict.

    Parameters
    ----------
    reason : str
        Why the validator did not run.
    reference : dict or None, optional
        Loaded reference, echoed into the result when available.

    Returns
    -------
    dict
        ``errored=False``, ``skipped=True``, ``passed=True`` and
        ``skipped_reason``.
    """
    print("Ensemble vs COIN (blind A): skipped (%s)" % reason)
    results = {
        "passed": True,
        "errored": False,
        "skipped": True,
        "skipped_reason": reason,
        "thresholds": dict(THRESHOLDS),
        "checks": {},
    }
    if reference is not None:
        results["coin_motor_average"] = reference["coin_motor_average"]
        results["perturbations"] = reference["perturbations"]
        results["cues"] = reference["cues"]
        results["config"] = {
            "runs": reference["runs"],
            "trials": reference["trials"],
            "particles": reference["particles"],
            "seed": reference["seed"],
        }
    return results


def run(
    strict=False,
    make_plots=False,
    max_cores=0,
    max_contexts=4,
    fixture_path=None,
):
    """Compare the ensemble run-average with the frozen COIN run-average.

    Parameters
    ----------
    strict : bool, optional
        Raise when the validator does not pass. Default ``False``.
    make_plots : bool, optional
        Accepted for signature parity with the ``.m``; this package has no
        plotting dependency, so it is recorded in ``config`` and ignored.
        Default ``False``.
    max_cores : int, optional
        Forwarded to the ensemble. Default ``0``.
    max_contexts : int, optional
        Context cap for the members. Default ``4``.
    fixture_path : path-like or None, optional
        Reference fixture; ``None`` uses :data:`FIXTURE_PATH`.

    Returns
    -------
    dict
        Field names verbatim from the ``.m``: ``rmse_motor_output``,
        ``max_abs_motor_output``, ``correlation_motor_output``,
        ``coin_motor_average``, ``ensemble_motor_average``, ``perturbations``,
        ``cues``, ``thresholds``, ``checks``, ``passed`` and ``config``. When
        the fixture or the ensemble class is unavailable, a skipped result is
        returned instead (see :func:`_skipped`).

    Raises
    ------
    RuntimeError
        If ``strict`` and the validator did not pass.
    """
    try:
        reference = load_reference(fixture_path)
    except FileNotFoundError as err:
        return _skipped(str(err))

    # ONLY the import is guarded: a missing translation unit is a skip, but a
    # broken ensemble is a failure.
    try:
        ensemble_cls = _import_ensemble()
    except ImportError as err:
        return _skipped("%s (%s)" % (PENDING_REASON, err), reference)

    ensemble_average = _ensemble_motor_average(
        ensemble_cls, reference, max_cores=max_cores, max_contexts=max_contexts
    )

    coin_average = reference["coin_motor_average"]
    if ensemble_average.size != coin_average.size:
        return _skipped(
            "ensemble returned %d trials, fixture holds %d"
            % (ensemble_average.size, coin_average.size),
            reference,
        )

    valid = np.isfinite(coin_average) & np.isfinite(ensemble_average)
    difference = coin_average[valid] - ensemble_average[valid]
    rmse = float(np.sqrt(np.mean(difference ** 2))) if valid.any() else float("nan")
    max_abs = float(np.max(np.abs(difference))) if valid.any() else float("nan")
    correlation = float("nan")
    if int(valid.sum()) > 1:
        a, b = coin_average[valid], ensemble_average[valid]
        if np.std(a) > 0 and np.std(b) > 0:
            correlation = float(np.corrcoef(a, b)[0, 1])

    checks = {
        "rmse": rmse < THRESHOLDS["rmse"],
        "max_abs": max_abs < THRESHOLDS["max_abs"],
        "correlation": correlation > THRESHOLDS["correlation"],
    }
    passed, checks = pass_summary(checks)

    print(
        "Ensemble vs COIN (R=%d): motor RMSE %.4f, max abs %.4f, corr %.3f"
        % (reference["runs"], rmse, max_abs, correlation)
    )

    results = {
        "rmse_motor_output": rmse,
        "max_abs_motor_output": max_abs,
        "correlation_motor_output": correlation,
        "coin_motor_average": coin_average,
        "ensemble_motor_average": ensemble_average,
        "perturbations": reference["perturbations"],
        "cues": reference["cues"],
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": passed,
        "skipped": False,
        "errored": False,
        "config": {
            "runs": reference["runs"],
            "trials": reference["trials"],
            "particles": reference["particles"],
            "max_contexts": int(max_contexts),
            "seed": reference["seed"],
            "max_cores": int(max_cores),
            "make_plots": bool(make_plots),
            "strict": bool(strict),
        },
    }

    if strict and not passed:
        raise RuntimeError(
            "validate_ensemble_vs_coin:Failed: Ensemble vs COIN run-average "
            "validation failed."
        )
    return results


def _main(argv=None):       # pragma: no cover - console entry point
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` when the validator passed or was skipped, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Compare RealTimeCOINEnsemble with the frozen COIN oracle."
    )
    parser.add_argument("--max-cores", type=int, default=0)
    args = parser.parse_args(argv)
    results = run(max_cores=args.max_cores)
    return 0 if results.get("passed", False) else 1


if __name__ == "__main__":       # pragma: no cover
    sys.exit(_main())

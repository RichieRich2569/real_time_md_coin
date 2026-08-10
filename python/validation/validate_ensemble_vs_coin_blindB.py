"""Run-averaged ``RealTimeCOINEnsemble`` versus the offline COIN oracle.

Translated from ``validation/validate_ensemble_vs_coin_blindB.m`` (blind author
B), as a fixture replay.

Independence note. Authors A and B wrote their MATLAB validators blind to each
other, and their SCIENTIFIC content is kept independent here: different
paradigm size (``R = 16``, ``T = 90``, ``P = 80`` versus A's 30/75/100),
different gates, and B's distinctive two-alignment analysis. What IS shared is
the fixture PARSING (:func:`validation.validate_ensemble_vs_coin_blindA.
load_reference` and ``member_kwargs``), because both captures use the identical
``.mat`` layout and duplicating a loader would create two places for a
shape bug to hide. No threshold, metric or alignment decision is imported.

Fairness of the comparison (from the ``.m``)
--------------------------------------------
``COIN.m`` GENERATES its feedback internally as ``perturbation +
sensory_noise + motor_noise``, whereas the ensemble is FED the feedback ``y``.
To make both see the same effective feedback the capture:

* set COIN's ``sigma_motor_noise = 0`` and ``sigma_sensory_noise = 1e-3``, so
  each COIN run's generated feedback is, to within that tiny noise, the
  perturbation;
* fed the ensemble the perturbation itself as the observed feedback ``y``;
* gave the ensemble members COIN's hyperparameters, including the same tiny
  ``sigma_sensory_noise``.

The two then differ only by the Monte-Carlo error of averaging ``R``
independent process-noise realizations.

Alignment assumption (underdetermined by the spec)
--------------------------------------------------
COIN stores, at trial ``t``, the pre-feedback predicted motor output for trial
``t``. The ensemble's Phase-1 query surface exposes only ``motor_output`` read
AFTER each trial - there is no cue-conditioned predictive query. The exact
trial-phase correspondence is therefore not fixed by the spec, so this validator
evaluates BOTH natural alignments - lag 0 (same index) and lag 1 (the ensemble
leads COIN by one trial) - reports both, and gates on the better-fitting one.

Availability guard
------------------
``RealTimeCOINEnsemble`` is live, so this validator runs and gates for real. The
import guard around it is kept as a safety net only: a MISSING ensemble class (or
a missing fixture) yields ``skipped=True`` rather than a spurious failure, while
a present-but-broken ensemble fails. The pinned API::

    RealTimeCOINEnsemble(runs=..., seed=..., max_cores=0, **member_kwargs)
    .simulate(q_seq, y_seq) -> object/mapping exposing a run-averaged
                               ``motor_output`` trace of shape (T,)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

if __package__ in (None, ""):        # pragma: no cover - direct-script launch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validation.pass_summary import pass_summary
from validation.validate_ensemble_vs_coin_blindA import (
    _import_ensemble,
    load_reference,
    member_kwargs,
)

__all__ = ["THRESHOLDS", "FIXTURE_PATH", "PENDING_REASON", "run"]

#: Frozen reference capture for author B's paradigm.
FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "oracle"
    / "ensemble_blindB_coin.mat"
)

#: Reported if the ensemble class is ever unimportable (safety net only:
#: RealTimeCOINEnsemble ships with the package).
PENDING_REASON = "RealTimeCOINEnsemble unavailable"

#: Pass gates, verbatim from the ``.m``.
#:
#: Rationale (the ``.m``'s): both estimators average ``R`` independent runs
#: whose feedback is, to within ``sigmaR ~ 1e-3``, the same deterministic
#: perturbation, so they estimate the same posterior-mean trajectory and differ
#: only by particle-filter / run-averaging Monte-Carlo error. The gates are
#: looser than the implementation-versus-implementation gate in
#: ``validate_original_coin_monte_carlo`` (RMSE 0.03) because here the two sides
#: use INDEPENDENT RNG streams and independent run sets rather than a shared
#: feedback stream.
THRESHOLDS = {
    "rmse": 0.08,
    "correlation": 0.85,
}


def _fit(a, b):
    """RMSE and Pearson correlation over the finite overlap of two traces.

    Transliterated from ``local_fit`` in the ``.m``.

    Parameters
    ----------
    a, b : array_like
        Equal-length traces.

    Returns
    -------
    rmse : float
        Root-mean-square difference over the finite overlap, or ``nan``.
    correlation : float
        Pearson correlation over the same overlap, or ``nan`` when fewer than
        two points overlap or either sample is constant.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))
    correlation = float("nan")
    if int(valid.sum()) > 1 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0:
        correlation = float(np.corrcoef(a[valid], b[valid])[0, 1])
    return rmse, correlation


def _ensemble_motor_average(ensemble_cls, reference, max_cores=0, max_contexts=4):
    """Run the ensemble on B's paradigm and return its run-averaged trace.

    Parameters
    ----------
    ensemble_cls : type
        The ``RealTimeCOINEnsemble`` class.
    reference : dict
        A :func:`validation.validate_ensemble_vs_coin_blindA.load_reference`
        result.
    max_cores : int, optional
        Forwarded to the ensemble. Default ``0``.
    max_contexts : int, optional
        Context cap for the members; the ``.m`` fixes 4. Default ``4``.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` run-averaged motor output.

    Notes
    -----
    Nothing is caught here: only the class IMPORT is treated as a skippable
    dependency, so a broken ensemble API surfaces as a failure rather than as a
    passing skip.
    """
    ensemble = ensemble_cls(
        runs=reference["runs"],
        seed=reference["seed"],
        max_cores=max_cores,
        **member_kwargs(reference, max_contexts),
    )
    # The perturbation IS the observed feedback fed to the ensemble.
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
    print("Ensemble vs COIN (blind B): skipped (%s)" % reason)
    results = {
        "passed": True,
        "errored": False,
        "skipped": True,
        "skipped_reason": reason,
        "thresholds": dict(THRESHOLDS),
        "checks": {},
    }
    if reference is not None:
        results["coin_run_average"] = reference["coin_motor_average"]
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
    """Compare the ensemble run-average with COIN's, under both alignments.

    Parameters
    ----------
    strict : bool, optional
        Raise when the validator does not pass. Default ``False``.
    make_plots : bool, optional
        Accepted for signature parity with the ``.m`` and ignored (no plotting
        dependency). Default ``False``.
    max_cores : int, optional
        Forwarded to the ensemble. Default ``0``.
    max_contexts : int, optional
        Context cap for the members. Default ``4``.
    fixture_path : path-like or None, optional
        Reference fixture; ``None`` uses :data:`FIXTURE_PATH`.

    Returns
    -------
    dict
        Field names verbatim from the ``.m``: ``rmse``, ``correlation``,
        ``rmse_lag0``, ``rmse_lag1``, ``correlation_lag0``,
        ``correlation_lag1``, ``chosen_alignment_lag``, ``coin_run_average``,
        ``ensemble_run_average``, ``perturbations``, ``cues``, ``thresholds``,
        ``checks``, ``passed`` and ``config``. A skipped result is returned when
        the fixture or the ensemble class is unavailable.

    Raises
    ------
    RuntimeError
        If ``strict`` and the validator did not pass.
    """
    try:
        reference = load_reference(
            FIXTURE_PATH if fixture_path is None else fixture_path
        )
    except FileNotFoundError as err:
        return _skipped(str(err))

    # ONLY the import is guarded; see _ensemble_motor_average.
    try:
        ensemble_cls = _import_ensemble()
    except ImportError as err:
        return _skipped("%s (%s)" % (PENDING_REASON, err), reference)

    ensemble_motor = _ensemble_motor_average(
        ensemble_cls, reference, max_cores=max_cores, max_contexts=max_contexts
    )

    coin_average = reference["coin_motor_average"]
    trials = reference["trials"]
    if ensemble_motor.size != trials:
        return _skipped(
            "ensemble returned %d trials, fixture holds %d"
            % (ensemble_motor.size, trials),
            reference,
        )

    # Lag 0: same index. Lag 1: the ensemble leads COIN by one trial, so
    # coin[1:] is matched against ensemble[:-1] (MATLAB's coinAvg(2:T) versus
    # ensMotor(1:T-1), 1-based).
    rmse0, corr0 = _fit(coin_average, ensemble_motor)
    rmse1, corr1 = _fit(coin_average[1:trials], ensemble_motor[0:trials - 1])

    # NaN-safe choice: a NaN rmse1 must not win the comparison, and `<` is
    # already False against NaN, so lag 0 is the default exactly as in the .m.
    if rmse1 < rmse0:
        best_rmse, best_corr, alignment = rmse1, corr1, 1
    else:
        best_rmse, best_corr, alignment = rmse0, corr0, 0

    checks = {
        "rmse": best_rmse < THRESHOLDS["rmse"],
        "correlation": best_corr > THRESHOLDS["correlation"],
    }
    passed, checks = pass_summary(checks)

    print(
        "Ensemble vs COIN: RMSE %.4f (lag %d), corr %.3f "
        "[lag0 rmse %.4f, lag1 rmse %.4f]"
        % (best_rmse, alignment, best_corr, rmse0, rmse1)
    )

    results = {
        "rmse": best_rmse,
        "correlation": best_corr,
        "rmse_lag0": rmse0,
        "rmse_lag1": rmse1,
        "correlation_lag0": corr0,
        "correlation_lag1": corr1,
        "chosen_alignment_lag": alignment,
        "coin_run_average": coin_average,
        "ensemble_run_average": ensemble_motor,
        "perturbations": reference["perturbations"],
        "cues": reference["cues"],
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": passed,
        "skipped": False,
        "errored": False,
        "config": {
            "runs": reference["runs"],
            "trials": trials,
            "particles": reference["particles"],
            "max_contexts": int(max_contexts),
            "seed": reference["seed"],
            "sigma_sensory": reference["sigma_sensory"],
            "max_cores": int(max_cores),
            "make_plots": bool(make_plots),
            "strict": bool(strict),
        },
    }

    if strict and not passed:
        raise RuntimeError(
            "validate_ensemble_vs_coin:Failed: Ensemble vs COIN validation "
            "failed."
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

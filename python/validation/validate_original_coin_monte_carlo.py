"""Multi-seed comparison with the offline COIN oracle.

Translated from ``validation/validate_original_coin_monte_carlo.m``.

This validator does not prove the model correct from first principles - that is
what the Kalman validators do. It checks that the streaming ``RealTimeCOIN``
produces the same predictive motor outputs as the original offline ``COIN.m``
over several random seeds, on the feedback stream COIN itself generated.

Fixture replay
--------------
MATLAB re-runs ``COIN.m`` per seed. Here each seed's trace is a frozen fixture
(``tests/fixtures/oracle/coin_trace_seed<seed>.mat``) replayed by
:func:`validation.compare_original_coin.run`. When the fixtures are absent the
validator reports ``skipped=True`` with ``passed=True`` rather than failing: an
unrun check is not evidence of a defect.

The consequence is that ``seeds`` selects among the FROZEN traces; it cannot
name an arbitrary integer the way the ``.m`` can. ``seeds=None`` (the default)
uses every trace present.

Gates
-----
Verbatim from the ``.m``::

    mean_rmse         < 0.03
    worst_correlation > 0.95

The rationale is the ``.m``'s: this is an implementation-versus-implementation
comparison on an IDENTICAL feedback stream, so near-exact agreement is expected
and the gate is tighter than the Kalman validators' 0.05.

One honest caveat about the cross-language setting. In MATLAB both COIN and
RealTimeCOIN are seeded from the same ``rng(seed)`` call, so their particle
filters share a random stream and the agreement is partly common-random-number
agreement. Python cannot reproduce MATLAB's stream, so here the two filters are
INDEPENDENTLY randomised and the residual is pure Monte-Carlo error at
``P = 100`` particles. That makes the Python numbers a strictly harder test of
the same gate, not an easier one - the gate is left untouched.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

if __package__ in (None, ""):        # pragma: no cover - direct-script launch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validation import compare_original_coin
from validation.pass_summary import pass_summary

__all__ = ["THRESHOLDS", "run"]

#: Pass gates, verbatim from the ``.m``.
THRESHOLDS = {
    "mean_rmse": 0.03,
    "worst_correlation": 0.95,
}


def _percentile(values, percent):
    """Linearly-interpolated percentile of the finite entries.

    Parameters
    ----------
    values : array_like
        Sample; non-finite entries are dropped.
    percent : float
        Percentile in ``[0, 100]``.

    Returns
    -------
    float
        The percentile, or ``nan`` when nothing is finite.

    Notes
    -----
    Transliterated from ``local_percentile`` in the ``.m`` (sort, then index
    ``1 + (n - 1) * p / 100`` with linear interpolation), which is the same rule
    as ``numpy.percentile``'s default ``method="linear"``. Written out rather
    than delegated so the correspondence with the ``.m`` stays visible.
    """
    sample = np.sort(np.asarray(values, dtype=float))
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan")
    index = 1.0 + (sample.size - 1) * float(percent) / 100.0
    low = int(np.floor(index))
    high = int(np.ceil(index))
    if low == high:
        return float(sample[low - 1])
    return float(
        sample[low - 1] + (index - low) * (sample[high - 1] - sample[low - 1])
    )


def run(
    seeds=None,
    trials=None,
    particles=None,
    strict=False,
    fixture_dir=None,
):
    """Score ``RealTimeCOIN`` against the frozen offline-COIN traces.

    Parameters
    ----------
    seeds : sequence of int or None, optional
        Trace seeds to replay; ``None`` uses every fixture present.
    trials : int or None, optional
        Cross-check forwarded to :func:`validation.compare_original_coin.run`.
        The frozen traces hold ``T = 60``, matching the compact suite; a
        different value raises rather than silently comparing a resized
        paradigm.
    particles : int or None, optional
        Cross-check on the fixtures' particle count, likewise forwarded.
    strict : bool, optional
        Raise when the validator does not pass. Default ``False``.
    fixture_dir : path-like or None, optional
        Directory holding the traces; ``None`` uses
        :data:`validation.compare_original_coin.ORACLE_FIXTURE_DIR`.

    Returns
    -------
    dict
        Field names verbatim from the ``.m``: ``seeds``, ``rmse``,
        ``correlation``, ``mean_rmse``, ``median_rmse``, ``percentile95_rmse``,
        ``mean_correlation``, ``worst_correlation``, ``rmse_motor_output``,
        ``correlation_motor_output``, ``thresholds``, ``passed``, ``checks``
        and ``config``. When no fixture is available the dict instead carries
        ``passed=True``, ``skipped=True`` and ``skipped_reason``.

    Raises
    ------
    RuntimeError
        If ``strict`` and the validator did not pass.
    """
    available = compare_original_coin.available_seeds(fixture_dir)
    requested = (
        list(available) if seeds is None else [int(s) for s in np.ravel(seeds)]
    )
    missing = [s for s in requested if s not in available]

    if not requested or missing:
        reason = (
            "offline-COIN trace fixtures missing (%s) in %s; the oracle "
            "comparison needs traces exported from MATLAB - see "
            "python/tests/fixtures/oracle/README.md"
            % (
                ", ".join(str(s) for s in missing) or "none available",
                fixture_dir
                if fixture_dir is not None
                else compare_original_coin.ORACLE_FIXTURE_DIR,
            )
        )
        print("Original COIN Monte Carlo: skipped (%s)" % reason)
        return {
            "passed": True,
            "skipped": True,
            "skipped_reason": reason,
            "seeds": requested,
            "available_seeds": available,
            "thresholds": dict(THRESHOLDS),
            "checks": {},
            "config": {
                "seeds": requested,
                "trials": trials,
                "particles": particles,
                "strict": bool(strict),
            },
        }

    rmse = np.zeros(len(requested))
    correlation = np.zeros(len(requested))
    for i, seed in enumerate(requested):
        one = compare_original_coin.run(
            seed=seed,
            trials=trials,
            particles=particles,
            fixture_dir=fixture_dir,
        )
        rmse[i] = one["rmse_motor_output"]
        correlation[i] = one["correlation_motor_output"]

    mean_rmse = float(np.nanmean(rmse))
    mean_correlation = float(np.nanmean(correlation))
    # np.min, which PROPAGATES nan, unlike MATLAB's min, which omits it. This is
    # a deliberate divergence from the .m: a correlation that could not be
    # computed is not evidence of agreement, so it must sink the gate rather
    # than be skipped over.
    worst_correlation = float(np.min(correlation))

    checks = {
        "mean_rmse": mean_rmse < THRESHOLDS["mean_rmse"],
        "worst_correlation": worst_correlation > THRESHOLDS["worst_correlation"],
    }
    passed, checks = pass_summary(checks)

    results = {
        "seeds": list(requested),
        "rmse": rmse,
        "correlation": correlation,
        "mean_rmse": mean_rmse,
        "median_rmse": float(np.nanmedian(rmse)),
        "percentile95_rmse": _percentile(rmse, 95),
        "mean_correlation": mean_correlation,
        "worst_correlation": worst_correlation,
        "rmse_motor_output": mean_rmse,
        "correlation_motor_output": mean_correlation,
        "thresholds": dict(THRESHOLDS),
        "passed": passed,
        "checks": checks,
        "skipped": False,
        "config": {
            "seeds": list(requested),
            "trials": trials,
            "particles": particles,
            "strict": bool(strict),
        },
    }

    print(
        "Original COIN Monte Carlo: mean RMSE %.4f, 95th RMSE %.4f, "
        "worst corr %.3f"
        % (mean_rmse, results["percentile95_rmse"], worst_correlation)
    )

    if strict and not passed:
        raise RuntimeError(
            "validate_original_coin_monte_carlo:Failed: Original COIN Monte "
            "Carlo validation failed."
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
        description="Compare RealTimeCOIN with the frozen offline-COIN traces."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="trace seeds to replay (default: every fixture present)",
    )
    args = parser.parse_args(argv)
    results = run(seeds=args.seeds)
    if results.get("skipped"):
        return 0
    print()
    print("%-8s %10s %12s" % ("seed", "rmse", "correlation"))
    for seed, value, corr in zip(
        results["seeds"], results["rmse"], results["correlation"]
    ):
        print("%-8d %10.5f %12.5f" % (seed, value, corr))
    print()
    print(
        "mean rmse %.5f (gate < %.3f), worst corr %.5f (gate > %.3f) -> %s"
        % (
            results["mean_rmse"],
            THRESHOLDS["mean_rmse"],
            results["worst_correlation"],
            THRESHOLDS["worst_correlation"],
            "PASS" if results["passed"] else "FAIL",
        )
    )
    return 0 if results["passed"] else 1


if __name__ == "__main__":       # pragma: no cover
    sys.exit(_main())

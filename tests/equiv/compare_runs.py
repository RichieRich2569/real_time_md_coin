"""Compare Python and MATLAB equivalence-battery run-averages.

Replacement for ``tests/+equiv/compareRuns.m``.

Why this is not the MATLAB comparison
-------------------------------------
``compareRuns.m`` is a BIT-IDENTITY detector: it diffs two MATLAB captures and
reports ``classA`` when the max absolute difference is exactly zero. That test
is meaningless across languages. MATLAB draws from a Mersenne-Twister stream and
this package draws from NumPy's PCG64; there is no seeding that makes them emit
the same variates, and the COIN particle filter consumes randomness at every one
of its nine pipeline steps. A single differing variate on trial 1 re-labels
contexts, reorders particles and changes every subsequent draw.

What CAN be compared is the DISTRIBUTION the two implementations sample from.
Both sides replay the identical ``(q, y)`` input stream under ``R = 20``
independent model seeds; the two run-averages are then two independent
Monte-Carlo estimates of the same expectation, and they must agree to within
Monte-Carlo error.

The band
--------
For every trial ``t`` and every entry ``i`` of every quantity::

    |py_mean[i, t] - matlab_mean[i, t]|  <=  3 * SE[i, t] + FLOORS[quantity]

    SE[i, t] = sqrt( matlab_std[i, t]^2 / R_matlab + py_std[i, t]^2 / R_py )

``SE`` is the exact standard error of the DIFFERENCE of two independent sample
means (the Welch form, with no equal-variance assumption). Both standard
deviations are population estimates (``ddof=0``), matching how the two means are
formed.

Both terms are needed. An earlier version of this band used
``3 * matlab_std / sqrt(R)`` alone, on the theory that the two sides run the
same model on identical inputs so one side's spread stands in for both. That is
wrong twice over, and measurably so:

* it omits the Python side's sampling error, making the band a factor ``sqrt(2)``
  too narrow even under equal variances - and the variances are NOT equal here
  (the Python across-seed spread on the context quantities runs ~35% larger than
  the fixture's), so the shortfall is worse than ``sqrt(2)``;
* it is a one-sided estimate from only 20 samples, so wherever the 20 MATLAB
  seeds happened to agree the band collapsed onto the floor even though the
  underlying quantity is genuinely variable.

The correction was validated with a NULL CONTROL: comparing 20 Python seeds
against 20 DIFFERENT Python seeds - the same code, so the only difference is
Monte-Carlo noise - through this exact band math. Under the two-sample ``SE``
the null needs no floor at all (no entry of any quantity exceeds ``3 * SE``),
whereas under the old one-sided band the same null would have required floors of
0.002-0.027. In other words the old band failed a comparison of the
implementation with ITSELF; the new one does not.

``FLOORS`` then keeps the band away from zero where both spreads collapse - the
early trials, and the many sorted-tail context entries that are identically zero
in every seed. Without a floor those entries would demand exact equality, which
is what cannot hold across languages.

Floor values and their rationale
--------------------------------
Each floor is set from the MEASURED requirement - the largest
``deviation - 3 * SE`` over the whole battery, reported per quantity as
``required_floor`` - with a stated safety margin. Nothing here was tuned until
the tests went green; the numbers below are the measurements.

``motor`` / ``state_mean`` : ``2.5e-3``
    Both are in feedback units. Largest measured requirement is ``1.0e-3``
    (``scalar_missing``), so this carries a 2.5x margin. In the
    lowest-amplitude scenario (``scalar_2ctx``, whose run-averaged motor output
    peaks at 0.05) the floor is 5% of signal range - but that scenario's own
    requirement is ``1e-4``, so there the floor is pure slack and the sampling
    term is what the comparison rests on. (These two quantities coincide
    numerically - see :data:`VALUE_FIELDS`.)
``pred_ctx`` / ``resp`` / ``counts`` : ``4e-2``
    All three are probability/occupancy vectors in ``[0, 1]``, and all three are
    near one-hot per seed: a single context usually takes almost all the mass,
    so the 20-seed run-average of each entry is heavy-tailed rather than
    Gaussian. A heavy-tailed mean makes a three-sigma band under-cover in the
    tail, and it makes the 20-sample standard-deviation estimate itself noisy.
    Largest measured requirements are ``1.03e-2`` (``pred_ctx``,
    ``scalar_2ctx``), ``4.1e-3`` (``resp``) and ``1.62e-2`` (``counts``,
    ``scalar_missing``), so ``4e-2`` carries a 2.5-10x margin. It stays small
    against the ``3 * SE`` sampling term, which peaks at 0.07-0.18 per
    scenario and dominates the band nearly everywhere.

    ``counts`` gets the SAME floor as the other two. It is a normalised
    occupancy FRACTION in ``[0, 1]``, not an integer count, so there is no
    reason to treat it differently - and a large floor here would be
    particularly damaging, since ``counts`` never has a sampling term above
    0.18 and any floor near that value makes the check vacuous.

Is the cross-language deviation real?
-------------------------------------
The context quantities deviate by up to ~0.077 between the languages, which
looks alarming until it is compared with the null control. Across the six
scenarios, 20-Python-seeds versus 20 DIFFERENT Python seeds produces maximum
deviations of 0.026-0.108 - in four of the six scenarios AS LARGE AS OR LARGER
THAN the Python-versus-MATLAB deviation (e.g. ``md2_basic`` ``resp``: null
0.108 versus cross-language 0.060). Averaging 20 seeds simply does not pin these
near-one-hot quantities down more tightly than that. The deviations are
Monte-Carlo noise, not a port defect.

The signed per-slot bias tells the same story: Python's top sorted slot averages
+0.019 relative to MATLAB in ``scalar_2ctx``, against a per-trial standard error
of ~0.034 for a single entry.

Context labels are arbitrary
----------------------------
Context labels are per-particle and per-run; the global alignment makes them
consistent WITHIN a run but nothing ties run 3's "context 2" to run 7's, let
alone to MATLAB's. So the three context-indexed quantities are compared as
SORTED-DESCENDING vectors: each ``(context,)`` slice is sorted per trial per
seed BEFORE averaging.

This tests the MULTISET of context masses (how mass is distributed over however
many contexts are active), not label identity. A permutation of the contexts is
invisible to it, by design. What it still catches: a wrong number of occupied
contexts, mass concentrated in too few or too many, or a systematically
different novel-context probability.

NaN pads
--------
The fixture stores ``L = max_contexts + 2`` context slots and NaN-pads the ones
the MATLAB query did not return. NaN means "no context here", i.e. ZERO mass, so
pads are zero-filled before sorting. The Python vectors are zero-padded to the
same length so the two sorted vectors are directly comparable.
"""

from __future__ import annotations

import numpy as np

# Relative imports so this package works whether it is reached as
# ``tests.equiv.compare_runs`` (the repo root on sys.path, the pyproject
# ``pythonpath`` setting) or as ``equiv.compare_runs`` (tests/ on sys.path,
# which is how pytest's rootdir insertion makes ``helpers`` importable).
from .capture_run import capture_run
from .scenario_battery import SCENARIO_NAMES, load_scenario

__all__ = [
    "PARITY_RUNS",
    "PARITY_BASE_SEED",
    "BAND_SIGMAS",
    "FLOORS",
    "CONTEXT_FIELDS",
    "VALUE_FIELDS",
    "python_seeds",
    "python_run_average",
    "matlab_run_average",
    "compare_scenario",
    "compare_all",
    "format_report",
]

#: Number of independent model seeds averaged on the Python side. Matches the
#: 20 MATLAB seeds stored in every fixture, so the two standard errors match.
PARITY_RUNS = 20

#: Root entropy for the Python seed derivation. Fixed so a parity failure is
#: reproducible from the scenario name alone.
PARITY_BASE_SEED = 20250810

#: Width of the Monte-Carlo band, in standard errors of the DIFFERENCE of the
#: two run-averages.
BAND_SIGMAS = 3.0

#: Additive band floors per quantity; see the module docstring for the rationale
#: and for the null-control measurements they are derived from.
FLOORS = {
    "motor": 2.5e-3,
    "state_mean": 2.5e-3,
    "pred_ctx": 4e-2,
    "resp": 4e-2,
    "counts": 4e-2,
}

#: Quantities compared after sorting their context axis descending.
CONTEXT_FIELDS = ("pred_ctx", "resp", "counts")

#: Quantities compared entry-wise in their natural (state-dimension) order.
#:
#: These two coincide numerically in this model: ``motor_output()`` is the
#: predictive FEEDBACK mean and ``state_moments()[0]`` the predictive STATE
#: mean, and with no bias term and an identity observation map the two agree to
#: rounding (measured <= 7e-16 in every fixture). They are both recorded because
#: the MATLAB capture records both and they are distinct code paths, but they do
#: not constitute two independent checks.
VALUE_FIELDS = ("motor", "state_mean")


def python_seeds(scenario, runs=PARITY_RUNS, base_seed=PARITY_BASE_SEED):
    """Derive the Python model seeds for one scenario.

    Parameters
    ----------
    scenario : dict
        A :func:`tests.equiv.scenario_battery.load_scenario` result; its
        ``seed`` (the MATLAB model seed) is used purely as stable per-scenario
        entropy.
    runs : int, optional
        Number of seeds. Default :data:`PARITY_RUNS`.
    base_seed : int, optional
        Root entropy. Default :data:`PARITY_BASE_SEED`.

    Returns
    -------
    list of numpy.random.SeedSequence
        Independent child sequences, safe to hand to ``numpy.random.default_rng``.

    Notes
    -----
    These deliberately do NOT match the MATLAB seeds stored in the fixture. The
    two languages' streams cannot be aligned, so matching integers would only be
    misleading; what matters is that the seeds are independent and reproducible.
    """
    return np.random.SeedSequence([int(base_seed), int(scenario["seed"])]).spawn(
        int(runs)
    )


def _sorted_context(array):
    """Zero-fill NaN pads and sort each context slice descending.

    Parameters
    ----------
    array : numpy.ndarray
        ``(L, T, R)`` (or ``(L, T)``) context-indexed masses; NaN marks a pad.

    Returns
    -------
    numpy.ndarray
        Same shape, NaN replaced by ``0`` and axis 0 sorted descending.
    """
    filled = np.nan_to_num(np.asarray(array, dtype=float), nan=0.0)
    # numpy sorts ascending only; negate, sort, negate back.
    return -np.sort(-filled, axis=0)


def _pad_to(array, length):
    """Zero-pad a context-indexed array along axis 0 up to ``length``.

    Parameters
    ----------
    array : numpy.ndarray
        Array whose axis 0 is the context axis.
    length : int
        Target size of axis 0; must be at least ``array.shape[0]``.

    Returns
    -------
    numpy.ndarray
        The array, zero-extended along axis 0.
    """
    deficit = int(length) - array.shape[0]
    if deficit <= 0:
        return array
    pad = [(0, deficit)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad, mode="constant", constant_values=0.0)


def _reduce(captures, context_length):
    """Reduce a stack of captures to per-quantity mean and std over runs.

    Parameters
    ----------
    captures : dict
        ``{field: (n, T, R) array}``; context fields are NOT yet sorted.
    context_length : int
        Common context-axis length the context fields are padded to.

    Returns
    -------
    dict
        ``{field: {"mean": (n, T), "std": (n, T)}}``. ``std`` is the population
        standard deviation over the run axis (``ddof=0``).
    """
    out = {}
    for field, stack in captures.items():
        if field in CONTEXT_FIELDS:
            stack = _sorted_context(_pad_to(stack, context_length))
        out[field] = {
            "mean": np.mean(stack, axis=2),
            "std": np.std(stack, axis=2),
        }
    return out


def python_run_average(scenario, seeds=None, context_length=None):
    """Replay a scenario under many seeds and reduce to mean/std per trial.

    Parameters
    ----------
    scenario : dict
        A :func:`tests.equiv.scenario_battery.load_scenario` result.
    seeds : sequence or None, optional
        Model seeds; ``None`` uses :func:`python_seeds`.
    context_length : int or None, optional
        Context-axis length to pad to; ``None`` uses the model's natural
        ``max_contexts + 1``.

    Returns
    -------
    dict
        ``{field: {"mean", "std"}}``, as produced by :func:`_reduce`.
    """
    if seeds is None:
        seeds = python_seeds(scenario)
    natural = int(scenario["args"]["max_contexts"]) + 1
    if context_length is None:
        context_length = natural

    per_seed = [capture_run(scenario, seed) for seed in seeds]
    stacks = {
        field: np.stack([run[field] for run in per_seed], axis=2)
        for field in per_seed[0]
    }
    return _reduce(stacks, context_length)


def matlab_run_average(scenario, context_length=None):
    """Reduce the frozen MATLAB capture to mean/std per trial.

    Parameters
    ----------
    scenario : dict
        A :func:`tests.equiv.scenario_battery.load_scenario` result.
    context_length : int or None, optional
        Context-axis length to pad to; ``None`` uses the fixture's own ``L``.

    Returns
    -------
    dict
        ``{field: {"mean", "std"}}``, reduced exactly as the Python side is.
    """
    stacks = {field: array for field, array in scenario["matlab"].items()}
    if context_length is None:
        context_length = stacks["pred_ctx"].shape[0]
    return _reduce(stacks, context_length)


def compare_scenario(
    scenario,
    runs=PARITY_RUNS,
    base_seed=PARITY_BASE_SEED,
    floors=None,
    band_sigmas=BAND_SIGMAS,
):
    """Compare one scenario's Python and MATLAB run-averages.

    Parameters
    ----------
    scenario : dict or str
        A :func:`tests.equiv.scenario_battery.load_scenario` result, or a
        scenario name to load.
    runs : int, optional
        Number of Python seeds. Default :data:`PARITY_RUNS`.
    base_seed : int, optional
        Root entropy for the Python seeds. Default :data:`PARITY_BASE_SEED`.
    floors : dict or None, optional
        Per-quantity band floors; ``None`` uses :data:`FLOORS`.
    band_sigmas : float, optional
        Band width in standard errors. Default :data:`BAND_SIGMAS`.

    Returns
    -------
    dict
        Keys:

        ``name``
            Scenario name.
        ``runs`` / ``matlab_runs``
            Seeds averaged on each side.
        ``fields``
            ``{field: {...}}`` with, per quantity: ``max_deviation`` (largest
            ``|py_mean - matlab_mean|``, over all entries), ``max_excess``
            (largest ``deviation - band``; negative means comfortably inside),
            ``max_band``, ``max_sampling_term`` (the band without its floor),
            ``required_floor`` (the smallest floor that would have sufficed -
            negative means the sampling term alone was enough), ``floor``,
            ``worst_index`` / ``worst_trial`` / ``worst_deviation`` /
            ``worst_band`` all taken AT the largest-excess entry (which is not
            in general where ``max_deviation`` occurs), ``n_violations``,
            ``n_entries`` and ``passed``.
        ``max_deviation`` / ``max_excess``
            Maxima over all quantities.
        ``passed``
            ``True`` when no entry of any quantity exceeds its band.
    """
    if isinstance(scenario, str):
        scenario = load_scenario(scenario)
    if floors is None:
        floors = FLOORS

    matlab_runs = int(scenario["matlab_seeds"].size)
    # Pad both sides to a common context capacity so the sorted vectors line up.
    context_length = max(
        int(scenario["matlab"]["pred_ctx"].shape[0]),
        int(scenario["args"]["max_contexts"]) + 1,
    )

    seeds = python_seeds(scenario, runs=runs, base_seed=base_seed)
    python = python_run_average(scenario, seeds, context_length)
    matlab = matlab_run_average(scenario, context_length)

    report_fields = {}
    for field in VALUE_FIELDS + CONTEXT_FIELDS:
        py_mean = python[field]["mean"]
        ml_mean = matlab[field]["mean"]
        if py_mean.shape != ml_mean.shape:
            raise AssertionError(
                "scenario %s field %r: python shape %s vs MATLAB shape %s"
                % (scenario["name"], field, py_mean.shape, ml_mean.shape)
            )
        deviation = np.abs(py_mean - ml_mean)
        # Welch standard error of the difference of the two sample means.
        standard_error = np.sqrt(
            matlab[field]["std"] ** 2 / matlab_runs
            + python[field]["std"] ** 2 / runs
        )
        band = band_sigmas * standard_error + floors[field]
        excess = deviation - band
        flat = int(np.argmax(excess))
        worst = np.unravel_index(flat, excess.shape)
        report_fields[field] = {
            "max_deviation": float(np.max(deviation)),
            "max_excess": float(excess.flat[flat]),
            "max_band": float(np.max(band)),
            "max_sampling_term": float(band_sigmas * np.max(standard_error)),
            # Smallest floor that would have made this quantity pass, i.e. how
            # much of the agreement the floor is responsible for. Negative means
            # the sampling term alone was enough.
            "required_floor": float(np.max(deviation - band_sigmas * standard_error)),
            "floor": float(floors[field]),
            "worst_index": int(worst[0]),
            "worst_trial": int(worst[1]),
            "worst_deviation": float(deviation[worst]),
            "worst_band": float(band[worst]),
            "n_entries": int(deviation.size),
            "n_violations": int(np.count_nonzero(excess > 0)),
            "passed": bool(np.all(excess <= 0)),
        }

    return {
        "name": scenario["name"],
        "runs": int(runs),
        "matlab_runs": matlab_runs,
        "band_sigmas": float(band_sigmas),
        "fields": report_fields,
        "max_deviation": max(f["max_deviation"] for f in report_fields.values()),
        "max_excess": max(f["max_excess"] for f in report_fields.values()),
        "passed": all(f["passed"] for f in report_fields.values()),
    }


def compare_all(names=SCENARIO_NAMES, **kwargs):
    """Compare every scenario in the battery.

    Parameters
    ----------
    names : sequence of str, optional
        Scenario names; default is the whole battery.
    **kwargs
        Forwarded to :func:`compare_scenario`.

    Returns
    -------
    dict
        ``{"scenarios": [report, ...], "passed": bool, "max_excess": float}``.
    """
    reports = [compare_scenario(name, **kwargs) for name in names]
    return {
        "scenarios": reports,
        "passed": all(r["passed"] for r in reports),
        "max_excess": max(r["max_excess"] for r in reports),
    }


def format_report(report):
    """Render one scenario report as a fixed-width table.

    Parameters
    ----------
    report : dict
        A :func:`compare_scenario` result.

    Returns
    -------
    str
        One header line plus one line per quantity.
    """
    lines = [
        "%s (python runs=%d, matlab runs=%d): %s"
        % (
            report["name"],
            report["runs"],
            report["matlab_runs"],
            "PASS" if report["passed"] else "FAIL",
        ),
        # dev@worst / band@worst are BOTH read at argmax(excess); max|dev| is a
        # separate maximum over all entries and generally sits elsewhere.
        "  %-11s %10s %10s %10s %10s %10s %9s"
        % (
            "quantity",
            "max|dev|",
            "dev@worst",
            "band@worst",
            "excess",
            "req_floor",
            "viol/N",
        ),
    ]
    for field in VALUE_FIELDS + CONTEXT_FIELDS:
        stats = report["fields"][field]
        lines.append(
            "  %-11s %10.3e %10.3e %10.3e %10.2e %10.2e %4d/%-4d"
            % (
                field,
                stats["max_deviation"],
                stats["worst_deviation"],
                stats["worst_band"],
                stats["max_excess"],
                stats["required_floor"],
                stats["n_violations"],
                stats["n_entries"],
            )
        )
    return "\n".join(lines)

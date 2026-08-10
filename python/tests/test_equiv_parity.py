"""Cross-language parity: Python ``RealTimeCOIN`` versus the MATLAB fixtures.

This is the anchor test of the port. It replays six frozen scenarios - scalar
and multi-dimensional, with and without missing observations - through the
Python model and checks that the run-averaged observable surface matches the
frozen MATLAB run-average inside a Monte-Carlo band. See
:mod:`tests.equiv.compare_runs` for the band definition and
``tests/fixtures/oracle/README.md`` for the fixture formats.

Test layout
-----------
* Fast, unmarked structural tests (fixture integrity, capture shapes, seed
  determinism) run in the ordinary suite; they catch a drifted or mis-parsed
  fixture in milliseconds.
* The parity comparison itself is ``@pytest.mark.validation`` and
  ``@pytest.mark.slow``: it runs 6 scenarios x 20 seeds, which is roughly
  **18 minutes** on a current laptop (one capture costs 3-12 s depending on the
  scenario's particle count). It is computed ONCE in a session-scoped fixture
  and shared by the per-scenario tests, so parametrising does not multiply the
  cost. Deselect it with ``-m "not slow"`` when iterating.
* The oracle validators get thin wrappers here too, so the whole cross-language
  surface fails in one place.
"""

from __future__ import annotations

import numpy as np
import pytest

from equiv.capture_run import capture_run
from equiv.compare_runs import (
    FLOORS,
    PARITY_RUNS,
    compare_all,
    format_report,
    python_seeds,
)
from equiv.scenario_battery import (
    SCENARIO_NAMES,
    fixtures_available,
    load_scenario,
    scenario_battery,
)

#: Applied to the equivalence tests only. The oracle-validator wrappers at the
#: bottom of this module depend on a DIFFERENT fixture family
#: (``tests/fixtures/oracle/``) and gate themselves on their validator's own
#: ``skipped`` flag, so they must not be governed by this marker.
requires_equiv_fixtures = pytest.mark.skipif(
    not fixtures_available(),
    reason="equivalence fixtures absent (see tests/fixtures/oracle/README.md)",
)


# ----------------------------------------------------------------------
# Fast structural checks
# ----------------------------------------------------------------------


@requires_equiv_fixtures
def test_battery_loads_every_scenario():
    """All six fixtures load and self-validate against the re-derived schedule.

    ``load_scenario`` re-derives the deterministic half of MATLAB's
    ``makeInputs`` (cue blocks, missing-trial cadence) and raises on any
    disagreement, so merely loading is a real assertion about fixture
    provenance.
    """
    scenarios = scenario_battery()
    assert [s["name"] for s in scenarios] == list(SCENARIO_NAMES)
    for scenario in scenarios:
        assert scenario["q"].size == scenario["trials"]
        assert scenario["y"].shape == (scenario["dim"], scenario["trials"])
        assert scenario["matlab_seeds"].size == PARITY_RUNS


@requires_equiv_fixtures
@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_fixture_context_pads_are_trailing(name):
    """Every NaN pad sits in the trailing context slots, never mid-vector.

    The zero-fill rule (NaN pad == zero mass) is only sound if pads mark absent
    capacity rather than holes in the middle of an occupied vector.
    """
    scenario = load_scenario(name)
    for field in ("pred_ctx", "resp", "counts"):
        array = scenario["matlab"][field]
        is_nan = np.isnan(array)
        # Along the context axis a valid pad pattern is a run of False then a
        # run of True, i.e. is_nan is non-decreasing down the axis.
        assert np.all(np.diff(is_nan.astype(int), axis=0) >= 0), (
            "%s/%s has a NaN pad above a finite entry" % (name, field)
        )


@requires_equiv_fixtures
def test_capture_shapes_and_determinism():
    """A capture has the documented shapes and is reproducible from its seed."""
    scenario = load_scenario("md2_basic")
    seed = python_seeds(scenario, runs=1)[0]
    first = capture_run(scenario, seed)
    second = capture_run(scenario, python_seeds(scenario, runs=1)[0])

    contexts = scenario["args"]["max_contexts"] + 1
    trials = scenario["trials"]
    assert first["motor"].shape == (scenario["dim"], trials)
    assert first["state_mean"].shape == (scenario["dim"], trials)
    for field in ("pred_ctx", "resp", "counts"):
        assert first[field].shape == (contexts, trials)
        np.testing.assert_allclose(np.sum(first[field], axis=0), 1.0, atol=1e-9)
    for field in first:
        np.testing.assert_array_equal(first[field], second[field])


@requires_equiv_fixtures
def test_missing_trials_still_produce_finite_output():
    """A scenario with NaN feedback trials still yields a finite surface."""
    scenario = load_scenario("scalar_missing")
    assert np.isnan(scenario["y"]).any()
    capture = capture_run(scenario, python_seeds(scenario, runs=1)[0])
    for field, values in capture.items():
        assert np.all(np.isfinite(values)), "%s went non-finite" % field


# ----------------------------------------------------------------------
# The parity comparison
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def parity_reports():
    """Run the whole battery once and share the reports across tests.

    Returns
    -------
    dict
        ``{scenario name: compare_scenario report}``.
    """
    result = compare_all()
    print()
    for report in result["scenarios"]:
        print(format_report(report))
        print()
    return {report["name"]: report for report in result["scenarios"]}


@requires_equiv_fixtures
@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_run_average_matches_matlab(parity_reports, name):
    """Python and MATLAB run-averages agree inside the Monte-Carlo band.

    A failure here is a claim about the PORT, not about tolerances: the band is
    the three-sigma sampling error of the difference of the two run-averages,
    estimated from both sides' own across-seed spread, so an excess means the
    Python model samples a measurably different distribution from the MATLAB
    one on identical inputs. Before widening anything, re-run the null control
    described in :mod:`tests.equiv.compare_runs` - Python seeds against
    different Python seeds - and check whether the same code disagrees with
    itself by as much.
    """
    report = parity_reports[name]
    failing = {
        field: stats
        for field, stats in report["fields"].items()
        if not stats["passed"]
    }
    assert not failing, (
        "%s: %d quantity/quantities outside the band\n%s"
        % (
            name,
            len(failing),
            "\n".join(
                "  %-11s max|dev|=%.4e worst band=%.4e excess=%.3e "
                "(%d/%d entries, worst at context %d trial %d)"
                % (
                    field,
                    stats["max_deviation"],
                    stats["worst_band"],
                    stats["max_excess"],
                    stats["n_violations"],
                    stats["n_entries"],
                    stats["worst_index"],
                    stats["worst_trial"],
                )
                for field, stats in failing.items()
            ),
        )
    )


@requires_equiv_fixtures
@pytest.mark.validation
@pytest.mark.slow
def test_parity_floors_are_not_doing_all_the_work(parity_reports):
    """Every quantity's band is driven by sampling error, not by its floor.

    A guard against the failure mode where a floor is quietly raised until
    everything passes. The check is PER QUANTITY, deliberately: an earlier
    version asked only that SOME quantity per scenario be informative, which
    ``pred_ctx`` satisfied trivially and which therefore could never fire for
    the quantity that actually had an oversized floor.

    The requirement is that the three-sigma sampling term exceed the floor by at
    least 2x SOMEWHERE in the battery, per quantity. Aggregating over scenarios
    rather than demanding it of every scenario keeps the check from being
    brittle where a quantity happens to be unusually reproducible, while still
    firing decisively on an oversized floor: an earlier ``counts`` floor of 0.35
    sat above every scenario's peak sampling term of <= 0.18, and this assertion
    rejects it.
    """
    peaks = {}
    for report in parity_reports.values():
        for field, stats in report["fields"].items():
            peaks[field] = max(
                peaks.get(field, 0.0), stats["max_sampling_term"]
            )
    offenders = [
        "%s: 3*SE peaks at %.3e across the battery but the floor is %.3e"
        % (field, peak, FLOORS[field])
        for field, peak in peaks.items()
        if peak <= 2.0 * FLOORS[field]
    ]
    assert not offenders, "floor dominates the band:\n  " + "\n  ".join(offenders)


@requires_equiv_fixtures
@pytest.mark.validation
@pytest.mark.slow
def test_parity_agreement_is_not_only_the_floor(parity_reports):
    """Most entries clear the band on the sampling term alone.

    Complements :func:`test_parity_floors_are_not_doing_all_the_work`: that one
    checks the band's SHAPE, this one checks that the floor is a tail correction
    rather than the thing carrying the result. If a quantity needed its floor at
    more than a few percent of entries, the comparison would have degenerated
    into "within a fixed tolerance".
    """
    for name, report in parity_reports.items():
        for field, stats in report["fields"].items():
            # required_floor is max(deviation - 3*SE) over entries; <= 0 means
            # the sampling term alone sufficed everywhere.
            assert stats["required_floor"] < FLOORS[field], (
                "%s/%s needs a floor of %.3e, at or above the configured %.3e"
                % (name, field, stats["required_floor"], FLOORS[field])
            )


# ----------------------------------------------------------------------
# Oracle validator wrappers
# ----------------------------------------------------------------------


@pytest.mark.validation
@pytest.mark.slow
def test_original_coin_monte_carlo_passes():
    """``RealTimeCOIN`` matches the frozen offline-COIN traces."""
    from validation import validate_original_coin_monte_carlo

    results = validate_original_coin_monte_carlo.run()
    if results.get("skipped"):
        pytest.skip(results["skipped_reason"])
    assert results["passed"], (
        "mean rmse %.5f (gate < %.3f), worst corr %.5f (gate > %.3f); "
        "per-seed rmse %s"
        % (
            results["mean_rmse"],
            results["thresholds"]["mean_rmse"],
            results["worst_correlation"],
            results["thresholds"]["worst_correlation"],
            np.array2string(np.asarray(results["rmse"]), precision=5),
        )
    )


@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("blind", ["A", "B"])
def test_ensemble_vs_coin(blind):
    """``RealTimeCOINEnsemble`` run-averaging matches COIN's.

    Skips until the ensemble translation unit merges; both validators report
    ``skipped=True`` with ``skipped_reason`` rather than failing.
    """
    import importlib

    module = importlib.import_module(
        "validation.validate_ensemble_vs_coin_blind%s" % blind
    )
    results = module.run()
    if results.get("skipped"):
        pytest.skip(results["skipped_reason"])
    assert results["passed"], "blind %s checks: %s" % (blind, results["checks"])

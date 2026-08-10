"""Pytest wrappers around the Kalman-reference scientific validators.

These are thin: each test runs one validator at its documented default seed and
asserts the overall ``passed`` flag, so a regression in the filter shows up as a
test failure rather than only as a number in a report. The validators themselves
own the metrics, the gates and the console output - see
:mod:`validation.validate_single_context_kalman`,
:mod:`validation.validate_multidim_kalman` and
:mod:`validation.validate_particle_convergence`.

The two fixture-gating tests exist because the oracle-comparison path of the
convergence validator is otherwise dead code in a checkout without the
offline-COIN traces: they drive it both ways with a synthetic archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ``validation`` is a namespace package living beside ``tests/``, and it is NOT
# installed by ``pip install -e .`` (which packages ``realtimecoin`` only).
# pytest's default import mode puts ``tests/`` on sys.path, not its parent, so
# the parent is prepended here; that makes ``pytest tests/`` work as well as
# ``python -m pytest tests/``. Adding ``pythonpath = ["."]`` under
# ``[tool.pytest.ini_options]`` would make this bootstrap unnecessary.
_PYTHON_ROOT = str(Path(__file__).resolve().parents[1])
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from validation.kalman_reference import kalman_reference_step
from validation.uniform_ks import uniform_ks
from validation.validate_multidim_kalman import (
    THRESHOLDS as MD_THRESHOLDS,
    run as run_multidim_kalman,
)
from validation.validate_particle_convergence import (
    THRESHOLDS as CONVERGENCE_THRESHOLDS,
    run as run_particle_convergence,
)
from validation.validate_single_context_kalman import (
    THRESHOLDS as SCALAR_THRESHOLDS,
    run as run_single_context_kalman,
)

pytestmark = pytest.mark.validation


def _assert_contract(results):
    """Assert the module contract every ``validate_*.run`` shares.

    Parameters
    ----------
    results : dict
        A validator's return value.
    """
    assert results["errored"] is False
    assert isinstance(results["passed"], bool)
    assert isinstance(results["checks"], dict)
    assert results["checks"]


def test_single_context_kalman_passes_at_default_seed():
    """The scalar model reduces to the analytic Kalman filter.

    ``strict=True`` doubles as an assertion that a PASSING run returns its
    metrics normally instead of raising - the behaviour the suite driver
    depends on.
    """
    results = run_single_context_kalman(strict=True)
    _assert_contract(results)

    assert results["predictive_mean_rmse"] < SCALAR_THRESHOLDS["mean_rmse"]
    assert (
        results["variance_relative_error"]
        < SCALAR_THRESHOLDS["variance_relative_error"]
    )
    assert results["feedback_ks"] < SCALAR_THRESHOLDS["feedback_ks"]
    # The independently computed analytic PIT is reported alongside the model's.
    assert np.isfinite(results["analytic_feedback_ks"])
    # The production moments and the validation mirror's must agree to
    # round-off - this is the two-implementation cross-check.
    assert results["moment_cross_check_max_abs_diff"] < 1e-12
    assert results["passed"]


def test_multidim_kalman_passes_at_default_seed():
    """The N-dimensional model reduces to the multivariate Kalman filter."""
    results = run_multidim_kalman(strict=True)
    _assert_contract(results)

    assert results["dim"] == 2
    assert results["predictive_mean_rmse"] < MD_THRESHOLDS["mean_rmse"]
    assert (
        results["variance_relative_error"]
        < MD_THRESHOLDS["variance_relative_error"]
    )
    assert results["feedback_ks"] < MD_THRESHOLDS["feedback_ks"]
    assert results["passed"]


def test_multidim_kalman_runs_at_dim_one():
    """``dim=1`` routes through the SCALAR pipeline and must not blow up.

    At ``N == 1`` the model runs the scalar pipeline: it returns plain floats
    from ``predictive_feedback_moments`` (so the Mahalanobis solve needs the
    promotion the validator applies) and it IGNORES the covariance overrides
    (so the validator must hand it the equivalent scalar sigmas instead). Only
    the Kalman-agreement metrics are asserted - 25 trials is far too small a
    sample for the KS gate, whose null expectation is about
    ``0.87 / sqrt(25) = 0.17``.
    """
    results = run_multidim_kalman(trials=25, particles=60, dim=1)

    assert results["dim"] == 1
    assert results["predictive_mean_rmse"] < MD_THRESHOLDS["mean_rmse"]
    assert (
        results["variance_relative_error"]
        < MD_THRESHOLDS["variance_relative_error"]
    )
    assert np.all(np.isfinite(results["feedback_p_values"]))


def test_strict_raises_when_a_gate_fails(monkeypatch):
    """``strict=True`` turns a failed gate into an ``AssertionError``.

    The gate is made impossible rather than the data made bad, so the test is
    deterministic instead of resting on a statistic landing the wrong side of a
    threshold.
    """
    monkeypatch.setitem(SCALAR_THRESHOLDS, "mean_rmse", -1.0)

    with pytest.raises(AssertionError):
        run_single_context_kalman(strict=True, trials=10, particles=40)

    # Without strict the same run reports the failure instead of raising.
    results = run_single_context_kalman(trials=10, particles=40)
    assert results["passed"] is False
    assert results["checks"]["mean_rmse"] is False


def test_particle_convergence_passes_at_default_seed():
    """Calibration and runtime behave sanely as the particle count grows."""
    results = run_particle_convergence()
    _assert_contract(results)

    assert (
        results["best_feedback_ks"]
        < CONVERGENCE_THRESHOLDS["best_feedback_ks"]
    )
    assert results["checks"]["runtime_nondecreasing"]
    assert results["passed"]


def test_particle_convergence_skips_oracle_gate_without_fixtures(tmp_path):
    """A missing oracle archive SKIPS the RMSE gate instead of failing it."""
    results = run_particle_convergence(
        particles=(8, 12), trials=6, datasets=1, oracle_dir=tmp_path
    )

    assert results["checks"]["best_rmse"] is None
    assert results["best_rmse"] is None
    assert results["original_coin_rmse"] == [None, None]
    assert "oracle fixtures missing" in results["skipped_reason"]
    # A skipped check must not drag the overall verdict down by itself.
    assert results["passed"] == all(
        v for v in results["checks"].values() if v is not None
    )


def test_particle_convergence_uses_oracle_fixture_when_present(tmp_path):
    """A present oracle archive is replayed, turning the RMSE gate into a bool."""
    trials = 12
    rng = np.random.default_rng(0)
    feedback = 0.2 * rng.standard_normal(trials)
    np.savez(
        tmp_path / "compare_original_coin_p8.npz",
        cues=np.ones(trials),
        feedback=feedback,
        # A non-constant reference trace, so the reported correlation is
        # defined as well as the RMSE.
        motor_output=0.1 * rng.standard_normal(trials),
        max_contexts=np.array(2),
        sigma_motor_noise=np.array(0.0),
    )

    results = run_particle_convergence(
        particles=(8,), trials=6, datasets=1, oracle_dir=tmp_path
    )

    assert results["skipped_reason"] is None
    assert isinstance(results["checks"]["best_rmse"], bool)
    assert np.isfinite(results["best_rmse"])
    assert results["best_rmse"] == results["best_original_coin_rmse"]


def test_kalman_reference_step_matches_hand_computed_scalar():
    """Spot-check the shared recursion against a closed-form scalar step."""
    m, p, a, d, q, r, y = 0.5, 0.4, 0.8, 0.1, 0.09, 0.25, 1.2

    pred_mean, pred_cov, post_mean, post_cov = kalman_reference_step(
        m, p, a, d, q, r, y
    )

    expected_pred_mean = a * m + d
    expected_state_cov = a * p * a + q
    expected_pred_cov = expected_state_cov + r
    gain = expected_state_cov / expected_pred_cov
    assert pred_mean == pytest.approx(expected_pred_mean)
    assert pred_cov == pytest.approx(expected_pred_cov)
    assert post_mean == pytest.approx(
        expected_pred_mean + gain * (y - expected_pred_mean)
    )
    assert post_cov == pytest.approx(expected_state_cov * (1.0 - gain))


def test_kalman_reference_step_scalar_is_the_one_dimensional_case():
    """The N == 1 matrix call reproduces the scalar call exactly."""
    scalar = kalman_reference_step(0.5, 0.4, 0.8, 0.1, 0.09, 0.25, 1.2)
    matrix = kalman_reference_step(
        np.array([0.5]),
        np.array([[0.4]]),
        np.array([[0.8]]),
        np.array([0.1]),
        np.array([[0.09]]),
        np.array([[0.25]]),
        np.array([1.2]),
    )
    for expected, actual in zip(scalar, matrix):
        flat = float(np.asarray(actual).reshape(-1)[0])
        assert flat == pytest.approx(expected)


def test_uniform_ks_is_zero_for_a_perfect_grid():
    """An exactly uniform sample has KS distance 1/n (its own discretisation)."""
    n = 100
    grid = (np.arange(1, n + 1) - 0.5) / n
    assert uniform_ks(grid) == pytest.approx(0.5 / n)
    assert np.isnan(uniform_ks([np.nan, np.inf]))

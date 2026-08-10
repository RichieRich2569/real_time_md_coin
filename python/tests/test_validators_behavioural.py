"""Tests for the behavioural validators and their helpers (unit C4).

Two very different kinds of test live here:

* FAST UNIT TESTS of :mod:`validation.pass_summary` and
  :mod:`validation.best_label_map` - pure functions, no model runs, always
  collected.
* ``@pytest.mark.validation`` WRAPPERS that run the three behavioural
  validators end to end at the SUITE seed (:data:`run_validation.DEFAULT_SEED`
  plus the stage offset) with the compact profile's sizes, and assert their
  ``passed`` flag. These are slow (each builds and runs particle filters) and
  seed-sensitive by design - ``context_recovery`` in particular passes at the
  suite seed and is documented as marginal at others - so they deliberately do
  NOT use an arbitrary seed.

The three end-to-end tests take roughly EIGHT MINUTES combined; budget for that
before running the default suite. Run only the fast ones with
``pytest -m "not validation"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ``python/validation`` is a namespace package next to ``python/tests``; pytest
# puts only the test directory on sys.path, so the package root is added here.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from validation import run_validation                       # noqa: E402
from validation.best_label_map import best_label_map        # noqa: E402
from validation.pass_summary import pass_summary            # noqa: E402
from validation.validate_context_recovery import (          # noqa: E402
    run as run_context_recovery,
)
from validation.validate_p_values_extended import (         # noqa: E402
    run as run_p_values_extended,
)
from validation.validate_stress_cases import (              # noqa: E402
    run as run_stress_cases,
)


def _suite_seed(stage):
    """Seed the compact suite gives ``stage``."""
    return run_validation.DEFAULT_SEED + run_validation.STAGE_SEED_OFFSETS[stage]


# --------------------------------------------------------------------------
# pass_summary
# --------------------------------------------------------------------------


def test_pass_summary_all_true():
    """Every truthy check passes and the flags come back as plain bools."""
    passed, checks = pass_summary({"a": True, "b": 1, "c": np.bool_(True)})
    assert passed is True
    assert checks == {"a": True, "b": True, "c": True}
    assert all(isinstance(v, bool) for v in checks.values())


@pytest.mark.parametrize(
    "value",
    [
        False,
        0,
        0.0,
        float("nan"),
        float("inf"),
        None,
        np.array([True, True]),
        "yes",
        # A numeric STRING: np.asarray("1", dtype=float) would succeed, but
        # MATLAB's isnumeric('1') is false, so it must fail the check.
        "1",
        b"1",
    ],
)
def test_pass_summary_rejects(value):
    """nan / None / non-scalars / zeros / text all count as failures."""
    passed, checks = pass_summary({"only": value})
    assert passed is False
    assert checks["only"] is False


def test_pass_summary_does_not_mutate_input():
    """The caller's mapping is left untouched (MATLAB returns by value)."""
    original = {"a": 1.0, "b": float("nan")}
    passed, checks = pass_summary(original)
    assert passed is False
    assert checks is not original
    assert original["a"] == 1.0
    assert np.isnan(original["b"])


def test_pass_summary_preserves_order():
    """Check order is preserved so error messages read predictably."""
    _passed, checks = pass_summary({"z": True, "a": True, "m": False})
    assert list(checks) == ["z", "a", "m"]


# --------------------------------------------------------------------------
# best_label_map
# --------------------------------------------------------------------------


def test_best_label_map_identity():
    """Already-aligned labels map to themselves with perfect accuracy."""
    true_labels = [0, 0, 1, 1, 1]
    mapped, accuracy, mapping = best_label_map(true_labels, true_labels)
    assert accuracy == pytest.approx(1.0)
    assert mapping == {0: 0, 1: 1}
    np.testing.assert_array_equal(mapped, true_labels)


def test_best_label_map_swapped():
    """A pure label swap is recovered exactly."""
    true_labels = [0, 0, 1, 1]
    inferred = [1, 1, 0, 0]
    mapped, accuracy, mapping = best_label_map(true_labels, inferred)
    assert accuracy == pytest.approx(1.0)
    assert mapping == {0: 1, 1: 0}
    np.testing.assert_array_equal(mapped, true_labels)


def test_best_label_map_partial_agreement():
    """Accuracy is the best achievable agreement, not the raw agreement."""
    true_labels = [0, 0, 0, 1]
    inferred = [1, 1, 1, 1]
    _mapped, accuracy, mapping = best_label_map(true_labels, inferred)
    # One inferred label can only be sent to one target: 3/4 is the optimum.
    assert accuracy == pytest.approx(0.75)
    assert mapping == {1: 0}


def test_best_label_map_ignores_non_finite():
    """nan entries are excluded from scoring and pass through ``mapped``."""
    true_labels = [0, 0, 1, np.nan]
    inferred = [1, 1, 0, np.nan]
    mapped, accuracy, _mapping = best_label_map(true_labels, inferred)
    assert accuracy == pytest.approx(1.0)
    assert np.isnan(mapped[-1])


def test_best_label_map_empty():
    """No scorable trials: identity output, nan accuracy, empty mapping."""
    mapped, accuracy, mapping = best_label_map([np.nan], [np.nan])
    assert np.isnan(accuracy)
    assert mapping == {}
    assert np.isnan(mapped[0])


def test_best_label_map_large_k_beats_matlab_identity_fallback():
    """K = 12 is recovered exactly - the case MATLAB's fallback gets wrong.

    ``validation_best_label_map.m`` brute-forces all ``K!`` permutations only
    for ``K <= 8`` and, without ``matchpairs``, silently returns the IDENTITY
    map for larger ``K``. Here 12 contexts are permuted by a derangement, so the
    identity map would score 0; the Hungarian solver must score 1.
    """
    k = 12
    # Derangement: true label g is inferred as (g + 5) mod 12 - no fixed points.
    permutation = [(g + 5) % k for g in range(k)]
    true_labels = []
    inferred = []
    for g in range(k):
        # Unequal block sizes so the optimum is not accidentally symmetric.
        repeats = 3 + g
        true_labels.extend([g] * repeats)
        inferred.extend([permutation[g]] * repeats)

    mapped, accuracy, mapping = best_label_map(true_labels, inferred)

    assert accuracy == pytest.approx(1.0)
    assert mapping == {permutation[g]: g for g in range(k)}
    np.testing.assert_array_equal(mapped, true_labels)
    # Sanity: the identity map really would have been wrong here.
    assert np.mean(np.asarray(inferred) == np.asarray(true_labels)) == 0.0


def test_best_label_map_never_merges_two_contexts():
    """The mapping is injective: distinct inferred labels get distinct targets."""
    rng = np.random.default_rng(7)
    true_labels = rng.integers(0, 5, size=200)
    inferred = rng.integers(0, 5, size=200)
    _mapped, _accuracy, mapping = best_label_map(true_labels, inferred)
    assert len(set(mapping.values())) == len(mapping)


# --------------------------------------------------------------------------
# End-to-end validators (slow, run at the suite seed)
# --------------------------------------------------------------------------


@pytest.mark.validation
def test_p_values_extended_passes_at_suite_seed():
    """PIT self-calibration passes its KS gates with the compact sizes."""
    suite = run_validation.compact_config()
    results = run_p_values_extended(
        seed=_suite_seed("p_values_extended"),
        num_datasets=suite["pit_datasets"],
        trials=suite["pit_trials"],
        particles=suite["pit_particles"],
    )
    assert results["passed"], results["checks"]


@pytest.mark.validation
def test_context_recovery_passes_at_suite_seed():
    """Known latent contexts are recovered after optimal relabelling.

    Seed-sensitive by construction (see the validator's header): the gates act
    on the average over ``num_seeds`` seeds, and the suite seed is the
    documented one.
    """
    suite = run_validation.compact_config()
    results = run_context_recovery(
        seed=_suite_seed("context_recovery"),
        trials=suite["context_trials"],
        particles=suite["context_particles"],
    )
    assert results["passed"], results["checks"]


@pytest.mark.validation
def test_stress_cases_pass_at_suite_seed():
    """The qualitative behavioural probes all behave as expected."""
    suite = run_validation.compact_config()
    results = run_stress_cases(
        seed=_suite_seed("stress_cases"),
        trials=suite["stress_trials"],
        particles=suite["stress_particles"],
    )
    assert results["passed"], results["checks"]

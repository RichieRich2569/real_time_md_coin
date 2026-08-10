"""Deterministic (config, inputs) cases for the cross-language equivalence test.

Port of ``tests/+equiv/scenarioBattery.m``.

What is re-derived and what is loaded
-------------------------------------
The MATLAB battery builds each scenario's cue sequence ``q`` and feedback
stream ``y`` from a dedicated ``RandStream('twister', 'Seed', 7000 + i)``.
NumPy cannot reproduce that stream, so ``y`` CANNOT be regenerated here.

Instead every scenario LOADS ``q`` and ``y`` from the frozen fixture
``tests/fixtures/equiv/<name>.mat`` captured from the live MATLAB session. That
guarantees the two languages consume byte-identical inputs, which is the whole
point of the battery: any difference in the recorded surface is then attributable
to the model, never to the driving data.

The parts of ``makeInputs`` that do NOT depend on the random stream are still
re-derived here and CHECKED against the fixture:

* the cue schedule ``q`` - contiguous blocks of
  ``blockLen = max(1, round(T / (2 * cues)))`` trials cycling through
  ``1 .. cues`` - is fully deterministic, so it is regenerated and asserted
  equal to the fixture's ``q`` element-wise;
* the missing-observation cadence - whole-trial NaN at MATLAB indices
  ``step:step:T`` with ``step = max(1, round(1 / missing_frac))`` - is likewise
  deterministic, so the fixture's NaN column pattern is asserted to match.

Together with the shape assertions (``dim``, ``T``, ``max_contexts``, the number
of stored seeds) this pins down everything about the fixture except the actual
random numbers, so a fixture regenerated from a drifted ``scenarioBattery.m``
would be caught rather than silently replayed.

MATLAB rounding note
--------------------
``round`` in MATLAB is round-half-AWAY-from-zero; Python's built-in ``round``
and ``numpy.round`` are round-half-to-EVEN. The two disagree at exact ``.5``
(e.g. ``round(2.5)`` is 3 in MATLAB, 2 in Python), so :func:`_matlab_round`
implements the MATLAB rule. It matters: ``scalar_missing`` has
``1 / 0.20 == 5.0`` exactly, and other configurations can land on a half.

Fixture layout
--------------
Loaded with ``scipy.io.loadmat(..., squeeze_me=True)``, which drops singleton
dimensions. The stored arrays are, in MATLAB (column-major) terms:

============  =========================  ==================================
variable      MATLAB shape               shape seen after ``squeeze_me``
============  =========================  ==================================
``q``         ``1 x T``                  ``(T,)``
``y``         ``dim x T``                ``(T,)`` if ``dim == 1`` else ``(dim, T)``
``motor``     ``dim x T x R``            ``(T, R)`` if ``dim == 1`` else ``(dim, T, R)``
``stateMean`` ``dim x T x R``            same as ``motor``
``predCtx``   ``L x T x R``              ``(L, T, R)``
``resp``      ``L x T x R``              ``(L, T, R)``
``counts``    ``L x T x R``              ``(L, T, R)``
``seeds``     ``1 x R``                  ``(R,)``
``name``      char                       0-d ``<U`` array
============  =========================  ==================================

``L = max_contexts + 2`` is a fixed capacity, one entry wider than the
``max_contexts + 1`` the queries actually return, so every trial carries exactly
one all-NaN pad row. NaN pads mean "the MATLAB query returned no entry for this
context slot" and are treated as ZERO mass by
:func:`tests.equiv.compare_runs`; the loaders here keep them as NaN so the pad
pattern stays inspectable.

``scipy.io.loadmat`` returns MATLAB's column-major data already transposed into
C order by SciPy, so no manual ``.T`` is needed anywhere: index ``[dim, trial,
seed]`` directly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.io

__all__ = [
    "SPECS",
    "FIXTURE_DIR",
    "SCENARIO_NAMES",
    "fixtures_available",
    "load_scenario",
    "scenario_battery",
]

#: Directory holding the frozen MATLAB equivalence fixtures.
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "equiv"

#: Scenario table, transcribed verbatim from the ``specs`` cell array in
#: ``scenarioBattery.m``. ``seed`` is the MATLAB model seed; it is NOT used to
#: seed the Python model (the streams cannot match) but IS used as stable
#: entropy for the Python seed derivation, so the Python seeds differ per
#: scenario in a reproducible way. ``input_seed`` (``7000 + i``) is recorded for
#: provenance only.
SPECS = (
    # name, seed, particles, max_contexts, dim, trials, cues, missing_frac
    ("scalar_2ctx", 11, 100, 3, 1, 45, 2, 0.00),
    ("scalar_missing", 12, 100, 3, 1, 45, 2, 0.20),
    ("scalar_3ctx_small", 13, 40, 4, 1, 50, 3, 0.00),
    ("md2_basic", 14, 80, 3, 2, 40, 2, 0.00),
    ("md2_missing", 15, 80, 3, 2, 40, 2, 0.20),
    ("md3_basic", 16, 60, 4, 3, 35, 2, 0.00),
)

#: Scenario names in battery order.
SCENARIO_NAMES = tuple(spec[0] for spec in SPECS)


def _matlab_round(value):
    """Round half away from zero, as MATLAB's ``round`` does.

    Parameters
    ----------
    value : float
        Value to round.

    Returns
    -------
    int
        ``value`` rounded with ties broken away from zero, unlike Python's
        banker's rounding.

    Examples
    --------
    >>> _matlab_round(2.5), _matlab_round(-2.5), _matlab_round(11.25)
    (3, -3, 11)
    """
    return int(math.floor(value + 0.5)) if value >= 0 else -int(
        math.floor(-value + 0.5)
    )


def _expected_cue_schedule(trials, cues):
    """Re-derive the deterministic cue schedule of ``makeInputs``.

    Parameters
    ----------
    trials : int
        Number of trials ``T``.
    cues : int
        Number of distinct cue labels cycled through.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` array of 1-based cue labels: contiguous blocks of
        ``max(1, round(T / (2 * cues)))`` trials, label advancing
        ``mod(label, cues) + 1`` per block.
    """
    block_len = max(1, _matlab_round(trials / (2.0 * cues)))
    schedule = np.zeros(trials, dtype=np.int64)
    label = 1
    start = 0                      # 0-based; MATLAB's t is 1-based
    while start < trials:
        stop = min(trials, start + block_len)
        schedule[start:stop] = label
        label = (label % cues) + 1
        start = stop
    return schedule


def _expected_missing_mask(trials, missing_frac):
    """Re-derive the deterministic missing-observation cadence.

    Parameters
    ----------
    trials : int
        Number of trials ``T``.
    missing_frac : float
        Requested fraction of missing trials; ``0`` means none.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` boolean mask, ``True`` where the whole trial is NaN. MATLAB
        blanks ``y(:, step:step:T)`` with ``step = max(1, round(1 /
        missing_frac))``, i.e. 1-based indices, hence the ``step - 1`` offset.
    """
    mask = np.zeros(trials, dtype=bool)
    if missing_frac > 0:
        step = max(1, _matlab_round(1.0 / missing_frac))
        mask[step - 1:: step] = True
    return mask


def fixtures_available(fixture_dir=None):
    """Report whether every scenario fixture is present.

    Parameters
    ----------
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`FIXTURE_DIR`.

    Returns
    -------
    bool
        ``True`` only when all six ``<name>.mat`` files exist.
    """
    root = Path(fixture_dir) if fixture_dir is not None else FIXTURE_DIR
    return all((root / ("%s.mat" % name)).is_file() for name in SCENARIO_NAMES)


def _as_dim_trial(array, dim, trials, label):
    """Normalise a squeezed ``dim x T`` or ``dim x T x R`` array to full rank.

    ``squeeze_me=True`` drops MATLAB's leading singleton dimension for the
    scalar scenarios, so a ``1 x T x R`` array arrives as ``(T, R)``. This
    restores the leading axis so downstream code can index ``[dim, trial, ...]``
    uniformly.

    Parameters
    ----------
    array : numpy.ndarray
        Squeezed array from the fixture.
    dim : int
        Expected state dimension.
    trials : int
        Expected trial count.
    label : str
        Variable name, used in the error message.

    Returns
    -------
    numpy.ndarray
        Array whose first two axes are ``(dim, trials)``.

    Raises
    ------
    AssertionError
        If the restored shape does not start with ``(dim, trials)``.
    """
    array = np.asarray(array, dtype=float)
    if dim == 1 and (array.ndim == 1 or array.shape[0] != 1):
        array = array[None, ...]
    if array.shape[:2] != (dim, trials):
        raise AssertionError(
            "fixture variable %r has shape %s; expected leading (dim, T) = %s"
            % (label, array.shape, (dim, trials))
        )
    return array


def load_scenario(name, fixture_dir=None):
    """Load one scenario: spec metadata plus the frozen MATLAB capture.

    Parameters
    ----------
    name : str
        Scenario name; one of :data:`SCENARIO_NAMES`.
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`FIXTURE_DIR`.

    Returns
    -------
    dict
        Keys:

        ``name``, ``seed``, ``dim``, ``trials``, ``cues``, ``missing_frac``
            Spec metadata (``seed`` is the MATLAB model seed).
        ``args``
            ``dict`` of ``RealTimeCOIN`` constructor keywords
            (``num_particles``, ``max_contexts``, ``state_dim``).
        ``q``
            ``(T,)`` int array of 1-based raw cue values, from the fixture.
        ``y``
            ``(dim, T)`` float array of feedback, NaN where missing, from the
            fixture.
        ``matlab_seeds``
            ``(R,)`` int array of the MATLAB model seeds used for the capture.
        ``matlab``
            ``dict`` of the MATLAB capture arrays: ``motor`` and ``state_mean``
            with shape ``(dim, T, R)``; ``pred_ctx``, ``resp`` and ``counts``
            with shape ``(L, T, R)`` where ``L = max_contexts + 2`` and NaN marks
            an absent context slot.

    Raises
    ------
    FileNotFoundError
        If the fixture file is missing.
    KeyError
        If ``name`` is not a battery scenario.
    AssertionError
        If the fixture disagrees with the re-derived schedule, the NaN cadence,
        or the expected shapes.
    """
    try:
        spec = next(s for s in SPECS if s[0] == name)
    except StopIteration:                                     # pragma: no cover
        raise KeyError("unknown equivalence scenario %r" % name) from None
    _, seed, particles, max_contexts, dim, trials, cues, missing_frac = spec

    root = Path(fixture_dir) if fixture_dir is not None else FIXTURE_DIR
    path = root / ("%s.mat" % name)
    if not path.is_file():
        raise FileNotFoundError(
            "missing equivalence fixture %s (see tests/fixtures/oracle/README.md"
            " for how the .mat fixtures are produced)" % path
        )
    raw = scipy.io.loadmat(str(path), squeeze_me=True)

    stored_name = str(np.asarray(raw["name"]).reshape(-1)[0])
    if stored_name != name:
        raise AssertionError(
            "fixture %s stores name %r, expected %r" % (path, stored_name, name)
        )

    q = np.asarray(raw["q"], dtype=np.int64).reshape(-1)
    if q.size != trials:
        raise AssertionError(
            "fixture %s has %d cue entries, spec says T = %d"
            % (path, q.size, trials)
        )
    expected_q = _expected_cue_schedule(trials, cues)
    if not np.array_equal(q, expected_q):
        raise AssertionError(
            "fixture %s cue schedule disagrees with the re-derived "
            "scenarioBattery.m schedule (fixture=%s, expected=%s)"
            % (path, q.tolist(), expected_q.tolist())
        )

    y = _as_dim_trial(raw["y"], dim, trials, "y")
    # A missing trial is blanked across ALL dimensions, so any-NaN == all-NaN.
    missing = np.isnan(y)
    if not np.array_equal(missing.any(axis=0), missing.all(axis=0)):
        raise AssertionError(
            "fixture %s has partially-NaN feedback columns; scenarioBattery.m "
            "only blanks whole trials" % path
        )
    expected_missing = _expected_missing_mask(trials, missing_frac)
    if not np.array_equal(missing.any(axis=0), expected_missing):
        raise AssertionError(
            "fixture %s missing-trial cadence disagrees with the re-derived "
            "pattern (fixture=%s, expected=%s)"
            % (
                path,
                np.flatnonzero(missing.any(axis=0)).tolist(),
                np.flatnonzero(expected_missing).tolist(),
            )
        )

    seeds = np.asarray(raw["seeds"], dtype=np.int64).reshape(-1)
    runs = int(seeds.size)

    matlab = {
        "motor": _as_dim_trial(raw["motor"], dim, trials, "motor"),
        "state_mean": _as_dim_trial(raw["stateMean"], dim, trials, "stateMean"),
    }
    for key, var in (
        ("pred_ctx", "predCtx"),
        ("resp", "resp"),
        ("counts", "counts"),
    ):
        array = np.asarray(raw[var], dtype=float)
        if array.ndim != 3 or array.shape[1:] != (trials, runs):
            raise AssertionError(
                "fixture %s variable %r has shape %s; expected (L, %d, %d)"
                % (path, var, array.shape, trials, runs)
            )
        if array.shape[0] < max_contexts + 1:
            raise AssertionError(
                "fixture %s variable %r has capacity %d < max_contexts + 1 = %d"
                % (path, var, array.shape[0], max_contexts + 1)
            )
        matlab[key] = array

    for key in ("motor", "state_mean"):
        if matlab[key].shape[2] != runs:
            raise AssertionError(
                "fixture %s variable %r has %d runs, seeds has %d"
                % (path, key, matlab[key].shape[2], runs)
            )

    return {
        "name": name,
        "seed": int(seed),
        "dim": int(dim),
        "trials": int(trials),
        "cues": int(cues),
        "missing_frac": float(missing_frac),
        "args": {
            "num_particles": int(particles),
            "max_contexts": int(max_contexts),
            "state_dim": int(dim),
        },
        "q": q,
        "y": y,
        "matlab_seeds": seeds,
        "matlab": matlab,
    }


def scenario_battery(fixture_dir=None):
    """Load every scenario in battery order.

    Parameters
    ----------
    fixture_dir : path-like or None, optional
        Directory searched; ``None`` uses :data:`FIXTURE_DIR`.

    Returns
    -------
    list of dict
        One :func:`load_scenario` result per entry of :data:`SPECS`.
    """
    return [load_scenario(name, fixture_dir) for name in SCENARIO_NAMES]

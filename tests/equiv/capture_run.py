"""Replay one scenario and record the per-trial observable surface.

Port of ``tests/+equiv/captureRun.m``.

The recorded surface is exactly the one the MATLAB capture used, so the two
languages' recordings line up field for field:

===============  ====================================================
field            MATLAB query / Python method
===============  ====================================================
``motor``        ``motor_output()``
``pred_ctx``     ``predicted_context_probabilities_vector()``
``resp``         ``responsibilities_vector()``
``counts``       ``sampled_context_count()``
``state_mean``   first output of ``state_moments()``
===============  ====================================================

Note that these are five recorded FIELDS but not five independent checks:
``motor`` and ``state_mean`` coincide numerically in this model (the predictive
feedback mean equals the predictive state mean with no bias term and an identity
observation map - measured agreement <= 7e-16 in every fixture). Both are
recorded because the MATLAB capture records both and they exercise distinct code
paths, but they carry the same information.

Loop-order fidelity matters and is preserved verbatim: for every trial
``observe_q(q[t])``, then ``observe_y(y[:, t])``, THEN the five queries. Reading
the queries before ``observe_y`` would sample the prior rather than the
posterior and would not be comparable with the fixture.

Query-name mapping. MATLAB's ``predicted_context_probabilities`` /
``responsibilities`` return a numeric row vector, which ``captureRun`` columnises;
the Python package splits the two return shapes into ``*_vector`` (array) and
``*_map`` (dict) methods, so the ``*_vector`` forms are the faithful
counterparts. ``sampled_context_count`` has the same name in both.

Where MATLAB's ``captureRun`` swallows a query error into ``NaN`` (via
``firstOutput``), this port lets the exception propagate: a query that cannot be
evaluated is a bug worth surfacing, and the MATLAB guard existed only to keep a
deprecation-shim rename from aborting a capture.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOIN

__all__ = ["FIELDS", "capture_run"]

#: Recorded fields, in the order the MATLAB capture lists them.
FIELDS = ("motor", "pred_ctx", "resp", "counts", "state_mean")


def _feedback_argument(column):
    """Convert one feedback column into the argument ``observe_y`` expects.

    Parameters
    ----------
    column : numpy.ndarray
        ``(dim,)`` feedback for one trial; all-NaN marks a missing trial.

    Returns
    -------
    float or numpy.ndarray
        A plain ``float`` when ``dim == 1`` (so the scalar branch sees MATLAB's
        scalar, not a length-1 array), otherwise the array unchanged. NaN is
        passed through rather than converted to ``None``: ``observe_y``
        interprets it as a missing observation itself, and for the
        multi-dimensional path the per-entry NaN pattern is what builds the
        observation mask.
    """
    if column.size == 1:
        return float(column[0])
    return column


def capture_run(scenario, seed):
    """Stream a scenario through ``RealTimeCOIN`` and record the surface.

    Parameters
    ----------
    scenario : dict
        A :func:`tests.equiv.scenario_battery.load_scenario` result. Only
        ``args``, ``q``, ``y``, ``dim`` and ``trials`` are used.
    seed : int or numpy.random.SeedSequence or numpy.random.Generator
        Passed straight to the model's ``rng`` keyword, i.e. through
        ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``motor`` and ``state_mean`` with shape ``(dim, T)``; ``pred_ctx``,
        ``resp`` and ``counts`` with shape ``(max_contexts + 1, T)``. Every
        entry is a plain float; nothing is NaN-padded (the Python queries always
        return a full-capacity vector).

    Notes
    -----
    Cue values are passed through as the fixture's RAW 1-based MATLAB labels.
    The package's cue registry maps distinct raw values to consecutive internal
    labels, so only the IDENTITY of repeated values matters, not their
    numeric value.
    """
    model = RealTimeCOIN(rng=seed, **scenario["args"])

    dim = int(scenario["dim"])
    trials = int(scenario["trials"])
    contexts = int(scenario["args"]["max_contexts"]) + 1
    q = scenario["q"]
    y = scenario["y"]

    motor = np.empty((dim, trials), dtype=float)
    state_mean = np.empty((dim, trials), dtype=float)
    pred_ctx = np.empty((contexts, trials), dtype=float)
    resp = np.empty((contexts, trials), dtype=float)
    counts = np.empty((contexts, trials), dtype=float)

    for t in range(trials):
        model.observe_q(int(q[t]))
        model.observe_y(_feedback_argument(y[:, t]))

        motor[:, t] = np.asarray(model.motor_output(), dtype=float).reshape(-1)
        pred_ctx[:, t] = np.asarray(
            model.predicted_context_probabilities_vector(), dtype=float
        ).reshape(-1)
        resp[:, t] = np.asarray(
            model.responsibilities_vector(), dtype=float
        ).reshape(-1)
        counts[:, t] = np.asarray(
            model.sampled_context_count(), dtype=float
        ).reshape(-1)
        mean, _cov = model.state_moments()
        state_mean[:, t] = np.asarray(mean, dtype=float).reshape(-1)

    return {
        "motor": motor,
        "pred_ctx": pred_ctx,
        "resp": resp,
        "counts": counts,
        "state_mean": state_mean,
    }

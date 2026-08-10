"""Recovery of known latent contexts from synthetic data.

Translated from ``validation/validate_context_recovery.m``.

Synthetic data gives us the hidden context ``c_t``, but mixture labels are
arbitrary: an inferred context labelled ``2`` may correspond to the true context
labelled ``0``. Context recovery is therefore scored only AFTER finding the
label permutation that minimises mismatch with the true labels
(:func:`validation.best_label_map.best_label_map`), which separates genuine
inference failure from harmless label switching.

Seed averaging
--------------
The recovery accuracy sits close to its 0.65 gate, so a single-seed run is
seed-sensitive: the same code can pass or fail purely from which particle-filter
noise draw it happened to use. To remove that flip the experiment is run over a
small set of seeds derived from ``seed`` (``seed + 0 .. num_seeds-1``), the
recovery metrics are computed on each, and the gates act on the ACROSS-SEED
AVERAGE of accuracy, true-context posterior mass and mean recovery lag. The
top-level metric fields hold the seed-averaged values, per-seed detail is under
``results["per_seed"]``, and the trace fields (cues / feedback / inferred /
matched contexts) come from the first (representative) seed.

Deviations from the MATLAB source
---------------------------------
* Labels are 0-based throughout (contexts ``0``/``1``, inferred labels from
  ``argmax``), where the ``.m`` is 1-based. Raw CUE VALUES are still ``1`` and
  ``2`` as in the ``.m``; the model's cue registry maps raw values to internal
  labels in order of first appearance, so the raw values are cosmetic.
* ``mapping`` is a plain ``dict`` rather than a ``containers.Map``, so the
  posterior-mass loop indexes it directly instead of testing ``isKey``.
* Synthetic data comes from a local :class:`numpy.random.Generator` and each run
  seeds its own model generator, instead of MATLAB's single global stream.
"""

from __future__ import annotations

import warnings

import numpy as np

from realtimecoin import RealTimeCOIN

from validation.best_label_map import best_label_map
from validation.pass_summary import pass_summary

__all__ = ["run", "THRESHOLDS"]

#: Gate rationale (verbatim from the ``.m``): these are qualitative recovery
#: gates, not calibration gates, so they are intentionally generous.
#: ``context_accuracy`` (0.65) is the best-relabelled hard-assignment agreement;
#: ``posterior_true_context`` (0.45) is the softer average posterior mass on the
#: true context; ``mean_recovery_lag`` (<= 20 trials) bounds how quickly the
#: filter re-locks after a context switch. Averaging over ``num_seeds`` seeds
#: removes the cross-seed flips that used to occur right at the accuracy
#: boundary.
THRESHOLDS = {
    "context_accuracy": 0.65,
    "posterior_true_context": 0.45,
    "mean_recovery_lag": 20,
}

_DEFAULTS = {
    "trials": 120,
    "particles": 150,
    "num_seeds": 5,
}

# --- Data-generating process (transliterated from the .m) ------------------
_A = np.array([0.92, 0.90])         # per-context retention
_D = np.array([0.000, 0.045])       # per-context drift
_SIGMA_Q = 0.025                    # process-noise standard deviation
_SIGMA_R = 0.045                    # sensory-noise standard deviation
_CUE_PROB = np.array([[0.88, 0.12], [0.12, 0.88]])      # p(cue | context)


def run(seed=1401, strict=False, **overrides):
    """Run the seed-averaged context-recovery validator.

    Parameters
    ----------
    seed : int, optional
        Base seed; run ``i`` uses ``seed + i``. Default 1401 (the MATLAB
        default).
    strict : bool, optional
        Raise instead of returning ``passed=False``. Default ``False``.
    **overrides
        ``trials`` (120), ``particles`` (150), ``num_seeds`` (5).

    Returns
    -------
    dict
        ``context_accuracy``, ``mean_posterior_true_context``,
        ``recovery_lags``, ``mean_recovery_lag``,
        ``mean_inferred_context_count`` (all seed-averaged), the representative
        seed's ``true_context`` / ``inferred_context`` / ``matched_context`` /
        ``posterior_true_context`` / ``feedback`` / ``cues`` traces, plus
        ``thresholds``, ``checks``, ``passed``, ``per_seed``, ``seeds`` and
        ``config``.

    Raises
    ------
    RuntimeError
        If ``strict`` and any gate fails.
    """
    cfg = _config(seed, strict, overrides)
    seeds = [cfg["seed"] + i for i in range(cfg["num_seeds"])]

    per_seed = [_run_one_seed(s, cfg) for s in seeds]

    # Across-seed averages: these are what the gates act on.
    avg_accuracy = float(np.mean([r["accuracy"] for r in per_seed]))
    avg_posterior = float(
        np.mean([r["mean_posterior_true_context"] for r in per_seed])
    )
    avg_inferred_count = float(
        np.mean([r["mean_inferred_context_count"] for r in per_seed])
    )

    # (num_seeds, num_switches); the switch times are deterministic, so every
    # row has the same length.
    lag_matrix = np.vstack([r["recovery_lags"] for r in per_seed])
    avg_lags = _nanmean(lag_matrix, axis=0)
    avg_mean_lag = _nanmean(
        np.asarray([r["mean_recovery_lag"] for r in per_seed], dtype=float)
    )

    checks = {
        "context_accuracy": avg_accuracy > THRESHOLDS["context_accuracy"],
        "posterior_true_context": avg_posterior
        > THRESHOLDS["posterior_true_context"],
        "mean_recovery_lag": avg_mean_lag <= THRESHOLDS["mean_recovery_lag"],
    }
    passed, checks = pass_summary(checks)

    rep = per_seed[0]           # representative seed for the trace fields
    results = {
        "context_accuracy": avg_accuracy,
        "mean_posterior_true_context": avg_posterior,
        "recovery_lags": avg_lags,
        "mean_recovery_lag": avg_mean_lag,
        "mean_inferred_context_count": avg_inferred_count,
        "true_context": rep["true_context"],
        "inferred_context": rep["inferred_context"],
        "matched_context": rep["matched_context"],
        "posterior_true_context": rep["posterior_true_context"],
        "feedback": rep["feedback"],
        "cues": rep["cues"],
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": passed,
        "per_seed": per_seed,
        "seeds": seeds,
        "config": cfg,
    }

    print(
        "Context recovery: accuracy %.3f, true-context posterior %.3f, "
        "mean lag %.1f trials" % (avg_accuracy, avg_posterior, avg_mean_lag)
    )

    if cfg["strict"] and not passed:
        raise RuntimeError(
            "validate_context_recovery:Failed: context recovery validation "
            "failed (%s)." % _failed_names(checks)
        )
    return results


def _run_one_seed(seed, cfg):
    """Execute one context-recovery experiment at a fixed seed.

    Parameters
    ----------
    seed : int
        Seed for both the synthetic data and the model.
    cfg : dict
        Resolved configuration.

    Returns
    -------
    dict
        ``seed``, ``accuracy``, ``mean_posterior_true_context``,
        ``recovery_lags``, ``mean_recovery_lag``,
        ``mean_inferred_context_count``, ``true_context``,
        ``inferred_context``, ``matched_context``, ``posterior_true_context``,
        ``feedback`` and ``cues``.
    """
    trials = cfg["trials"]
    data_rng = np.random.default_rng(seed)
    true_context = _synthetic_context_sequence(trials)

    model = RealTimeCOIN(
        num_particles=cfg["particles"],
        max_contexts=4,
        prior_mean_retention=0.9,
        prior_precision_retention=200,
        prior_mean_drift=0.015,
        prior_precision_drift=80,
        sigma_process_noise=_SIGMA_Q,
        sigma_sensory_noise=_SIGMA_R,
        sigma_motor_noise=0.0,
        # [seed, 1], not seed: the model must not share a stream with data_rng.
        rng=np.random.default_rng([seed, 1]),
    )

    state = 0.0
    c_slots = model.max_contexts + 1
    cues = np.zeros(trials, dtype=int)
    feedback = np.zeros(trials)
    inferred = np.zeros(trials, dtype=int)
    resp_history = np.zeros((c_slots, trials))
    inferred_count = np.zeros(trials)

    for t in range(trials):
        c = int(true_context[t])
        cues[t] = _sample_categorical(data_rng, _CUE_PROB[c]) + 1
        state = _A[c] * state + _D[c] + _SIGMA_Q * data_rng.standard_normal()
        feedback[t] = state + _SIGMA_R * data_rng.standard_normal()

        model.observe_q(int(cues[t]))
        model.observe_y(float(feedback[t]))

        diagnostics = model.diagnostics()
        # (P_modal, C) -> (C,): the particle-averaged responsibility of each
        # ALIGNED global context.
        resp = np.mean(diagnostics["responsibilities"], axis=0)
        resp_history[: resp.size, t] = resp
        inferred[t] = int(np.argmax(resp))
        inferred_count[t] = diagnostics["C"]

    mapped, accuracy, mapping = best_label_map(true_context, inferred)

    # Posterior mass on whichever inferred label the optimal map sends to the
    # true context of that trial.
    posterior_true = np.zeros(trials)
    for t in range(trials):
        for inferred_label in range(c_slots):
            if mapping.get(inferred_label, None) == true_context[t]:
                posterior_true[t] += resp_history[inferred_label, t]

    switches = np.nonzero(np.diff(true_context) != 0)[0] + 1
    lags = _recovery_lags(mapped, true_context, switches)

    return {
        "seed": seed,
        "accuracy": accuracy,
        "mean_posterior_true_context": float(np.mean(posterior_true)),
        "recovery_lags": lags,
        "mean_recovery_lag": _nanmean(lags),
        "mean_inferred_context_count": float(np.mean(inferred_count)),
        "true_context": true_context,
        "inferred_context": inferred,
        "matched_context": mapped,
        "posterior_true_context": posterior_true,
        "feedback": feedback,
        "cues": cues,
    }


def _synthetic_context_sequence(trials):
    """Deterministic A-B-A context schedule.

    Parameters
    ----------
    trials : int
        Number of trials ``T``.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` 0-based context labels: ``floor(T/3)`` trials of context 0,
        the same of context 1, then the remainder back in context 0.
    """
    third = trials // 3
    return np.concatenate(
        [
            np.zeros(third, dtype=int),
            np.ones(third, dtype=int),
            np.zeros(trials - 2 * third, dtype=int),
        ]
    )


def _recovery_lags(mapped, true_context, switches):
    """Trials taken to re-lock onto the true context after each switch.

    Parameters
    ----------
    mapped : array_like
        ``(T,)`` relabelled inferred contexts.
    true_context : array_like
        ``(T,)`` true contexts.
    switches : array_like
        Indices of the trials on which the true context changes.

    Returns
    -------
    numpy.ndarray
        ``(len(switches),)`` lags in trials (0 = recovered immediately),
        ``nan`` where the target context is never reached again.
    """
    mapped = np.asarray(mapped, dtype=float)
    true_context = np.asarray(true_context)
    lags = np.full(len(switches), np.nan)
    for i, t0 in enumerate(switches):
        target = true_context[t0]
        hits = np.nonzero(mapped[t0:] == target)[0]
        if hits.size:
            lags[i] = float(hits[0])
    return lags


def _sample_categorical(rng, p):
    """Draw one 0-based index from unnormalised probabilities.

    Inverse-CDF draw, mirroring ``validation_sample_categorical.m``. Kept local
    rather than imported from unit C3's ``validation/sample_categorical.py``
    because that module's generator-passing signature is not pinned.

    Parameters
    ----------
    rng : numpy.random.Generator
        Generator supplying the single uniform draw.
    p : array_like
        Non-negative weights; an all-zero vector becomes uniform.

    Returns
    -------
    int
        The sampled 0-based index.
    """
    weights = np.asarray(p, dtype=float).reshape(-1).copy()
    weights[~np.isfinite(weights) | (weights < 0)] = 0.0
    total = weights.sum()
    weights = (
        np.full(weights.size, 1.0 / weights.size) if total <= 0 else weights / total
    )
    index = int(np.searchsorted(np.cumsum(weights), rng.random(), side="left"))
    return min(index, weights.size - 1)


def _nanmean(values, axis=None):
    """Mean ignoring ``nan`` (MATLAB's ``mean(..., 'omitnan')``).

    Parameters
    ----------
    values : array_like
        Sample.
    axis : int or None, optional
        Reduction axis.

    Returns
    -------
    float or numpy.ndarray
        The mean of the finite entries; ``nan`` where a slice is all-``nan``.
        numpy warns on an all-``nan`` slice and MATLAB does not, so the warning
        is suppressed to keep the two behaviours identical.
    """
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(values, axis=axis)
    return float(out) if axis is None else out


def _failed_names(checks):
    """Comma-separated names of the failing checks (for error messages)."""
    return ", ".join(name for name, ok in checks.items() if not ok) or "none"


def _config(seed, strict, overrides):
    """Resolve the run configuration, rejecting unknown overrides.

    Parameters
    ----------
    seed : int
        Base seed.
    strict : bool
        Strict flag.
    overrides : dict
        Caller overrides; keys must be a subset of :data:`_DEFAULTS`.

    Returns
    -------
    dict
        The resolved configuration.

    Raises
    ------
    TypeError
        If an override key is not recognised.
    """
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise TypeError(
            "validate_context_recovery.run() got unexpected keyword "
            "argument(s): %s." % ", ".join(sorted(unknown))
        )
    cfg = dict(_DEFAULTS)
    cfg.update(overrides)
    cfg["trials"] = int(cfg["trials"])
    cfg["particles"] = int(cfg["particles"])
    cfg["num_seeds"] = int(cfg["num_seeds"])
    cfg["seed"] = int(seed)
    cfg["strict"] = bool(strict)
    return cfg


if __name__ == "__main__":       # pragma: no cover
    # Invoke as ``python -m validation.validate_context_recovery`` from
    # ``python/``: this is a namespace package, so running the file by path
    # would not put the package root on sys.path.
    run()

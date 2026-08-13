"""Hand-designed behavioural stress checks.

Translated from ``validation/validate_stress_cases.m``.

These cases are not proofs of calibration. They are deliberately interpretable
probes of qualitative behaviour: stable data should not create many contexts,
abrupt changes should increase contextual uncertainty or create a new context,
A-B-A data should tend to reuse an old context, and sensory-noise settings should
change posterior uncertainty in the expected direction.

Deviations from the MATLAB source
---------------------------------
* Raw cue values stay ``1 .. 4`` as in the ``.m``; the model's registry maps
  them to 0-based internal labels in order of first appearance. Reported
  ``top_context`` values are 0-based aligned global labels.
* Each case builds its own :class:`numpy.random.Generator` (seeded
  ``seed + offset``, mirroring the ``.m``'s per-case ``rng`` call) for the
  observation noise, and seeds the model's generator identically.
* Block edges are computed with MATLAB's round-half-away-from-zero rule
  (``floor(x + 0.5)``) rather than numpy's round-half-to-even, so the block
  boundaries match the ``.m`` exactly.
* ``mean(state_filtered_var(i_observed))`` becomes a gather with the 0-based
  per-particle context index, because ``i_observed`` here is that index rather
  than MATLAB's linear index into a contexts-by-particles plane.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOIN

from validation.pass_summary import pass_summary

__all__ = ["run", "THRESHOLDS"]

#: Qualitative behavioural gates. ``single_context_mean_count`` bounds context
#: proliferation on stationary data, ``abrupt_creates_context`` demands that a
#: step change instantiates a second context, ``cap_max_contexts`` checks the
#: ``max_contexts`` cap is honoured under pressure from four distinct blocks, and
#: ``high_noise_uncertainty_ratio`` demands the posterior state variance grow by
#: at least 25% when the sensory noise grows eightfold.
THRESHOLDS = {
    "single_context_mean_count": 1.5,
    "abrupt_creates_context": 1.5,
    "cap_max_contexts": 2,
    "high_noise_uncertainty_ratio": 1.25,
}

_DEFAULTS = {
    "trials": 90,
    "particles": 100,
}

EPS = float(np.finfo(float).eps)


def run(seed=1501, strict=False, **overrides):
    """Run the behavioural stress-case validator.

    Parameters
    ----------
    seed : int, optional
        Base seed; case ``i`` uses ``seed + i``. Default 1501 (the MATLAB
        default).
    strict : bool, optional
        Raise instead of returning ``passed=False``. Default ``False``.
    **overrides
        ``trials`` (90), ``particles`` (100).

    Returns
    -------
    dict
        One sub-dict per case - ``single_context_no_cue_change``,
        ``abrupt_perturbation_switch``, ``aba_repeated_context``,
        ``uninformative_cues``, ``misleading_cues``, ``high_sensory_noise``,
        ``low_sensory_noise``, ``max_context_cap`` - plus ``thresholds``,
        ``checks``, ``passed`` and ``config``.

    Raises
    ------
    RuntimeError
        If ``strict`` and any gate fails.
    """
    cfg = _config(seed, strict, overrides)
    trials = cfg["trials"]

    single = _run_case(
        _constant_signal(trials, 0.0), np.ones(trials, dtype=int), cfg, 1
    )
    abrupt = _run_case(
        _block_signal(trials, [0.0, 0.4]), _block_cues(trials, [1, 2]), cfg, 2
    )
    aba = _run_case(
        _block_signal(trials, [0.0, 0.4, 0.0]),
        _block_cues(trials, [1, 2, 1]),
        cfg, 3,
    )
    uninformative = _run_case(
        _block_signal(trials, [0.0, 0.35]),
        _random_cues(trials, cfg["seed"] + 4),
        cfg, 4,
    )
    misleading = _run_case(
        _block_signal(trials, [0.0, 0.35]), _block_cues(trials, [2, 1]), cfg, 5
    )
    high_noise = _run_noise_case(cfg, 0.12, 6)
    low_noise = _run_noise_case(cfg, 0.015, 7)
    cap = _run_case(
        _block_signal(trials, [0.0, 0.3, -0.2, 0.45]),
        _block_cues(trials, [1, 2, 3, 4]),
        cfg, 8, max_contexts=2,
    )

    checks = {
        "single_context_mean_count": single["mean_context_count"]
        <= THRESHOLDS["single_context_mean_count"],
        "abrupt_creates_context": abrupt["max_context_count"]
        >= THRESHOLDS["abrupt_creates_context"],
        "cap_respected": cap["max_context_count"]
        <= THRESHOLDS["cap_max_contexts"],
        "noise_uncertainty": high_noise["mean_state_variance"]
        > THRESHOLDS["high_noise_uncertainty_ratio"]
        * low_noise["mean_state_variance"],
    }
    passed, checks = pass_summary(checks)

    results = {
        "single_context_no_cue_change": single,
        "abrupt_perturbation_switch": abrupt,
        "aba_repeated_context": aba,
        "uninformative_cues": uninformative,
        "misleading_cues": misleading,
        "high_sensory_noise": high_noise,
        "low_sensory_noise": low_noise,
        "max_context_cap": cap,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": passed,
        "config": cfg,
    }

    print(
        "Stress cases: single mean K %.2f, abrupt max K %.0f, "
        "high/low var ratio %.2f"
        % (
            single["mean_context_count"],
            abrupt["max_context_count"],
            high_noise["mean_state_variance"]
            / max(low_noise["mean_state_variance"], EPS),
        )
    )

    if cfg["strict"] and not passed:
        raise RuntimeError(
            "validate_stress_cases:Failed: stress-case validation failed (%s)."
            % _failed_names(checks)
        )
    return results


def _run_case(signal, cues, cfg, seed_offset, max_contexts=5):
    """Run one cued stress case and summarise its behaviour.

    Parameters
    ----------
    signal : numpy.ndarray
        ``(T,)`` noise-free perturbation schedule.
    cues : numpy.ndarray
        ``(T,)`` raw cue values presented alongside the signal.
    cfg : dict
        Resolved configuration.
    seed_offset : int
        Added to ``cfg["seed"]`` for this case.
    max_contexts : int, optional
        Context cap of the model under test. Default 5.

    Returns
    -------
    dict
        ``mean_context_count``, ``max_context_count``, ``final_context_count``,
        ``mean_state_variance``, ``mean_prediction_error``, ``context_count``,
        ``top_context`` and ``reuses_initial_context_in_final_block``.
    """
    seed = cfg["seed"] + seed_offset
    data_rng = np.random.default_rng(seed)
    model = RealTimeCOIN(
        num_particles=cfg["particles"],
        max_contexts=max_contexts,
        prior_mean_retention=0.92,
        prior_precision_retention=300,
        prior_mean_drift=0.0,
        prior_precision_drift=100,
        sigma_process_noise=0.025,
        sigma_sensory_noise=0.04,
        sigma_motor_noise=0.0,
        # [seed, 1], not seed: the model must not share a stream with data_rng.
        rng=np.random.default_rng([seed, 1]),
    )

    trials = signal.size
    context_count = np.zeros(trials)
    top_context = np.zeros(trials, dtype=int)
    state_variance = np.zeros(trials)
    prediction_error = np.zeros(trials)

    for t in range(trials):
        y = float(signal[t] + 0.04 * data_rng.standard_normal())
        cue = int(cues[t])
        # Read the model's one-step prediction BEFORE it sees the trial.
        prediction = model.predictive_motor_output(cue)
        model.observe_q(cue)
        model.observe_y(y)

        diagnostics = model.diagnostics()
        resp = np.mean(diagnostics["responsibilities"], axis=0)      # (C,)
        top_context[t] = int(np.argmax(resp))
        context_count[t] = diagnostics["C"]
        state_variance[t] = _mean_observed_state_variance(model)
        prediction_error[t] = abs(y - prediction)

    return {
        "mean_context_count": float(np.mean(context_count)),
        "max_context_count": float(np.max(context_count)),
        "final_context_count": float(context_count[-1]),
        "mean_state_variance": _nanmean(state_variance),
        "mean_prediction_error": _nanmean(prediction_error),
        "context_count": context_count,
        "top_context": top_context,
        "reuses_initial_context_in_final_block": _final_block_reuses_initial(
            top_context
        ),
    }


def _run_noise_case(cfg, sensory_noise, seed_offset):
    """Run one un-cued case at a given sensory-noise level.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    sensory_noise : float
        Sensory-noise standard deviation used BOTH to generate the data and to
        configure the model, so the model is correctly specified.
    seed_offset : int
        Added to ``cfg["seed"]`` for this case.

    Returns
    -------
    dict
        ``mean_context_count``, ``max_context_count``, ``final_context_count``,
        ``mean_state_variance`` and ``context_count``.
    """
    seed = cfg["seed"] + seed_offset
    data_rng = np.random.default_rng(seed)
    model = RealTimeCOIN(
        num_particles=cfg["particles"],
        max_contexts=3,
        sigma_process_noise=0.025,
        sigma_sensory_noise=sensory_noise,
        sigma_motor_noise=0.0,
        # [seed, 1], not seed: the model must not share a stream with data_rng.
        rng=np.random.default_rng([seed, 1]),
    )

    trials = cfg["trials"]
    state_variance = np.zeros(trials)
    context_count = np.zeros(trials)
    for t in range(1, trials + 1):      # 1-based t, as in the .m's sin(t/12)
        y = float(
            0.1 * np.sin(t / 12.0) + sensory_noise * data_rng.standard_normal()
        )
        model.observe_q(1)
        model.observe_y(y)
        diagnostics = model.diagnostics()
        state_variance[t - 1] = _mean_observed_state_variance(model)
        context_count[t - 1] = diagnostics["C"]

    return {
        "mean_context_count": float(np.mean(context_count)),
        "max_context_count": float(np.max(context_count)),
        "final_context_count": float(context_count[-1]),
        "mean_state_variance": _nanmean(state_variance),
        "context_count": context_count,
    }


def _mean_observed_state_variance(model):
    """Particle-mean filtered state variance of the currently observed context.

    MATLAB writes ``mean(raw.state_filtered_var(raw.i_observed))`` using linear
    indices into a contexts-by-particles plane; this package stores
    ``i_observed`` as the ``(P,)`` array of 0-based context columns, so the same
    gather is spelled with an explicit particle index.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Model whose particle state is read.

    Returns
    -------
    float
        The particle-averaged variance.
    """
    d = model.D
    particles = np.arange(model.num_particles)
    return float(np.mean(d.state_filtered_var[particles, d.i_observed]))


# --------------------------------------------------------------------------
# Signal / cue schedules
# --------------------------------------------------------------------------


def _constant_signal(trials, value):
    """``(T,)`` constant signal."""
    return np.full(trials, float(value))


def _block_signal(trials, values):
    """``(T,)`` piecewise-constant signal split into equal-length blocks.

    Parameters
    ----------
    trials : int
        Number of trials ``T``.
    values : sequence of float
        One value per block.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` signal.

    Notes
    -----
    Block edges use MATLAB's round-half-AWAY-from-zero (``floor(x + 0.5)``)
    rather than numpy's round-half-to-even, so the boundaries land exactly where
    ``round(linspace(1, T + 1, n + 1))`` puts them in the ``.m``.
    """
    values = np.asarray(values, dtype=float)
    edges = np.floor(np.linspace(0.0, trials, values.size + 1) + 0.5).astype(int)
    signal = np.zeros(trials)
    for i, value in enumerate(values):
        signal[edges[i]:edges[i + 1]] = value
    return signal


def _block_cues(trials, values):
    """``(T,)`` integer cue schedule with one raw cue value per block."""
    return np.round(_block_signal(trials, values)).astype(int)


def _random_cues(trials, seed):
    """``(T,)`` cue schedule of independent draws from ``{1, 2}``.

    Parameters
    ----------
    trials : int
        Number of trials.
    seed : int
        Base seed of the case. The ``.m`` draws the cue schedule from the same
        global stream as that case's observation noise; here the schedule is
        built up front, so it uses a generator seeded ``[seed, 2]`` - distinct
        from the case's noise generator (``seed``) AND from its model generator
        (``[seed, 1]``). The extra entropy word is load-bearing: seeding it with
        ``seed`` alone would make the cue schedule a deterministic function of
        the very noise sequence the case then draws, correlating cue and
        feedback.

    Returns
    -------
    numpy.ndarray
        ``(T,)`` raw cue values.
    """
    rng = np.random.default_rng([seed, 2])
    return 1 + (rng.random(trials) > 0.5).astype(int)


def _final_block_reuses_initial(top_context):
    """Whether the modal context of the last third matches the first third.

    Parameters
    ----------
    top_context : numpy.ndarray
        ``(T,)`` per-trial highest-responsibility context labels.

    Returns
    -------
    bool
        ``True`` when the two modes agree. Ties resolve to the SMALLEST label,
        matching MATLAB's ``mode``. An empty block yields ``False``, because
        MATLAB's ``mode([]) == mode([])`` is ``NaN == NaN``, i.e. false (only
        reachable for ``trials < 3``).
    """
    trials = top_context.size
    third = trials // 3
    first_block = top_context[:third]
    last_block = top_context[2 * third:]
    if first_block.size == 0 or last_block.size == 0:
        return False
    return bool(_mode(first_block) == _mode(last_block))


def _mode(labels):
    """Most frequent non-negative integer label, smallest on a tie.

    ``np.argmax(np.bincount(...))`` returns the smallest index among ties, which
    is exactly MATLAB's ``mode`` tie rule. Callers must pass a non-empty array.
    """
    return int(np.argmax(np.bincount(np.asarray(labels, dtype=int))))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _nanmean(values):
    """Mean ignoring ``nan`` (MATLAB's ``mean(..., 'omitnan')``).

    ``nan`` ONLY - infinities propagate, as they do in MATLAB. Empty and
    all-``nan`` inputs short-circuit so numpy never warns where MATLAB would
    not.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return float("nan")
    return float(np.nanmean(values))


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
            "validate_stress_cases.run() got unexpected keyword argument(s): "
            "%s." % ", ".join(sorted(unknown))
        )
    cfg = dict(_DEFAULTS)
    cfg.update(overrides)
    cfg["trials"] = int(cfg["trials"])
    cfg["particles"] = int(cfg["particles"])
    cfg["seed"] = int(seed)
    cfg["strict"] = bool(strict)
    return cfg


if __name__ == "__main__":       # pragma: no cover
    # Invoke as ``python -m validation.validate_stress_cases`` from the repo root:
    # this is a namespace package, so running the file by path would not put the
    # package root on sys.path.
    run()

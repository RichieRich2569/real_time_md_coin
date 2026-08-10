"""Posterior predictive calibration diagnostics (PIT self-calibration).

Translated from ``validation/validate_p_values_extended.m``.

For a calibrated continuous predictive CDF ``F_t`` the probability integral
transform (PIT)::

    p_t = F_t(x_t)

is Uniform(0, 1) over repeated draws from the data-generating process. Cues are
discrete, so this validator uses the RANDOMISED discrete PIT
``F(q_t-) + u * Pr(q_t)``, which removes the atom-induced non-uniformity of a
plain discrete CDF. State and parameter ranks are included as posterior
diagnostics; the feedback and cue PITs are the primary prequential calibration
checks.

The data-generating process is a 2-context HMM: the latent context drives the
CUE emissions, while a single AR(1) latent state evolves independently of the
context (that is what the ``.m`` does - ``context_offset`` only seeds the
initial state; see :func:`_synthetic_stream`).

Deviations from the MATLAB source
---------------------------------
``validate_p_values_extended.m`` scores each trial with the class queries
``coin.predictive_state_feedback_cdf(y, q)`` and ``coin.predictive_cue_p_value(q)``.
Both queries exist in this package (:mod:`realtimecoin.queries_density`) and are
gated directly by :mod:`validation.validate_p_values`; here the two PITs are
recomputed INDEPENDENTLY instead, so this validator is a second implementation
rather than a re-run of the production one:

* **Feedback PIT** - :func:`_predictive_feedback_components` rebuilds the
  predictive feedback MIXTURE (per-particle, per-context weights, means and
  variances) from the raw particle state exactly as
  ``validation_predictive_feedback_moments.m`` rebuilds it, then evaluates the
  Gaussian-mixture CDF at ``y`` instead of collapsing the mixture to two
  moments. Collapsing first would be WRONG for a PIT: the predictive
  distribution is genuinely multi-modal across contexts, and a single Gaussian
  with the same mean and variance has a different CDF. The resulting value is
  algebraically identical to ``predictive_state_feedback_cdf.m``
  (``sum(W .* normal_cdf(y, M, V)) / P``; dividing by ``P`` and normalising the
  weights by their total agree because each particle's weight row sums to one)
  except in two corners, both unreachable at these priors:

  1. the novel-slot fallback is written ``d / max(1 - a, eps)`` and
     ``Q / max(1 - a^2, eps)`` after ``validation_predictive_feedback_moments.m``,
     whereas the class path's
     :func:`realtimecoin.numerics.stationary_state_mean` returns 0 for a
     retention factor within ``eps`` of 1;
  2. a degenerate (all-zero) weight row falls back to UNIFORM ``1/C`` after the
     same ``.m`` helper, whereas the class path's
     :func:`realtimecoin.numerics.normalize_columns` collapses it onto ``e_0``.

  Measured agreement with the class path
  (:func:`realtimecoin.queries_core.preview_predictive_feedback`) is 2e-16.
* **Cue PIT** - :func:`_predictive_cue_p_value` applies the randomised discrete
  PIT to the pmf from :func:`realtimecoin.context.preview_cue_pmf` (which is
  implemented), transliterating ``predictive_cue_p_value.m`` including its
  zero-mass treatment of a never-seen cue label.
* The raw particle state is read as ``model.D`` rather than as
  ``model.diagnostics()["raw"]``. They are the same object, but going through
  ``diagnostics()`` would trigger the lazy cross-particle context alignment
  (expensive) for a quantity that does not need it. The aligned ``diagnostics()``
  IS used for the retention/drift ranks, which are per-global-context.
* Synthetic data and the PIT randomisation ``u`` are drawn from a local
  :class:`numpy.random.Generator`, and each dataset's model gets its own
  generator seeded from ``(seed, dataset)``. MATLAB seeds one global stream, so
  the numbers differ; the distributions being tested do not.
* ``make_plots`` is accepted for signature compatibility and ignored - this
  package has no plotting dependency.
"""

from __future__ import annotations

import numpy as np

from realtimecoin import RealTimeCOIN
from realtimecoin.context import preview_cue_pmf
from realtimecoin.state import peek_cue_label

from validation.pass_summary import pass_summary

try:                                    # pragma: no cover - exercised by merge
    from validation.mixture_utils import mixture_cdf
except ImportError:                     # pragma: no cover
    mixture_cdf = None                  # see _mixture_cdf below
try:                                    # pragma: no cover
    from validation.uniform_ks import uniform_ks
except ImportError:                     # pragma: no cover
    uniform_ks = None                   # see _uniform_ks below

__all__ = ["run", "THRESHOLDS"]

EPS = float(np.finfo(float).eps)

#: Gate rationale (verbatim from the ``.m``): ``feedback_ks`` and ``cue_ks``
#: (both 0.08) are tighter than the Kalman validators' single-stream feedback
#: gate (0.15) because this validator pools ``num_datasets * trials`` PIT values,
#: sharply reducing the sampling variance of the KS statistic.
#: ``state_rank_ks`` (0.15) is looser because the mixture-CDF rank uses the
#: FILTERED posterior, a noisier diagnostic than the predictive feedback PIT.
THRESHOLDS = {
    "feedback_ks": 0.08,
    "cue_ks": 0.08,
    "state_rank_ks": 0.15,
}

_DEFAULTS = {
    "num_datasets": 25,
    "trials": 80,
    "particles": 100,
    "make_plots": False,
}

# --- Data-generating process (transliterated from the .m) ------------------
_A = 0.9                    # retention of the latent AR(1) state
_D = 0.02                   # drift of the latent AR(1) state
_SIGMA_Q = 0.03             # process-noise standard deviation
_SIGMA_R = 0.05             # sensory-noise standard deviation
_CUE_PROB = np.array([[0.85, 0.15], [0.15, 0.85]])      # p(cue | context)
_TRANSITION = np.array([[0.96, 0.04], [0.04, 0.96]])    # p(context' | context)
#: Only entry 0 is ever used: it seeds the initial state. The state then evolves
#: as a single AR(1) chain regardless of context, exactly as in the ``.m``.
_CONTEXT_OFFSET = np.array([-0.25, 0.25])


def run(seed=1201, strict=False, **overrides):
    """Run the extended PIT calibration validator.

    Parameters
    ----------
    seed : int, optional
        Base seed for the synthetic data, the PIT randomisation and the model
        generators. Default 1201 (the MATLAB default).
    strict : bool, optional
        Raise instead of returning ``passed=False``. Default ``False``.
    **overrides
        ``num_datasets`` (25), ``trials`` (80), ``particles`` (100),
        ``make_plots`` (``False``, ignored).

    Returns
    -------
    dict
        ``feedback_p_values``, ``cue_p_values``, ``state_rank_values``,
        ``retention_rank_values``, ``drift_rank_values``, ``feedback_ks``,
        ``cue_ks``, ``state_rank_ks``, ``retention_rank_mean``,
        ``drift_rank_mean``, ``feedback_mean``, ``cue_mean``, ``thresholds``,
        ``checks``, ``passed`` and ``config``.

    Raises
    ------
    RuntimeError
        If ``strict`` and any gate fails.
    """
    cfg = _config(seed, strict, overrides)

    feedback_p = []
    cue_p = []
    state_rank = []
    retention_rank = []
    drift_rank = []

    data_rng = np.random.default_rng(cfg["seed"])
    for dataset in range(cfg["num_datasets"]):
        _run_one_dataset(
            cfg, dataset, data_rng,
            feedback_p, cue_p, state_rank, retention_rank, drift_rank,
        )

    results = {
        "feedback_p_values": np.asarray(feedback_p),
        "cue_p_values": np.asarray(cue_p),
        "state_rank_values": np.asarray(state_rank),
        "retention_rank_values": np.asarray(retention_rank),
        "drift_rank_values": np.asarray(drift_rank),
    }
    results["feedback_ks"] = _uniform_ks(results["feedback_p_values"])
    results["cue_ks"] = _uniform_ks(results["cue_p_values"])
    results["state_rank_ks"] = _uniform_ks(results["state_rank_values"])
    results["retention_rank_mean"] = _nanmean(results["retention_rank_values"])
    results["drift_rank_mean"] = _nanmean(results["drift_rank_values"])
    results["feedback_mean"] = _nanmean(results["feedback_p_values"])
    results["cue_mean"] = _nanmean(results["cue_p_values"])
    results["thresholds"] = dict(THRESHOLDS)

    checks = {
        "feedback_ks": results["feedback_ks"] < THRESHOLDS["feedback_ks"],
        "cue_ks": results["cue_ks"] < THRESHOLDS["cue_ks"],
        "state_rank_ks": results["state_rank_ks"] < THRESHOLDS["state_rank_ks"],
    }
    results["passed"], results["checks"] = pass_summary(checks)
    results["config"] = cfg

    print(
        "Extended PIT: feedback KS %.3f, cue KS %.3f, state-rank KS %.3f"
        % (results["feedback_ks"], results["cue_ks"], results["state_rank_ks"])
    )

    if cfg["strict"] and not results["passed"]:
        raise RuntimeError(
            "validate_p_values_extended:Failed: extended p-value validation "
            "failed (%s)." % _failed_names(results["checks"])
        )
    return results


def _run_one_dataset(
    cfg, dataset, data_rng,
    feedback_p, cue_p, state_rank, retention_rank, drift_rank,
):
    """Simulate one synthetic stream and append its per-trial PIT values.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    dataset : int
        0-based dataset index; part of the model's seed.
    data_rng : numpy.random.Generator
        Generator for the synthetic data and the PIT randomisation.
    feedback_p, cue_p, state_rank, retention_rank, drift_rank : list
        Accumulators, appended to in place (one entry per trial).

    Returns
    -------
    None
    """
    model = RealTimeCOIN(
        num_particles=cfg["particles"],
        max_contexts=4,
        prior_mean_retention=_A,
        prior_precision_retention=1e6,
        prior_mean_drift=_D,
        prior_precision_drift=1e6,
        sigma_process_noise=_SIGMA_Q,
        sigma_sensory_noise=_SIGMA_R,
        sigma_motor_noise=0.0,
        # dataset + 1, not dataset: numpy's SeedSequence IGNORES trailing zero
        # entropy words, so default_rng([seed, 0]) is byte-identical to
        # default_rng(seed) - which is data_rng. Without the offset, dataset 0's
        # particle noise would be drawn from the very stream that generated its
        # data.
        rng=np.random.default_rng([cfg["seed"], dataset + 1]),
    )

    context = 0
    state = float(_CONTEXT_OFFSET[context])

    for _trial in range(cfg["trials"]):
        context, cue, state, feedback = _synthetic_stream(
            data_rng, context, state
        )

        # Score BEFORE the model sees the trial: these are prequential
        # (one-step-ahead) predictive p-values.
        feedback_p.append(_predictive_feedback_cdf(model, feedback, cue))
        cue_p.append(_predictive_cue_p_value(model, cue, data_rng.random()))

        model.observe_q(cue)
        model.observe_y(feedback)

        raw = model.D
        state_rank.append(
            _mixture_cdf(
                state,
                raw.responsibilities,
                raw.state_filtered_mean,
                raw.state_filtered_var,
            )
        )

        diagnostics = model.diagnostics()
        w = np.asarray(diagnostics["responsibilities"], dtype=float)
        w = w / max(w.sum(), EPS)
        retention_rank.append(float(np.sum(w * (diagnostics["retention"] <= _A))))
        drift_rank.append(float(np.sum(w * (diagnostics["drift"] <= _D))))


def _synthetic_stream(data_rng, context, state):
    """Advance the 2-context HMM by one trial.

    Parameters
    ----------
    data_rng : numpy.random.Generator
        Generator for every synthetic draw.
    context : int
        Current 0-based latent context.
    state : float
        Current latent AR(1) state.

    Returns
    -------
    context : int
        Next latent context.
    cue : int
        Raw cue value (1 or 2, mirroring the ``.m``; the model's cue registry
        maps raw values to 0-based labels in order of first appearance).
    state : float
        Next latent state.
    feedback : float
        Noisy observation of the next state.

    Notes
    -----
    The latent state does NOT depend on the context - only the cue does. That is
    the ``.m``'s behaviour (``context_offset`` seeds ``s`` once, then ``s`` is
    driven by ``a * s + d`` alone), and it is preserved deliberately.
    """
    context = _sample_categorical(data_rng, _TRANSITION[context])
    cue = _sample_categorical(data_rng, _CUE_PROB[context]) + 1
    state = _A * state + _D + _SIGMA_Q * data_rng.standard_normal()
    feedback = state + _SIGMA_R * data_rng.standard_normal()
    return context, int(cue), float(state), float(feedback)


# --------------------------------------------------------------------------
# Predictive PITs (standing in for the unit-C2 class queries)
# --------------------------------------------------------------------------


def _predictive_feedback_cdf(model, y, cue):
    """Predictive CDF of the next feedback at ``y`` (feedback PIT).

    Stand-in for ``RealTimeCOIN.predictive_state_feedback_cdf``; see the module
    docstring for why it is computed here.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Model whose particle state is read (never mutated).
    y : float
        Observed feedback of the upcoming trial.
    cue : float or None
        RAW cue value of the upcoming trial.

    Returns
    -------
    float
        ``P(Y <= y)`` under the predictive Gaussian mixture, in [0, 1].
    """
    q_label = peek_cue_label(model, cue)
    weights, means, variances = _predictive_feedback_components(model, q_label)
    return _mixture_cdf(y, weights, means, variances)


def _predictive_feedback_components(model, q_label):
    """Per-particle, per-context predictive feedback mixture components.

    Transliteration of ``validation_predictive_feedback_moments.m`` up to (but
    not including) its final collapse to two moments: the same weights, means
    and variances are returned as arrays so the caller can evaluate the mixture
    CDF.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Model whose particle state is read.
    q_label : int or None
        0-based cue label of the upcoming trial, clamped to the last
        instantiated cue column; ``None`` marginalises the cue out.

    Returns
    -------
    weights : numpy.ndarray
        ``(P, C)`` context weights, each row summing to one.
    means : numpy.ndarray
        ``(P, C)`` predictive feedback means.
    variances : numpy.ndarray
        ``(P, C)`` predictive feedback variances.
    """
    d = model.D
    particles = np.arange(model.num_particles)

    # prior[p, :] = transition row out of particle p's currently sampled
    # context, renormalised.
    prior = _normalise_rows(d.local_transition_matrix[particles, d.context, :])
    if q_label is None:
        weights = prior
    else:
        q_col = min(int(q_label), d.local_cue_matrix.shape[2] - 1)
        weights = _normalise_rows(prior * d.local_cue_matrix[:, :, q_col])

    # One-step-ahead AR(1) prediction: s' = a s + d, V' = a^2 V + Q.
    state_mean = d.retention * d.state_filtered_mean + d.drift          # (P, C)
    state_var = (
        d.retention ** 2 * d.state_filtered_var + model.sigma_process_noise ** 2
    )                                                                   # (P, C)

    # The novel slot has no filtered state: use the stationary distribution of
    # its own sampled dynamics. Particles at the context cap have no novel slot.
    rows = np.nonzero(d.n_active < model.max_contexts)[0]
    if rows.size:
        cols = d.n_active[rows]
        a = d.retention[rows, cols]
        drift = d.drift[rows, cols]
        state_mean[rows, cols] = drift / np.fmax(1.0 - a, EPS)
        state_var[rows, cols] = model.sigma_process_noise ** 2 / np.fmax(
            1.0 - a ** 2, EPS
        )

    means = state_mean + d.bias                                         # (P, C)
    variances = (
        state_var + model.sigma_sensory_noise ** 2 + model.sigma_motor_noise ** 2
    )                                                                   # (P, C)
    return weights, means, variances


def _predictive_cue_p_value(model, cue, u):
    """Randomised predictive p-value for a cue label (cue PIT).

    Transliteration of ``predictive_cue_p_value.m``::

        p = F(q-) + u * f(q)

    where ``f`` is the predictive cue pmf and ``F(q-)`` the cumulative mass of
    the labels below ``q``. The ``u * f(q)`` term spreads the atom into a
    continuous ``[F(q-), F(q)]``, so ``p`` is uniform under a correct model.

    Parameters
    ----------
    model : realtimecoin.RealTimeCOIN
        Model whose particle state is read; the cue registry is NOT mutated.
    cue : float or None
        Raw cue value; ``None`` yields ``nan`` (the p-value is undefined).
    u : float
        Uniform variate in [0, 1]. Passed in explicitly (never drawn here) so
        the whole validator stays reproducible from one seed.

    Returns
    -------
    float
        The p-value in [0, 1], or ``nan``.
    """
    q_label = peek_cue_label(model, cue)
    if q_label is None:
        return float("nan")
    pmf, labels = preview_cue_pmf(model)
    if q_label >= pmf.size:
        # A never-seen label sits beyond the pmf and carries zero mass.
        pmf = np.pad(pmf, (0, q_label + 1 - pmf.size))
        labels = np.arange(pmf.size)
    f = float(pmf[q_label])
    f_minus = float(pmf[labels < q_label].sum())
    return float(min(max(f_minus + u * f, 0.0), 1.0))


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _normalise_rows(x):
    """Row-normalise a non-negative matrix, falling back to uniform.

    The transposed counterpart of ``normalize_columns`` inside
    ``validation_predictive_feedback_moments.m`` (MATLAB works on a
    contexts-by-particles matrix; this package is particles-leading).

    Parameters
    ----------
    x : numpy.ndarray
        ``(P, C)`` weights; non-finite and negative entries are treated as zero.

    Returns
    -------
    numpy.ndarray
        ``(P, C)`` rows summing to one; an all-zero row becomes uniform.
    """
    out = np.array(x, dtype=float)
    out[~np.isfinite(out) | (out < 0)] = 0.0
    total = out.sum(axis=1)
    degenerate = total <= 0
    if np.any(degenerate):
        out[degenerate, :] = 1.0 / out.shape[1]
        total[degenerate] = 1.0
    return out / total[:, None]


def _mixture_cdf(x, weights, means, variances):
    """CDF of a Gaussian mixture at ``x``.

    Delegates to ``validation.mixture_utils.mixture_cdf`` (unit C3) when that
    module is importable, and otherwise evaluates the same expression locally so
    this validator runs standalone before C3 merges. The local branch is a
    transliteration of ``validation_mixture_cdf.m``.

    Parameters
    ----------
    x : float
        Query point.
    weights : array_like
        Mixture weights (any shape; normalised by their total).
    means, variances : array_like
        Component moments, same shape as ``weights``.

    Returns
    -------
    float
        The mixture CDF in [0, 1], or ``nan`` when the weights carry no mass.
    """
    if mixture_cdf is not None:
        # Copy the weights before handing them over: callers pass model-owned
        # arrays (D.responsibilities), and a literal transliteration of
        # validation_mixture_cdf.m's `w(~isfinite(w) | w < 0) = 0` built on
        # np.asarray would zero entries of the live particle state in place.
        return float(
            mixture_cdf(x, np.array(weights, dtype=float), means, variances)
        )

    from realtimecoin.statics import normal_cdf

    w = np.array(weights, dtype=float)
    w[~np.isfinite(w) | (w < 0)] = 0.0
    total = w.sum()
    if total <= 0:
        return float("nan")
    w = w / total
    p = float(np.sum(w * normal_cdf(x, means, np.fmax(variances, 0.0))))
    return float(min(max(p, 0.0), 1.0))


def _uniform_ks(p):
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0, 1).

    Delegates to ``validation.uniform_ks.uniform_ks`` (unit C3) when importable;
    the local fallback transliterates ``validation_uniform_ks.m`` so this
    validator runs standalone before C3 merges.

    Parameters
    ----------
    p : array_like
        PIT values; non-finite entries are dropped and the rest clipped to
        [0, 1].

    Returns
    -------
    float
        The largest vertical gap between the empirical CDF and ``F(u) = u``, or
        ``nan`` for an empty sample.
    """
    if uniform_ks is not None:
        return float(uniform_ks(p))

    values = np.asarray(p, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = np.sort(np.clip(values, 0.0, 1.0))
    n = values.size
    if n == 0:
        return float("nan")
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - values)),
                     np.max(np.abs(lower - values))))


def _sample_categorical(rng, p):
    """Draw one 0-based index from unnormalised probabilities.

    Inverse-CDF draw, mirroring ``validation_sample_categorical.m``. Kept local
    rather than imported from unit C3's ``validation/sample_categorical.py``
    because that module's generator-passing signature is not pinned; the
    algorithm is four lines and identical either way.

    Parameters
    ----------
    rng : numpy.random.Generator
        Generator supplying the single uniform draw.
    p : array_like
        Non-negative weights; non-finite/negative entries count as zero, and an
        all-zero vector becomes uniform.

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
    # side="left" reproduces MATLAB's find(rand <= cumsum(p), 1); the clamp
    # guards the corner where rounding leaves cumsum(p) fractionally below u.
    index = int(np.searchsorted(np.cumsum(weights), rng.random(), side="left"))
    return min(index, weights.size - 1)


def _nanmean(values):
    """Mean ignoring ``nan`` (MATLAB's ``mean(..., 'omitnan')``).

    ``nan`` ONLY - infinities propagate, as they do in MATLAB. The empty and
    all-``nan`` cases are short-circuited so numpy never warns where MATLAB
    would not.

    Parameters
    ----------
    values : array_like
        Sample.

    Returns
    -------
    float
        The mean of the non-``nan`` entries, or ``nan`` when there are none.
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

    The Python stand-in for MATLAB's ``inputParser``.

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
        The resolved configuration, including ``seed`` and ``strict``.

    Raises
    ------
    TypeError
        If an override key is not recognised.
    """
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise TypeError(
            "validate_p_values_extended.run() got unexpected keyword "
            "argument(s): %s." % ", ".join(sorted(unknown))
        )
    cfg = dict(_DEFAULTS)
    cfg.update(overrides)
    cfg["num_datasets"] = int(cfg["num_datasets"])
    cfg["trials"] = int(cfg["trials"])
    cfg["particles"] = int(cfg["particles"])
    cfg["make_plots"] = bool(cfg["make_plots"])
    cfg["seed"] = int(seed)
    cfg["strict"] = bool(strict)
    return cfg


if __name__ == "__main__":       # pragma: no cover
    # Invoke as ``python -m validation.validate_p_values_extended`` from
    # ``python/``: this is a namespace package, so running the file by path
    # would not put the package root on sys.path.
    run()

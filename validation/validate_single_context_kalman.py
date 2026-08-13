"""Independent Kalman reference validation of the scalar model.

Translated from ``validation/validate_single_context_kalman.m``.

In the one-context, scalar, linear-Gaussian case the COIN state model reduces to
a Kalman filter::

    s_t = a s_{t-1} + d + eps_t,    eps_t ~ N(0, sigma_Q^2)
    y_t = s_t + eta_t,              eta_t ~ N(0, sigma_R^2)

If the retention and drift priors are made very precise (precision 1e12) and
``max_contexts == 1``, RealTimeCOIN should be doing nothing but the Kalman
predict-update recursion. This validator is therefore an EXTERNAL mathematical
reference, not a comparison against the original offline COIN.

Deviation from the MATLAB original: the two implementations swap roles
---------------------------------------------------------------------
MATLAB deliberately reads the two quantities from two DIFFERENT
implementations, so the validator cross-checks production code against
validation code:

* predictive moments  <- ``validation_predictive_feedback_moments.m`` (validation)
* PIT                 <- ``coin.predictive_state_feedback_cdf``        (production)

This port keeps that two-implementation design but SWAPS which side supplies
which quantity:

* predictive moments  <- ``model.predictive_feedback_moments`` (production)
* PIT                 <- :func:`validation.feedback_moments.predictive_feedback_mixture`
  plus :func:`validation.mixture_utils.mixture_cdf`           (validation)

The swap is a deliberate INDEPENDENT MIRROR, not a workaround for a missing
query: ``RealTimeCOIN.predictive_state_feedback_cdf`` exists and is exercised by
``validate_p_values`` / ``validate_p_values_extended``, so gating on it here
would re-test it rather than add coverage. In this orientation the RMSE gate
measures PRODUCTION moments against the analytic Kalman filter - a slightly
stronger claim than MATLAB's - while the mirror supplies the PIT.

Both implementations are therefore still exercised, and the reported
``moment_cross_check_max_abs_diff`` makes their agreement explicit: it is the
largest per-trial discrepancy between the production moments and the validation
mirror's, and should sit at round-off. Swapping back to
``model.predictive_state_feedback_cdf(y, 1)`` would reproduce MATLAB's exact
assignment and turn the mirror into a redundant third check.

Run as a script for a MATLAB-style console summary::

    python -m validation.validate_single_context_kalman --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from realtimecoin import RealTimeCOIN
from realtimecoin.statics import normal_cdf

from .feedback_moments import predictive_feedback_mixture
from .kalman_reference import kalman_reference_step
from .mixture_utils import mixture_cdf, mixture_moments
from .pass_summary import pass_summary
from .uniform_ks import uniform_ks

__all__ = ["run", "THRESHOLDS", "DEFAULT_TRIALS", "DEFAULT_PARTICLES"]

EPS = float(np.finfo(float).eps)

#: Trial/particle counts of ``run_validation.m``'s compact profile
#: (``kalman_trials`` / ``kalman_particles``), so the metrics reported here are
#: directly comparable with the MATLAB suite's.
DEFAULT_TRIALS = 80
DEFAULT_PARTICLES = 180

#: Gates transliterated verbatim from the MATLAB validator. ``mean_rmse`` and
#: ``variance_relative_error`` bound agreement with the analytic Kalman moments;
#: ``feedback_ks`` is a SINGLE-STREAM PIT gate and is deliberately looser than
#: the pooled ``p_values_extended`` gate (0.08), which averages many datasets and
#: so estimates the KS statistic far more tightly.
THRESHOLDS = {
    "mean_rmse": 0.05,
    "variance_relative_error": 0.35,
    "feedback_ks": 0.15,
}

# Ground-truth dynamics, identical to the MATLAB script.
_A = 0.82
_D = 0.035
_SIGMA_Q = 0.025
_SIGMA_R = 0.05


def run(seed=0, strict=False, **overrides):
    """Run the scalar single-context Kalman validation.

    Parameters
    ----------
    seed : int, optional
        Seed for the whole validator. Two independent streams are spawned from
        it with :class:`numpy.random.SeedSequence`: one drives the synthetic
        data, the other the model's particle filter. (MATLAB seeds one global
        stream that both share; the split is required here because the Python
        model owns its generator.)
    strict : bool, optional
        When True, raise :class:`AssertionError` if any gate fails - MATLAB's
        ``Strict`` flag. When False (the default) the metrics are returned with
        ``passed=False`` instead.
    **overrides
        ``trials`` (default 80) and ``particles`` (default 180).

    Returns
    -------
    dict
        ``predictive_mean_rmse``, ``predictive_variance_relative_error`` (also
        aliased as ``variance_relative_error``), ``feedback_ks``,
        ``analytic_feedback_ks``, ``moment_cross_check_max_abs_diff``,
        the per-trial traces
        (``realtime_predictive_mean``, ``kalman_predictive_mean``,
        ``realtime_predictive_variance``, ``kalman_predictive_variance``,
        ``feedback_p_values``, ``feedback``), ``thresholds``, ``checks``,
        ``passed``, ``errored`` and ``config``.

    Raises
    ------
    TypeError
        On an unrecognised override.
    AssertionError
        If ``strict`` and any gate fails.
    """
    trials = int(overrides.pop("trials", DEFAULT_TRIALS))
    particles = int(overrides.pop("particles", DEFAULT_PARTICLES))
    if overrides:
        raise TypeError(
            "validate_single_context_kalman.run got unexpected keyword "
            "argument(s): %s." % ", ".join(sorted(overrides))
        )

    data_seed, model_seed = np.random.SeedSequence(seed).spawn(2)
    data_rng = np.random.default_rng(data_seed)

    model = RealTimeCOIN(
        num_particles=particles,
        max_contexts=1,
        prior_mean_retention=_A,
        prior_precision_retention=1e12,
        prior_mean_drift=_D,
        prior_precision_drift=1e12,
        sigma_process_noise=_SIGMA_Q,
        sigma_sensory_noise=_SIGMA_R,
        sigma_motor_noise=0.0,
        rng=np.random.default_rng(model_seed),
    )

    # Kalman filter and generating process both start at the stationary
    # distribution of s_t = a s + d + N(0, sigma_Q^2).
    m = _D / (1.0 - _A)
    p_cov = _SIGMA_Q ** 2 / (1.0 - _A ** 2)
    s = m

    kalman_mean = np.zeros(trials)
    kalman_var = np.zeros(trials)
    rt_mean = np.zeros(trials)
    rt_var = np.zeros(trials)
    rt_pit = np.zeros(trials)
    analytic_pit = np.zeros(trials)
    y_trace = np.zeros(trials)
    cross_check = np.zeros(trials)

    for t in range(trials):
        s = _A * s + _D + _SIGMA_Q * data_rng.standard_normal()
        y = s + _SIGMA_R * data_rng.standard_normal()
        y_trace[t] = y

        pred_mean, y_var, m, p_cov = kalman_reference_step(
            m, p_cov, _A, _D, _SIGMA_Q ** 2, _SIGMA_R ** 2, y
        )
        kalman_mean[t] = pred_mean
        kalman_var[t] = y_var

        # Cue label 0 is MATLAB's cue column 1 (the only instantiated column).
        # Moments from PRODUCTION code, PIT from the independent mirror; see
        # the module docstring for why the two roles are swapped vs MATLAB.
        rt_mean[t], rt_var[t] = model.predictive_feedback_moments(0)
        weights, means, variances = predictive_feedback_mixture(model, 0)
        rt_pit[t] = mixture_cdf(y, weights, means, variances)
        analytic_pit[t] = float(normal_cdf(y, pred_mean, y_var))

        mirror_mean, mirror_var = mixture_moments(weights, means, variances)
        cross_check[t] = max(
            abs(mirror_mean - rt_mean[t]), abs(mirror_var - rt_var[t])
        )

        model.observe_q(1)
        model.observe_y(y)

    mean_rmse = float(np.sqrt(np.mean((rt_mean - kalman_mean) ** 2)))
    var_rel_error = float(
        np.median(np.abs(rt_var - kalman_var) / np.fmax(kalman_var, EPS))
    )
    feedback_ks = uniform_ks(rt_pit)
    analytic_ks = uniform_ks(analytic_pit)

    passed, checks = pass_summary(
        {
            "mean_rmse": mean_rmse < THRESHOLDS["mean_rmse"],
            "variance_relative_error": (
                var_rel_error < THRESHOLDS["variance_relative_error"]
            ),
            "feedback_ks": feedback_ks < THRESHOLDS["feedback_ks"],
        }
    )

    results = {
        "predictive_mean_rmse": mean_rmse,
        "predictive_variance_relative_error": var_rel_error,
        # Alias under the bare gate name, so callers can key on either.
        "variance_relative_error": var_rel_error,
        "feedback_ks": feedback_ks,
        "analytic_feedback_ks": analytic_ks,
        # Largest per-trial gap between the production moments and the
        # validation mirror's. Reported, not gated: MATLAB has no counterpart,
        # and a gate on it would be inventing a threshold.
        "moment_cross_check_max_abs_diff": float(np.max(cross_check)),
        "realtime_predictive_mean": rt_mean,
        "kalman_predictive_mean": kalman_mean,
        "realtime_predictive_variance": rt_var,
        "kalman_predictive_variance": kalman_var,
        "feedback_p_values": rt_pit,
        "feedback": y_trace,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": passed,
        "errored": False,
        "config": {"seed": seed, "trials": trials, "particles": particles},
    }

    print(
        "Single-context Kalman: mean RMSE %.4f, variance rel. error %.3f, "
        "PIT KS %.3f" % (mean_rmse, var_rel_error, feedback_ks)
    )

    if strict and not passed:
        raise AssertionError(
            "Single-context Kalman validation failed: %r" % (checks,)
        )
    return results


def _main(argv=None):
    """Command-line entry point printing the MATLAB-style summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    results = run(
        seed=args.seed,
        strict=args.strict,
        trials=args.trials,
        particles=args.particles,
    )
    print("  analytic PIT KS   %.3f" % results["analytic_feedback_ks"])
    print(
        "  moment cross-check (production vs mirror) max |diff| %.3e"
        % results["moment_cross_check_max_abs_diff"]
    )
    for name, ok in results["checks"].items():
        print("  %-26s %s" % (name, "PASS" if ok else "FAIL"))
    print("passed: %s" % results["passed"])
    return 0 if results["passed"] else 1


if __name__ == "__main__":   # pragma: no cover - script entry point
    raise SystemExit(_main())

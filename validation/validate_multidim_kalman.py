"""Independent multivariate Kalman reference validation of the MD model.

Translated from ``validation/validate_multidim_kalman.m``.

In the one-context, N-dimensional, linear-Gaussian case the COIN state model
reduces to a multivariate Kalman filter::

    s_t = A s_{t-1} + d + w_t,   w_t ~ N(0, Q)
    y_t = s_t + v_t,             v_t ~ N(0, R)

With very precise dynamics priors and ``max_contexts == 1``, RealTimeCOIN should
be doing only the multivariate predict-update recursion. CORRELATED (full) ``Q``
and ``R`` are used on purpose, so the matrix Kalman gain and the Cholesky
likelihood are genuinely exercised rather than collapsing to N independent
scalar filters.

Calibration is assessed with a multivariate probability-integral transform: the
Mahalanobis distance of each innovation under the predictive feedback covariance
is chi-square_N distributed when the model is calibrated, so
``chi2.cdf(mahalanobis, N)`` should be Uniform(0, 1). MATLAB spells that CDF
``gammainc(mahalanobis / 2, N / 2)`` (the regularised lower incomplete gamma);
:func:`scipy.stats.chi2.cdf` is the same function.

Run as a script for a MATLAB-style console summary::

    python -m validation.validate_multidim_kalman --seed 0 --dim 2
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import chi2

from realtimecoin import RealTimeCOIN

from .kalman_reference import kalman_reference_step
from .pass_summary import pass_summary
from .uniform_ks import uniform_ks

__all__ = [
    "run",
    "THRESHOLDS",
    "DEFAULT_TRIALS",
    "DEFAULT_PARTICLES",
    "DEFAULT_DIM",
]

EPS = float(np.finfo(float).eps)

#: Compact-profile counts from ``run_validation.m`` (``kalman_trials`` /
#: ``kalman_particles``), and ``Dim`` 2 as the suite passes it.
DEFAULT_TRIALS = 80
DEFAULT_PARTICLES = 180
DEFAULT_DIM = 2

#: Gates match :mod:`validation.validate_single_context_kalman` exactly - the
#: scalar and multivariate Kalman references are deliberately held to one
#: standard, so a regression in either shows up against the same numbers.
THRESHOLDS = {
    "mean_rmse": 0.05,
    "variance_relative_error": 0.35,
    "feedback_ks": 0.15,
}

_A = 0.82
_DRIFT = 0.035


def _noise_covariances(n):
    """Build the correlated process and observation covariances.

    ``Q = 0.025^2 (I + 0.3 (1 - I))`` and ``R = 0.05^2 (I - 0.2 (1 - I))``:
    symmetric, positive definite, and non-diagonal so the filter cannot pass by
    treating the dimensions independently.

    Parameters
    ----------
    n : int
        State dimension.

    Returns
    -------
    q : numpy.ndarray
        ``(N, N)`` process-noise covariance.
    r : numpy.ndarray
        ``(N, N)`` observation-noise covariance.
    """
    eye = np.eye(n)
    off = np.ones((n, n)) - eye
    q = 0.025 ** 2 * (eye + 0.3 * off)
    r = 0.05 ** 2 * (eye - 0.2 * off)
    return (q + q.T) / 2.0, (r + r.T) / 2.0


def run(seed=0, strict=False, **overrides):
    """Run the multi-dimensional single-context Kalman validation.

    Parameters
    ----------
    seed : int, optional
        Seed for the whole validator; independent data and model streams are
        spawned from it with :class:`numpy.random.SeedSequence`.
    strict : bool, optional
        When True, raise :class:`AssertionError` if any gate fails (MATLAB's
        ``Strict`` flag).
    **overrides
        ``trials`` (default 80), ``particles`` (default 180) and ``dim``
        (default 2).

    Returns
    -------
    dict
        ``dim``, ``predictive_mean_rmse``,
        ``predictive_variance_relative_error`` (also aliased as
        ``variance_relative_error``), ``feedback_ks``, the per-trial traces
        (``realtime_predictive_mean``, ``kalman_predictive_mean``,
        ``variance_relative_error_trace``, ``feedback_p_values``),
        ``thresholds``, ``checks``, ``passed``, ``errored`` and ``config``.

    Raises
    ------
    TypeError
        On an unrecognised override.
    AssertionError
        If ``strict`` and any gate fails.
    """
    trials = int(overrides.pop("trials", DEFAULT_TRIALS))
    particles = int(overrides.pop("particles", DEFAULT_PARTICLES))
    n = int(overrides.pop("dim", DEFAULT_DIM))
    if overrides:
        raise TypeError(
            "validate_multidim_kalman.run got unexpected keyword argument(s): "
            "%s." % ", ".join(sorted(overrides))
        )

    data_seed, model_seed = np.random.SeedSequence(seed).spawn(2)
    data_rng = np.random.default_rng(data_seed)

    a_mat = _A * np.eye(n)
    d_vec = _DRIFT * np.ones(n)
    q_cov, r_cov = _noise_covariances(n)

    # The model IGNORES the covariance overrides at state_dim == 1 (it runs the
    # scalar pipeline off the sigma_* properties and warns if they are passed),
    # so at N == 1 the same Q / R are supplied as scalar standard deviations
    # instead. Q and R are diagonal there, so the two forms are identical.
    if n == 1:
        noise_kwargs = {
            "sigma_process_noise": float(np.sqrt(q_cov[0, 0])),
            "sigma_sensory_noise": float(np.sqrt(r_cov[0, 0])),
            "sigma_motor_noise": 0.0,
        }
    else:
        noise_kwargs = {
            "process_noise_covariance": q_cov,
            "observation_noise_covariance": r_cov,
        }

    model = RealTimeCOIN(
        num_particles=particles,
        max_contexts=1,
        state_dim=n,
        prior_mean_retention=_A,
        prior_precision_retention=1e12,
        prior_mean_drift=_DRIFT,
        prior_precision_drift=1e12,
        rng=np.random.default_rng(model_seed),
        **noise_kwargs,
    )

    l_q = np.linalg.cholesky(q_cov)
    l_r = np.linalg.cholesky(r_cov)

    # Filter and generating process both start at the shared stationary
    # distribution (P0 = Q / (1 - a^2), valid because A = a I).
    m = d_vec / (1.0 - _A)
    p_cov = q_cov / (1.0 - _A ** 2)
    s = m.copy()

    kalman_mean = np.zeros((n, trials))
    rt_mean = np.zeros((n, trials))
    var_rel = np.zeros(trials)
    rt_pit = np.zeros(trials)

    for t in range(trials):
        s = a_mat @ s + d_vec + l_q @ data_rng.standard_normal(n)
        y = s + l_r @ data_rng.standard_normal(n)

        pred_mean, s_cov, m, p_cov = kalman_reference_step(
            m, p_cov, a_mat, d_vec, q_cov, r_cov, y
        )
        kalman_mean[:, t] = pred_mean

        model.observe_q(1)
        # Cue LABEL 0 here is MATLAB's cue label 1 (labels are 0-based in the
        # Python translation); only one cue column exists, so it is clamped
        # to that column either way.
        mu, sigma = model.predictive_feedback_moments(0)
        # At N == 1 the model runs the SCALAR pipeline and returns plain
        # floats, so promote before any linear algebra.
        mu = np.atleast_1d(np.asarray(mu, dtype=float)).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(n, n)
        rt_mean[:, t] = mu
        # fmax, not the builtin max: MATLAB's max(x, eps) drops a NaN in favour
        # of eps, whereas Python's max propagates it.
        var_rel[t] = np.linalg.norm(sigma - s_cov, "fro") / np.fmax(
            np.linalg.norm(s_cov, "fro"), EPS
        )

        # Mahalanobis PIT: (y - mu)' Sigma^{-1} (y - mu) ~ chi-square_N.
        innovation = y - mu
        mahalanobis = float(innovation @ np.linalg.solve(sigma, innovation))
        rt_pit[t] = float(chi2.cdf(mahalanobis, n))

        model.observe_y(y)

    mean_rmse = float(
        np.sqrt(np.mean((rt_mean.reshape(-1) - kalman_mean.reshape(-1)) ** 2))
    )
    var_rel_error = float(np.median(var_rel))
    feedback_ks = uniform_ks(rt_pit)

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
        "dim": n,
        "predictive_mean_rmse": mean_rmse,
        "predictive_variance_relative_error": var_rel_error,
        "variance_relative_error": var_rel_error,
        "feedback_ks": feedback_ks,
        "realtime_predictive_mean": rt_mean,
        "kalman_predictive_mean": kalman_mean,
        "variance_relative_error_trace": var_rel,
        "feedback_p_values": rt_pit,
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "passed": passed,
        "errored": False,
        "config": {
            "seed": seed,
            "trials": trials,
            "particles": particles,
            "dim": n,
        },
    }

    print(
        "Multi-dim Kalman (N=%d): mean RMSE %.4f, variance rel. error %.3f, "
        "PIT KS %.3f" % (n, mean_rmse, var_rel_error, feedback_ks)
    )

    if strict and not passed:
        raise AssertionError(
            "Multi-dimensional Kalman validation failed: %r" % (checks,)
        )
    return results


def _main(argv=None):
    """Command-line entry point printing the MATLAB-style summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    results = run(
        seed=args.seed,
        strict=args.strict,
        trials=args.trials,
        particles=args.particles,
        dim=args.dim,
    )
    for name, ok in results["checks"].items():
        print("  %-26s %s" % (name, "PASS" if ok else "FAIL"))
    print("passed: %s" % results["passed"])
    return 0 if results["passed"] else 1


if __name__ == "__main__":   # pragma: no cover - script entry point
    raise SystemExit(_main())

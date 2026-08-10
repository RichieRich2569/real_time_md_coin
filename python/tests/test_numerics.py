"""Unit tests for the deterministic numerical helpers.

These are all exact / analytic checks: every helper here is deterministic, so
the assertions pin the documented thresholds and fallback branches rather than
sampling statistics.
"""

import numpy as np
import pytest

from realtimecoin.numerics import (
    EPS,
    REALMAX,
    REALMIN,
    categorical_jeffreys,
    choljitter,
    ensure_pd,
    gaussian_jeffreys,
    gaussian_jeffreys_multi,
    gaussian_log_lik_chol,
    gaussian_pdf_columns_md,
    jeffreys_finite_clip,
    mixture_density_on_grid,
    normalize_columns,
    normalize_probability,
    precision_to_variance,
    regularize_covariance,
    renormalize_global_weights,
    safe_divide,
    safe_inverse,
    safe_log,
    stationary_state_cov_md,
    stationary_state_mean,
    stationary_state_mean_md,
    stationary_state_var,
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _is_positive_definite(a):
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


# --------------------------------------------------------------------------
# ensure_pd / choljitter / regularize_covariance
# --------------------------------------------------------------------------


def test_choljitter_reconstructs_a_positive_definite_matrix():
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    factor, ok = choljitter(a)
    assert ok
    assert np.allclose(np.triu(factor, 1), 0.0)   # lower triangular
    assert np.allclose(factor @ factor.T, a)


def test_choljitter_handles_a_singular_matrix_with_jitter():
    # Rank-1 (PSD but singular): the escalating jitter must rescue it.
    a = np.outer([1.0, 2.0], [1.0, 2.0])
    factor, ok = choljitter(a)
    assert ok
    assert np.allclose(factor @ factor.T, a, atol=1e-8)


def test_choljitter_symmetrises_its_input():
    a = np.array([[4.0, 2.0], [0.0, 3.0]])
    factor, ok = choljitter(a)
    assert ok
    assert np.allclose(factor @ factor.T, (a + a.T) / 2.0)


def test_choljitter_diagonal_fallback_for_an_indefinite_matrix():
    # Strongly indefinite: eight jitter escalations reach only ~1e-4 * scale,
    # which cannot repair a -5 eigenvalue, so the diagonal fallback fires.
    a = np.array([[1.0, 5.0], [5.0, 1.0]])
    factor, ok = choljitter(a)
    assert not ok
    assert np.allclose(factor, np.diag([1.0, 1.0]))


def test_choljitter_diagonal_fallback_stays_finite_for_nan_input():
    # MATLAB's max(NaN, eps) is eps, so the last-resort diagonal factor is
    # finite even when the input is poisoned; the whole point of the branch is
    # to degrade gracefully rather than propagate the nan.
    a = np.array([[np.nan, 0.0], [0.0, 4.0]])
    factor, ok = choljitter(a)
    assert not ok
    assert np.all(np.isfinite(factor))
    assert factor[1, 1] == pytest.approx(2.0)


def test_gaussian_log_lik_chol_stays_finite_for_a_poisoned_covariance():
    value = gaussian_log_lik_chol(np.array([0.1, 0.2]), np.array([[np.nan, 0.0], [0.0, 1.0]]))
    assert np.isfinite(value)


def test_ensure_pd_load_of_an_empty_matrix_is_a_usable_1x1():
    out, _ = ensure_pd("load", np.zeros((0, 0)))
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(EPS)
    assert np.isfinite(safe_inverse(out)[0, 0])


def test_safe_inverse_of_a_non_finite_matrix_propagates_rather_than_raising():
    # MATLAB rcond(nan matrix) is nan and nan < 1e-12 is false, so the direct
    # inverse runs and the nan propagates. It must not raise.
    out = safe_inverse(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    assert np.all(np.isnan(out) | np.isfinite(out))


def test_safe_log_of_nan_is_finite():
    assert np.isfinite(safe_log(np.array([np.nan]))[0])


def test_ensure_pd_load_is_positive_definite():
    covar = np.array([[0.0, 0.0], [0.0, 0.0]])
    out, ok = ensure_pd("load", covar)
    assert ok
    assert _is_positive_definite(out)
    assert np.allclose(out, out.T)


def test_ensure_pd_load_zeroes_non_finite_entries():
    covar = np.array([[1.0, np.nan], [np.inf, 2.0]])
    out = regularize_covariance(covar)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T)
    assert _is_positive_definite(out)


def test_ensure_pd_eigclip_projects_onto_the_psd_cone():
    a = np.array([[1.0, 5.0], [5.0, 1.0]])   # eigenvalues 6 and -4
    out, _ = ensure_pd("eigclip", a)
    eigenvalues = np.linalg.eigvalsh(out)
    assert np.all(eigenvalues >= -1e-12)
    assert np.allclose(out, out.T)
    assert np.max(eigenvalues) == pytest.approx(6.0)


def test_ensure_pd_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="UnknownMode"):
        ensure_pd("nope", np.eye(2))


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 3)),                            # zero
        np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),  # rank 1
        np.diag([1e-14, 1.0, 1e3]),                  # badly scaled
    ],
)
def test_regularize_covariance_repairs_semi_definite_input(matrix):
    # "load" only diagonally loads a PSD-but-singular covariance; it makes no
    # claim about genuinely indefinite input (that is what "eigclip" is for).
    out = regularize_covariance(matrix)
    assert _is_positive_definite(out)


# --------------------------------------------------------------------------
# safe_* helpers
# --------------------------------------------------------------------------


def test_safe_divide_guards_small_divisors():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 0.0, EPS / 2])
    z = safe_divide(a, b)
    assert np.array_equal(z, np.array([0.5, 0.0, 0.0]))
    assert np.all(np.isfinite(z))


def test_safe_log_floors_at_realmin():
    y = safe_log(np.array([0.0, -1.0, 1.0, np.e]))
    assert y[0] == pytest.approx(np.log(REALMIN))
    assert y[1] == pytest.approx(np.log(REALMIN))
    assert y[2] == pytest.approx(0.0)
    assert y[3] == pytest.approx(1.0)
    assert np.all(np.isfinite(y))


def test_safe_inverse_uses_the_true_inverse_when_well_conditioned():
    a = np.array([[2.0, 0.0], [0.0, 4.0]])
    assert np.allclose(safe_inverse(a), np.diag([0.5, 0.25]))


def test_safe_inverse_falls_back_to_the_pseudo_inverse():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])   # singular
    out = safe_inverse(a)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, np.linalg.pinv(a))


def test_precision_to_variance_maps_zero_to_infinity():
    v = precision_to_variance(np.array([0.0, 4.0, 0.5]))
    assert np.isinf(v[0])
    assert v[1] == pytest.approx(0.25)
    assert v[2] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Gaussian densities
# --------------------------------------------------------------------------


def test_gaussian_log_lik_chol_matches_the_scalar_normal():
    from realtimecoin.statics import normal_pdf

    y = np.array([0.4])
    s = np.array([[2.0]])
    assert gaussian_log_lik_chol(y, s) == pytest.approx(
        float(np.log(normal_pdf(0.4, 0.0, 2.0)))
    )


def test_gaussian_log_lik_chol_matches_the_closed_form():
    y = np.array([0.5, -1.0])
    s = np.array([[2.0, 0.4], [0.4, 1.0]])
    expected = -0.5 * (
        2 * np.log(2 * np.pi)
        + np.log(np.linalg.det(s))
        + y @ np.linalg.solve(s, y)
    )
    assert gaussian_log_lik_chol(y, s) == pytest.approx(expected)


def test_gaussian_pdf_columns_md_integrates_to_one():
    grid = np.linspace(-8, 8, 801)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    cov = np.array([[1.5, 0.3], [0.3, 0.8]])
    d = gaussian_pdf_columns_md(points, np.array([0.2, -0.4]), cov).reshape(xx.shape)
    mass = _trapezoid(_trapezoid(d, grid, axis=1), grid)
    assert mass == pytest.approx(1.0, abs=1e-4)


def test_gaussian_pdf_columns_md_matches_the_scalar_normal():
    from realtimecoin.statics import normal_pdf

    points = np.array([[-1.0], [0.0], [2.0]])
    d = gaussian_pdf_columns_md(points, np.array([0.5]), np.array([[2.0]]))
    assert np.allclose(d, normal_pdf(points.reshape(-1), 0.5, 2.0))


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------


def test_normalize_columns_normalises_the_context_axis():
    x = np.array([[1.0, 3.0], [2.0, 2.0]])   # (particles, contexts)
    out = normalize_columns(x)
    assert np.allclose(out.sum(axis=-1), 1.0)
    assert np.allclose(out[0], np.array([0.25, 0.75]))


def test_normalize_columns_zero_slice_collapses_to_e1():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 1.0]])
    out = normalize_columns(x)
    assert np.allclose(out[0], np.array([1.0, 0.0]))
    assert np.allclose(out[1], np.array([0.5, 0.5]))
    assert np.allclose(out[2], np.array([1.0, 0.0]))


def test_normalize_columns_can_target_another_axis():
    x = np.array([[1.0, 3.0], [3.0, 1.0]])
    out = normalize_columns(x, axis=0)
    assert np.allclose(out.sum(axis=0), 1.0)


def test_normalize_probability_is_strictly_positive():
    p = normalize_probability(np.array([1.0, 0.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)
    assert p[1] == pytest.approx(REALMIN, rel=1e-6)


def test_normalize_probability_uniform_fallback():
    assert np.allclose(normalize_probability(np.zeros(4)), np.ones(4) / 4)
    assert np.allclose(
        normalize_probability(np.array([np.nan, -1.0])), np.ones(2) / 2
    )


def test_renormalize_global_weights_keeps_an_empty_franchise_empty():
    assert np.allclose(renormalize_global_weights(np.zeros(3)), np.zeros(3))
    w = renormalize_global_weights(np.array([1.0, np.nan, 3.0]))
    assert np.allclose(w, np.array([0.25, 0.0, 0.75]))


# --------------------------------------------------------------------------
# Mixture density
# --------------------------------------------------------------------------


def test_mixture_density_on_grid_scalar_integrates_to_one():
    grid = np.linspace(-15, 15, 6001)
    weights = np.array([[0.5, 0.5], [1.0, 0.0]])       # (P, C)
    means = np.array([[-2.0, 3.0], [0.0, 0.0]])
    variances = np.array([[1.0, 2.0], [0.5, 1.0]])
    d = mixture_density_on_grid(grid, weights, means, variances, 2.0, 1)
    assert _trapezoid(d, grid) == pytest.approx(1.0, abs=1e-4)


def test_mixture_density_on_grid_skips_zero_weight_components():
    grid = np.linspace(-5, 5, 201)
    weights = np.array([[1.0, 0.0]])
    means = np.array([[0.0, 1e6]])
    variances = np.array([[1.0, 1.0]])
    d = mixture_density_on_grid(grid, weights, means, variances, 1.0, 1)
    from realtimecoin.statics import normal_pdf

    assert np.allclose(d, normal_pdf(grid, 0.0, 1.0))


def test_mixture_density_on_grid_md_integrates_to_one():
    grid = np.linspace(-10, 10, 401)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    weights = np.array([[1.0]])                        # (P, C)
    means = np.zeros((1, 1, 2))
    variances = np.array([[[[1.0, 0.2], [0.2, 1.4]]]])
    d = mixture_density_on_grid(points, weights, means, variances, 1.0, 2)
    mass = _trapezoid(_trapezoid(d.reshape(xx.shape), grid, axis=1), grid)
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_mixture_density_on_grid_rejects_a_mis_shaped_grid():
    with pytest.raises(ValueError, match="GridDimensionMismatch"):
        mixture_density_on_grid(
            np.zeros((5, 3)), np.ones((1, 1)), np.zeros((1, 1, 2)),
            np.array([[np.eye(2)]]), 1.0, 2,
        )


def test_mixture_density_on_grid_zero_normalizer_leaves_the_sum_unscaled():
    grid = np.linspace(-3, 3, 51)
    weights = np.array([[1.0]])
    means = np.array([[0.0]])
    variances = np.array([[1.0]])
    scaled = mixture_density_on_grid(grid, weights, means, variances, 0.0, 1)
    unscaled = mixture_density_on_grid(grid, weights, means, variances, 1.0, 1)
    assert np.allclose(scaled, unscaled)


# --------------------------------------------------------------------------
# Stationary moments
# --------------------------------------------------------------------------


def test_stationary_state_mean_scalar():
    a = np.array([0.5, 1.0, 0.9])
    d = np.array([0.5, 1.0, 0.1])
    m = stationary_state_mean(a, d)
    assert m[0] == pytest.approx(1.0)
    assert m[1] == 0.0            # a == 1: no finite stationary mean
    assert m[2] == pytest.approx(1.0)


def test_stationary_state_var_scalar():
    a = np.array([0.0, 0.5, 1.0, -1.0])
    v = stationary_state_var(a, 0.2)
    assert v[0] == pytest.approx(0.04)
    assert v[1] == pytest.approx(0.04 / 0.75)
    assert v[2] == 0.0            # |a| == 1: no finite stationary variance
    assert v[3] == 0.0


def test_stationary_state_mean_md_solves_the_fixed_point():
    a = np.array([[0.5, 0.1], [0.0, 0.4]])
    d = np.array([1.0, 2.0])
    m = stationary_state_mean_md(a, d)
    assert np.allclose(a @ m + d, m)


def test_stationary_state_mean_md_pinv_fallback_stays_finite():
    a = np.eye(2)                 # I - A is singular
    m = stationary_state_mean_md(a, np.array([1.0, 0.0]))
    assert np.all(np.isfinite(m))


def test_stationary_state_cov_md_solves_the_lyapunov_equation():
    a = np.array([[0.6, 0.2], [-0.1, 0.5]])
    q = np.array([[0.4, 0.05], [0.05, 0.3]])
    p = stationary_state_cov_md(a, q)
    residual = p - (a @ p @ a.T + q)
    assert np.max(np.abs(residual)) < 1e-10
    assert np.allclose(p, p.T)
    assert np.all(np.linalg.eigvalsh(p) >= -1e-12)


def test_stationary_state_cov_md_matches_the_scalar_formula():
    a = np.array([[0.5]])
    q = np.array([[0.09]])
    p = stationary_state_cov_md(a, q)
    assert p[0, 0] == pytest.approx(0.09 / (1 - 0.25))
    assert p[0, 0] == pytest.approx(float(stationary_state_var(0.5, 0.3)))


def test_stationary_state_cov_md_pinv_fallback_on_the_stability_boundary():
    a = np.eye(2)                 # I - kron(A, A) is singular
    q = np.diag([0.1, 0.2])
    p = stationary_state_cov_md(a, q)
    assert np.all(np.isfinite(p))
    assert np.allclose(p, p.T)
    assert np.all(np.linalg.eigvalsh(p) >= -1e-12)


# --------------------------------------------------------------------------
# Jeffreys divergences
# --------------------------------------------------------------------------


def test_jeffreys_finite_clip():
    assert jeffreys_finite_clip(np.inf) == REALMAX
    assert jeffreys_finite_clip(np.nan) == REALMAX
    assert jeffreys_finite_clip(-1e-18) == 0.0
    assert jeffreys_finite_clip(2.5) == pytest.approx(2.5)


def test_gaussian_jeffreys_is_zero_for_identical_gaussians():
    assert gaussian_jeffreys(1.0, 2.0, 1.0, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_gaussian_jeffreys_matches_the_closed_form():
    m1, v1, m2, v2 = 0.0, 1.0, 1.0, 4.0
    expected = 0.5 * (v1 / v2 + v2 / v1 + (m1 - m2) ** 2 * (1 / v1 + 1 / v2) - 2)
    assert gaussian_jeffreys(m1, v1, m2, v2) == pytest.approx(expected)


def test_gaussian_jeffreys_treats_infinite_variance_as_diffuse():
    d = gaussian_jeffreys(0.0, np.inf, 0.0, 1.0)
    assert np.isfinite(d)
    assert d > 0


def test_gaussian_jeffreys_multi_collapses_to_the_scalar_case():
    d_multi = gaussian_jeffreys_multi(
        np.array([0.0]), np.array([[1.0]]), np.array([1.0]), np.array([[4.0]])
    )
    d_scalar = gaussian_jeffreys(0.0, 1.0, 1.0, 4.0)
    assert d_multi == pytest.approx(d_scalar, rel=1e-8)


def test_gaussian_jeffreys_multi_is_symmetric_and_non_negative():
    m1 = np.array([0.0, 1.0])
    s1 = np.array([[1.0, 0.2], [0.2, 2.0]])
    m2 = np.array([1.0, -1.0])
    s2 = np.array([[2.0, -0.3], [-0.3, 1.0]])
    forward = gaussian_jeffreys_multi(m1, s1, m2, s2)
    backward = gaussian_jeffreys_multi(m2, s2, m1, s1)
    assert forward == pytest.approx(backward, rel=1e-10)
    assert forward > 0


def test_categorical_jeffreys_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert categorical_jeffreys(p, p) == pytest.approx(0.0, abs=1e-12)


def test_categorical_jeffreys_zero_pads_the_shorter_vector():
    d = categorical_jeffreys(np.array([0.5, 0.5]), np.array([0.5, 0.25, 0.25]))
    assert np.isfinite(d)
    assert d > 0


def test_categorical_jeffreys_matches_the_closed_form():
    p = np.array([0.25, 0.75])
    q = np.array([0.5, 0.5])
    expected = np.sum((p - q) * np.log(p / q))
    assert categorical_jeffreys(p, q) == pytest.approx(expected, rel=1e-9)

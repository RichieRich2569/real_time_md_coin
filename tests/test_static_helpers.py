"""Unit tests for the public static helpers.

Translation of ``tests/test_static_helpers.m`` (the MATLAB plain-function
test), extended with the degenerate branches the MATLAB test did not cover:
an all ``-inf`` log-sum-exp row, a single-context and a reducible stationary
distribution, and the ``v <= 0`` branches of the normal pdf / cdf.
"""

import numpy as np
import pytest

from realtimecoin.statics import (
    EPS,
    log_sum_exp,
    normal_cdf,
    normal_pdf,
    sample_num_tables,
    stationary_distribution,
    systematic_resampling,
)

# numpy 2 renamed trapz -> trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def test_normal_pdf_integrates_to_one():
    x = np.linspace(-6, 6, 2001)
    p = normal_pdf(x, 0, 1)
    assert abs(_trapezoid(p, x) - 1) < 1e-4


def test_normal_pdf_matches_closed_form():
    x = np.array([-1.5, 0.0, 2.25])
    expected = np.exp(-0.5 * (x - 0.5) ** 2 / 2.0) / np.sqrt(2 * np.pi * 2.0)
    assert np.allclose(normal_pdf(x, 0.5, 2.0), expected)


def test_normal_pdf_degenerate_variance_is_a_spike():
    x = np.array([-1.0, 0.0, np.sqrt(EPS) / 2, 1.0])
    p = normal_pdf(x, 0.0, 0.0)
    assert p[0] == 0.0
    assert p[3] == 0.0
    assert p[1] == pytest.approx(1.0 / np.sqrt(EPS))
    assert p[2] == pytest.approx(1.0 / np.sqrt(EPS))


def test_normal_pdf_non_finite_results_become_realmax():
    # A nan query would give a nan density; the MATLAB guard replaces every
    # non-finite entry with realmax so downstream weights stay finite.
    p = normal_pdf(np.array([np.nan, 0.0]), 0.0, 1.0)
    assert p[0] == np.finfo(float).max
    assert np.isfinite(p[1])


def test_normal_pdf_floors_tiny_variance_at_eps():
    # A vanishing (but positive, non-scalar-degenerate) variance floors at eps.
    p = normal_pdf(np.array([0.0]), 0.0, np.array([1e-320]))
    assert p[0] == pytest.approx(1.0 / np.sqrt(2 * np.pi * EPS))


def test_normal_cdf_matches_known_quantiles():
    assert normal_cdf(0.0, 0.0, 1.0) == pytest.approx(0.5)
    assert normal_cdf(1.96, 0.0, 1.0) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96, 0.0, 1.0) == pytest.approx(0.025, abs=1e-3)


def test_normal_cdf_is_bounded_and_monotone():
    x = np.linspace(-40, 40, 501)
    p = normal_cdf(x, 1.0, 4.0)
    assert np.all(p >= 0) and np.all(p <= 1)
    assert np.all(np.diff(p) >= -1e-15)


def test_normal_pdf_nan_variance_floors_at_eps():
    # MATLAB's max(NaN, eps) is eps, so a poisoned variance still gives a
    # finite density rather than nan.
    p = normal_pdf(np.array([0.0, 1.0]), 0.0, np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(p))


def test_normal_cdf_nan_clamps_to_zero():
    # MATLAB's min(max(p, 0), 1) maps a nan to 0.
    p = normal_cdf(np.array([np.nan]), 0.0, 1.0)
    assert p[0] == 0.0


def test_normal_cdf_degenerate_variance_is_a_step():
    x = np.array([-1.0, 0.0, 1.0])
    p = normal_cdf(x, 0.0, -1.0)
    assert np.array_equal(p, np.array([0.0, 1.0, 1.0]))


def test_log_sum_exp_matches_direct_sum():
    log_p = np.log(np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]]))
    lse = log_sum_exp(log_p, 0)
    assert np.max(np.abs(np.exp(lse) - np.sum(np.exp(log_p), axis=0))) < 1e-12


def test_log_sum_exp_defaults_to_the_trailing_context_axis():
    # The default is -1, not MATLAB's dim = 1, because the layout is
    # transposed: contexts are the trailing axis here.
    log_p = np.log(np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]]))
    assert np.allclose(log_sum_exp(log_p), log_sum_exp(log_p, -1))
    assert np.allclose(np.exp(log_sum_exp(log_p)), np.sum(np.exp(log_p), axis=-1))


def test_log_sum_exp_all_minus_inf_row():
    log_p = np.array([[-np.inf, -np.inf], [0.0, 0.0]])
    lse = log_sum_exp(log_p, 1)
    assert lse[0] == -np.inf
    assert lse[1] == pytest.approx(np.log(2.0))


def test_log_sum_exp_is_shift_stable():
    log_p = np.array([[-1e4, -1e4 + np.log(2.0)]])
    lse = log_sum_exp(log_p, 1)
    assert lse[0] == pytest.approx(-1e4 + np.log(3.0))


def test_stationary_distribution_is_stationary_and_normalised():
    t = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = stationary_distribution(t)
    assert np.max(np.abs(pi @ t - pi)) < 1e-10
    assert abs(pi.sum() - 1) < 1e-12
    # Closed form for a 2-state chain: pi = [b, a] / (a + b) with a = T(1, 2).
    assert np.allclose(pi, np.array([2.0 / 3.0, 1.0 / 3.0]))


def test_stationary_distribution_single_context():
    pi = stationary_distribution(np.array([[1.0]]))
    assert pi.shape == (1,)
    assert pi[0] == pytest.approx(1.0)


def test_stationary_distribution_reducible_chain():
    # State 0 is absorbing and state 1 leaks into it: all mass ends up on 0.
    t = np.array([[1.0, 0.0], [0.5, 0.5]])
    pi = stationary_distribution(t)
    assert np.all(pi >= 0)
    assert pi.sum() == pytest.approx(1.0)
    assert np.max(np.abs(pi @ t - pi)) < 1e-10
    assert np.allclose(pi, np.array([1.0, 0.0]))


def test_stationary_distribution_uniform_for_doubly_stochastic():
    t = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])
    pi = stationary_distribution(t)
    assert np.allclose(pi, np.ones(3) / 3)


def test_systematic_resampling_returns_valid_indices():
    rng = np.random.default_rng(4)
    idx = systematic_resampling(rng, [0.0, 0.25, 0.75])
    assert idx.size == 3
    assert np.all(idx >= 0) and np.all(idx <= 2)
    # A zero weight can never be selected.
    assert not np.any(idx == 0)


def test_systematic_resampling_uniform_fallback_on_zero_mass():
    rng = np.random.default_rng(0)
    idx = systematic_resampling(rng, np.zeros(5))
    assert idx.size == 5
    assert sorted(idx.tolist()) == [0, 1, 2, 3, 4]


def test_systematic_resampling_ignores_non_finite_weights():
    rng = np.random.default_rng(1)
    idx = systematic_resampling(rng, [np.nan, 1.0, np.inf, 1.0])
    assert np.all((idx == 1) | (idx == 3))
    assert idx.size == 4


def test_systematic_resampling_is_low_variance():
    # With n particles and weights w, index i must appear floor(n w_i) or
    # ceil(n w_i) times: that is the defining property of the scheme.
    rng = np.random.default_rng(7)
    w = np.array([0.1, 0.2, 0.7])
    n = w.size
    idx = systematic_resampling(rng, w)
    counts = np.bincount(idx, minlength=n)
    assert np.all(counts >= np.floor(n * w))
    assert np.all(counts <= np.ceil(n * w))


def test_sample_num_tables_bounds():
    rng = np.random.default_rng(4)
    counts = np.array([[0.0, 3.0], [2.0, 5.0]])
    tables = sample_num_tables(rng, np.ones((2, 2)), counts)
    assert np.all(tables >= 0)
    assert np.all(tables <= counts)


def test_sample_num_tables_zero_base_gives_no_tables():
    rng = np.random.default_rng(2)
    tables = sample_num_tables(rng, np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.all(tables == 0)


@pytest.mark.statistical
@pytest.mark.parametrize("base, n", [(0.5, 10), (2.0, 20), (5.0, 5), (1.0, 50)])
def test_sample_num_tables_mean_matches_the_antoniak_expectation(base, n):
    # E[tables] = sum_{i=1..n} base / (base + i - 1).
    rng = np.random.default_rng(6)
    draws = np.array(
        [sample_num_tables(rng, np.array([base]), np.array([float(n)]))[0]
         for _ in range(20_000)]
    )
    expected = sum(base / (base + i - 1) for i in range(1, n + 1))
    assert draws.mean() == pytest.approx(expected, abs=0.05)
    assert draws.min() >= 0 and draws.max() <= n


def test_sample_num_tables_first_customer_always_opens_a_table():
    rng = np.random.default_rng(3)
    # With one customer the probability b / (b + 0) is 1, so a table always
    # opens regardless of the uniform draw.
    tables = sample_num_tables(rng, np.array([0.5, 10.0]), np.array([1.0, 1.0]))
    assert np.array_equal(tables, np.array([1.0, 1.0]))

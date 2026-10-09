"""Tests for src.scaling."""

import numpy as np
import pytest

from src.scaling import (
    compute_cumulative_impact,
    estimate_scaling_exponent,
    ks_distance,
    simulate_white_impact_with_mu,
    ternary_sign_process,
)


# ── compute_cumulative_impact ─────────────────────────────────────────────────

def test_cumulative_impact_sum():
    """Non-overlapping window sums must be correct for a known input."""
    j = np.arange(1.0, 11.0)   # [1, 2, ..., 10]
    h = 5
    result = compute_cumulative_impact(j, h)

    assert len(result) == 2, f"Expected 2 windows (floor(10/5)), got {len(result)}"
    np.testing.assert_allclose(result[0], 15.0, rtol=1e-12)   # 1+2+3+4+5
    np.testing.assert_allclose(result[1], 40.0, rtol=1e-12)   # 6+7+8+9+10


def test_cumulative_impact_partial_window():
    """Trailing elements that do not form a complete window must be discarded."""
    j = np.ones(11)            # 11 elements, h=5 → 2 full windows, 1 leftover
    result = compute_cumulative_impact(j, h=5)
    assert len(result) == 2
    np.testing.assert_allclose(result, [5.0, 5.0])


# ── estimate_scaling_exponent ─────────────────────────────────────────────────

def test_scaling_exponent_known():
    """Estimate beta must recover exactly 0.5 for values = sqrt(params)."""
    params = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    values = np.sqrt(params)   # log(values) = 0.5 * log(params) → beta = 0.5
    result = estimate_scaling_exponent(values, params)
    assert abs(result["beta"] - 0.5) < 1e-10, (
        f"Estimated beta={result['beta']:.8f}, expected 0.5"
    )


def test_scaling_exponent_dict_keys():
    """Result dict must contain 'beta', 'intercept', 'r2', and 'se_beta'."""
    params = np.array([1.0, 2.0, 4.0])
    values = params ** 0.3
    result = estimate_scaling_exponent(values, params)
    assert {"beta", "intercept", "r2", "se_beta"} <= set(result.keys())


# ── simulate_white_impact_with_mu ─────────────────────────────────────────────

def test_white_impact_variance():
    """Var(j_t) must be within 5% of kappa * mu."""
    rng = np.random.default_rng(42)
    kappa, mu = 2.0, 0.05
    j = simulate_white_impact_with_mu(T=200_000, kappa=kappa, mu=mu, rng=rng)
    expected_var = kappa * mu   # = 0.10
    rel_err = abs(np.var(j) - expected_var) / expected_var
    assert rel_err < 0.05, (
        f"Relative var error {rel_err:.4f} > 5%; "
        f"sample var={np.var(j):.5f}, expected={expected_var:.5f}"
    )


# ── ternary_sign_process ──────────────────────────────────────────────────────

def test_ternary_sign_range():
    """Every output value must be in {-1, 0, +1}."""
    rng = np.random.default_rng(42)
    eta = rng.standard_normal(50_000)
    eps = ternary_sign_process(eta, a=0.5)
    unique = set(np.unique(eps))
    assert unique <= {-1.0, 0.0, 1.0}, (
        f"Unexpected values in output: {unique - {-1.0, 0.0, 1.0}}"
    )


def test_ternary_sign_symmetry():
    """For symmetric latent innovations, P(eps=+1) ≈ P(eps=-1) within 2%."""
    rng = np.random.default_rng(42)
    n = 100_000
    eta = rng.standard_normal(n)
    eps = ternary_sign_process(eta, a=0.5)
    p_pos = float(np.mean(eps == 1.0))
    p_neg = float(np.mean(eps == -1.0))
    assert abs(p_pos - p_neg) < 0.02, (
        f"|P(+1) - P(-1)| = {abs(p_pos - p_neg):.4f} exceeds 0.02"
    )


# ── ks_distance ───────────────────────────────────────────────────────────────

def test_ks_distance_same():
    """KS distance between a sample and itself must be exactly 0."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1_000)
    assert ks_distance(x, x) == 0.0


def test_ks_distance_different():
    """KS distance between clearly separated distributions must be large (> 0.5)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1_000)
    y = rng.standard_normal(1_000) + 5.0   # shifted by 5 sigma
    dist = ks_distance(x, y)
    assert dist > 0.5, f"KS distance {dist:.4f} not large enough (expected > 0.5)"

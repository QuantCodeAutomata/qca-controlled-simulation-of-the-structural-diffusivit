"""Tests for src.simulation."""

import numpy as np
import pytest

from src.simulation import (
    simulate_counterfactual_returns,
    simulate_full_system,
    simulate_impact_increments,
    simulate_innovations,
)


T = 100_000
B = 1_000
SIGMA_Y = 1.5
C = 2.0
U0 = 0.5
DT = 1.0


def _make_rngs(seed_eta=42, seed_xi=123):
    return (
        np.random.default_rng(seed_eta),
        np.random.default_rng(seed_xi),
    )


# ── simulate_innovations ──────────────────────────────────────────────────────

def test_innovations_shape():
    """simulate_innovations must return an array of shape (T+B,)."""
    rng, _ = _make_rngs()
    eta = simulate_innovations(T, B, rng)
    assert eta.shape == (T + B,)


# ── simulate_counterfactual_returns ───────────────────────────────────────────

def test_counterfactual_returns_shape():
    """simulate_counterfactual_returns must return shape (T,) after discarding burn-in."""
    _, rng_xi = _make_rngs()
    y = simulate_counterfactual_returns(T, B, SIGMA_Y, DT, rng_xi)
    assert y.shape == (T,)


def test_counterfactual_returns_variance():
    """Var(y_t) must be within 5% of sigma_Y^2 * dt for T=100 000."""
    _, rng_xi = _make_rngs()
    y = simulate_counterfactual_returns(T, B, SIGMA_Y, DT, rng_xi)
    expected_var = SIGMA_Y ** 2 * DT
    rel_err = abs(np.var(y) - expected_var) / expected_var
    assert rel_err < 0.05, (
        f"Relative error {rel_err:.4f} exceeds 5%; "
        f"sample var={np.var(y):.4f}, expected={expected_var:.4f}"
    )


# ── simulate_impact_increments ────────────────────────────────────────────────

def test_impact_increments_variance():
    """Var(j_t) must be within 5% of c^2 (all-pass filter has unit l^2-norm)."""
    rng_eta, _ = _make_rngs()
    eta = simulate_innovations(T, B, rng_eta)
    j = simulate_impact_increments(eta, U0, C, B)
    expected_var = C ** 2
    rel_err = abs(np.var(j) - expected_var) / expected_var
    assert rel_err < 0.05, (
        f"Relative error {rel_err:.4f} exceeds 5%; "
        f"sample var={np.var(j):.4f}, expected={expected_var:.4f}"
    )


# ── simulate_full_system ──────────────────────────────────────────────────────

def test_full_system_keys():
    """Result dict must contain the five required keys."""
    result = simulate_full_system(T, B, SIGMA_Y, U0, C, DT, seed_eta=42, seed_xi=123)
    required = {"x", "y", "j", "eta", "xi"}
    assert required <= set(result.keys()), (
        f"Missing keys: {required - set(result.keys())}"
    )


def test_full_system_structure():
    """x must equal y + j element-wise (the additive decomposition)."""
    result = simulate_full_system(T, B, SIGMA_Y, U0, C, DT, seed_eta=42, seed_xi=123)
    np.testing.assert_allclose(
        result["x"], result["y"] + result["j"],
        rtol=1e-12, atol=1e-12,
        err_msg="x != y + j: additive decomposition violated",
    )


def test_independence_of_rngs():
    """eta and xi must be statistically independent (|corr| < 0.01)."""
    result = simulate_full_system(
        T=50_000, B=500, sigma_Y=1.0, u0=0.5, c=1.0, dt=1.0,
        seed_eta=42, seed_xi=123,
    )
    eta = result["eta"]   # shape (T+B,)
    xi = result["xi"]     # shape (T+B,)
    corr = float(np.corrcoef(eta, xi)[0, 1])
    assert abs(corr) < 0.01, (
        f"|corr(eta, xi)| = {abs(corr):.4f} >= 0.01; "
        "RNGs may not be independent"
    )

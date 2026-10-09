"""Tests for src.impact_cost."""

import numpy as np
import pytest

from src.impact_cost import (
    estimate_impact_cost,
    estimate_impact_cost_batched,
    theoretical_proportional_score,
    verify_whiteness_of_impact,
)


# ── estimate_impact_cost ──────────────────────────────────────────────────────

def test_impact_cost_positive_case():
    """Impact cost must be positive when q_t is anti-correlated with future j.

    Setting q[t] = -j[t+1] means the participant sells when tomorrow's impact
    is positive (price rising), making the future-sum term dominate negatively,
    so C_H = -(1/T)*sum(q * future_j) > 0.
    """
    rng = np.random.default_rng(42)
    T = 50_000
    H_C = 5
    j = rng.standard_normal(T)
    q = np.zeros(T)
    q[: T - 1] = -j[1:]    # q[t] = -j[t+1] → anti-correlated with future j

    cost = estimate_impact_cost(q, j, H_C)
    assert cost > 0, f"Expected positive impact cost, got {cost:.6f}"


def test_impact_cost_zero_case():
    """Impact cost must be near zero for q and j that are statistically independent."""
    rng_j = np.random.default_rng(42)
    rng_q = np.random.default_rng(999)
    T = 100_000
    H_C = 5
    j = rng_j.standard_normal(T)
    q = rng_q.standard_normal(T)   # independent of j

    cost = estimate_impact_cost(q, j, H_C)
    # For independent unit-variance series: std(C_H) ≈ sqrt(H_C / T) ≈ 0.007
    # Use a generous 10-sigma bound
    bound = 10.0 * np.sqrt(H_C / T)
    assert abs(cost) < bound, (
        f"|cost|={abs(cost):.4f} exceeds {bound:.4f} (10-sigma bound)"
    )


# ── theoretical_proportional_score ───────────────────────────────────────────

def test_theoretical_score_positive():
    """K(u0, z1, z2) must be positive for valid parameter inputs."""
    K = theoretical_proportional_score(u0=0.5, z1=0.2, z2=0.8)
    assert K > 0, f"Theoretical score is not positive: K={K}"


def test_theoretical_score_range():
    """K must increase when z2 increases (with u0, z1 fixed)."""
    u0, z1 = 0.5, 0.2
    K_small = theoretical_proportional_score(u0, z1, z2=0.5)
    K_large = theoretical_proportional_score(u0, z1, z2=0.8)
    assert K_large > K_small, (
        f"K did not increase with z2: K(z2=0.5)={K_small:.4f}, K(z2=0.8)={K_large:.4f}"
    )


# ── estimate_impact_cost_batched ──────────────────────────────────────────────

def test_batched_cost_consistency():
    """Batched mean must agree with the full-series estimate to within 5%."""
    rng = np.random.default_rng(42)
    T = 100_000
    H_C = 5
    n_batches = 10
    j = rng.standard_normal(T)
    q = np.zeros(T)
    q[: T - 1] = -j[1:]     # same anti-correlated setup as positive case

    full_cost = estimate_impact_cost(q, j, H_C)
    mean_batched, _ = estimate_impact_cost_batched(q, j, H_C, n_batches)

    rel_err = abs(mean_batched - full_cost) / (abs(full_cost) + 1e-8)
    assert rel_err < 0.05, (
        f"Batched mean {mean_batched:.4f} differs from full {full_cost:.4f} "
        f"by {rel_err*100:.1f}% (> 5%)"
    )


# ── verify_whiteness_of_impact ────────────────────────────────────────────────

def test_whiteness_of_white_noise():
    """For i.i.d. j, max off-zero autocovariance must be small relative to variance."""
    rng = np.random.default_rng(42)
    T = 200_000
    max_lag = 20
    j = rng.standard_normal(T)

    result = verify_whiteness_of_impact(j, max_lag)
    lag0_var = result["lag0_var"]          # ≈ 1.0
    max_off = result["max_off_zero_abs"]   # ≈ O(1/sqrt(T))

    # max|R[t]| for t!=0 should be << R[0]; bound at 10/sqrt(T) of R[0]
    threshold = lag0_var * 10.0 / np.sqrt(T)
    assert max_off < threshold, (
        f"max_off_zero_abs={max_off:.4e} >= {threshold:.4e} "
        f"(lag0_var={lag0_var:.4f})"
    )

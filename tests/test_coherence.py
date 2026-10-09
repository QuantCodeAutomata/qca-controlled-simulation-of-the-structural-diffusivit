"""Tests for src.coherence."""

import numpy as np
import pytest

from src.coherence import (
    compute_distortion_profile,
    detectability_proxy,
    fit_power_law,
    solve_ncoh_star,
    theoretical_ncoh_star,
)


LAM = 0.1
ALPHA = 0.05
SIGMA_Y = 1.0
DT = 1.0


# ── compute_distortion_profile ────────────────────────────────────────────────

def test_distortion_profile_shape():
    """Output array must have shape (max_lag+1,)."""
    n_coh = 10
    max_lag = 20
    profile = compute_distortion_profile(LAM, ALPHA, n_coh, max_lag)
    assert len(profile) == max_lag + 1, (
        f"Expected length {max_lag + 1}, got {len(profile)}"
    )


def test_distortion_profile_zero_at_lag0():
    """Lag-0 entry of the distortion profile must be exactly 0 by definition."""
    profile = compute_distortion_profile(LAM, ALPHA, n_coh=10, max_lag=20)
    assert profile[0] == 0.0, f"profile[0] = {profile[0]}, expected 0.0"


def test_distortion_profile_nonzero_at_positive_lags():
    """Positive-lag entries within n_coh must be nonzero when lam and alpha are positive."""
    n_coh = 5
    max_lag = 20
    profile = compute_distortion_profile(LAM, ALPHA, n_coh, max_lag)
    # Lags 1..n_coh should be non-zero for lam > 0
    for h in range(1, n_coh + 1):
        assert profile[h] != 0.0, f"profile[{h}] unexpectedly zero"


# ── detectability_proxy ───────────────────────────────────────────────────────

def test_detectability_proxy_positive():
    """Proxy must be positive when the distortion profile has non-zero positive-lag entries."""
    profile = compute_distortion_profile(LAM, ALPHA, n_coh=5, max_lag=20)
    proxy = detectability_proxy(profile, sigma_Y=SIGMA_Y, dt=0.01)
    assert proxy > 0.0, f"Detectability proxy is non-positive: {proxy}"


def test_detectability_proxy_zero_for_flat_profile():
    """Proxy must be 0 for an all-zero distortion profile (no distortion)."""
    zero_profile = np.zeros(21)
    proxy = detectability_proxy(zero_profile, sigma_Y=SIGMA_Y, dt=DT)
    assert proxy == 0.0, f"Expected 0.0 for zero profile, got {proxy}"


# ── solve_ncoh_star ───────────────────────────────────────────────────────────

def test_solve_ncoh_star_decreasing():
    """n_coh* must be non-increasing in mu (higher participation → shorter coherence)."""
    mu_vals = [0.001, 0.01, 0.1]
    ncoh_vals = [
        solve_ncoh_star(mu=mu, N_D=250, S=0.5, lam=LAM, alpha=ALPHA, sigma_Y=SIGMA_Y)
        for mu in mu_vals
    ]
    for i in range(len(ncoh_vals) - 1):
        assert ncoh_vals[i] >= ncoh_vals[i + 1], (
            f"n_coh* not non-increasing at index {i}: "
            f"{ncoh_vals[i]} < {ncoh_vals[i+1]}"
        )


def test_solve_ncoh_star_returns_positive_int():
    """Return value must be a positive integer."""
    result = solve_ncoh_star(mu=0.01, N_D=250, S=0.5, lam=LAM, alpha=ALPHA,
                             sigma_Y=SIGMA_Y)
    assert isinstance(result, int)
    assert result >= 1


# ── fit_power_law ─────────────────────────────────────────────────────────────

def test_power_law_fit_known():
    """OLS fit must recover the exact slope for a noise-free power law."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    slope_true = 2.5
    y = x ** slope_true
    result = fit_power_law(x, y)
    assert abs(result["slope"] - slope_true) < 1e-10, (
        f"Fitted slope {result['slope']:.10f} != {slope_true}"
    )


def test_power_law_fit_keys():
    """Result dict must contain 'slope', 'intercept', and 'r2'."""
    x = np.array([1.0, 2.0, 4.0])
    y = x ** 1.3
    result = fit_power_law(x, y)
    assert {"slope", "intercept", "r2"} <= set(result.keys())


# ── theoretical_ncoh_star ─────────────────────────────────────────────────────

def test_theoretical_ncoh_negative_onethird():
    """theoretical_ncoh_star must be strictly decreasing in mu (exponent = -1/3)."""
    mu_vals = [0.001, 0.01, 0.1, 1.0]
    ncoh_vals = [
        theoretical_ncoh_star(mu=mu, S=1.0, lam=LAM, alpha=ALPHA)
        for mu in mu_vals
    ]
    for i in range(len(ncoh_vals) - 1):
        assert ncoh_vals[i] > ncoh_vals[i + 1], (
            f"theoretical_ncoh_star not decreasing: "
            f"ncoh({mu_vals[i]})={ncoh_vals[i]:.4f}, "
            f"ncoh({mu_vals[i+1]})={ncoh_vals[i+1]:.4f}"
        )


def test_theoretical_ncoh_power_law_slope():
    """Verify the -1/3 scaling exponent numerically by fitting log-log regression."""
    mu_vals = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ncoh_vals = np.array([
        theoretical_ncoh_star(mu=mu, S=1.0, lam=LAM, alpha=ALPHA)
        for mu in mu_vals
    ])
    result = fit_power_law(mu_vals, ncoh_vals)
    assert abs(result["slope"] - (-1.0 / 3.0)) < 1e-10, (
        f"Slope {result['slope']:.10f} != -1/3"
    )

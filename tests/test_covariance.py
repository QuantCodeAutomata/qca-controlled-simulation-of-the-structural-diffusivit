"""Tests for src.covariance."""

import numpy as np
import pytest

from src.covariance import (
    compute_sample_variance,
    estimate_autocovariance,
    estimate_crosscovariance,
    rms_deviation,
    verify_structural_identity,
)


RNG = np.random.default_rng(42)
T = 100_000
MAX_LAG = 20


def _white(seed=42, n=T) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


# ── estimate_autocovariance ───────────────────────────────────────────────────

def test_autocovariance_lag0_equals_variance():
    """R_xx[0] must equal the sample variance (same T-normalisation)."""
    x = _white(seed=0)
    R = estimate_autocovariance(x, MAX_LAG)
    sample_var = compute_sample_variance(x)
    assert R[0] == pytest.approx(sample_var, rel=1e-10)


def test_autocovariance_symmetry():
    """R_xx[-t] must equal R_xx[t] for every positive lag (symmetric process)."""
    x = _white(seed=1)
    R = estimate_autocovariance(x, MAX_LAG)
    for t in range(1, MAX_LAG + 1):
        assert R[-t] == pytest.approx(R[t], rel=1e-10), (
            f"R[{-t}]={R[-t]:.6e} != R[{t}]={R[t]:.6e}"
        )


def test_autocovariance_white_noise():
    """Off-zero lags of iid N(0,1) must be small relative to lag-0 variance."""
    x = _white(seed=2)
    R = estimate_autocovariance(x, MAX_LAG)
    # For iid series, |R[t]| ~ 1/sqrt(T) for t != 0
    threshold = 10.0 / np.sqrt(T)   # generous margin (~10 sigma)
    for t in range(1, MAX_LAG + 1):
        assert abs(R[t]) < threshold, (
            f"|R[{t}]|={abs(R[t]):.4e} >= {threshold:.4e}"
        )


# ── estimate_crosscovariance ──────────────────────────────────────────────────

def test_crosscovariance_shape():
    """Keys must cover every integer in [-max_lag, max_lag]."""
    x = _white(seed=3)
    y = _white(seed=4)
    R = estimate_crosscovariance(x, y, MAX_LAG)
    expected_keys = set(range(-MAX_LAG, MAX_LAG + 1))
    assert set(R.keys()) == expected_keys, (
        f"Key mismatch: got {len(R)} keys, expected {len(expected_keys)}"
    )


# ── verify_structural_identity ────────────────────────────────────────────────

def test_structural_identity_verification():
    """For white j independent of y, both off-zero and lag-0 structural errors must be small.

    The identity is  R_yj[t] + R_jy[t] + R_jj[t] = dt*(sigma_X^2 - sigma_Y^2)*delta[t].
    When y and j are independent and both white, all cross-covariances vanish and
    R_jj[0] = Var(j), giving zero residual once theoretical parameters are set
    consistently.
    """
    rng = np.random.default_rng(7)
    sigma_Y_val = 1.0
    c_val = 1.5
    dt = 1.0
    n = 100_000

    y = rng.normal(0.0, sigma_Y_val, n)
    j = rng.normal(0.0, c_val, n)

    # Sample covariances
    R_yj = estimate_crosscovariance(y, j, MAX_LAG)
    R_jy = estimate_crosscovariance(j, y, MAX_LAG)
    R_jj = estimate_autocovariance(j, MAX_LAG)

    # Set theoretical parameters from sample moments
    sigma_Y2 = float(np.var(y)) / dt   # ≈ sigma_Y_val^2
    sigma_X2 = float(np.var(y + j)) / dt  # ≈ sigma_Y_val^2 + c_val^2

    result = verify_structural_identity(
        R_yj, R_jy, R_jj, dt, sigma_X2, sigma_Y2, MAX_LAG
    )

    # All errors should be small: O(1/sqrt(T))
    threshold = 15.0 / np.sqrt(n)
    assert result["max_abs_off_zero_error"] < threshold, (
        f"max off-zero structural error {result['max_abs_off_zero_error']:.4e}"
        f" exceeds {threshold:.4e}"
    )
    assert result["lag0_error"] < threshold, (
        f"lag-0 structural error {result['lag0_error']:.4e} exceeds {threshold:.4e}"
    )


# ── rms_deviation ─────────────────────────────────────────────────────────────

def test_rms_deviation_zero():
    """rms_deviation(x, x) must be exactly 0 for any array x."""
    x = _white(seed=5)
    assert rms_deviation(x, x) == 0.0

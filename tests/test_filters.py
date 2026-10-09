"""Tests for src.filters."""

import numpy as np
import pytest

from src.filters import (
    build_allpass_filter,
    build_biexponential_filter,
    build_lti_filter,
    compute_filter_truncation_length,
    verify_allpass_property,
)


# ── build_allpass_filter ──────────────────────────────────────────────────────

def test_allpass_filter_shape():
    """Returned array must have length H+1."""
    for u0 in (0.3, 0.5, 0.8):
        for H in (10, 50, 100):
            U = build_allpass_filter(u0, H)
            assert U.shape == (H + 1,), f"Expected ({H + 1},), got {U.shape}"


def test_allpass_filter_lag0():
    """U[0] must equal u0 exactly."""
    u0 = 0.6
    U = build_allpass_filter(u0, H=50)
    assert U[0] == pytest.approx(u0)


def test_allpass_filter_h1():
    """U[1] must equal -(1 - u0^2), i.e. -(1 - u0^2) * u0^0."""
    u0 = 0.4
    U = build_allpass_filter(u0, H=50)
    expected = -(1.0 - u0 ** 2)
    assert U[1] == pytest.approx(expected)


def test_allpass_filter_tail_decay():
    """Coefficients for h >= 1 must decay geometrically with ratio u0."""
    u0 = 0.5
    H = 50
    U = build_allpass_filter(u0, H)
    # Check ratio U[h+1] / U[h] == u0 for h = 1, ..., H-1
    for h in range(1, H):
        ratio = U[h + 1] / U[h]
        assert ratio == pytest.approx(u0, rel=1e-9), (
            f"Ratio at h={h} was {ratio}, expected {u0}"
        )


def test_allpass_filter_invalid_u0():
    """ValueError must be raised for u0 <= 0 or u0 >= 1."""
    with pytest.raises(ValueError):
        build_allpass_filter(0.0, H=10)
    with pytest.raises(ValueError):
        build_allpass_filter(1.0, H=10)
    with pytest.raises(ValueError):
        build_allpass_filter(-0.5, H=10)
    with pytest.raises(ValueError):
        build_allpass_filter(1.5, H=10)


# ── build_biexponential_filter ────────────────────────────────────────────────

def test_biexponential_filter_sum_near_zero():
    """Sum of L over a large window is near zero (infinite-series limit = 0)."""
    # sum_{h=0}^{H} L[h] = z2^{H+1} - z1^{H+1} → 0 as H → ∞
    z1, z2, H = 0.3, 0.7, 500
    L = build_biexponential_filter(z1, z2, H)
    assert abs(np.sum(L)) < 1e-6, f"sum(L)={np.sum(L):.2e} not near zero"


def test_biexponential_filter_invalid():
    """ValueError must be raised when the ordering constraint is violated."""
    with pytest.raises(ValueError):
        build_biexponential_filter(0.5, 0.3, H=10)   # z1 > z2
    with pytest.raises(ValueError):
        build_biexponential_filter(0.5, 0.5, H=10)   # z1 == z2
    with pytest.raises(ValueError):
        build_biexponential_filter(0.0, 0.5, H=10)   # z1 == 0
    with pytest.raises(ValueError):
        build_biexponential_filter(0.3, 1.0, H=10)   # z2 == 1


# ── compute_filter_truncation_length ─────────────────────────────────────────

def test_truncation_length():
    """Returned H must satisfy base**H < tol."""
    for base in (0.3, 0.5, 0.8, 0.9):
        for tol in (1e-4, 1e-8, 1e-12):
            H = compute_filter_truncation_length(base, tol)
            assert base ** H < tol, (
                f"base={base}, tol={tol}: base**H={base**H:.2e} not < {tol}"
            )


# ── verify_allpass_property ───────────────────────────────────────────────────

def test_verify_allpass_property_lag0():
    """Lag-0 autocorrelation of the truncated all-pass filter must be positive.

    For the infinite filter sum(U[h]^2) = 1 exactly; truncation only
    reduces it slightly, so the result must still be well above zero.
    """
    u0 = 0.5
    H = 200   # long enough that truncation error is negligible
    C = verify_allpass_property(u0, H, L=20)
    assert C[0] > 0, f"C[0]={C[0]:.6f} is not positive"


# ── build_lti_filter ─────────────────────────────────────────────────────────

def test_lti_filter_shape():
    """Output of build_lti_filter must match the input signal's shape."""
    rng = np.random.default_rng(0)
    coeffs = build_allpass_filter(u0=0.5, H=20)
    signal = rng.standard_normal(1000)
    out = build_lti_filter(coeffs, signal)
    assert out.shape == signal.shape, (
        f"Expected shape {signal.shape}, got {out.shape}"
    )

"""
Simulation routines for the structural diffusivity / impact-whitening framework.

Provides independent innovation streams, counterfactual return simulation,
all-pass-filtered impact increments, and biexponential participant flow.
All functions use numpy Generator objects for reproducibility and are
compatible with numpy >= 1.17 random infrastructure.
"""

import numpy as np

from src.filters import (
    build_allpass_filter,
    build_biexponential_filter,
    build_lti_filter,
    compute_filter_truncation_length,
)


def simulate_innovations(T: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Draw i.i.d. N(0,1) innovations of length T+B.

    Parameters
    ----------
    T : int
        Number of usable time steps (after burn-in).
    B : int
        Burn-in period; first B samples are discarded by downstream callers.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Float64 array of shape (T+B,).
    """
    return rng.standard_normal(T + B)


def simulate_counterfactual_returns(
    T: int,
    B: int,
    sigma_Y: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate information-bearing counterfactual returns y_t = sigma_Y * sqrt(dt) * xi_t.

    The innovations xi_t are i.i.d. N(0,1) and are drawn independently of
    the impact innovations eta (they use a separate RNG stream).

    Parameters
    ----------
    T : int
        Number of usable samples returned (after burn-in B is discarded).
    B : int
        Burn-in period drawn and discarded internally.
    sigma_Y : float
        Annualised (or per-unit) volatility of the counterfactual price.
    dt : float
        Time-step size; returns are scaled by sqrt(dt).
    rng : np.random.Generator
        Seeded random generator (must be independent of eta's generator).

    Returns
    -------
    np.ndarray
        Float64 array of shape (T,) containing usable return samples.
    """
    xi_full = rng.standard_normal(T + B)
    return (sigma_Y * np.sqrt(dt) * xi_full[B:]).astype(np.float64)


def simulate_impact_increments(
    eta: np.ndarray,
    u0: float,
    c: float,
    B: int,
) -> np.ndarray:
    """Simulate impact increments j_t = c * (U * eta)[t] via the all-pass filter.

    Parameters
    ----------
    eta : np.ndarray
        Innovation array of length T+B produced by simulate_innovations.
    u0 : float
        All-pass filter pole parameter, strictly in (0, 1).
    c : float
        Impact scale coefficient.
    B : int
        Burn-in period; the first B samples of the filtered output are discarded.

    Returns
    -------
    np.ndarray
        Float64 array of shape (T,) = (len(eta)-B,) of usable impact increments.
    """
    H = compute_filter_truncation_length(u0)
    U = build_allpass_filter(u0, H)
    filtered = build_lti_filter(U, eta)
    return (c * filtered[B:]).astype(np.float64)


def simulate_participant_flow(
    eta: np.ndarray,
    z1: float,
    z2: float,
    B: int,
) -> np.ndarray:
    """Simulate participant flow q_t = (L * eta)[t] via the biexponential filter.

    Parameters
    ----------
    eta : np.ndarray
        Innovation array of length T+B.
    z1 : float
        Fast-decay pole, strictly between 0 and z2.
    z2 : float
        Slow-decay pole, strictly between z1 and 1.
    B : int
        Burn-in period; first B filtered samples are discarded.

    Returns
    -------
    np.ndarray
        Float64 array of shape (T,) = (len(eta)-B,) of usable flow samples.
    """
    H = compute_filter_truncation_length(z2)
    L = build_biexponential_filter(z1, z2, H)
    filtered = build_lti_filter(L, eta)
    return filtered[B:].astype(np.float64)


def simulate_full_system(
    T: int,
    B: int,
    sigma_Y: float,
    u0: float,
    c: float,
    dt: float,
    seed_eta: int = 42,
    seed_xi: int = 123,
) -> dict:
    """Run a full two-stream simulation and return all key quantities.

    Two independent numpy Generators are created—one for the impact
    innovations eta and one for the counterfactual return innovations xi—
    ensuring the two processes are statistically independent.

    Parameters
    ----------
    T : int
        Number of usable time steps.
    B : int
        Burn-in period; first B samples are discarded in each stream.
    sigma_Y : float
        Counterfactual return volatility.
    u0 : float
        All-pass filter pole parameter.
    c : float
        Impact scale coefficient.
    dt : float
        Time-step size; counterfactual returns are scaled by sqrt(dt).
    seed_eta : int, optional
        Seed for the impact innovation RNG (default 42).
    seed_xi : int, optional
        Seed for the counterfactual return RNG (default 123).

    Returns
    -------
    dict with keys:
        'eta' : np.ndarray, shape (T+B,) — raw impact innovations
        'xi'  : np.ndarray, shape (T+B,) — raw counterfactual innovations
        'y'   : np.ndarray, shape (T,)   — counterfactual returns
        'j'   : np.ndarray, shape (T,)   — impact increments
        'x'   : np.ndarray, shape (T,)   — realized returns (y + j)
    """
    rng_eta = np.random.default_rng(seed_eta)
    rng_xi = np.random.default_rng(seed_xi)

    eta = simulate_innovations(T, B, rng_eta)
    xi = simulate_innovations(T, B, rng_xi)

    y = (sigma_Y * np.sqrt(dt) * xi[B:]).astype(np.float64)
    j = simulate_impact_increments(eta, u0, c, B)

    return {
        "eta": eta,
        "xi": xi,
        "y": y,
        "j": j,
        "x": (y + j).astype(np.float64),
    }

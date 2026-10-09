"""
Scaling-law analysis for cumulative market impact.

Studies how the standard deviation of cumulative impact I_h = sum_{k=1}^h j_k
scales with h (horizon) and mu (participation rate), and provides OLS
log-log regression utilities for exponent estimation.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_cumulative_impact(j: np.ndarray, h: int) -> np.ndarray:
    """Compute non-overlapping window sums of j over horizon h.

    Partitions j into floor(T/h) non-overlapping blocks of length h and
    returns the sum within each block.

    Parameters
    ----------
    j : np.ndarray
        Impact increment series of length T.
    h : int
        Window size (horizon).

    Returns
    -------
    np.ndarray
        Float64 array of shape (floor(T/h),) where each entry is the
        cumulative impact over one non-overlapping window.
    """
    T = len(j)
    n_windows = T // h
    return j[: n_windows * h].reshape(n_windows, h).sum(axis=1).astype(np.float64)


def estimate_scaling_exponent(
    values: np.ndarray, params: np.ndarray
) -> dict:
    """OLS regression of log|values| on log|params| to estimate a power-law exponent.

    Fits the model  log|values| = beta * log|params| + intercept + epsilon.

    Parameters
    ----------
    values : np.ndarray
        Dependent variable (e.g. std of cumulative impact); positive entries.
    params : np.ndarray
        Independent variable (e.g. horizon h or mu); positive entries.

    Returns
    -------
    dict with keys:
        'beta'      : float, estimated slope (power-law exponent)
        'intercept' : float, estimated intercept (log of prefactor)
        'r2'        : float, coefficient of determination
        'se_beta'   : float, OLS standard error of beta
    """
    log_v = np.log(np.abs(values)).astype(np.float64)
    log_p = np.log(np.abs(params)).astype(np.float64)

    n = len(log_v)
    X = np.column_stack([log_p, np.ones(n)])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, log_v, rcond=None)
    beta, intercept = float(coeffs[0]), float(coeffs[1])

    y_hat = beta * log_p + intercept
    ss_res = float(np.sum((log_v - y_hat) ** 2))
    ss_tot = float(np.sum((log_v - log_v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    denom = float(np.sum((log_p - log_p.mean()) ** 2))
    se_beta = float(np.sqrt(ss_res / max(n - 2, 1) / denom)) if denom > 0.0 else 0.0

    return {"beta": beta, "intercept": intercept, "r2": float(r2), "se_beta": se_beta}


def simulate_white_impact_with_mu(
    T: int, kappa: float, mu: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate i.i.d. white impact increments j_t ~ N(0, kappa*mu).

    Parameters
    ----------
    T : int
        Number of samples.
    kappa : float
        Per-unit-mu variance coefficient.
    mu : float
        Participation rate (scales variance).
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray
        Float64 array of shape (T,).
    """
    return rng.normal(0.0, np.sqrt(kappa * mu), T).astype(np.float64)


def ternary_sign_process(eta_latent: np.ndarray, a: float) -> np.ndarray:
    """Threshold a latent Gaussian series into a ternary sign process.

    Maps each element to:
        +1  if eta_latent > a
        -1  if eta_latent < -a
         0  otherwise

    Parameters
    ----------
    eta_latent : np.ndarray
        Latent Gaussian innovations.
    a : float
        Threshold (positive); controls the probability of inaction.

    Returns
    -------
    np.ndarray
        Float64 array of shape (T,) with values in {-1, 0, +1}.
    """
    out = np.zeros_like(eta_latent, dtype=np.float64)
    out[eta_latent > a] = 1.0
    out[eta_latent < -a] = -1.0
    return out


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov distance between two empirical distributions.

    Parameters
    ----------
    x : np.ndarray
        First sample.
    y : np.ndarray
        Second sample.

    Returns
    -------
    float
        KS statistic (maximum absolute difference between ECDFs).
    """
    stat, _ = ks_2samp(x, y)
    return float(stat)


def estimate_scaling_grid(
    mus: np.ndarray,
    hs: np.ndarray,
    kappa: float,
    T_per: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate white impact and measure how std of cumulative impact scales.

    For each (mu, h) pair, generates T_per i.i.d. impact increments
    j_t ~ N(0, kappa*mu), accumulates them over non-overlapping windows of
    size h, and records the empirical standard deviation alongside the
    theoretical value sqrt(kappa * mu * h).

    Parameters
    ----------
    mus : np.ndarray
        Grid of participation rates mu (positive).
    hs : np.ndarray
        Grid of horizons h (positive integers).
    kappa : float
        Variance coefficient.
    T_per : int
        Number of impact increments to simulate per (mu, h) cell.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    pd.DataFrame
        Columns: ['mu', 'h', 'std_I', 'theoretical_std'].
    """
    records = []
    for mu in mus:
        for h in hs:
            j = simulate_white_impact_with_mu(T_per, kappa, float(mu), rng)
            windows = compute_cumulative_impact(j, int(h))
            std_I = float(np.std(windows))
            theoretical_std = float(np.sqrt(kappa * float(mu) * float(h)))
            records.append(
                {"mu": float(mu), "h": int(h), "std_I": std_I, "theoretical_std": theoretical_std}
            )
    return pd.DataFrame(records, columns=["mu", "h", "std_I", "theoretical_std"])

"""
Statistical diagnostics for market microstructure simulations.

Provides Ljung-Box serial-correlation tests, diffusivity checks,
quantile-quantile comparisons, Monte Carlo summaries, and ACF
confidence-interval estimation.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as sm_acf


def ljung_box_test(x: np.ndarray, lags: list) -> pd.DataFrame:
    """Ljung-Box portmanteau test for serial correlation.

    Wraps statsmodels.stats.diagnostic.acorr_ljungbox and returns
    a tidy DataFrame indexed by lag.

    Parameters
    ----------
    x : np.ndarray
        Time series to test.
    lags : list
        List of lag values at which to evaluate the test statistic.

    Returns
    -------
    pd.DataFrame
        Columns: ['lb_stat', 'lb_pvalue'], one row per lag.
    """
    result = acorr_ljungbox(x, lags=lags, return_df=True)
    return result[["lb_stat", "lb_pvalue"]].copy()


def diffusivity_check(
    x: np.ndarray,
    max_lag: int,
    dt: float,
    sigma2_theoretical: float,
) -> dict:
    """Assess whether a return series x satisfies the diffusivity conditions.

    A diffusive process has:
        - Var(x_t) ≈ sigma2_theoretical * dt
        - Cov(x_t, x_{t-s}) ≈ 0  for s != 0

    The series is declared diffusive when the maximum absolute off-zero
    autocovariance is below the asymptotic 3/sqrt(T) bound.

    Parameters
    ----------
    x : np.ndarray
        Return series of length T.
    max_lag : int
        Number of lags to check for serial correlation.
    dt : float
        Time-step size.
    sigma2_theoretical : float
        Theoretical variance per unit time.

    Returns
    -------
    dict with keys:
        'lag0_var'           : float, sample variance of x
        'theoretical_var'    : float, sigma2_theoretical * dt
        'off_zero_max_abs'   : float, max |autocovariance| for lags 1..max_lag
        'is_diffusive'       : bool, True if off_zero_max_abs < 3/sqrt(T)
    """
    from src.covariance import estimate_autocovariance

    T = len(x)
    acov = estimate_autocovariance(x, max_lag)
    lag0_var = float(acov[0])
    off_zero_vals = [abs(acov[t]) for t in range(1, max_lag + 1)]
    off_zero_max = float(max(off_zero_vals)) if off_zero_vals else 0.0
    threshold = 3.0 / np.sqrt(T)

    return {
        "lag0_var": lag0_var,
        "theoretical_var": float(sigma2_theoretical * dt),
        "off_zero_max_abs": off_zero_max,
        "is_diffusive": bool(off_zero_max < threshold),
    }


def qq_comparison(
    x: np.ndarray,
    y: np.ndarray,
    n_quantiles: int = 100,
) -> dict:
    """Empirical quantile-quantile comparison and two-sample KS test.

    Computes n_quantiles evenly spaced empirical quantiles of x and y
    and performs a KS test to assess distributional similarity.

    Parameters
    ----------
    x : np.ndarray
        First sample.
    y : np.ndarray
        Second sample.
    n_quantiles : int, optional
        Number of quantile points (default 100).

    Returns
    -------
    dict with keys:
        'x_quantiles' : np.ndarray, quantiles of x
        'y_quantiles' : np.ndarray, quantiles of y
        'ks_stat'     : float, KS two-sample statistic
        'ks_pvalue'   : float, associated p-value
    """
    probs = np.linspace(0.0, 1.0, n_quantiles)
    x_q = np.quantile(x, probs)
    y_q = np.quantile(y, probs)
    ks_stat, ks_pvalue = ks_2samp(x, y)

    return {
        "x_quantiles": x_q,
        "y_quantiles": y_q,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def monte_carlo_summary(results: list) -> dict:
    """Summarise Monte Carlo scalar results with a 95% confidence interval.

    Uses the normal approximation CI:
        [mean - 1.96*std/sqrt(n), mean + 1.96*std/sqrt(n)]

    Parameters
    ----------
    results : list
        List of scalar floats from independent Monte Carlo runs.

    Returns
    -------
    dict with keys:
        'mean'    : float
        'std'     : float
        'ci_low'  : float, lower bound of 95% CI
        'ci_high' : float, upper bound of 95% CI
        'n'       : int, number of runs
    """
    arr = np.asarray(results, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(n) if n > 0 else 0.0

    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "n": n,
    }


def estimate_acf_with_ci(x: np.ndarray, max_lag: int) -> dict:
    """Compute the sample ACF with 95% pointwise confidence intervals.

    Delegates to statsmodels acf with alpha=0.05 for Bartlett-formula CIs.

    Parameters
    ----------
    x : np.ndarray
        Univariate time series.
    max_lag : int
        Number of lags to compute (lags 0..max_lag).

    Returns
    -------
    dict with keys:
        'acf'    : np.ndarray, ACF values for lags 0..max_lag
        'ci_low' : np.ndarray, lower CI bound for each lag
        'ci_high': np.ndarray, upper CI bound for each lag
    """
    acf_vals, confint = sm_acf(x, nlags=max_lag, alpha=0.05)

    return {
        "acf": acf_vals.astype(np.float64),
        "ci_low": confint[:, 0].astype(np.float64),
        "ci_high": confint[:, 1].astype(np.float64),
    }

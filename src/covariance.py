"""
Sample covariance estimation and structural identity verification.

All estimators use T-normalisation (divide by T rather than T-lag) for
consistency across lags when summing covariance functions.
"""

import numpy as np
from scipy.signal import correlate as sp_correlate


def estimate_autocovariance(x: np.ndarray, max_lag: int) -> dict:
    """Estimate sample autocovariance R_xx[t] for lags t in [-max_lag, max_lag].

    Uses T-normalisation (denominator = len(x)) for all lags so that
    the estimator remains consistent when aggregating across lags.

    R_xx[t] = (1/T) * sum_{s} (x[s+t] - mu_x) * (x[s] - mu_x)

    Parameters
    ----------
    x : np.ndarray
        Univariate time series of length T.
    max_lag : int
        Maximum absolute lag to evaluate.

    Returns
    -------
    dict
        Mapping {lag (int): autocovariance (float)} for lag in
        range(-max_lag, max_lag+1).
    """
    T = len(x)
    x_c = (x - x.mean()).astype(np.float64)
    # sp_correlate(a, b, 'full')[T-1+t] = sum_s a[s+t]*b[s]
    full = sp_correlate(x_c, x_c, mode='full', method='fft') / T
    return {t: float(full[T - 1 + t]) for t in range(-max_lag, max_lag + 1)}


def estimate_crosscovariance(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> dict:
    """Estimate sample cross-covariance R_xy[t] for lags t in [-max_lag, max_lag].

    Positive lag t means x leads y:
        R_xy[t] = (1/T) * sum_s (x[s+t] - mu_x) * (y[s] - mu_y)

    Parameters
    ----------
    x : np.ndarray
        First time series of length T.
    y : np.ndarray
        Second time series of length T (same length as x).
    max_lag : int
        Maximum absolute lag.

    Returns
    -------
    dict
        Mapping {lag (int): cross-covariance (float)}.
    """
    T = len(x)
    x_c = (x - x.mean()).astype(np.float64)
    y_c = (y - y.mean()).astype(np.float64)
    full = sp_correlate(x_c, y_c, mode='full', method='fft') / T
    return {t: float(full[T - 1 + t]) for t in range(-max_lag, max_lag + 1)}


def verify_structural_identity(
    R_yj: dict,
    R_jy: dict,
    R_jj: dict,
    dt: float,
    sigma_X2: float,
    sigma_Y2: float,
    max_lag: int,
) -> dict:
    """Check the structural covariance identity at every lag.

    The identity states that for all t:
        R_yj[t] + R_jy[t] + R_jj[t] = dt * (sigma_X^2 - sigma_Y^2) * delta[t]

    where delta[t] = 1 if t == 0, else 0.

    Parameters
    ----------
    R_yj : dict
        Cross-covariance of counterfactual returns with impact (y leads j).
    R_jy : dict
        Cross-covariance of impact with counterfactual returns (j leads y).
    R_jj : dict
        Autocovariance of impact increments.
    dt : float
        Time-step size.
    sigma_X2 : float
        Variance parameter of realized returns per unit time.
    sigma_Y2 : float
        Variance parameter of counterfactual returns per unit time.
    max_lag : int
        Lags evaluated in the supplied dicts (non-negative only is enough,
        but the dict may contain negative lags too).

    Returns
    -------
    dict with keys:
        'S'                    : dict {lag: R_yj+R_jy+R_jj}
        'theoretical'          : dict {lag: dt*(sigma_X2-sigma_Y2)*delta[t]}
        'error'                : dict {lag: S[t]-theoretical[t]}
        'max_abs_off_zero_error': float, max |error[t]| for t != 0
        'lag0_error'           : float, |error[0]|
    """
    lags = list(range(-max_lag, max_lag + 1))
    sigma_diff = dt * (sigma_X2 - sigma_Y2)

    S = {t: R_yj[t] + R_jy[t] + R_jj[t] for t in lags}
    theoretical = {t: sigma_diff if t == 0 else 0.0 for t in lags}
    error = {t: S[t] - theoretical[t] for t in lags}

    off_zero_errors = [abs(error[t]) for t in lags if t != 0]
    max_abs_off_zero = float(np.max(off_zero_errors)) if off_zero_errors else 0.0

    return {
        "S": S,
        "theoretical": theoretical,
        "error": error,
        "max_abs_off_zero_error": max_abs_off_zero,
        "lag0_error": float(abs(error[0])),
    }


def compute_sample_variance(x: np.ndarray) -> float:
    """Compute sample variance of x after demeaning.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    float
        Variance estimate, i.e. mean of (x - mean(x))^2.
    """
    return float(np.var(x - x.mean()))


def rms_deviation(estimated: np.ndarray, theoretical: np.ndarray) -> float:
    """Compute root-mean-square deviation between two arrays.

    Parameters
    ----------
    estimated : np.ndarray
        Estimated values.
    theoretical : np.ndarray
        Reference (theoretical) values of the same shape.

    Returns
    -------
    float
        RMSD = sqrt(mean((estimated - theoretical)^2)).
    """
    return float(np.sqrt(np.mean((np.asarray(estimated) - np.asarray(theoretical)) ** 2)))

"""
Coherence-length analysis for stealth execution.

Studies how the covariance distortion introduced by a participant's
correlated order flow is detectable to an outside observer, and
derives the optimal coherence length n_coh* that keeps detectability
just below a budget threshold S.
"""

import numpy as np


def compute_distortion_profile(
    lam: float,
    alpha: float,
    n_coh: int,
    max_lag: int,
) -> np.ndarray:
    """Build the tapered covariance-distortion profile for lag 0..max_lag.

    The profile is non-zero only on lags 1..n_coh and uses a linear taper:
        profile[h] = (lam**2 + 2*lam*alpha) * (n_coh - h + 1) / n_coh
                     for h in 1..n_coh
        profile[h] = 0   otherwise (including h = 0)

    Parameters
    ----------
    lam : float
        Impact coefficient lambda.
    alpha : float
        Cross-term coefficient alpha.
    n_coh : int
        Coherence length; number of lags over which distortion is spread.
    max_lag : int
        Size of the returned array minus one; must be >= n_coh.

    Returns
    -------
    np.ndarray
        Float64 array of shape (max_lag+1,) indexed by lag h = 0..max_lag.
    """
    profile = np.zeros(max_lag + 1, dtype=np.float64)
    amplitude = lam ** 2 + 2.0 * lam * alpha
    for h in range(1, min(n_coh, max_lag) + 1):
        profile[h] = amplitude * (n_coh - h + 1) / n_coh
    return profile


def detectability_proxy(
    distortion: np.ndarray,
    sigma_Y: float,
    dt: float,
) -> float:
    """Compute the detectability proxy (normalised RMS distortion).

    Std_hat = (1 / (sigma_Y * sqrt(dt))) * sqrt(sum_{t >= 1} distortion[t]^2)

    A larger value means the participant's covariance distortion is more
    easily distinguished from noise by an outside observer.

    Parameters
    ----------
    distortion : np.ndarray
        Distortion profile array; index 0 is lag-0 (unused in sum), indices
        1.. are lags 1..max_lag.
    sigma_Y : float
        Benchmark (counterfactual) return volatility.
    dt : float
        Time-step size.

    Returns
    -------
    float
        Detectability proxy Std_hat.
    """
    rms_distortion = float(np.sqrt(np.sum(distortion[1:] ** 2)))
    return rms_distortion / (sigma_Y * np.sqrt(dt))


def solve_ncoh_star(
    mu: float,
    N_D: float,
    S: float,
    lam: float,
    alpha: float,
    sigma_Y: float,
    max_ncoh: int = 1000,
) -> int:
    """Find the largest coherence length n_coh* that keeps detectability below threshold.

    The detectability threshold is:
        threshold = S / (sigma_Y * sqrt(1 / (mu * N_D)))
                  = S * sqrt(mu * N_D) / sigma_Y

    The internal time-step is set to dt = 1/(mu * N_D) so that
        detectability_proxy(..., dt=1/(mu*N_D)) = sqrt(mu*N_D)/sigma_Y * sqrt(sum d^2)

    and the comparison reduces to  sqrt(sum d[1:]^2) <= S / sqrt(mu * N_D) * ... 
    Equivalently the search finds the largest n_coh with proxy <= threshold.

    Parameters
    ----------
    mu : float
        Participation rate.
    N_D : float
        Number of trading days (or bins) in the detection window.
    S : float
        Detectability budget.
    lam : float
        Impact coefficient lambda.
    alpha : float
        Cross-term coefficient alpha.
    sigma_Y : float
        Benchmark return volatility.
    max_ncoh : int, optional
        Upper bound on search (default 1000).

    Returns
    -------
    int
        Largest n_coh in 1..max_ncoh such that detectability_proxy <= threshold.
        Returns 1 if even n_coh=1 exceeds the threshold.
    """
    dt = 1.0 / (mu * N_D)
    threshold = S / (sigma_Y * np.sqrt(1.0 / (mu * N_D)))

    best = 1
    for n_coh in range(1, max_ncoh + 1):
        profile = compute_distortion_profile(lam, alpha, n_coh, max_lag=n_coh)
        proxy = detectability_proxy(profile, sigma_Y, dt)
        if proxy <= threshold:
            best = n_coh
        else:
            break
    return best


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS log-log regression: log(y) = slope * log(x) + intercept.

    Parameters
    ----------
    x : np.ndarray
        Positive independent variable values.
    y : np.ndarray
        Positive dependent variable values.

    Returns
    -------
    dict with keys:
        'slope'     : float, power-law exponent
        'intercept' : float, log of prefactor
        'r2'        : float, coefficient of determination
    """
    log_x = np.log(np.asarray(x, dtype=np.float64))
    log_y = np.log(np.asarray(y, dtype=np.float64))

    X = np.column_stack([log_x, np.ones_like(log_x)])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_y, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    y_hat = slope * log_x + intercept
    ss_res = float(np.sum((log_y - y_hat) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return {"slope": slope, "intercept": intercept, "r2": float(r2)}


def theoretical_ncoh_star(
    mu: float, S: float, lam: float, alpha: float
) -> float:
    """Analytical approximation for the optimal coherence length n_coh*.

    Derived from balancing the distortion budget:
        C = (S / (lam + 2*alpha))^(2/3)
        n_coh_star = C * mu^(-1/3)

    Parameters
    ----------
    mu : float
        Participation rate.
    S : float
        Detectability budget.
    lam : float
        Impact coefficient lambda.
    alpha : float
        Cross-term coefficient alpha.

    Returns
    -------
    float
        Theoretical optimal coherence length.
    """
    C = (S / (lam + 2.0 * alpha)) ** (2.0 / 3.0)
    return float(C * mu ** (-1.0 / 3.0))

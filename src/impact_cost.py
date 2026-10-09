"""
Impact-cost estimation and whiteness diagnostics for market microstructure.

The core object of study is the temporary-impact cost C_H, which measures
how much a participant pays (on average) due to the adverse price movement
triggered by their own order flow over the next H_C steps.
"""

import numpy as np

from src.covariance import estimate_autocovariance


def compute_future_sum_kernel(H_C: int) -> np.ndarray:
    """Build the causal future-sum convolution kernel of length H_C+1.

    The kernel is [0, 1, 1, ..., 1] (H_C ones preceded by a single zero).
    Convolving a signal j with this kernel and reading position t yields
    sum_{h=1}^{H_C} j[t+h] when applied in the non-causal direction.

    Parameters
    ----------
    H_C : int
        Forecast horizon; number of future steps to accumulate.

    Returns
    -------
    np.ndarray
        Float64 array of shape (H_C+1,).
    """
    kernel = np.ones(H_C + 1, dtype=np.float64)
    kernel[0] = 0.0
    return kernel


def estimate_impact_cost(q: np.ndarray, j: np.ndarray, H_C: int) -> float:
    """Estimate the H_C-step impact cost C_H.

    Computes:
        C_H = -(1/T) * sum_{t=0}^{T-H_C-1} q[t] * sum_{h=1}^{H_C} j[t+h]

    The future sum is evaluated via cumulative summation for O(T) cost.
    Only the T-H_C terms with a complete H_C-step lookahead are included.

    Parameters
    ----------
    q : np.ndarray
        Participant flow (order imbalance) series of length T.
    j : np.ndarray
        Impact increment series of length T (same length as q).
    H_C : int
        Forecast horizon over which future impact is accumulated.

    Returns
    -------
    float
        Estimated impact cost.
    """
    T = len(q)
    j_cum = np.cumsum(j)
    # future_sum[t] = j[t+1] + ... + j[t+H_C] = j_cum[t+H_C] - j_cum[t]
    T_valid = T - H_C
    future_sums = j_cum[H_C:T] - j_cum[:T_valid]
    return float(-np.sum(q[:T_valid] * future_sums) / T)


def theoretical_proportional_score(u0: float, z1: float, z2: float) -> float:
    """Compute the theoretical proportional impact-cost score K(u0, z1, z2).

    Defined as:
        K = (1 - u0^2) * (z2 - z1) / ((1 - z1*u0) * (1 - z2*u0))

    This is the analytical expectation of the normalised impact cost for a
    participant whose flow is driven by the biexponential filter with poles
    z1 < z2 and whose impact follows the all-pass filter with pole u0.

    Parameters
    ----------
    u0 : float
        All-pass filter pole.
    z1 : float
        Biexponential fast-decay pole.
    z2 : float
        Biexponential slow-decay pole.

    Returns
    -------
    float
        Score K.
    """
    return float((1.0 - u0 ** 2) * (z2 - z1) / ((1.0 - z1 * u0) * (1.0 - z2 * u0)))


def estimate_impact_cost_batched(
    q: np.ndarray,
    j: np.ndarray,
    H_C: int,
    n_batches: int,
) -> tuple:
    """Estimate impact cost with uncertainty via non-overlapping batch means.

    Splits q and j into n_batches equal-length segments, computes
    estimate_impact_cost on each, and returns the grand mean and
    standard error across batches.

    Parameters
    ----------
    q : np.ndarray
        Participant flow series of length T.
    j : np.ndarray
        Impact increment series of length T.
    H_C : int
        Forecast horizon.
    n_batches : int
        Number of non-overlapping batches.

    Returns
    -------
    tuple
        (mean, std_err) where std_err = std(batch_costs) / sqrt(n_batches).
    """
    T = len(q)
    batch_size = T // n_batches
    costs = []
    for k in range(n_batches):
        start = k * batch_size
        end = start + batch_size
        costs.append(estimate_impact_cost(q[start:end], j[start:end], H_C))
    costs = np.array(costs, dtype=np.float64)
    mean = float(costs.mean())
    std_err = float(costs.std(ddof=1) / np.sqrt(n_batches))
    return mean, std_err


def verify_whiteness_of_impact(j: np.ndarray, max_lag: int) -> dict:
    """Assess whether impact increments j_t are serially uncorrelated.

    An information-neutral (all-pass-filtered) impact process should be
    white: autocovariance zero at all non-zero lags.

    Parameters
    ----------
    j : np.ndarray
        Impact increment series of length T.
    max_lag : int
        Maximum lag passed to estimate_autocovariance.

    Returns
    -------
    dict with keys:
        'acov'            : dict {lag: autocovariance} from estimate_autocovariance
        'max_off_zero_abs': float, max |acov[t]| for t != 0
        'lag0_var'        : float, acov[0] (variance of j)
    """
    acov = estimate_autocovariance(j, max_lag)
    off_zero = [abs(v) for t, v in acov.items() if t != 0]
    return {
        "acov": acov,
        "max_off_zero_abs": float(max(off_zero)) if off_zero else 0.0,
        "lag0_var": float(acov[0]),
    }

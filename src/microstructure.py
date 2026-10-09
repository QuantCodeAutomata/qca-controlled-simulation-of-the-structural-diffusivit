"""
Empirical microstructure utilities for synthetic high-frequency data analysis.

Provides generators for synthetic HF order-flow and price data (used as
fallback when real tick data is unavailable), binning utilities, and
simple neutrality classifiers.
"""

import numpy as np
import pandas as pd

from src.simulation import simulate_impact_increments


# Default burn-in when not explicitly parameterised
_DEFAULT_BURNIN = 100


def generate_synthetic_hf_data(
    T: int,
    sigma_X: float,
    lambda_X: float,
    N_per_bin: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic high-frequency binned return data.

    Each bin return is drawn from the Kyle (1985) model:
        x_t | N_t  ~  N(0, lambda_X^2 * N_t)

    where N_t ~ Poisson(N_per_bin) is the random trade count per bin.

    Parameters
    ----------
    T : int
        Number of time bins.
    sigma_X : float
        Realised return volatility used to set the constant bid-ask spread
        (spread = 2 * lambda_X; sigma_X stored as a metadata column).
    lambda_X : float
        Kyle lambda — per-trade price impact volatility.
    N_per_bin : int
        Expected number of trades per bin.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    pd.DataFrame
        Columns: ['bin', 'x_t', 'N_t', 'spread'].
    """
    N_t = rng.poisson(N_per_bin, T).astype(np.int64)
    N_t = np.maximum(N_t, 1)
    x_t = rng.normal(0.0, lambda_X * np.sqrt(N_t.astype(np.float64))).astype(np.float64)
    spread = np.full(T, 2.0 * lambda_X, dtype=np.float64)

    return pd.DataFrame(
        {"bin": np.arange(T), "x_t": x_t, "N_t": N_t, "spread": spread}
    )


def estimate_lambda_X(df: pd.DataFrame) -> float:
    """Estimate the Kyle lambda from binned return and trade-count data.

    Uses the moment estimator:
        lambda_X = sqrt(sum(x_t^2) / sum(N_t))

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'x_t' (bin returns) and 'N_t' (trade counts).

    Returns
    -------
    float
        Estimated Kyle lambda.
    """
    return float(np.sqrt(np.sum(df["x_t"].values ** 2) / np.sum(df["N_t"].values)))


def build_bins(
    prices: np.ndarray,
    timestamps: np.ndarray,
    bin_size_sec: int,
) -> pd.DataFrame:
    """Aggregate a tick-level price series into fixed-width calendar-time bins.

    Parameters
    ----------
    prices : np.ndarray
        Tick-level price (or log-price) array of length N.
    timestamps : np.ndarray
        Unix-epoch seconds corresponding to each tick, length N.
    bin_size_sec : int
        Width of each bin in seconds.

    Returns
    -------
    pd.DataFrame
        Columns: ['bin_start', 'return', 'n_trades'].
        Each row represents one bin; 'return' is the last-minus-first
        price in the bin (or NaN if the bin is empty).
    """
    t_min = int(timestamps[0])
    t_max = int(timestamps[-1])
    bin_edges = np.arange(t_min, t_max + bin_size_sec, bin_size_sec)

    records = []
    for b_start in bin_edges[:-1]:
        b_end = b_start + bin_size_sec
        mask = (timestamps >= b_start) & (timestamps < b_end)
        n_trades = int(mask.sum())
        if n_trades > 0:
            ret = float(prices[mask][-1] - prices[mask][0])
        else:
            ret = float("nan")
        records.append({"bin_start": b_start, "return": ret, "n_trades": n_trades})

    return pd.DataFrame(records, columns=["bin_start", "return", "n_trades"])


def generate_synthetic_execution_data(
    T: int,
    mu: float,
    sigma_Y: float,
    u0: float,
    c: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a synthetic order-execution dataset with impact and flow.

    The data-generating process is:
        eta_t   ~ i.i.d. N(0, 1)          (impact innovations)
        j_t     = c * (U * eta)[t]         (all-pass filtered impact)
        y_t     ~ i.i.d. N(0, sigma_Y^2)  (counterfactual returns)
        x_t     = y_t + j_t               (realised returns)
        q_t     ~ i.i.d. N(0, mu^2)       (participant flow proxy)

    Parameters
    ----------
    T : int
        Number of usable time steps.
    mu : float
        Standard deviation of the participant flow (participation scale).
    sigma_Y : float
        Counterfactual return volatility.
    u0 : float
        All-pass filter pole parameter.
    c : float
        Impact scale coefficient.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    pd.DataFrame
        Columns: ['t', 'y_t', 'j_t', 'x_t', 'q_t', 'mu'].
    """
    B = _DEFAULT_BURNIN
    eta = rng.standard_normal(T + B)
    j = simulate_impact_increments(eta, u0, c, B)
    y = rng.normal(0.0, sigma_Y, T).astype(np.float64)
    x = y + j
    q = rng.normal(0.0, mu, T).astype(np.float64)

    return pd.DataFrame(
        {
            "t": np.arange(T),
            "y_t": y,
            "j_t": j,
            "x_t": x,
            "q_t": q,
            "mu": np.full(T, mu, dtype=np.float64),
        }
    )


def classify_neutrality(
    df: pd.DataFrame,
    alpha_threshold: float = 0.1,
) -> pd.Series:
    """Classify each bin as information-neutral based on participant flow magnitude.

    A bin is neutral (1) if |q_t| <= alpha_threshold, non-neutral (0) otherwise.
    Neutral bins are bins where the participant is effectively inactive, so
    the return is driven purely by the counterfactual (public information).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'q_t' column (participant flow).
    alpha_threshold : float, optional
        Flow magnitude threshold for neutrality (default 0.1).

    Returns
    -------
    pd.Series
        Boolean Series (int 0/1) of the same length as df, indexed like df.
    """
    return (df["q_t"].abs() <= alpha_threshold).astype(int)

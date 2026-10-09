"""
Filter construction and application utilities for market microstructure models.

Provides causal all-pass filters (for information-neutral impact whitening),
biexponential participant-flow filters, and supporting diagnostics.
"""

import numpy as np
from scipy.signal import lfilter


def build_allpass_filter(u0: float, H: int) -> np.ndarray:
    """Build causal all-pass filter coefficients truncated to length H+1.

    Coefficients are defined by:
        U[0]   = u0
        U[h]   = -(1 - u0**2) * u0**(h-1)   for h >= 1

    The infinite-horizon filter satisfies the all-pass identity:
        sum_{h>=0} U[h] * U[h+t] = delta[t]

    Parameters
    ----------
    u0 : float
        Pole parameter, strictly between 0 and 1.
    H : int
        Truncation length; the returned array has shape (H+1,).

    Returns
    -------
    np.ndarray
        Float64 array of shape (H+1,) containing U[0], ..., U[H].

    Raises
    ------
    ValueError
        If u0 is not in the open interval (0, 1).
    """
    if not (0.0 < u0 < 1.0):
        raise ValueError(f"u0 must satisfy 0 < u0 < 1, got {u0}")

    h = np.arange(H + 1, dtype=np.float64)
    U = np.where(h == 0, u0, -(1.0 - u0 ** 2) * u0 ** (h - 1))
    return U


def build_biexponential_filter(z1: float, z2: float, H: int) -> np.ndarray:
    """Build biexponential (leader-minus-follower) filter coefficients.

    Coefficients are defined by:
        L[h] = (1 - z1) * z1**h - (1 - z2) * z2**h   for h >= 0

    Parameters
    ----------
    z1 : float
        Fast-decay pole, strictly between 0 and z2.
    z2 : float
        Slow-decay pole, strictly between z1 and 1.
    H : int
        Truncation length; the returned array has shape (H+1,).

    Returns
    -------
    np.ndarray
        Float64 array of shape (H+1,).

    Raises
    ------
    ValueError
        If the constraint 0 < z1 < z2 < 1 is violated.
    """
    if not (0.0 < z1 < z2 < 1.0):
        raise ValueError(
            f"Parameters must satisfy 0 < z1 < z2 < 1, got z1={z1}, z2={z2}"
        )

    h = np.arange(H + 1, dtype=np.float64)
    L = (1.0 - z1) * z1 ** h - (1.0 - z2) * z2 ** h
    return L


def build_lti_filter(coeffs: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Apply a causal FIR filter to a signal via scipy.signal.lfilter.

    Computes the output:
        y[t] = sum_{k=0}^{len(coeffs)-1} coeffs[k] * signal[t-k]

    This is equivalent to full convolution trimmed to len(signal), but uses
    the numerically efficient direct-form-II implementation in lfilter.

    Parameters
    ----------
    coeffs : np.ndarray
        FIR filter coefficients (b-polynomial), shape (H+1,).
    signal : np.ndarray
        Input signal, shape (T,).

    Returns
    -------
    np.ndarray
        Filtered signal of shape (T,), same length as *signal*.
    """
    return lfilter(coeffs.astype(np.float64), [1.0], signal.astype(np.float64))


def verify_allpass_property(u0: float, H: int, L: int) -> np.ndarray:
    """Compute the autocorrelation of truncated all-pass filter coefficients.

    Evaluates:
        C[t] = sum_{h=0}^{H-t} U[h] * U[h+t]   for t = 0, 1, ..., L

    For the true infinite-horizon filter, C[0] = 1 and C[t] = 0 for t > 0.
    Truncation introduces small deviations that vanish as H grows.

    Parameters
    ----------
    u0 : float
        Pole parameter, strictly between 0 and 1.
    H : int
        Filter truncation length passed to build_allpass_filter.
    L : int
        Maximum lag to evaluate.

    Returns
    -------
    np.ndarray
        Float64 array of shape (L+1,) with C[0], ..., C[L].
    """
    U = build_allpass_filter(u0, H)
    # np.correlate(U, U, 'full')[H + t] = sum_h U[h] * U[h + t]
    full_corr = np.correlate(U, U, mode="full")
    return full_corr[H: H + L + 1].astype(np.float64)


def compute_filter_truncation_length(base: float, tol: float = 1e-10) -> int:
    """Return the smallest H such that base**H < tol.

    Useful for determining a safe truncation horizon so that filter tail
    energy below *tol* is negligible.

    Parameters
    ----------
    base : float
        Decay rate, strictly between 0 and 1.
    tol : float, optional
        Tolerance threshold (default 1e-10).

    Returns
    -------
    int
        Smallest non-negative integer H with base**H < tol.

    Raises
    ------
    ValueError
        If base is not in the open interval (0, 1).
    """
    if not (0.0 < base < 1.0):
        raise ValueError(f"base must satisfy 0 < base < 1, got {base}")

    if tol >= 1.0:
        return 0

    return int(np.ceil(np.log(tol) / np.log(base)))

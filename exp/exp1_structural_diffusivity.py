"""
Experiment 1: Controlled Simulation of the Structural Diffusivity Identity
and Information-Neutral Impact Whitening.

Verifies numerically that:
  (i)  The all-pass filter preserves the white-noise spectrum of the impact.
  (ii) The structural identity R_yj(τ) + R_jy(τ) + R_jj(τ) = dt·(σ_X²−σ_Y²)·δ(τ)
       holds across multiple Monte Carlo replications and u0 values.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.filters import verify_allpass_property, compute_filter_truncation_length
from src.simulation import simulate_full_system
from src.covariance import (
    estimate_autocovariance,
    estimate_crosscovariance,
    verify_structural_identity,
    compute_sample_variance,
)
from src.impact_cost import verify_whiteness_of_impact
from src.diagnostics import ljung_box_test, diffusivity_check


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _set_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _write_results_md(save_dir: str, header: str, summary: dict) -> None:
    path = os.path.join(save_dir, "RESULTS.md")
    with open(path, "a") as fh:
        fh.write(f"\n## {header}\n\n")
        for key, block in summary.items():
            fh.write(f"### {key}\n")
            if isinstance(block, dict):
                for k, v in block.items():
                    if isinstance(v, dict):
                        fh.write(f"  **{k}**\n")
                        for kk, vv in v.items():
                            fh.write(f"    - {kk}: {vv}\n")
                    else:
                        fh.write(f"  - {k}: {v}\n")
            else:
                fh.write(f"  {block}\n")
            fh.write("\n")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _run_single_rep(T, B, u0, c, sigma_Y, dt, max_lag, seed1, seed2):
    """One Monte Carlo replication: simulate, estimate covariances, verify identity."""
    sim = simulate_full_system(
        T=T, B=B, sigma_Y=sigma_Y, u0=u0, c=c, dt=dt,
        seed_eta=seed1, seed_xi=seed2,
    )
    y, j, x = sim["y"], sim["j"], sim["x"]

    lags = list(range(-max_lag, max_lag + 1))

    # ---- covariances (lags -max_lag … +max_lag) ----
    R_yy = estimate_autocovariance(y, max_lag)
    R_xx = estimate_autocovariance(x, max_lag)
    R_jj = estimate_autocovariance(j, max_lag)
    R_jy = estimate_crosscovariance(j, y, max_lag)
    R_yj = estimate_crosscovariance(y, j, max_lag)

    # ---- structural identity ----
    sigma_X2 = compute_sample_variance(x)
    sigma_Y2 = compute_sample_variance(y)

    id_result = verify_structural_identity(
        R_yj=R_yj, R_jy=R_jy, R_jj=R_jj,
        dt=dt, sigma_X2=sigma_X2, sigma_Y2=sigma_Y2, max_lag=max_lag,
    )

    # ---- whiteness of j ----
    whiteness = verify_whiteness_of_impact(j, max_lag)

    # ---- Ljung-Box on j (pass a list of lags) ----
    lb_lags = list(range(1, min(21, max_lag + 1)))
    lb = ljung_box_test(j, lb_lags)
    lb_pvalue = float(lb["lb_pvalue"].iloc[-1])

    # ---- diffusivity ----
    diff_y = diffusivity_check(y, max_lag=max_lag, dt=dt, sigma2_theoretical=sigma_Y ** 2)
    diff_x = diffusivity_check(x, max_lag=max_lag, dt=dt, sigma2_theoretical=sigma_X2)

    # ---- off-zero max for y, x (dict keys are integers) ----
    R_yy_off = float(max(abs(R_yy[t]) for t in R_yy if t != 0))
    R_xx_off = float(max(abs(R_xx[t]) for t in R_xx if t != 0))

    metrics = {
        "lag0_var_x":                  float(R_xx[0]),
        "lag0_var_y":                  float(R_yy[0]),
        "lag0_var_j":                  float(R_jj[0]),
        "max_off_zero_Rjj":            float(whiteness["max_off_zero_abs"]),
        "max_off_zero_Ryy":            R_yy_off,
        "max_off_zero_Rxx":            R_xx_off,
        "identity_max_off_zero_error": float(id_result["max_abs_off_zero_error"]),
        "identity_lag0_error":         float(id_result["lag0_error"]),
        "lb_pvalue":                   lb_pvalue,
        "y_is_diffusive":              int(bool(diff_y["is_diffusive"])),
        "x_is_diffusive":              int(bool(diff_x["is_diffusive"])),
    }

    # LHS of structural identity as ndarray
    S_lhs = np.array([R_yj[t] + R_jy[t] + R_jj[t] for t in lags])
    R_jj_arr = np.array([R_jj[t] for t in lags])

    return metrics, R_jj_arr, S_lhs, sigma_X2, sigma_Y2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_rjj(lags, Rjj_mean, u0, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, Rjj_mean, width=0.6, color="steelblue", alpha=0.75, label=r"$\hat{R}_{jj}(\tau)$")
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xlabel(r"Lag $\tau$", fontsize=12)
    ax.set_ylabel("Autocovariance", fontsize=12)
    ax.set_title(
        rf"Autocovariance of Impact $j_t$  ($u_0={u0}$)"
        "\n" r"(Expected: approximate $\delta$-function at $\tau=0$)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.tight_layout()
    path = os.path.join(save_dir, f"exp1_Rjj_u0_{u0:.1f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_identity(lags, S_mean, sigma_X2_avg, sigma_Y2_avg, dt, max_lag, u0, save_dir):
    rhs = np.zeros(len(lags))
    rhs[max_lag] = dt * (sigma_X2_avg - sigma_Y2_avg)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, S_mean, color="steelblue", linewidth=1.5,
            label=r"LHS: $R_{yj}(\tau)+R_{jy}(\tau)+R_{jj}(\tau)$")
    ax.plot(lags, rhs, "r--", linewidth=2,
            label=r"RHS: $dt\,(\sigma_X^2-\sigma_Y^2)\,\delta(\tau)$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"Lag $\tau$", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        rf"Structural Identity Verification  ($u_0={u0}$)"
        "\n" r"(LHS should coincide with the RHS $\delta$-spike)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.tight_layout()
    path = os.path.join(save_dir, f"exp1_identity_u0_{u0:.1f}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_metric_bars(u0_values, metric_key, mc_summaries, save_dir):
    """Bar chart of a scalar metric (mean ± std) across u0 values."""
    means = [mc_summaries[u0]["mean"][metric_key] for u0 in u0_values]
    stds  = [mc_summaries[u0]["std"][metric_key]  for u0 in u0_values]
    labels = [str(u0) for u0 in u0_values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xlabel(r"$u_0$", fontsize=12)
    ax.set_ylabel(metric_key.replace("_", " "), fontsize=11)
    ax.set_title(f"MC average of {metric_key}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(save_dir, f"exp1_{metric_key}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp1(save_dir: str = "results") -> dict:
    """
    Experiment 1 – Structural diffusivity identity and impact whitening.

    Parameters
    ----------
    save_dir : str
        Directory for PNG plots and RESULTS.md.

    Returns
    -------
    dict
        Nested summary keyed by u0 value.
    """
    os.makedirs(save_dir, exist_ok=True)
    _set_style()

    # ---- parameters ----
    T          = 50_000
    B          = 1_000
    dt         = 1.0
    sigma_Y    = 1.0
    u0_values  = [0.2, 0.5, 0.8]
    c          = 0.5
    max_lag    = 50
    M          = 5
    seed_base  = 0

    lags = np.arange(-max_lag, max_lag + 1)

    full_summary   = {}
    mc_summaries   = {}   # per u0: {mean, std} of scalar metrics

    for u0_idx, u0 in enumerate(u0_values):
        print(f"\n{'='*60}")
        print(f"  u0 = {u0}")
        print(f"{'='*60}")

        # ---- (a) verify all-pass property ----
        H_ap = compute_filter_truncation_length(u0)
        ap_acov = verify_allpass_property(u0, H=H_ap, L=max_lag)
        allpass_ok = bool(abs(ap_acov[0] - 1.0) < 0.01 and np.max(np.abs(ap_acov[1:])) < 0.05)
        print(f"  All-pass property verified: {allpass_ok} (lag0={ap_acov[0]:.4f}, max_off_zero={np.max(np.abs(ap_acov[1:])):.4e})")

        # ---- (b) Monte Carlo loop ----
        rep_metrics   = []
        Rjj_reps      = []
        S_lhs_reps    = []
        sigma_X2_reps = []
        sigma_Y2_reps = []

        for rep in range(M):
            seed1 = seed_base * 1_000_000 + u0_idx * 10_000 + rep
            seed2 = seed1 + 500_000

            metrics, R_jj, S_lhs, sx2, sy2 = _run_single_rep(
                T=T, B=B, u0=u0, c=c, sigma_Y=sigma_Y, dt=dt,
                max_lag=max_lag, seed1=seed1, seed2=seed2,
            )
            rep_metrics.append(metrics)
            Rjj_reps.append(R_jj)
            S_lhs_reps.append(S_lhs)
            sigma_X2_reps.append(sx2)
            sigma_Y2_reps.append(sy2)

            print(
                f"    rep {rep:2d} | identity_lag0_err={metrics['identity_lag0_error']:.4e}"
                f" | max_off_zero_Rjj={metrics['max_off_zero_Rjj']:.4e}"
                f" | LB p-val={metrics['lb_pvalue']:.3f}"
            )

        # ---- aggregate metrics (manual mean/std over reps) ----
        all_keys = list(rep_metrics[0].keys())
        mean_m = {k: float(np.mean([r[k] for r in rep_metrics])) for k in all_keys}
        std_m  = {k: float(np.std( [r[k] for r in rep_metrics], ddof=1)) for k in all_keys}

        mc_summaries[u0] = {"mean": mean_m, "std": std_m}

        print(f"\n  MC mean identity_lag0_error     : {mean_m['identity_lag0_error']:.4e}  ± {std_m['identity_lag0_error']:.4e}")
        print(f"  MC mean identity_off_zero_error : {mean_m['identity_max_off_zero_error']:.4e}  ± {std_m['identity_max_off_zero_error']:.4e}")
        print(f"  MC mean max_off_zero_Rjj        : {mean_m['max_off_zero_Rjj']:.4e}  ± {std_m['max_off_zero_Rjj']:.4e}")
        print(f"  MC mean lb_pvalue               : {mean_m['lb_pvalue']:.3f}")

        # ---- (c) plots ----
        Rjj_mean = np.mean(Rjj_reps,   axis=0)
        S_mean   = np.mean(S_lhs_reps, axis=0)
        sX2_avg  = float(np.mean(sigma_X2_reps))
        sY2_avg  = float(np.mean(sigma_Y2_reps))

        p1 = _plot_rjj(lags, Rjj_mean, u0, save_dir)
        p2 = _plot_identity(lags, S_mean, sX2_avg, sY2_avg, dt, max_lag, u0, save_dir)
        print(f"  Saved: {p1}")
        print(f"  Saved: {p2}")

        full_summary[f"u0_{u0}"] = {
            "allpass_ok": allpass_ok,
            "mean_metrics": mean_m,
            "std_metrics":  std_m,
        }

    # ---- cross-u0 metric bar charts ----
    for metric in ("identity_max_off_zero_error", "max_off_zero_Rjj", "lb_pvalue"):
        p = _plot_metric_bars(u0_values, metric, mc_summaries, save_dir)
        print(f"Saved: {p}")

    # ---- persist results ----
    _write_results_md(save_dir, "EXP1 – Structural Diffusivity Identity", full_summary)
    print(f"\nResults appended to {os.path.join(save_dir, 'RESULTS.md')}")

    return full_summary


if __name__ == "__main__":
    run_exp1()

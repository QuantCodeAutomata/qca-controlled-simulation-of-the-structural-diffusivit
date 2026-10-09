"""
Experiment 4: Controlled Simulation of the Detectability and
Coherence-Length Bound.

Verifies numerically that the detectability coherence length satisfies

    n_coh*(μ) ∝ μ^{−1/3}

by solving the optimisation problem embedded in solve_ncoh_star and
comparing with the closed-form theoretical prediction.

Three plots are produced:
  1. n_coh*(μ) vs μ (log-log) with theoretical overlay  → exp4_ncoh_vs_mu.png
  2. Power-law fit with annotated slope                  → exp4_power_law_fit.png
  3. Sensitivity to S: n_coh*(μ) for S ∈ {0.005,0.01,0.02} → exp4_sensitivity_S.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coherence import (
    compute_distortion_profile,
    detectability_proxy,
    solve_ncoh_star,
    fit_power_law,
    theoretical_ncoh_star,
)


# ---------------------------------------------------------------------------
# Style helper
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

def _write_results_md(save_dir: str, header: str, content: dict) -> None:
    path = os.path.join(save_dir, "RESULTS.md")
    with open(path, "a") as fh:
        fh.write(f"\n## {header}\n\n")
        for section, data in content.items():
            fh.write(f"### {section}\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    fh.write(f"  - {k}: {v}\n")
            elif isinstance(data, (list, np.ndarray)):
                fh.write("  " + ", ".join(f"{x:.5g}" for x in data) + "\n")
            else:
                fh.write(f"  {data}\n")
            fh.write("\n")


# ---------------------------------------------------------------------------
# Core solve loop
# ---------------------------------------------------------------------------

def _compute_ncoh_grid(mu_grid, S, lam, alpha, sigma_Y, N_D, label=""):
    """Return (n_coh_solved, n_coh_theoretical) arrays for a mu_grid."""
    n_solved      = np.empty(len(mu_grid))
    n_theoretical = np.empty(len(mu_grid))

    for i, mu in enumerate(mu_grid):
        # solve_ncoh_star(mu, N_D, S, lam, alpha, sigma_Y, max_ncoh=1000)
        n_solved[i] = solve_ncoh_star(
            mu=mu, N_D=N_D, S=S, lam=lam, alpha=alpha, sigma_Y=sigma_Y,
        )
        n_theoretical[i] = theoretical_ncoh_star(
            mu=mu, S=S, lam=lam, alpha=alpha,
        )

    if label:
        print(f"  [{label}]  n_coh* range: [{n_solved.min():.3f}, {n_solved.max():.3f}]")

    return n_solved, n_theoretical


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_ncoh_vs_mu(mu_grid, n_solved, n_theoretical, save_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(mu_grid, n_solved, "o-", color="steelblue", linewidth=1.8,
              markersize=5, label=r"$n_{\mathrm{coh}}^*$ (solved)")
    ax.loglog(mu_grid, n_theoretical, "r--", linewidth=1.5,
              label=r"Theoretical: $C\cdot\mu^{-1/3}$")

    # Annotate reference slope
    mu_lo, mu_hi = mu_grid[[5, -5]]
    y_lo = n_solved[5]
    y_hi = y_lo * (mu_hi / mu_lo) ** (-1 / 3)
    ax.loglog([mu_lo, mu_hi], [y_lo, y_hi], "k:", linewidth=1.2,
              label=r"slope $= -1/3$")

    ax.set_xlabel(r"Impact rate $\mu$", fontsize=12)
    ax.set_ylabel(r"$n_{\mathrm{coh}}^*(\mu)$", fontsize=12)
    ax.set_title(
        r"Detectability Coherence Length $n_{\mathrm{coh}}^*$ vs $\mu$",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    fig.tight_layout()
    path = os.path.join(save_dir, "exp4_ncoh_vs_mu.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_power_law_fit(mu_grid, n_solved, fitted_slope, fitted_intercept, save_dir):
    log_mu = np.log(mu_grid)
    log_n  = np.log(n_solved)
    log_n_fit = fitted_intercept + fitted_slope * log_mu

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log-log data + fitted line
    ax = axes[0]
    # Theoretical line anchored to data mean (same slope -1/3)
    theoretical_intercept = float(np.mean(log_n - (-1 / 3) * log_mu))
    log_n_theory = -1.0 / 3.0 * log_mu + theoretical_intercept

    ax.scatter(log_mu, log_n, color="steelblue", s=30, zorder=3,
               label=r"$\log n_{\mathrm{coh}}^*$")
    ax.plot(log_mu, log_n_fit, "r-", linewidth=1.8,
            label=rf"fit: slope $= {fitted_slope:.4f}$")
    ax.plot(log_mu, log_n_theory, color="grey", linestyle="--", linewidth=1.2,
            label=r"theoretical slope $-\frac{1}{3}$")
    ax.set_xlabel(r"$\log(\mu)$", fontsize=12)
    ax.set_ylabel(r"$\log(n_{\mathrm{coh}}^*)$", fontsize=12)
    ax.set_title("Log-log Regression", fontsize=11)
    ax.legend(fontsize=10)

    # Residuals
    ax2 = axes[1]
    residuals = log_n - log_n_fit
    ax2.scatter(log_mu, residuals, color="tomato", s=25, zorder=3)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel(r"$\log(\mu)$", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_title(
        rf"Fit residuals  (slope $= {fitted_slope:.4f}$,"
        rf" expected $-\frac{{1}}{{3}} \approx {-1/3:.4f}$)",
        fontsize=10,
    )

    fig.suptitle("Power-Law Fit: $\\log n_{\\mathrm{coh}}^* = a + \\hat{\\beta}\\log\\mu$",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, "exp4_power_law_fit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_sensitivity_S(mu_grid, S_values, n_solved_dict, save_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "tomato", "seagreen"]

    for col, S_val in zip(colors, S_values):
        n_vals = n_solved_dict[S_val]
        ax.loglog(mu_grid, n_vals, "o-", color=col, linewidth=1.6,
                  markersize=5, label=rf"$S = {S_val}$")

    # Reference slope
    mu_lo, mu_hi = mu_grid[[3, -3]]
    ref_n_lo = n_solved_dict[S_values[1]][3]
    ref_n_hi = ref_n_lo * (mu_hi / mu_lo) ** (-1 / 3)
    ax.loglog([mu_lo, mu_hi], [ref_n_lo, ref_n_hi], "k:", linewidth=1.2,
              label=r"slope $-\frac{1}{3}$")

    ax.set_xlabel(r"Impact rate $\mu$", fontsize=12)
    ax.set_ylabel(r"$n_{\mathrm{coh}}^*(\mu)$", fontsize=12)
    ax.set_title(
        r"Sensitivity to $S$: Coherence Length $n_{\mathrm{coh}}^*$ vs $\mu$",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, "exp4_sensitivity_S.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp4(save_dir: str = "results") -> dict:
    """
    Experiment 4 – Detectability and coherence-length bound.

    Parameters
    ----------
    save_dir : str
        Directory for PNG plots and RESULTS.md.

    Returns
    -------
    dict
        ``{'fitted_slope', 'theoretical_slope', 'ncoh_grid', 'examples', 'plots'}``
    """
    os.makedirs(save_dir, exist_ok=True)
    _set_style()

    # ---- parameters ----
    mu_grid  = np.logspace(-3, np.log10(0.2), 30)
    N_D      = 20_000
    S        = 0.01
    lam      = 0.5
    alpha    = 0.3
    sigma_Y  = 1.0

    # ---- (1-2) solve n_coh* across mu_grid ----
    print("\nExp 4: solving n_coh*(μ) for baseline parameters …")
    n_solved, n_theoretical = _compute_ncoh_grid(
        mu_grid=mu_grid, S=S, lam=lam, alpha=alpha,
        sigma_Y=sigma_Y, N_D=N_D, label="baseline",
    )

    # ---- (3) normalise theoretical by its value at mu_grid[0] ----
    # (preserves slope; scale is anchored to the solved curve)
    if n_theoretical[0] > 0 and n_solved[0] > 0:
        scale_factor = n_solved[0] / n_theoretical[0]
    else:
        scale_factor = 1.0
    n_theoretical_scaled = n_theoretical * scale_factor

    # ---- (4) power-law fit ----
    fit_result = fit_power_law(x=mu_grid, y=n_solved)
    fitted_slope     = float(fit_result["slope"])
    fitted_intercept = float(fit_result["intercept"])
    theoretical_slope = -1.0 / 3.0

    print(f"  Fitted slope:      {fitted_slope:.5f}")
    print(f"  Theoretical slope: {theoretical_slope:.5f}")
    print(f"  Discrepancy:       {abs(fitted_slope - theoretical_slope):.5f}")

    # ---- (5) spot checks ----
    mu_check = [0.01, 0.10]
    examples = {}
    for mu_c in mu_check:
        n_c = solve_ncoh_star(
            mu=mu_c, N_D=N_D, S=S, lam=lam, alpha=alpha, sigma_Y=sigma_Y,
        )
        examples[mu_c] = float(n_c)
        print(f"  n_coh*(μ={mu_c:.2f}) = {n_c:.3f}")

    # ---- (6) plots ----
    plots = []

    p = _plot_ncoh_vs_mu(mu_grid, n_solved, n_theoretical_scaled, save_dir)
    plots.append(p)
    print(f"Saved: {p}")

    p = _plot_power_law_fit(mu_grid, n_solved, fitted_slope, fitted_intercept, save_dir)
    plots.append(p)
    print(f"Saved: {p}")

    # ---- (7) sensitivity to S ----
    S_values = [0.005, 0.01, 0.02]
    n_solved_dict = {}

    print("\nExp 4: sensitivity sweep over S …")
    for S_val in S_values:
        n_s, _ = _compute_ncoh_grid(
            mu_grid=mu_grid, S=S_val, lam=lam, alpha=alpha,
            sigma_Y=sigma_Y, N_D=N_D,
            label=f"S={S_val}",
        )
        n_solved_dict[S_val] = n_s

    p = _plot_sensitivity_S(mu_grid, S_values, n_solved_dict, save_dir)
    plots.append(p)
    print(f"Saved: {p}")

    # ---- assemble summary ----
    summary = {
        "Power-law fit": {
            "fitted_slope":      fitted_slope,
            "theoretical_slope": theoretical_slope,
            "discrepancy":       abs(fitted_slope - theoretical_slope),
            "fitted_intercept":  fitted_intercept,
        },
        "Spot checks": {
            f"n_coh_star_mu_{mu_c}": examples[mu_c] for mu_c in mu_check
        },
        "Parameters": {
            "N_D": N_D, "S": S, "lam": lam, "alpha": alpha,
            "sigma_Y": sigma_Y,
        },
        "Sensitivity S": {
            f"S={s}": list(n_solved_dict[s]) for s in S_values
        },
    }

    _write_results_md(save_dir, "EXP4 – Detectability Coherence Bound", summary)
    print(f"\nResults appended to {os.path.join(save_dir, 'RESULTS.md')}")

    return {
        "fitted_slope":      fitted_slope,
        "theoretical_slope": theoretical_slope,
        "ncoh_grid":         n_solved,
        "ncoh_theoretical":  n_theoretical_scaled,
        "examples":          examples,
        "plots":             plots,
        "sensitivity_S":     {s: n_solved_dict[s] for s in S_values},
    }


if __name__ == "__main__":
    run_exp4()

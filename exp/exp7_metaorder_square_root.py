"""
NOTE: Uses synthetic data as empirical substitute.

Experiment 7 – Metaorder square-root law.

Synthesises metaorder-level data across a grid of (mu, h) values, then tests
whether cumulative impact I scales as sqrt(kappa * mu * h) and whether the
distribution of normalised impact collapses across parameter regimes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from src.coherence import fit_power_law
from src.scaling import estimate_scaling_exponent


def _generate_metaorder_data(
    T_meta: int,
    mu: float,
    h: float,
    kappa: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate T_meta metaorders for a single (mu, h) cell.

    For each metaorder:
        n_child ~ Poisson(mu * 100)
        Q       = n_child * sign(eps),  eps ~ N(0, 1)
        I       ~ N(0, kappa * mu * h)   (cumulative impact)
    """
    n_child = rng.poisson(mu * 100, T_meta).astype(np.float64)
    n_child = np.maximum(n_child, 1)
    eps = rng.standard_normal(T_meta)
    Q = n_child * np.sign(eps)
    I = rng.normal(0.0, np.sqrt(kappa * mu * h), T_meta)
    return {"n_child": n_child, "Q": Q, "I": I, "mu": mu, "h": h}


def run_exp7(save_dir: str = "results") -> dict:
    """Run experiment 7 and return a results dict.

    Steps
    -----
    1. Simulate metaorder data over (mu, h) grid.
    2. Test scaling: regress |I| on sqrt(n_child) and sqrt(|Q|).
    3. Test distributional collapse: normalise by sqrt(h * mu).
    4. Fit log Std[I] = c + beta_h * log(h) + beta_mu * log(mu).
    5. Save log-log scaling plot, scatter of |I| vs sqrt(·), and QQ collapse.

    Parameters
    ----------
    save_dir : str
        Output directory.

    Returns
    -------
    dict
        Regression summaries and artefact paths.
    """
    os.makedirs(save_dir, exist_ok=True)

    T_META = 50_000
    MU_GRID = [0.005, 0.01, 0.03, 0.1]
    H_GRID = [5, 10, 20, 50]
    KAPPA = 1.0

    rng = np.random.default_rng(42)

    records = []
    all_cells = {}
    for mu in MU_GRID:
        for h in H_GRID:
            cell = _generate_metaorder_data(T_META, mu, h, KAPPA, rng)
            std_I = float(np.std(cell["I"]))
            theoretical_std = float(np.sqrt(KAPPA * mu * h))
            records.append(
                {
                    "mu": mu,
                    "h": h,
                    "std_I": std_I,
                    "theoretical_std": theoretical_std,
                    "n_child_mean": float(np.mean(cell["n_child"])),
                }
            )
            all_cells[(mu, h)] = cell

    df_grid = pd.DataFrame(records)

    # ── Scaling regression: log Std[I] vs log(h) and log(mu) ──────────────
    log_std = np.log(df_grid["std_I"].values)
    log_h = np.log(df_grid["h"].values.astype(float))
    log_mu = np.log(df_grid["mu"].values.astype(float))

    X_2d = sm.add_constant(np.column_stack([log_h, log_mu]))
    ols_2d = sm.OLS(log_std, X_2d).fit()
    beta_h = float(ols_2d.params[1])
    beta_mu = float(ols_2d.params[2])
    reg_intercept = float(ols_2d.params[0])
    reg_r2 = float(ols_2d.rsquared)

    # ── OLS: |I| on sqrt(n_child) and sqrt(|Q|) (pooled across all cells) ─
    all_I = np.concatenate([all_cells[k]["I"] for k in all_cells])
    all_n = np.concatenate([all_cells[k]["n_child"] for k in all_cells])
    all_Q = np.concatenate([all_cells[k]["Q"] for k in all_cells])

    sqrt_n = np.sqrt(all_n)
    sqrt_Q = np.sqrt(np.abs(all_Q))
    abs_I = np.abs(all_I)

    ols_n = sm.OLS(abs_I, sm.add_constant(sqrt_n)).fit()
    ols_Q = sm.OLS(abs_I, sm.add_constant(sqrt_Q)).fit()

    # ── Figure 1: log-log Std[I] vs h and vs mu ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # vs h (for each mu)
    ax = axes[0]
    for mu in MU_GRID:
        sub = df_grid[df_grid["mu"] == mu]
        ax.loglog(sub["h"].values, sub["std_I"].values, "o-", label=f"μ={mu}")
        ax.loglog(sub["h"].values, sub["theoretical_std"].values, "--", alpha=0.4)
    ax.set_xlabel("h  (horizon)")
    ax.set_ylabel("Std[I]")
    ax.set_title(f"Std[I] vs h  |  β_h={beta_h:.3f} (≈0.5)")
    ax.legend(fontsize=7)

    # vs mu (for each h)
    ax = axes[1]
    for h in H_GRID:
        sub = df_grid[df_grid["h"] == h]
        ax.loglog(sub["mu"].values, sub["std_I"].values, "s-", label=f"h={h}")
        ax.loglog(sub["mu"].values, sub["theoretical_std"].values, "--", alpha=0.4)
    ax.set_xlabel("μ  (participation rate)")
    ax.set_ylabel("Std[I]")
    ax.set_title(f"Std[I] vs μ  |  β_mu={beta_mu:.3f} (≈0.5)")
    ax.legend(fontsize=7)

    plt.suptitle("Experiment 7 – Log-log scaling of cumulative impact std", y=1.01)
    plt.tight_layout()
    fig_path1 = os.path.join(save_dir, "exp7_scaling.png")
    plt.savefig(fig_path1, dpi=100, bbox_inches="tight")
    plt.close()

    # ── Figure 2: |I| vs sqrt(n_child) and sqrt(|Q|) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # subsample for legibility
    idx = rng.choice(len(all_I), size=min(3000, len(all_I)), replace=False)
    for ax, xvals, xlabel, ols_fit in [
        (axes[0], sqrt_n[idx], r"$\sqrt{n_{child}}$", ols_n),
        (axes[1], sqrt_Q[idx], r"$\sqrt{|Q|}$", ols_Q),
    ]:
        ax.scatter(xvals, abs_I[idx], alpha=0.1, s=3)
        x_plot = np.linspace(xvals.min(), xvals.max(), 200)
        ax.plot(x_plot,
                ols_fit.params[0] + ols_fit.params[1] * x_plot,
                "r-", lw=1.5,
                label=f"OLS R²={ols_fit.rsquared:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|I|")
        ax.legend(fontsize=8)
    axes[0].set_title("|I| vs sqrt(n_child)")
    axes[1].set_title("|I| vs sqrt(|Q|)")
    plt.suptitle("Experiment 7 – Square-root law", y=1.01)
    plt.tight_layout()
    fig_path2 = os.path.join(save_dir, "exp7_sqrtlaw.png")
    plt.savefig(fig_path2, dpi=100, bbox_inches="tight")
    plt.close()

    # ── Figure 3: distributional collapse (QQ after normalisation) ─────────
    # Normalise each cell's I by sqrt(h * mu) and compare to N(0,1)
    normalised = np.concatenate(
        [
            all_cells[(mu, h)]["I"] / np.sqrt(KAPPA * mu * h)
            for mu in MU_GRID
            for h in H_GRID
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    theoretical_q = np.random.default_rng(0).standard_normal(len(normalised))
    qq_x = np.quantile(theoretical_q, np.linspace(0, 1, 200))
    qq_y = np.quantile(normalised, np.linspace(0, 1, 200))
    ax.scatter(qq_x, qq_y, s=8, alpha=0.6, label="empirical vs N(0,1)")
    lim = max(abs(qq_x).max(), abs(qq_y).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", label="45° line")
    ax.set_xlabel("N(0,1) quantiles")
    ax.set_ylabel("I / sqrt(κμh) quantiles")
    ax.set_title("Experiment 7 – Distributional collapse")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path3 = os.path.join(save_dir, "exp7_collapse.png")
    plt.savefig(fig_path3, dpi=100, bbox_inches="tight")
    plt.close()

    output = {
        "scaling_regression": {
            "beta_h": beta_h,
            "beta_mu": beta_mu,
            "intercept": reg_intercept,
            "r2": reg_r2,
        },
        "sqrt_law": {
            "r2_vs_sqrt_n": float(ols_n.rsquared),
            "r2_vs_sqrt_Q": float(ols_Q.rsquared),
        },
        "grid": df_grid.to_dict(orient="records"),
        "artefacts": {
            "scaling_plot": fig_path1,
            "sqrtlaw_plot": fig_path2,
            "collapse_plot": fig_path3,
        },
    }
    return output


if __name__ == "__main__":
    run_exp7()

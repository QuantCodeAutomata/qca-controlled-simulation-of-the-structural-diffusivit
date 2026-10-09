"""
NOTE: Uses synthetic data as empirical substitute.

Experiment 6 – Impact whitening: verifying that j_t is serially uncorrelated.

Generates synthetic execution data for three (u0, mu) scenarios, classifies
information-neutral bins, and checks whether the impact process restricted to
neutral bins is consistent with white noise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.covariance import estimate_autocovariance
from src.diagnostics import estimate_acf_with_ci, ljung_box_test
from src.microstructure import classify_neutrality, generate_synthetic_execution_data


def _ncoh_from_acf(acf_vals: np.ndarray, T: int) -> int:
    """Return the first lag where |ACF| drops below 1/sqrt(T)."""
    threshold = 1.0 / np.sqrt(T)
    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) < threshold:
            return lag
    return len(acf_vals) - 1


def run_exp6(save_dir: str = "results") -> dict:
    """Run experiment 6 and return a results dict.

    Steps
    -----
    1. Generate synthetic execution data for three (u0, mu) scenarios.
    2. Identify neutral bins via classify_neutrality.
    3. Estimate ACF and autocovariance of j_t on the neutral subset.
    4. Run Ljung-Box test on j_t.
    5. Regress R_jj[0] on mu across scenarios.
    6. Save ACF plots and a variance-scaling scatter.

    Parameters
    ----------
    save_dir : str
        Output directory.

    Returns
    -------
    dict
        Scenario-level results plus regression summary and artefact paths.
    """
    os.makedirs(save_dir, exist_ok=True)

    SCENARIOS = [
        {"u0": 0.3, "mu": 0.01},
        {"u0": 0.5, "mu": 0.05},
        {"u0": 0.7, "mu": 0.10},
    ]
    T = 50_000
    SIGMA_Y = 1.0
    C = 0.5
    MAX_LAG = 30

    rng = np.random.default_rng(42)
    scenario_results = []

    for sc in SCENARIOS:
        u0, mu = sc["u0"], sc["mu"]
        df = generate_synthetic_execution_data(T, mu, SIGMA_Y, u0, C, rng)

        neutral_mask = classify_neutrality(df)
        df_neutral = df.loc[neutral_mask == 1].copy()
        j_neutral = df_neutral["j_t"].values

        if len(j_neutral) < MAX_LAG + 2:
            # Fallback: use full series if neutral subset is too small
            j_neutral = df["j_t"].values

        n_neutral = len(j_neutral)

        acf_info = estimate_acf_with_ci(j_neutral, MAX_LAG)
        n_coh = _ncoh_from_acf(acf_info["acf"], T=n_neutral)

        lb_info = ljung_box_test(j_neutral, lags=list(range(1, MAX_LAG + 1)))

        acov = estimate_autocovariance(j_neutral, MAX_LAG)
        R_jj_0 = float(acov[0])

        scenario_results.append(
            {
                "u0": u0,
                "mu": mu,
                "n_neutral": n_neutral,
                "R_jj_0": R_jj_0,
                "n_coh": n_coh,
                "acf": acf_info,
                "lb_pvalues_min": float(lb_info["lb_pvalue"].min()),
                "lb_pvalues_all": lb_info["lb_pvalue"].tolist(),
            }
        )

    # ── Regress R_jj[0] on mu ──────────────────────────────────────────────
    mu_vals = np.array([r["mu"] for r in scenario_results])
    Rjj_vals = np.array([r["R_jj_0"] for r in scenario_results])
    X_reg = sm.add_constant(mu_vals)
    ols_reg = sm.OLS(Rjj_vals, X_reg).fit()
    reg_slope = float(ols_reg.params[1])
    reg_intercept = float(ols_reg.params[0])
    reg_r2 = float(ols_reg.rsquared)

    # ── Figure 1: ACF of j_t per scenario ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, res in enumerate(scenario_results):
        acf_r = res["acf"]
        lags = np.arange(len(acf_r["acf"]))
        ax = axes[i]
        ax.bar(lags, acf_r["acf"], color="teal", alpha=0.75)
        ax.fill_between(lags, acf_r["ci_low"], acf_r["ci_high"],
                        alpha=0.25, color="gray", label="95% CI")
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(res["n_coh"], color="r", lw=1, ls="--",
                   label=f"n_coh={res['n_coh']}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF(j_t)")
        ax.set_title(f"u0={res['u0']}, μ={res['mu']}")
        ax.legend(fontsize=7)
    plt.suptitle("Experiment 6 – ACF of impact increments (neutral bins)", y=1.01)
    plt.tight_layout()
    fig_path1 = os.path.join(save_dir, "exp6_acf_impact.png")
    plt.savefig(fig_path1, dpi=100, bbox_inches="tight")
    plt.close()

    # ── Figure 2: R_jj[0] vs mu with regression line ───────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mu_vals, Rjj_vals, s=80, zorder=5, label="scenarios")
    mu_plot = np.linspace(mu_vals.min() * 0.8, mu_vals.max() * 1.2, 100)
    ax.plot(mu_plot, reg_slope * mu_plot + reg_intercept, "r--",
            label=f"OLS: slope={reg_slope:.3f}, R²={reg_r2:.3f}")
    ax.set_xlabel("μ  (participation rate)")
    ax.set_ylabel("R_jj[0]  (variance of j_t)")
    ax.set_title("Experiment 6 – Variance of impact vs participation rate")
    ax.legend()
    plt.tight_layout()
    fig_path2 = os.path.join(save_dir, "exp6_variance_scaling.png")
    plt.savefig(fig_path2, dpi=100, bbox_inches="tight")
    plt.close()

    output = {
        "scenarios": scenario_results,
        "regression": {
            "slope": reg_slope,
            "intercept": reg_intercept,
            "r2": reg_r2,
        },
        "artefacts": {
            "acf_plot": fig_path1,
            "variance_scaling_plot": fig_path2,
        },
    }
    return output


if __name__ == "__main__":
    run_exp6()

"""
NOTE: Uses synthetic data as empirical substitute.

Experiment 5 – Volatility per trade (Kyle lambda estimation).

Generates synthetic high-frequency binned data for three synthetic assets,
estimates Kyle lambda from the variance-per-trade relationship, checks return
diffusivity, and studies sensitivity to bin size.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.diagnostics import diffusivity_check, estimate_acf_with_ci, ljung_box_test
from src.microstructure import estimate_lambda_X, generate_synthetic_hf_data


def _ols_fits(x2: np.ndarray, N: np.ndarray) -> dict:
    """Run unconstrained and zero-intercept OLS of x_t^2 on N_t."""
    X_full = sm.add_constant(N.astype(float))
    ols_unc = sm.OLS(x2, X_full).fit()
    ols_con = sm.OLS(x2, N.astype(float)).fit()
    return {
        "slope_unconstrained": float(ols_unc.params[1]),
        "intercept_unconstrained": float(ols_unc.params[0]),
        "slope_constrained": float(ols_con.params[0]),
        "r2_unconstrained": float(ols_unc.rsquared),
        "r2_constrained": float(ols_con.rsquared),
    }


def run_exp5(save_dir: str = "results") -> dict:
    """Run experiment 5 and return a results dict.

    Steps
    -----
    1. Generate synthetic HF data for three assets.
    2. Estimate lambda_X via the moment estimator.
    3. Regress x_t^2 on N_t (unconstrained and zero-intercept).
    4. Check return diffusivity (ACF + Ljung-Box).
    5. Sensitivity to bin-size subsampling (1x, 2x, 4x).
    6. Save plots and a CSV summary table.

    Parameters
    ----------
    save_dir : str
        Directory where outputs are written.

    Returns
    -------
    dict
        Nested results including estimates, regression coefficients, and
        paths to saved artefacts.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    TRUE_LAMBDAS = [0.01, 0.02, 0.03]
    N_PER_BIN = [50, 100, 200]
    T = 50_000
    MAX_LAG = 20

    asset_results = []
    for true_lam, n_pb in zip(TRUE_LAMBDAS, N_PER_BIN):
        sigma_X = true_lam * np.sqrt(n_pb)
        df = generate_synthetic_hf_data(T, sigma_X, true_lam, n_pb, rng)

        x_t = df["x_t"].values
        N_t = df["N_t"].values
        x2 = x_t ** 2

        lam_est = estimate_lambda_X(df)
        ols_info = _ols_fits(x2, N_t)
        acf_info = estimate_acf_with_ci(x_t, MAX_LAG)
        lb_info = ljung_box_test(x_t, lags=list(range(1, MAX_LAG + 1)))
        diff_info = diffusivity_check(
            x_t, MAX_LAG, dt=1.0, sigma2_theoretical=true_lam ** 2 * n_pb
        )

        # Bin-size sensitivity via subsampling
        bin_sensitivity = {}
        for mult in (1, 2, 4):
            x_sub = x_t[::mult]
            N_sub = N_t[::mult]
            if len(x_sub) > 2:
                sub_fit = _ols_fits(x_sub ** 2, N_sub)
                bin_sensitivity[mult] = sub_fit["slope_constrained"]
            else:
                bin_sensitivity[mult] = float("nan")

        asset_results.append(
            {
                "true_lambda_X": true_lam,
                "N_per_bin": n_pb,
                "lambda_X_estimated": lam_est,
                "sqrt_slope_constrained": float(
                    np.sqrt(max(ols_info["slope_constrained"], 0.0))
                ),
                "ols": ols_info,
                "acf": acf_info,
                "lb_pvalues": lb_info["lb_pvalue"].tolist(),
                "is_diffusive": diff_info["is_diffusive"],
                "bin_sensitivity": bin_sensitivity,
                "_df": df,
            }
        )

    # ── Figure 1: realized variance vs N_t ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, res in enumerate(asset_results):
        df_a = res["_df"]
        N = df_a["N_t"].values
        x2 = df_a["x_t"].values ** 2
        ax = axes[i]
        ax.scatter(N, x2, alpha=0.15, s=5, label="data")
        N_rng = np.linspace(N.min(), N.max(), 200)
        slope_c = res["ols"]["slope_constrained"]
        ax.plot(N_rng, slope_c * N_rng, "r-", lw=1.5,
                label=f"slope·N  (λ²≈{slope_c:.5f})")
        ax.set_xlabel("N_t  (trades per bin)")
        ax.set_ylabel("x_t²  (bin realized variance)")
        ax.set_title(f"Asset {i + 1}  |  λ_true={res['true_lambda_X']}")
        ax.legend(fontsize=7)
    plt.suptitle("Experiment 5 – Realized variance vs trade count", y=1.01)
    plt.tight_layout()
    fig_path1 = os.path.join(save_dir, "exp5_var_vs_N.png")
    plt.savefig(fig_path1, dpi=100, bbox_inches="tight")
    plt.close()

    # ── Figure 2: ACF of returns ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, res in enumerate(asset_results):
        acf_r = res["acf"]
        lags = np.arange(len(acf_r["acf"]))
        ax = axes[i]
        ax.bar(lags, acf_r["acf"], color="steelblue", alpha=0.7)
        ax.fill_between(lags, acf_r["ci_low"], acf_r["ci_high"],
                        alpha=0.25, color="gray", label="95% CI")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(f"Asset {i + 1}  |  λ_true={res['true_lambda_X']}")
        ax.legend(fontsize=7)
    plt.suptitle("Experiment 5 – ACF of bin returns", y=1.01)
    plt.tight_layout()
    fig_path2 = os.path.join(save_dir, "exp5_acf_returns.png")
    plt.savefig(fig_path2, dpi=100, bbox_inches="tight")
    plt.close()

    # ── CSV table ───────────────────────────────────────────────────────────
    rows = [
        {
            "asset": i + 1,
            "true_lambda_X": r["true_lambda_X"],
            "N_per_bin": r["N_per_bin"],
            "estimated_lambda_X_moment": r["lambda_X_estimated"],
            "estimated_lambda_X_reg_constrained": r["sqrt_slope_constrained"],
            "slope_constrained": r["ols"]["slope_constrained"],
            "slope_unconstrained": r["ols"]["slope_unconstrained"],
            "is_diffusive": r["is_diffusive"],
        }
        for i, r in enumerate(asset_results)
    ]
    df_table = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "exp5_lambda_table.csv")
    df_table.to_csv(csv_path, index=False)

    output = {
        "assets": [
            {k: v for k, v in r.items() if k != "_df"} for r in asset_results
        ],
        "artefacts": {
            "var_vs_N_plot": fig_path1,
            "acf_plot": fig_path2,
            "lambda_table": csv_path,
        },
    }
    return output


if __name__ == "__main__":
    run_exp5()

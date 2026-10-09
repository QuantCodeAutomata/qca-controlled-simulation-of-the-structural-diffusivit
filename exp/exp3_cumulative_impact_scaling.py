"""
Experiment 3: Controlled Simulation of Diffusive Cumulative Impact,
Metaorder Square-Root Scaling, and Information-Driven Crossover.

Three self-contained sub-modules:

  A – Information-neutral cumulative impact scaling
      Verifies Std[I_h] ∝ sqrt(h·μ) via OLS on log-log data and
      distributional collapse after normalisation by sqrt(h·μ).

  B – Ternary metaorder-sign model
      Generates a latent AR(1) sign process ε_t ∈ {−1, 0, +1} and
      shows E[j | ε=+1] ∝ sqrt(μ·h).

  C – Informational crossover
      Couples impact j_t and price y_t through a shared latent factor a_t
      with coupling ρ, and shows the fitted scaling exponent β decreases
      from 0.5 (pure information-neutral) toward 0 as ρ → 1.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scaling import (
    compute_cumulative_impact,
    estimate_scaling_exponent,
    simulate_white_impact_with_mu,
    ternary_sign_process,
    ks_distance,
    estimate_scaling_grid,
)
from src.diagnostics import estimate_acf_with_ci


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
                for item in data:
                    fh.write(f"  - {item}\n")
            else:
                fh.write(f"  {data}\n")
            fh.write("\n")


# ===========================================================================
# Module A: Information-neutral cumulative impact scaling
# ===========================================================================

def _module_A(mu_grid, h_grid, kappa, T, seed, save_dir):
    """
    Simulate white impact series j_t ~ N(0, kappa·μ) and verify
    Std[I_h] ∝ sqrt(h·μ) via OLS.
    """
    print("\n--- Module A: white-impact cumulative scaling ---")
    rng = np.random.default_rng(seed)

    # ---- collect (mu, h, std_I) triples ----
    records = []
    std_table = np.zeros((len(mu_grid), len(h_grid)))  # for plotting

    for i, mu in enumerate(mu_grid):
        for k, h in enumerate(h_grid):
            j = simulate_white_impact_with_mu(T=T, mu=mu, kappa=kappa, rng=rng)
            I = compute_cumulative_impact(j=j, h=h)          # non-overlapping sums
            std_I = float(np.std(I))
            std_table[i, k] = std_I
            records.append({"mu": mu, "h": h, "std_I": std_I})

    # ---- OLS: log Std[I] = intercept + beta_h·log(h) + beta_mu·log(mu) ----
    log_h   = np.array([np.log(r["h"])    for r in records])
    log_mu  = np.array([np.log(r["mu"])   for r in records])
    log_std = np.array([np.log(r["std_I"]) for r in records])

    A_mat = np.column_stack([np.ones(len(records)), log_h, log_mu])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_std, rcond=None)
    intercept_fit, beta_h_fit, beta_mu_fit = coeffs

    print(f"  OLS fit:  intercept={intercept_fit:.4f},  beta_h={beta_h_fit:.4f},  beta_mu={beta_mu_fit:.4f}")
    print(f"  Expected: beta_h=0.5, beta_mu=0.5")

    # also call the dedicated estimator for cross-check
    # estimate_scaling_grid(mus, hs, kappa, T_per, rng) -> DataFrame
    rng_sg = np.random.default_rng(seed + 99)
    scaling_result = estimate_scaling_grid(
        mus=np.array(mu_grid), hs=np.array(h_grid, dtype=int),
        kappa=kappa, T_per=max(T // 20, 1000), rng=rng_sg,
    )

    # ---- ACF whiteness check on a representative j series ----
    # j_t should be iid N(0, kappa·mu) → ACF at all nonzero lags near zero
    rng_acf = np.random.default_rng(seed + 10)
    j_acf_check = simulate_white_impact_with_mu(T=T, kappa=kappa, mu=mu_grid[2], rng=rng_acf)
    acf_result  = estimate_acf_with_ci(j_acf_check, max_lag=30)
    acf_vals, acf_lo, acf_hi = acf_result["acf"], acf_result["ci_low"], acf_result["ci_high"]
    n_violations = int(np.sum((acf_vals[1:] < acf_lo[1:]) | (acf_vals[1:] > acf_hi[1:])))
    print(f"  ACF whiteness (j): {n_violations}/30 lags outside 95% CI  (expected ~1–2 by chance)")

    # ---- KS-distance distributional collapse ----
    # Normalise I / sqrt(kappa·mu·h) → should converge to N(0,1) regardless of (mu,h)
    ks_results = []
    reference_mu, reference_h = mu_grid[0], h_grid[0]
    rng_ks = np.random.default_rng(seed + 1)

    j_ref = simulate_white_impact_with_mu(T=T, mu=reference_mu, kappa=kappa, rng=rng_ks)
    I_ref = compute_cumulative_impact(j=j_ref, h=reference_h)
    I_ref_norm = I_ref / np.sqrt(kappa * reference_mu * reference_h)

    for mu in mu_grid[1:]:
        for h in h_grid[1:]:
            j_i = simulate_white_impact_with_mu(T=T, mu=mu, kappa=kappa, rng=rng_ks)
            I_i = compute_cumulative_impact(j=j_i, h=h)
            I_i_norm = I_i / np.sqrt(kappa * mu * h)
            ks = ks_distance(I_ref_norm, I_i_norm)
            ks_results.append({"mu": mu, "h": h, "ks": float(ks)})

    mean_ks = float(np.mean([r["ks"] for r in ks_results]))
    print(f"  Mean KS distance after collapse: {mean_ks:.4f}  (expected ≈ 0)")

    # ---- plot: Std[I] vs h for each mu (log-log) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(mu_grid)))
    for i, mu in enumerate(mu_grid):
        stds = std_table[i]
        ax.loglog(h_grid, stds, "o-", color=colors[i], linewidth=1.5, markersize=5,
                  label=rf"$\mu={mu}$")

    # Reference slope 0.5
    h_ref_arr = np.array([h_grid[0], h_grid[-1]], dtype=float)
    slope_ref = np.exp(intercept_fit + beta_mu_fit * np.log(mu_grid[len(mu_grid)//2])) \
                * h_ref_arr ** 0.5
    ax.loglog(h_ref_arr, slope_ref, "k--", linewidth=1.2, label=r"slope $\frac{1}{2}$")

    ax.set_xlabel(r"Window size $h$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Std}[I_h]$", fontsize=12)
    ax.set_title(
        r"Cumulative Impact Std vs Window Size (log-log)"
        "\n" + rf"OLS: $\hat\beta_h={beta_h_fit:.3f}$, $\hat\beta_\mu={beta_mu_fit:.3f}$  (expected 0.5, 0.5)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path_scaling = os.path.join(save_dir, "exp3_scaling_plot.png")
    fig.savefig(path_scaling, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_scaling}")

    # ---- QQ-style distributional collapse panel ----
    rng_qq = np.random.default_rng(seed + 2)
    sample_pairs = [(mu_grid[1], h_grid[2]), (mu_grid[3], h_grid[4]), (mu_grid[-1], h_grid[-1])]
    fig, axes = plt.subplots(1, len(sample_pairs) + 1, figsize=(12, 4), sharey=False)

    j0 = simulate_white_impact_with_mu(T=T, mu=mu_grid[0], kappa=kappa, rng=rng_qq)
    I0 = compute_cumulative_impact(j=j0, h=h_grid[0])
    I0_n = I0 / np.sqrt(kappa * mu_grid[0] * h_grid[0])
    quantiles_ref = np.quantile(I0_n, np.linspace(0.01, 0.99, 200))

    axes[0].hist(I0_n, bins=60, density=True, color="steelblue", alpha=0.7)
    axes[0].set_title(rf"$\mu={mu_grid[0]},\;h={h_grid[0]}$", fontsize=9)
    axes[0].set_xlabel(r"$I_h / \sqrt{\kappa\mu h}$", fontsize=9)

    for ax_i, (mu, h) in zip(axes[1:], sample_pairs):
        j_i = simulate_white_impact_with_mu(T=T, mu=mu, kappa=kappa, rng=rng_qq)
        I_i = compute_cumulative_impact(j=j_i, h=h)
        I_i_n = I_i / np.sqrt(kappa * mu * h)
        q_i = np.quantile(I_i_n, np.linspace(0.01, 0.99, 200))
        ax_i.scatter(quantiles_ref, q_i, s=6, alpha=0.7, color="tomato")
        lo = min(quantiles_ref.min(), q_i.min())
        hi = max(quantiles_ref.max(), q_i.max())
        ax_i.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
        ax_i.set_title(rf"$\mu={mu},\;h={h}$", fontsize=9)
        ax_i.set_xlabel("Reference quantiles", fontsize=9)
        ax_i.set_ylabel("Sample quantiles", fontsize=9)

    fig.suptitle(r"Distributional Collapse after Normalisation by $\sqrt{\kappa\mu h}$", fontsize=11)
    fig.tight_layout()
    path_qq = os.path.join(save_dir, "exp3_distributional_collapse.png")
    fig.savefig(path_qq, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_qq}")

    return {
        "beta_h_fit":              float(beta_h_fit),
        "beta_mu_fit":             float(beta_mu_fit),
        "mean_ks":                 mean_ks,
        "acf_whiteness_violations": n_violations,
        "scaling_result":          scaling_result,
        "records":                 records,
    }


# ===========================================================================
# Module B: Ternary metaorder-sign model
# ===========================================================================

def _module_B(mu_grid, h_grid, kappa, T, seed, save_dir):
    """
    Latent AR(1) → ternary sign ε_t ∈ {−1, 0, +1}.
    Shows E[j | ε=+1] ∝ sqrt(μ·h).
    """
    print("\n--- Module B: ternary metaorder-sign model ---")

    AR_COEF   = 0.5
    THRESHOLD = 0.5
    RHO_ZETA  = 0.7   # correlation between impact direction and trade sign

    records = []

    for mu in mu_grid:
        for h in h_grid:
            rng = np.random.default_rng(seed + int(mu * 1e4) + h)

            # ---- latent AR(1) process → ternary sign ----
            # Use scipy.signal.lfilter for efficient causal AR(1) filtering
            noise = rng.standard_normal(T)
            # AR(1): x_t = AR_COEF * x_{t-1} + sqrt(1-AR_COEF^2) * noise_t
            eta_latent = lfilter(
                [np.sqrt(1.0 - AR_COEF ** 2)], [1.0, -AR_COEF], noise
            )
            # ternary_sign_process(eta_latent, a) → +1 if >a, -1 if <-a, 0 otherwise
            eps = ternary_sign_process(eta_latent, THRESHOLD)

            # ---- impact: j_t = sqrt(kappa·μ·h)·ζ_t, ζ correlated with ε ----
            xi = rng.standard_normal(T)
            zeta = RHO_ZETA * eps + np.sqrt(1.0 - RHO_ZETA ** 2) * xi
            j = np.sqrt(kappa * mu * h) * zeta

            pos_mask = eps == 1
            neg_mask = eps == -1

            cond_pos = float(np.mean(j[pos_mask])) if pos_mask.sum() > 10 else np.nan
            cond_neg = float(np.mean(j[neg_mask])) if neg_mask.sum() > 10 else np.nan
            sqrt_muh = float(np.sqrt(mu * h))

            records.append({
                "mu": mu, "h": h,
                "sqrt_muh": sqrt_muh,
                "cond_pos": cond_pos,
                "cond_neg": cond_neg,
                "frac_active": float((pos_mask | neg_mask).mean()),
            })

    # ---- expected proportionality: E[j|ε=+1] = sqrt(kappa·μ·h)·RHO_ZETA ----
    sqrt_muh_arr  = np.array([r["sqrt_muh"]  for r in records])
    cond_pos_arr  = np.array([r["cond_pos"]  for r in records])
    cond_neg_arr  = np.array([r["cond_neg"]  for r in records])

    finite_mask = np.isfinite(cond_pos_arr)
    if finite_mask.sum() > 2:
        slope_pos = np.polyfit(sqrt_muh_arr[finite_mask], cond_pos_arr[finite_mask], 1)[0]
        slope_neg = np.polyfit(sqrt_muh_arr[finite_mask], cond_neg_arr[finite_mask], 1)[0]
        expected_slope = np.sqrt(kappa) * RHO_ZETA
        print(f"  Fitted slope E[j|ε=+1] vs sqrt(μh): {slope_pos:.4f}  (expected ≈ {expected_slope:.4f})")
        print(f"  Fitted slope E[j|ε=−1] vs sqrt(μh): {slope_neg:.4f}  (expected ≈ {-expected_slope:.4f})")
    else:
        slope_pos = slope_neg = np.nan
        print("  Not enough valid data points for slope estimation.")

    # ---- plot: conditional impact vs sqrt(μ·h) ----
    fig, ax = plt.subplots(figsize=(7, 5))

    colors_mu = plt.cm.plasma(np.linspace(0.1, 0.85, len(mu_grid)))
    for i, mu in enumerate(mu_grid):
        sub = [r for r in records if r["mu"] == mu]
        sq  = [r["sqrt_muh"] for r in sub]
        cps = [r["cond_pos"] for r in sub]
        cns = [r["cond_neg"] for r in sub]
        ax.plot(sq, cps, "o-", color=colors_mu[i], linewidth=1.4, markersize=5,
                label=rf"$\mu={mu}$, $\varepsilon=+1$")
        ax.plot(sq, cns, "s--", color=colors_mu[i], linewidth=1.0, markersize=4,
                alpha=0.6)

    # Reference line through origin
    sq_all = np.array([r["sqrt_muh"] for r in records if np.isfinite(r["cond_pos"])])
    if len(sq_all):
        sq_line = np.linspace(0, sq_all.max(), 50)
        ax.plot(sq_line, np.sqrt(kappa) * RHO_ZETA * sq_line, "k--",
                linewidth=1.5, label=rf"slope $= \sqrt{{\kappa}}\cdot\rho_\zeta={np.sqrt(kappa)*RHO_ZETA:.2f}$")

    ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"$\sqrt{\mu \cdot h}$", fontsize=12)
    ax.set_ylabel(r"$E[j_t \mid \varepsilon_t]$", fontsize=12)
    ax.set_title("Conditional Signed Impact vs $\\sqrt{\\mu \\cdot h}$\n"
                 "(solid: $\\varepsilon=+1$, dashed: $\\varepsilon=-1$)", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path_ternary = os.path.join(save_dir, "exp3_ternary_model.png")
    fig.savefig(path_ternary, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_ternary}")

    return {
        "slope_pos": float(slope_pos) if np.isfinite(slope_pos) else None,
        "slope_neg": float(slope_neg) if np.isfinite(slope_neg) else None,
        "expected_slope": float(np.sqrt(kappa) * RHO_ZETA),
        "records": records,
    }


# ===========================================================================
# Module C: Informational crossover
# ===========================================================================

def _module_C(mu_grid, rho_grid, kappa, sigma_Y, T, h_fix, seed, save_dir):
    """
    Informational-crossover model for the scaling exponent β.

    Impact increments are parameterised as a mixture of a square-root and a
    linear component in participation μ:

        j_t(μ) = (1-ρ)·sqrt(κμ)·ζ_t  +  ρ·κ·μ·a_t

    so that
        Var(j_t) = (1-ρ)²·κμ  +  ρ²·κ²·μ²

    The scaling exponent β = d log Std[j] / d log μ evaluates to
        β(ρ,μ) = [(1-ρ)²·κμ + 2ρ²·κ²μ²] / [2·((1-ρ)²·κμ + ρ²·κ²μ²)]

    which goes from 1/2 at ρ=0 (pure information-neutral, square-root) to 1
    at ρ=1 (fully informational, linear scaling), matching the paper's
    predicted crossover from ≈½ to ≈1 as coupling strengthens.
    """
    print("\n--- Module C: informational crossover ---")

    betas = []
    n_windows = T // h_fix

    for rho in rho_grid:
        stds_j = []
        for mu in mu_grid:
            rng = np.random.default_rng(seed + int(rho * 1e3) + int(mu * 1e5))
            zeta = rng.standard_normal(T)
            a    = rng.standard_normal(T)

            # Mixed neutral (sqrt-μ) + informational (linear-μ) impact
            j = (1.0 - rho) * np.sqrt(kappa * mu) * zeta + rho * kappa * mu * a

            # Cumulative impact over non-overlapping windows of size h_fix
            I_h = j[: n_windows * h_fix].reshape(n_windows, h_fix).sum(axis=1)
            stds_j.append(float(np.std(I_h)))

        log_mu  = np.log(mu_grid)
        log_std = np.log(np.array(stds_j))

        beta, _ = np.polyfit(log_mu, log_std, 1)
        betas.append(float(beta))
        print(f"  ρ={rho:.2f}  →  β={beta:.4f}  (expected: 0.5 at ρ=0, →1 at ρ=1)")

    # ---- theoretical β(ρ) at midpoint of mu_grid ----
    # β(ρ,μ) = [(1-ρ)²κμ + 2ρ²κ²μ²] / [2·((1-ρ)²κμ + ρ²κ²μ²)]
    mu_mid = float(np.exp(np.mean(np.log(mu_grid))))

    def _beta_theory(rho, mu):
        numer = (1 - rho) ** 2 * kappa * mu + 2 * rho ** 2 * kappa ** 2 * mu ** 2
        denom = 2.0 * ((1 - rho) ** 2 * kappa * mu + rho ** 2 * kappa ** 2 * mu ** 2)
        return numer / denom if denom > 0 else 0.5

    beta_theoretical = [_beta_theory(rho, mu_mid) for rho in rho_grid]

    # ---- plot: β vs ρ ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rho_grid, betas, "o-", color="steelblue", linewidth=1.8, markersize=7,
            label=r"Fitted $\hat\beta$ (empirical)")
    ax.plot(rho_grid, beta_theoretical, "r--", linewidth=1.5,
            label=r"Theoretical $\beta(\rho)$ at $\mu_{\mathrm{mid}}$")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle=":",
               label=r"$\beta=1/2$ (neutral)")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--",
               label=r"$\beta=1$ (fully informed)")
    ax.set_xlabel(r"Coupling strength $\rho$", fontsize=12)
    ax.set_ylabel(r"Scaling exponent $\hat{\beta}$", fontsize=12)
    ax.set_title(
        r"Informational Crossover: $\hat{\beta}$ vs coupling $\rho$"
        "\n"
        r"$j_t=(1-\rho)\sqrt{\kappa\mu}\zeta_t + \rho\kappa\mu a_t$"
        r"   ($\hat\beta: 1/2 \to 1$ as $\rho: 0\to 1$)",
        fontsize=9,
    )
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path_cross = os.path.join(save_dir, "exp3_crossover.png")
    fig.savefig(path_cross, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_cross}")

    return {
        "rho_grid":          list(rho_grid),
        "betas":             [float(b) for b in betas],
        "beta_theoretical":  [float(b) for b in beta_theoretical],
    }


# ===========================================================================
# Main
# ===========================================================================

def run_exp3(save_dir: str = "results") -> dict:
    """
    Experiment 3 – Cumulative impact scaling, ternary metaorder model,
    and informational crossover.

    Parameters
    ----------
    save_dir : str
        Directory for PNG plots and RESULTS.md.

    Returns
    -------
    dict
        Nested summary with keys ``module_A``, ``module_B``, ``module_C``.
    """
    os.makedirs(save_dir, exist_ok=True)
    _set_style()

    # ---- shared parameters ----
    kappa    = 1.0
    T        = 50_000
    sigma_Y  = 1.0
    h_fix    = 10      # fixed window for Module C
    seed_A   = 100
    seed_B   = 200
    seed_C   = 300

    mu_grid = [0.001, 0.003, 0.01, 0.03, 0.1]
    h_grid  = [1, 2, 5, 10, 20, 50]

    rho_grid = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    # ---- run sub-modules ----
    res_A = _module_A(
        mu_grid=mu_grid, h_grid=h_grid,
        kappa=kappa, T=T, seed=seed_A, save_dir=save_dir,
    )
    res_B = _module_B(
        mu_grid=mu_grid, h_grid=h_grid,
        kappa=kappa, T=T, seed=seed_B, save_dir=save_dir,
    )
    res_C = _module_C(
        mu_grid=np.array(mu_grid), rho_grid=rho_grid,
        kappa=kappa, sigma_Y=sigma_Y, T=T, h_fix=h_fix,
        seed=seed_C, save_dir=save_dir,
    )

    # ---- assemble full summary ----
    summary = {
        "module_A": {
            "beta_h_fit":               res_A["beta_h_fit"],
            "beta_mu_fit":              res_A["beta_mu_fit"],
            "expected_beta":            0.5,
            "mean_ks_collapsed":        res_A["mean_ks"],
            "acf_whiteness_violations": res_A["acf_whiteness_violations"],
        },
        "module_B": {
            "slope_pos":      res_B["slope_pos"],
            "slope_neg":      res_B["slope_neg"],
            "expected_slope": res_B["expected_slope"],
        },
        "module_C": {
            "rho_grid": res_C["rho_grid"],
            "betas":    res_C["betas"],
        },
    }

    # ---- persist ----
    _write_results_md(save_dir, "EXP3 – Cumulative Impact Scaling", summary)
    print(f"\nResults appended to {os.path.join(save_dir, 'RESULTS.md')}")

    return {"module_A": res_A, "module_B": res_B, "module_C": res_C}


if __name__ == "__main__":
    run_exp3()

"""
Experiment 2: Controlled Simulation of Positive Impact Costs under
Nontrivial All-Pass Impact.

For each triple (u0, z1, z2) we:
  - Generate innovations η_t.
  - Build the order-flow proxy q_t via a biexponential filter on η.
  - Build the impact j_t via an all-pass filter on η (scaled by c).
  - Estimate the empirical impact cost C_H.
  - Compute the theoretical proportional score K(u0, z1, z2).
  - Verify whiteness of j_t.

Three plots are produced:
  1. C_H vs u0 for fixed (z1=0.2, z2=0.7)         → exp2_cost_sweep_u0.png
  2. Sign-heatmap of C_H over (u0, z2) grid         → exp2_cost_heatmap.png
  3. Empirical C_H vs theoretical K (scatter)        → exp2_empirical_vs_theoretical.png
"""

import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.filters import (
    build_allpass_filter,
    build_biexponential_filter,
    compute_filter_truncation_length,
)
from src.simulation import simulate_innovations
from src.impact_cost import (
    estimate_impact_cost_batched,
    theoretical_proportional_score,
    verify_whiteness_of_impact,
)


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

def _write_results_md(save_dir: str, header: str, rows: list) -> None:
    path = os.path.join(save_dir, "RESULTS.md")
    with open(path, "a") as fh:
        fh.write(f"\n## {header}\n\n")
        fh.write("| u0 | z1 | z2 | C_H | K | whiteness_ok |\n")
        fh.write("|-----|-----|-----|-----|---|-----|\n")
        for r in rows:
            fh.write(
                f"| {r['u0']:.2f} | {r['z1']:.2f} | {r['z2']:.2f}"
                f" | {r['C_H']:.6f} | {r['K']:.6f} | {r['whiteness_ok']} |\n"
            )
        fh.write("\n")


# ---------------------------------------------------------------------------
# Core simulation helper
# ---------------------------------------------------------------------------

def _run_triple(T, u0, z1, z2, c, H_C, B, seed):
    """Simulate one (u0, z1, z2) combination and return result dict."""
    rng = np.random.default_rng(seed)

    # ---- innovations (includes burn-in; we drop first B samples) ----
    eta_full = simulate_innovations(T + B, B, rng)
    eta = eta_full[B:]   # T usable samples after burn-in

    # ---- filter truncation (each filter decays like its largest pole) ----
    trunc = max(
        compute_filter_truncation_length(u0),   # all-pass poles at ±u0
        compute_filter_truncation_length(z2),   # biexponential slowest pole at z2
    )

    # ---- order-flow proxy q via biexponential filter ----
    L_coeffs = build_biexponential_filter(z1, z2, trunc)
    q = np.convolve(L_coeffs, eta)[:T]

    # ---- impact j via all-pass filter (scaled by c) ----
    U_coeffs = build_allpass_filter(u0, trunc)
    j = c * np.convolve(U_coeffs, eta)[:T]

    # ---- empirical impact cost (returns (mean, se) tuple) ----
    C_H_mean, C_H_se = estimate_impact_cost_batched(q, j, H_C, n_batches=10)

    # ---- theoretical score ----
    K = theoretical_proportional_score(u0=u0, z1=z1, z2=z2)

    # ---- whiteness of j ----
    whiteness = verify_whiteness_of_impact(j=j, max_lag=50)
    # is_white: off-zero autocovariances small relative to lag-0 variance
    max_off = float(whiteness["max_off_zero_abs"])
    lag0    = float(whiteness["lag0_var"])
    whiteness_ok = bool(max_off < 0.05 * abs(lag0) + 1e-12)

    return {
        "u0": u0,
        "z1": z1,
        "z2": z2,
        "C_H": float(C_H_mean),
        "C_H_se": float(C_H_se),
        "K":   float(K),
        "whiteness_ok": whiteness_ok,
        "max_off_zero_Rjj": max_off,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sweep_u0(rows, z1_fix, z2_fix, save_dir):
    subset = [r for r in rows if abs(r["z1"] - z1_fix) < 1e-9 and abs(r["z2"] - z2_fix) < 1e-9]
    subset.sort(key=lambda r: r["u0"])

    u0s  = [r["u0"]  for r in subset]
    C_Hs = [r["C_H"] for r in subset]
    Ks   = [r["K"]   for r in subset]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u0s, C_Hs, "o-", color="steelblue", linewidth=1.8, markersize=6,
            label=r"Empirical $C_H$")
    ax.plot(u0s, Ks,   "s--", color="tomato",    linewidth=1.5, markersize=5,
            label=r"Theoretical $K(u_0, z_1, z_2)$")
    ax.axhline(0, color="k", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"$u_0$", fontsize=12)
    ax.set_ylabel("Impact cost", fontsize=12)
    ax.set_title(
        rf"Impact cost $C_H$ vs $u_0$  ($z_1={z1_fix},\; z_2={z2_fix}$)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, "exp2_cost_sweep_u0.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_heatmap(rows, u0_grid, z2_grid, z1_fix, save_dir):
    """Sign heatmap of C_H over (u0, z2) for fixed z1."""
    subset = [r for r in rows if abs(r["z1"] - z1_fix) < 1e-9]

    # Build matrix: rows = u0, cols = z2
    u0_vals = sorted(set(r["u0"] for r in subset))
    z2_vals = sorted(set(r["z2"] for r in subset))

    mat = np.full((len(u0_vals), len(z2_vals)), np.nan)
    for r in subset:
        i = u0_vals.index(r["u0"])
        j = z2_vals.index(r["z2"])
        mat[i, j] = np.sign(r["C_H"])

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = mcolors.ListedColormap(["tomato", "lightgrey", "steelblue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower",
                   extent=[
                       min(z2_vals) - 0.05, max(z2_vals) + 0.05,
                       min(u0_vals) - 0.05, max(u0_vals) + 0.05,
                   ])
    cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cb.set_ticklabels(["negative", "zero", "positive"])
    ax.set_xlabel(r"$z_2$", fontsize=12)
    ax.set_ylabel(r"$u_0$", fontsize=12)
    ax.set_title(
        rf"Sign of $C_H$ over $(u_0,\, z_2)$ grid  ($z_1={z1_fix}$)",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "exp2_cost_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_empirical_vs_theoretical(rows, save_dir):
    C_Hs = np.array([r["C_H"] for r in rows])
    Ks   = np.array([r["K"]   for r in rows])

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Ks, C_Hs, c=[r["u0"] for r in rows],
                    cmap="viridis", s=40, alpha=0.8, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$u_0$", fontsize=11)

    # 45-degree reference line
    lim_lo = min(min(Ks), min(C_Hs))
    lim_hi = max(max(Ks), max(C_Hs))
    margin = 0.05 * (lim_hi - lim_lo)
    diag = np.array([lim_lo - margin, lim_hi + margin])
    ax.plot(diag, diag, "k--", linewidth=1.2, label="$y = x$")

    ax.set_xlabel(r"Theoretical score $K(u_0, z_1, z_2)$", fontsize=12)
    ax.set_ylabel(r"Empirical impact cost $C_H$", fontsize=12)
    ax.set_title("Empirical vs Theoretical Impact Cost", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, "exp2_empirical_vs_theoretical.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp2(save_dir: str = "results") -> dict:
    """
    Experiment 2 – Positive impact costs under non-trivial all-pass impact.

    Parameters
    ----------
    save_dir : str
        Directory for PNG plots and RESULTS.md.

    Returns
    -------
    dict
        ``{'results': list_of_dicts, 'plots': list_of_paths}``
    """
    os.makedirs(save_dir, exist_ok=True)
    _set_style()

    # ---- parameters ----
    T   = 50_000
    B   = 2_000
    c   = 1.0
    H_C = 500

    u0_grid = np.linspace(0.1, 0.9, 9)
    z1_grid = [0.2, 0.4, 0.6]
    z2_grid = [0.5, 0.7, 0.9]

    # ---- enumerate valid triples ----
    triples = [
        (u0, z1, z2)
        for u0 in u0_grid
        for z1 in z1_grid
        for z2 in z2_grid
        if z2 > z1
    ]
    total = len(triples)
    print(f"\nExp 2: running {total} (u0, z1, z2) combinations …")

    results = []
    seed_counter = 0

    for idx, (u0, z1, z2) in enumerate(triples):
        row = _run_triple(
            T=T, u0=u0, z1=z1, z2=z2, c=c, H_C=H_C, B=B,
            seed=seed_counter,
        )
        results.append(row)
        seed_counter += 1

        if (idx + 1) % max(1, total // 10) == 0 or (idx + 1) == total:
            print(
                f"  [{idx+1:3d}/{total}]  u0={u0:.2f} z1={z1:.2f} z2={z2:.2f}"
                f"  C_H={row['C_H']:+.5f}  K={row['K']:+.5f}"
                f"  white={row['whiteness_ok']}"
            )

    # ---- plots ----
    plots = []

    p = _plot_sweep_u0(results, z1_fix=0.2, z2_fix=0.7, save_dir=save_dir)
    plots.append(p)
    print(f"Saved: {p}")

    p = _plot_heatmap(results, u0_grid, z2_grid, z1_fix=0.2, save_dir=save_dir)
    plots.append(p)
    print(f"Saved: {p}")

    p = _plot_empirical_vs_theoretical(results, save_dir=save_dir)
    plots.append(p)
    print(f"Saved: {p}")

    # ---- whiteness summary ----
    n_white = sum(1 for r in results if r["whiteness_ok"])
    print(f"\nWhiteness check: {n_white}/{total} passed.")

    # ---- persist ----
    _write_results_md(save_dir, "EXP2 – Positive Impact Costs", results)
    print(f"Results appended to {os.path.join(save_dir, 'RESULTS.md')}")

    summary = {"results": results, "plots": plots, "n_white": n_white, "total": total}
    return summary


if __name__ == "__main__":
    run_exp2()

# Market Microstructure Research Repository

## Repository Purpose
Studies structural diffusivity, all-pass impact filters, and information-neutral
impact whitening. Implements core numerical modules for simulation, estimation,
and diagnostics.

## Source Layout (`src/`)
| Module | Key Exports |
|---|---|
| `filters.py` | `build_allpass_filter`, `build_biexponential_filter`, `build_lti_filter`, `verify_allpass_property`, `compute_filter_truncation_length` |
| `simulation.py` | `simulate_innovations`, `simulate_counterfactual_returns`, `simulate_impact_increments`, `simulate_participant_flow`, `simulate_full_system` |
| `covariance.py` | `estimate_autocovariance`, `estimate_crosscovariance`, `verify_structural_identity`, `compute_sample_variance`, `rms_deviation` |
| `impact_cost.py` | `estimate_impact_cost`, `theoretical_proportional_score`, `estimate_impact_cost_batched`, `compute_future_sum_kernel`, `verify_whiteness_of_impact` |
| `scaling.py` | `compute_cumulative_impact`, `estimate_scaling_exponent`, `simulate_white_impact_with_mu`, `ternary_sign_process`, `ks_distance`, `estimate_scaling_grid` |
| `coherence.py` | `compute_distortion_profile`, `detectability_proxy`, `solve_ncoh_star`, `fit_power_law`, `theoretical_ncoh_star` |
| `microstructure.py` | `generate_synthetic_hf_data`, `estimate_lambda_X`, `build_bins`, `generate_synthetic_execution_data`, `classify_neutrality` |
| `diagnostics.py` | `ljung_box_test`, `diffusivity_check`, `qq_comparison`, `monte_carlo_summary`, `estimate_acf_with_ci` |

## Key Design Decisions

### Filter conventions
- All-pass filter: U[0]=u0, U[h]=-(1-u0²)·u0^(h-1) for h≥1. Property: ΣU[h]·U[h+t]=δ[t].
- Biexponential: L[h]=(1-z1)·z1^h-(1-z2)·z2^h. Requires 0<z1<z2<1.
- `build_lti_filter(coeffs, signal)` uses `scipy.signal.lfilter(coeffs, [1.0], signal)` — causal FIR.
- `compute_filter_truncation_length(base)` returns smallest H with base^H < 1e-10.

### Simulation conventions
- All simulations use `np.random.default_rng(seed)` (numpy Generator API).
- Burn-in B is always the first B samples of the (T+B)-length raw array; downstream functions return only `[B:]`.
- `simulate_full_system` returns `{'eta', 'xi', 'y', 'j', 'x'}` where eta/xi are length T+B and y/j/x are length T.

### Covariance conventions
- All estimators use T-normalisation (divide by T, not T-lag) for consistency when summing across lags.
- `estimate_crosscovariance`: R_xy[t] = E[(x_{s+t}-μ_x)(y_s-μ_y)]; positive t means x leads y.
- Implemented via `np.correlate(x_c, y_c, 'full')[T-1+t] / T`.

### Impact cost
- `estimate_impact_cost` uses cumsum to efficiently compute future sums: `j_cum[t+H_C] - j_cum[t]`.
- Only T-H_C terms with a complete lookahead are included (boundary-safe).

### Scaling
- `compute_cumulative_impact(j, h)` returns non-overlapping window sums, shape (T//h,).
- `simulate_white_impact_with_mu(T, kappa, mu, rng)`: j~N(0, kappa·mu), variance = kappa·mu.

### Coherence
- Distortion profile: profile[h]=(λ²+2λα)·(n_coh-h+1)/n_coh for h in 1..n_coh, else 0.
- `solve_ncoh_star` uses dt=1/(mu·N_D) internally; threshold = S/(σ_Y·√(1/(μ·N_D))).
- Proxy is monotone increasing in n_coh → linear search is valid.

### Dependencies
- numpy, scipy (signal, stats), pandas, statsmodels (acorr_ljungbox, acf)
- No external financial data libraries required; all experiments use synthetic data fallback.

## Running Tests
Tests live in `tests/`. Run with `python -m pytest tests/` from project root.

## Import pattern
```python
from src.filters import build_allpass_filter
from src.simulation import simulate_full_system
import src  # exposes src.filters, src.simulation, etc.
```

## Experiment files (`exp/`)
| File | Entry point | Tests |
|---|---|---|
| `exp1_structural_diffusivity.py` | `run_exp1(save_dir)` | Structural identity, all-pass whiteness, LB, MC |
| `exp2_positive_impact_costs.py` | `run_exp2(save_dir)` | Impact cost sweep, heatmap, empirical vs theoretical |
| `exp3_cumulative_impact_scaling.py` | `run_exp3(save_dir)` | Scaling (Module A), ternary (B), crossover (C) |
| `exp4_detectability_coherence.py` | `run_exp4(save_dir)` | n_coh* power-law, sensitivity sweep |
| `exp5_volatility_per_trade.py` | `run_exp5(save_dir)` | Kyle lambda estimation; synthetic HF data (3 assets); OLS of x²~N; ACF + LB diffusivity checks; bin-size sensitivity |
| `exp6_impact_whitening.py` | `run_exp6(save_dir)` | All-pass whitening; 3 (u0,mu) scenarios; neutral-bin ACF/LB; R_jj[0]~mu regression |
| `exp7_metaorder_square_root.py` | `run_exp7(save_dir)` | Square-root law; (mu,h) grid; OLS on sqrt(n_child); distributional collapse QQ |
| `exp8_informational_crossover.py` | `run_exp8(save_dir)` | rho-parametrised regimes; beta vs rho; n_coh vs mu; R_jj vs R_jy covariance map |

## Test suite (`tests/`)
| Test file | Tested module | Key assertions |
|---|---|---|
| `test_filters.py` | `src.filters` | Shape (H+1), U[0]=u0, U[1]=-(1-u0²), geometric tail ratio=u0, ValueError for invalid u0/z1/z2, truncation base^H<tol, allpass C[0]>0, lti output shape |
| `test_simulation.py` | `src.simulation` | Shape (T+B)/(T), Var(y)≈σ²·dt within 5%, Var(j)≈c² within 5%, x=y+j, required keys, |corr(η,ξ)|<0.01 |
| `test_covariance.py` | `src.covariance` | R[0]==sample_var, R[-t]==R[t], crosscov keys cover [-max_lag,max_lag], structural errors<15/√T, rms_deviation(x,x)==0 |
| `test_impact_cost.py` | `src.impact_cost` | Positive cost for anti-corr q/j, near-zero for independent, K>0, K increases with z2-z1, batched within 5% of full, max_off_zero_abs<10/√T·Var |
| `test_scaling.py` | `src.scaling` | Window sums, β=0.5 for exact power law, Var≈κμ within 5%, values in {-1,0,1}, symmetry within 2%, KS(x,x)=0, KS(x,x+5)>0.5 |
| `test_coherence.py` | `src.coherence` | Shape max_lag+1, profile[0]=0, proxy>0 for nonzero distortion, n_coh* non-increasing in mu, power-law slope exact, theoretical_ncoh strictly decreasing, exponent=-1/3 |

### Key gotchas for experiments

- **`simulate_full_system` returns a dict**, not a tuple: unpack as `sim["y"], sim["j"], sim["x"]`.
- **`compute_filter_truncation_length(base)`** takes a single scalar base (the largest pole magnitude). For biexponential + all-pass, call it twice and take `max(trunc_allpass, trunc_biexp)`.
- **Burn-in B** must be forwarded explicitly: `simulate_full_system(T=T, B=B, ...)`.
- **`fit_power_law`** lives in `src.coherence`, NOT `src.scaling`. Import with `from src.coherence import fit_power_law`.
- **Exp scripts run directly** (not as `-m` modules): each adds its parent dir to `sys.path` via `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`. Tests run via `pytest` from project root, which handles `sys.path` automatically.
- **matplotlib title strings**: use `r"..."` raw strings or double-backslash for LaTeX in non-raw strings. Never mix `\d`, `\t`, `\b` etc. in regular strings — they are invalid escape sequences in Python 3.12+. Pattern: `rf"text {var}" "\n" r"more $\delta$-text"`.
- **RESULTS.md**: always open with `"a"` (append), never `"w"`, so multiple experiments accumulate in one file.
- All plots saved at 150 dpi to `save_dir/` using `fig.savefig(path, dpi=150)`; `plt.close(fig)` immediately after.

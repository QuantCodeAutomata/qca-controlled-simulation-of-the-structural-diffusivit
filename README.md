# Structural Diffusivity and Information-Neutral Impact Whitening

Numerical reproduction of the theoretical framework for participant-level
market impact described in the paper on structural diffusivity and all-pass
impact filters.  All experiments are self-contained Python simulations; **no
live market-data API access is required**.

---

## Research Objectives

The eight experiments cover:

| # | Title | Type |
|---|-------|------|
| 1 | Structural diffusivity identity & information-neutral whitening | Synthetic |
| 2 | Positive impact costs under nontrivial all-pass impact | Synthetic |
| 3 | Diffusive cumulative impact, metaorder square-root scaling, & informational crossover | Synthetic |
| 4 | Detectability and coherence-length bound | Semi-analytic |
| 5 | Volatility-per-trade scaling & efficient-price diffusivity baseline | Synthetic HF proxy |
| 6 | Information-neutral impact whitening & participation-based variance scaling | Synthetic HF proxy |
| 7 | Diffusive cumulative impact, metaorder square-root law, child-order count dominance | Synthetic HF proxy |
| 8 | Informational crossover, coherence-length scaling, pure-surprise vs all-pass models | Synthetic HF proxy |

---

## Repository Layout

```
.
├── data/           # No raw data committed; see data/README.md
├── exp/            # Runnable experiment scripts (exp1_*.py … exp8_*.py)
├── src/            # Core library modules
│   ├── filters.py          # All-pass, biexponential, and generic LTI filters
│   ├── simulation.py       # Synthetic time-series generators
│   ├── covariance.py       # Auto/cross-covariance estimators & identity checker
│   ├── impact_cost.py      # Impact cost functionals
│   ├── scaling.py          # Cumulative-impact & scaling-exponent utilities
│   ├── coherence.py        # Detectability proxy & coherence-length solver
│   ├── microstructure.py   # Synthetic HF market-data generators (exp 5–8)
│   └── diagnostics.py      # ACF, Ljung-Box, diffusivity, Monte-Carlo helpers
├── tests/          # pytest suite (49 tests)
├── results/        # Saved plots and RESULTS.md summary
└── AGENTS.md       # Persistent AI-agent memory for this repo
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full test suite
pytest tests/ -v

# Run all eight experiments (writes plots + metrics to results/)
python exp/exp1_structural_diffusivity.py
python exp/exp2_positive_impact_costs.py
python exp/exp3_cumulative_impact_scaling.py
python exp/exp4_detectability_coherence.py
python exp/exp5_volatility_per_trade.py
python exp/exp6_impact_whitening.py
python exp/exp7_metaorder_square_root.py
python exp/exp8_informational_crossover.py
```

---

## Key Theoretical Elements

### All-pass impact filter (paper Eq. 3.3)

```
U[0]   =  u₀
U[h]   = -(1-u₀²) u₀^(h-1)   for h ≥ 1,   0 < u₀ < 1
```

Whiteness property: Σ_{h≥0} U[h] U[h+t] = δ[t]

### Structural covariance identity (paper Eq. 2.2)

```
R_yj[t] + R_jy[t] + R_jj[t]  =  Δt (σ_X² - σ_Y²) δ[t]
```

In the information-neutral regime (R_jy = R_yj = 0) this simplifies to
R_jj[t] = (σ_X² - σ_Y²) Δt δ[t], confirming that impact increments are white.

### Informational crossover (Exp. 3 Module C)

Impact variance is modelled as:

```
j_t = (1-ρ)·√(κμ)·ζ_t  +  ρ·κμ·a_t
```

The scaling exponent β = d log Std[j] / d log μ goes from **½ at ρ=0**
(information-neutral, square-root law) to **1 at ρ=1** (fully informed,
linear impact), reproducing the paper's predicted crossover.

### Coherence-length bound (Exp. 4)

```
n_coh* ~ C · μ^(-1/3),    C = (S / (λ + 2α))^(2/3)
```

Higher participation ⟹ shorter detectable coherence horizon.

---

## Results

Experiment outputs (PNG plots, CSV tables, metric summaries) are stored in
[`results/RESULTS.md`](results/RESULTS.md).

---

## Dependencies

See `requirements.txt`.  Core: `numpy`, `scipy`, `pandas`, `matplotlib`,
`seaborn`, `statsmodels`.

---

## Citation / Methodology

All formulas follow the paper's notation exactly.  Custom implementations are
annotated with `# Custom — Context7 found no library equivalent (paper Eq. X.X)`.
Standard library wrappers cite the upstream package in a comment.

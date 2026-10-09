# Data Directory

All experiments in this repository are **fully synthetic** (experiments 1–4, 6–8)
or use synthetic market-microstructure surrogates (experiment 5).

No external datasets are required or downloaded.  All time-series inputs are
generated at runtime by the simulation modules in `src/` using NumPy's
reproducible random-number generators with fixed seeds.

## Empirical experiments (5–8)

Experiments 5–8 include synthetic high-frequency market-data proxies that
replicate the statistical structure described in the paper (midpoint returns,
participant flow, trade counts, spread).  If you have access to real TAQ-style
trade/quote data you can substitute it by implementing the loader interfaces
documented in `src/microstructure.py`.

Real Massive-API data fetches are **never committed** to version control;
add any downloaded files to `.gitignore` before staging.

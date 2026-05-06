# INFO5574 Final Project — NFLX vs Gold (GLD) in the Tails

This project studies whether **Netflix (NFLX)** and **gold (proxied by GLD)** become more “related” during **extreme market stress** than they look during normal times.

The key idea is that standard correlation/regression mostly reflects “average days”, while risk in real markets is driven by **tail events**. We therefore focus on **left-tail crash days** and model dependence using **copulas** (to capture non-linear / tail dependence).

## Team workflow (3 members; Modules A/B/C)

- **Module A (daily preprocessing & merge)**: Clean the two raw daily datasets (NFLX and GLD), align trading days via an **inner join on `Date`**, address missingness / calendar mismatches, and export daily aligned **prices** and **log returns** for downstream analysis.
- **Module B (statistics, hypothesis testing, tail risk)**: Run descriptive statistics, correlation (Pearson/Spearman), OLS regression (with coefficient t-tests using robust standard errors), then compare dependence in the **left tail** (worst 5% NFLX days) using copulas and joint tail probability summaries.
- **Module C (visualization & insights, daily)**: Produce daily visuals (normalized dual-axis trend, returns scatter + regression line, correlation heatmap, rolling correlation, and tail-day highlighting) to support the “average regime vs extreme regime” narrative.

## Data sources (2 independent datasets)

- **Gold proxy: GLD (SPDR® Gold Shares)**
  - **File:** `navhist-us-en-gld(navhist).csv`
  - **Source:** State Street Global Advisors (historical NAV)
  - **Why GLD:** It trades on the US calendar, aligning naturally with US equities.

- **Netflix equity: NFLX**
  - **File:** `NFLX.csv`
  - **Source:** Kaggle (daily OHLCV)
  - **Field used:** `Adj Close`

## Method of combining datasets + missing values

- Parse both datasets as **daily time series**.
- Keep `Date` + one price column from each dataset.
- **Inner join** on `Date` to align trading days (this automatically handles missing dates due to holidays / data gaps).

## Transformations (tail-risk focus)

- Convert prices to **daily log returns**:
  - `NFLX_logret = log(NFLX_AdjClose).diff()`
  - `GLD_logret  = log(GLD_NAV).diff()`
- Define **crash days** as the worst **5%** of NFLX daily log returns (left tail).

## Technique choice + assumptions (what we fit)

### Baseline (average-day) modeling

- **Correlation** (Pearson + Spearman) on daily returns
- **OLS regression** (with robust standard errors):
  - `NFLX_logret ~ 1 + GLD_logret`

Assumptions (OLS, in practice for returns):
- Linearity in parameters and correct model specification
- Errors are mean-zero and (for classical t-tests) homoskedastic/independent; we use **HC1 robust SE** to relax homoskedasticity

### Tail dependence (non-linear dependence)

- **Copula models** on the empirical CDF–transformed returns (rank-based uniforms), comparing:
  - Gaussian copula (no tail dependence)
  - Student-t copula (symmetric tail dependence)
  - Clayton (lower-tail dependence)
  - Gumbel (upper-tail dependence)

Interpretation:
- If tail-focused copulas fit better (lower AIC) and joint tail probabilities differ across tails, that supports a **“doom correlation”** narrative (dependence changes in extremes).

## How to run (B module: statistics + hypothesis tests)

Run the analysis script:

```bash
py b_module_stats_tail.py
```

Outputs are written to `outputs/`:
- `combined_daily_prices.csv`
- `combined_daily_logreturns.csv`
- `descriptive_stats.csv`
- `correlation_report.csv`
- `tail_correlation_report.csv`
- `ols_summary.txt` (includes coefficient t-tests)
- `copula_fit_full.json`
- `copula_fit_tail.json`
- `joint_tail_probabilities.csv`

## Notebooks (all 3 modules)

- `notebooks/01_preprocess_merge_daily.ipynb` — Module A (daily cleaning + merge + export)
- `notebooks/02_stats_tail_copula.ipynb` — Module B (daily returns + tail risk + copulas)
- `notebooks/03_viz_insights_daily.ipynb` — Module C (daily visuals: normalized trend / returns scatter+fit / heatmap / rolling corr / tail highlight)

Recommended run order: **A → B → C**

## Suggested write-up structure (paper/presentation)

- **EDA**: show distributions of returns + summary stats
- **Average-day results**: correlation + OLS fit (and limitations)
- **Tail results**: compare tail vs full-sample correlation; report joint tail probabilities
- **Copula insight**: best-fit copula by AIC; explain what that implies about tail dependence
- **Conclusion**: what happens “on normal days” vs “on crash days”

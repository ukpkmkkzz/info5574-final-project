# INFO5574 Final Project Paper  
**Title:** Doom Correlation in the Left Tail: Copula-Based Dependence Between Netflix (NFLX) and Gold (GLD)  
**Authors:** <Name 1>, <Name 2>, <Name 3>  
**Date:** May 2026  

## Background & hypothesis

Financial risk is driven disproportionately by extreme events. Relationships that appear weak on “average days” can change abruptly during market stress, which matters for hedging, diversification, and stress testing.

This project asks whether **Netflix (NFLX)** and **gold** (proxied by **SPDR Gold Shares, GLD**) exhibit a “doom correlation” regime: **near-independence in normal markets but changed dependence in the left tail** during severe equity selloffs.

We test the following hypotheses on **daily log returns**:

- **H0 (no tail dependence):** Dependence between NFLX and GLD is not meaningfully different in extreme left-tail NFLX days compared to the full sample.
- **H1 (doom correlation / tail dependence):** Dependence changes in the left tail (e.g., stronger negative association or asymmetric joint tail behavior).

## Data sources (two independent datasets)

1) **NFLX daily prices (Adjusted Close used)**  
- Source: Kaggle (Netflix daily OHLCV dataset)  
- File: `NFLX.csv`

2) **GLD historical NAV (gold proxy)**  
- Source: State Street Global Advisors (historical NAV export)  
- File: `navhist-us-en-gld(navhist).csv`

We use **GLD rather than spot gold** to match the US trading calendar of equities.

## Methodology

### Data preparation

- Parse both datasets as daily time series and convert price columns to numeric.
- Keep `Date` and one price column from each series: `NFLX_AdjClose` and `GLD_NAV`.
- Align calendars via an **inner join on `Date`** (keeps only days where both markets trade).
- Transform prices to daily **log returns**:
  - `NFLX_logret = log(NFLX_AdjClose).diff()`
  - `GLD_logret  = log(GLD_NAV).diff()`

### Tail-risk definition

Define “crash days” as the **worst 5%** of NFLX daily log returns (left tail). This isolates extreme NFLX downside events and allows a direct comparison of dependence in:
- **Full sample** vs  
- **Left-tail subsample** (bottom 5% of NFLX days)

### Models and tests

We treat this as an **explanatory analysis** (not a predictive model).

1) **Correlation (baseline dependence)**  
- Pearson correlation and Spearman rank correlation between `NFLX_logret` and `GLD_logret` (full sample and left tail).

2) **OLS regression (average-day linear relationship)**  
\[
\text{NFLX\_logret} = \alpha + \beta \cdot \text{GLD\_logret} + \varepsilon
\]
We report coefficient tests using **HC1 robust standard errors** to reduce sensitivity to heteroskedasticity (common in financial returns).

3) **Copula comparison (non-linear / tail dependence)**  
We compare dependence models on rank-transformed uniforms (empirical CDF) using:
- Gaussian copula (no tail dependence)
- Student-t copula (symmetric tail dependence)
- Archimedean copulas when parameter ranges are valid (data-driven; not always applicable)

## Results

### Data description and summary statistics

After aligning on shared trading days and computing log returns, we have:
- **Daily prices:** 4,793 matched dates  
- **Daily log returns:** **n = 4,792** (after 1-day differencing)

Key summary statistics (n = 4,792):

| Variable | Mean | Std | Min | Max | Skew | Excess Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| NFLX_logret | 0.001166 | 0.03250 | -0.4326 | 0.3522 | -0.8053 | 23.2399 |
| GLD_logret | 0.000302 | 0.01112 | -0.0960 | 0.0684 | -0.3668 | 5.2120 |

### Correlation (full sample vs left tail)

- **Full sample:** Pearson \(r = 0.0083\) (p = 0.5655), Spearman \(\rho = -0.0104\) (p = 0.4696)  
- **Left tail (NFLX bottom 5%, n = 240):** Pearson \(r = 0.0078\) (p = 0.9048), Spearman \(\rho = -0.0236\) (p = 0.7165)

Both full-sample and tail-sample correlations are near zero and not statistically significant.

### OLS regression (robust SE)

Model: `NFLX_logret ~ 1 + GLD_logret` (HC1 robust SE), n = 4,792

| Term | Coef | Robust SE | z | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| Intercept | 0.0012 | 0.0005 | 2.47 | 0.014 | [0.000, 0.002] |
| GLD_logret | 0.0243 | 0.0507 | 0.48 | 0.632 | [-0.075, 0.124] |

Interpretation:
- The estimated slope \(\beta\) is **not statistically significant** (p = 0.632).  
- Model fit is negligible (**R² ≈ 0.000**), implying GLD returns do not explain NFLX daily returns in a linear mean relationship.

### Copula and joint tail summaries

Copula fits (AIC comparison) produced **near-zero dependence** (estimated \(\rho \approx 0\)) in both:
- Full sample (n = 4,792)  
- Left-tail subsample (n = 240)

Empirical joint tail probabilities at q = 0.05:

| Event | Probability |
|---|---:|
| \(P(\text{NFLX}\le q,\,\text{GLD}\le q)\) | 0.00355 |
| \(P(\text{NFLX}\le q,\,\text{GLD}\ge 1-q)\) | 0.00376 |
| \(P(\text{NFLX}\ge 1-q,\,\text{GLD}\ge 1-q)\) | 0.00396 |
| \(P(\text{NFLX}\ge 1-q,\,\text{GLD}\le q)\) | 0.00230 |

These probabilities are small and do not indicate strong asymmetry consistent with a pronounced “doom correlation” in this pairing.

## Conclusions

Across daily data in this sample, we find **little evidence** that GLD and NFLX exhibit a strong average-day relationship or an intensified left-tail dependence:

- Correlations are near zero in both the full sample and the NFLX left tail.
- OLS shows the GLD coefficient is not significant and explains essentially none of the variation in NFLX daily returns.
- Copula comparisons also suggest dependence near zero.

This is informative: “safe haven” claims are often **asset- and regime-specific**. Gold may hedge broad equity stress in some contexts, but it does not necessarily hedge **single-name, idiosyncratic equity tail risk** (NFLX) in a stable way.

### Challenges and changes from the original plan

Our initial plan aggregated the data to annual averages, which produced too few observations for meaningful inference. We corrected this by:
- switching to **daily alignment**,
- modeling **daily log returns**, and
- explicitly comparing **full-sample** vs **tail-sample** dependence.

### Reproducibility (code + data)

- GitHub repo (code + notebooks + outputs): `https://github.com/ukpkmkkzz/info5574-final-project`  
- Key code:
  - `b_module_stats_tail.py`
  - `notebooks/01_preprocess_merge_daily.ipynb`
  - `notebooks/02_stats_tail_copula.ipynb`
  - `notebooks/03_viz_insights_daily.ipynb`
- Data files used:
  - `NFLX.csv`
  - `navhist-us-en-gld(navhist).csv`

## References (background reading)

- Longin, F., & Solnik, B. (2001). *Extreme correlation of international equity markets.* Journal of Finance.  
- Baur, D. G., & Lucey, B. M. (2010). *Is Gold a Hedge or a Safe Haven?* Financial Review.  
- Baur, D. G., & McDermott, T. K. (2010). *Is gold a safe haven?* International evidence.  
- Patton, A. J. (2006). *Modelling asymmetric exchange rate dependence.* International Economic Review.  
- Embrechts, P., McNeil, A., & Straumann, D. (2002). *Correlation and dependence in risk management: properties and pitfalls.*  

## Teammate rating (peer review)

- <Name 1>: <0–10>  
- <Name 2>: <0–10>  
- <Name 3>: <0–10>  


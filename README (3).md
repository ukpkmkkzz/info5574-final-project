# INFO5574 Final Project — updated README

This repository previously described an **annual-aggregation** workflow (too few samples for meaningful inference).

The project has been updated to a **daily, tail-risk** workflow (extreme events / “doom correlation”), with a full B-module analysis script that produces:
- Descriptive statistics
- Correlation (Pearson/Spearman)
- OLS regression with coefficient t-tests (robust SE)
- Tail (worst 5%) subsample analysis
- Copula-based dependence comparison (Gaussian vs Student-t, plus valid Archimedean cases)

Please see the main project README:

- `README.md`

Run the B-module analysis:

```bash
py b_module_stats_tail.py
```

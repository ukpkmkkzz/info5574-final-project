from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.distributions.copula.api import (
    ClaytonCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class Config:
    nflx_csv: Path = ROOT / "NFLX.csv"
    gld_csv: Path = ROOT / "navhist-us-en-gld(navhist).csv"
    tail_q: float = 0.05  # left-tail quantile for "crash days"


def _rank_to_uniform(x: pd.Series) -> np.ndarray:
    """
    Empirical CDF transform using ranks -> (0,1).
    Avoids exactly 0/1 which can break copula log-likelihoods.
    """
    r = st.rankdata(x.to_numpy(), method="average")
    n = len(r)
    return (r - 0.5) / n


def load_and_clean_nflx(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[["Date", "Adj Close"]].rename(columns={"Adj Close": "NFLX_AdjClose"})
    df["NFLX_AdjClose"] = pd.to_numeric(df["NFLX_AdjClose"], errors="coerce")
    df = df.dropna(subset=["NFLX_AdjClose"])
    return df


def load_and_clean_gld(path: Path) -> pd.DataFrame:
    # File has a few metadata rows; header row begins at line with "Date,NAV,..."
    # This export is sometimes not UTF-8; try common encodings for Windows/finance exports.
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, skiprows=3, encoding=enc)
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 - intentionally broad for encoding fallback
            last_err = e
    if last_err is not None:
        raise last_err
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[["Date", "NAV"]].rename(columns={"NAV": "GLD_NAV"})
    df["GLD_NAV"] = pd.to_numeric(df["GLD_NAV"], errors="coerce")
    df = df.dropna(subset=["GLD_NAV"])
    return df


def merge_daily(nflx: pd.DataFrame, gld: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(nflx, gld, on="Date", how="inner")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["NFLX_logret"] = np.log(out["NFLX_AdjClose"]).diff()
    out["GLD_logret"] = np.log(out["GLD_NAV"]).diff()
    out = out.dropna(subset=["NFLX_logret", "GLD_logret"]).reset_index(drop=True)
    return out


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["NFLX_AdjClose", "GLD_NAV", "NFLX_logret", "GLD_logret"]
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew(numeric_only=True)
    stats["kurtosis_excess"] = df[cols].kurtosis(numeric_only=True)
    return stats


def correlation_report(df: pd.DataFrame) -> pd.DataFrame:
    x = df["NFLX_logret"]
    y = df["GLD_logret"]
    pearson = st.pearsonr(x, y)
    spearman = st.spearmanr(x, y)
    return pd.DataFrame(
        [
            {"metric": "pearson_r", "value": pearson.statistic, "p_value": pearson.pvalue},
            {
                "metric": "spearman_rho",
                "value": spearman.statistic,
                "p_value": spearman.pvalue,
            },
        ]
    )


def ols_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    y = df["NFLX_logret"]
    X = sm.add_constant(df["GLD_logret"])
    model = sm.OLS(y, X, missing="drop")
    return model.fit(cov_type="HC1")  # robust SE (common for returns)


def tail_subset(df: pd.DataFrame, tail_q: float) -> pd.DataFrame:
    cutoff = df["NFLX_logret"].quantile(tail_q)
    return df[df["NFLX_logret"] <= cutoff].copy()


def fit_copulas(u: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    """
    Fit a few common copulas by MLE and return log-likelihood / AIC.
    Note: Copula fits here are empirical and meant for comparison/insight.
    """

    def safe_ll(cop, *args) -> float:
        dens = cop.pdf(np.column_stack([u, v]), *args)
        dens = np.clip(dens, 1e-300, np.inf)
        return float(np.sum(np.log(dens)))

    rows: list[dict] = []

    # Gaussian copula
    gc = GaussianCopula()
    gc.fit_corr_param(np.column_stack([u, v]))
    ll = safe_ll(gc)
    rho = float(np.asarray(gc.corr)[0, 1])
    rows.append({"copula": "gaussian", "params": {"rho": rho}, "loglik": ll, "k": 1})

    # Student t copula (estimate rho and df via crude search)
    best = None
    for df_ in [3, 4, 5, 7, 10, 15, 20, 30]:
        tc = StudentTCopula(df=df_)
        tc.fit_corr_param(np.column_stack([u, v]))
        ll_ = safe_ll(tc)
        cand = (ll_, df_, float(np.asarray(tc.corr)[0, 1]))
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None
    ll, df_, rho = best
    rows.append({"copula": "t", "params": {"rho": rho, "df": int(df_)}, "loglik": ll, "k": 2})

    # Archimedean copulas require valid theta ranges; use Kendall's tau sign to decide.
    tau = st.kendalltau(u, v).statistic

    # Clayton (lower-tail dependence), theta > 0
    if tau is not None and tau > 0:
        clay = ClaytonCopula()
        theta = float(clay.fit_corr_param(np.column_stack([u, v])))
        if theta > 0:
            ll = safe_ll(clay, theta)
            rows.append({"copula": "clayton", "params": {"theta": theta}, "loglik": ll, "k": 1})

    # Gumbel (upper-tail dependence), theta >= 1
    if tau is not None and tau > 0:
        gum = GumbelCopula()
        theta = float(gum.fit_corr_param(np.column_stack([u, v])))
        if theta >= 1:
            ll = safe_ll(gum, theta)
            rows.append({"copula": "gumbel", "params": {"theta": theta}, "loglik": ll, "k": 1})

    out = pd.DataFrame(rows)
    n = len(u)
    out["aic"] = 2 * out["k"] - 2 * out["loglik"]
    out["n"] = n
    return out.sort_values("aic").reset_index(drop=True)


def joint_tail_probabilities(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Empirical "joint tail" probabilities for left-tail and right-tail events.
    Useful for the 'asymmetry' narrative without overclaiming.
    """
    x = df["NFLX_logret"]
    y = df["GLD_logret"]
    xl = x.quantile(q)
    yl = y.quantile(q)
    xu = x.quantile(1 - q)
    yu = y.quantile(1 - q)

    n = len(df)
    out = pd.DataFrame(
        [
            {
                "event": f"P(NFLX<=q, GLD<=q) @ q={q:.2f}",
                "prob": float(((x <= xl) & (y <= yl)).mean()),
            },
            {
                "event": f"P(NFLX<=q, GLD>=1-q) @ q={q:.2f}",
                "prob": float(((x <= xl) & (y >= yu)).mean()),
            },
            {
                "event": f"P(NFLX>=1-q, GLD>=1-q) @ q={q:.2f}",
                "prob": float(((x >= xu) & (y >= yu)).mean()),
            },
            {
                "event": f"P(NFLX>=1-q, GLD<=q) @ q={q:.2f}",
                "prob": float(((x >= xu) & (y <= yl)).mean()),
            },
        ]
    )
    out["n"] = n
    return out


def main() -> None:
    cfg = Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nflx = load_and_clean_nflx(cfg.nflx_csv)
    gld = load_and_clean_gld(cfg.gld_csv)
    merged = merge_daily(nflx, gld)
    merged_ret = add_returns(merged)

    merged.to_csv(OUT_DIR / "combined_daily_prices.csv", index=False)
    merged_ret.to_csv(OUT_DIR / "combined_daily_logreturns.csv", index=False)

    desc = descriptive_stats(merged_ret)
    desc.to_csv(OUT_DIR / "descriptive_stats.csv")

    corr = correlation_report(merged_ret)
    corr.to_csv(OUT_DIR / "correlation_report.csv", index=False)

    res = ols_regression(merged_ret)
    (OUT_DIR / "ols_summary.txt").write_text(res.summary().as_text(), encoding="utf-8")

    tail = tail_subset(merged_ret, cfg.tail_q)
    tail_corr = correlation_report(tail)
    tail_corr.to_csv(OUT_DIR / "tail_correlation_report.csv", index=False)

    # Copula fit on full sample and tail sample (empirical CDF transform)
    u = _rank_to_uniform(merged_ret["NFLX_logret"])
    v = _rank_to_uniform(merged_ret["GLD_logret"])
    cop_full = fit_copulas(u, v)
    cop_full.to_json(OUT_DIR / "copula_fit_full.json", orient="records", indent=2)

    ut = _rank_to_uniform(tail["NFLX_logret"])
    vt = _rank_to_uniform(tail["GLD_logret"])
    cop_tail = fit_copulas(ut, vt)
    cop_tail.to_json(OUT_DIR / "copula_fit_tail.json", orient="records", indent=2)

    joint = joint_tail_probabilities(merged_ret, cfg.tail_q)
    joint.to_csv(OUT_DIR / "joint_tail_probabilities.csv", index=False)

    # Small console print for quick verification
    print("Wrote outputs to:", OUT_DIR)
    print("Merged days:", len(merged), "Return days:", len(merged_ret), "Tail days:", len(tail))
    print("OLS beta(GLD_logret):", res.params.get("GLD_logret", math.nan))


if __name__ == "__main__":
    main()


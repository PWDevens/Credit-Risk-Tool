"""Feature diagnostics — the evidence step before more feature engineering.

(1) Information Value (IV) of the engineered features.
(2) Information Value of the base features.
(3) Collinearity: Spearman |rho|>=0.8 pairs + VIF for the engineered numerics.

IV is model-free predictive power (credit-industry standard):
    <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 suspicious/leakage
Run:  python modeling/diagnose_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

import features as F  # noqa: E402
from common import data as D  # noqa: E402

TARGET = F.PD_TARGET


def iv_of(df: pd.DataFrame, col: str, bins: int = 10) -> float:
    """Total Information Value of one column vs the binary target."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        key = pd.qcut(s, q=bins, duplicates="drop").astype("object").where(s.notna(), "<NA>")
    else:
        key = s.astype("object").where(s.notna(), "<NA>")
    g = df.groupby(key, observed=True)[TARGET]
    tab = pd.DataFrame({"n": g.size(), "bad": g.sum()})
    tab["good"] = tab["n"] - tab["bad"]
    dist_bad = (tab["bad"] + 0.5) / tab["bad"].sum()
    dist_good = (tab["good"] + 0.5) / tab["good"].sum()
    woe = np.log(dist_good / dist_bad)
    return float(((dist_good - dist_bad) * woe).sum())


def band(iv: float) -> str:
    if pd.isna(iv):
        return "?"
    return ("useless" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3
            else "strong" if iv < 0.5 else "suspicious")


def iv_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for c in cols:
        try:
            iv = iv_of(df, c)
        except Exception:
            iv = np.nan
        rows.append({"feature": c, "IV": round(iv, 4) if iv == iv else iv, "band": band(iv)})
    return pd.DataFrame(rows).sort_values("IV", ascending=False, na_position="last")


def main() -> None:
    df = D.load_frame("train")
    print(f"train rows={len(df):,}  bad_rate={df[TARGET].mean():.4f}")

    eng = F._ENGINEERED_NUMERIC + ["RiskCluster"]
    print("\n=============== (1) ENGINEERED FEATURES — IV ===============")
    print(iv_table(df, eng).to_string(index=False))

    print("\n=============== (2) BASE FEATURES — IV ===============")
    tb = iv_table(df, F.MODEL_FEATURES)
    print("top 20:")
    print(tb.head(20).to_string(index=False))
    print("\nbottom 8 (weakest base features):")
    print(tb.tail(8).to_string(index=False))

    print("\n=============== (3) COLLINEARITY ===============")
    num = [c for c in (F.MODEL_FEATURES + F._ENGINEERED_NUMERIC) if c not in F.CATEGORICAL_FEATURES]
    X = df[num].apply(pd.to_numeric, errors="coerce")
    corr = X.corr(method="spearman")
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= 0.8:
                pairs.append((round(float(r), 3), cols[i], cols[j]))
    pairs.sort(key=lambda x: -abs(x[0]))
    print(f"Spearman |rho| >= 0.8 pairs ({len(pairs)}):")
    for r, a, b in pairs[:25]:
        print(f"  {r:+.3f}  {a}  <->  {b}")

    from sklearn.linear_model import LinearRegression
    Xi = X.fillna(X.median())
    Xi = (Xi - Xi.mean()) / Xi.std(ddof=0)
    Xi = Xi.loc[:, Xi.notna().all()]
    print("\nVIF for engineered numerics (>5 high, >10 severe multicollinearity):")
    for c in F._ENGINEERED_NUMERIC:
        if c not in Xi.columns:
            print(f"  {c:52s} VIF=   n/a")
            continue
        y = Xi[c].to_numpy()
        Xo = Xi.drop(columns=[c]).to_numpy()
        r2 = LinearRegression().fit(Xo, y).score(Xo, y)
        vif = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
        print(f"  {c:52s} VIF={vif:8.2f}")


if __name__ == "__main__":
    main()

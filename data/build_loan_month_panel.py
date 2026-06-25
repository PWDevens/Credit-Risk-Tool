"""Phase 2 — build the loan-month (person-period) panel for the discrete-time hazard model.

One row per loan per month it was alive. The target `defaulted_this_month` is 1 only in the
month a loan defaulted, else 0. This is the data substrate for a hazard model h(t|x): the chance
a loan defaults *in month t given it survived to t*. Building it this way also recovers the
`Current` loans v1 dropped — they enter honestly as *censored* (still-alive) observations.

See docs/04-discrete-time-hazard-model.md. Run:
    .venv\\Scripts\\python.exe data\\build_loan_month_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
import features as F  # noqa: E402

RAW_CSV = REPO_ROOT / "data" / "raw" / "prosperLoanData.csv"
PANEL_PATH = REPO_ROOT / "data" / "processed" / "loan_month_panel.parquet"

TIME_COLS = ["t", "t_frac"]
HAZARD_TARGET = "defaulted_this_month"
# Carried through but NOT covariates (used for splitting / labels / bookkeeping).
META_COLS = ["loan_id", "Term", "T_obs", "is_event", "LoanOriginationDate", "LoanOriginationQuarter"]
MONTHS_PER_DAY = 12.0 / 365.25


def covariate_cols() -> list:
    """Borrower covariates (origination-time, constant over a loan's months) + the time features.
    Macro is intentionally excluded from the base hazard model (see spec)."""
    return list(F.MODEL_FEATURES) + list(F._ENGINEERED_NUMERIC) + ["RiskCluster"] + TIME_COLS


def _observed_duration(df: pd.DataFrame) -> pd.Series:
    """Observed months T_obs. Closed loans: ClosedDate - origination (the true lifetime). Open /
    Current loans: months-since-origination at the data snapshot (the censoring time). Clipped to
    [1, Term]."""
    orig = pd.to_datetime(df["LoanOriginationDate"], errors="coerce")
    closed = pd.to_datetime(df["ClosedDate"], errors="coerce")
    months_to_close = ((closed - orig).dt.days * MONTHS_PER_DAY).round()
    snap_months = pd.to_numeric(df["LoanMonthsSinceOrigination"], errors="coerce")
    t_obs = months_to_close.where(closed.notna(), snap_months)
    return t_obs.clip(lower=1, upper=pd.to_numeric(df["Term"], errors="coerce"))


def build_panel() -> pd.DataFrame:
    print(f"Loading {RAW_CSV.name} ...")
    raw = pd.read_csv(RAW_CSV, low_memory=False)

    # --- population scope: post-2009 originations, drop Cancelled, need a usable Term ----------
    orig = pd.to_datetime(raw["LoanOriginationDate"], errors="coerce")
    term = pd.to_numeric(raw["Term"], errors="coerce")
    keep = (orig >= F.POST_2009_CUTOFF) & (raw["LoanStatus"] != "Cancelled") & (term >= 1)
    df = raw[keep].copy().reset_index(drop=True)
    df["loan_id"] = np.arange(len(df))
    df["is_event"] = df["LoanStatus"].isin(F.BAD_STATUSES)
    df["T_obs"] = _observed_duration(df).astype("Int64")
    df = df[df["T_obs"].notna() & (df["T_obs"] >= 1)].copy()
    df["T_obs"] = df["T_obs"].astype(int)
    df["Term"] = pd.to_numeric(df["Term"], errors="coerce").astype(int)

    # --- covariates (reuse the production feature path) ---------------------------------------
    df = F.add_derived_features(df)
    df = F.feature_engineering(df)
    if F.RISK_CLUSTER_PATH.exists():
        df["RiskCluster"] = F.assign_risk_cluster(df, joblib.load(F.RISK_CLUSTER_PATH))
    else:
        print("  WARNING: risk_cluster.joblib missing — building panel without RiskCluster.")
    df = F.cast_categoricals(df)

    cov = [c for c in covariate_cols() if c not in TIME_COLS and c in df.columns]

    # --- sanity checks -------------------------------------------------------------------------
    n_events = int(df["is_event"].sum())
    n_current = int((df["LoanStatus"] == "Current").sum())
    print(f"  loans in scope: {len(df):,} | events (Defaulted+Chargedoff): {n_events:,} | "
          f"Current (censored): {n_current:,}")
    closed = pd.to_datetime(df["ClosedDate"], errors="coerce")
    cm = ((closed - pd.to_datetime(df["LoanOriginationDate"], errors="coerce")).dt.days
          * MONTHS_PER_DAY).round()
    diff = (cm - pd.to_numeric(df["LoanMonthsSinceOrigination"], errors="coerce")).abs()
    # Informational only: for CLOSED loans, ClosedDate-origination (true lifetime) legitimately
    # differs from LoanMonthsSinceOrigination (age at the ~2014 snapshot) — a closed loan closed
    # before the snapshot. We deliberately use ClosedDate as the true duration.
    print(f"  (info) closed-loan median |ClosedDate-age - snapshot-age|: {np.nanmedian(diff):.1f} months")
    # Guardrails: events are the post-2009 Defaulted+Chargedoff loans (NOT the ~17k all-time count
    # — most defaults are pre-2009 originations, out of scope); Current loans recovered; and every
    # observed duration sits within [1, Term].
    assert n_events > 4000, f"expected a few thousand post-2009 events, got {n_events}"
    assert n_current > 1000, f"expected Current loans recovered as censored, got {n_current}"
    assert bool((df["T_obs"] >= 1).all() and (df["T_obs"] <= df["Term"]).all()), "T_obs out of [1, Term]"

    # --- expand to person-period rows ----------------------------------------------------------
    # dedupe columns: Term (and any overlap) lives in both the covariates and META_COLS.
    base = df[list(dict.fromkeys(cov + META_COLS))].reset_index(drop=True)
    panel = base.loc[base.index.repeat(base["T_obs"].to_numpy())].reset_index(drop=True)
    panel["t"] = panel.groupby("loan_id", sort=False).cumcount() + 1
    panel["t_frac"] = panel["t"] / panel["Term"]
    panel[HAZARD_TARGET] = ((panel["t"] == panel["T_obs"]) & panel["is_event"]).astype(int)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    pos = int(panel[HAZARD_TARGET].sum())
    print(f"\nPanel: {len(panel):,} loan-month rows | positives (default months): {pos:,} "
          f"| event rate: {pos / len(panel):.5f}")
    print(f"saved {PANEL_PATH}")
    return panel


if __name__ == "__main__":
    build_panel()

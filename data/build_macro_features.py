"""Pull point-in-time macro features (national unemployment + fed funds rate) from FRED.

Builds a monthly lookup keyed by year-month, joined onto each loan by LoanOriginationDate.
Macro conditions AT ORIGINATION are known at the underwriting decision, so this is
leakage-safe (a true point-in-time feature). Converts a static through-the-cycle PD toward
a point-in-time PD — the v3-plan macro overlay.

  UNRATE   = civilian unemployment rate, % (monthly)
  FEDFUNDS = effective federal funds rate, % (monthly)

Three sources, tried in order per series (the public fredgraph.csv endpoint is flaky and can
hang, so it's the last resort):
  1. LOCAL CSV  — data/raw/UNRATE.csv and data/raw/FEDFUNDS.csv. Most reliable: download each
     in your browser from
        https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE
        https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS
     (or the "Download -> CSV" button on each series page), save into data/raw/, then run this.
  2. FRED API   — set a free key:  $env:FRED_API_KEY = "your_key"   (get one at
     https://fredaccount.stlouisfed.org/apikeys). Uses the JSON API on api.stlouisfed.org.
  3. fredgraph.csv with a longer timeout + retries.

Run:  python data/build_macro_features.py
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MACRO_CSV = RAW_DIR / "macro_monthly.csv"
MACRO_FEATURES = ["macro_unemployment", "macro_fedfunds"]

# FRED series id -> our column name. Add state series (e.g. "CAUR","TXUR") later for a
# geography join on BorrowerState.
SERIES = {"UNRATE": "macro_unemployment", "FEDFUNDS": "macro_fedfunds"}
GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
API_URL = ("https://api.stlouisfed.org/fred/series/observations"
           "?series_id={}&api_key={}&file_type=json")


def _to_monthly(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df = df.iloc[:, :2].copy()
    df.columns = ["date", colname]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[colname] = pd.to_numeric(df[colname], errors="coerce")
    df["ym"] = df["date"].dt.to_period("M").astype(str)
    return df[["ym", colname]].dropna()


def _fetch_local(series_id: str, colname: str):
    path = RAW_DIR / f"{series_id}.csv"
    if not path.exists():
        return None
    return _to_monthly(pd.read_csv(path), colname)


def _fetch_api(series_id: str, colname: str, key: str):
    req = urllib.request.Request(API_URL.format(series_id, key),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        obs = pd.DataFrame(json.load(resp)["observations"])
    return _to_monthly(obs.rename(columns={"value": colname})[["date", colname]], colname)


def _fetch_graph(series_id: str, colname: str, retries: int = 3):
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(GRAPH_URL.format(series_id),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            return _to_monthly(pd.read_csv(io.StringIO(raw)), colname)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"    fredgraph attempt {attempt}/{retries} failed: {exc}")
    raise RuntimeError(f"fredgraph failed for {series_id}: {last}")


def _fetch(series_id: str, colname: str) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    sources = [("local CSV", lambda: _fetch_local(series_id, colname))]
    if key:
        sources.append(("FRED API", lambda: _fetch_api(series_id, colname, key)))
    sources.append(("fredgraph.csv", lambda: _fetch_graph(series_id, colname)))

    for src_name, fn in sources:
        try:
            t = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {series_id}: {src_name} failed ({exc})")
            continue
        if t is not None and len(t):
            print(f"  {colname} ({series_id}): {len(t):,} monthly obs via {src_name}")
            return t
    raise RuntimeError(
        f"Could not fetch {series_id} from any source. Easiest fix: open "
        f"{GRAPH_URL.format(series_id)} in your browser, save it as "
        f"data/raw/{series_id}.csv, and re-run.")


def build_macro_table() -> pd.DataFrame:
    macro = None
    for sid, col in SERIES.items():
        t = _fetch(sid, col)
        macro = t if macro is None else macro.merge(t, on="ym", how="outer")
    return macro.sort_values("ym")


def add_derived_columns(macro: pd.DataFrame) -> pd.DataFrame:
    """Add all derived macro columns (single source of truth, reproducible). Per raw series:

      TRAILING / point-in-time clean (safe as origination-time features):
        *_ma12 : 12-month trailing moving average (dampens one-month spikes)
        *_ewma : exponential moving average, halflife 6 months
        *_ttc  : TTC_WEIGHT*ewma + (1-TTC_WEIGHT)*long_run_mean (anchored toward the cycle mean)
        *_gap  : raw - long_run_mean (cyclical deviation from normal)
        *_yoy  : (raw - raw 12 months ago) / raw 12 months ago  (trailing year-over-year change)
        *_ln   : natural log of the level (helps LINEAR models; no effect on trees)

      CALENDAR-period (NOTE: forward-looking WITHIN the period — a Jan loan 'sees' the whole
      year/quarter, so these are mild look-ahead; prefer the trailing forms above for PIT):
        *_annual_avg / *_annual_change       : calendar-year mean and its YoY % change
        *_quarterly_avg / *_quarterly_change : calendar-quarter mean and its QoQ % change

    Long-run mean uses features.MACRO_LONGRUN_WINDOW. See docs/01-feature-engineering.md.
    """
    import features as F  # local import: only needed when this script is run

    macro = macro.sort_values("ym").reset_index(drop=True)
    lo, hi = F.MACRO_LONGRUN_WINDOW
    w = F.TTC_WEIGHT
    year = macro["ym"].str[:4].astype(int)
    yq = year.astype(str) + "Q" + ((macro["ym"].str[5:7].astype(int) - 1) // 3 + 1).astype(str)
    for col in SERIES.values():
        s = pd.to_numeric(macro[col], errors="coerce")
        # trailing / point-in-time-clean
        macro[f"{col}_ma12"] = s.rolling(12, min_periods=1).mean()
        ewma = s.ewm(halflife=6).mean()
        macro[f"{col}_ewma"] = ewma
        longrun = float(s[(macro["ym"] >= lo) & (macro["ym"] <= hi)].mean())
        macro[f"{col}_ttc"] = w * ewma + (1.0 - w) * longrun
        macro[f"{col}_gap"] = s - longrun
        macro[f"{col}_yoy"] = (s - s.shift(12)) / s.shift(12)
        macro[f"{col}_ln"] = np.log(s)
        # calendar-period (look-ahead within the period — kept for reference/testing)
        macro[f"{col}_annual_avg"] = s.groupby(year).transform("mean")
        macro[f"{col}_annual_change"] = year.map(s.groupby(year).mean().pct_change())
        macro[f"{col}_quarterly_avg"] = s.groupby(yq).transform("mean")
        macro[f"{col}_quarterly_change"] = yq.map(s.groupby(yq).mean().pct_change())
    return macro


def assign_macro(df: pd.DataFrame, macro_path: Path = MACRO_CSV) -> pd.DataFrame:
    """Join macro_unemployment / macro_fedfunds onto df by LoanOriginationDate's year-month."""
    macro = pd.read_csv(macro_path, dtype={"ym": str}).set_index("ym")
    ym = pd.to_datetime(df["LoanOriginationDate"], errors="coerce").dt.to_period("M").astype(str)
    out = df.copy()
    for col in MACRO_FEATURES:
        out[col] = ym.map(macro[col])
    return out


def main() -> None:
    print("Building macro table (local CSV -> FRED API -> fredgraph.csv) ...")
    macro = build_macro_table()
    macro = add_derived_columns(macro)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    macro.to_csv(MACRO_CSV, index=False)
    print(f"saved {MACRO_CSV}  ({len(macro):,} rows, {macro.shape[1]} cols)")
    print(macro.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()

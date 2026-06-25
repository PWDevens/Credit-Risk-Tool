"""Pull point-in-time STATE unemployment (BLS LAUS via FRED) and smooth it exactly like the
national series, but per state.

For each loan we already join the *national* unemployment + fed funds at origination
(build_macro_features.py). This adds a **regional cross-section**: the borrower's own state's
unemployment rate in the origination month. Unlike the national macro (which only varies by
date, so on a random split it acts as a vintage proxy), state unemployment varies *between
borrowers at the same time* — so it is far less of a vintage artifact and has a real shot at
helping out-of-time too.

FRED series id per state = "{ABBR}UR" (e.g. CAUR, TXUR, NYUR, DCUR) — seasonally adjusted,
monthly, 1976-present. Reuses build_macro_features._fetch (local CSV -> FRED API -> fredgraph).
The FRED JSON API (api.stlouisfed.org) needs a free key:
    $env:FRED_API_KEY = "your_key"     # https://fredaccount.stlouisfed.org/apikeys
(The keyless fredgraph endpoint is rate-limited and often times out for 51 series.)

Output: data/raw/state_monthly.csv in LONG format (one row per ym x state) with the same
TTC-anchored / gap / yoy columns the national table has.

Run:  python data/build_state_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))

from build_macro_features import _fetch  # noqa: E402  (local CSV -> API -> fredgraph)

RAW_DIR = REPO_ROOT / "data" / "raw"
STATE_CSV = RAW_DIR / "state_monthly.csv"

# 50 states + DC (USPS abbreviations). FRED unemployment series = "{ABBR}UR".
STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY",
]


def build_state_table() -> pd.DataFrame:
    """Fetch every state's unemployment series and stack into a long (ym, state, rate) frame."""
    frames = []
    for abbr in STATE_ABBRS:
        t = _fetch(f"{abbr}UR", "state_unemployment")   # raises with guidance if unreachable
        t = t.assign(state=abbr)
        frames.append(t)
    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["state", "ym"]).reset_index(drop=True)


def add_derived_columns(long: pd.DataFrame) -> pd.DataFrame:
    """Per-state trailing / point-in-time-clean smoothing, identical in spirit to the national
    series (build_macro_features.add_derived_columns), but computed within each state group:

      state_unemployment_ma12 : 12-month trailing moving average
      state_unemployment_ewma : exponential moving average, halflife 6 months
      state_unemployment_ttc  : w*ewma + (1-w)*state_long_run_mean  (anchored toward the cycle)
      state_unemployment_gap  : raw - state_long_run_mean           (cyclical deviation)
      state_unemployment_yoy  : (raw - raw 12m ago) / raw 12m ago   (trailing year-over-year)

    Long-run mean is computed PER STATE over features.MACRO_LONGRUN_WINDOW so each state is
    anchored to its OWN normal (a 4% state and an 8% state shouldn't share one mean).
    """
    import features as F  # local import: only needed when this script is run

    lo, hi = F.MACRO_LONGRUN_WINDOW
    w = F.TTC_WEIGHT
    out = long.sort_values(["state", "ym"]).copy()
    g = out.groupby("state", group_keys=False)
    s = out["state_unemployment"]

    out["state_unemployment_ma12"] = g["state_unemployment"].transform(
        lambda x: x.rolling(12, min_periods=1).mean())
    ewma = g["state_unemployment"].transform(lambda x: x.ewm(halflife=6).mean())
    out["state_unemployment_ewma"] = ewma
    longrun = out.assign(_in=(out["ym"] >= lo) & (out["ym"] <= hi))
    lr = (longrun[longrun["_in"]].groupby("state")["state_unemployment"].mean())
    out["_lr"] = out["state"].map(lr)
    out["state_unemployment_ttc"] = w * ewma + (1.0 - w) * out["_lr"]
    out["state_unemployment_gap"] = s - out["_lr"]
    out["state_unemployment_yoy"] = g["state_unemployment"].transform(
        lambda x: (x - x.shift(12)) / x.shift(12))
    out["state_unemployment_ln"] = np.log(s)
    return out.drop(columns=["_lr"])


def main() -> None:
    print(f"Building state unemployment table for {len(STATE_ABBRS)} states "
          "(local CSV -> FRED API -> fredgraph) ...")
    long = build_state_table()
    long = add_derived_columns(long)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(STATE_CSV, index=False)
    print(f"saved {STATE_CSV}  ({len(long):,} rows, {long['state'].nunique()} states, "
          f"{long.shape[1]} cols)")
    print(long[long["state"] == "CA"].tail(3).to_string(index=False))


if __name__ == "__main__":
    main()

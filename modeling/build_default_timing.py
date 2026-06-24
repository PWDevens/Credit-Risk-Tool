"""Derive the empirical default-timing curve for the v2 financial engine.

Time-to-default (ClosedDate - LoanOriginationDate) as a fraction of term, over defaulted
post-2009 loans, histogrammed to a density on [0,1]. The finance engine uses it to spread
the model's lifetime PD into a per-month marginal hazard. Saves models/default_timing.json
(gitignored, regenerable). The engine has a sane fallback if the file is absent.

Run:  python modeling/build_default_timing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
import features as F  # noqa: E402

BINS = 20


def main() -> None:
    df = pd.read_csv(F.RAW_CSV, low_memory=False,
                     usecols=["LoanStatus", "LoanOriginationDate", "ClosedDate", "Term"])
    df = df[df["LoanStatus"].isin(F.BAD_STATUSES)].copy()
    orig = pd.to_datetime(df["LoanOriginationDate"], errors="coerce")
    closed = pd.to_datetime(df["ClosedDate"], errors="coerce")
    df = df[(closed.notna()) & (orig >= F.POST_2009_CUTOFF)]
    months = (closed[df.index] - orig[df.index]).dt.days / 30.44
    frac = (months / df["Term"]).clip(0, 1)
    frac = frac[frac.notna() & (df["Term"] > 0)]

    hist, _edges = np.histogram(frac, bins=BINS, range=(0, 1))
    dens = hist / hist.sum()

    (REPO_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "models" / "default_timing.json").write_text(
        json.dumps({"bins": BINS, "density": dens.tolist(), "n": int(hist.sum())}, indent=2))
    peak = (np.argmax(dens) + 0.5) / BINS
    print(f"default-timing curve from {int(hist.sum()):,} defaults; peak at ~{peak:.0%} of life")
    print("density:", dens.round(3).tolist())


if __name__ == "__main__":
    main()

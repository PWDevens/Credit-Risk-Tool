"""Phase 1 — calibration diagnostics for the production PD model (calibrated XGBoost).

A PD must be a true probability for the Expected-Loss / pricing engine, not just a good ranking.
This reports two calibration views on the held-out test set:
  1. by predicted-PD decile  — predicted vs actual default rate in each risk band
  2. by vintage (origination year) — does the model stay calibrated across cohorts / over time?

  python modeling/calibration_report.py

Saves model-results/calibration_decile.csv and calibration_vintage.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from common import data as D, metrics as M  # noqa: E402
import features as F  # noqa: E402

RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
PD_FINE = REPO_ROOT / "modeling" / "probability-of-default" / "pd_xgboost.joblib"
FMT = {"pred_PD": "{:.3f}".format, "actual_PD": "{:.3f}".format, "gap": "{:+.3f}".format}


def main() -> None:
    if not PD_FINE.exists():
        raise SystemExit("pd_xgboost.joblib not found — run finetune_xgboost.py first.")
    art = joblib.load(PD_FINE)
    pre, model, cols = art["preprocessor"], art["model"], art["feature_cols"]

    test = D.load_frame("test")
    p = model.predict_proba(pre.transform(test[cols]))[:, 1]   # calibrated PD
    y = test[F.PD_TARGET].astype(int).to_numpy()

    # 1. By predicted-PD decile.
    dec = M.calibration_table(y, p, bins=10)
    print("=== Calibration by predicted-PD decile (held-out test set) ===")
    print(dec.to_string(index=False, formatters=FMT))

    # 2. By vintage (origination year).
    year = pd.to_datetime(test["LoanOriginationDate"], errors="coerce").dt.year
    vint = (pd.DataFrame({"year": year, "y": y, "p": p}).dropna(subset=["year"])
            .groupby("year").agg(n=("y", "size"), pred_PD=("p", "mean"), actual_PD=("y", "mean")))
    vint["gap"] = vint["pred_PD"] - vint["actual_PD"]
    vint = vint.reset_index()
    vint["year"] = vint["year"].astype(int)
    print("\n=== Calibration by vintage (origination year, test set) ===")
    print(vint.to_string(index=False, formatters=FMT))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dec.to_csv(RESULTS_DIR / "calibration_decile.csv", index=False)
    vint.to_csv(RESULTS_DIR / "calibration_vintage.csv", index=False)
    mae = float((dec["gap"].abs() * dec["n"]).sum() / dec["n"].sum())
    print(f"\nWeighted mean |predicted - actual| across deciles: {mae:.4f}")
    print(f"saved calibration_decile.csv + calibration_vintage.csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

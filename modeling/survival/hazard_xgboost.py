"""Phase 2 — fit a calibrated discrete-time XGBoost hazard h(t|x) on the loan-month panel.

The model's predicted probability for a (loan, month) row IS the monthly hazard: the chance the
loan defaults in that month given it survived to it. Reuses the production preprocessor + XGBoost
config. Most loan-months are non-events, so we down-sample survivors for the fit and re-calibrate
on an UN-sampled, loan-disjoint slice to recover true probabilities (needed for the ECL).

    .venv\\Scripts\\python.exe modeling\\survival\\hazard_xgboost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "modeling" / "probability-of-default"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from common.finetune import build_preprocessor  # noqa: E402
from common import data as D  # noqa: E402
import finetune_xgboost as xgb  # noqa: E402
from build_loan_month_panel import PANEL_PATH, covariate_cols, HAZARD_TARGET  # noqa: E402

OUT = REPO_ROOT / "modeling" / "probability-of-default" / "pd_hazard_xgboost.joblib"
NEG_PER_POS = 20          # survivors kept per default-month row in the fit pool
SEED = 42


def fit_hazard(budget=None) -> dict:
    if not PANEL_PATH.exists():
        raise SystemExit("loan_month_panel.parquet missing — run data/build_loan_month_panel.py.")
    panel = pd.read_parquet(PANEL_PATH)
    feats = [c for c in covariate_cols() if c in panel.columns]

    # Out-of-time split by origination (a loan has one origination date -> grouped by loan for free).
    orig = pd.to_datetime(panel["LoanOriginationDate"], errors="coerce")
    cut = pd.Timestamp(D.OOT_CUTOFF)
    tr, te = panel[orig < cut], panel[orig >= cut]
    print(f"panel {len(panel):,} rows | OOT train {len(tr):,} | test {len(te):,}")

    # Loan-level fit/cal split so calibration sees UN-sampled rows (true base rate), no leakage.
    rng = np.random.RandomState(SEED)
    loans = tr["loan_id"].unique()
    cal_loans = set(rng.choice(loans, size=max(int(0.15 * len(loans)), 1), replace=False))
    is_cal = tr["loan_id"].isin(cal_loans)
    fit_pool, cal_pool = tr[~is_cal], tr[is_cal]

    # Down-sample survivor rows in the fit pool; keep every default-month row.
    pos = fit_pool[fit_pool[HAZARD_TARGET] == 1]
    neg_all = fit_pool[fit_pool[HAZARD_TARGET] == 0]
    neg = neg_all.sample(n=min(len(pos) * NEG_PER_POS, len(neg_all)), random_state=SEED)
    fit = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED)
    spw = float((fit[HAZARD_TARGET] == 0).sum() / max((fit[HAZARD_TARGET] == 1).sum(), 1))
    print(f"fit {len(fit):,} (pos {len(pos):,}, neg {len(neg):,}, scale_pos_weight {spw:.1f}) | "
          f"cal {len(cal_pool):,}")

    pre = build_preprocessor(feats).fit(fit[feats])
    est = xgb.build(xgb.SEED, spw)
    est.fit(pre.transform(fit[feats]), fit[HAZARD_TARGET])
    # Isotonic recalibration on the un-sampled cal pool -> hazards are true probabilities again.
    cal = CalibratedClassifierCV(FrozenEstimator(est), method="isotonic").fit(
        pre.transform(cal_pool[feats]), cal_pool[HAZARD_TARGET])

    auc = roc_auc_score(te[HAZARD_TARGET], cal.predict_proba(pre.transform(te[feats]))[:, 1])
    print(f"hazard row-level test ROC-AUC: {auc:.4f}  (smoke signal; survival metrics in benchmark)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    art = {"preprocessor": pre, "model": cal, "estimator": est, "feature_cols": feats}
    joblib.dump(art, OUT)
    print(f"saved {OUT}")
    return {**art, "test_auc": float(auc)}


if __name__ == "__main__":
    fit_hazard()

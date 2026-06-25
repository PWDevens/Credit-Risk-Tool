"""Phase 2 — scikit-survival benchmark + survival-metric suite, reported out-of-time.

An independent continuous-time model (Random Survival Forest) so we can say "we evaluated the
alternative," plus the metrics that judge a survival model properly: time-dependent AUC, IPCW
concordance, and integrated Brier score. We also score our discrete-time hazard model on IPCW
concordance (ranking loans by lifetime PD) so the two are comparable.

scikit-survival is an optional dependency — if it isn't installed, this skips cleanly.

    .venv\\Scripts\\python.exe modeling\\survival\\benchmark_sksurv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from common.finetune import build_preprocessor  # noqa: E402
from common import data as D  # noqa: E402
from survival import term_structure as TS  # noqa: E402
from build_loan_month_panel import PANEL_PATH, covariate_cols, TIME_COLS  # noqa: E402

RESULTS = REPO_ROOT / "modeling" / "model-results" / "hazard_survival_metrics.csv"
SEED = 42
RSF_SAMPLE = 8000


def _loan_level(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the panel to one row per loan (covariates are constant within a loan)."""
    return panel.sort_values(["loan_id", "t"]).groupby("loan_id", sort=False).first().reset_index()


def main() -> None:
    try:
        from sksurv.util import Surv
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import (cumulative_dynamic_auc, concordance_index_ipcw,
                                     integrated_brier_score)
    except Exception as exc:  # noqa: BLE001
        print(f"scikit-survival not installed — skipping benchmark ({exc}).")
        return
    if not PANEL_PATH.exists():
        print("loan_month_panel.parquet missing — run data/build_loan_month_panel.py.")
        return

    loans = _loan_level(pd.read_parquet(PANEL_PATH))
    feats = [c for c in covariate_cols() if c not in TIME_COLS and c in loans.columns]
    orig = pd.to_datetime(loans["LoanOriginationDate"], errors="coerce")
    cut = pd.Timestamp(D.OOT_CUTOFF)
    tr, te = loans[orig < cut], loans[orig >= cut]
    print(f"loans: train {len(tr):,} | test {len(te):,}")

    pre = build_preprocessor(feats).fit(tr[feats])
    Xtr = np.nan_to_num(np.asarray(pre.transform(tr[feats]), dtype=float))
    Xte = np.nan_to_num(np.asarray(pre.transform(te[feats]), dtype=float))
    ytr = Surv.from_arrays(tr["is_event"].to_numpy(bool), tr["T_obs"].to_numpy(float))
    yte = Surv.from_arrays(te["is_event"].to_numpy(bool), te["T_obs"].to_numpy(float))

    # Sample the train loans so the RSF fit stays quick.
    rng = np.random.RandomState(SEED)
    if len(tr) > RSF_SAMPLE:
        idx = rng.choice(len(tr), RSF_SAMPLE, replace=False)
        Xtr_s, ytr_s = Xtr[idx], ytr[idx]
    else:
        Xtr_s, ytr_s = Xtr, ytr

    rsf = RandomSurvivalForest(n_estimators=100, max_depth=6, min_samples_leaf=50,
                               n_jobs=-1, random_state=SEED).fit(Xtr_s, ytr_s)

    # Monthly evaluation grid, kept strictly inside both train and test follow-up (sksurv requires it).
    hi = float(min(ytr_s["time"].max(), yte["time"].max()))
    lo = float(max(ytr_s["time"].min(), yte["time"].min())) + 1.0
    times = np.arange(np.ceil(lo) + 1, np.floor(hi) - 1, 6.0)

    rows = []
    try:
        risk = rsf.predict(Xte)
        _, auc_mean = cumulative_dynamic_auc(ytr_s, yte, risk, times)
        cidx = concordance_index_ipcw(ytr_s, yte, risk)[0]
        surv = np.vstack([fn(times) for fn in rsf.predict_survival_function(Xte)])
        ibs = integrated_brier_score(ytr_s, yte, surv, times)
        rows += [("rsf", "time_dependent_AUC_mean", auc_mean),
                 ("rsf", "concordance_ipcw", cidx), ("rsf", "integrated_brier", ibs)]
        print(f"RSF: td-AUC={auc_mean:.4f}  c-index(ipcw)={cidx:.4f}  IBS={ibs:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  RSF metric step failed ({exc}) — continuing.")

    # Our discrete-time hazard model: rank test loans by lifetime PD, score IPCW concordance.
    # Batched: expand every test loan to its loan-months in ONE frame, score once, then collapse
    # to a lifetime PD per loan (1 - prod(1-h)). This replaces a per-loan preprocessor.transform
    # (tens of thousands of calls) with a single transform over the whole test set.
    if TS.available():
        try:
            art = joblib.load(TS.HAZARD_PATH)
            hcols = art["feature_cols"]
            terms = te["Term"].astype(int).clip(lower=1).to_numpy()
            big = te.loc[te.index.repeat(terms)].reset_index(drop=True)
            big["t"] = np.concatenate([np.arange(1, n + 1) for n in terms])
            big["t_frac"] = big["t"] / big["Term"].astype(int).to_numpy()
            h = art["model"].predict_proba(art["preprocessor"].transform(big[hcols]))[:, 1]
            h = np.clip(h, 1e-6, 1 - 1e-6)
            # Grouped product of (1-h) per loan via summed logs -> lifetime PD = 1 - S(term).
            loan_idx = np.repeat(np.arange(len(te)), terms)
            log_surv = np.bincount(loan_idx, weights=np.log1p(-h), minlength=len(te))
            risk_h = 1.0 - np.exp(log_surv)
            h_cidx = concordance_index_ipcw(ytr_s, yte, risk_h)[0]
            rows.append(("hazard_xgboost", "concordance_ipcw", h_cidx))
            print(f"hazard_xgboost: c-index(ipcw)={h_cidx:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  hazard concordance step failed ({exc}) — continuing.")
    else:
        print("hazard artifact missing — skipping its concordance comparison.")

    if rows:
        out = pd.DataFrame(rows, columns=["model", "metric", "value"])
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(RESULTS, index=False)
        print(f"saved {RESULTS}")


if __name__ == "__main__":
    main()

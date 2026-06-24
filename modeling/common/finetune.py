"""Shared fine-tuning harness for the per-model PD challengers.

One encoder, one split, one FLAML cross-validated-AUC objective, one calibrate+eval+save
path — so finetune_xgboost / _lightgbm / _rf differ ONLY in the estimator and its search
space. That keeps the model comparison fair and the losing experiments easy to read side
by side (they are kept on purpose, as due-diligence artifacts).

Encoding (fit on training rows, applied unchanged to cal/test/scoring):
  NOMINAL (no order)   -> OneHotEncoder(handle_unknown='ignore')
  ORDINAL (real order) -> OrdinalEncoder with explicit category order ("label encoding")
  NUMERIC              -> passthrough
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data"))

import features as F  # noqa: E402
from . import data as D, metrics as M  # noqa: E402

OUT_DIR = REPO_ROOT / "modeling" / "probability-of-default"
RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
CV_FOLDS = 3

ORDINAL_ORDERS = {
    "IncomeRange": ["Not employed", "$0", "$1-24,999", "$25,000-49,999",
                    "$50,000-74,999", "$75,000-99,999", "$100,000+"],
    "bankcard_util_bucket": ["<=30%", "30-50%", "50-75%", "75-100%", ">100%"],
}
ORDINAL_FEATURES = list(ORDINAL_ORDERS)
NOMINAL_FEATURES = [c for c in F.CATEGORICAL_FEATURES if c not in ORDINAL_FEATURES]
NUMERIC_FEATURES = [c for c in F.MODEL_FEATURES if c not in F.CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("ord", OrdinalEncoder(
                categories=[ORDINAL_ORDERS[c] for c in ORDINAL_FEATURES],
                handle_unknown="use_encoded_value", unknown_value=-1,
            ), ORDINAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _coerce(params: dict, int_keys) -> dict:
    p = dict(params)
    for k in int_keys:
        if p.get(k) is not None:
            p[k] = int(round(p[k]))
    return p


def _flaml_search(build_estimator, search_space, seed_config, int_keys, X, y, spw, budget):
    """FLAML search; objective = mean CV AUC (encoder re-fit per fold, no leakage)."""
    from flaml import tune

    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42).split(X, y))

    def evaluate(config):
        params = _coerce(config, int_keys)
        aucs = []
        for tr, va in folds:
            Xt, Xv, yt, yv = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
            pre = build_preprocessor().fit(Xt)
            est = build_estimator(params, spw)
            est.fit(pre.transform(Xt), yt)
            aucs.append(roc_auc_score(yv, est.predict_proba(pre.transform(Xv))[:, 1]))
        return {"auc": float(np.mean(aucs))}

    analysis = tune.run(
        evaluate, config=search_space, metric="auc", mode="max",
        time_budget_s=budget, num_samples=-1,
        points_to_evaluate=[seed_config],
        low_cost_partial_config={"n_estimators": 50}, verbose=1,
    )
    return _coerce(analysis.best_config, int_keys), float(analysis.best_result["auc"])


def run_finetune(name, build_estimator, search_space, seed_config, int_keys) -> dict:
    """Tune `build_estimator` with FLAML, refit + isotonic-calibrate, evaluate on test.

    build_estimator(params, scale_pos_weight) -> an unfitted sklearn-API classifier.
    Saves pd_<name>.joblib + pd_<name>_best_config.json + pd_finetune_<name>.csv.
    """
    budget = int(os.environ.get("FLAML_TIME_BUDGET", "180"))
    train, test = D.load_frame("train"), D.load_frame("test")
    Xtr, ytr = D.pd_Xy(train)
    Xte, yte = D.pd_Xy(test)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    print(f"\n=== {name}: FLAML search (budget={budget}s, {CV_FOLDS}-fold CV AUC) ===")
    best, cv_auc = _flaml_search(build_estimator, search_space, seed_config, int_keys, Xtr, ytr, spw, budget)
    print(f"[{name}] best CV AUC = {cv_auc:.4f}  config = {json.dumps(best)}")

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        Xtr, ytr, test_size=0.20, random_state=42, stratify=ytr)
    pre = build_preprocessor().fit(X_fit)
    X_fit_e, X_cal_e, X_te_e = pre.transform(X_fit), pre.transform(X_cal), pre.transform(Xte)

    est = build_estimator(best, spw)
    est.fit(X_fit_e, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(est), method="isotonic").fit(X_cal_e, y_cal)

    raw_prob = est.predict_proba(X_te_e)[:, 1]
    cal_prob = calibrated.predict_proba(X_te_e)[:, 1]
    rows = [
        {"model": f"{name}_raw", **M.pd_metrics(yte, raw_prob)},
        {"model": f"{name}_calibrated", **M.pd_metrics(yte, cal_prob)},
    ]
    base_csv = RESULTS_DIR / "pd_baseline_automl.csv"
    if base_csv.exists():
        b = pd.read_csv(base_csv)
        b = b[b["model"] == "automl"]
        if not b.empty:
            rows.append({"model": "automl_baseline",
                         **{k: b.iloc[0][k] for k in ["AUC", "Gini", "KS", "Brier", "LogLoss"]}})

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / f"pd_finetune_{name}.csv", index=False)
    print("\n" + res.to_string(index=False))

    # Save the calibrated model (for scoring) AND the raw fitted estimator (for SHAP,
    # which explains the pre-calibration booster) + the preprocessor + feature order.
    joblib.dump({"preprocessor": pre, "model": calibrated, "estimator": est,
                 "features": F.MODEL_FEATURES, "best_config": best},
                OUT_DIR / f"pd_{name}.joblib")
    test_auc = float(M.pd_metrics(yte, cal_prob)["AUC"])
    (RESULTS_DIR / f"pd_{name}_best_config.json").write_text(json.dumps(
        {"best_config": best, "cv_auc": cv_auc, "test_auc_calibrated": test_auc}, indent=2))
    print(f"saved pd_{name}.joblib + config + metrics")
    return {"name": name, "cv_auc": cv_auc, "test_auc_calibrated": test_auc}

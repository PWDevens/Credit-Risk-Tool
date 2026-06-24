"""Shared fine-tuning harness for the per-model PD challengers.

One encoder, one split, one FLAML cross-validated-AUC objective, one calibrate+eval path —
so the model scripts differ ONLY in the estimator and its search space.

Encoding (fit on training rows, applied unchanged to cal/test/scoring):
  NOMINAL (no order)   -> OneHotEncoder(handle_unknown='ignore')
  ORDINAL (real order) -> OrdinalEncoder with explicit category order ("label encoding")
  NUMERIC              -> passthrough for trees; median-impute + StandardScaler for linear
                          models (scale_numeric=True), which can't take NaN or raw scales.

build_preprocessor() takes the actual feature columns, so the same harness serves the
4-version feature-set matrix (base / +cluster / +engineered / +both) — see run_matrix.py.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

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


def build_preprocessor(feature_cols, scale_numeric: bool = False) -> ColumnTransformer:
    """Partition the given feature columns into numeric / ordinal / nominal and encode.

    Dynamic so any feature subset works. scale_numeric=True (linear models) median-imputes
    and standardizes the numeric block; trees use passthrough.
    """
    cat = set(F.CATEGORICAL_FEATURES)
    ordinal = [c for c in feature_cols if c in ORDINAL_ORDERS]
    nominal = [c for c in feature_cols if c in cat and c not in ORDINAL_ORDERS]
    numeric = [c for c in feature_cols if c not in cat]
    num_tf = (Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
              if scale_numeric else "passthrough")
    return ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric),
            ("ord", OrdinalEncoder(categories=[ORDINAL_ORDERS[c] for c in ordinal],
                                   handle_unknown="use_encoded_value", unknown_value=-1), ordinal),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
        ],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _coerce(params: dict, int_keys) -> dict:
    p = dict(params)
    for k in int_keys:
        if p.get(k) is not None:
            p[k] = int(round(p[k]))
    return p


def _flaml_search(build_estimator, search_space, seed_config, int_keys, X, y, spw, budget,
                  scale_numeric):
    """FLAML search; objective = mean CV AUC (encoder re-fit per fold, no leakage)."""
    from flaml import tune

    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42).split(X, y))
    cols = list(X.columns)

    def evaluate(config):
        params = _coerce(config, int_keys)
        aucs = []
        for tr, va in folds:
            Xt, Xv, yt, yv = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
            pre = build_preprocessor(cols, scale_numeric).fit(Xt)
            est = build_estimator(params, spw)
            est.fit(pre.transform(Xt), yt)
            aucs.append(roc_auc_score(yv, est.predict_proba(pre.transform(Xv))[:, 1]))
        return {"auc": float(np.mean(aucs))}

    kwargs = dict(config=search_space, metric="auc", mode="max", time_budget_s=budget,
                  num_samples=-1, points_to_evaluate=[seed_config], verbose=0)
    if "n_estimators" in search_space:  # low-cost warm start only for ensemble models
        kwargs["low_cost_partial_config"] = {"n_estimators": 50}
    analysis = tune.run(evaluate, **kwargs)
    return _coerce(analysis.best_config, int_keys), float(analysis.best_result["auc"])


def _train_one(build_estimator, search_space, seed_config, int_keys, *,
               include_engineered, include_cluster, scale_numeric, budget, include_macro=False):
    """Tune + refit + isotonic-calibrate on one feature-set. Returns everything needed to
    score or save. Shared by run_finetune (saves a model) and evaluate_featureset (matrix)."""
    train, test = D.load_frame("train"), D.load_frame("test")
    Xtr, ytr = D.pd_Xy(train, include_engineered, include_cluster, include_macro)
    Xte, yte = D.pd_Xy(test, include_engineered, include_cluster, include_macro)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    best, cv_auc = _flaml_search(build_estimator, search_space, seed_config, int_keys,
                                 Xtr, ytr, spw, budget, scale_numeric)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        Xtr, ytr, test_size=0.20, random_state=42, stratify=ytr)
    pre = build_preprocessor(list(Xtr.columns), scale_numeric).fit(X_fit)
    est = build_estimator(best, spw)
    est.fit(pre.transform(X_fit), y_fit)
    cal = CalibratedClassifierCV(FrozenEstimator(est), method="isotonic").fit(pre.transform(X_cal), y_cal)
    Xte_e = pre.transform(Xte)
    return {"best": best, "cv_auc": cv_auc, "pre": pre, "est": est, "cal": cal,
            "feature_cols": list(Xtr.columns), "yte": yte, "Xte_e": Xte_e,
            "raw": est.predict_proba(Xte_e)[:, 1], "cal_prob": cal.predict_proba(Xte_e)[:, 1]}


def evaluate_featureset(name, build_estimator, search_space, seed_config, int_keys, *,
                        include_engineered, include_cluster, include_macro=False,
                        scale_numeric=False, budget=None) -> dict:
    """Lightweight matrix cell: tune+eval on one (model, feature-set), return metrics only."""
    budget = budget or int(os.environ.get("FLAML_TIME_BUDGET", "120"))
    r = _train_one(build_estimator, search_space, seed_config, int_keys,
                   include_engineered=include_engineered, include_cluster=include_cluster,
                   include_macro=include_macro, scale_numeric=scale_numeric, budget=budget)
    mc = M.pd_metrics(r["yte"], r["cal_prob"])
    return {"model": name, "cv_auc": round(r["cv_auc"], 4),
            "test_auc_cal": round(mc["AUC"], 4), "gini": round(mc["Gini"], 4),
            "ks": round(mc["KS"], 4), "brier": round(mc["Brier"], 4),
            "n_features": len(r["feature_cols"])}


def run_finetune(name, build_estimator, search_space, seed_config, int_keys, *,
                 scale_numeric=False, include_engineered=True, include_cluster=True) -> dict:
    """Full path for a shipped per-model challenger: tune on the v4 feature-set (default),
    save pd_<name>.joblib + pd_<name>_best_config.json + pd_finetune_<name>.csv."""
    budget = int(os.environ.get("FLAML_TIME_BUDGET", "180"))
    print(f"\n=== {name}: FLAML search (budget={budget}s, {CV_FOLDS}-fold CV AUC) ===")
    r = _train_one(build_estimator, search_space, seed_config, int_keys,
                   include_engineered=include_engineered, include_cluster=include_cluster,
                   scale_numeric=scale_numeric, budget=budget)
    print(f"[{name}] best CV AUC = {r['cv_auc']:.4f}  config = {json.dumps(r['best'])}")

    rows = [
        {"model": f"{name}_raw", **M.pd_metrics(r["yte"], r["raw"])},
        {"model": f"{name}_calibrated", **M.pd_metrics(r["yte"], r["cal_prob"])},
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

    joblib.dump({"preprocessor": r["pre"], "model": r["cal"], "estimator": r["est"],
                 "features": F.MODEL_FEATURES, "feature_cols": r["feature_cols"],
                 "best_config": r["best"]}, OUT_DIR / f"pd_{name}.joblib")
    test_auc = float(M.pd_metrics(r["yte"], r["cal_prob"])["AUC"])
    (RESULTS_DIR / f"pd_{name}_best_config.json").write_text(json.dumps(
        {"best_config": r["best"], "cv_auc": r["cv_auc"], "test_auc_calibrated": test_auc}, indent=2))
    print(f"saved pd_{name}.joblib + config + metrics")
    return {"name": name, "cv_auc": r["cv_auc"], "test_auc_calibrated": test_auc}

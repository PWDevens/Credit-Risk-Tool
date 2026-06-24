"""Fine-tuned PD model — a FLAML-tuned, calibrated XGBoost challenger to the AutoML baseline.

Step E (PD). Reuses the SAME split and feature contract as the AutoML baseline (via
features.py + common.data) so the head-to-head on the held-out test set is fair.

Pipeline:
  1. Encode (fit on training rows, applied unchanged to cal/test/scoring):
       NOMINAL (no order)  -> OneHotEncoder(handle_unknown='ignore')
       ORDINAL (real order)-> OrdinalEncoder with explicit category order
       NUMERIC             -> passthrough
  2. TUNE hyperparameters with FLAML. The search objective is 3-fold CV AUC with the
     encoder re-fit inside each fold (no leakage) and no early stopping (so the metric
     is unbiased). The search is seeded with our best manual config, so it only has to
     improve on it.
  3. Refit the best config, isotonic-calibrate on a holdout (PD must be a true
     probability for EL = PD x LGD x EAD), and score raw vs calibrated on the test set
     next to the AutoML baseline.

Run:  python modeling/probability-of-default/finetune_xgboost.py
Tune the search budget:  FLAML_TIME_BUDGET=600 python .../finetune_xgboost.py
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
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

import features as F  # noqa: E402
from common import data as D, metrics as M  # noqa: E402

OUT_DIR = REPO_ROOT / "modeling" / "probability-of-default"
RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
MODEL_PATH = OUT_DIR / "pd_xgboost.joblib"
CONFIG_PATH = RESULTS_DIR / "pd_xgboost_best_config.json"

TIME_BUDGET = int(os.environ.get("FLAML_TIME_BUDGET", "180"))  # seconds of search
CV_FOLDS = 3

# --- feature partitions for encoding ---------------------------------------- #
ORDINAL_ORDERS = {
    "IncomeRange": ["Not employed", "$0", "$1-24,999", "$25,000-49,999",
                    "$50,000-74,999", "$75,000-99,999", "$100,000+"],
    "bankcard_util_bucket": ["<=30%", "30-50%", "50-75%", "75-100%", ">100%"],
}
ORDINAL_FEATURES = list(ORDINAL_ORDERS)
NOMINAL_FEATURES = [c for c in F.CATEGORICAL_FEATURES if c not in ORDINAL_FEATURES]
NUMERIC_FEATURES = [c for c in F.MODEL_FEATURES if c not in F.CATEGORICAL_FEATURES]

# Fixed params (not searched). scale_pos_weight is set from the data in main().
FIXED_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def build_preprocessor() -> ColumnTransformer:
    """One-hot for nominal, ordinal-encode for ordinal, passthrough numeric."""
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


def _coerce_ints(params: dict) -> dict:
    params = dict(params)
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        if k in params:
            params[k] = int(round(params[k]))
    return params


def tune_hyperparameters(X, y, spw):
    """FLAML search; objective = mean CV AUC (encoder re-fit per fold, no early stopping)."""
    from flaml import tune

    base = {**FIXED_PARAMS, "scale_pos_weight": spw}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    def evaluate(config):
        params = _coerce_ints({**base, **config})
        aucs = []
        for tr, va in folds:
            Xt, Xv = X.iloc[tr], X.iloc[va]
            yt, yv = y.iloc[tr], y.iloc[va]
            pre = build_preprocessor().fit(Xt)
            model = XGBClassifier(**params)
            model.fit(pre.transform(Xt), yt)
            aucs.append(roc_auc_score(yv, model.predict_proba(pre.transform(Xv))[:, 1]))
        return {"auc": float(np.mean(aucs))}

    search_space = {
        "n_estimators": tune.lograndint(50, 800),
        "max_depth": tune.randint(3, 9),
        "learning_rate": tune.loguniform(0.01, 0.3),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "min_child_weight": tune.randint(1, 10),
        "reg_lambda": tune.loguniform(0.1, 5.0),
        "reg_alpha": tune.loguniform(1e-3, 5.0),
        "gamma": tune.uniform(0.0, 3.0),
    }
    # Seed with the best manual config so FLAML only has to improve on AUC ~0.745.
    seed = {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "reg_lambda": 1.0, "reg_alpha": 0.001, "gamma": 0.0}

    print(f"FLAML search: time_budget={TIME_BUDGET}s, {CV_FOLDS}-fold CV AUC ...")
    analysis = tune.run(
        evaluate,
        config=search_space,
        metric="auc",
        mode="max",
        time_budget_s=TIME_BUDGET,
        num_samples=-1,
        points_to_evaluate=[seed],
        low_cost_partial_config={"n_estimators": 50},
        verbose=1,
    )
    best = _coerce_ints(analysis.best_config)
    print(f"best CV AUC = {analysis.best_result['auc']:.4f}")
    print(f"best config = {json.dumps(best)}")
    return best, float(analysis.best_result["auc"])


def main() -> None:
    train, test = D.load_frame("train"), D.load_frame("test")
    Xtr_all, ytr_all = D.pd_Xy(train)
    Xte, yte = D.pd_Xy(test)

    spw = float((ytr_all == 0).sum() / max((ytr_all == 1).sum(), 1))
    best_config, best_cv_auc = tune_hyperparameters(Xtr_all, ytr_all, spw)

    # Refit the winning config: train on a fit fold, calibrate on a holdout, test on test.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        Xtr_all, ytr_all, test_size=0.20, random_state=42, stratify=ytr_all
    )
    pre = build_preprocessor().fit(X_fit)
    X_fit_e, X_cal_e, X_te_e = pre.transform(X_fit), pre.transform(X_cal), pre.transform(Xte)

    clf = XGBClassifier(**{**FIXED_PARAMS, "scale_pos_weight": spw, **best_config})
    clf.fit(X_fit_e, y_fit, verbose=False)

    calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    calibrated.fit(X_cal_e, y_cal)

    raw_prob = clf.predict_proba(X_te_e)[:, 1]
    cal_prob = calibrated.predict_proba(X_te_e)[:, 1]
    rows = [
        {"model": "xgboost_flaml_raw", **M.pd_metrics(yte, raw_prob)},
        {"model": "xgboost_flaml_calibrated", **M.pd_metrics(yte, cal_prob)},
    ]
    base_csv = RESULTS_DIR / "pd_baseline_automl.csv"
    if base_csv.exists():
        base = pd.read_csv(base_csv)
        b = base[base["model"] == "automl"]
        if not b.empty:
            rows.append({"model": "automl_baseline",
                         **{k: b.iloc[0][k] for k in ["AUC", "Gini", "KS", "Brier", "LogLoss"]}})

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "pd_finetune_xgboost.csv", index=False)
    print("\n" + res.to_string(index=False))

    joblib.dump(
        {"preprocessor": pre, "model": calibrated, "features": F.MODEL_FEATURES,
         "best_config": best_config},
        MODEL_PATH,
    )
    CONFIG_PATH.write_text(json.dumps(
        {"best_config": best_config, "cv_auc": best_cv_auc,
         "test_auc_calibrated": float(M.pd_metrics(yte, cal_prob)["AUC"])}, indent=2))
    print(f"\nsaved {MODEL_PATH}\nsaved {CONFIG_PATH}")


if __name__ == "__main__":
    main()

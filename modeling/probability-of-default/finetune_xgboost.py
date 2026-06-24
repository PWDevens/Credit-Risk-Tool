"""Fine-tuned PD model — FLAML-tuned, calibrated XGBoost challenger to the AutoML baseline.

One of three per-model challengers (see also finetune_lightgbm.py, finetune_rf.py), kept
side by side as a due-diligence record of model selection. All three share the encoder,
split, CV-AUC objective, and calibration in common/finetune.py — they differ only in the
estimator and its search space below.

Run:  python modeling/probability-of-default/finetune_xgboost.py
Budget:  FLAML_TIME_BUDGET=600 python .../finetune_xgboost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))

from flaml import tune  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from common.finetune import run_finetune  # noqa: E402

INT_KEYS = ["n_estimators", "max_depth", "min_child_weight"]


def build(params: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_jobs=-1, random_state=42, scale_pos_weight=scale_pos_weight, **params,
    )


SEARCH_SPACE = {
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
# Seed the search with the best manual config so FLAML only has to improve on it.
SEED = {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_lambda": 1.0, "reg_alpha": 0.001, "gamma": 0.0}


if __name__ == "__main__":
    run_finetune("xgboost", build, SEARCH_SPACE, SEED, INT_KEYS)

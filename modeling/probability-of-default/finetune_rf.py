"""Fine-tuned PD model — FLAML-tuned, calibrated Random Forest challenger to the AutoML baseline.

One of three per-model challengers (see also finetune_xgboost.py, finetune_lightgbm.py),
kept side by side as a due-diligence record of model selection. Shares the encoder, split,
CV-AUC objective, and calibration in common/finetune.py; differs only in the estimator and
its search space below. Class imbalance is handled with class_weight='balanced' (RF has no
scale_pos_weight), so the scale_pos_weight argument is accepted but unused.

Run:  python modeling/probability-of-default/finetune_rf.py
Budget:  FLAML_TIME_BUDGET=600 python .../finetune_rf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))

from flaml import tune  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from common.finetune import run_finetune  # noqa: E402

INT_KEYS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split"]


def build(params: dict, scale_pos_weight: float) -> RandomForestClassifier:
    # RF handles imbalance via class_weight, not scale_pos_weight (unused here).
    return RandomForestClassifier(
        n_jobs=-1, random_state=42, class_weight="balanced", **params,
    )


SEARCH_SPACE = {
    "n_estimators": tune.lograndint(50, 600),
    "max_depth": tune.randint(3, 20),
    "min_samples_leaf": tune.randint(1, 20),
    "min_samples_split": tune.randint(2, 20),
    "max_features": tune.uniform(0.1, 1.0),
}
SEED = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5,
        "min_samples_split": 2, "max_features": 0.3}


if __name__ == "__main__":
    run_finetune("rf", build, SEARCH_SPACE, SEED, INT_KEYS)

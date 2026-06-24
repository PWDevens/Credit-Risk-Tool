"""Fine-tuned PD model — FLAML-tuned, calibrated LightGBM challenger to the AutoML baseline.

One of three per-model challengers (see also finetune_xgboost.py, finetune_rf.py), kept
side by side as a due-diligence record of model selection. Shares the encoder, split,
CV-AUC objective, and calibration in common/finetune.py; differs only in the estimator
and its search space below.

Run:  python modeling/probability-of-default/finetune_lightgbm.py
Budget:  FLAML_TIME_BUDGET=600 python .../finetune_lightgbm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))

from flaml import tune  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from common.finetune import run_finetune  # noqa: E402

INT_KEYS = ["n_estimators", "num_leaves", "min_child_samples", "max_depth"]


def build(params: dict, scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_jobs=-1, random_state=42, verbosity=-1,
        scale_pos_weight=scale_pos_weight, subsample_freq=1, **params,
    )


SEARCH_SPACE = {
    "n_estimators": tune.lograndint(50, 800),
    "num_leaves": tune.randint(7, 256),
    "max_depth": tune.randint(3, 12),
    "learning_rate": tune.loguniform(0.01, 0.3),
    "min_child_samples": tune.randint(5, 100),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0),
    "reg_lambda": tune.loguniform(1e-3, 5.0),
    "reg_alpha": tune.loguniform(1e-3, 5.0),
}
SEED = {"n_estimators": 300, "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "reg_alpha": 0.001}


if __name__ == "__main__":
    run_finetune("lightgbm", build, SEARCH_SPACE, SEED, INT_KEYS)

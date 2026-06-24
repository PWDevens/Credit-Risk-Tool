"""Fine-tuned PD model — FLAML-tuned, calibrated ElasticNet Logistic Regression challenger.

The interpretable, regulator-friendly model in the bakeoff. Logistic is linear in its inputs,
so it won't out-discriminate the GBMs on AUC — but ElasticNet (L1+L2) handles the heavy
multicollinearity the diagnostic found (it selects away duplicates) and yields an auditable,
points-style scorecard. Shares the harness in common/finetune.py; differs only in the
estimator + search space. Uses scale_numeric=True (median-impute + StandardScaler) because
logistic, unlike the trees, cannot take NaN or raw feature scales.

Run:  python modeling/probability-of-default/finetune_logistic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "modeling"))

from flaml import tune  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from common.finetune import run_finetune  # noqa: E402

INT_KEYS: list = []
SCALE_NUMERIC = True


def build(params: dict, scale_pos_weight: float) -> LogisticRegression:
    # class_weight='balanced' handles the ~24% bad rate (scale_pos_weight is a tree concept).
    # max_iter trimmed + looser tol so saga (slow on the ~190 one-hot columns) converges in
    # a tractable time for the matrix; ample for a calibrated AUC.
    return LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight="balanced",
        max_iter=2000, tol=1e-3, random_state=42, **params,
    )


SEARCH_SPACE = {
    "C": tune.loguniform(1e-3, 10.0),       # inverse regularization strength
    "l1_ratio": tune.uniform(0.0, 1.0),     # 0 = ridge (L2), 1 = lasso (L1), between = elasticnet
}
SEED = {"C": 1.0, "l1_ratio": 0.5}


if __name__ == "__main__":
    run_finetune("logistic", build, SEARCH_SPACE, SEED, INT_KEYS, scale_numeric=SCALE_NUMERIC)

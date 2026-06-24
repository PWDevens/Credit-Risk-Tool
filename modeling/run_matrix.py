"""Run the 4-model x 4-feature-set PD matrix.

Models : XGBoost, LightGBM, Random Forest, ElasticNet Logistic Regression.
Versions (what each adds on top of the base features):
  v1_base       : base features only (no cluster, no engineering)
  v2_cluster    : base + RiskCluster
  v3_engineered : base + engineered features
  v4_all        : base + RiskCluster + engineered features

AutoML is NOT re-run here — it stays the existing FE-free baseline (pd_baseline_automl.csv),
shown as the reference line in results_visual.py. Writes model-results/version_matrix.csv.

Budget per cell via FLAML_TIME_BUDGET (default 120s). 16 cells.
  FLAML_TIME_BUDGET=120 python modeling/run_matrix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "modeling" / "probability-of-default"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from common.finetune import evaluate_featureset, RESULTS_DIR  # noqa: E402
import finetune_xgboost as xgb   # noqa: E402
import finetune_lightgbm as lgbm  # noqa: E402
import finetune_rf as rf          # noqa: E402
import finetune_logistic as logit  # noqa: E402

# (name, build, search_space, seed, int_keys, scale_numeric)
MODELS = [
    ("xgboost", xgb.build, xgb.SEARCH_SPACE, xgb.SEED, xgb.INT_KEYS, False),
    ("lightgbm", lgbm.build, lgbm.SEARCH_SPACE, lgbm.SEED, lgbm.INT_KEYS, False),
    ("rf", rf.build, rf.SEARCH_SPACE, rf.SEED, rf.INT_KEYS, False),
    ("logistic", logit.build, logit.SEARCH_SPACE, logit.SEED, logit.INT_KEYS, True),
]
# (version, include_engineered, include_cluster, include_macro)
VERSIONS = [
    ("v1_base", False, False, False),
    ("v2_cluster", False, True, False),
    ("v3_engineered", True, False, False),
    ("v4_all", True, True, False),
    ("v5_macro", True, True, True),   # v4 + point-in-time macro overlay
]


def main() -> None:
    rows = []
    for vname, eng, clu, mac in VERSIONS:
        for mname, build, space, seed, int_keys, scale in MODELS:
            print(f"\n##### {vname}  x  {mname} #####", flush=True)
            try:
                res = evaluate_featureset(mname, build, space, seed, int_keys,
                                          include_engineered=eng, include_cluster=clu,
                                          include_macro=mac, scale_numeric=scale)
            except Exception as exc:  # keep the matrix going if one cell errors
                print(f"  FAILED: {exc}")
                res = {"model": mname, "cv_auc": float("nan"), "test_auc_cal": float("nan")}
            res["version"] = vname
            rows.append(res)
            print(f"  -> cv_auc={res.get('cv_auc')}  test_auc_cal={res.get('test_auc_cal')}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "version_matrix.csv"
    df.to_csv(out, index=False)
    print("\n================ MATRIX COMPLETE ================")
    pivot = df.pivot(index="model", columns="version", values="test_auc_cal")
    pivot = pivot.reindex(columns=[v[0] for v in VERSIONS])
    print(pivot.to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

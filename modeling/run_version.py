"""Run ONE feature-set version across all four PD challengers, with per-version
IV/WOE + best-model explainability drivers. Reports everything for that version, then exits
(the caller runs v1..v5 in sequence and reports after each).

Versions (CUMULATIVE / nested ablation — each adds a layer on top of the previous):
  v1_base        base features only
  v2_cluster     base + RiskCluster
  v3_engineered  base + cluster + engineered features
  v4_macro       base + cluster + engineered + macro (national TTC; state overlay = future ver.)

Each bar answers "what does THIS layer add on top of everything before it." Macro is last,
the most honest test of it: does the economy add signal beyond the borrower's own data?

Constant references (not re-run): AutoML 0.750, Prosper grade champion 0.648.

  FLAML_TIME_BUDGET=180 python modeling/run_version.py v1_base

Appends the 4 model rows to model-results/version_matrix.csv (upsert by version) so
results_visual.py can plot all versions at the end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "modeling" / "probability-of-default"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from common.finetune import _train_one, RESULTS_DIR  # noqa: E402
from common import data as D, metrics as M  # noqa: E402
from diagnose_features import iv_of, band  # noqa: E402
import finetune_xgboost as xgb  # noqa: E402
import finetune_lightgbm as lgbm  # noqa: E402
import finetune_rf as rf  # noqa: E402
import finetune_logistic as logit  # noqa: E402

# version -> (include_engineered, include_cluster, macro_set)  [cumulative: each adds a layer]
VERSIONS = {
    "v1_base": (False, False, None),
    "v2_cluster": (False, True, None),
    "v3_engineered": (True, True, None),
    "v4_macro": (True, True, "ttc"),         # national TTC macro (state overlay = future version)
}
MODELS = [
    ("xgboost", xgb.build, xgb.SEARCH_SPACE, xgb.SEED, xgb.INT_KEYS, False),
    ("lightgbm", lgbm.build, lgbm.SEARCH_SPACE, lgbm.SEED, lgbm.INT_KEYS, False),
    ("rf", rf.build, rf.SEARCH_SPACE, rf.SEED, rf.INT_KEYS, False),
    ("logistic", logit.build, logit.SEARCH_SPACE, logit.SEED, logit.INT_KEYS, True),
]


def _drivers(name: str, r: dict, top: int = 12):
    """Top explainability drivers for the best model: SHAP for trees, |coef| for logistic."""
    Xe, est = r["Xte_e"], r["est"]
    feats = list(Xe.columns)
    if name == "logistic":
        coef = np.asarray(est.coef_).reshape(-1)
        order = np.argsort(np.abs(coef))[::-1][:top]
        return "standardized coefficient", [(feats[i], float(coef[i])) for i in order]
    import shap
    Xs = Xe.sample(min(500, len(Xe)), random_state=42)
    sv = shap.TreeExplainer(est).shap_values(Xs)
    if isinstance(sv, list):
        vals = np.asarray(sv[1])
    else:
        vals = np.asarray(sv)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top]
    return "mean |SHAP|", [(feats[i], float(mean_abs[i])) for i in order]


def main(version: str) -> None:
    if version not in VERSIONS:
        raise SystemExit(f"unknown version {version!r}; choices: {list(VERSIONS)}")
    eng, clu, mac = VERSIONS[version]        # mac is a macro_set string or None
    budget = int(os.environ.get("FLAML_TIME_BUDGET", "180"))
    print(f"\n############### {version}  "
          f"(engineered={eng}, cluster={clu}, macro={mac}, budget={budget}s) ###############")

    train = D.load_frame("train")
    cols = D._feature_cols(eng, clu, mac)
    print(f"\n=== {version}: IV / WOE table (all {len(cols)} features) ===")
    iv = sorted(((c, iv_of(train, c)) for c in cols), key=lambda kv: -kv[1])
    print(f"{'feature':46s} {'IV':>8s}  band")
    for c, v in iv:
        print(f"{c:46s} {v:8.4f}  {band(v)}")

    rows, arts = [], {}
    for mname, build, space, seed, int_keys, scale in MODELS:
        print(f"\n--- {version} x {mname}: tuning ({budget}s) ---", flush=True)
        r = _train_one(build, space, seed, int_keys, include_engineered=eng,
                       include_cluster=clu, macro_set=mac, scale_numeric=scale, budget=budget)
        mc = M.pd_metrics(r["yte"], r["cal_prob"])
        row = {"version": version, "model": mname, "cv_auc": round(r["cv_auc"], 4),
               "test_auc_cal": round(mc["AUC"], 4), "gini": round(mc["Gini"], 4),
               "ks": round(mc["KS"], 4), "brier": round(mc["Brier"], 4),
               "n_features": len(r["feature_cols"])}
        rows.append(row)
        arts[mname] = r
        print(f"   {mname}: cv_auc={row['cv_auc']}  test_auc_cal={row['test_auc_cal']}", flush=True)

    res = pd.DataFrame(rows).sort_values("test_auc_cal", ascending=False)
    print(f"\n=== {version}: MODEL RESULTS ===")
    print(res.to_string(index=False))

    best = res.iloc[0]["model"]
    kind, drv = _drivers(best, arts[best])
    print(f"\n=== {version}: top drivers for best model = {best} ({kind}) ===")
    for feat, val in drv:
        print(f"  {val:+12.4f}  {feat}")

    out = RESULTS_DIR / "version_matrix.csv"
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev["version"] != version]
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(out, index=False)
    print(f"\nsaved/updated {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1_base")

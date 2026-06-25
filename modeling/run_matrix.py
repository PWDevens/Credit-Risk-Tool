"""Crash-resilient runner for the cumulative v1->v4 feature-set matrix.

run_version.py runs all four models in ONE process, so a native crash (OpenMP/MKL conflict —
seen on the logistic and lightgbm cells) takes down the whole version with no traceback. This
runner isolates every (version, model) cell in its own subprocess with capped math-library
threads, appends to version_matrix.csv incrementally, and is resumable (skips cells already in
the CSV). Then it regenerates the chart.

AutoML is NOT re-run — it stays the FE-free baseline (pd_baseline_automl.csv), shown as the
reference line in results_visual.py.

  python modeling/run_matrix.py                 # run all missing cells, then plot
  FLAML_TIME_BUDGET=180 python modeling/run_matrix.py
  python modeling/run_matrix.py --cell v3_engineered logistic 180   # worker (internal)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "modeling" / "probability-of-default"))
sys.path.insert(0, str(REPO_ROOT / "data"))

RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
OUT = RESULTS_DIR / "version_matrix.csv"

# Cumulative (nested) feature-set versions -> (include_engineered, include_cluster, macro_set).
VERSIONS = {
    "v1_base": (False, False, None),
    "v2_cluster": (False, True, None),
    "v3_engineered": (True, True, None),
    "v4_macro": (True, True, "ttc"),     # national TTC macro (state overlay = future version)
}
MODELS = ["xgboost", "lightgbm", "rf", "logistic"]


def _model(name):
    import finetune_xgboost as xgb
    import finetune_lightgbm as lgbm
    import finetune_rf as rf
    import finetune_logistic as logit
    m = {"xgboost": (xgb, False), "lightgbm": (lgbm, False),
         "rf": (rf, False), "logistic": (logit, True)}
    mod, scale = m[name]
    return mod.build, mod.SEARCH_SPACE, mod.SEED, mod.INT_KEYS, scale


def run_cell(version: str, model: str, budget: int) -> dict:
    from common.finetune import _train_one
    from common import metrics as M
    eng, clu, macro = VERSIONS[version]
    build, space, seed, int_keys, scale = _model(model)
    r = _train_one(build, space, seed, int_keys, include_engineered=eng, include_cluster=clu,
                   macro_set=macro, scale_numeric=scale, budget=budget)
    mc = M.pd_metrics(r["yte"], r["cal_prob"])
    return {"version": version, "model": model, "cv_auc": round(r["cv_auc"], 4),
            "test_auc_cal": round(mc["AUC"], 4), "gini": round(mc["Gini"], 4),
            "ks": round(mc["KS"], 4), "brier": round(mc["Brier"], 4),
            "n_features": len(r["feature_cols"])}


def _done() -> set:
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT)
    return set(zip(df["version"], df["model"]))


def _append(row: dict) -> None:
    df = (pd.concat([pd.read_csv(OUT), pd.DataFrame([row])], ignore_index=True)
          if OUT.exists() else pd.DataFrame([row]))
    df = df.drop_duplicates(["version", "model"], keep="last")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)


def main() -> None:
    budget = int(os.environ.get("FLAML_TIME_BUDGET", "180"))
    # Cap every math-library thread pool + tolerate duplicate OpenMP runtimes (the crash cause).
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               KMP_DUPLICATE_LIB_OK="TRUE")
    done = _done()
    todo = [(v, m) for v in VERSIONS for m in MODELS if (v, m) not in done]
    print(f"{len(done)} cells already done; running {len(todo)} (budget={budget}s each)")
    for version, model in todo:
        tag = f"{version} x {model}"
        print(f"\n##### {tag} #####", flush=True)
        cmd = [sys.executable, __file__, "--cell", version, model, str(budget)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=budget * 6, env=env)
            row = None
            for line in p.stdout.splitlines():
                if line.startswith("RESULT "):
                    row = json.loads(line[len("RESULT "):])
            if row is None:
                print(f"  FAILED (exit {p.returncode}): " + " | ".join(p.stderr.splitlines()[-3:]))
                continue
            _append(row)
            print(f"  -> {tag}: test_auc_cal={row['test_auc_cal']}  cv_auc={row['cv_auc']}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT {tag}")

    print("\nregenerating chart ...", flush=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "modeling" / "results_visual.py")], env=env)
    print("DONE")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--cell":
        print(f"RESULT {json.dumps(run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4])))}")
    else:
        main()

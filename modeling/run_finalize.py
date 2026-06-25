"""Finalize Part C: the base-model (v1, no macro) OUT-OF-TIME reference + backfill the one
failed cell, so the comparison is apples-to-apples. Isolated subprocesses (OMP capped) so a
crash can't take the run down; results upserted into model-results/partc_matrix.csv.

  FLAML_TIME_BUDGET=60 python modeling/run_finalize.py
"""
from __future__ import annotations

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
OUT = RESULTS_DIR / "partc_matrix.csv"

# (version, include_engineered, include_cluster, macro, split, model)  macro: none|raw|ttc|robust
CELLS = [
    ("v1_base", "0", "0", "none", "oot", "xgboost"),    # base OOT reference
    ("v1_base", "0", "0", "none", "oot", "lightgbm"),
    ("v1_base", "0", "0", "none", "oot", "rf"),
    ("v1_base", "0", "0", "none", "oot", "logistic"),
    ("v5_all", "1", "1", "ttc", "random", "lightgbm"),  # backfill the one failed cell
]


def _model(name):
    import finetune_xgboost as xgb
    import finetune_lightgbm as lgbm
    import finetune_rf as rf
    import finetune_logistic as logit
    m = {"xgboost": (xgb, False), "lightgbm": (lgbm, False),
         "rf": (rf, False), "logistic": (logit, True)}
    mod, scale = m[name]
    return mod.build, mod.SEARCH_SPACE, mod.SEED, mod.INT_KEYS, scale


def run_cell(version, eng, clu, macro, split, model, budget) -> float:
    from common.finetune import _train_one
    from common import metrics as M
    build, space, seed, int_keys, scale = _model(model)
    macro_set = None if macro == "none" else macro
    r = _train_one(build, space, seed, int_keys, include_engineered=(eng == "1"),
                   include_cluster=(clu == "1"), macro_set=macro_set, split_mode=split,
                   scale_numeric=scale, budget=int(budget))
    return float(M.pd_metrics(r["yte"], r["cal_prob"])["AUC"])


def main() -> None:
    budget = int(os.environ.get("FLAML_TIME_BUDGET", "60"))
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    rows = []
    for (version, eng, clu, macro, split, model) in CELLS:
        tag = f"{version} | {split}+{macro} | {model}"
        print(f"\n##### {tag}  (budget={budget}s) #####", flush=True)
        cmd = [sys.executable, __file__, "--cell", version, eng, clu, macro, split, model, str(budget)]
        auc = float("nan")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=budget * 5, env=env)
            for line in p.stdout.splitlines():
                if line.startswith("RESULT "):
                    auc = round(float(line.split()[1]), 4)
            if auc != auc:
                print(f"  no RESULT (exit {p.returncode}): " + " | ".join(p.stderr.splitlines()[-2:]))
        except Exception as exc:  # noqa: BLE001
            print(f"  crash/timeout: {exc}")
        rows.append({"version": version, "split": split, "macro": macro,
                     "model": model, "test_auc_cal": auc})
        print(f"  -> {tag}: test_auc_cal={auc}", flush=True)

    new = pd.DataFrame(rows)
    key = ["version", "split", "macro", "model"]
    if OUT.exists():
        prev = pd.read_csv(OUT)
        prev = prev[~prev.set_index(key).index.isin(new.set_index(key).index)]
        out = pd.concat([prev, new], ignore_index=True)
    else:
        out = new
    out.to_csv(OUT, index=False)
    print(f"\nDONE — updated {OUT}")


if __name__ == "__main__":
    if len(sys.argv) >= 9 and sys.argv[1] == "--cell":
        v, e, c, ma, s, mo, b = sys.argv[2:9]
        print(f"RESULT {run_cell(v, e, c, ma, s, mo, b):.6f}")
    else:
        main()

"""Part C — does the macro lift survive out-of-time, and does TTC anchoring help?

Reruns v3 (base + macro) and v5 (base + cluster + engineered + macro) across the 2x2 grid of
  split in {random, oot}   (oot: train < 2013, test 2013-14 — unseen vintages)
  macro in {raw, ttc}       (raw point-in-time level vs TTC-anchored + cyclical gap)
for all four models. Writes model-results/partc_matrix.csv (incrementally) + prints the table.

Robustness: each cell runs in its OWN subprocess so a hard crash (e.g. LightGBM/OpenMP) only
loses that cell, not the whole run. Results are appended cell-by-cell and the run is RESUMABLE
(re-running skips cells already in the CSV). Threads are capped to avoid OpenMP oversubscription.

  FLAML_TIME_BUDGET=120 python modeling/run_partc.py            # orchestrator (run all cells)
  python modeling/run_partc.py --cell v3_macro oot ttc xgboost 120   # one cell (worker)
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

# version -> (include_engineered, include_cluster)
PARTC_VERSIONS = {"v3_macro": (False, False), "v5_all": (True, True)}
CONDITIONS = [("random", "raw"), ("random", "ttc"), ("oot", "raw"), ("oot", "ttc")]
MODEL_NAMES = ["xgboost", "lightgbm", "rf", "logistic"]


def _model(name):
    import finetune_xgboost as xgb
    import finetune_lightgbm as lgbm
    import finetune_rf as rf
    import finetune_logistic as logit
    m = {"xgboost": (xgb, False), "lightgbm": (lgbm, False),
         "rf": (rf, False), "logistic": (logit, True)}
    mod, scale = m[name]
    return mod.build, mod.SEARCH_SPACE, mod.SEED, mod.INT_KEYS, scale


def run_cell(version, split, macro, model_name, budget) -> float:
    """Worker: train one (version, split, macro, model) cell; return calibrated test AUC."""
    from common.finetune import _train_one
    from common import metrics as M
    eng, clu = PARTC_VERSIONS[version]
    build, space, seed, int_keys, scale = _model(model_name)
    r = _train_one(build, space, seed, int_keys, include_engineered=eng, include_cluster=clu,
                   macro_set=macro, split_mode=split, scale_numeric=scale, budget=int(budget))
    return float(M.pd_metrics(r["yte"], r["cal_prob"])["AUC"])


def _append(version, split, macro, model, auc):
    row = pd.DataFrame([{"version": version, "split": split, "macro": macro,
                         "model": model, "test_auc_cal": auc}])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def _print_tables() -> None:
    df = pd.read_csv(OUT)
    df["cond"] = df["split"] + "+" + df["macro"]
    order = ["random+raw", "random+ttc", "oot+raw", "oot+ttc"]
    print("\n================ PART C: macro lift, random vs OOT, raw vs TTC ================")
    for vname in PARTC_VERSIONS:
        sub = df[df["version"] == vname]
        if sub.empty:
            continue
        print(f"\n--- {vname} (calibrated test AUC) ---")
        piv = sub.pivot_table(index="model", columns="cond", values="test_auc_cal").reindex(columns=order)
        print(piv.to_string())


def orchestrate() -> None:
    budget = int(os.environ.get("FLAML_TIME_BUDGET", "120"))
    done = set()
    if OUT.exists():
        for r in pd.read_csv(OUT).itertuples():
            done.add((r.version, r.split, r.macro, r.model))

    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")  # avoid OpenMP crash
    for version in PARTC_VERSIONS:
        for split, macro in CONDITIONS:
            for model in MODEL_NAMES:
                key = (version, split, macro, model)
                if key in done:
                    print(f"skip (done): {version} {split}+{macro} {model}", flush=True)
                    continue
                tag = f"{version} | {split}+{macro} | {model}"
                print(f"\n##### {tag}  (budget={budget}s) #####", flush=True)
                cmd = [sys.executable, __file__, "--cell", version, split, macro, model, str(budget)]
                auc = float("nan")
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=budget * 5, env=env)
                    for line in p.stdout.splitlines():
                        if line.startswith("RESULT "):
                            auc = round(float(line.split()[1]), 4)
                    if auc != auc:  # nan -> show why
                        print(f"  no RESULT (exit {p.returncode}). stderr tail:\n"
                              + "\n".join(p.stderr.splitlines()[-3:]))
                except Exception as exc:  # noqa: BLE001
                    print(f"  cell crashed/timed out: {exc}")
                _append(version, split, macro, model, auc)
                print(f"  -> {tag}: test_auc_cal={auc}", flush=True)

    _print_tables()
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    if len(sys.argv) >= 7 and sys.argv[1] == "--cell":
        _v, _s, _m, _mod, _b = sys.argv[2:7]
        print(f"RESULT {run_cell(_v, _s, _m, _mod, _b):.6f}")
    else:
        orchestrate()

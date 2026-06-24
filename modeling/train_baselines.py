"""Train the three AutoGluon baseline models (PD / EAD / LGD) on the shared split.

Saves each predictor under its metric folder, writes a metrics table per metric to
modeling/model-results/, and dumps models/feature_defaults.json (median/mode per feature)
so the frontend can score from partial inputs.

Training budget is intentionally small for fast iteration — raise it for a real baseline:
    AUTOML_TIME_LIMIT=600 AUTOML_PRESET=best_quality python modeling/train_baselines.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

import features as F  # noqa: E402
from autogluon.tabular import TabularPredictor  # noqa: E402
from common import data as D, metrics as M  # noqa: E402

TIME_LIMIT = int(os.environ.get("AUTOML_TIME_LIMIT", "120"))
PRESET = os.environ.get("AUTOML_PRESET", "medium_quality")

RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
MODELS_DIR = REPO_ROOT / "models"
PD_DIR = REPO_ROOT / "modeling" / "probability-of-default" / "automl_model"
EAD_DIR = REPO_ROOT / "modeling" / "exposure-at-default" / "automl_model"
LGD_DIR = REPO_ROOT / "modeling" / "loss-given-default" / "automl_model"


def _save_metrics(name: str, rows: list[dict]) -> None:
    out = RESULTS_DIR / f"{name}_baseline_automl.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  -> {out}")


def _dump_defaults(train: pd.DataFrame) -> None:
    """Median (numeric) / mode (categorical) per feature, from the full PD population."""
    X, _ = D.pd_Xy(train)
    defaults: dict = {}
    for col in F.MODEL_FEATURES:
        s = X[col]
        if col in F.CATEGORICAL_FEATURES:
            mode = s.mode(dropna=True)
            defaults[col] = (str(mode.iloc[0]) if not mode.empty else None)
        else:
            defaults[col] = float(pd.to_numeric(s, errors="coerce").median())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "feature_defaults.json").write_text(json.dumps(defaults, indent=2))
    print(f"  -> {MODELS_DIR / 'feature_defaults.json'} ({len(defaults)} features)")


def train_pd(train, test):
    print("\n[PD] probability of default — full resolved population")
    Xtr, ytr = D.pd_Xy(train)
    Xte, yte = D.pd_Xy(test)
    pred = TabularPredictor(label=F.PD_TARGET, problem_type="binary", eval_metric="roc_auc",
                            path=str(PD_DIR)).fit(
        Xtr.assign(**{F.PD_TARGET: ytr}), time_limit=TIME_LIMIT, presets=PRESET, verbosity=1)
    proba = pred.predict_proba(Xte)
    pcol = 1 if 1 in proba.columns else proba.columns[-1]
    model = M.pd_metrics(yte, proba[pcol].to_numpy())

    rows = [{"model": "automl", **model}]
    # Champion benchmark: Prosper's own grade ranking (higher ProsperScore = lower risk).
    champ = test[["ProsperScore", F.PD_TARGET]].dropna()
    if len(champ) > 100:
        risk = -champ["ProsperScore"].to_numpy()  # invert: higher score -> lower PD
        rows.append({"model": "prosper_score_champion",
                     **M.pd_metrics(champ[F.PD_TARGET], risk, calibration=False)})
    _save_metrics("pd", rows)
    return model


def train_reg(name, dir_, Xy, train, test, lazy_fn):
    print(f"\n[{name.upper()}] — defaulted loans only")
    Xtr, ytr = Xy(train)
    Xte, yte = Xy(test)
    pred = TabularPredictor(label="_y", problem_type="regression",
                            eval_metric="root_mean_squared_error", path=str(dir_)).fit(
        Xtr.assign(_y=ytr.to_numpy()), time_limit=TIME_LIMIT, presets=PRESET, verbosity=1)
    yhat = pred.predict(Xte).to_numpy()
    rows = [{"model": "automl", **M.reg_metrics(yte, yhat)}]
    rows.append({"model": "lazy_baseline", **M.reg_metrics(yte, lazy_fn(train, test, ytr))})
    _save_metrics(name, rows)
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resplit train/test from raw first, so baselines always train on a fresh,
    # reproducible split (random_state fixed in features.py). Skip with SKIP_RESPLIT=1.
    if os.environ.get("SKIP_RESPLIT") != "1":
        print("Resplitting train/test from raw via features.py ...")
        F.main()

    train, test = D.load_frame("train"), D.load_frame("test")
    _dump_defaults(train)

    train_pd(train, test)

    # EAD lazy baseline: assume full exposure (= LoanOriginalAmount).
    def ead_lazy(tr, te, ytr):
        d = te[te[F.PD_TARGET] == 1]
        y = F.build_ead_label(d)
        return d.loc[y.notna(), "LoanOriginalAmount"].to_numpy()

    train_reg("ead", EAD_DIR, D.ead_Xy, train, test, ead_lazy)

    # LGD lazy baseline: constant mean LGD over the training defaults.
    def lgd_lazy(tr, te, ytr):
        d = te[te[F.PD_TARGET] == 1]
        y = F.build_lgd_label(d)
        n = int(y.notna().sum())
        return [float(ytr.mean())] * n

    train_reg("lgd", LGD_DIR, D.lgd_Xy, train, test, lgd_lazy)
    print("\nDone. Baselines trained, metrics + feature_defaults.json written.")


if __name__ == "__main__":
    main()

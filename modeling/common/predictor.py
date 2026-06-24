"""RiskPredictor — the single in-process scoring API the frontend uses.

Loads the three AutoGluon predictors + feature defaults, and exposes assess(), which
fills any unspecified feature with its trained default and returns PD / LGD / EAD / EL.
The `family` argument is the AutoML-vs-fine-tuned toggle hook (only 'automl' at v1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data"))
import features as F  # noqa: E402

MODEL_DIRS = {
    "automl": {
        "pd": REPO_ROOT / "modeling" / "probability-of-default" / "automl_model",
        "ead": REPO_ROOT / "modeling" / "exposure-at-default" / "automl_model",
        "lgd": REPO_ROOT / "modeling" / "loss-given-default" / "automl_model",
    },
}
DEFAULTS_PATH = REPO_ROOT / "models" / "feature_defaults.json"


def available_families() -> list[str]:
    """Families whose three model dirs all exist on disk."""
    return [
        fam
        for fam, dirs in MODEL_DIRS.items()
        if all(p.exists() for p in dirs.values())
    ]


class RiskPredictor:
    def __init__(self, family: str = "automl"):
        if family not in MODEL_DIRS:
            raise ValueError(f"unknown family {family!r}; choices: {list(MODEL_DIRS)}")
        self.family = family
        dirs = MODEL_DIRS[family]
        self.models = {k: TabularPredictor.load(str(p)) for k, p in dirs.items()}
        self.defaults = json.loads(DEFAULTS_PATH.read_text())

    def _row(self, inputs: dict) -> pd.DataFrame:
        row = dict(self.defaults)
        row.update({k: v for k, v in inputs.items() if v is not None})
        df = pd.DataFrame([row]).reindex(columns=F.MODEL_FEATURES)
        return F.cast_categoricals(df)

    def assess(self, inputs: dict) -> dict:
        """inputs: a (partial) dict over features.MODEL_FEATURES. Returns pd/lgd/ead/el."""
        X = self._row(inputs)

        proba = self.models["pd"].predict_proba(X)
        pcol = 1 if 1 in proba.columns else proba.columns[-1]
        pd_ = float(proba[pcol].iloc[0])

        ead = max(float(self.models["ead"].predict(X).iloc[0]), 0.0)
        lgd = float(np.clip(self.models["lgd"].predict(X).iloc[0], 0.0, 1.0))
        return {"pd": pd_, "lgd": lgd, "ead": ead, "el": pd_ * lgd * ead}

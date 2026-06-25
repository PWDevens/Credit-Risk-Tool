"""RiskPredictor — the single in-process scoring API the frontend uses.

EAD and LGD are always the AutoGluon baselines. PD can come from either the AutoGluon
baseline ('automl') or the fine-tuned, calibrated XGBoost challenger ('finetuned') —
the `family` argument is the app's toggle. Only PD has a tuned challenger; EAD/LGD
are identical across families. For the finetuned family, explain_pd() returns per-borrower
SHAP reason codes from the XGBoost booster (the adverse-action-style "why").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "modeling"))
import features as F  # noqa: E402
from survival import term_structure as TS  # noqa: E402  (hazard PD term structure; available() guards use)

PD_DIR = REPO_ROOT / "modeling" / "probability-of-default" / "automl_model"
EAD_DIR = REPO_ROOT / "modeling" / "exposure-at-default" / "automl_model"
LGD_DIR = REPO_ROOT / "modeling" / "loss-given-default" / "automl_model"
PD_FINE = REPO_ROOT / "modeling" / "probability-of-default" / "pd_xgboost.joblib"
DEFAULTS_PATH = REPO_ROOT / "models" / "feature_defaults.json"

FAMILY_LABELS = {"automl": "AutoML baseline", "finetuned": "XGBoost (fine-tuned PD)"}


def available_families() -> list[str]:
    """Families whose required artifacts all exist on disk (EAD/LGD shared)."""
    fams = []
    reg = EAD_DIR.exists() and LGD_DIR.exists()
    if PD_DIR.exists() and reg:
        fams.append("automl")
    if PD_FINE.exists() and reg:
        fams.append("finetuned")
    return fams


class RiskPredictor:
    def __init__(self):
        self.ead = TabularPredictor.load(str(EAD_DIR))
        self.lgd = TabularPredictor.load(str(LGD_DIR))
        self.pd_automl = TabularPredictor.load(str(PD_DIR)) if PD_DIR.exists() else None
        self.pd_fine = joblib.load(PD_FINE) if PD_FINE.exists() else None
        # Persisted KMeans pipeline (impute->quantile-scale->cluster) for RiskCluster; the
        # SAME object fit on train is applied here in production (no re-fit, no leakage).
        self.risk_cluster = joblib.load(F.RISK_CLUSTER_PATH) if F.RISK_CLUSTER_PATH.exists() else None
        # Latest TTC-anchored macro (a new loan is originated 'now'); a through-the-cycle
        # calibration that shifts the PD level with the economy. Cached once.
        self.macro = F.current_macro()
        self.defaults = json.loads(DEFAULTS_PATH.read_text())
        self._explainer = None

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reproduce, for a scoring row, exactly what the fine-tuned model saw in training:
        engineered features, the RiskCluster label, and the (TTC-anchored) macro overlay at
        today's conditions. Columns the model wasn't trained on are dropped by the preprocessor."""
        Xfe = F.feature_engineering(X)
        if self.risk_cluster is not None:
            Xfe["RiskCluster"] = F.assign_risk_cluster(Xfe, self.risk_cluster)
        for col, val in self.macro.items():
            Xfe[col] = val
        return Xfe

    def _row(self, inputs: dict) -> pd.DataFrame:
        row = dict(self.defaults)
        row.update({k: v for k, v in inputs.items() if v is not None})
        df = pd.DataFrame([row]).reindex(columns=F.MODEL_FEATURES)
        return F.cast_categoricals(df)

    def _pd(self, X: pd.DataFrame, family: str) -> float:
        if family == "finetuned":
            if self.pd_fine is None:
                raise RuntimeError("XGBoost PD model not found — run finetune_xgboost.py")
            # The fine-tuned model was trained on base + engineered + RiskCluster + TTC macro,
            # so reproduce all of them on the scoring row before the preprocessor sees it.
            Xe = self.pd_fine["preprocessor"].transform(self._engineer(X))
            return float(self.pd_fine["model"].predict_proba(Xe)[:, 1][0])
        proba = self.pd_automl.predict_proba(X)
        pcol = 1 if 1 in proba.columns else proba.columns[-1]
        return float(proba[pcol].iloc[0])

    def assess(self, inputs: dict, family: str = "automl") -> dict:
        """inputs: a (partial) dict over features.MODEL_FEATURES. Returns pd/lgd/ead/el."""
        X = self._row(inputs)
        pd_ = self._pd(X, family)
        ead = max(float(self.ead.predict(X).iloc[0]), 0.0)
        lgd = float(np.clip(self.lgd.predict(X).iloc[0], 0.0, 1.0))
        return {"pd": pd_, "lgd": lgd, "ead": ead, "el": pd_ * lgd * ead, "family": family}

    def term_structure(self, inputs: dict, term: int):
        """Borrower-specific monthly marginal-default vector from the discrete-time hazard model
        (modeling/survival/term_structure.py), or None if the hazard artifact is absent.

        Built on the SAME engineered + TTC-macro row the PD model scores (via .to_dict('records')
        to preserve column dtypes), so the loss timing is consistent with the rest of the
        assessment. The vector sums to the hazard's own lifetime PD; the caller decides how to
        combine its *shape* with the headline lifetime PD.
        """
        if not TS.available():
            return None
        row = self._engineer(self._row(inputs)).to_dict("records")[0]
        m = TS.marginal_pd(TS.hazard_curve(row, int(term)))
        return m if m.size and float(m.sum()) > 0 else None

    def economic_context(self) -> dict:
        """The TTC-anchored macro overlay currently feeding PD (percent units), for display.

        Empty if no macro file is present. These values shift the whole PD *level* with the
        cycle (reserves/pricing), not the applicant ranking — see docs/01-feature-engineering.md.
        """
        return dict(self.macro)

    def explain_pd(self, inputs: dict, top: int = 6) -> list[tuple[str, float]]:
        """Top SHAP drivers of this borrower's PD from the XGBoost booster.

        Returns (encoded_feature, log_odds_contribution); positive raises PD. SHAP
        explains the pre-calibration booster, the honest "what drove the score" story.
        """
        if self.pd_fine is None:
            return []
        import shap

        X = self._row(inputs)
        Xe = self.pd_fine["preprocessor"].transform(self._engineer(X))
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.pd_fine["estimator"])
        sv = self._explainer.shap_values(Xe)
        vals = np.asarray(sv[1] if isinstance(sv, list) else sv).reshape(-1)
        feats = list(Xe.columns)
        # Macro / regional features are identical for every applicant on a given day — they shift
        # the whole PD level with the cycle, but they don't explain why THIS borrower is riskier
        # than the next. So drop them from the per-borrower "Why?" reason codes. They still drive
        # the model; they're just not a valid adverse-action explanation.
        ranked = [i for i in np.argsort(np.abs(vals))[::-1]
                  if not feats[i].startswith(("macro_", "state_"))]
        return [(feats[i], float(vals[i])) for i in ranked[:top]]

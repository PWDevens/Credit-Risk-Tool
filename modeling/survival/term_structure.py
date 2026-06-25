"""Phase 2 — turn a discrete-time hazard h(t|x) into a PD term structure.

Given a fitted hazard model (the chance of defaulting in month t, given survival to t), these
pure functions produce, for one loan:
  * survival  S(t)         = chance still alive at the end of month t
  * marginal_pd(t)         = chance of defaulting *in* month t   (this is the term structure)
  * lifetime_pd            = chance of ever defaulting over the loan's life

The marginal-PD vector is exactly what modeling/common/finance.py needs to spread loss over the
loan's life — a borrower-specific shape instead of one fixed empirical curve.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HAZARD_PATH = REPO_ROOT / "modeling" / "probability-of-default" / "pd_hazard_xgboost.joblib"
_EPS = 1e-6
_CACHE: dict = {}


def available() -> bool:
    """True if the fitted hazard artifact exists (the finance engine falls back if not)."""
    return HAZARD_PATH.exists()


def _load(artifact=None):
    if artifact is not None:
        return artifact
    if "art" not in _CACHE:
        _CACHE["art"] = joblib.load(HAZARD_PATH)
    return _CACHE["art"]


def hazard_curve(loan_row, term: int, artifact=None) -> np.ndarray:
    """Monthly hazards h[1..term] for one borrower: build a (term x covariate) frame with the
    borrower's covariates held constant and the time features (`t`, `t_frac`) varying, then score
    the fitted hazard model. `loan_row` is a dict / Series of covariate -> value."""
    term = int(term)
    if term <= 0:
        return np.array([])
    art = _load(artifact)
    cols = art["feature_cols"]
    row = dict(loan_row)
    t = np.arange(1, term + 1)
    data = {}
    for c in cols:
        if c == "t":
            data[c] = t
        elif c == "t_frac":
            data[c] = t / term
        else:
            data[c] = row.get(c, np.nan)
    X = pd.DataFrame(data, columns=cols)
    h = art["model"].predict_proba(art["preprocessor"].transform(X))[:, 1]
    return np.clip(h, _EPS, 1 - _EPS)


def survival_from_hazard(h) -> np.ndarray:
    """S(t) = product over k<=t of (1 - h[k]); chance of surviving to the end of month t."""
    h = np.clip(np.asarray(h, dtype=float), _EPS, 1 - _EPS)
    return np.cumprod(1.0 - h) if h.size else h


def marginal_pd(h) -> np.ndarray:
    """marginal_pd(t) = S(t-1) * h[t]: chance of defaulting *in* month t. Length == len(h); sums to
    the lifetime PD."""
    h = np.clip(np.asarray(h, dtype=float), _EPS, 1 - _EPS)
    if h.size == 0:
        return h
    s = np.cumprod(1.0 - h)
    s_prev = np.concatenate([[1.0], s[:-1]])
    return s_prev * h


def lifetime_pd(h) -> float:
    """1 - S(term): chance of ever defaulting over the loan's life."""
    h = np.clip(np.asarray(h, dtype=float), _EPS, 1 - _EPS)
    if h.size == 0:
        return 0.0
    return float(1.0 - np.cumprod(1.0 - h)[-1])

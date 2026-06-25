"""Load the processed splits and build per-metric (X, y) from the feature manifest.

PD trains on the full resolved population; EAD and LGD train on defaulted rows only,
with labels constructed from the LP_* support columns (see features.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data"))
import features as F  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_frame(split: str) -> pd.DataFrame:
    """split in {'train', 'test'}."""
    df = pd.read_csv(PROCESSED_DIR / f"{split}_data.csv", low_memory=False)
    return F.cast_categoricals(df)


OOT_CUTOFF = "2013-01-01"   # out-of-time split: train < cutoff, test >= cutoff (by origination)


def _feature_cols(include_engineered: bool = False, include_cluster: bool = False,
                  macro_set=None) -> list:
    """Base features, optionally + engineered numerics, + RiskCluster, + a macro set.

    macro_set: None | 'raw' (point-in-time level) | 'ttc' (TTC-anchored + gap). Used by the
    matrix and by Part C (raw vs TTC under random vs OOT). AutoML stays all-off.
    """
    cols = list(F.MODEL_FEATURES)
    if include_engineered:
        cols += F._ENGINEERED_NUMERIC
    if include_cluster:
        cols += ["RiskCluster"]
    if macro_set == "raw":
        cols += F.MACRO_FEATURES_RAW
    elif macro_set == "ttc":                 # canonical national (== F.MACRO_FEATURES)
        cols += F.MACRO_FEATURES_TTC
    elif macro_set == "ttc_geo":             # national + regional (state) TTC overlay
        cols += F.MACRO_FEATURES_TTC + F.STATE_FEATURES_TTC
    elif macro_set == "state":               # state overlay only (to isolate its marginal value)
        cols += F.STATE_FEATURES_TTC
    elif macro_set == "robust":
        cols += F.MACRO_FEATURES_ROBUST
    return cols


def pd_Xy(df: pd.DataFrame, include_engineered: bool = False, include_cluster: bool = False,
          include_macro: bool = False):
    """include_macro=True maps to the canonical (TTC-anchored) macro set."""
    cols = _feature_cols(include_engineered, include_cluster, "ttc" if include_macro else None)
    return df[cols].copy(), df[F.PD_TARGET].astype(int)


def pd_split(split_mode: str = "random", *, include_engineered: bool = False,
             include_cluster: bool = False, macro_set=None):
    """Return (X_train, y_train, X_test, y_test) for a given split mode + feature set.

    split_mode='random' uses the stored stratified split; 'oot' re-splits the full frame by
    LoanOriginationDate (train < OOT_CUTOFF, test >= OOT_CUTOFF) to test generalization to
    unseen vintages. Note: RiskCluster is fit on the random-train split, so under 'oot' it
    carries a mild fit-window mismatch (documented; cluster's marginal IV makes it minor).
    """
    cols = _feature_cols(include_engineered, include_cluster, macro_set)
    if split_mode == "random":
        tr, te = load_frame("train"), load_frame("test")
    elif split_mode == "oot":
        full = pd.concat([load_frame("train"), load_frame("test")], ignore_index=True)
        orig = pd.to_datetime(full["LoanOriginationDate"], errors="coerce")
        cut = pd.Timestamp(OOT_CUTOFF)
        tr, te = full[orig < cut].copy(), full[orig >= cut].copy()
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")
    return (tr[cols].copy(), tr[F.PD_TARGET].astype(int),
            te[cols].copy(), te[F.PD_TARGET].astype(int))


def _defaulted(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[F.PD_TARGET] == 1].copy()


def ead_Xy(df: pd.DataFrame, include_engineered: bool = False, include_cluster: bool = False):
    d = _defaulted(df)
    y = F.build_ead_label(d)
    m = y.notna()
    return d.loc[m, _feature_cols(include_engineered, include_cluster)].copy(), y[m]


def lgd_Xy(df: pd.DataFrame, include_engineered: bool = False, include_cluster: bool = False):
    d = _defaulted(df)
    y = F.build_lgd_label(d)
    m = y.notna()
    return d.loc[m, _feature_cols(include_engineered, include_cluster)].copy(), y[m]

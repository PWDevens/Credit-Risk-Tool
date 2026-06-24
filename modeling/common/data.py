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


def pd_Xy(df: pd.DataFrame):
    return df[F.MODEL_FEATURES].copy(), df[F.PD_TARGET].astype(int)


def _defaulted(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[F.PD_TARGET] == 1].copy()


def ead_Xy(df: pd.DataFrame):
    d = _defaulted(df)
    y = F.build_ead_label(d)
    m = y.notna()
    return d.loc[m, F.MODEL_FEATURES].copy(), y[m]


def lgd_Xy(df: pd.DataFrame):
    d = _defaulted(df)
    y = F.build_lgd_label(d)
    m = y.notna()
    return d.loc[m, F.MODEL_FEATURES].copy(), y[m]

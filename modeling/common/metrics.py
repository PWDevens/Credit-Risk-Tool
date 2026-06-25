"""Scorecard metrics. PD uses discrimination + calibration; EAD/LGD use regression error."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def pd_metrics(y_true, y_score, calibration: bool = True) -> dict:
    """Discrimination always; calibration (Brier/log-loss) only when y_score is a
    genuine probability in [0, 1]. A pure ranking (e.g. Prosper's grade) sets
    calibration=False — AUC/Gini/KS are rank-based and stay valid."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    auc = roc_auc_score(y_true, y_score)
    ks = ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic
    # PR_AUC (average precision) focuses on the positive/default class. SECONDARY metric: it is
    # prevalence-dependent (a random model scores BaseRate), so always read it next to BaseRate
    # and never compare it across splits with different default rates (e.g. random vs OOT).
    out = {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks,
           "PR_AUC": average_precision_score(y_true, y_score),
           "BaseRate": float(np.mean(y_true))}
    if calibration:
        out["Brier"] = brier_score_loss(y_true, y_score)
        out["LogLoss"] = log_loss(y_true, y_score)
    return out


def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Decile calibration: bucket borrowers by predicted PD, compare predicted vs actual.

    A well-calibrated PD matters more than a well-ranked one for Expected Loss / pricing — when
    the model says 5%, about 5% of those loans should default. Returns one row per bucket with
    the count, mean predicted PD, actual default rate, and the gap (predicted - actual).
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                       "p": np.asarray(y_score, dtype=float)})
    # qcut by predicted PD; drop duplicate edges if scores are bunched (duplicates='drop').
    df["bucket"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    g = (df.groupby("bucket")
         .agg(n=("y", "size"), pred_PD=("p", "mean"), actual_PD=("y", "mean"))
         .reset_index(drop=True))
    g["gap"] = g["pred_PD"] - g["actual_PD"]
    return g


def reg_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }

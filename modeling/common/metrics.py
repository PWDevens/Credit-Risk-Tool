"""Scorecard metrics. PD uses discrimination + calibration; EAD/LGD use regression error."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
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
    out = {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks}
    if calibration:
        out["Brier"] = brier_score_loss(y_true, y_score)
        out["LogLoss"] = log_loss(y_true, y_score)
    return out


def reg_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }

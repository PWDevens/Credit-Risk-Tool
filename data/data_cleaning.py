"""
================
Design principle
----------------
Imputation statistics (medians, modes, group medians) are LEARNED ON TRAIN
ONLY, returned to the caller, and then re-applied unchanged to validation /
test / scoring data. Computing these statistics over the full dataset leaks
holdout information into the training fold via the fill values — the exact
issue a model-risk validation review (SR 11-7) would flag.

Three imputation modes, per the request:
  * numeric  -> column median
  * categorical -> column mode
  * group-wise median -> median of a numeric column WITHIN each category of a
    grouping column (e.g. DebtToIncomeRatio imputed by IncomeRange band)

Plus, for credit data specifically:
  * constant_fill -> for informative nulls (e.g. prior-Prosper-loan fields are
    null because the borrower is a first-timer; that means 0, not "unknown")
  * add_missing_flags -> emit a `<col>_was_missing` indicator BEFORE filling,
    so the model can use missingness itself as signal (thin-file detection)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Policy: WHAT to do with each column. Contains no data — just the plan.       #
# --------------------------------------------------------------------------- #
@dataclass
class ImputationPlan:
    numeric_median: Sequence[str] = ()              # fill with column median
    categorical_mode: Sequence[str] = ()            # fill with column mode
    grouped_median: Mapping[str, str] = field(default_factory=dict)
    #   {column_to_fill: grouping_column}
    constant_fill: Mapping[str, Any] = field(default_factory=dict)
    #   {column: constant_value}   e.g. {"TotalProsperLoans": 0}
    add_missing_flags: Sequence[str] = ()           # emit <col>_was_missing


# --------------------------------------------------------------------------- #
# FIT: learn the fill values on training data only.                           #
# --------------------------------------------------------------------------- #
def fit_imputation(df: pd.DataFrame, plan: ImputationPlan) -> dict:
    """Compute and RETURN the median / mode / group-median values from `df`.

    The returned dict is the auditable artifact: persist it (json/pickle) and
    you have a frozen, reproducible imputation rule for scoring.
    """
    stats: dict[str, Any] = {
        "numeric_median": {},
        "categorical_mode": {},
        "grouped_median": {},
        "constant_fill": dict(plan.constant_fill),  # echoed for completeness
    }

    for col in plan.numeric_median:
        _require(df, col)
        med = df[col].median()
        if pd.isna(med):
            warnings.warn(f"[numeric_median] '{col}' is all-NaN on train; skipped.")
            continue
        stats["numeric_median"][col] = med

    for col in plan.categorical_mode:
        _require(df, col)
        modes = df[col].mode(dropna=True)
        if modes.empty:
            warnings.warn(f"[categorical_mode] '{col}' is all-NaN on train; skipped.")
            continue
        stats["categorical_mode"][col] = modes.iloc[0]

    for col, by in plan.grouped_median.items():
        _require(df, col)
        _require(df, by)
        group_med = df.groupby(by, observed=True)[col].median()
        stats["grouped_median"][col] = {
            "by": by,
            "groups": group_med.dropna().to_dict(),  # per-category medians
            "fallback": df[col].median(),            # for unseen/NaN groups
        }

    return stats


# --------------------------------------------------------------------------- #
# APPLY: fill a (possibly different) frame using learned stats.               #
# --------------------------------------------------------------------------- #
def apply_imputation(df: pd.DataFrame, plan: ImputationPlan, stats: dict) -> pd.DataFrame:
    """Return a cleaned COPY of `df` using stats learned by `fit_imputation`."""
    out = df.copy()

    # Missing flags first, so they capture the ORIGINAL nulls (pre-fill).
    for col in plan.add_missing_flags:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype("int8")

    # Group-wise median: map each null row's group -> that group's median,
    # falling back to the global median for unseen or NaN-median groups.
    for col, info in stats["grouped_median"].items():
        if col not in out.columns:
            continue
        by, groups, fallback = info["by"], info["groups"], info["fallback"]
        mask = out[col].isna()
        if mask.any():
            filled = out.loc[mask, by].map(groups)
            if not pd.isna(fallback):
                filled = filled.fillna(fallback)
            out.loc[mask, col] = filled

    # Plain numeric median.
    for col, med in stats["numeric_median"].items():
        if col in out.columns:
            out[col] = out[col].fillna(med)

    # Categorical mode.
    for col, mode_val in stats["categorical_mode"].items():
        if col in out.columns:
            out[col] = out[col].fillna(mode_val)

    # Constant fill (informative nulls).
    for col, const in stats["constant_fill"].items():
        if col in out.columns:
            out[col] = out[col].fillna(const)

    return out


# --------------------------------------------------------------------------- #
# Convenience wrapper: fit on train, apply to train (+ optional test).        #
# --------------------------------------------------------------------------- #
def clean_data(
    train: pd.DataFrame,
    plan: ImputationPlan,
    *,
    test: pd.DataFrame | None = None,
):
    """Fit on `train`, return (clean_train, stats) or (clean_train, clean_test, stats).

    The stats are fit on TRAIN ONLY and reused for TEST — never refit on test.
    """
    stats = fit_imputation(train, plan)
    clean_train = apply_imputation(train, plan, stats)
    if test is None:
        return clean_train, stats
    clean_test = apply_imputation(test, plan, stats)
    return clean_train, clean_test, stats


def _require(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")


# --------------------------------------------------------------------------- #
# Example plan tuned to the Prosper data dictionary.                          #
# --------------------------------------------------------------------------- #
PROSPER_PLAN = ImputationPlan(
    numeric_median=[
        "BankcardUtilization", "RevolvingCreditBalance", "AvailableBankcardCredit",
        "OpenRevolvingMonthlyPayment", "CurrentCreditLines", "OpenCreditLines",
        "TotalCreditLinespast7years", "InquiriesLast6Months", "TotalInquiries",
        "AmountDelinquent", "EmploymentStatusDuration",
    ],
    categorical_mode=[
        "EmploymentStatus", "Occupation", "BorrowerState", "IncomeRange",
    ],
    grouped_median={
        # DTI distribution differs by income band -> impute within band.
        "DebtToIncomeRatio": "IncomeRange",
    },
    constant_fill={
        # Null == no prior Prosper relationship -> 0, not "missing".
        "TotalProsperLoans": 0, "TotalProsperPaymentsBilled": 0,
        "OnTimeProsperPayments": 0, "ProsperPaymentsLessThanOneMonthLate": 0,
        "ProsperPaymentsOneMonthPlusLate": 0, "ProsperPrincipalBorrowed": 0,
        "ProsperPrincipalOutstanding": 0,
    },
    add_missing_flags=[
        "DebtToIncomeRatio",   # capped/unavailable DTI is itself a signal
        "TotalProsperLoans",   # doubles as a thin-file / repeat-borrower flag
        "EmploymentStatusDuration",
    ],
)


if __name__ == "__main__":
    # Minimal smoke test on synthetic rows.
    demo = pd.DataFrame({
        "IncomeRange": ["$25,000-49,999", "$50,000-74,999", "$25,000-49,999", None],
        "DebtToIncomeRatio": [0.20, np.nan, 0.35, np.nan],
        "BankcardUtilization": [0.5, 0.9, np.nan, 0.3],
        "EmploymentStatus": ["Employed", None, "Self-employed", "Employed"],
        "TotalProsperLoans": [np.nan, 2, np.nan, 1],
    })
    plan = ImputationPlan(
        numeric_median=["BankcardUtilization"],
        categorical_mode=["EmploymentStatus", "IncomeRange"],
        grouped_median={"DebtToIncomeRatio": "IncomeRange"},
        constant_fill={"TotalProsperLoans": 0},
        add_missing_flags=["TotalProsperLoans"],
    )
    cleaned, learned = clean_data(demo, plan)
    print(cleaned)
    print("\nlearned stats:", learned)
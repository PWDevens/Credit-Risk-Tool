"""
features.py — authoritative, executable feature manifest for the Credit Risk Scorecard.

Single source of truth for:
  * which raw columns are model FEATURES vs targets / benchmarks / excluded,
  * the population filter (resolved loans, post-2009 originations),
  * the derived features,
  * the label builders for all three metrics (PD / EAD / LGD).

Governance companion: DATA_DICTIONARY.md (per-variable rationale). Where the two
disagree, THIS FILE wins; the dictionary explains why.

    EL = PD x LGD x EAD
      PD  : classifier, full resolved population.       label = is_bad
      EAD : regressor, DEFAULTED loans only.            label = LoanOriginalAmount - LP_CustomerPrincipalPayments
      LGD : regressor, DEFAULTED loans only.            label = LP_NetPrincipalLoss / EAD, clipped [0, 1]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "prosperLoanData.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# --------------------------------------------------------------------------- #
# Population (DATA_DICTIONARY §1, §11.5, §11.6)                                 #
# --------------------------------------------------------------------------- #
BAD_STATUSES = {"Chargedoff", "Defaulted"}
GOOD_STATUSES = {"Completed"}
POST_2009_CUTOFF = pd.Timestamp("2009-07-01")  # ProsperScore/Rating feature break

# --------------------------------------------------------------------------- #
# Raw feature columns by bucket (DATA_DICTIONARY §2-4)                          #
# --------------------------------------------------------------------------- #
CREDIT_BUREAU = [
    "CreditScoreRangeLower", "CreditScoreRangeUpper",
    "CurrentCreditLines", "OpenCreditLines", "TotalCreditLinespast7years",
    "TotalTrades", "TradesNeverDelinquent (percentage)", "TradesOpenedLast6Months",
    "OpenRevolvingAccounts", "OpenRevolvingMonthlyPayment", "RevolvingCreditBalance",
    "BankcardUtilization", "AvailableBankcardCredit",
    "CurrentDelinquencies", "AmountDelinquent", "DelinquenciesLast7Years",
    "PublicRecordsLast10Years", "PublicRecordsLast12Months",
    "InquiriesLast6Months", "TotalInquiries", "DebtToIncomeRatio",
]
APPLICATION_NUMERIC = [
    "Term", "LoanOriginalAmount", "StatedMonthlyIncome", "EmploymentStatusDuration",
]
APPLICATION_CATEGORICAL = [
    "IncomeRange", "IncomeVerifiable", "EmploymentStatus", "Occupation",
    "BorrowerState", "ListingCategory (numeric)", "IsBorrowerHomeowner", "CurrentlyInGroup",
]
# Prior-Prosper history (Bucket 3): informative nulls -> fill 0 + is_repeat_borrower.
PRIOR_PROSPER = [
    "TotalProsperLoans", "TotalProsperPaymentsBilled", "OnTimeProsperPayments",
    "ProsperPaymentsLessThanOneMonthLate", "ProsperPaymentsOneMonthPlusLate",
    "ProsperPrincipalBorrowed", "ProsperPrincipalOutstanding", "ScorexChangeAtTimeOfListing",
]

# Derived features (DATA_DICTIONARY §9).
DERIVED_NUMERIC = [
    "credit_history_months", "credit_score_mid", "loan_to_income",
    "stated_income_log", "is_repeat_borrower", "income_unverified", "dti_capped_flag",
]
DERIVED_CATEGORICAL = ["bankcard_util_bucket"]

# The model input contract. Every downstream consumer reads this.
MODEL_FEATURES = (
    CREDIT_BUREAU + APPLICATION_NUMERIC + APPLICATION_CATEGORICAL
    + PRIOR_PROSPER + DERIVED_NUMERIC + DERIVED_CATEGORICAL
)
CATEGORICAL_FEATURES = APPLICATION_CATEGORICAL + DERIVED_CATEGORICAL

# Kept in the processed frame for label building / benchmarking — NOT features.
LABEL_SUPPORT = [
    "LoanStatus", "LoanOriginalAmount",
    "LP_CustomerPrincipalPayments", "LP_NetPrincipalLoss",
    "LP_GrossPrincipalLoss", "LP_NonPrincipalRecoverypayments",
]
BENCHMARK_COLS = ["ProsperScore", "ProsperRating (numeric)"]

# Raw columns needed only to compute derived features (not retained as features).
_DERIVE_SOURCES = ["DateCreditPulled", "FirstRecordedCreditLine"]

PD_TARGET = "is_bad"


# --------------------------------------------------------------------------- #
# Derived features                                                             #
# --------------------------------------------------------------------------- #
def _util_bucket(s: pd.Series) -> pd.Series:
    """Bankcard utilization -> ordinal bucket with an explicit Missing level."""
    bins = [-np.inf, 0.30, 0.50, 0.75, 1.00, np.inf]
    labels = ["<=30%", "30-50%", "50-75%", "75-100%", ">100%"]
    out = pd.cut(s, bins=bins, labels=labels)
    return out.astype("object").where(out.notna(), "Missing")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the §9 derived features. Run BEFORE imputation so null-aware flags see raw nulls."""
    df = df.copy()

    df["credit_score_mid"] = df[["CreditScoreRangeLower", "CreditScoreRangeUpper"]].mean(axis=1)

    pulled = pd.to_datetime(df.get("DateCreditPulled"), errors="coerce")
    first = pd.to_datetime(df.get("FirstRecordedCreditLine"), errors="coerce")
    df["credit_history_months"] = (pulled - first).dt.days / 30.44

    annual_income = df["StatedMonthlyIncome"] * 12
    df["loan_to_income"] = df["LoanOriginalAmount"] / annual_income.replace(0, np.nan)
    df["stated_income_log"] = np.log1p(df["StatedMonthlyIncome"].clip(lower=0))

    # is_repeat_borrower from RAW missingness, BEFORE the prior-Prosper null->0 fill.
    df["is_repeat_borrower"] = df["TotalProsperLoans"].notna().astype("int8")
    df["income_unverified"] = (
        ~df["IncomeVerifiable"].astype("boolean").fillna(False)
    ).astype("int8")
    df["dti_capped_flag"] = (df["DebtToIncomeRatio"] >= 10.01).astype("int8")
    df["bankcard_util_bucket"] = _util_bucket(df["BankcardUtilization"])

    # Informative nulls -> 0 (no prior Prosper relationship), never median.
    for col in PRIOR_PROSPER:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# --------------------------------------------------------------------------- #
# Population filter & label builders                                           #
# --------------------------------------------------------------------------- #
def filter_population(df: pd.DataFrame, mode: str = "resolved") -> pd.DataFrame:
    """Post-2009 originations, resolved loans only (DATA_DICTIONARY §1)."""
    orig = pd.to_datetime(df["LoanOriginationDate"], errors="coerce")
    df = df[orig >= POST_2009_CUTOFF]
    if mode == "resolved":
        keep = df["LoanStatus"].isin(BAD_STATUSES | GOOD_STATUSES)
    else:  # pragma: no cover - reserved for the optional seasoned-Current mode
        raise NotImplementedError(f"population mode {mode!r} not implemented")
    return df[keep].copy()


def build_pd_target(df: pd.DataFrame) -> pd.Series:
    return df["LoanStatus"].isin(BAD_STATUSES).astype("int8")


def build_ead_label(df: pd.DataFrame) -> pd.Series:
    """Outstanding principal at default (no CCF for installment loans)."""
    ead = df["LoanOriginalAmount"] - df["LP_CustomerPrincipalPayments"]
    return ead.clip(lower=0)


def build_lgd_label(df: pd.DataFrame) -> pd.Series:
    """Fraction of exposure lost = net principal loss / EAD, clipped to [0, 1]."""
    ead = build_ead_label(df)
    lgd = df["LP_NetPrincipalLoss"] / ead.replace(0, np.nan)
    return lgd.clip(0, 1)


# --------------------------------------------------------------------------- #
# Frame assembly                                                               #
# --------------------------------------------------------------------------- #
def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Force the declared categorical features to string dtype (preserving NA).

    Needed because CSV round-trips lose dtype and some categoricals are numeric
    codes (e.g. ListingCategory) that AutoGluon would otherwise treat as numeric.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
            df[col] = df[col].map(lambda v: v if (isinstance(v, float) and np.isnan(v)) else str(v))
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filtered+derived frame: MODEL_FEATURES + label-support + benchmarks + is_bad."""
    df = add_derived_features(df)
    keep = list(dict.fromkeys(MODEL_FEATURES + LABEL_SUPPORT + BENCHMARK_COLS))
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out[PD_TARGET] = build_pd_target(df).to_numpy()
    return cast_categoricals(out)


def main() -> None:
    print(f"Reading {RAW_CSV} ...")
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    pop = filter_population(raw, mode="resolved")
    frame = prepare(pop)

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame[PD_TARGET]
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train_data.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_data.csv", index=False)

    bad = int(frame[PD_TARGET].sum())
    print(f"  raw rows:          {len(raw):>8,}")
    print(f"  resolved post-2009:{len(frame):>8,}")
    print(f"  bad rate:          {bad / len(frame):>8.3%}  ({bad:,} bad)")
    print(f"  defaulted (EAD/LGD population): {bad:,}")
    print(f"  features: {len(MODEL_FEATURES)}  |  train={len(train):,}  test={len(test):,}")
    print(f"  wrote {PROCESSED_DIR / 'train_data.csv'} and test_data.csv")


if __name__ == "__main__":
    main()

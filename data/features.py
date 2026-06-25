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

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "prosperLoanData.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Macro overlay (economic conditions at origination), pulled by build_macro_features.py; the
# pipeline joins it only if the file exists (graceful no-op otherwise).
MACRO_CSV = REPO_ROOT / "data" / "raw" / "macro_monthly.csv"

# --- Through-the-cycle (TTC) anchoring (see docs/01-feature-engineering.md, .pipeline/time-smoothing.md)
# Raw point-in-time unemployment is spiky (2009 GFC, 2020 COVID) and correlates with vintage,
# so on a random split it inflates AUC by letting the model memorize cohort default rates.
# TTC anchoring smooths it (EWMA) and shrinks toward a long-run mean. build_macro_features.py
# produces the *_ma12 / *_ewma / *_ttc / *_gap / *_yoy columns.
TTC_WEIGHT = 0.5                                # macro_ttc = w*smoothed_PIT + (1-w)*long_run_mean
MACRO_LONGRUN_WINDOW = ("1990-01", "2019-12")   # stable anchor window (excludes the COVID spike)

MACRO_FEATURES_RAW = ["macro_unemployment", "macro_fedfunds"]                  # point-in-time level
MACRO_FEATURES_TTC = ["macro_unemployment_ttc", "macro_fedfunds_ttc", "macro_unemployment_gap"]
MACRO_FEATURES_ROBUST = ["macro_unemployment_gap", "macro_unemployment_yoy",   # gap + trailing YoY
                         "macro_fedfunds_gap", "macro_fedfunds_yoy"]

# CANONICAL macro form for this project = TTC-anchored. DECIDED after the v1-v5 feature matrix
# + the Part C out-of-time study: raw point-in-time macro is partly a vintage proxy that does
# NOT generalize out-of-time, whereas TTC-anchored macro generalizes best for the GBMs and
# rescues the logistic model from overfitting the vintage levels. "macro" unqualified = this.
MACRO_FEATURES = MACRO_FEATURES_TTC

# --- STATE (regional) unemployment overlay (data/build_state_features.py) -------------------
# The borrower's OWN state's unemployment at origination, TTC-smoothed per state (each state
# anchored to its own long-run mean). Cross-sectional (varies between borrowers on the same
# date), so unlike national macro it is not just a vintage proxy. Joined by BorrowerState x ym.
STATE_CSV = REPO_ROOT / "data" / "raw" / "state_monthly.csv"
STATE_FEATURES_RAW = ["state_unemployment"]
STATE_FEATURES_TTC = ["state_unemployment_ttc", "state_unemployment_gap"]   # canonical state set

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

# The model input contract for the AutoML BASELINE. Every downstream consumer reads this.
# Feature names are the real Prosper / derived column names (no bucket prefix): the bucket
# is already conveyed by the list variable each name lives in, so prefixing the strings
# would be redundant, would decouple the names from the raw data + DATA_DICTIONARY, and
# would force renaming the actual DataFrame columns everywhere for no modeling benefit.
MODEL_FEATURES = (
    CREDIT_BUREAU + APPLICATION_NUMERIC + APPLICATION_CATEGORICAL
    + PRIOR_PROSPER + DERIVED_NUMERIC + DERIVED_CATEGORICAL
)
CATEGORICAL_FEATURES = APPLICATION_CATEGORICAL + DERIVED_CATEGORICAL + ["RiskCluster"]

# Engineered features (feature_engineering()). Deliberately NOT in MODEL_FEATURES, so the
# AutoML baseline stays feature-engineering-free; the fine-tuned challengers opt in via
# MODEL_FEATURES + ENGINEERED_FEATURES.
# The 9 numeric ones are produced by feature_engineering(); RiskCluster (categorical) is the
# KMeans segment produced by assign_risk_cluster() and appended here so the challengers
# select it. RiskCluster is also in CATEGORICAL_FEATURES so it is one-hot encoded, not
# treated as a number.
_ENGINEERED_NUMERIC = [
    # NOTE: Credit_and_Affordability_LTI was DROPPED — it was a perfect duplicate of the base
    # derived feature loan_to_income (Spearman 1.0, VIF 5.6M in the diagnostic).
    "Credit_and_Affordability_EstMonthlyDebtObligation",
    "Credit_and_Affordability_DisposableIncome",
    "Credit_and_Affordability_ResidualIncome",   # cash left after existing debt + the new loan
    "Credit_and_Affordability_ReferencePTI",      # new-loan payment / income at a reference APR
    "Credit_and_Affordability_TotalUtilization",
    "Credit_and_Affordability_OpenToTotalLineRatio",
    "Credit_and_Affordability_InquiryVelocity",
    "Credit_and_Affordability_RecentDelinquencyShare",
    "Historical_Prosper_Activity_OnTimeRate",
    "Historical_Prosper_Activity_OutstandingDebtRatio",
    # New-information flags (not transforms of existing features):
    "Affordability_IncomeBand_Mismatch",   # stated income inconsistent with self-reported band
    "Affordability_Income_Undefined",       # zero/undefined income (the DisposableIncome sentinel)
]
ENGINEERED_FEATURES = _ENGINEERED_NUMERIC + ["RiskCluster"]

# Reference APR used to compute a rate-independent loan payment for Residual Income and
# Reference-rate PTI. The real BorrowerRate is an excluded price field, so we price every
# loan at one fixed benchmark rate to get a comparable monthly payment.
REFERENCE_APR = 0.15

# ----------------------------------------------------------------------------- #
# Monotonic-constraint direction map (for LightGBM/XGBoost monotone_constraints) #
# ----------------------------------------------------------------------------- #
# In plain language: each number says which way risk is ALLOWED to move as the feature goes up.
#   +1  = as this feature goes UP, predicted default risk may only go UP   (or stay flat)
#   -1  = as this feature goes UP, predicted default risk may only go DOWN (or stay flat)
#    0  = no constraint (let the data decide; used where direction is genuinely ambiguous)
# Constraints encode domain/regulatory knowledge and stop the model from carving
# noise-driven wiggles, improving robustness and defensibility. Only numeric features get a
# direction; one-hot categoricals have no natural ordering, so they are left unconstrained.
MONOTONE_DIRECTIONS = {
    # --- more of these = MORE risk (+1) ---
    "DebtToIncomeRatio": +1,            # more debt relative to income
    "BankcardUtilization": +1,          # closer to maxed-out cards
    "InquiriesLast6Months": +1,         # recent credit-seeking
    "TotalInquiries": +1,
    "CurrentDelinquencies": +1,         # currently behind on accounts
    "AmountDelinquent": +1,
    "DelinquenciesLast7Years": +1,
    "PublicRecordsLast10Years": +1,     # bankruptcies / judgments
    "PublicRecordsLast12Months": +1,
    "RevolvingCreditBalance": +1,       # more revolving debt carried
    "LoanOriginalAmount": +1,           # bigger loan to repay
    "Term": +1,                         # longer exposure window
    "loan_to_income": +1,               # bigger loan relative to income
    "Credit_and_Affordability_EstMonthlyDebtObligation": +1,
    "Credit_and_Affordability_ReferencePTI": +1,        # heavier new-loan payment burden
    "Credit_and_Affordability_TotalUtilization": +1,
    "Credit_and_Affordability_InquiryVelocity": +1,
    "Credit_and_Affordability_RecentDelinquencyShare": +1,
    "Historical_Prosper_Activity_OutstandingDebtRatio": +1,
    "Affordability_IncomeBand_Mismatch": +1,            # stated income looks inconsistent
    "Affordability_Income_Undefined": +1,               # zero/undefined income
    # --- more of these = LESS risk (-1) ---
    "credit_score_mid": -1,             # higher bureau score
    "CreditScoreRangeLower": -1,
    "CreditScoreRangeUpper": -1,
    "StatedMonthlyIncome": -1,          # more income
    "stated_income_log": -1,
    "credit_history_months": -1,        # longer, deeper credit history
    "AvailableBankcardCredit": -1,      # more unused credit headroom
    "Credit_and_Affordability_DisposableIncome": -1,    # more cash buffer
    "Credit_and_Affordability_ResidualIncome": -1,      # more cash left after the new loan
    "Historical_Prosper_Activity_OnTimeRate": -1,       # better prior payment record
    # everything else (counts of lines/trades, prior-Prosper volumes, categoricals): 0 / omit.
}

# Annual-income bounds implied by each self-reported IncomeRange band (for the plausibility
# flag below). Used to check stated monthly income against the band the borrower selected.
INCOME_BANDS = {
    "$0": (0, 0),
    "Not employed": (0, 0),
    "$1-24,999": (1, 24_999),
    "$25,000-49,999": (25_000, 49_999),
    "$50,000-74,999": (50_000, 74_999),
    "$75,000-99,999": (75_000, 99_999),
    "$100,000+": (100_000, np.inf),
}

# RiskCluster — an unsupervised KMeans segment built from the features below (base + a few
# engineered). Fit on TRAIN only (leakage-safe), persisted, and re-applied in production.
# Added to ENGINEERED_FEATURES / CATEGORICAL_FEATURES once built (see build_risk_clusters.py).
RISK_CLUSTER_PATH = REPO_ROOT / "models" / "risk_cluster.joblib"
CLUSTER_FEATURES = [
    "loan_to_income",  # was Credit_and_Affordability_LTI (dropped); identical information
    "DebtToIncomeRatio",
    "BankcardUtilization",
    "InquiriesLast6Months",
    "Credit_and_Affordability_DisposableIncome",
    "Credit_and_Affordability_TotalUtilization",
    "Credit_and_Affordability_OpenToTotalLineRatio",
    "Credit_and_Affordability_InquiryVelocity",
]

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
# Feature Engineering to Improve Modeling                                           #
# --------------------------------------------------------------------------- #
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers advanced financial risk metrics while safely handling data leakage

    and first-time borrower edge cases.
    """
    df = df.copy()

    # =========================================================================
    # 1. CREDIT CAPACITY, LEVERAGE & AFFORDABILITY METRICS
    # =========================================================================

    # (LTI was dropped — duplicate of base loan_to_income.)

    # Reconstruct Estimated Monthly Debt Obligation from DTI and Stated Income
    df["Credit_and_Affordability_EstMonthlyDebtObligation"] = (
        df["DebtToIncomeRatio"] * df["StatedMonthlyIncome"]
    )

    # Absolute Disposable Income Proxy (Dollar cash buffer left over BEFORE the new loan)
    df["Credit_and_Affordability_DisposableIncome"] = (
        df["StatedMonthlyIncome"]
        - df["Credit_and_Affordability_EstMonthlyDebtObligation"]
    )

    # New-loan monthly payment at a fixed REFERENCE_APR (the real rate is an excluded price
    # field, so we price every loan at one benchmark rate for a comparable payment).
    _c = REFERENCE_APR / 12.0
    ref_payment = df["LoanOriginalAmount"] * _c / (1.0 - (1.0 + _c) ** (-df["Term"]))

    # Reference-rate PTI: the NEW loan's payment as a share of monthly income (vs DTI, which
    # is ALL existing debt). A single-loan affordability ratio.
    df["Credit_and_Affordability_ReferencePTI"] = ref_payment.div(df["StatedMonthlyIncome"])

    # Residual income (VA-underwriting style): absolute dollars left after existing debt AND
    # the new loan payment. Captures resilience that ratios miss — two borrowers at the same
    # DTI differ greatly if one keeps $300/mo and the other $3,000/mo.
    df["Credit_and_Affordability_ResidualIncome"] = (
        df["Credit_and_Affordability_DisposableIncome"] - ref_payment
    )

    # Re-engineered Total Revolving Credit Utilization
    total_revolving_limit = (
        df["RevolvingCreditBalance"] + df["AvailableBankcardCredit"]
    )
    df["Credit_and_Affordability_TotalUtilization"] = (
        df["RevolvingCreditBalance"].div(total_revolving_limit).fillna(0)
    )
    df["Credit_and_Affordability_TotalUtilization"] = df[
        "Credit_and_Affordability_TotalUtilization"
    ].clip(0.0, 2.0)

    # =========================================================================
    # 2. CREDIT DEPTH, VELOCITY & VINTAGE METRICS (BUREAU DATA)
    # =========================================================================

    # Credit Line Utilization Burden (How extended are they across open lines?)
    df["Credit_and_Affordability_OpenToTotalLineRatio"] = (
        df["OpenCreditLines"].div(df["CurrentCreditLines"]).fillna(0)
    )

    # Inquiry Velocity (Credit-seeking density in the last 6 months)
    df["Credit_and_Affordability_InquiryVelocity"] = df[
        "InquiriesLast6Months"
    ].div(df["TotalInquiries"] + 1)

    # Delinquency Vintage Ratio (Is delinquency recent or historical?)
    df["Credit_and_Affordability_RecentDelinquencyShare"] = df[
        "CurrentDelinquencies"
    ].div(df["DelinquenciesLast7Years"] + 1)

    # =========================================================================
    # 3. HISTORICAL PROSPER ACTIVITY (ANTI-DATA LEAKAGE STRATEGY)
    # =========================================================================

    # Explicitly calculate historical metrics ONLY if they are a repeat borrower.
    # First-time borrowers are filled with -1 so tree models (XGBoost/LightGBM)
    # can explicitly isolate them without mixing them up with "perfect" history.

    is_repeat = df["TotalProsperLoans"] > 0

    # Initialize columns with -1 sentinel value
    df["Historical_Prosper_Activity_OnTimeRate"] = -1.0
    df["Historical_Prosper_Activity_OutstandingDebtRatio"] = -1.0

    # Calculate only for rows where true historical data actually exists
    df.loc[is_repeat, "Historical_Prosper_Activity_OnTimeRate"] = df[
        "OnTimeProsperPayments"
    ].div(df["TotalProsperPaymentsBilled"])
    df.loc[is_repeat, "Historical_Prosper_Activity_OutstandingDebtRatio"] = df[
        "ProsperPrincipalOutstanding"
    ].div(df["ProsperPrincipalBorrowed"])

    # Clean up any potential calculation NaNs within the subsetted repeat borrowers
    df["Historical_Prosper_Activity_OnTimeRate"] = df[
        "Historical_Prosper_Activity_OnTimeRate"
    ].fillna(-1.0)
    df["Historical_Prosper_Activity_OutstandingDebtRatio"] = df[
        "Historical_Prosper_Activity_OutstandingDebtRatio"
    ].fillna(-1.0)

    # =========================================================================
    # 4. NEW-INFORMATION FLAGS (data-quality / plausibility, not transforms)
    # =========================================================================

    # (#5) Income-band plausibility: does stated monthly income, annualized, fall OUTSIDE the
    # IncomeRange band the borrower selected? A mismatch is a self-report-reliability / fraud
    # signal that no existing feature captures. Binary IS the right encoding — it's a yes/no
    # consistency check, not a magnitude (a signed gap would just re-encode income, which the
    # model already has). Missing IncomeRange -> not flagged (cannot check).
    annual = df["StatedMonthlyIncome"] * 12
    lo = df["IncomeRange"].map(lambda b: INCOME_BANDS.get(b, (-np.inf, np.inf))[0])
    hi = df["IncomeRange"].map(lambda b: INCOME_BANDS.get(b, (-np.inf, np.inf))[1])
    df["Affordability_IncomeBand_Mismatch"] = (
        ((annual < lo) | (annual > hi)) & df["IncomeRange"].notna()
    ).astype("int8")

    # (#6) Sentinel/undefined flag: zero or undefined stated income is what drives the -1
    # sentinels in LTI / DisposableIncome. An explicit binary separates "undefined" from a
    # genuinely low value (cleaner than relying on -1 alone, and essential for any linear /
    # WOE model where -1 would otherwise be read as a magnitude).
    df["Affordability_Income_Undefined"] = (df["StatedMonthlyIncome"] <= 0).astype("int8")

    # --- Validation hardening: guarantee no inf / NaN in engineered features ----------
    # Several ratios divide by a column that can be exactly 0 (zero income, zero credit
    # lines, zero prior payments). `.fillna()` only catches 0/0 -> NaN, not x/0 -> +/-inf;
    # the validation found inf in LTI and OpenToTotalLineRatio. inf crashes sklearn's
    # RandomForest and poisons tree splits, so replace inf with NaN and fill ALL remaining
    # engineered NaN with the function's own -1 "undefined" sentinel. Using -1 (not a
    # median) keeps this leakage-free — no statistic is learned from the data.
    df[_ENGINEERED_NUMERIC] = (
        df[_ENGINEERED_NUMERIC].replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    )

    return df


# --------------------------------------------------------------------------- #
# RiskCluster — unsupervised borrower segmentation                            #
# --------------------------------------------------------------------------- #
def fit_risk_clusters(df: pd.DataFrame, k_range=range(3, 8), sample: int = 4000,
                      random_state: int = 42):
    """Fit median-impute -> StandardScaler -> KMeans on CLUSTER_FEATURES (TRAIN ONLY).

    Scaling is required because KMeans uses Euclidean distance and the features live on
    very different scales (dollars vs ratios). Imputation handles the raw NaNs (DTI etc.);
    -1 sentinels in the engineered inputs are left as-is. `k` is chosen by the best
    silhouette score over `k_range` (computed on a random subsample for speed, since
    silhouette is O(n^2)). Returns (fitted_pipeline, info_dict). Persist the pipeline and
    re-apply it unchanged to test / production so no statistic leaks across the split.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import QuantileTransformer

    # QuantileTransformer (rank-based -> normal), not StandardScaler/RobustScaler: the
    # engineered inputs have pathological outliers (absurd self-reported incomes flowing into
    # DisposableIncome, DTI-cap rows). A mean/std or IQR scaler lets those few extremes
    # dominate, so KMeans isolates 1-2 of them as singleton clusters (silhouette ~1.0,
    # useless). Rank-based scaling bounds the tails so clusters form on the bulk structure.
    def _scaler():
        return QuantileTransformer(output_distribution="normal",
                                   n_quantiles=min(1000, len(df)), random_state=random_state)

    X = df[CLUSTER_FEATURES]
    pre = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", _scaler())]).fit(X)
    Xs = pre.transform(X)

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(Xs), size=min(sample, len(Xs)), replace=False)
    scores, best_k, best_score = {}, None, -1.0
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(Xs)
        s = float(silhouette_score(Xs[idx], labels[idx]))
        scores[k] = s
        if s > best_score:
            best_k, best_score = k, s

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", _scaler()),
        ("kmeans", KMeans(n_clusters=best_k, random_state=random_state, n_init=10)),
    ]).fit(X)
    return pipe, {"best_k": best_k, "silhouette": best_score, "scores": scores}


def assign_risk_cluster(df: pd.DataFrame, pipe=None) -> pd.Series:
    """Return the RiskCluster label (string, e.g. '0'..'k-1') for each row.

    Loads the persisted pipeline if none is passed. Expects CLUSTER_FEATURES to be present,
    so call AFTER feature_engineering(). Same scaler+KMeans used in training and production.
    """
    if pipe is None:
        pipe = joblib.load(RISK_CLUSTER_PATH)
    labels = pipe.predict(df[CLUSTER_FEATURES])
    return pd.Series(labels, index=df.index).astype(str)


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


def assign_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join point-in-time macro (unemployment, fed funds) at the origination month.

    Macro at origination is known at the underwriting decision -> leakage-safe. No-op (NaN
    columns) if build_macro_features.py hasn't been run yet, so the pipeline never breaks.
    """
    out = df.copy()
    if not MACRO_CSV.exists():
        for c in MACRO_FEATURES_RAW + MACRO_FEATURES_TTC + MACRO_FEATURES_ROBUST:
            out[c] = np.nan
        return out
    macro = pd.read_csv(MACRO_CSV, dtype={"ym": str}).set_index("ym")
    ym = pd.to_datetime(out["LoanOriginationDate"], errors="coerce").dt.to_period("M").astype(str)
    for c in macro.columns:        # join ALL macro columns (raw + smoothed + ttc + gap + yoy)
        out[c] = ym.map(macro[c])
    return out


def current_macro(macro_path: Path = MACRO_CSV) -> dict:
    """Latest available macro values, for scoring a NEW loan at today's economic conditions.

    A new application is originated 'now', so its point-in-time macro is the most recent month
    in the table. Returns {col: value} for the canonical MACRO_FEATURES (TTC-anchored); empty
    if no macro file. This is a through-the-cycle calibration: it shifts every current PD with
    the economy (matters for EL/reserves/pricing) without changing the ranking among same-day
    applicants (macro is identical for all of them).
    """
    if not macro_path.exists():
        return {}
    m = pd.read_csv(macro_path, dtype={"ym": str}).sort_values("ym")
    avail = m.dropna(subset=[c for c in MACRO_FEATURES if c in m.columns])
    last = (avail if not avail.empty else m).iloc[-1]
    return {c: float(last[c]) for c in MACRO_FEATURES if c in m.columns}


def assign_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join the borrower's STATE unemployment (TTC-smoothed) at origination, by BorrowerState x
    origination month. No-op (NaN columns) until build_state_features.py has been run."""
    out = df.copy()
    state_cols = STATE_FEATURES_RAW + STATE_FEATURES_TTC
    if not STATE_CSV.exists():
        for c in state_cols:
            out[c] = np.nan
        return out
    st = pd.read_csv(STATE_CSV, dtype={"ym": str, "state": str})
    ym = pd.to_datetime(out["LoanOriginationDate"], errors="coerce").dt.to_period("M").astype(str)
    key = pd.DataFrame({"ym": ym.to_numpy(), "state": out["BorrowerState"].astype("string").to_numpy()})
    merged = key.merge(st[["ym", "state", *state_cols]], on=["ym", "state"], how="left")
    for c in state_cols:                       # null BorrowerState (~5%) -> NaN; trees handle it
        out[c] = merged[c].to_numpy()
    return out


def current_state_features(state: str | None, state_path: Path = STATE_CSV) -> dict:
    """Latest TTC state values for a given BorrowerState, for scoring a NEW loan. Empty if no
    state file or unknown state (caller leaves NaN, which the tree models handle)."""
    if state is None or not state_path.exists():
        return {}
    st = pd.read_csv(state_path, dtype={"ym": str, "state": str}).sort_values("ym")
    sub = st[st["state"] == str(state)].dropna(subset=STATE_FEATURES_TTC)
    if sub.empty:
        return {}
    last = sub.iloc[-1]
    return {c: float(last[c]) for c in STATE_FEATURES_TTC if c in st.columns}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filtered+derived+engineered frame: MODEL_FEATURES + ENGINEERED_FEATURES (+ macro if
    pulled) + label-support + benchmarks + is_bad. Engineered/macro columns ride along in the
    processed data so the challengers can opt in; the AutoML baseline selects MODEL_FEATURES
    and ignores them."""
    df = add_derived_features(df)
    df = feature_engineering(df)
    macro_cols = []
    if MACRO_CSV.exists():           # only join when the macro data has been fetched
        df = assign_macro_features(df)
        macro_cols = list(dict.fromkeys(
            MACRO_FEATURES_RAW + MACRO_FEATURES_TTC + MACRO_FEATURES_ROBUST))  # all variants ride along
    state_cols = []
    if STATE_CSV.exists():           # regional overlay rides along once fetched
        df = assign_state_features(df)
        state_cols = list(dict.fromkeys(STATE_FEATURES_RAW + STATE_FEATURES_TTC))
    # LoanOriginationDate is retained (non-feature) so an out-of-time split can key on vintage.
    keep = list(dict.fromkeys(
        MODEL_FEATURES + ENGINEERED_FEATURES + macro_cols + state_cols
        + ["LoanOriginationDate"] + LABEL_SUPPORT + BENCHMARK_COLS))
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

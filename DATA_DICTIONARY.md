# Prosper Loan Data — Annotated Data Dictionary & Feature Treatment

Reference for the consumer credit-risk scorecard project. This is the source
data dictionary **annotated with the modeling decision for each variable** —
keep, exclude, benchmark, or loss-only — and the reason. It is the governance
companion to `features.py` (the executable manifest) and `data_cleaning.py`
(imputation). Where the two disagree, `features.py` is authoritative; this file
explains *why*.

The project decomposes credit loss into the regulatory identity:

> **Expected Loss = PD × LGD × EAD**

- **PD** (probability of default) — the classification model. Features come
  *only* from origination-time fields (Sections 2–4).
- **LGD / EAD** (loss severity / exposure) — built from the realized-outcome
  `LP_*` fields (Section 7). These fields construct **labels**, never PD features.

## Treatment legend

| Tag | Meaning |
|-----|---------|
| `TARGET` | Used to construct the default label. |
| `FEATURE` | Model input. Known at the underwriting decision; not endogenous. |
| `FEATURE*` | Usable feature, but needs care (high-cardinality, self-reported, or fair-lending optics). |
| `BENCHMARK` | Excluded as a feature — it is Prosper's *own* risk-model output. Retained to benchmark our model against the incumbent. |
| `EXCLUDE (price)` | Endogenous: set *by* underwriting. Cannot be an input to a model that sets the price. |
| `LOSS-ONLY` | Realized-outcome field. Constructs LGD/EAD labels; never a PD feature (leakage). |
| `DERIVE` | Not a direct feature; used to compute a derived feature or to scope the population. |
| `EXCLUDE` | Identifier, post-origination outcome, or funding result that postdates the decision. |

---

## 1. Target & population

| Variable | Description | Type | Treatment |
|----------|-------------|------|-----------|
| `LoanStatus` | Current loan state: Cancelled, Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, PastDue. | categorical | `TARGET` — `Chargedoff`/`Defaulted` → 1; `Completed` → 0; unresolved (`Current`, `PastDue`) → dropped. |
| `ClosedDate` | Close date (Cancelled/Completed/Chargedoff/Defaulted only). | date | `EXCLUDE` — post-origination. |
| `LoanMonthsSinceOrigination` | Months since origination. | numeric | `DERIVE` — population filter only (seasoning); never a feature. |

**Population rule.** A large share of loans are `Current` — too young to have
defaulted. Coding them as "good" teaches the model that recent vintages are
safe (the *incomplete performance window* trap). Default scope keeps only
**resolved** loans; an optional `seasoned` mode additionally admits fully
seasoned `Current` loans. See `features.filter_population`.

---

## 2. Credit-bureau features — `FEATURE` (Bucket 1)

All "at the time the credit profile was pulled," i.e. application-time. This is
the engine of the scorecard.

| Variable | Description | Type | Imputation |
|----------|-------------|------|-----------|
| `CreditScoreRangeLower` / `CreditScoreRangeUpper` | Bureau score range. | numeric | median → derived `credit_score_mid`. |
| `CurrentCreditLines` | Current credit lines. | numeric | median |
| `OpenCreditLines` | Open credit lines. | numeric | median |
| `TotalCreditLinespast7years` | Credit lines opened in last 7 yrs. | numeric | median |
| `TotalTrades` | Trade lines ever opened. | numeric | median |
| `TradesNeverDelinquent` | Trades never delinquent. | numeric | median |
| `TradesOpenedLast6Months` | Trades opened in last 6 mo. | numeric | median |
| `OpenRevolvingAccounts` | Open revolving accounts. | numeric | median |
| `OpenRevolvingMonthlyPayment` | Monthly payment on revolving accts. | numeric | median |
| `RevolvingCreditBalance` | Revolving balance ($). | numeric | median |
| `BankcardUtilization` | % of revolving credit used. | numeric | median → derived `bankcard_util_bucket`. |
| `AvailableBankcardCredit` | Available bankcard credit ($). | numeric | median |
| `CurrentDelinquencies` | Accounts currently delinquent. | numeric | median |
| `AmountDelinquent` | Dollars currently delinquent. | numeric | median |
| `DelinquenciesLast7Years` | Delinquencies in last 7 yrs. | numeric | median |
| `PublicRecordsLast10Years` | Public records, last 10 yrs. | numeric | median |
| `PublicRecordsLast12Months` | Public records, last 12 mo. | numeric | median |
| `InquiriesLast6Months` | Inquiries, last 6 mo. | numeric | median |
| `TotalInquiries` | Total inquiries. | numeric | median |
| `DebtToIncomeRatio` | DTI, **capped at 10.01** (1001%); null when unavailable. | numeric | **group-median by `IncomeRange`**; cap → derived `dti_capped_flag`. |

Supporting raw dates (not features themselves):

| Variable | Description | Treatment |
|----------|-------------|-----------|
| `DateCreditPulled` | Date credit profile pulled. | `DERIVE` → `credit_history_months`. |
| `FirstRecordedCreditLine` | Date first credit line opened. | `DERIVE` → `credit_history_months`. |

---

## 3. Application / stated fields — `FEATURE` / `FEATURE*` (Bucket 2)

| Variable | Description | Type | Treatment / note |
|----------|-------------|------|------------------|
| `Term` | Loan length (months). | numeric | `FEATURE` |
| `LoanOriginalAmount` | Origination amount ($). | numeric | `FEATURE` (also an EAD input). |
| `StatedMonthlyIncome` | Self-reported monthly income. | numeric | `FEATURE*` — pair with `IncomeVerifiable`; log-transformed. |
| `IncomeRange` | Income band. | categorical | `FEATURE` — also the grouping key for DTI imputation. |
| `IncomeVerifiable` | Borrower has income documentation. | boolean | `FEATURE` → derived `income_unverified`. |
| `EmploymentStatus` | Employment status at listing. | categorical | `FEATURE` (mode-imputed). |
| `EmploymentStatusDuration` | Months in current employment status. | numeric | `FEATURE` (median-imputed). |
| `Occupation` | Self-selected occupation. | categorical | `FEATURE*` — high-cardinality; target-encode/bin. |
| `BorrowerState` | Two-letter state. | categorical | `FEATURE*` — high-cardinality **and** fair-lending optics (geography can proxy protected class). |
| `ListingCategory` | Loan purpose (coded 0–20). | categorical | `FEATURE` — map to labels (Section 9). |
| `IsBorrowerHomeowner` | Homeowner flag. | boolean | `FEATURE` |
| `CurrentlyInGroup` | In a Prosper group at listing. | boolean | `FEATURE` |

---

## 4. Prior Prosper history — `FEATURE`, informative nulls (Bucket 3)

Null here means **no prior Prosper relationship** (a first-time borrower), not a
random gap. Fill with `0` and emit `is_repeat_borrower`; never median-impute.

| Variable | Description | Type |
|----------|-------------|------|
| `TotalProsperLoans` | Prior Prosper loans. | numeric |
| `TotalProsperPaymentsBilled` | Prior payments billed. | numeric |
| `OnTimeProsperPayments` | Prior on-time payments. | numeric |
| `ProsperPaymentsLessThanOneMonthLate` | Prior payments <1 mo late. | numeric |
| `ProsperPaymentsOneMonthPlusLate` | Prior payments ≥1 mo late. | numeric |
| `ProsperPrincipalBorrowed` | Prior principal borrowed ($). | numeric |
| `ProsperPrincipalOutstanding` | Prior principal outstanding ($). | numeric |
| `ScorexChangeAtTimeOfListing` | Score change vs last Prosper loan. | numeric |

Treatment: `constant_fill = 0` + `add_missing_flags` → `is_repeat_borrower`.

---

## 5. Prosper's own model outputs — `BENCHMARK`, not features (Bucket 4a)

Available at origination, but each is the output of Prosper's *own* risk model.
Training on them partially recovers their model rather than building an
independent one — the apparent skill is circular. **Excluded as features;
retained as the champion benchmark** for the challenger tab (our model's AUC
vs. Prosper's grade ranking).

| Variable | Description |
|----------|-------------|
| `ProsperScore` | Custom risk score, 1–10 (10 = lowest risk). Post-2009. |
| `ProsperRating (numeric)` | Rating 0–7 (HR…AA). Post-2009. |
| `ProsperRating (Alpha)` | Rating AA–HR. Post-2009. |
| `CreditGrade` | Pre-2009 credit grade (mutually exclusive with the above). |
| `EstimatedLoss` | Estimated principal loss on charge-off. Post-2009. |
| `EstimatedEffectiveYield` | Estimated effective yield. Post-2009. |
| `EstimatedReturn` | Estimated return (yield − loss rate). Post-2009. |

> **2009 feature break.** `ProsperScore`/`ProsperRating`/`Estimated*` exist only
> post-July-2009; `CreditGrade` only pre-2009. Cleanest scope: **post-2009
> originations only** — consistent feature availability, more relevant regime.

---

## 6. Price terms — `EXCLUDE (price)`, endogenous (Bucket 4b)

Set *by* underwriting in response to risk. For an origination model that decides
the price, the price cannot be an input; doing so leaks Prosper's risk ranking
back in.

| Variable | Description |
|----------|-------------|
| `BorrowerAPR` | Borrower APR. |
| `BorrowerRate` | Borrower interest rate. |
| `LenderYield` | Rate less servicing fee. |
| `MonthlyLoanPayment` | Scheduled payment — `f(rate, amount, term)`, so it carries the same endogeneity. |

---

## 7. Loss / exposure fields — `LOSS-ONLY` (LGD & EAD)

Realized-outcome fields. They **construct LGD/EAD labels** and are **strictly
forbidden as PD features** — every one postdates default. (`EL = PD × LGD × EAD`.)

| Variable | Description | Role |
|----------|-------------|------|
| `LP_GrossPrincipalLoss` | Gross charged-off amount. | LGD numerator. |
| `LP_NetPrincipalLoss` | Principal uncollected after recoveries. | Realized net loss. |
| `LP_CustomerPrincipalPayments` | Pre-charge-off principal paid. | **EAD** = `LoanOriginalAmount − principal paid`. |
| `LP_CustomerPayments` | Pre-charge-off gross payments. | Cash-flow context. |
| `LP_InterestandFees` | Pre-charge-off interest & fees paid. | Cash-flow context. |
| `LP_ServiceFees` | Investor servicing fees. | Net-recovery adjustment. |
| `LP_CollectionFees` | Investor collection fees. | Net-recovery adjustment. |
| `LP_NonPrincipalRecoverypayments` | Interest/fee portion of recoveries. | Recovery composition. |

> For installment loans EAD is the outstanding principal at default (no undrawn
> commitment / CCF). LGD = `1 − net recoveries / EAD`; nominal recoveries are
> fine for v1, PV-discounted recoveries are the rigorous refinement. Basel also
> expects **downturn LGD** (severity under stress), worth a note in the model doc.

---

## 8. Hard excludes — `EXCLUDE`

**Identifiers** (no predictive content; drop): `ListingKey`, `ListingNumber`,
`LoanKey`, `LoanNumber`, `MemberKey`, `GroupKey`, `ListingCreationDate`,
`LoanOriginationQuarter`.

**Dates used only for derivation / scoping**: `LoanOriginationDate`
(vintage/seasoning), `DateCreditPulled` + `FirstRecordedCreditLine` (→
`credit_history_months`).

**Post-origination outcomes** (leakage): `LoanCurrentDaysDelinquent`,
`LoanFirstDefaultedCycleNumber`, `LoanMonthsSinceOrigination` (filter-only).

**Funding-outcome fields** (postdate the underwriting decision): `PercentFunded`,
`Investors`, `InvestmentFromFriendsCount`, `InvestmentFromFriendsAmount`,
`Recommendations`.

---

## 9. Derived features

Built in `features.add_derived_features`, **before imputation** (the null-aware
flags depend on raw missingness).

| Derived feature | Definition | Source |
|-----------------|-----------|--------|
| `credit_history_months` | `DateCreditPulled − FirstRecordedCreditLine` (months). | bureau dates |
| `credit_score_mid` | Mean of score range bounds. | `CreditScoreRange*` |
| `loan_to_income` | `LoanOriginalAmount / (StatedMonthlyIncome × 12)`. | capacity ratio |
| `stated_income_log` | `log1p(StatedMonthlyIncome)`. | income skew |
| `is_repeat_borrower` | `1` if `TotalProsperLoans` non-null. | thin-file signal |
| `income_unverified` | `1` if `IncomeVerifiable` is false. | stated-income reliability |
| `dti_capped_flag` | `1` if `DebtToIncomeRatio ≥ 10.01`. | DTI cap signal |
| `bankcard_util_bucket` | Utilization binned (`≤30/30–50/50–75/75–100/>100%/Missing`). | non-linearity + explicit missing level |

---

## 10. Coded-value reference

**`ListingCategory`:** 0 Not Available · 1 Debt Consolidation · 2 Home
Improvement · 3 Business · 4 Personal · 5 Student · 6 Auto · 7 Other · 8
Baby&Adoption · 9 Boat · 10 Cosmetic · 11 Engagement Ring · 12 Green · 13
Household · 14 Large Purchases · 15 Medical/Dental · 16 Motorcycle · 17 RV · 18
Taxes · 19 Vacation · 20 Wedding.

**`ProsperRating (numeric)`:** 0 N/A · 1 HR · 2 E · 3 D · 4 C · 5 B · 6 A · 7 AA.

**`ProsperScore`:** 1–10, higher = lower risk.

---

## 11. Modeling rules — one-page summary

1. **Leakage.** A PD feature must be knowable at the underwriting decision.
   Outcome fields (`LoanStatus`, `LP_*`, `Loan*Delinquent*`,
   `LoanMonthsSinceOrigination`) build labels only.
2. **Endogeneity.** Rate/APR/yield and `MonthlyLoanPayment` are set *by*
   underwriting → excluded from an origination model.
3. **Circularity.** Prosper's own score/rating/estimates are benchmark targets,
   not features.
4. **Informative nulls.** Prior-Prosper fields → fill 0 + `is_repeat_borrower`,
   not median.
5. **Performance window.** Drop unresolved (`Current`) loans so immature
   vintages aren't labeled good.
6. **2009 break.** Scope to post-2009 originations for consistent features.
7. **DTI cap.** `10.01` = capped (≥1000%); flag it, don't treat as a real value.
8. **Fair lending.** `BorrowerState`/`Occupation` are usable but watch proxy
   risk; document in the model card.
9. **Three models, three validations.** PD, LGD, and EAD are developed and
   validated separately; LGD validation leans on calibration / predicted-vs-
   actual loss, not AUC.

# Credit Risk Scorecard

A lender-facing credit-risk tool that estimates the **Expected Loss** of a consumer
loan from borrower and loan characteristics, built on the
[Prosper Loan dataset](https://www.kaggle.com/) (~113k loans, 81 columns).

It models each component of the regulatory credit-loss identity and combines them:

```
Expected Loss (EL) = PD × LGD × EAD
```

| Component | Meaning | Model |
|-----------|---------|-------|
| **PD**  | Probability of Default        | Calibrated binary classifier (XGBoost), benchmarked against an AutoML baseline **and** Prosper's own grade ranking |
| **LGD** | Loss Given Default (fraction) | Fractional regressor, benchmarked against a mean-LGD baseline |
| **EAD** | Exposure at Default ($)       | Regressor, benchmarked against a full-exposure baseline |

A local **Streamlit** app lets a lender enter a borrower/loan and see PD, LGD, EAD,
and the resulting Expected Loss in dollars, with a per-borrower explanation of the
main risk drivers.

> **Governance.** [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) is the per-variable treatment
> authority — which fields are features, benchmarks, loss-only labels, or excluded, and why.
> `features.py` is the executable feature manifest; where the two disagree, `features.py`
> wins. The build plan lives in [`.pipeline/PROJECT_PLAN.md`](.pipeline/PROJECT_PLAN.md).

---

## Project status

🚧 **In development.**

| Step | Component | Status |
|------|-----------|--------|
| A | Feature manifest + population/labels (`features.py`)             | ✅ Done |
| B | AutoML baselines — PD/EAD/LGD + lazy/champion (`train_baselines.py`) | ✅ Done |
| C | `RiskPredictor` serving interface (`modeling/common/predictor.py`)   | ✅ Done |
| D | Streamlit Win98 frontend (`app/app.py`)                          | ✅ Done — **v1** |
| E | Fine-tuned PD (LightGBM, calibrated) + SHAP, behind the toggle    | ✅ PD shipped |

**v1 = Steps A–D**: a locally-run Win98 dashboard scoring PD/LGD/EAD/EL on AutoML
baselines, with the AutoML-vs-fine-tuned toggle in place. Step E slots fine-tuned
models in behind that toggle with no UI change.

---

## Repository layout

```
data/
  loader.py            Pull raw data from Kaggle (via kagglehub)
  features.py          Authoritative feature manifest: treatment, population, derived features
  data_cleaning.py     Imputation pipeline (driven by the manifest)
  data_processor.py    Train/test split (stratified on the binary target)
  data_analysis.py     EDA toolkit (missing report, WOE/IV ranking, target-rate views)
  raw/                 Raw downloaded dataset (gitignored)
  processed/           Train/test splits (gitignored)
modeling/
  probability-of-default/  PD baselines + XGBoost model
  exposure-at-default/     EAD baseline + model
  loss-given-default/      LGD baseline + model
  model-results/           Saved baseline/model metrics tables
  common/                  Shared metrics + model I/O (planned)
app/                   Streamlit frontend
models/                Serialized pd/lgd/ead model artifacts
```

> `features.py` is the single source of truth for model inputs. Only fields knowable **at
> the underwriting decision** are used. Excluded as inputs: **outcome fields** (`LP_*`
> payments/losses, `ClosedDate`, delinquency cycles — these build EAD/LGD *labels*),
> **price fields** set by underwriting (`BorrowerRate`, `BorrowerAPR`, `LenderYield`,
> `MonthlyLoanPayment`), and **Prosper's own scores/ratings** (kept as benchmarks, not
> features). See `DATA_DICTIONARY.md` §11 for the full rule set.

---

## Getting started

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

### 2. Credentials

Kaggle API credentials are read from `secrets/.env` (gitignored). Provide your
Kaggle username and key there for `loader.py`.

### 3. Build the data and models

```bash
python data/loader.py            # downloads the raw dataset to data/raw/
python data/features.py          # population filter + labels -> data/processed/ splits
python modeling/train_baselines.py   # trains AutoGluon PD/EAD/LGD -> models/ + metrics
```

`train_baselines.py` uses a small AutoML budget by default for fast iteration; for a
production-grade baseline raise it:

```bash
AUTOML_TIME_LIMIT=600 AUTOML_PRESET=best_quality python modeling/train_baselines.py
```

### 4. Run the app

```bash
streamlit run app/app.py
```

---

## Modeling notes

- **Target.** A loan is "bad" when `LoanStatus ∈ {Defaulted, Chargedoff}` → 1; `Completed`
  → 0. Unresolved loans (`Current`, `PastDue*`, `FinalPaymentInProgress`) are **dropped** —
  coding them as good would label immature vintages safe (incomplete performance window).
- **Population scope.** Post-July-2009 originations only, so feature availability is
  consistent (`ProsperScore`/`ProsperRating`/`Estimated*` are post-2009; `CreditGrade` is
  pre-2009).
- **PD metrics.** AUC, Gini (= 2·AUC−1), KS statistic, plus calibration via Brier score
  and log-loss. PD is calibrated so the Expected Loss product is meaningful. Benchmarked
  against both an AutoML model and Prosper's own grade ranking (challenger vs. champion).
- **EAD.** Installment loans have no undrawn commitment (no CCF); the label is outstanding
  principal at default, `LoanOriginalAmount − LP_CustomerPrincipalPayments`. Lazy baseline
  assumes full exposure.
- **LGD.** Label = `1 − net recoveries / EAD` (`LP_NetPrincipalLoss / EAD`), clipped to
  [0, 1]; expected bimodal, so a two-stage cure/severity approach is on the table.
  PV-discounted recoveries and a Basel **downturn-LGD** view are noted refinements.
  Validated on calibration / predicted-vs-actual loss, not AUC.
- **Imputation highlights.** Credit-bureau numerics → median; `DebtToIncomeRatio` →
  group-median by `IncomeRange` (cap `10.01` flagged, not treated as a value); prior-Prosper
  fields → fill 0 + `is_repeat_borrower` (informative nulls, never median).
- **Explainability & fair lending.** SHAP gives global and per-borrower attributions,
  surfaced in the app. `BorrowerState`/`Occupation` can proxy protected class — handled
  with care and documented in the model card.

### Platform note

`auto-sklearn` (used for the PD AutoML baseline) does **not** run on native Windows. Run
Step 2 inside WSL2 or Docker (Linux). All other steps run natively on Windows.

---

## Disclaimer

This is an educational / research project on a public dataset. It is **not** a production
underwriting system and should not be used for real lending decisions.

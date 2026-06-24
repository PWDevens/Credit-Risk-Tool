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
| **PD**  | Probability of Default        | Calibrated binary classifier (**LightGBM**, fine-tuned), benchmarked against an AutoML baseline **and** Prosper's own grade ranking |
| **LGD** | Loss Given Default (fraction) | Regressor, benchmarked against a mean-LGD baseline |
| **EAD** | Exposure at Default ($)       | Regressor, benchmarked against a full-exposure baseline |

A local **Streamlit** app (styled as a Windows-98 underwriting terminal) lets a lender enter
a borrower/loan and: see PD, LGD, EAD and the resulting Expected Loss; toggle between the
AutoML and fine-tuned (LightGBM) PD models; read a per-borrower **SHAP** explanation of the
risk drivers; and get **risk-based pricing** — a discounted lifetime expected loss, expected
profit / RAROC, and a recommended APR.

> **Governance.** [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) is the per-variable treatment
> authority — which fields are features, benchmarks, loss-only labels, or excluded, and why.
> `features.py` is the executable feature manifest; where the two disagree, `features.py`
> wins. The build plan lives in [`.pipeline/PROJECT_PLAN.md`](.pipeline/PROJECT_PLAN.md).

---

## Project status

🚧 **v1 complete; v2 in progress.**

| Step | Component | Status |
|------|-----------|--------|
| A | Feature manifest + population/labels (`features.py`)                 | ✅ Done |
| B | AutoML baselines — PD/EAD/LGD + lazy/champion (`train_baselines.py`) | ✅ Done |
| C | `RiskPredictor` serving interface (`modeling/common/predictor.py`)   | ✅ Done |
| D | Streamlit Win98 frontend (`app/app.py`)                             | ✅ Done |
| E | Fine-tuned PD — LightGBM + SHAP, behind the toggle                   | ✅ Done |

**v1 (Steps A–E) is complete**: a locally-run Win98 dashboard scoring PD/LGD/EAD/EL, with a
working AutoML-vs-LightGBM PD toggle and per-borrower SHAP. The LightGBM model was chosen
after a like-for-like comparison against XGBoost and Random Forest (kept as `finetune_*.py`
due-diligence scripts).

**v2 (in progress).** A financial engine (`modeling/common/finance.py`) adds a discounted
**lifetime ECL**, **expected profit / RAROC**, and **risk-based pricing** (break-even and
target-return APR), surfaced in the app's "Financials" panel. Remaining: feature engineering
to lift PD accuracy, plus prepayment, macro/stress scenarios, and a two-stage LGD model.

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
  common/                  Shared code: data, metrics, RiskPredictor, FLAML harness, finance engine
  probability-of-default/  PD AutoML baseline + fine-tuned challengers (finetune_xgboost/lightgbm/rf.py)
  exposure-at-default/     EAD AutoML baseline
  loss-given-default/      LGD AutoML baseline
  model-results/           Saved metrics tables (gitignored)
  train_baselines.py       Trains the three AutoGluon baselines (resplits from raw first)
  build_default_timing.py  Builds the default-timing curve for the finance engine
app/                   Streamlit Win98 app (app.py)
models/                Serialized artifacts: AutoGluon dirs, pd_lightgbm.joblib,
                       feature_defaults.json, default_timing.json (all gitignored)
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
production-grade baseline raise it (env vars; the example uses bash syntax):

```bash
AUTOML_TIME_LIMIT=600 AUTOML_PRESET=best_quality python modeling/train_baselines.py
```

To enable the fine-tuned PD toggle and the data-built default-timing curve (optional — the
app falls back to the AutoML PD and a built-in timing curve without them):

```bash
python modeling/probability-of-default/finetune_lightgbm.py   # fine-tuned PD (LightGBM) + SHAP
python modeling/build_default_timing.py                       # default-timing curve for pricing
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
- **Fine-tuned PD.** LightGBM, FLAML-tuned and isotonic-calibrated, chosen after a
  like-for-like comparison with XGBoost and Random Forest (all on the same split/features).
  It ties the AutoML baseline within statistical noise; shipped for its single-model
  transparency and fast SHAP. EAD/LGD stay on the AutoML baseline under both toggle states.
- **Explainability & fair lending.** SHAP gives per-borrower attributions for the fine-tuned
  PD model, surfaced in the app's "Why?" panel. `BorrowerState`/`Occupation` can proxy
  protected class — handled with care and documented in the model card.

### Financial engine (v2)

The app turns PD/LGD/EAD into lender decisions via `modeling/common/finance.py`: a discounted
**lifetime ECL** (PD term-structure × amortizing EAD × discounting at the loan's rate),
**expected profit / RAROC**, and **risk-based pricing** (break-even APR and the APR that hits
a target RAROC). The offered APR is a financial *input/output*, never a model feature — which
is exactly why the price can be solved for. See the module's plain-English header for the
assumptions (the PD-timing approximation; prepayment is not yet modeled).

### Platform note

Everything runs natively on **Windows** (and macOS/Linux). The AutoML baselines use
**AutoGluon** and the fine-tuned PD model uses **LightGBM** — no WSL2, Docker, or Linux-only
dependency is required.

---

## Disclaimer

This is an educational / research project on a public dataset. It is **not** a production
underwriting system and should not be used for real lending decisions.

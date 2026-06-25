# Credit Risk Scorecard

A lender-facing credit-risk tool that estimates the **Expected Loss** of a consumer loan from
borrower and loan characteristics — and turns it into **pricing and provisioning decisions** —
built on the [Prosper Loan dataset](https://www.kaggle.com/) (~113k loans, 81 columns).

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

**Scope (v1 → v3).** The project is built in layers, each a recognizable credit-risk discipline:

- **v1 — the loss machinery.** Calibrated PD/LGD/EAD models, the `EL` product, the Win98 app,
  and an AutoML-vs-fine-tuned PD toggle with per-borrower SHAP.
- **v2 — financial decisioning** (`modeling/common/finance.py`). Turns the risk estimates into
  money: a discounted **lifetime ECL**, **expected profit / RAROC**, and **risk-based pricing**
  (break-even and target-return APR).
- **v3 — making it defensible** (in progress). A measured **feature-engineering** pass, a
  point-in-time **macro overlay** kept in *through-the-cycle (TTC)–anchored* form, and
  **out-of-time (vintage) validation** — the way credit models are actually judged. Plan and
  decision records: [`docs/V3_PLAN.md`](docs/V3_PLAN.md), [`docs/macro-decision.md`](docs/macro-decision.md).

See [Project status](#project-status) for the per-step breakdown.

> **Governance.** [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) is the per-variable treatment
> authority — which fields are features, benchmarks, loss-only labels, or excluded, and why.
> `features.py` is the executable feature manifest; where the two disagree, `features.py`
> wins. The build plan lives in [`.pipeline/PROJECT_PLAN.md`](.pipeline/PROJECT_PLAN.md).

---

## Project status

🚧 **v1 & v2 complete; v3 in progress.**

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

**v2 — complete.** A financial engine (`modeling/common/finance.py`) turns PD/LGD/EAD into
lender decisions: a discounted **lifetime ECL**, **expected profit / RAROC**, and **risk-based
pricing** (break-even and target-return APR), surfaced in the app's "Financials" panel.

**v3 (in progress) — depth, time-awareness, and validation.** The theme is making the model
*defensible* the way credit risk is actually validated, not adding surface area (full plan:
[docs/V3_PLAN.md](docs/V3_PLAN.md)). Landed so far:

- ✅ **Out-of-time (vintage) validation** — train on earlier originations, test on later
  (`run_partc.py` / `run_finalize.py`). This is *the* credit-risk validation standard and is now
  the lens every feature decision is judged through.
- ✅ **Feature-engineering pass** — engineered affordability/credit ratios + an unsupervised
  `RiskCluster`, measured in a cumulative **v1→v4** ablation matrix (`run_version.py`,
  `docs/finetuning_matrix.png`). Honest finding: they're largely **redundant** once clustering is
  in — the ~0.75 AUC ceiling is information-bound, not model-bound.
- ✅ **Point-in-time macro overlay** — national unemployment + fed funds at origination, kept in
  **through-the-cycle (TTC)–anchored** form after the out-of-time study showed raw macro is partly
  a vintage proxy ([docs/macro-decision.md](docs/macro-decision.md)).

Remaining:

- ⏳ **Discrete-time hazard model** (loan-month panel + LightGBM) → a model-driven PD term
  structure feeding the ECL engine, benchmarked against scikit-survival.
- ⏳ **ECL backtest** vs realized dollar losses by vintage, an **SR 11-7-style model card**, and a
  results notebook with the money charts.
- 🔭 **Future:** regional (state) unemployment overlay (pipeline built — `build_state_features.py` —
  needs the data pulled), prepayment / competing-risks, a two-stage LGD model, and CCAR/IFRS-9
  stress scenarios.

---

## Repository layout

```
data/
  loader.py                Pull raw data from Kaggle (via kagglehub)
  features.py              Authoritative manifest: treatment, population, derived + engineered
                           features, RiskCluster, macro/state joins, PD/EAD/LGD labels
  data_cleaning.py         Imputation pipeline (manifest-driven)
  data_processor.py        Train/test split (stratified on the binary target)
  data_analysis.py         EDA toolkit (missing report, WOE/IV ranking, target-rate views)
  build_macro_features.py  Pull + TTC-smooth NATIONAL macro (FRED unemployment + fed funds)
  build_state_features.py  Pull + TTC-smooth PER-STATE unemployment (future regional overlay)
  raw/                     Raw dataset + macro panels (gitignored; macro_monthly.csv tracked)
  processed/               Train/test splits (gitignored)
modeling/
  common/
    data.py                Load splits + build per-metric (X, y) with feature-set switches
    metrics.py             PD/EAD/LGD metric suites (AUC/Gini/KS/Brier, calibration)
    predictor.py           RiskPredictor — the in-process scoring API the app calls
    finetune.py            Shared FLAML tune + isotonic-calibrate + eval harness
    finance.py             Financial engine: amortization, lifetime ECL, RAROC, risk-based pricing
  probability-of-default/  PD AutoML baseline + finetune_{xgboost,lightgbm,rf,logistic}.py
  exposure-at-default/     EAD AutoML baseline
  loss-given-default/      LGD AutoML baseline
  train_baselines.py       Trains the three AutoGluon baselines (resplits from raw first)
  build_default_timing.py  Default-timing curve for the finance engine
  build_risk_clusters.py   Fits the persisted KMeans RiskCluster pipeline (train-only)
  diagnose_features.py     IV/WOE + collinearity (VIF / Spearman) feature diagnostics
  run_matrix.py            Crash-resilient cumulative v1->v4 feature-set matrix (per-cell isolation)
  run_version.py           Single-version run with IV/WOE table + explainability drivers
  run_partc.py             Out-of-time study: {random, OOT} x {raw, TTC} macro grid
  run_finalize.py          No-macro OOT baseline + matrix-cell backfills
  results_visual.py        Plots the matrix -> docs/finetuning_matrix.png
  model-results/           Saved metric tables + matrices (gitignored)
app/
  app.py                   Streamlit Win98 underwriting terminal (PD/LGD/EAD/EL, toggle, SHAP, financials)
docs/
  V3_PLAN.md               v3 roadmap (time-aware hazard, OOT validation, packaging)
  macro-decision.md        Why macro is used in TTC-anchored form (the out-of-time study write-up)
  finetuning_matrix.png    The v1->v4 challenger chart (tracked deliverable)
models/                    Serialized artifacts (gitignored): risk_cluster.joblib,
                           feature_defaults.json, default_timing.json; pd_*.joblib live beside
                           the PD scripts, AutoGluon dirs as modeling/<metric>/automl_model/
DATA_DICTIONARY.md         Per-variable treatment authority (features / benchmarks / labels / excluded)
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
  Shipped on the base + engineered + RiskCluster + TTC-macro feature set (see the macro bullet
  below). It ties the AutoML baseline within statistical noise; shipped for its single-model
  transparency and fast SHAP. EAD/LGD stay on the AutoML baseline under both toggle states.
- **Macro overlay (TTC-anchored).** Unemployment + fed funds at origination are added as a
  *through-the-cycle*–anchored feature (smoothed and shrunk toward a long-run mean), not the
  raw point-in-time level. We tested this hard: raw macro lifts AUC ~+0.015 on a random split
  but that lift is **partly a vintage proxy** — out-of-time it shrinks (XGBoost) and even turns
  *negative* for the logistic model. TTC anchoring generalizes best out-of-time (LightGBM
  ~+0.025) and rescues the logistic model from overfitting. Macro shifts the **PD level** with
  the economy (it's a cycle calibration for EL/reserves/pricing), not the applicant ranking.
  Full plain-language write-up + the random-vs-out-of-time results table:
  [docs/macro-decision.md](docs/macro-decision.md).
- **Recommended next overlay — regional (state) unemployment.** The national macro only varies
  by date, so on a random split it partly acts as a vintage proxy. The borrower's *own state's*
  unemployment at origination is **cross-sectional** (it differs between borrowers on the same
  day), so it's far less of a vintage artifact and has a real shot at helping out-of-time. The
  pipeline is **already built and TTC-smoothed per state** (`data/build_state_features.py`,
  `features.assign_state_features`, `macro_set='ttc_geo'`); it just needs the 51 BLS/FRED state
  series pulled (a free FRED API key) to activate. Slated for a future version.
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

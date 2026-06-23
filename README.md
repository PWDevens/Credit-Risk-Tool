# Credit Risk Scorecard

A lender-facing credit-risk tool that estimates the **Expected Loss** of a consumer
loan from borrower and loan characteristics, built on the
[Prosper Loan dataset](https://www.kaggle.com/) (~113k loans, 81 columns).

It models each component of the standard credit-risk identity and combines them:

```
Expected Loss (EL) = PD × EAD × LGD
```

| Component | Meaning | Model |
|-----------|---------|-------|
| **PD**  | Probability of Default        | Calibrated binary classifier (XGBoost), benchmarked against an AutoML baseline |
| **EAD** | Exposure at Default ($)       | Regressor, benchmarked against a full-exposure (CCF=100%) baseline |
| **LGD** | Loss Given Default (fraction) | Fractional regressor, benchmarked against a mean-LGD baseline |

A local **Streamlit** app lets a lender enter a borrower/loan and see PD, EAD, LGD,
and the resulting Expected Loss in dollars, with a per-borrower explanation of the
main risk drivers.

---

## Project status

🚧 **In development.** See [`.pipeline/PROJECT_PLAN.md`](.pipeline/PROJECT_PLAN.md) for the
full plan and current state.

| Step | Component | Status |
|------|-----------|--------|
| 1 | Data cleaning pipeline            | In progress |
| 2 | PD baseline (AutoML)              | Not started |
| 3 | PD model (XGBoost, calibrated)    | Not started |
| 4 | EAD model (+ lazy baseline)       | Not started |
| 5 | LGD model (+ lazy baseline)       | Not started |
| 6 | Streamlit frontend                | Not started |

---

## Repository layout

```
data/
  loader.py            Pull raw data from Kaggle (via kagglehub)
  data-processor.py    Clean + split into train/test
  data_analysis.py     EDA toolkit (missing report, WOE/IV ranking, target-rate views)
  data-cleaning.ipynb  Exploratory cleaning notebook
  raw/                 Raw downloaded dataset (gitignored)
  processed/           Train/test splits (gitignored)
modeling/
  probability-of-default/  PD baseline + XGBoost model
  exposure-at-default/     EAD baseline + model
  loss-given-default/      LGD baseline + model
  common/                  Shared feature list, metrics, model I/O
app/                   Streamlit frontend
models/                Serialized pd/ead/lgd model artifacts
```

> The shared origination-time feature list in `modeling/common/` is the single source of
> truth for model inputs. Only features known **at loan origination** are used — outcome
> columns (`LP_*` payments/losses, `ClosedDate`, delinquency cycles, Prosper's own ratings)
> are excluded as inputs to avoid leakage, and are used only to construct EAD/LGD targets.

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

### 3. Get the data

```bash
python data/loader.py            # downloads the raw dataset to data/raw/
python data/data-processor.py    # cleans + writes train/test to data/processed/
```

### 4. Run the app (once models are trained)

```bash
streamlit run app/app.py
```

---

## Modeling notes

- **Target.** A loan is "bad" (defaulted) when `LoanStatus ∈ {Defaulted, Chargedoff}`.
  In-flight loans (`Current`, `PastDue*`, `FinalPaymentInProgress`) are excluded from the
  PD target since their final outcome is unknown.
- **PD metrics.** AUC, Gini (= 2·AUC−1), KS statistic, plus calibration via Brier score
  and log-loss. PD outputs are calibrated so the Expected Loss product is meaningful.
- **EAD.** Target ≈ outstanding principal at default
  (`LoanOriginalAmount − LP_CustomerPrincipalPayments`); lazy baseline assumes full exposure.
- **LGD.** Target = `LP_NetPrincipalLoss / EAD` (net of recoveries), clipped to [0, 1];
  expected to be bimodal, so a two-stage cure/severity approach is on the table.
- **Explainability.** SHAP provides global and per-borrower attributions, surfaced in the app.

### Platform note

`auto-sklearn` (used for the PD AutoML baseline) does **not** run on native Windows. Run
Step 2 inside WSL2 or Docker (Linux). All other steps run natively on Windows.

---

## Disclaimer

This is an educational / research project on a public dataset. It is **not** a production
underwriting system and should not be used for real lending decisions.

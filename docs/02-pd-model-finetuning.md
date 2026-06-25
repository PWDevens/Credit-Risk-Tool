# PD Model Finetuning — tree ensembles, baselines, and the information ceiling

How the production probability-of-default model was chosen: a strong AutoML **baseline**, three
fine-tuned tree **challengers** searched and calibrated against it, and an honest account of the
**accuracy ceiling** on public Prosper data. All figures are the out-of-time test numbers regenerated
into `modeling/model-results/` by the per-model finetune scripts.

---

## 1. The baseline to beat

Two reference points anchor every challenger (`pd_baseline_automl.csv`):

| Reference | Test AUC | Note |
|---|---|---|
| **AutoGluon AutoML** (no feature engineering) | **0.7498** | the bar a hand-tuned model must clear to justify itself |
| Prosper's own score (`ProsperScore`) | 0.6482 | the incumbent ranking, kept as a **benchmark, not a feature** |

A challenger that can't beat 0.7498 isn't worth its added complexity; clearing 0.648 just means we've
matched the lender's own model, which is table stakes.

## 2. The challengers — search, then calibrate

Three tree families were tuned with a FLAML-style hyperparameter search (cross-validated AUC), then
**isotonically recalibrated** so the scores are true probabilities the ECL engine can trust. Each
`pd_finetune_<model>.csv` reports the raw and calibrated test metrics against the shared baseline:

| Model | Raw AUC | Calibrated AUC | Brier (raw → cal) | LogLoss (raw → cal) | CV AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.7632 | **0.7612** | 0.186 → **0.154** | 0.546 → **0.471** | 0.7673 |
| LightGBM | 0.7618 | 0.7612 | 0.187 → 0.154 | 0.547 → 0.470 | 0.7683 |
| Random Forest | 0.7268 | 0.7266 | 0.178 → 0.162 | 0.532 → 0.494 | 0.7316 |

Two things to read here:

1. **Calibration is nearly free on discrimination and large on sharpness.** Isotonic recalibration
   costs XGBoost ~0.002 AUC but cuts Brier from 0.186 to 0.154 and LogLoss from 0.546 to 0.471 — the
   scores stop being merely *ranked* and start being *correct in level*, which is what
   `PD × LGD × EAD` needs. (Why this matters is detailed in
   [`03-validation-methodology.md`](03-validation-methodology.md).)
2. **The trees that win, win together.** XGBoost and LightGBM land in a dead heat at **0.7612**
   calibrated; Random Forest (0.7266) does not even clear the 0.7498 AutoML baseline.

## 3. What shipped, and why the losers are kept

**Shipped: the calibrated XGBoost** (`pd_xgboost.joblib`) — tied with LightGBM on AUC, marginally
better calibrated, and the family the rest of the stack (hazard model, SHAP) reuses. The chosen
hyperparameters (`pd_xgboost_best_config.json`): 600 trees, depth 5, learning rate 0.03,
subsample/colsample 0.8, `min_child_weight` 5, `reg_lambda` 1.0.

The LightGBM and Random Forest challengers are **deliberately retained** as committed artifacts
(`pd_finetune_lightgbm.csv`, `pd_finetune_rf.csv`, their scripts and best-configs). They're the
due-diligence record that the choice was contested and measured, not assumed — RF's underperformance
is itself evidence that the gradient-boosted families are the right tool here.

## 4. The ceiling is the finding

![Out-of-time challenger matrix](finetuning_matrix.png)

Across the v1→v4 build, all the gradient-boosted models converge at **~0.745–0.750 AUC** before the
macro overlay, then step to **0.7612** once leakage-safe TTC macro is added (see
[`01-feature-engineering.md`](01-feature-engineering.md) for the ablation). The flat pre-macro
plateau across XGBoost / LightGBM / RF / logistic and an independent FLAML search is the point: the
limit is **information-bound, not model-bound** — the public Prosper fields simply don't carry more
separable signal, so the honest deliverable is a *measured* number on the out-of-time set, not an
inflated one from a leakier setup. The shipped challenger is calibrated XGBoost at **test AUC 0.7612**.

## 5. Reproduce

```
.venv\Scripts\python.exe modeling\run_version.py        # builds the v1->v4 matrix inputs
.venv\Scripts\python.exe modeling\results_visual.py     # -> docs/finetuning_matrix.png
```

The per-model finetune scripts under `modeling/` (xgboost / lightgbm / rf) write the
`pd_finetune_*.csv` and `pd_*_best_config.json` tables; the AutoML baseline writes
`pd_baseline_automl.csv`. See the model card ([`06-model-card.md`](06-model-card.md)) for intended use
and limitations.

# Phase 2 — Discrete-time hazard model (detailed plan)

**Goal:** replace the single lifetime PD + empirical default-timing curve in the finance engine
with a **covariate-driven monthly hazard** `h(t | x)` — the chance a loan defaults *in month t,
given it survived to t and given the borrower x*. From that hazard we get a per-loan **PD term
structure** (how default risk is spread over the life of the loan), which is what a discounted
lifetime ECL really needs. As a bonus it **recovers the 56,576 `Current` loans** v1 dropped, by
entering them honestly as *censored* (still-alive) observations instead of throwing them away.

This plan is concrete enough to build from. It uses only fields confirmed present in the data.

---

## 1. Why this, and why now

- **Today:** `finance.py` takes one lifetime PD and smears it across the loan's months with a
  *fixed empirical curve* — the same shape for every loan. That's a placeholder.
- **Want:** a hazard that depends on **both the borrower and the month**, so a thin-file borrower
  and a thick-file borrower get different *shapes*, not just different levels. Early-life vs
  late-life default risk is the whole point of a term structure.
- **Validation fit:** it slots straight into the out-of-time discipline already built (Part C) —
  train on older vintages, test on newer.

## 2. The loan-month panel (the core data step)

Reshape each loan into **one row per month it was alive** ("person-period" format). A loan that
ran 14 months becomes 14 rows, `t = 1..14`. The target is **`defaulted_this_month`** — `1` only
in the month the loan defaulted, `0` every other month. Rows after the event don't exist.

### Event vs. censoring (from `LoanStatus`)

| LoanStatus | meaning | treatment | event? |
|---|---|---|---|
| `Defaulted`, `Chargedoff` | went bad | event at the default month | **1** in last month |
| `Completed` | paid off | survived, censored at payoff | 0 throughout |
| `Current` | still paying at snapshot | **censored** at months-observed | 0 throughout |
| `FinalPaymentInProgress` | finishing | treat as `Completed` (censored) | 0 |
| `Past Due (*)` | delinquent, not yet charged off | **censor** at last observed month (conservative); revisit treating `>120 days` as event | 0 |
| `Cancelled` (5 rows) | never really a loan | drop | — |

### Observed duration `T_obs` (get this right — it's the #1 correctness risk)

- **Closed loans** (`Completed` / `Defaulted` / `Chargedoff`): `T_obs` = whole months between
  `LoanOriginationDate` and `ClosedDate`. This is the *true* lifetime — do **not** use
  `LoanMonthsSinceOrigination` here, which is measured to the data snapshot and can overshoot.
- **`Current` loans** (no `ClosedDate`): `T_obs` = `LoanMonthsSinceOrigination` (months to the
  snapshot). This is the censoring time.
- **Sanity check to run first:** for closed loans, confirm `ClosedDate − LoanOriginationDate`
  agrees with `LoanMonthsSinceOrigination` within a month or two; investigate if not.

### Covariates per row

- The **same origination-time features the production model uses** (base + engineered +
  `RiskCluster`), held **constant** across a loan's months (they're known at origination).
- The **month index `t`** as a feature — plus a small **spline / piecewise encoding of `t`** so
  the hazard shape can bend (default risk typically rises then falls over a loan's life).
- *(Optional v2 of this phase)* **time-varying macro**: instead of origination-month macro, join
  the macro for calendar month = origination + `t`. This lets the hazard react to the economy as
  the loan ages. Start origination-fixed; add this once the base model works.

### Scale

~84k usable loans × ~20–30 observed months ≈ **2–3 million loan-month rows**, with defaults a tiny
fraction of them. Plan for it (next section).

## 3. The hazard model

- **Estimator:** the same **XGBoost** classifier, fit on the panel. Its predicted probability for
  a `(loan, month)` row *is* the discrete-time hazard `h(t | x)`. Reuse the existing
  `build_preprocessor` + FLAML harness; the only new column is the time encoding.
- **Severe class imbalance** (most loan-months are non-events): handle with **`scale_pos_weight`**
  (already wired) and/or **down-sample the non-event rows** (keep all event rows + a random sample
  of survivors, with case weights to undo the sampling). Down-sampling also tames the 2–3M rows.
- **Calibrate** (isotonic) and **SHAP-explain** exactly as the production PD model does — the
  hazard then plugs into the same explainability story.

## 4. From hazard to a PD term structure

For each loan, walk its months and compound the survival probability:

```
S(0) = 1
S(t) = S(t-1) · (1 − h(t | x))            # chance of surviving to the end of month t
marginal_PD(t) = S(t-1) · h(t | x)        # chance of defaulting *in* month t
lifetime_PD   = 1 − S(T) = Σ marginal_PD(t)
```

`marginal_PD(t)` over `t = 1..T` **is the term structure** the ECL engine wants.

## 5. Wire into the finance engine

- In `finance.py`, replace the empirical default-timing curve with the per-loan `marginal_PD(t)`
  vector from the hazard model. The discounted lifetime ECL becomes
  `Σ_t marginal_PD(t) · LGD · EAD(t) · discount(t)` — same structure, now with a *model-driven*,
  borrower-specific timing instead of one fixed shape.
- Keep the empirical curve as a fallback when the hazard model artifact is absent (mirrors how the
  app already falls back gracefully).

## 6. Benchmark — scikit-survival

Stand up a **continuous-time** benchmark so we can say "we evaluated the alternative":
`RandomSurvivalForest` and `GradientBoostingSurvivalAnalysis`. They output a survival function;
evaluate it on a **monthly grid** and difference it to get comparable marginal PDs. Ship the
discrete-time panel + XGBoost as production (better monthly-ECL fit, reuses our calibration/SHAP);
use scikit-survival as the benchmark + metrics layer.

## 7. Evaluation & validation

- **Survival metrics** (from scikit-survival, model-agnostic): time-dependent AUC
  (`cumulative_dynamic_auc`), IPCW concordance (`concordance_index_ipcw`), and integrated Brier
  score (`integrated_brier_score`).
- **Timing calibration:** predicted vs realized default month distribution (does the term
  structure's *shape* match reality, not just the lifetime total).
- **Out-of-time:** train on earlier vintages (`LoanOriginationQuarter` / year), test on later —
  the same split discipline as Part C. Report all of the above on the OOT set.

## 8. File plan

- `data/build_loan_month_panel.py` — builds + caches the person-period panel (the §2 logic);
  writes `data/processed/loan_month_panel.parquet`. The one piece of genuinely new data work.
- `modeling/survival/hazard_xgboost.py` — fit the discrete-time hazard (reuses `finetune` harness),
  save `pd_hazard_xgboost.joblib`.
- `modeling/survival/term_structure.py` — hazard → `S(t)` / `marginal_PD(t)` / lifetime PD (§4),
  the bridge the finance engine calls.
- `modeling/survival/benchmark_sksurv.py` — scikit-survival fit + the survival-metric suite (§6–7).
- `modeling/common/finance.py` — swap the timing curve for the model term structure (§5).
- `requirements.txt` — add `scikit-survival`.

## 9. Sequencing (milestones)

1. **Panel + sanity checks** (`build_loan_month_panel.py`) — get `T_obs`/event/censor right; this
   is the foundation and the main risk. Validate counts: events ≈ 17,010; survivors include the
   56,576 `Current` loans.
2. **Hazard model** — fit, handle imbalance, calibrate; confirm it reproduces the marginal lifetime
   PD roughly in line with the current PD model on resolved loans.
3. **Term structure → finance** — wire `marginal_PD(t)` into the ECL; eyeball curve shapes.
4. **Benchmark + metrics + OOT** — scikit-survival, survival metrics, out-of-time report.
5. **Write-up** — feeds Phase 3 (ECL backtest) and Phase 4 (model card / notebook).

## 10. Risks & gotchas

- **Duration definition** (§2) is the biggest trap — snapshot-vs-close confusion silently corrupts
  every hazard. Sanity-check before modeling.
- **Leakage:** use only origination-time covariates (or properly lagged time-varying macro). Never
  feed a row anything that encodes the future (e.g. `LoanCurrentDaysDelinquent` at snapshot).
- **Imbalance + size:** down-sample survivors with case weights; don't let 3M rows dictate budget.
- **Past-due ambiguity:** start by censoring them; document the choice and test `>120 days`-as-event
  as a sensitivity.
- **Calibration after down-sampling:** sampling distorts the base rate — recalibrate (isotonic on a
  held-out, un-sampled slice) so the hazards are true probabilities for the ECL.

## 11. "Phase 2 done" looks like

- A cached loan-month panel with validated event/censor counts (the `Current` loans recovered).
- A calibrated XGBoost hazard producing per-loan **PD term-structure curves**, SHAP-explainable.
- `finance.py` computing lifetime ECL from the **model** term structure, not the empirical curve.
- A scikit-survival benchmark with time-dependent AUC / IPCW-concordance / integrated-Brier, all
  reported **out-of-time** — the honest, validator-ready version.

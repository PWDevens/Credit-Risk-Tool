# v3 Plan — A time-aware, out-of-time-validated credit model

**Theme: depth over breadth.** v1 built the `EL = PD × LGD × EAD` machinery; v2 added a
fine-tuned, explainable PD model and a financial decisioning engine (lifetime ECL, expected
profit / RAROC, risk-based pricing). v3 is not about adding surface area — it's about making
the model **defensible**: time-aware, validated the way credit risk is actually validated,
and tied back to realized dollars.

Across three model families (XGBoost / LightGBM / Random Forest) and a FLAML search, PD
discrimination converged at **~0.745–0.750 AUC**. That tells us the ceiling is **feature-bound,
not model-bound** — so v3's accuracy work is feature engineering, and the bigger wins are in
*how we model time* and *how we validate*, not in swapping algorithms again.

---

## Design decision: discrete-time hazard, with scikit-survival as a benchmark

The v2 finance engine spreads a single lifetime PD over the loan's life using an **empirical**
default-timing curve. v3 replaces that with a real, covariate-driven hazard `h(t | x)`.

**Primary approach — discrete-time hazard via a person-period panel (no survival library):**
reshape to one row per loan per month it was alive, with a binary `defaulted_this_month`
target and the month index (or a spline of it) as a feature, then fit the existing **LightGBM**
classifier on it. The predicted probability *is* the monthly hazard. This:

- produces the **monthly marginal PD** the ECL engine needs, with no discretization step;
- reuses the v2 **calibration + SHAP** infrastructure unchanged;
- lets the hazard depend on covariates **and** time (time-varying effects);
- naturally **recovers the `Current` loans v1 dropped** — they enter as *censored* observations
  (exposure up to the snapshot), fixing the "incomplete performance window" compromise honestly.

**Benchmark / metrics — scikit-survival:** it is continuous-time (Cox, `RandomSurvivalForest`,
`GradientBoostingSurvivalAnalysis`), so feeding monthly ECL would require evaluating its survival
function on a monthly grid and differencing it — an extra step. It earns its place as (a) an
**independent continuous-time benchmark** and (b) the **evaluation toolkit** we want regardless:
`cumulative_dynamic_auc` (time-dependent AUC), `concordance_index_ipcw`, `integrated_brier_score`.

> **Bottom line:** ship the discrete-time panel + LightGBM as the production hazard model;
> use scikit-survival as the benchmark and survival-metrics layer. Better technical fit for
> monthly-granularity ECL, and a stronger "I evaluated the alternative" narrative.

---

## Workstream 1 — Lift the ~0.75 AUC (feature engineering)

The only real accuracy lever, since the model-family ceiling is already reached.

- **WOE / target encoding** for high-cardinality nominals (`Occupation`, `BorrowerState`)
  instead of one-hot — reuse the existing `woe_iv()` / `iv_ranking()` in `data_analysis.py`.
  Highest-probability single win.
- **Interaction terms** (e.g. DTI × utilization, inquiries × delinquencies) and **pruning** of
  the redundant credit-score trio (`credit_score_mid` + the two raw bounds).
- **Point-in-time macro overlay:** join unemployment / interest-rate environment on
  `LoanOriginationDate` (+ `BorrowerState`). Both an accuracy lever and a finance differentiator
  (point-in-time vs through-the-cycle PD).
- **Honest target:** public Prosper data realistically tops out around **0.76–0.79 AUC**. The
  deliverable is a measured lift on the out-of-time set (Workstream 3), not an inflated number.

**Deliverable:** a feature-engineering pass with before/after lift reported on the OOT set, and
a short note on which features moved the needle (and which didn't).

## Workstream 2 — Discrete-time hazard model (PD term structure)

- **Build the loan-month panel:** event = default month (from `ClosedDate − LoanOriginationDate`);
  censor time = `LoanMonthsSinceOrigination` for still-current loans. Handle the heavy non-event
  class imbalance (most loan-months are non-default) with weighting / downsampling.
- **Fit the LightGBM hazard** with the month index as a feature → `h(t | x)`; wire it into
  `finance.py` to replace the empirical timing curve. Calibrate and SHAP-explain as in v2.
- **Benchmark** with scikit-survival (`RandomSurvivalForest` / `GradientBoostingSurvivalAnalysis`).
- **Evaluate** both with time-dependent AUC, IPCW concordance, and integrated Brier score.

**Deliverable:** a hazard model producing PD term-structure curves, a survival-metrics comparison
vs the scikit-survival benchmark, and an updated ECL that uses model-driven monthly hazards.

## Workstream 3 — Validation rigor (the portfolio differentiator) — DO FIRST

- **Out-of-time (vintage) validation:** train on earlier originations, test on later (e.g. train
  ≤ 2012, test 2013–2014). This is *the* credit-risk validation standard and far more convincing
  than a random split. Stand this up first so every later change is judged honestly.
- **ECL backtest vs realized dollar losses** by vintage — ties the financial engine to reality
  (predicted-vs-actual loss, calibration-by-vintage plots).
- **Model card (SR 11-7 style)** documenting intended use, data, metrics, limitations, and the
  fair-lending treatment of geography/occupation.
- **Reject inference / selection bias** note — Prosper only shows *funded* loans, so the model is
  conditional on Prosper's own approval. Acknowledge it; sketch a reweighting/Heckman approach.

**Deliverable:** an OOT validation harness, an ECL backtest, and the model-card / bias write-ups.

## Workstream 4 — Packaging

- A results **notebook / write-up** with the money charts: PD term-structure curves,
  calibration-by-vintage, OOT performance, and the ECL backtest. For a portfolio, the narrative
  is half the value.

**Deliverable:** a self-contained results notebook and a README refresh pointing to it.

---

## Sequence

**3 → 1 → 2 → 4.** Build the out-of-time validation harness *first* so feature work and the
hazard model are measured honestly; finish with the write-up that tells the story.

## What "v3 done" looks like

- PD discrimination reported on an **out-of-time** set (not a random split), with a measured,
  honestly-framed lift from feature engineering.
- A **discrete-time hazard model** driving a model-based PD term structure in the ECL engine,
  benchmarked against scikit-survival with proper survival metrics.
- An **ECL backtest** that ties predicted losses to realized dollars by vintage.
- A **model card** and a **results notebook** — the artifacts that make it read as a credible,
  finance-literate credit-risk project rather than a Kaggle-style accuracy chase.

## Deferred beyond v3

Prepayment / competing-risks survival, a two-stage (cure-rate → severity) LGD model, macro
**stress scenarios** (CCAR/IFRS 9-style), and SHAP-driven **ECOA adverse-action** reason codes.

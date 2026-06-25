# Macro feature decision: why we use *through-the-cycle (TTC)–anchored* macro

**Decision (2026-06):** when the PD model uses the economy as a feature, it uses the
**TTC-anchored** form of unemployment and the fed funds rate — not the raw point-in-time level.
This note explains, in plain language, how we got there and why.

---

## 1. The question

We have an economic overlay: the national **unemployment rate** and **fed funds rate** in the
month each loan was made. The economy at origination is known when you underwrite, so it's a
fair (leakage-safe) feature. The question was simply: **does adding it make the model better,
and if so, in what form?**

There are three candidate forms (all built by `data/build_macro_features.py`):

| form | what it is |
|---|---|
| **raw** | the actual unemployment / fed funds level that month (spiky) |
| **TTC-anchored** | the level smoothed over time and pulled toward its long-run average |
| **robust** | the cyclical "gap" from normal + the trailing year-over-year change |

## 2. The first answer looked great — and was misleading

On our normal test (a **random split** — shuffle all loans 2009–2014, train on 80%, test on
20%), adding **raw** macro lifted AUC by about **+0.015** for the tree models and pushed the
best model to ~0.76, *beating* the AutoML baseline (0.750). Exciting.

But there's a trap. Unemployment was high in 2009–2011 (~9–10%) and lower by 2013–2014
(~6–7%), so **the unemployment number doubles as a "date stamp" for when the loan was made.**
And the early loans defaulted more (worse economy *and* Prosper's early underwriting was
looser). On a random split, the model sees some 2010 loans in training *and* some in testing,
so it learns "loans stamped ~9.5% unemployment go bad" — it's **memorizing which vintages were
bad and reading the vintage off the unemployment value**, not learning a general economic
effect. That's a number that looks powerful in a backtest but wouldn't help on a real future
applicant.

## 3. The honest test: out-of-time, with a fair baseline

To catch that trap we ran an **out-of-time (OOT) split** — train on the **old** loans
(pre-2013), test on the **newer** ones (2013–2014), the way a model is actually used: built
today, applied to applicants you haven't seen. We ran every model two ways (random vs OOT) and
two macro forms (raw vs TTC), then added a **base model with no macro at all** as the reference
so we could measure macro's *true* contribution (the OOT test is also just a harder, different
cohort — 12.5% bad vs ~25% — and we didn't want to credit macro for that).

**Macro's isolated contribution (macro model − base model, same split):**

| model | lift on RANDOM | lift OOT (raw macro) | lift OOT (**TTC** macro) |
|---|---|---|---|
| XGBoost | +0.015 | +0.008 | +0.007 |
| LightGBM | +0.014 | **+0.017** | **+0.025** |
| Random Forest | +0.014 | +0.003 | +0.001 |
| Logistic | −0.001 | **−0.024** | +0.000 |

Three things fall out of this table:

1. **Macro is *partly* real, not pure date-stamp.** Out-of-time, it still adds ~+0.008
   (XGBoost) to ~+0.017 (LightGBM). The economy genuinely carries default signal. But for
   XGBoost the random-split number (+0.015) was about **half** inflation — only half survived.

2. **Raw macro can actively hurt.** For the logistic model, raw macro is **−0.024**
   out-of-time: it overfits the exact unemployment levels it saw in training and mis-fires on
   the new period.

3. **TTC anchoring fixes both problems.** It *rescues* the logistic model (−0.024 → ~0) and
   *improves* LightGBM (+0.017 → **+0.025**). TTC smooths the spikes and shrinks toward the
   long-run average, so the model leans on the cycle's general shape rather than memorizing one
   era's exact numbers — which is precisely what generalizes to a new period.

   (On a random split TTC looks identical to raw for the trees — they only care about the
   *order* of values, which smoothing preserves. Its benefit only shows up out-of-time. That's
   the whole point: it buys robustness, not a higher backtest score.)

## 4. The decision

**Use TTC-anchored macro.** It delivers a real, out-of-time-validated lift for the gradient-
boosted models (LightGBM ~+0.025, XGBoost ~+0.007) and protects the interpretable model from
the vintage-overfit that raw macro causes. The deployable benefit is **smaller than the +0.015
random-split headline but genuinely positive and downturn-robust** — the right trade for a
credit model that has to survive a changing economy.

Raw macro is kept in the code (`MACRO_FEATURES_RAW`) only for reproducing the experiments
above; it is not the production form.

## 5. How it's wired in the repo

- **`data/features.py`** — `MACRO_FEATURES = MACRO_FEATURES_TTC` makes TTC the canonical macro
  set; "macro" anywhere unqualified means these columns. `current_macro()` returns the latest
  available month's TTC values for scoring a brand-new loan.
- **`modeling/common/data.py`** — `macro_set='ttc'` (the default whenever macro is on) selects
  the TTC columns; `'raw'`/`'robust'` remain available for experiments.
- **`modeling/common/finetune.py`** — the shipped per-model challengers train on
  base + engineered + cluster + **TTC macro**.
- **`modeling/common/predictor.py`** — at scoring, the latest TTC macro is attached to the
  applicant row. A new loan is originated *now*, so its macro is the most recent month.

### A subtlety worth stating

For a batch of applicants scored on the **same day**, macro is *identical* for all of them, so
it does **not** change their *ranking*. What it does is shift the **whole PD level** up or down
with the economy. That doesn't matter for "who is riskier than whom," but it matters a lot for
the **dollar outputs** — expected loss, lifetime ECL, reserves, and risk-based pricing should
rise in a downturn and ease in a recovery. So macro lives in the model as a **through-the-cycle
calibration of the PD level**, which is exactly where a credit-risk practitioner would want it.

---

*Reproduce: `modeling/run_version.py` (the v1–v5 feature matrix) and `modeling/run_partc.py`
(the random-vs-OOT × raw-vs-TTC grid) write `modeling/model-results/version_matrix.csv` and
`partc_matrix.csv`. `modeling/run_finalize.py` adds the no-macro OOT baseline. See also
`.pipeline/time-smoothing.md`.*

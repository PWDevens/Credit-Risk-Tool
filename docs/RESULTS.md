# Results — the v3 money charts

v3 is about making the model **defensible**: time-aware, validated out-of-time, and tied back to
realized dollars. This write-up collects the four charts that carry that story. Each is regenerated
by a script in `modeling/` (see [Reproduce](#reproduce)); the modeling assumptions and limitations
behind them live in the [model card](MODEL_CARD.md).

---

## 1. PD term structure — *when* defaults happen

![PD term structure](pd_term_structure.png)

The discrete-time hazard model predicts a **monthly** probability of default `h(t | x)`. Accumulated
over the loan's life, it gives each borrower a **cumulative default curve** — the chance of having
defaulted by month *t*. The three lines are representative borrowers at the 10th, 50th, and 90th
percentiles of predicted PD: risk separates cleanly, and the curves are **steeper early** then flatten,
matching how consumer defaults concentrate in the first two years. This term structure — not a flat
average — is what drives the timing of the lifetime ECL in `finance.py`.

## 2. Calibration by vintage — predicted vs actual default rate

![PD calibration by vintage](calibration_by_vintage.png)

Discrimination is only half the story; the probabilities have to mean what they say. Per origination
**vintage**, predicted PD tracks the actual default rate closely — the weighted mean gap across
deciles is ≈ **0.017**. The early vintages are the load-bearing evidence (2009 n≈446 … 2012 n≈1.6k);
the latest vintage (2014, n≈10) is too thin to read and is shown only for completeness. The model is
**well-calibrated**, not just well-ranked.

## 3. Out-of-time performance — the honest lens

![Out-of-time challenger matrix](finetuning_matrix.png)

Every feature decision in v3 is judged on an **out-of-time** split (train on earlier originations,
test on later) rather than a random split — the credit-risk standard. The cumulative v1→v4 matrix
shows PD discrimination converging at **~0.745–0.750 AUC** across XGBoost, LightGBM, and Random
Forest. The flat ceiling is the finding: it is **information-bound**, not model-bound, so the honest
deliverable is a measured number on the out-of-time set, not an inflated one. (Calibrated XGBoost is
the shipped challenger at test AUC **0.7612**.)

## 4. ECL backtest — predicted vs realized **dollars** by vintage

![ECL backtest](ecl_backtest.png)

The credibility centerpiece: for the resolved post-2009 book, predicted lifetime EL
(`PD × LGD × EAD`, undiscounted) summed by vintage vs **realized** dollar losses
(`LP_NetPrincipalLoss`). Overall the engine predicts ≈ **$31.7M** against ≈ **$31.2M** realized — a
**1.02** predicted/realized ratio, i.e. calibrated to within ~2% in dollars. Per vintage:

| Vintage | Loans | Defaults | Predicted EL | Realized loss | Ratio |
|--------:|------:|---------:|-------------:|--------------:|------:|
| 2009 | 2,034 | 308 | $0.89M | $0.80M | 1.11 |
| 2010 | 5,625 | 937 | $3.06M | $2.49M | 1.23 |
| 2011 | 7,633 | 2,145 | $10.56M | $9.85M | 1.07 |
| 2012 | 8,140 | 2,622 | $14.89M | $15.64M | 0.95 |
| 2013\* | 2,635 | 338 | $2.31M | $2.44M | 0.95 |
| 2014\* | 69 | 0 | $0.005M | $0.00M | — |
| **ALL** | **26,136** | **6,350** | **$31.71M** | **$31.22M** | **1.02** |

\* out-of-time (origination ≥ 2013). The OOT vintages are thin, so the dollar read there is
directional; the 2009–2012 in-sample vintages carry the calibration.

---

## Reproduce

From the repo root with the project `.venv` active, in order:

```bash
.venv\Scripts\python.exe modeling\calibration_report.py    # calibration tables (decile + vintage)
.venv\Scripts\python.exe modeling\ecl_backtest.py          # predicted-vs-realized $ backtest
.venv\Scripts\python.exe modeling\run_version.py           # OOT matrix inputs ...
.venv\Scripts\python.exe modeling\results_visual.py        # ... -> docs/finetuning_matrix.png
.venv\Scripts\python.exe modeling\results_charts.py        # builds charts 1, 2, 4 into docs/
```

`results_charts.py` reads the CSVs the earlier scripts write (`modeling/model-results/`) and the
fitted hazard artifact, and emits `docs/pd_term_structure.png`, `docs/calibration_by_vintage.png`,
and a committed `docs/ecl_backtest.png`. See the [model card](MODEL_CARD.md) for intended use, data,
metrics, limitations, fair-lending treatment, and the reject-inference note.

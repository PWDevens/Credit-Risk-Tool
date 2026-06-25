"""Phase 4 — build the results write-up's money charts into docs/ (committed PNGs).

The results write-up (docs/results.md) renders on GitHub, so every embedded chart must be a
committed file under docs/ (model-results/ is gitignored). This script builds the two charts that
had no committed PNG yet — PD term structure and calibration-by-vintage — plus a committed copy of
the ECL backtest chart. The out-of-time matrix (docs/finetuning_matrix.png) already exists and is
reused as-is.

    .venv\\Scripts\\python.exe modeling\\results_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))
from common import data as D                       # noqa: E402  (load_frame, OOT_CUTOFF)
from survival import term_structure as TS          # noqa: E402  (available, hazard_curve, survival_from_hazard)
from ecl_backtest import plot_backtest             # noqa: E402  (reuse the grouped-bar plotter)

RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
DOCS_DIR = REPO_ROOT / "docs"
PD_FINE = REPO_ROOT / "modeling" / "probability-of-default" / "pd_xgboost.joblib"
CALIB_CSV = RESULTS_DIR / "calibration_vintage.csv"
ECL_CSV = RESULTS_DIR / "ecl_backtest_by_vintage.csv"

# Low -> High risk, visually ordered (green -> house purple -> warm orange).
TIER_COLORS = {"Low risk": "#3FA34D", "Median risk": "#534AB7", "High risk": "#D85A30"}


def plot_calibration_by_vintage(calib: pd.DataFrame, out_png: Path) -> None:
    """Grouped bars: predicted vs actual default rate per origination vintage."""
    years = calib["year"].astype(int).to_numpy()
    x = np.arange(len(years))
    w = 0.4

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, calib["pred_PD"], width=w, color="#534AB7", label="Predicted PD")
    ax.bar(x + w / 2, calib["actual_PD"], width=w, color="#D85A30", label="Actual PD")

    # Annotate each year group with its loan count.
    top = np.maximum(calib["pred_PD"].to_numpy(), calib["actual_PD"].to_numpy())
    for xi, ti, ni in zip(x, top, calib["n"].to_numpy()):
        ax.text(xi, ti, f"n={int(ni):,}", ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_title("PD calibration by vintage (predicted vs actual default rate)")
    ax.set_xlabel("Origination vintage (year)")
    ax.set_ylabel("Default rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def pick_representative_borrowers(test: pd.DataFrame, art: dict,
                                  k_each: int = 1) -> list[tuple[str, dict]]:
    """Three labelled borrowers at the 10th/50th/90th percentiles of predicted PD (nearest-rank).

    Each borrower is returned as a record dict built from a *single-row DataFrame*
    (`clean.iloc[[pos]].to_dict("records")[0]`), which preserves every column's native dtype.

    We restrict candidates to rows with **complete** model features (`dropna`). A borrower with a
    NaN numeric feature (e.g. a missing DebtToIncomeRatio) makes term_structure.hazard_curve build
    an object-dtype column when it broadcasts that scalar NaN across the month grid, which the
    XGBoost hazard model rejects. Illustrative curves should use fully-populated borrowers anyway.
    """
    clean = test.dropna(subset=art["feature_cols"]).reset_index(drop=True)
    p = art["model"].predict_proba(art["preprocessor"].transform(clean[art["feature_cols"]]))[:, 1]
    order = np.argsort(p)                       # ascending predicted PD
    n = len(order)
    picks = {
        "Low risk": order[int(0.10 * (n - 1))],
        "Median risk": order[int(0.50 * (n - 1))],
        "High risk": order[int(0.90 * (n - 1))],
    }
    return [(label, clean.iloc[[pos]].to_dict("records")[0]) for label, pos in picks.items()]


def plot_pd_term_structure(reps: list[tuple[str, dict]], out_png: Path, term: int = 36) -> None:
    """Cumulative default probability (1 - S(t)) over months 1..term, one line per risk tier."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, row in reps:
        h = TS.hazard_curve(row, term)
        cdf = 1.0 - TS.survival_from_hazard(h)          # length == len(h) == term
        months = np.arange(1, len(cdf) + 1)
        ax.plot(months, cdf * 100.0, marker="", linewidth=2,
                color=TIER_COLORS.get(label, "#534AB7"), label=label)

    ax.set_title("PD term structure — cumulative default probability by risk tier")
    ax.set_xlabel("Months since origination")
    ax.set_ylabel("Cumulative default probability (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    if not CALIB_CSV.exists():
        raise SystemExit("calibration_vintage.csv not found — run modeling/calibration_report.py first.")
    if not ECL_CSV.exists():
        raise SystemExit("ecl_backtest_by_vintage.csv not found — run modeling/ecl_backtest.py first.")
    if not PD_FINE.exists():
        raise SystemExit("pd_xgboost.joblib not found — run finetune_xgboost.py first.")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 2 — calibration by vintage.
    calib = pd.read_csv(CALIB_CSV)
    plot_calibration_by_vintage(calib, DOCS_DIR / "calibration_by_vintage.png")
    print(f"saved {DOCS_DIR / 'calibration_by_vintage.png'}")

    # Chart 4 — committed copy of the ECL backtest (reuse ecl_backtest.plot_backtest).
    ecl = pd.read_csv(ECL_CSV)
    plot_backtest(ecl, DOCS_DIR / "ecl_backtest.png")
    print(f"saved {DOCS_DIR / 'ecl_backtest.png'}")

    # Chart 1 — PD term structure (needs the fitted hazard artifact).
    if TS.available():
        art = joblib.load(PD_FINE)
        test = D.load_frame("test")
        reps = pick_representative_borrowers(test, art)
        plot_pd_term_structure(reps, DOCS_DIR / "pd_term_structure.png")
        print(f"saved {DOCS_DIR / 'pd_term_structure.png'}")
    else:
        print("WARNING: pd_hazard_xgboost.joblib not found — skipping PD term-structure chart.")


if __name__ == "__main__":
    main()

"""Phase 3 — ECL backtest: predicted vs realized dollar losses by vintage.

Ties the Expected-Loss engine to reality. For the resolved post-2009 book, predicted lifetime EL
(production calibrated XGBoost PD x LGD x EAD, undiscounted) is summed by origination vintage and
compared against realized dollar losses (LP_NetPrincipalLoss). A predicted/realized ratio near 1.0
means the dollars are well calibrated. This is the dollar analogue of the probability calibration
in modeling/calibration_report.py.

Undiscounted EL is the right comparison because realized LP_NetPrincipalLoss is a nominal figure.
The Phase-2 hazard term structure governs loss *timing* (and the discounted ECL), but since
sum(marginal_pd) == lifetime_pd the undiscounted lifetime total is timing-independent — so it is
intentionally not used here.

    .venv\\Scripts\\python.exe modeling\\ecl_backtest.py

Saves model-results/ecl_backtest_by_vintage.csv and .png.
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
from common import data as D                 # noqa: E402  (OOT_CUTOFF, load_frame)
from common.predictor import RiskPredictor   # noqa: E402  (reuse .ead, .lgd AutoGluon predictors)
import features as F                          # noqa: E402  (PD_TARGET)

RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
PD_FINE = REPO_ROOT / "modeling" / "probability-of-default" / "pd_xgboost.joblib"
OUT_CSV = RESULTS_DIR / "ecl_backtest_by_vintage.csv"
OUT_PNG = RESULTS_DIR / "ecl_backtest_by_vintage.png"

FMT = {"predicted_el": "{:,.0f}".format, "realized_loss": "{:,.0f}".format,
       "ratio": "{:.2f}".format}


def score_book(frame: pd.DataFrame, rp: RiskPredictor, art: dict) -> pd.DataFrame:
    """Per-loan predicted EL ($) and realized loss ($), tagged with origination year.

    Predicted EL = lifetime_PD x LGD x EAD (undiscounted). PD is the production calibrated XGBoost
    scored on the processed frame — its macro columns are joined point-in-time at each loan's
    origination, so this is the honest "what the model would have predicted at origination" PD (not
    the live app's current_macro 'today' scoring). LGD/EAD are the production AutoGluon baselines.
    """
    pd_life = art["model"].predict_proba(art["preprocessor"].transform(frame[art["feature_cols"]]))[:, 1]
    lgd = np.clip(rp.lgd.predict(frame).to_numpy(dtype=float), 0.0, 1.0)
    ead = np.clip(rp.ead.predict(frame).to_numpy(dtype=float), 0.0, None)
    predicted_el = pd_life * lgd * ead
    realized = frame["LP_NetPrincipalLoss"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    year = pd.to_datetime(frame["LoanOriginationDate"], errors="coerce").dt.year

    book = pd.DataFrame({
        "year": year.to_numpy(),
        "is_bad": frame[F.PD_TARGET].astype(int).to_numpy(),
        "predicted_el": predicted_el,
        "realized_loss": realized,
    }).dropna(subset=["year"])
    book["year"] = book["year"].astype(int)
    return book


def aggregate_by_vintage(book: pd.DataFrame) -> pd.DataFrame:
    """Sum predicted vs realized dollars per origination year, plus an ALL summary row."""
    g = book.groupby("year").agg(
        n=("is_bad", "size"), n_default=("is_bad", "sum"),
        predicted_el=("predicted_el", "sum"), realized_loss=("realized_loss", "sum"),
    ).reset_index()
    oot_year = pd.Timestamp(D.OOT_CUTOFF).year
    g["ratio"] = np.where(g["realized_loss"] > 0, g["predicted_el"] / g["realized_loss"], np.nan)
    g["is_oot"] = (g["year"] >= oot_year)
    g = g.sort_values("year").reset_index(drop=True)

    tot_realized = float(g["realized_loss"].sum())
    all_row = {
        "year": "ALL", "n": int(g["n"].sum()), "n_default": int(g["n_default"].sum()),
        "predicted_el": float(g["predicted_el"].sum()), "realized_loss": tot_realized,
        "ratio": (float(g["predicted_el"].sum()) / tot_realized) if tot_realized > 0 else np.nan,
        "is_oot": "",
    }
    out = pd.concat([g.astype({"year": object}), pd.DataFrame([all_row])], ignore_index=True)
    return out


def plot_backtest(vint: pd.DataFrame, out_png: Path) -> None:
    """Grouped bars: predicted vs realized dollars per vintage year (ALL row excluded)."""
    v = vint[vint["year"] != "ALL"].copy()
    years = v["year"].astype(int).to_numpy()
    x = np.arange(len(years))
    w = 0.4

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, v["predicted_el"], width=w, color="#534AB7", label="Predicted EL")
    ax.bar(x + w / 2, v["realized_loss"], width=w, color="#D85A30", label="Realized loss")

    # Ratio callout above each year's taller bar.
    top = np.maximum(v["predicted_el"].to_numpy(), v["realized_loss"].to_numpy())
    for xi, ti, ri in zip(x, top, v["ratio"].to_numpy()):
        if np.isfinite(ri):
            ax.text(xi, ti, f"{ri:.2f}x", ha="center", va="bottom", fontsize=8, color="#333333")

    # Mark out-of-time vintages with an asterisk in the tick label.
    labels = [f"{y}*" if bool(o) else str(y) for y, o in zip(years, v["is_oot"].to_numpy())]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("ECL backtest — predicted vs realized loss by vintage")
    ax.set_xlabel("Origination vintage (year; * = out-of-time, >= 2013)")
    ax.set_ylabel("Dollars ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    if not PD_FINE.exists():
        raise SystemExit("pd_xgboost.joblib not found — run finetune_xgboost.py first.")
    art = joblib.load(PD_FINE)
    rp = RiskPredictor()

    frame = pd.concat([D.load_frame("train"), D.load_frame("test")], ignore_index=True)
    book = score_book(frame, rp, art)
    vint = aggregate_by_vintage(book)

    print("=== ECL backtest — predicted vs realized $ loss by vintage (resolved post-2009 book) ===")
    print(vint.to_string(index=False, formatters=FMT))
    overall = vint.loc[vint["year"] == "ALL", "ratio"].iloc[0]
    print(f"\nOverall predicted/realized dollar ratio: {overall:.2f}  "
          f"(1.00 = perfectly calibrated $)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vint.to_csv(OUT_CSV, index=False)
    plot_backtest(vint, OUT_PNG)
    print(f"saved {OUT_CSV.name} + {OUT_PNG.name} to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

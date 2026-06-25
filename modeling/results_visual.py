"""Plot the PD challenger matrix (v1->v4) for all four models + the AutoML reference line.

Reads model-results/version_matrix.csv (from run_matrix.py) and the existing AutoML baseline,
writes docs/finetuning_matrix.png.

    python modeling/results_visual.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "modeling" / "model-results"
OUT = REPO_ROOT / "docs" / "finetuning_matrix.png"

VERSIONS = ["v1_base", "v2_cluster", "v3_engineered", "v4_macro"]
VLABELS = ["v1\nbase", "v2\n+cluster", "v3\n+engineered", "v4\n+macro"]
MODELS = ["lightgbm", "xgboost", "rf", "logistic"]
STYLE = {  # color + marker so series are distinguishable without relying on color alone
    "lightgbm": ("#1D9E75", "o"), "xgboost": ("#534AB7", "^"),
    "rf": ("#D85A30", "s"), "logistic": ("#185FA5", "D"),
}


def main() -> None:
    m = pd.read_csv(RESULTS_DIR / "version_matrix.csv")
    automl_auc = prosper_auc = None
    base = RESULTS_DIR / "pd_baseline_automl.csv"
    if base.exists():
        a = pd.read_csv(base)
        am = a[a["model"] == "automl"]
        if not am.empty:
            automl_auc = float(am["AUC"].iloc[0])
        pc = a[a["model"] == "prosper_score_champion"]
        if not pc.empty:
            prosper_auc = float(pc["AUC"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        sub = m[m["model"] == model].set_index("version").reindex(VERSIONS)
        if sub["test_auc_cal"].notna().any():
            color, marker = STYLE[model]
            ax.plot(range(len(VERSIONS)), sub["test_auc_cal"], marker=marker, color=color,
                    linewidth=2, markersize=8, label=model)

    if automl_auc is not None:
        ax.axhline(automl_auc, ls="--", color="#888780", linewidth=2,
                   label=f"AutoML baseline ({automl_auc:.3f})")
    if prosper_auc is not None:
        ax.axhline(prosper_auc, ls=":", color="#A32D2D", linewidth=2,
                   label=f"Prosper default prediction ({prosper_auc:.3f})")

    # Span both the Prosper champion (~0.65) and the challenger cluster (~0.74-0.75).
    lo = min([prosper_auc] if prosper_auc is not None else [0.72]) - 0.01
    hi = max([automl_auc] if automl_auc is not None else [0.76]) + 0.015
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(VERSIONS)))
    ax.set_xticklabels(VLABELS)
    ax.set_ylabel("calibrated test AUC")
    ax.set_xlabel("feature-set version")
    ax.set_title("PD challengers across cumulative feature-set versions (v1–v4)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)

    # Caveat callout: the v4 macro jump is a RANDOM-split number. The out-of-time study (Part C)
    # showed ~half of it is a vintage proxy that fades on unseen vintages — so the durable gain is
    # smaller, and macro is shipped TTC-anchored. Annotate so the chart isn't read as a pure gain.
    if len(VERSIONS) >= 4:
        ax.annotate(
            "v4's macro gain is a random-split number.\n"
            "Out-of-time (Part C): ~half is a vintage proxy\n"
            "that fades on unseen vintages — see docs/01-feature-engineering.md",
            xy=(3, 0.760), xytext=(0.35, 0.672), fontsize=7.5, color="#5A3A00",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFF3D6", ec="#C8A24B", lw=1),
            arrowprops=dict(arrowstyle="->", color="#C8A24B", lw=1.4,
                            connectionstyle="arc3,rad=-0.2"))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

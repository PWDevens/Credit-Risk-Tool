"""Build the RiskCluster segmentation for the v3 fine-tuned challengers.

Fits the KMeans pipeline (median-impute -> StandardScaler -> KMeans, k by silhouette) on
the TRAIN split only, persists it to models/risk_cluster.joblib (re-used in production by
RiskPredictor), and writes a 'RiskCluster' column into the processed train/test CSVs so the
fine-tune scripts can pick it up as a categorical feature.

Run AFTER data/features.py and AFTER 'RiskCluster' is added to ENGINEERED_FEATURES /
CATEGORICAL_FEATURES.

    python modeling/build_risk_clusters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "modeling"))
sys.path.insert(0, str(REPO_ROOT / "data"))

import features as F  # noqa: E402
from common import data as D  # noqa: E402


def main() -> None:
    train, test = D.load_frame("train"), D.load_frame("test")

    pipe, info = F.fit_risk_clusters(train)
    print(f"RiskCluster: best_k={info['best_k']}  silhouette={info['silhouette']:.3f}")
    print(f"  silhouette by k: { {k: round(v, 3) for k, v in info['scores'].items()} }")

    (REPO_ROOT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, F.RISK_CLUSTER_PATH)

    for split, df in [("train", train), ("test", test)]:
        df["RiskCluster"] = F.assign_risk_cluster(df, pipe)
        df.to_csv(F.PROCESSED_DIR / f"{split}_data.csv", index=False)
        sizes = df["RiskCluster"].value_counts().sort_index().to_dict()
        # bad rate per cluster — a quick sanity check that clusters separate risk
        rates = df.groupby("RiskCluster")[F.PD_TARGET].mean().round(3).to_dict()
        print(f"  {split}: sizes={sizes}  bad_rate_by_cluster={rates}")

    print(f"saved {F.RISK_CLUSTER_PATH}")


if __name__ == "__main__":
    main()

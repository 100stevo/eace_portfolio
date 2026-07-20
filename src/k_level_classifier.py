"""
Module 6 (Pillar B - Supporting): Automated K-Level Classification
------------------------------------------------------------------
Uses K-Means on wearable gait/activity data to identify natural activity
groupings, maps them to clinical K-Levels (0-4), then trains a Random Forest
to automate the mapping for future incoming data.

K-Level clinical reference (Medicare Functional Classification Level):
  K0 - No ability/potential to ambulate safely
  K1 - Limited household ambulator
  K2 - Limited community ambulator
  K3 - Community ambulator, variable cadence
  K4 - Exceeds basic ambulation (e.g. child, athlete, high-demand adult)

Run standalone via:
    python k_level_classifier.py --in ../outputs/features.csv --outdir ../outputs
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

K_LEVEL_LABELS = ["K0", "K1", "K2", "K3", "K4"]
ACTIVITY_FEATURES = ["Daily_Step_Count", "Gait_Symmetry_Index"]


def classify_k_level(activity_df: pd.DataFrame, n_clusters: int = 5) -> dict:
    """
    Cluster wearable activity data into 5 groups, map clusters to K-Levels by
    ascending mean step count, and train a Random Forest to replicate the
    mapping for future incoming records.

    Parameters
    ----------
    activity_df : dataframe with ACTIVITY_FEATURES columns.
    n_clusters : number of K-Means clusters (default 5, matching K0-K4).

    Returns
    -------
    dict with keys: kmeans, scaler, cluster_to_klevel, rf_classifier,
    accuracy, labeled_df.
    """
    work = activity_df.dropna(subset=ACTIVITY_FEATURES).copy()
    X = work[ACTIVITY_FEATURES].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work["cluster"] = kmeans.fit_predict(X_scaled)

    # Map clusters -> K-Levels by ascending mean daily step count
    cluster_means = work.groupby("cluster")["Daily_Step_Count"].mean().sort_values()
    cluster_to_klevel = {
        cluster: K_LEVEL_LABELS[rank] for rank, cluster in enumerate(cluster_means.index)
    }
    work["K_Level"] = work["cluster"].map(cluster_to_klevel)

    # Train a Random Forest to replicate the cluster->K-Level mapping for new data
    X_train, X_test, y_train, y_test = train_test_split(
        X, work["K_Level"], test_size=0.2, random_state=42, stratify=work["K_Level"]
    )
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    return {
        "kmeans": kmeans,
        "scaler": scaler,
        "cluster_to_klevel": cluster_to_klevel,
        "rf_classifier": rf,
        "accuracy": accuracy,
        "classification_report": report,
        "labeled_df": work,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Classify K-Level from wearable activity data.")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    result = classify_k_level(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "k_level_model.pkl", "wb") as f:
        pickle.dump(
            {
                "kmeans": result["kmeans"],
                "scaler": result["scaler"],
                "cluster_to_klevel": result["cluster_to_klevel"],
                "rf_classifier": result["rf_classifier"],
            },
            f,
        )

    result["labeled_df"].to_csv(outdir / "k_level_labeled_activity.csv", index=False)

    with open(outdir / "k_level_summary.json", "w") as f:
        json.dump(
            {
                "cluster_to_klevel": result["cluster_to_klevel"],
                "rf_holdout_accuracy": result["accuracy"],
                "classification_report": result["classification_report"],
            },
            f,
            indent=2,
        )

    print(f"[k_level_classifier] RF holdout accuracy = {result['accuracy']:.3f}")
    print(f"[k_level_classifier] wrote k_level_model.pkl to {outdir}")


if __name__ == "__main__":
    _cli()

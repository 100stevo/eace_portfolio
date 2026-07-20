import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from feature_engineering import engineer_features  # noqa: E402
from outcomes_analysis import run_propensity_matching, run_mixed_effects_model  # noqa: E402
from predictive_model import train_rtd_model  # noqa: E402
from k_level_classifier import classify_k_level  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def step(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def main():
    # ---- Module 1 (pandas equivalent of the Polars loader) ----
    step("MODULE 1: Load & merge raw sources")
    military = pd.read_csv(DATA_DIR / "military_source.csv", parse_dates=["Injury_Date"])
    va = pd.read_csv(DATA_DIR / "va_source.csv")
    civilian = pd.read_csv(DATA_DIR / "civilian_source.csv")

    merged = va.merge(military, on="Patient_ID", how="left")
    merged = merged.merge(civilian, on="Patient_ID", how="left")
    merged = merged.rename(columns={"Days_Post_Injury": "Days_Since_Injury"})
    print(f"Merged: {merged.shape[0]} rows x {merged.shape[1]} cols")
    merged.to_csv(OUT_DIR / "merged_raw.csv", index=False)

    # ---- Module 2 ----
    step("MODULE 2: Feature engineering")
    features = engineer_features(merged)
    features["SMFA_Total"] = features[["SMFA_Mobility", "SMFA_Emotional"]].mean(axis=1)
    print(f"Features: {features.shape[0]} rows x {features.shape[1]} cols")
    print(features[["Cohort_Type_Flag", "Age_Group", "Injury_Severity_Index"]].head(3))
    features.to_csv(OUT_DIR / "features.csv", index=False)

    # ---- Module 3 note ----
    step("MODULE 3: EDA report (R Markdown)")
    print("src/eda_report.Rmd is ready to knit against outputs/features.csv")
    print("(requires R + tidyverse/table1/survminer -- not run in this Python sandbox)")

    # Table 1 (Python equivalent, since R isn't available here)
    baseline = features.sort_values("Days_Since_Injury").groupby("Patient_ID").first().reset_index()
    table1 = baseline.groupby("Cohort_Type")[["Age", "ISS_Score", "BMI"]].agg(["mean", "std"]).round(1)
    table1.to_html(OUT_DIR / "table1_demographics.html")
    print(table1)

    # ---- Module 4 ----
    step("MODULE 4: Propensity score matching + mixed-effects model")
    matched_df, psm_model = run_propensity_matching(baseline)
    matched_df.to_csv(OUT_DIR / "psm_matched_cohort.csv", index=False)
    mixed_results = run_mixed_effects_model(features, set(matched_df["Patient_ID"]))
    with open(OUT_DIR / "mixed_model_results.json", "w") as f:
        json.dump(mixed_results, f, indent=2)
    print(f"Engine: {mixed_results['engine']}")
    print(json.dumps(mixed_results["coefficients"], indent=2))

    # ---- Module 5 ----
    step("MODULE 5: RTD predictive model (XGBoost spec / GradientBoosting fallback here)")
    rtd_target = civilian.set_index("Patient_ID")["Return_to_Duty"]
    baseline_for_model = baseline.set_index("Patient_ID")
    baseline_for_model["Return_to_Duty"] = rtd_target
    baseline_for_model = baseline_for_model.reset_index()
    rtd_result = train_rtd_model(baseline_for_model, use_grid_search=True)
    print(f"Engine: {rtd_result['engine']}")
    print(f"CV AUC: {rtd_result['cv_auc']:.3f} | Holdout AUC: {rtd_result['holdout_auc']:.3f}")

    with open(OUT_DIR / "rtd_model.pkl", "wb") as f:
        pickle.dump({"model": rtd_result["model"], "feature_names": rtd_result["feature_names"]}, f)
    with open(OUT_DIR / "rtd_model_summary.json", "w") as f:
        json.dump(
            {
                "engine": rtd_result["engine"],
                "best_params": rtd_result["best_params"],
                "cv_auc": rtd_result["cv_auc"],
                "holdout_auc": rtd_result["holdout_auc"],
                "feature_importances": rtd_result.get("feature_importances_fallback"),
            },
            f,
            indent=2,
        )

    # ---- Module 6 ----
    step("MODULE 6: K-Level classification")
    k_result = classify_k_level(features)
    print(f"Cluster -> K-Level map: {k_result['cluster_to_klevel']}")
    print(f"RF holdout accuracy: {k_result['accuracy']:.3f}")
    k_result["labeled_df"].to_csv(OUT_DIR / "k_level_labeled_activity.csv", index=False)
    with open(OUT_DIR / "k_level_summary.json", "w") as f:
        json.dump(
            {
                "cluster_to_klevel": {str(k): v for k, v in k_result["cluster_to_klevel"].items()},
                "rf_holdout_accuracy": k_result["accuracy"],
            },
            f,
            indent=2,
        )
    with open(OUT_DIR / "k_level_model.pkl", "wb") as f:
        pickle.dump(
            {
                "kmeans": k_result["kmeans"],
                "scaler": k_result["scaler"],
                "cluster_to_klevel": k_result["cluster_to_klevel"],
                "rf_classifier": k_result["rf_classifier"],
            },
            f,
        )

    step("PIPELINE COMPLETE")
    print("All outputs written to /outputs")
    return {
        "features": features,
        "baseline": baseline,
        "matched_df": matched_df,
        "mixed_results": mixed_results,
        "rtd_result": rtd_result,
        "k_result": k_result,
    }


if __name__ == "__main__":
    main()

"""
Module 2: Feature Engineering Pipeline
----------------------------------------
Takes the merged raw dataset and derives the analytic feature set used by
Modules 3-6: Cohort_Type flag, Age_Group bins, Injury_Severity_Index,
Time_to_Rehab, and MICE-imputed SMFA scores.

Run standalone via:
    python feature_engineering.py --in ../outputs/merged_raw.parquet \
                                   --out ../outputs/features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

AGE_BINS = [18, 25, 35, 45, 200]
AGE_LABELS = ["18-25", "26-35", "36-45", "46+"]

SMFA_MISSINGNESS_CEILING = 0.20  # per TSD: do not impute beyond 20% missing


def flag_amputation_from_icd(icd_series: pd.Series) -> pd.Series:
    """Cohort_Type: 1 = Amputation (ICD-10-PCS codes starting with '84'), 0 = Salvage."""
    return icd_series.astype(str).str.startswith("84").astype(int)


def bin_age(age_series: pd.Series) -> pd.Series:
    return pd.cut(age_series, bins=AGE_BINS, labels=AGE_LABELS, right=True)


def compute_injury_severity_index(iss: pd.Series, surgeries_30d: pd.Series) -> pd.Series:
    """Composite score: z-scored ISS + z-scored surgical burden in first 30 days."""
    iss_z = (iss - iss.mean()) / iss.std(ddof=0)
    surg_z = (surgeries_30d - surgeries_30d.mean()) / surgeries_30d.std(ddof=0)
    return (0.7 * iss_z + 0.3 * surg_z).round(3)


def impute_smfa(df: pd.DataFrame, columns=("SMFA_Mobility", "SMFA_Emotional")) -> pd.DataFrame:
    """MICE-impute SMFA columns only where missingness is below the 20% ceiling."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        missing_frac = df[col].isna().mean()
        if missing_frac == 0:
            continue
        if missing_frac > SMFA_MISSINGNESS_CEILING:
            print(
                f"[feature_engineering] WARNING: {col} missingness "
                f"({missing_frac:.1%}) exceeds {SMFA_MISSINGNESS_CEILING:.0%} ceiling; "
                "skipping imputation, flagging instead."
            )
            df[f"{col}_high_missing_flag"] = df[col].isna().astype(int)
            continue

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        imputer = IterativeImputer(random_state=42, max_iter=15, sample_posterior=False)
        imputed_block = imputer.fit_transform(df[numeric_cols])
        df[numeric_cols] = imputed_block
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the full analytic feature set from the merged raw dataframe.

    Expected input columns (subset): ICD_10_PCS, Age, ISS_Score,
    Surgeries_First_30d, SMFA_Mobility, SMFA_Emotional, Time_to_Rehab.
    """
    df = df.copy()

    if "ICD_10_PCS" in df.columns:
        df["Cohort_Type_Flag"] = flag_amputation_from_icd(df["ICD_10_PCS"])
    elif "Cohort_Type" in df.columns:
        df["Cohort_Type_Flag"] = (df["Cohort_Type"] == "Amputation").astype(int)

    if "Age" in df.columns:
        df["Age_Group"] = bin_age(df["Age"])

    if {"ISS_Score", "Surgeries_First_30d"}.issubset(df.columns):
        df["Injury_Severity_Index"] = compute_injury_severity_index(
            df["ISS_Score"], df["Surgeries_First_30d"]
        )

    df = impute_smfa(df)

    return df


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Engineer analytic features.")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    in_path = Path(args.input_path)
    df = pd.read_parquet(in_path) if in_path.suffix == ".parquet" else pd.read_csv(in_path)

    features = engineer_features(df)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        features.to_parquet(out_path, index=False)
    else:
        features.to_csv(out_path, index=False)

    print(f"[feature_engineering] {features.shape[0]} rows x {features.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    _cli()

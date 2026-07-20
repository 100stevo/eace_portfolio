"""
Module 5 (Pillar B): Predictive Modeling for Return-to-Duty
---------------------------------------------------------------
Trains an XGBoost classifier (GridSearchCV-tuned, 10-fold StratifiedKFold CV)
to predict 24-month Return-to-Duty status, then computes SHAP values for
clinician-facing explainability.

Run standalone via:
    python predictive_model.py --in ../outputs/features.csv --outdir ../outputs
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier

    HAS_XGBOOST = False

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

FEATURE_COLUMNS = [
    "Age",
    "ISS_Score",
    "BMI",
    "Injury_Severity_Index",
    "Cohort_Type_Flag",
    "Pre_Existing_Diabetes",
    "Surgeries_First_30d",
    "SMFA_Mobility",
    "SMFA_Emotional",
    "PTSD_Checklist",
    "Time_to_Rehab",
]

PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
}


def _get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[cols].copy()
    for c in cols:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, cols


def train_rtd_model(df: pd.DataFrame, use_grid_search: bool = True) -> dict:
    """
    Train (and tune) a Return-to-Duty classifier.

    Parameters
    ----------
    df : dataframe containing FEATURE_COLUMNS and a `Return_to_Duty` target.
    use_grid_search : if False, trains a single default-parameter model
        (useful for fast local demos; the DoW pipeline should use True).

    Returns
    -------
    dict with keys: model, feature_names, cv_auc, holdout_auc,
    classification_report, shap_values (if shap installed), X_holdout.
    """
    work = df.dropna(subset=["Return_to_Duty"]).copy()
    X, feature_names = _get_feature_matrix(work)
    y = work["Return_to_Duty"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    if HAS_XGBOOST:
        base_model = XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        grid = {"xgb__" + k: v for k, v in PARAM_GRID.items()} if False else PARAM_GRID
    else:
        base_model = GradientBoostingClassifier(random_state=42)
        grid = {
            "n_estimators": PARAM_GRID["n_estimators"],
            "max_depth": PARAM_GRID["max_depth"],
            "learning_rate": PARAM_GRID["learning_rate"],
        }

    if use_grid_search:
        search = GridSearchCV(
            base_model, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        cv_auc = float(search.best_score_)
        best_params = search.best_params_
    else:
        model = base_model.fit(X_train, y_train)
        cv_auc = None
        best_params = {}

    holdout_proba = model.predict_proba(X_test)[:, 1]
    holdout_auc = float(roc_auc_score(y_test, holdout_proba))
    report = classification_report(
        y_test, (holdout_proba > 0.5).astype(int), output_dict=True, zero_division=0
    )

    result = {
        "model": model,
        "feature_names": feature_names,
        "engine": "xgboost.XGBClassifier" if HAS_XGBOOST else (
            "sklearn.GradientBoostingClassifier (DEMO FALLBACK - install xgboost "
            "for production model per TSD spec)"
        ),
        "best_params": best_params,
        "cv_auc": cv_auc,
        "holdout_auc": holdout_auc,
        "classification_report": report,
        "X_holdout": X_test,
        "y_holdout": y_test,
    }

    if HAS_SHAP:
        try:
            if HAS_XGBOOST:
                # Known shap/xgboost incompatibility: some XGBoost 2.x builds
                # serialize base_score as a JSON array string (e.g.
                # "[4.6333334E-1]") instead of a plain float, which older shap
                # parsers can't read (ValueError: could not convert string to
                # float). A save/reload round-trip through XGBoost's own JSON
                # format normalizes this. If it still fails, we fall back to
                # feature_importances_ rather than crashing the whole run --
                # `pip install --upgrade shap` or `pip install "xgboost<2"`
                # fixes it natively.
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_path = tmp.name
                model.save_model(tmp_path)
                model.load_model(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            result["shap_values"] = shap_values
        except Exception as e:
            print(
                f"[predictive_model] WARNING: SHAP TreeExplainer failed ({e}); "
                "falling back to model feature_importances_. This is a known "
                "shap/xgboost version-compatibility issue -- try "
                "`pip install --upgrade shap` or `pip install \"xgboost<2\"` "
                "to fix it natively.",
            )
            result["shap_values"] = None
            if hasattr(model, "feature_importances_"):
                result["feature_importances_fallback"] = dict(
                    zip(feature_names, model.feature_importances_.tolist())
                )
    else:
        # Fallback explainability: model feature_importances_ as a stand-in.
        # NOT a substitute for true SHAP values -- install `shap` in the DoW
        # environment (already in requirements.txt) to get per-prediction,
        # signed, additive attributions as specified in the TSD.
        result["shap_values"] = None
        result["feature_importances_fallback"] = dict(
            zip(feature_names, model.feature_importances_.tolist())
        )

    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train RTD prediction model.")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fast", action="store_true", help="skip GridSearchCV for a quick demo run")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    result = train_rtd_model(df, use_grid_search=not args.fast)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "rtd_model.pkl", "wb") as f:
        pickle.dump({"model": result["model"], "feature_names": result["feature_names"]}, f)

    summary = {
        "engine": result["engine"],
        "best_params": result["best_params"],
        "cv_auc": result["cv_auc"],
        "holdout_auc": result["holdout_auc"],
        "classification_report": result["classification_report"],
        "feature_importances_fallback": result.get("feature_importances_fallback"),
    }
    with open(outdir / "rtd_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[predictive_model] holdout AUC = {result['holdout_auc']:.3f}")
    print(f"[predictive_model] wrote rtd_model.pkl + rtd_model_summary.json to {outdir}")


if __name__ == "__main__":
    _cli()

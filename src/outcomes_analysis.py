"""
Module 4 (Pillar A): Comparative Outcomes Analysis
-----------------------------------------------------
Propensity Score Matching (Salvage vs. Amputation) followed by a Linear
Mixed-Effects Model on SMFA_Total over time, with Patient_ID as a random
intercept to account for repeated measures.

Run standalone via:
    python outcomes_analysis.py --in ../outputs/features.csv --outdir ../outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - falls back to OLS-with-cluster-SE demo
    HAS_STATSMODELS = False

COVARIATES = ["Age", "ISS_Score", "BMI", "Mechanism_of_Injury", "Pre_Existing_Diabetes"]


def run_propensity_matching(
    df: pd.DataFrame, caliper_sd: float = 0.2
) -> tuple[pd.DataFrame, LogisticRegression]:
    """
    1:1 nearest-neighbor propensity score matching between Cohort_Type groups.

    Parameters
    ----------
    df : baseline (one row per patient) dataframe containing Cohort_Type and COVARIATES.
    caliper_sd : maximum allowed distance in propensity-score SD units.

    Returns
    -------
    matched_df : the matched subset (treatment + matched control rows, with a
                 `match_id` column linking pairs).
    model : the fitted LogisticRegression propensity model (for diagnostics).
    """
    work = df.dropna(subset=COVARIATES + ["Cohort_Type"]).copy()
    base_covariates = [c for c in COVARIATES if c != "Mechanism_of_Injury"]
    mech_dummy_prefix = "Mechanism_of_Injury_"
    work = pd.get_dummies(work, columns=["Mechanism_of_Injury"], prefix="Mechanism_of_Injury", drop_first=True)
    mech_dummies = [c for c in work.columns if c.startswith(mech_dummy_prefix)]
    covariate_cols = base_covariates + mech_dummies

    X = work[covariate_cols].astype(float)
    y = (work["Cohort_Type"] == "Amputation").astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    work["propensity_score"] = model.predict_proba(X_scaled)[:, 1]

    caliper = caliper_sd * work["propensity_score"].std()

    treated = work[y == 1].reset_index(drop=True)
    control = work[y == 0].reset_index(drop=True)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity_score"]])
    distances, indices = nn.kneighbors(treated[["propensity_score"]])

    matched_pairs = []
    used_controls = set()
    for i, (dist, idx) in enumerate(zip(distances.ravel(), indices.ravel())):
        if dist <= caliper and idx not in used_controls:
            used_controls.add(idx)
            matched_pairs.append((treated.loc[i], control.loc[idx], i))

    if not matched_pairs:
        raise RuntimeError(
            "No matches found within caliper; consider widening caliper_sd "
            "or checking covariate overlap."
        )

    rows = []
    for match_id, (t_row, c_row, _) in enumerate(matched_pairs):
        t_row = t_row.copy()
        c_row = c_row.copy()
        t_row["match_id"] = match_id
        c_row["match_id"] = match_id
        rows.append(t_row)
        rows.append(c_row)

    matched_df = pd.DataFrame(rows).reset_index(drop=True)
    print(
        f"[outcomes_analysis] Matched {len(matched_pairs)} pairs "
        f"({len(matched_pairs)} amputation, {len(matched_pairs)} salvage) "
        f"within caliper={caliper:.4f}"
    )
    return matched_df, model


def run_mixed_effects_model(
    longitudinal_df: pd.DataFrame, matched_patient_ids: set
) -> dict:
    """
    Fit SMFA_Total ~ Cohort_Type + Days_Since_Injury, random intercept on Patient_ID,
    restricted to the matched cohort.
    """
    sub = longitudinal_df[longitudinal_df["Patient_ID"].isin(matched_patient_ids)].copy()
    sub = sub.dropna(subset=["SMFA_Total", "Cohort_Type", "Days_Since_Injury"])

    if HAS_STATSMODELS and len(sub) > 0:
        model = smf.mixedlm(
            "SMFA_Total ~ C(Cohort_Type) + Days_Since_Injury",
            data=sub,
            groups=sub["Patient_ID"],
        )
        result = model.fit(reml=True)
        return {
            "engine": "statsmodels.MixedLM",
            "coefficients": {k: float(v) for k, v in result.params.items()},
            "p_values": {k: float(v) for k, v in result.pvalues.items()},
            "conf_int_2.5%": {k: float(v) for k, v in result.conf_int()[0].items()},
            "conf_int_97.5%": {k: float(v) for k, v in result.conf_int()[1].items()},
            "n_obs": int(result.nobs),
            "n_groups": int(sub["Patient_ID"].nunique()),
        }

    # ---- Fallback demo path (statsmodels unavailable in this sandbox) ----
    # Cluster-robust OLS approximates the fixed-effects estimates; the random-
    # intercept variance component is not recovered by this fallback. This
    # path exists only so the pipeline can be demonstrated without network
    # access to install statsmodels -- in the DoW environment, statsmodels
    # is listed in requirements.txt and the MixedLM branch above will run.
    from sklearn.linear_model import LinearRegression

    sub["amputation_flag"] = (sub["Cohort_Type"] == "Amputation").astype(int)
    X = sub[["amputation_flag", "Days_Since_Injury"]].astype(float)
    yv = sub["SMFA_Total"].astype(float)
    ols = LinearRegression().fit(X, yv)

    # cluster-robust (by Patient_ID) standard errors via simple sandwich estimator
    resid = yv.values - ols.predict(X)
    Xd = np.column_stack([np.ones(len(X)), X.values])
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for pid, grp_idx in sub.groupby("Patient_ID").indices.items():
        Xg = Xd[grp_idx]
        ug = resid[grp_idx]
        score = Xg.T @ ug
        meat += np.outer(score, score)
    cluster_vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cluster_vcov))

    coef_names = ["Intercept", "amputation_flag", "Days_Since_Injury"]
    coefs = [ols.intercept_, ols.coef_[0], ols.coef_[1]]
    from scipy import stats as sstats

    dof = max(sub["Patient_ID"].nunique() - 1, 1)
    p_values = [
        2 * (1 - sstats.t.cdf(abs(c / s), dof)) if s > 0 else np.nan
        for c, s in zip(coefs, se)
    ]

    return {
        "engine": "sklearn.LinearRegression + cluster-robust SE (DEMO FALLBACK - "
        "install statsmodels for true MixedLM random-intercept model)",
        "coefficients": dict(zip(coef_names, [float(c) for c in coefs])),
        "std_errors": dict(zip(coef_names, [float(s) for s in se])),
        "p_values": dict(zip(coef_names, [float(p) for p in p_values])),
        "n_obs": int(len(sub)),
        "n_groups": int(sub["Patient_ID"].nunique()),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run PSM + mixed-effects outcomes analysis.")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    if "SMFA_Total" not in df.columns and {"SMFA_Mobility", "SMFA_Emotional"}.issubset(df.columns):
        df["SMFA_Total"] = df[["SMFA_Mobility", "SMFA_Emotional"]].mean(axis=1)

    baseline = df.sort_values("Days_Since_Injury").groupby("Patient_ID").first().reset_index()

    matched_df, _ = run_propensity_matching(baseline)
    matched_ids = set(matched_df["Patient_ID"])

    mixed_results = run_mixed_effects_model(df, matched_ids)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(outdir / "psm_matched_cohort.csv", index=False)
    with open(outdir / "mixed_model_results.json", "w") as f:
        json.dump(mixed_results, f, indent=2)

    print(f"[outcomes_analysis] wrote psm_matched_cohort.csv and mixed_model_results.json to {outdir}")


if __name__ == "__main__":
    _cli()

"""
Module 7: Interactive Dashboard & Reporting Engine
-----------------------------------------------------
Streamlit app for clinical decision support. Loads processed data and
pre-trained models from ../outputs and presents three tabs:
  1. Patient Demographics (Table 1)
  2. Comparative Outcomes (PSM + mixed-model trend)
  3. Prediction Tool (RTD probability + SHAP force plot)

Run via:
    streamlit run app.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

st.set_page_config(
    page_title="EACE Extremity Trauma Outcomes Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_features() -> pd.DataFrame:
    path = OUTPUTS_DIR / "features.csv"
    if not path.exists():
        st.error(f"Missing {path}. Run the Module 1-2 pipeline first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_rtd_model():
    path = OUTPUTS_DIR / "rtd_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_json(name: str):
    path = OUTPUTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    st.title("🦿 EACE Extremity Trauma Outcomes Analytics")
    st.caption("Fellowship ID: EACE-2026-0006 · Limb Salvage vs. Amputation Comparative Effectiveness")

    df = load_features()

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")
    age_groups = sorted(df["Age_Group"].dropna().unique().tolist()) if "Age_Group" in df else []
    selected_age = st.sidebar.multiselect("Age Group", age_groups, default=age_groups)

    mechanisms = sorted(df["Mechanism_of_Injury"].dropna().unique().tolist()) if "Mechanism_of_Injury" in df else []
    selected_mech = st.sidebar.multiselect("Injury Mechanism", mechanisms, default=mechanisms)

    cohorts = sorted(df["Cohort_Type"].dropna().unique().tolist()) if "Cohort_Type" in df else []
    selected_cohort = st.sidebar.multiselect("Cohort", cohorts, default=cohorts)

    filtered = df.copy()
    if selected_age:
        filtered = filtered[filtered["Age_Group"].isin(selected_age)]
    if selected_mech:
        filtered = filtered[filtered["Mechanism_of_Injury"].isin(selected_mech)]
    if selected_cohort:
        filtered = filtered[filtered["Cohort_Type"].isin(selected_cohort)]

    tab1, tab2, tab3 = st.tabs(
        ["📋 Patient Demographics", "📊 Comparative Outcomes", "🎯 Prediction Tool"]
    )

    # ---------------- TAB 1: Demographics ----------------
    with tab1:
        st.subheader("Table 1 — Baseline Demographics")
        baseline = filtered.sort_values("Days_Since_Injury").groupby("Patient_ID").first().reset_index() \
            if "Days_Since_Injury" in filtered.columns else filtered.drop_duplicates("Patient_ID")

        summary_cols = [c for c in ["Age", "ISS_Score", "BMI", "Cohort_Type",
                                     "Mechanism_of_Injury", "Pre_Existing_Diabetes"] if c in baseline.columns]
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(
                baseline.groupby("Cohort_Type")[[c for c in summary_cols if c != "Cohort_Type"]]
                .describe().T if "Cohort_Type" in baseline else baseline[summary_cols].describe().T,
                use_container_width=True,
            )
        with c2:
            if "Cohort_Type" in baseline.columns:
                fig = px.pie(baseline, names="Cohort_Type", title="Cohort Distribution")
                st.plotly_chart(fig, use_container_width=True)

        if "Mechanism_of_Injury" in baseline.columns:
            fig2 = px.histogram(
                baseline, x="Mechanism_of_Injury", color="Cohort_Type", barmode="group",
                title="Injury Mechanism by Cohort",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TAB 2: Comparative Outcomes ----------------
    with tab2:
        st.subheader("Propensity-Matched Functional Outcomes")

        matched_path = OUTPUTS_DIR / "psm_matched_cohort.csv"
        mixed_results = load_json("mixed_model_results.json")

        if matched_path.exists():
            matched = pd.read_csv(matched_path)
            st.markdown(f"**Matched sample:** {matched['match_id'].nunique()} pairs")
            if "propensity_score" in matched.columns:
                fig3 = px.histogram(
                    matched, x="propensity_score", color="Cohort_Type", barmode="overlay",
                    opacity=0.6, title="Propensity Score Overlap After Matching",
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Run `outcomes_analysis.py` to generate the matched cohort.")

        if "SMFA_Mobility" in filtered.columns and "Days_Since_Injury" in filtered.columns:
            trend = (
                filtered.groupby(["Cohort_Type", "Days_Since_Injury"])["SMFA_Mobility"]
                .mean().reset_index()
            )
            fig4 = px.line(
                trend, x="Days_Since_Injury", y="SMFA_Mobility", color="Cohort_Type",
                markers=True, title="SMFA Mobility Trend Over Time (Mixed-Model Input)",
            )
            st.plotly_chart(fig4, use_container_width=True)

        if mixed_results:
            st.markdown(f"**Model engine:** `{mixed_results.get('engine', 'n/a')}`")
            coef_df = pd.DataFrame(
                {
                    "coefficient": mixed_results.get("coefficients", {}),
                    "p_value": mixed_results.get("p_values", {}),
                }
            )
            st.dataframe(coef_df, use_container_width=True)
        else:
            st.info("Run `outcomes_analysis.py` to generate mixed-model results.")

    # ---------------- TAB 3: Prediction Tool ----------------
    with tab3:
        st.subheader("Return-to-Duty Prediction Tool")
        bundle = load_rtd_model()
        if bundle is None:
            st.info("Run `predictive_model.py` to train and save the RTD model.")
        else:
            model = bundle["model"]
            feature_names = bundle["feature_names"]

            st.markdown("Enter hypothetical patient features:")
            input_vals = {}
            cols = st.columns(3)
            defaults = filtered[feature_names].median(numeric_only=True) if set(feature_names).issubset(filtered.columns) else {}
            for i, feat in enumerate(feature_names):
                with cols[i % 3]:
                    default_val = float(defaults.get(feat, 0.0)) if hasattr(defaults, "get") else 0.0
                    input_vals[feat] = st.number_input(feat, value=round(default_val, 2))

            if st.button("Predict Return-to-Duty Probability", type="primary"):
                X_input = pd.DataFrame([input_vals])[feature_names]
                proba = model.predict_proba(X_input)[0, 1]
                st.metric("Predicted RTD Probability", f"{proba:.1%}")

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        title={"text": "Return-to-Duty Probability (%)"},
                        gauge={"axis": {"range": [0, 100]},
                               "bar": {"color": "#2c7fb8"},
                               "steps": [
                                   {"range": [0, 33], "color": "#fde0dd"},
                                   {"range": [33, 66], "color": "#fdbb84"},
                                   {"range": [66, 100], "color": "#c7e9c0"},
                               ]},
                    )
                )
                st.plotly_chart(gauge, use_container_width=True)

                try:
                    import shap

                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_input)
                    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
                    force_df = pd.DataFrame(
                        {"feature": feature_names, "shap_value": sv}
                    ).sort_values("shap_value")
                    fig5 = px.bar(
                        force_df, x="shap_value", y="feature", orientation="h",
                        color="shap_value", color_continuous_scale="RdBu",
                        title="Feature Contributions (SHAP) to This Prediction",
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                except ImportError:
                    if hasattr(model, "feature_importances_"):
                        fi_df = pd.DataFrame(
                            {"feature": feature_names, "importance": model.feature_importances_}
                        ).sort_values("importance")
                        fig5 = px.bar(
                            fi_df, x="importance", y="feature", orientation="h",
                            title="Global Feature Importance (SHAP unavailable — install `shap`)",
                        )
                        st.plotly_chart(fig5, use_container_width=True)


if __name__ == "__main__":
    main()

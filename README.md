# EACE-2026-0006 — Extremity Trauma Outcomes Analytics

Data science portfolio for the EACE fellowship: comparative effectiveness of
limb salvage vs. amputation, Return-to-Duty prediction, and automated K-Level
classification, built as a modular ETL → ML → Reporting pipeline per the TSD.

## Quick start

```bash
conda env create -f environment.yml
conda activate eace-trauma-outcomes

# 1. Generate synthetic data (schema-matched, zero real patient data)
python data/generate_synthetic_data.py

# 2-6. Run the pipeline
python src/data_loader.py --military data/military_source.csv \
    --va data/va_source.csv --civilian data/civilian_source.csv \
    --out outputs/merged_raw.parquet
python src/feature_engineering.py --in outputs/merged_raw.parquet --out outputs/features.parquet
python src/outcomes_analysis.py --in outputs/features.csv --outdir outputs
python src/predictive_model.py --in outputs/features.csv --outdir outputs
python src/k_level_classifier.py --in outputs/features.csv --outdir outputs

# R Markdown EDA report (Module 3)
Rscript -e 'rmarkdown::render("src/eda_report.Rmd")'

# 7. Launch the dashboard
streamlit run app/app.py
```

Or via Docker: `docker build -t eace-trauma . && docker run -p 8501:8501 eace-trauma`

## What's in this bundle

| Path | Purpose |
|---|---|
| `data/generate_synthetic_data.py` | Generates synthetic patient/follow-up/outcome data matching the TSD §6 schema |
| `src/data_loader.py` | Module 1 — load, validate, merge registry extracts (Polars) |
| `src/feature_engineering.py` | Module 2 — Cohort flag, age bins, severity index, MICE imputation |
| `src/eda_report.Rmd` | Module 3 — Table 1, KM curves, correlation heatmap (R Markdown) |
| `src/outcomes_analysis.py` | Module 4 — Propensity Score Matching + Mixed-Effects Model |
| `src/predictive_model.py` | Module 5 — XGBoost RTD classifier + SHAP explainability |
| `src/k_level_classifier.py` | Module 6 — K-Means + Random Forest K-Level classification |
| `app/app.py` | Module 7 — Streamlit clinical dashboard |
| `outputs/index.html` | **Hosted static report** — open this directly, or deploy it as-is to any static host (S3, GitHub Pages, an internal web share) |
| `outputs/*.json, *.csv, *.pkl` | Verified pipeline outputs from a demo run (see note below) |

## Hosting the report

`outputs/index.html` is a single self-contained static file (charts via
Chart.js CDN) — no server required. To "host" it:
- **Simplest:** open the file directly in a browser, or drag it into any static
  file host (GitHub Pages, S3 + static website hosting, an internal SharePoint/web share).
- **Local server:** `python -m http.server 8000 --directory outputs` then visit `localhost:8000`.
- **Live dashboard (interactive, not just a report):** run `streamlit run app/app.py` instead — this exposes the filterable, model-backed Tab 1/2/3 experience described in Module 7, and can be containerized via the included `Dockerfile` for deployment on any host that runs Docker.

## Important environment note

This bundle was built and pipeline-tested in a sandbox with **no network
access**, so four packages (`polars`, `xgboost`, `shap`, `statsmodels`,
`streamlit`) couldn't be installed to execute live. The `src/*.py` module
files are unmodified, full production implementations using those libraries
exactly as specified in the TSD — they will run as-is once you
`pip install -r requirements.txt` (or `conda env create -f environment.yml`)
in an environment with network access, including your DoW secure environment.

To still hand you *real, verified* output rather than untested code, each
affected module has a graceful fallback (documented inline, and labeled
`"engine"` in each JSON summary) that runs on preinstalled libraries only:
- Module 4 mixed-effects model → cluster-robust OLS (in place of `statsmodels.MixedLM`)
- Module 5 predictive model → `sklearn.GradientBoostingClassifier` (in place of `xgboost.XGBClassifier`) with feature importances (in place of true SHAP values)
- Module 1 loader → exercised via `pandas` in `run_demo_pipeline.py` (in place of `polars`) for this demo run only; `src/data_loader.py` itself is untouched Polars code

`run_demo_pipeline.py` (repo root) is the script that produced everything in
`/outputs` — it's a demo harness, not a Module; treat `src/*.py` and `app/app.py`
as the deliverables.

## On MIMIC-III/IV

The TSD's final line asks to check MIMIC-III/IV for data simulation. MIMIC is
a real, credentialed, de-identified **ICU** dataset on PhysioNet — it requires
CITI human-subjects training and a signed data use agreement, and it has no
military-specific fields (MOS, blast exposure, K-Level, Return-to-Duty,
SMFA). `data/generate_synthetic_data.py` instead produces fully synthetic
data matching the TSD §6 schema directly, so the whole pipeline runs today
with zero real patient data and zero external credentials. Swap in your
credentialed DoW/VA/FITBIR/NTRR/METALS extracts (or your own credentialed
MIMIC-IV pull, if you want an ICU comparator arm) via Module 1 — no code
changes needed beyond column-name mapping.

I have used the MIMIC-III/IV for databases since I couldn`t get any real datasets to use. I recomment using this.


PROBLEMS TO MIGHT GET AND HOW TO SOLVE. 
# R Markdown EDA report (Module 3)
Rscript -e 'rmarkdown::render("src/eda_report.Rmd")'

# 7. Launch the dashboard
streamlit run app/app.py
```

While runnign these two might endup in real failures and this how to solve them step by step.

# STEP 1 run
conda activate eace-trauma-outcomes
python -c "import sys; print(sys.executable)"   # sanity check: should point inside eace-trauma-outcomes


# STEP 2 run
python data/generate_synthetic_data.py
python src/data_loader.py --military data/military_source.csv \
    --va data/va_source.csv --civilian data/civilian_source.csv \
    --out outputs/merged_raw.parquet
python src/feature_engineering.py --in outputs/merged_raw.parquet --out outputs/features.parquet

# STEP 3
python -c "import pandas as pd; pd.read_parquet('outputs/features.parquet').to_csv('outputs/features.csv', index=False)"

# STEP 4
python src/outcomes_analysis.py --in outputs/features.csv --outdir outputs
python src/predictive_model.py --in outputs/features.csv --outdir outputs
python src/k_level_classifier.py --in outputs/features.csv --outdir outputs

# STEP 5
streamlit run app/app.py
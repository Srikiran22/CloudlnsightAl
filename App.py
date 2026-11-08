import numpy as np
import pandas as pd
import streamlit as st

from Utils.dataset_ui import init_session_state, invalidate_dataset_cache, render_sidebar, set_active_dataset
from Utils.logsys import get_logger
from Utils.paths import DATASETS_DIR, atomic_write
from Utils.theme import apply_theme, brand_logo_svg, section_card_header

logger = get_logger("App")

st.set_page_config(
    page_title="CloudInsight | Data Science Studio",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()
st.logo(brand_logo_svg(), size="large")
init_session_state()


def _generate_synthetic_benchmarks():
    """Generate high-quality domain datasets for instant analytical exploration."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. SaaS Churn Pair (Q1 baseline & Q2 drifted)
    n_saas = 600
    plan_tier_q1 = rng.choice(["Starter", "Pro", "Enterprise"], size=n_saas, p=[0.5, 0.35, 0.15])
    spend_q1 = np.round(rng.normal(120.0, 35.0, size=n_saas).clip(20.0, 500.0), 2)
    tickets_q1 = rng.poisson(2.0, size=n_saas)
    usage_q1 = np.round(rng.normal(45.0, 15.0, size=n_saas).clip(1.0, 160.0), 1)
    logit_q1 = -1.0 + 1.2 * tickets_q1 - 0.08 * usage_q1
    prob_q1 = 1 / (1 + np.exp(-logit_q1))
    churned_q1 = (prob_q1 > 0.5).astype(int)

    q1_df = pd.DataFrame({
        "customer_id": [f"CUST-{i:04d}" for i in range(1, n_saas + 1)],
        "plan_tier": plan_tier_q1,
        "monthly_spend_usd": spend_q1,
        "support_tickets": tickets_q1,
        "usage_hours": usage_q1,
        "churned": churned_q1,
    })
    q1_df.loc[rng.choice(n_saas, 18, replace=False), "usage_hours"] = np.nan
    q1_df = pd.concat([q1_df, q1_df.iloc[:5]], ignore_index=True)

    plan_tier_q2 = rng.choice(["Starter", "Pro", "Enterprise"], size=n_saas, p=[0.25, 0.45, 0.30])
    spend_q2 = np.round(rng.normal(165.0, 50.0, size=n_saas).clip(20.0, 600.0), 2)
    tickets_q2 = rng.poisson(3.2, size=n_saas)
    usage_q2 = np.round(rng.normal(38.0, 18.0, size=n_saas).clip(1.0, 160.0), 1)
    logit_q2 = -0.5 + 1.1 * tickets_q2 - 0.07 * usage_q2
    prob_q2 = 1 / (1 + np.exp(-logit_q2))
    churned_q2 = (prob_q2 > 0.5).astype(int)

    q2_df = pd.DataFrame({
        "customer_id": [f"CUST-{i:04d}" for i in range(1001, 1001 + n_saas)],
        "plan_tier": plan_tier_q2,
        "monthly_spend_usd": spend_q2,
        "support_tickets": tickets_q2,
        "usage_hours": usage_q2,
        "churned": churned_q2,
    })

    # 2. Housing Market Benchmark (Regression)
    n_house = 500
    sqft = rng.normal(1850, 450, n_house).clip(600, 4500)
    beds = rng.choice([1, 2, 3, 4, 5], size=n_house, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    baths = np.round(beds * 0.75 + rng.normal(0, 0.3, n_house)).clip(1, 4)
    age = rng.integers(1, 55, size=n_house)
    price = np.round(85000 + (sqft * 195) + (beds * 18000) + (baths * 24000) - (age * 1200) + rng.normal(0, 28000, n_house), -2)
    house_df = pd.DataFrame({
        "property_id": [f"PROP-{i:04d}" for i in range(1, n_house + 1)],
        "square_feet": np.round(sqft, 0),
        "bedrooms": beds,
        "bathrooms": baths,
        "property_age_years": age,
        "neighborhood_grade": rng.choice(["A", "B", "C", "D"], size=n_house, p=[0.2, 0.45, 0.25, 0.1]),
        "sale_price_usd": price,
    })

    # Save to disk
    datasets = [
        ("saas_customers_q1.csv", q1_df),
        ("saas_customers_q2_drifted.csv", q2_df),
        ("housing_market_prices.csv", house_df),
    ]
    for fname, frame in datasets:
        dest = DATASETS_DIR / fname
        atomic_write(dest, frame.to_csv(index=False), mode="w", encoding="utf-8")
        invalidate_dataset_cache(fname)

    set_active_dataset(q1_df, "saas_customers_q1.csv")


def render_home() -> None:
    render_sidebar()
    if st.session_state.pop("sample_loaded_notification", False):
        st.success("Benchmark datasets loaded into workspace catalog: saas_customers_q1.csv, saas_customers_q2_drifted.csv, housing_market_prices.csv.")

    # Workspace Header Card
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
              <h1 style="margin: 0; font-size: 1.65rem; font-weight: 750; letter-spacing: -0.02em;">
                CloudInsight <span style="color: var(--ci-accent);">Data Science Studio</span>
              </h1>
              <span style="font-size: 0.78rem; font-weight: 650; padding: 4px 10px; border-radius: 999px; background: var(--ci-tint); border: 1px solid var(--ci-border); color: var(--ci-accent);">
                PRODUCTION WORKBENCH v2.2
              </span>
            </div>
            <p style="margin: 0.2rem 0 0 0; color: var(--ci-text-2); font-size: 0.95rem; line-height: 1.55;">
              A local-first analytics & machine learning workbench for rigorous data profiling, statistical hypothesis testing, automated feature engineering, leak-free Scikit-Learn model pipelines, and audit reporting.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # 3 Core Data Science Pillars
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            section_card_header("Data Ingestion & Integrity", "Multi-format local tabular pipelines")
            st.markdown(
                '<p style="font-size: 0.88rem; color: var(--ci-text-2); line-height: 1.55; margin: 0;">'
                'Zero-cloud local ingestion for CSV, Parquet, Excel, JSON, TSV, and SQLite with automated schema inference and memory bounding.'
                '</p>',
                unsafe_allow_html=True
            )
    with c2:
        with st.container(border=True):
            section_card_header("Statistical Diagnostics & EDA", "Bivariate correlation & hypothesis testing")
            st.markdown(
                '<p style="font-size: 0.88rem; color: var(--ci-text-2); line-height: 1.55; margin: 0;">'
                'Pearson/Spearman correlation matrices, D’Agostino-Pearson normality tests, VIF multicollinearity audits, and Tukey IQR outlier detection.'
                '</p>',
                unsafe_allow_html=True
            )
    with c3:
        with st.container(border=True):
            section_card_header("Scikit-Learn Model Studio", "Predictive modeling & validation pipelines")
            st.markdown(
                '<p style="font-size: 0.88rem; color: var(--ci-text-2); line-height: 1.55; margin: 0;">'
                'Supervised classification & regression, 5-fold cross-validation, feature importance explainability, and HMAC-signed artifacts.'
                '</p>',
                unsafe_allow_html=True
            )

    left, right = st.columns([5, 4])
    with left:
        with st.container(border=True):
            section_card_header("End-to-End Data Science Lifecycle", "Systematic 6-stage analytical pipeline")
            st.markdown(
                """
                1. **Data Hub & Ingest** — Ingest tabular files (`CSV`, `Parquet`, `Excel`, `JSON`, `SQLite`) with instant column typing and null profiling.
                2. **Data Profiling & Hygiene** — Evaluate completeness, uniqueness, duplicate patterns, and composite Data Hygiene Index.
                3. **Data Cleaning & Imputation** — Statistical imputation (mean/median/constant/complete-case), outlier screening (Tukey IQR, Z-score bounds), and sanitized catalog export.
                4. **Feature Engineering Studio** — Standard/MinMax/Robust scalers, categorical encodings, log transforms, and Mutual Information feature selection.
                5. **Exploratory Analysis & Visuals** — Statistical tests, correlation matrices with significance filtering, and publication-ready Plotly charts.
                6. **Machine Learning & Inference** — Train, validate, and compare classification or regression algorithms with leak-free cross-validation, batch prediction, and executive PDF reports.
                """
            )
    with right:
        with st.container(border=True):
            section_card_header("Quick Start Benchmarks", "Instant domain datasets for exploration")
            st.markdown(
                "Load production-grade benchmark datasets to evaluate modeling, data cleaning, and statistical drift:"
            )

            if st.button(
                "Load sample datasets (SaaS Churn, Drift & Housing)",
                type="primary",
                use_container_width=True,
                help="Load curated test datasets into workspace. Existing files are preserved.",
            ):
                _generate_synthetic_benchmarks()
                st.session_state["sample_loaded_notification"] = True
                st.rerun()

            st.caption("Deterministic synthetic benchmarks calibrated for baseline evaluation, covariate drift, and continuous regression.")
            st.markdown(
                """
                - `saas_customers_q1.csv` — 605 customer records with churn labels, missingness and duplicates
                - `saas_customers_q2_drifted.csv` — Statistically shifted distributions for drift detection
                - `housing_market_prices.csv` — 500 residential records for continuous target regression modeling
                """
            )


pg = st.navigation(
    {
        "Overview": [
            st.Page(render_home, title="Workspace Home", icon=":material/home:", default=True),
            st.Page("Pages/Dashboard.py", title="Data Health & Profiler", icon=":material/speed:"),
        ],
        "Data Management": [
            st.Page("Pages/Upload.py", title="Data Hub & Ingest", icon=":material/folder_open:"),
            st.Page("Pages/Cleaning.py", title="Data Cleaning & Imputation", icon=":material/healing:"),
            st.Page("Pages/FeatureEngineering.py", title="Feature Engineering Studio", icon=":material/tune:"),
        ],
        "Exploratory Analysis": [
            st.Page("Pages/EDA.py", title="Statistical Diagnostics", icon=":material/analytics:"),
            st.Page("Pages/Visualization.py", title="Interactive Visual Analytics", icon=":material/bar_chart:"),
        ],
        "Machine Learning": [
            st.Page("Pages/ML.py", title="Model Studio & Training", icon=":material/precision_manufacturing:"),
            st.Page("Pages/Predict.py", title="Batch Scoring & Simulation", icon=":material/batch_prediction:"),
        ],
        "AI & Gemini Assistant": [
            st.Page("Pages/AIConverter.py", title="AI File Converter", icon=":material/transform:"),
            st.Page("Pages/AI.py", title="AI Summary & Insights", icon=":material/psychology:"),
            st.Page("Pages/AIChat.py", title="Ask AI (Chat)", icon=":material/forum:"),
        ],
        "Audit & Reports": [
            st.Page("Pages/Compare.py", title="Dataset Drift & Compare", icon=":material/compare_arrows:"),
            st.Page("Pages/Report.py", title="Data Science Audit Report", icon=":material/description:"),
        ],
    },
    expanded=True
)
pg.run()

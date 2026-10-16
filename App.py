import streamlit as st

from Utils.dataset_ui import init_session_state, render_sidebar
from Utils.Gemini import DEFAULT_GEMINI_MODEL
from Utils.theme import apply_theme, page_header, section_card_header

st.set_page_config(
    page_title="CloudInsight AI",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()
init_session_state()


def render_home():
    render_sidebar()
    if st.session_state.pop("sample_loaded_notification", False):
        st.success("Loaded `saas_customers_q1.csv` and `saas_customers_q2_drifted.csv`! Explore Cleaning, Compare, Dashboard, or ML in the sidebar.")

    with st.container(border=True):
        st.markdown(
            """
            <div style="margin-bottom: 0.25rem;">
              <h1 style="margin: 0; font-size: 1.55rem; font-weight: 750;">CloudInsight AI</h1>
            </div>
            <p style="margin: 0.25rem 0 0 0; color: var(--ci-text-2); font-size: 0.95rem;">
              A local-first analytics and machine learning workspace for ingesting, preparing, and modeling tabular datasets — with Gemini-assisted unstructured document extraction.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # 3 Architecture Pillars
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            section_card_header("Cloud & Local Storage", "Connect S3 buckets or local files")
            st.markdown(
                '<div style="font-size: 1.2rem; font-weight: 700; color: var(--ci-text); margin: 0.2rem 0 0.15rem 0;">Amazon S3 & Local</div>'
                '<div style="font-size: 0.83rem; color: var(--ci-muted); line-height: 1.4;">Streaming transfers, ambient IAM, session tokens, and bucket exports.</div>',
                unsafe_allow_html=True
            )
    with c2:
        with st.container(border=True):
            section_card_header("AI Intelligence Layer", f"Google Gemini ({DEFAULT_GEMINI_MODEL})")
            st.markdown(
                f'<div style="font-size: 1.2rem; font-weight: 700; color: var(--ci-text); margin: 0.2rem 0 0.15rem 0;">Gemini 3.8 Flash</div>'
                '<div style="font-size: 0.83rem; color: var(--ci-muted); line-height: 1.4;">Executive reports, interactive chat, and unstructured document extraction.</div>',
                unsafe_allow_html=True
            )
    with c3:
        with st.container(border=True):
            section_card_header("Predictive Modeling", "Scikit-Learn AutoML Pipelines")
            st.markdown(
                '<div style="font-size: 1.2rem; font-weight: 700; color: var(--ci-text); margin: 0.2rem 0 0.15rem 0;">Scikit-Learn</div>'
                '<div style="font-size: 0.83rem; color: var(--ci-muted); line-height: 1.4;">Classification, regression, 5-fold CV, and HMAC-signed models.</div>',
                unsafe_allow_html=True
            )

    left, right = st.columns([5, 4])
    with left:
        with st.container(border=True):
            section_card_header("Analytical Workflow", "5-step end-to-end data science lifecycle")
            st.markdown(
                """
                1. **Ingest** — Upload local tabular files (`CSV`, `Parquet`, `Excel`, `JSON`, `XML`, `PDF`) or stream directly from Amazon S3.
                2. **Prepare** — Remove duplicates, execute statistical imputation, and screen for schema and distribution drift.
                3. **Explore** — Diagnostic statistics, variance-capped correlations, Tukey IQR outliers, and peak-preserving Plotly charts.
                4. **Model** — Train classification or regression models with leak-free pipelines and cryptographically sign artifacts with HMAC-SHA256.
                5. **Share** — Compile an executive PDF analytics report with CJK Unicode typography and Gemini insights.
                """
            )
    with right:
        with st.container(border=True):
            section_card_header("Quick Start & Demo", "Instant synthetic exploration workspace")
            st.markdown(
                """
                Load sample datasets to explore customer churn, data hygiene scoring, and statistical distribution drift:
                """
            )

            if st.button("Load sample datasets (Q1 vs Q2 Drift & Churn)", type="primary", use_container_width=True):
                import numpy as np
                import pandas as pd
                from Utils.paths import DATASETS_DIR, atomic_write
                from Utils.dataset_ui import invalidate_dataset_cache, set_active_dataset

                rng = np.random.default_rng(42)
                n = 600
                q1_df = pd.DataFrame({
                    "customer_id": [f"CUST-{i:04d}" for i in range(1, n + 1)],
                    "plan_tier": rng.choice(["Starter", "Pro", "Enterprise"], size=n, p=[0.5, 0.35, 0.15]),
                    "monthly_spend_usd": np.round(rng.normal(120.0, 35.0, size=n).clip(20.0, 500.0), 2),
                    "support_tickets": rng.poisson(1.8, size=n),
                    "usage_hours": np.round(rng.normal(45.0, 15.0, size=n).clip(1.0, 160.0), 1),
                    "churned": rng.choice([0, 1], size=n, p=[0.82, 0.18]),
                })
                # Introduce missing values and duplicates so Cleaning & Dashboard Hygiene Index are interesting
                q1_df.loc[rng.choice(n, 18, replace=False), "usage_hours"] = np.nan
                q1_df = pd.concat([q1_df, q1_df.iloc[:5]], ignore_index=True)

                # Q2 dataset with deliberate statistical distribution drift (KS & TVD triggers)
                q2_df = pd.DataFrame({
                    "customer_id": [f"CUST-{i:04d}" for i in range(1001, 1001 + n)],
                    "plan_tier": rng.choice(["Starter", "Pro", "Enterprise"], size=n, p=[0.25, 0.45, 0.30]),
                    "monthly_spend_usd": np.round(rng.normal(165.0, 50.0, size=n).clip(20.0, 600.0), 2),
                    "support_tickets": rng.poisson(3.2, size=n),
                    "usage_hours": np.round(rng.normal(38.0, 18.0, size=n).clip(1.0, 160.0), 1),
                    "churned": rng.choice([0, 1], size=n, p=[0.70, 0.30]),
                })

                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                for fname, frame in [("saas_customers_q1.csv", q1_df), ("saas_customers_q2_drifted.csv", q2_df)]:
                    atomic_write(DATASETS_DIR / fname, frame.to_csv(index=False), mode="w", encoding="utf-8")
                    invalidate_dataset_cache(fname)

                set_active_dataset(q1_df, "saas_customers_q1.csv")
                st.session_state["sample_loaded_notification"] = True
                st.rerun()


pg = st.navigation(
    {
        "Start": [
            st.Page(render_home, title="Home", icon=":material/home:", default=True),
            st.Page("Pages/Dashboard.py", title="Data Overview", icon=":material/dashboard:"),
        ],
        "Data Sources": [
            st.Page("Pages/Upload.py", title="Import Files", icon=":material/upload_file:"),
            st.Page("Pages/CloudS3.py", title="Cloud Storage (S3)", icon=":material/cloud:"),
            st.Page("Pages/AIConverter.py", title="AI File Converter", icon=":material/auto_fix_high:"),
        ],
        "Prepare": [
            st.Page("Pages/Cleaning.py", title="Clean Data", icon=":material/cleaning_services:"),
            st.Page("Pages/Compare.py", title="Compare Datasets", icon=":material/compare_arrows:"),
        ],
        "Analyze & Model": [
            st.Page("Pages/EDA.py", title="Explore Columns", icon=":material/table_chart:"),
            st.Page("Pages/Visualization.py", title="Charts", icon=":material/bar_chart:"),
            st.Page("Pages/ML.py", title="Train Model", icon=":material/model_training:"),
            st.Page("Pages/Predict.py", title="Predict & Score", icon=":material/batch_prediction:"),
        ],
        "AI & Reports": [
            st.Page("Pages/AI.py", title="AI Summary", icon=":material/summarize:"),
            st.Page("Pages/AIChat.py", title="Ask AI (Chat)", icon=":material/forum:"),
            st.Page("Pages/Report.py", title="PDF Report", icon=":material/picture_as_pdf:"),
        ],
    },
    expanded=True
)
pg.run()


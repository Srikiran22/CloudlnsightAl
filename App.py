import streamlit as st

from Utils.dataset_ui import init_session_state, render_sidebar
from Utils.Gemini import DEFAULT_GEMINI_MODEL
from Utils.theme import apply_theme

st.set_page_config(
    page_title="CloudInsight AI",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()
init_session_state()


def render_home():
    st.title("CloudInsight AI")
    st.markdown("An analytics workspace for ingesting, understanding, and modeling tabular data — with Gemini-assisted conversion of unstructured files.")
    render_sidebar()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Storage", "Amazon S3")
    with c2:
        st.metric("AI", f"Gemini ({DEFAULT_GEMINI_MODEL})")
    with c3:
        st.metric("Modeling", "Scikit-Learn")

    st.markdown("---")

    left, right = st.columns([5, 4])
    with left:
        st.markdown(
            """
            #### How it works

            1. **Ingest** — upload a file or connect to S3. Structured formats parse natively; text and PDFs are structured by Gemini.
            2. **Prepare** — remove duplicates, impute or drop missing values, then compare datasets for schema and value drift.
            3. **Explore** — statistics, correlations, outlier detection, and interactive charts.
            4. **Model** — train classification or regression models, save them, and score new data.
            5. **Share** — generate a detailed PDF report, optionally including an AI-written executive summary.
            """
        )
    with right:
        st.markdown(
            """
            #### Getting started

            Load a dataset on the **Ingest data** page — everything else keys off
            the active dataset shown in the sidebar.

            A Gemini API key is only needed for AI conversion, insights,
            and chat; it is entered at runtime and cleared from session state after
            each task unless you choose to keep it for the session.
            """
        )

        if st.button("Load sample datasets (Q1 vs Q2 Drift & Churn)", type="primary"):
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
            st.success("Loaded `saas_customers_q1.csv` and `saas_customers_q2_drifted.csv`! Explore Cleaning, Compare, Dashboard, or ML in the sidebar.")


pg = st.navigation(
    {
        "Home": [
            st.Page(render_home, title="Home", icon=":material/home:", default=True),
        ],
        "Workspace": [
            st.Page("Pages/Upload.py", title="Ingest data", icon=":material/upload:", default=False),
            st.Page("Pages/Cleaning.py", title="Cleaning", icon=":material/cleaning_services:"),
            st.Page("Pages/Compare.py", title="Compare", icon=":material/compare_arrows:"),
            st.Page("Pages/EDA.py", title="EDA", icon=":material/query_stats:"),
            st.Page("Pages/Visualization.py", title="Visualize", icon=":material/bar_chart:"),
            st.Page("Pages/Dashboard.py", title="Dashboard", icon=":material/dashboard:"),
            st.Page("Pages/ML.py", title="Machine learning", icon=":material/model_training:"),
            st.Page("Pages/AI.py", title="AI insights", icon=":material/smart_toy:"),
            st.Page("Pages/Report.py", title="PDF report", icon=":material/description:"),
        ],
    }
)
pg.run()

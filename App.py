import streamlit as st

from Utils.dataset_ui import init_session_state, render_sidebar
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
        st.metric("AI", "Gemini 1.5 / 2.0")
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

            Load a dataset on the **Upload** page — everything else keys off
            the active dataset shown in the sidebar.

            A Gemini API key is only needed for AI conversion, insights,
            and chat; it is entered at runtime and wiped from memory after
            each task unless you choose to keep it for the session.
            """
        )


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

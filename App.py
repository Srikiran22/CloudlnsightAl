import streamlit as st

from Utils.dataset_ui import render_sidebar

st.set_page_config(
    page_title="CloudInsight AI",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize global session state
if "current_df" not in st.session_state:
    st.session_state["current_df"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None


def render_home():
    render_sidebar()
    st.title("☁️ CloudInsight AI")
    st.subheader("Intelligent Data Analytics Platform with AI-Powered Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cloud Storage", "Amazon S3")

    with col2:
        st.metric("AI Intelligence", "Gemini 1.5 / 2.0")

    with col3:
        st.metric("Machine Learning", "Scikit-Learn")

    st.markdown("---")

    st.markdown("""
### 🚀 Platform Workflow
1. **📂 Upload**: Ingest *any* data file — CSV, Excel, JSON, TSV, Parquet, XML, HTML, text, or PDF. Structured formats parse natively; unstructured ones are converted into clean tables by Gemini AI.
2. **🧹 Data Cleaning**: Remove duplicates and impute/drop missing values with custom strategies.
3. **⚖️ Compare**: Diff any two datasets — schema changes, missing-value drift, and numeric shifts.
4. **📊 EDA**: In-depth statistics, correlation heatmaps, missing value analysis, and IQR outlier detection.
5. **📈 Visualizations**: Interactive plotting and trend analysis.
6. **🤖 Machine Learning**: Automated classification/regression, model persistence, and prediction exports.
7. **💡 AI Insights**: Gemini-powered dataset interpretations and conversational Q&A.
8. **📄 Export & Reports**: Detailed multi-section PDF reports with templates and batch generation.
""")


pg = st.navigation(
    {
        "Home": [
            st.Page(render_home, title="Home", icon="☁️", default=True),
        ],
        "Workspace": [
            st.Page("Pages/Upload.py", title="Upload", icon="📂"),
            st.Page("Pages/Cleaning.py", title="Cleaning", icon="🧹"),
            st.Page("Pages/Compare.py", title="Compare", icon="⚖️"),
            st.Page("Pages/EDA.py", title="EDA", icon="📊"),
            st.Page("Pages/Visualization.py", title="Visualizations", icon="📈"),
            st.Page("Pages/ML.py", title="Machine Learning", icon="🤖"),
            st.Page("Pages/AI.py", title="AI Insights", icon="💡"),
            st.Page("Pages/Report.py", title="PDF Report", icon="📄"),
            st.Page("Pages/Dashboard.py", title="Dashboard", icon="🎛️"),
        ],
    }
)
pg.run()

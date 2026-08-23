import streamlit as st
import pandas as pd

from Utils.dataset_ui import render_sidebar, select_working_dataset
from Utils.logsys import get_logger

logger = get_logger("EDA")

st.title("Exploratory data analysis")
st.markdown("Distributions, missing data, correlations, and outliers for the active dataset.")

df, selected_file = select_working_dataset("Select Dataset for EDA:")
render_sidebar()

st.success(f"Loaded: `{selected_file}`")

st.subheader("Overview")
rows, columns = df.shape
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Rows", f"{rows:,}")
with col2:
    st.metric("Total Columns", columns)
with col3:
    st.metric("Numeric Features", len(df.select_dtypes(include="number").columns))
with col4:
    st.metric("Categorical Features", len(df.select_dtypes(exclude="number").columns))

st.subheader("Column structure & missing values")
missing_counts = df.isnull().sum()
missing_pcts = (missing_counts / max(rows, 1)) * 100

col_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.notnull().sum().values,
    "Missing Count": missing_counts.values,
    "Missing %": [f"{p:.2f}%" for p in missing_pcts.values],
    "Unique Values": df.nunique().values
})
st.dataframe(col_info, width="stretch")

st.subheader("Descriptive statistics")
tab_num, tab_cat = st.tabs(["Numerical", "Categorical"])

with tab_num:
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        st.dataframe(num_df.describe().transpose(), width="stretch")
    else:
        st.info("No numerical columns found in this dataset.")

with tab_cat:
    cat_df = df.select_dtypes(exclude="number")
    if not cat_df.empty:
        st.dataframe(cat_df.describe().transpose(), width="stretch")
    else:
        st.info("No categorical columns found in this dataset.")

st.subheader("Correlation matrix")
numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    correlation = numeric_df.corr()
    try:
        st.dataframe(correlation.style.background_gradient(cmap="coolwarm", axis=None).format(precision=3), width="stretch")
    except Exception as error:
        # styled rendering is a nicety; fall back to plain numbers, loudly
        logger.warning("correlation styling skipped: %s: %s", type(error).__name__, error)
        st.dataframe(correlation.round(3), width="stretch")
else:
    st.info("At least two numeric columns are required for correlation analysis.")

st.subheader("Outlier detection (IQR method)")
numeric_columns = numeric_df.columns

if len(numeric_columns) == 0:
    st.info("No numerical columns available for outlier detection.")
else:
    outlier_results = []

    for column in numeric_columns:
        series = df[column].dropna()
        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outliers_count = 0
            lower_bound = q1
            upper_bound = q3
        else:
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outliers_count = len(outliers)

        pct = (outliers_count / len(series)) * 100 if len(series) > 0 else 0

        outlier_results.append({
            "Column": column,
            "Q1 (25%)": round(q1, 3),
            "Q3 (75%)": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Outlier Count": outliers_count,
            "Outlier %": f"{pct:.2f}%"
        })

    outlier_table = pd.DataFrame(outlier_results)
    st.dataframe(outlier_table, width="stretch")

import streamlit as st
import pandas as pd

from Utils.dataset_ui import render_sidebar, select_working_dataset
from Utils.logsys import get_logger
from Utils.sampling import sample_for_analysis
from Utils.theme import page_header, section_card_header

logger = get_logger("EDA")

page_header(
    "Explore Columns",
    "Distributions, missing data, summary statistics, correlations, and outliers for the active dataset."
)

df, selected_file = select_working_dataset("Active Dataset:")
render_sidebar()

rows, columns = df.shape
num_features = len(df.select_dtypes(include="number").columns)
cat_features = len(df.select_dtypes(exclude="number").columns)
mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024) if not df.empty else 0.0

with st.container(border=True):
    section_card_header("Structural Overview", f"High-level dataset dimensions for {selected_file}")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Rows", f"{rows:,}")
    with k2:
        st.metric("Columns", columns)
    with k3:
        st.metric("Numeric", num_features)
    with k4:
        st.metric("Categorical", cat_features)
    with k5:
        st.metric("Memory", f"{mem_mb:.2f} MB")

st.markdown("---")

tab_schema, tab_stats, tab_corr, tab_outliers = st.tabs([
    "Columns & Missing Data",
    "Summary Statistics",
    "Correlations",
    "Unusual Values (Outliers)"
])

analysis_df, is_sampled, scope_note = sample_for_analysis(df)

with tab_schema:
    with st.container(border=True):
        section_card_header("Column Structure & Nullability", "Data types, non-null counts, and missingness ratios")
        if is_sampled:
            st.info(scope_note)

        missing_counts = analysis_df.isnull().sum()
        missing_pcts = (missing_counts / max(len(analysis_df), 1)) * 100

        col_info = pd.DataFrame({
            "Column": analysis_df.columns,
            "Data Type": analysis_df.dtypes.astype(str).values,
            "Non-Null Count": analysis_df.notnull().sum().values,
            "Missing Count": missing_counts.values,
            "Missing %": [f"{p:.2f}%" for p in missing_pcts.values],
            "Unique Values": analysis_df.nunique().values
        })
        st.dataframe(col_info, width="stretch", hide_index=True)

with tab_stats:
    with st.container(border=True):
        section_card_header("Descriptive Statistics", "Summary statistics across numerical and categorical features")
        tab_num, tab_cat = st.tabs(["Numerical Features", "Categorical Features"])

        with tab_num:
            num_df = analysis_df.select_dtypes(include="number")
            if not num_df.empty:
                st.dataframe(num_df.describe().transpose(), width="stretch")
            else:
                st.info("No numerical columns found in this dataset.")

        with tab_cat:
            cat_df = analysis_df.select_dtypes(exclude="number")
            if not cat_df.empty:
                st.dataframe(cat_df.describe().transpose(), width="stretch")
            else:
                st.info("No categorical columns found in this dataset.")

with tab_corr:
    with st.container(border=True):
        section_card_header("Correlation Matrix", "Pairwise Pearson correlation coefficients")
        numeric_df = analysis_df.select_dtypes(include="number")

        if numeric_df.shape[1] >= 2:
            corr_df = numeric_df
            if numeric_df.shape[1] > 50:
                st.info("Showing correlation for top 50 numeric columns by variance to ensure responsive display.")
                variances = numeric_df.var().sort_values(ascending=False)
                corr_df = numeric_df[variances.head(50).index]
            correlation = corr_df.corr()
            try:
                st.dataframe(correlation.style.background_gradient(cmap="coolwarm", axis=None).format(precision=3), width="stretch")
            except Exception as error:
                logger.warning("correlation styling skipped: %s: %s", type(error).__name__, error)
                st.dataframe(correlation.round(3), width="stretch")
        else:
            st.info("At least two numeric columns are required for correlation analysis.")

with tab_outliers:
    with st.container(border=True):
        section_card_header("IQR Outlier Audit (Tukey's Fences)", "1.5 × IQR threshold outlier detection")
        numeric_columns = analysis_df.select_dtypes(include="number").columns

        if len(numeric_columns) == 0:
            st.info("No numerical columns available for outlier detection.")
        else:
            outlier_results = []
            for column in numeric_columns:
                series = analysis_df[column].dropna()
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
            st.dataframe(outlier_table, width="stretch", hide_index=True)


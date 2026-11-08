import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from Utils.dataset_ui import render_sidebar, select_working_dataset
from Utils.eda_logic import audit_outliers, calculate_vif
from Utils.logsys import get_logger
from Utils.sampling import sample_for_analysis
from Utils.theme import metric_card, page_header, section_card_header

logger = get_logger("EDA")

page_header(
    "Statistical Diagnostics",
    "Comprehensive parametric and non-parametric exploratory data analysis, normality screening, VIF multicollinearity, and hypothesis tests.",
    badge="STATISTICAL AUDIT"
)

df, selected_file = select_working_dataset("Active dataset")
render_sidebar()

rows, columns = df.shape
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024) if not df.empty else 0.0

with st.container(border=True):
    section_card_header("Structural Feature Space", f"Dimensionality overview for `{selected_file}`")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Total Rows", f"{rows:,}")
    with k2:
        metric_card("Total Columns", columns)
    with k3:
        metric_card("Numeric Features", len(num_cols))
    with k4:
        metric_card("Categorical", len(cat_cols))
    with k5:
        metric_card("RAM Footprint", f"{mem_mb:.2f} MB")

analysis_df, is_sampled, scope_note = sample_for_analysis(df)
if is_sampled:
    st.info(scope_note)

tab_univariate, tab_corr, tab_vif, tab_outliers, tab_hypothesis = st.tabs([
    "1. Univariate & Normality",
    "2. Correlation & Covariance",
    "3. Multicollinearity (VIF)",
    "4. Outlier Diagnostics",
    "5. Group Hypothesis Testing"
])

# -----------------------------------------------------------------------------
# TAB 1: UNIVARIATE & NORMALITY
# -----------------------------------------------------------------------------
with tab_univariate:
    with st.container(border=True):
        section_card_header("Numerical Feature Distributions & Normality", "Summary statistics, skewness, kurtosis, and D'Agostino-Pearson normality tests")
        if not num_cols:
            st.info("No numerical columns found in this dataset.")
        else:
            stats_list = []
            for col in num_cols:
                series = analysis_df[col].dropna()
                if series.empty:
                    continue
                mean_val = series.mean()
                std_val = series.std()
                median_val = series.median()
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                skew_val = series.skew()
                kurt_val = series.kurtosis()

                # Normality test
                norm_status = "N/A (N < 8)"
                if len(series) >= 8:
                    try:
                        _, p_val = stats.normaltest(series)
                        norm_status = "Gaussian (p ≥ 0.05)" if p_val >= 0.05 else f"Non-Gaussian (p={p_val:.3e})"
                    except Exception:
                        norm_status = "Undetermined"

                stats_list.append({
                    "Feature": col,
                    "Mean": round(mean_val, 3),
                    "Std Dev": round(std_val, 3),
                    "Median": round(median_val, 3),
                    "IQR": round(q75 - q25, 3),
                    "Skewness": round(skew_val, 3),
                    "Kurtosis": round(kurt_val, 3),
                    "Distribution": norm_status
                })

            stats_df = pd.DataFrame(stats_list)
            st.dataframe(stats_df, width="stretch", hide_index=True)

        if cat_cols:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            section_card_header("Categorical Dimension Cardinality", "Unique levels and top frequency metrics")
            cat_summary = []
            for c in cat_cols:
                s = analysis_df[c].dropna()
                top_val = s.mode().iloc[0] if not s.empty else "None"
                top_freq = (s == top_val).sum() if not s.empty else 0
                cat_summary.append({
                    "Categorical Feature": c,
                    "Unique Values": analysis_df[c].nunique(),
                    "Top Level": str(top_val),
                    "Top Level Frequency": f"{top_freq:,} ({(top_freq / max(len(s), 1)) * 100:.1f}%)"
                })
            st.dataframe(pd.DataFrame(cat_summary), width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 2: CORRELATION & COVARIANCE
# -----------------------------------------------------------------------------
with tab_corr:
    with st.container(border=True):
        section_card_header("Pairwise Feature Correlation Matrix", "Measure linear (Pearson) or monotonic (Spearman) relationships")
        if len(num_cols) < 2:
            st.info("At least two numeric columns are required for correlation analysis.")
        else:
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                corr_method = st.selectbox("Correlation Metric", ["Pearson (Linear relationships)", "Spearman (Monotonic / Non-linear rank)"])
                method_key = "pearson" if "Pearson" in corr_method else "spearman"
            with col_c2:
                corr_sub_cols = st.multiselect("Filter columns to display (default: all)", num_cols, default=num_cols[:15])

            cols_to_corr = corr_sub_cols if corr_sub_cols else num_cols
            corr_matrix = analysis_df[cols_to_corr].corr(method=method_key)

            st.dataframe(
                corr_matrix.style.background_gradient(cmap="coolwarm", vmin=-1.0, vmax=1.0).format("{:.3f}"),
                width="stretch"
            )

            # Top positive and negative correlations
            corr_unstack = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).unstack().dropna()
            if not corr_unstack.empty:
                top_pairs = corr_unstack.sort_values(ascending=False)
                c_top1, c_top2 = st.columns(2)
                with c_top1:
                    st.caption("**Strongest Positive Correlations:**")
                    for (f1, f2), r in top_pairs.head(3).items():
                        st.markdown(f"- `{f1}` ↔ `{f2}`: **r = {r:+.3f}**")
                with c_top2:
                    st.caption("**Strongest Negative Correlations:**")
                    for (f1, f2), r in top_pairs.tail(3).items():
                        st.markdown(f"- `{f1}` ↔ `{f2}`: **r = {r:+.3f}**")


# -----------------------------------------------------------------------------
# TAB 3: MULTICOLLINEARITY (VIF)
# -----------------------------------------------------------------------------
with tab_vif, st.container(border=True):
    section_card_header(
        "Variance Inflation Factor (VIF) Diagnostic",
        "Identify severe multicollinearity among predictor variables (VIF ≥ 10 indicates high redundancy)."
    )
    if len(num_cols) < 2:
        st.info("Need at least 2 numerical features to evaluate multicollinearity.")
    else:
        vif_candidates = [c for c in num_cols if analysis_df[c].nunique() > 1]
        if len(vif_candidates) < 2:
            st.info("Not enough non-constant numeric features to compute VIF.")
        else:
            if st.button("Compute Variance Inflation Factors", type="primary", key="btn_run_vif"):
                with st.spinner("Calculating VIF across feature matrix..."):
                    vif_df = calculate_vif(analysis_df, vif_candidates)
                    st.session_state["vif_results"] = vif_df

            if "vif_results" in st.session_state:
                st.dataframe(st.session_state["vif_results"], width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 4: OUTLIER DIAGNOSTICS
# -----------------------------------------------------------------------------
with tab_outliers, st.container(border=True):
    section_card_header(
        "Tukey's IQR & Z-Score Outlier Audit",
        "Screen for abnormal observations exceeding 1.5 × IQR bounds or |Z| ≥ 3.0 standard deviations."
    )
    if not num_cols:
        st.info("No numerical features available for outlier detection.")
    else:
        outlier_df = audit_outliers(analysis_df, num_cols, k=1.5)
        st.dataframe(outlier_df, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 5: GROUP HYPOTHESIS TESTING
# -----------------------------------------------------------------------------
with tab_hypothesis, st.container(border=True):
    section_card_header(
        "Statistical Hypothesis Testing Across Groups",
        "Evaluate whether numerical feature means differ significantly across categories (One-Way ANOVA F-Test)."
    )
    if not num_cols or not cat_cols:
        st.info("Requires at least one numerical variable and one categorical grouping variable.")
    else:
        h_c1, h_c2 = st.columns(2)
        with h_c1:
            group_col = st.selectbox("Categorical Grouping Variable", cat_cols, key="h_group")
        with h_c2:
            num_target = st.selectbox("Numerical Metric", num_cols, key="h_target")

        group_counts = analysis_df[group_col].value_counts()
        valid_groups = group_counts[group_counts >= 3].index.tolist()

        if len(valid_groups) < 2:
            st.warning(f"Categorical variable `{group_col}` needs at least 2 distinct groups with ≥ 3 records each.")
        else:
            group_data = [analysis_df[analysis_df[group_col] == g][num_target].dropna().values for g in valid_groups]
            f_stat, p_val = stats.f_oneway(*group_data)

            m_h1, m_h2, m_h3 = st.columns(3)
            with m_h1:
                metric_card("F-Statistic", f"{f_stat:.3f}")
            with m_h2:
                p_formatted = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
                metric_card("p-value", p_formatted)
            with m_h3:
                sig_label = "Statistically Significant" if p_val < 0.05 else "Not Significant"
                metric_card("Verdict (α = 0.05)", sig_label, subtext="H₀: All group means equal")

            # Group breakdown summary
            grp_summary = analysis_df[analysis_df[group_col].isin(valid_groups)].groupby(group_col)[num_target].agg(
                ["count", "mean", "std", "median"]
            ).reset_index()
            st.dataframe(grp_summary.round(3), width="stretch", hide_index=True)

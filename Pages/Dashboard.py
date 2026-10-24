import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from Utils.dataset_ui import dataframe_fingerprint, render_sidebar, select_working_dataset
from Utils.quality import quality_metrics
from Utils.Charts import create_histogram_plot, create_box_violin_plot, create_hygiene_gauge
from Utils.sampling import sample_for_visualization
from Utils.theme import chart_palette, page_header, section_card_header, metric_card

page_header(
    "Data Health & Profiler",
    "Overall data hygiene score, structural integrity audit, missingness matrix, and interactive slice-and-dice filter.",
    badge="DATA AUDIT"
)

df, selected_file = select_working_dataset("Active dataset")
render_sidebar()

# Cache quality metrics by dataset fingerprint to avoid repeated O(N) full dataset scans on reruns
df_fp = dataframe_fingerprint(df)
metrics_cache_key = f"dash_quality_metrics_{selected_file}_{df_fp}"
if metrics_cache_key not in st.session_state:
    st.session_state[metrics_cache_key] = quality_metrics(df)
metrics = st.session_state[metrics_cache_key]
rows, cols = metrics["rows"], metrics["cols"]
missing_cells = metrics["missing_cells"]
completeness_score = metrics["completeness"]
dup_rows = metrics["duplicate_rows"]
uniqueness_score = metrics["uniqueness"]
health_index = metrics["index"]

col_gauge_card, col_metrics_card = st.columns([1.2, 1.8])

with col_gauge_card:
    with st.container(border=True):
        section_card_header("Data Hygiene Index", "Composite quality score (completeness & uniqueness)")
        st.plotly_chart(create_hygiene_gauge(health_index), width="stretch", theme="streamlit")

with col_metrics_card:
    with st.container(border=True):
        section_card_header("Quality & Hygiene Audit", f"Structural diagnostics for `{selected_file}`")
        kpi_r1_1, kpi_r1_2, kpi_r1_3 = st.columns(3)
        with kpi_r1_1:
            metric_card("Completeness", f"{completeness_score:.1f}%", subtext="Non-null cell ratio")
        with kpi_r1_2:
            metric_card("Uniqueness", f"{uniqueness_score:.1f}%", subtext="Distinct row ratio")
        with kpi_r1_3:
            metric_card("Total Rows", f"{rows:,}", subtext=f"{cols} attributes")

        kpi_r2_1, kpi_r2_2, kpi_r2_3 = st.columns(3)
        with kpi_r2_1:
            metric_card("Missing Cells", f"{missing_cells:,}", subtext="Null / NaN values")
        with kpi_r2_2:
            metric_card("Duplicate Rows", f"{dup_rows:,}", subtext="Exact identical rows")
        with kpi_r2_3:
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024) if not df.empty else 0.0
            metric_card("Memory Footprint", f"{mem_mb:.2f} MB", subtext="Deep RAM usage")

tab_profiler, tab_filter, tab_dist = st.tabs([
    "Column Profiler & Missingness",
    "Interactive Slice & Dice Filter",
    "Distribution Snapshots"
])

# -----------------------------------------------------------------------------
# TAB 1: COLUMN PROFILER & MISSINGNESS
# -----------------------------------------------------------------------------
with tab_profiler:
    with st.container(border=True):
        section_card_header("Attribute Completeness & Type Breakdown", "Per-column null counts, uniqueness, and memory consumption")
        null_counts = df.isnull().sum()
        null_pcts = (null_counts / max(len(df), 1)) * 100
        uniques = df.nunique()
        dtypes = df.dtypes.astype(str)

        prof_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
            "Missing Count": null_counts.values,
            "Missing %": null_pcts.values,
            "Unique Values": uniques.values,
            "Cardinality": ["Single Value (Constant)" if u == 1 else "Binary (2)" if u == 2 else "Low (3-20)" if u <= 20 else "High (>20)" for u in uniques.values]
        }).sort_values(by="Missing Count", ascending=False)

        def style_missing(series):
            styles = []
            for val in series:
                if val > 15:
                    styles.append("background-color: rgba(239, 68, 68, 0.35); color: #fca5a5; font-weight: 600;")
                elif val > 0:
                    styles.append("background-color: rgba(245, 158, 11, 0.25); color: #fcd34d;")
                else:
                    styles.append("")
            return styles

        st.dataframe(
            prof_df.style.apply(style_missing, subset=["Missing %"]).format({"Missing %": "{:.2f}%"}),
            width="stretch",
            hide_index=True
        )

        # Missingness bar chart
        cols_with_missing = prof_df[prof_df["Missing Count"] > 0]
        if not cols_with_missing.empty:
            fig_miss = px.bar(
                cols_with_missing,
                x="Column",
                y="Missing %",
                title="Features with Missing Values (%)",
                color="Missing %",
                color_continuous_scale="Reds",
            )
            fig_miss.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_miss, width="stretch", theme="streamlit")
        else:
            st.success("Perfect data integrity: zero missing values detected across all columns!")


# -----------------------------------------------------------------------------
# TAB 2: INTERACTIVE SLICE & DICE FILTER
# -----------------------------------------------------------------------------
with tab_filter:
    with st.container(border=True):
        section_card_header("Interactive Slice & Dice Filter", "Filter rows by numeric thresholds or discrete categories in real time")
        filter_cols = st.multiselect("Select columns to filter on", df.columns.tolist())
        filtered_df = df

        if filter_cols:
            col_chunks = st.columns(min(len(filter_cols), 4))
            for idx, fcol in enumerate(filter_cols):
                chunk = col_chunks[idx % len(col_chunks)]
                with chunk:
                    if pd.api.types.is_numeric_dtype(df[fcol]):
                        series = df[fcol].dropna()
                        finite_series = series[np.isfinite(series)]
                        if finite_series.empty:
                            st.caption(f"{fcol}: no finite numeric values to filter")
                            continue
                        min_v = float(finite_series.min())
                        max_v = float(finite_series.max())
                        if min_v < max_v:
                            selected_range = st.slider(f"{fcol}:", min_value=min_v, max_value=max_v, value=(min_v, max_v))
                            if selected_range[0] > min_v or selected_range[1] < max_v:
                                mask = (
                                    np.isfinite(filtered_df[fcol])
                                    & (filtered_df[fcol] >= selected_range[0])
                                    & (filtered_df[fcol] <= selected_range[1])
                                )
                                filtered_df = filtered_df[mask]
                        else:
                            st.caption(f"{fcol}: constant value {min_v}")
                    else:
                        cardinality = int(df[fcol].nunique(dropna=True))
                        if cardinality <= 50:
                            has_nulls = bool(df[fcol].isna().any())
                            null_label = "(Missing / Null)"
                            base_vals = sorted(df[fcol].dropna().astype(str).unique().tolist())
                            unique_vals = [null_label] + base_vals if has_nulls else base_vals
                            chosen_vals = st.multiselect(f"{fcol}:", unique_vals, default=unique_vals)
                            if len(chosen_vals) < len(unique_vals):
                                keep_nulls = null_label in chosen_vals
                                non_null_chosen = set(chosen_vals) - {null_label}
                                mask = (
                                    (filtered_df[fcol].isna() if keep_nulls else False)
                                    | (filtered_df[fcol].notna() & filtered_df[fcol].astype(str).isin(non_null_chosen))
                                )
                                filtered_df = filtered_df[mask]
                        else:
                            st.caption(f"{fcol}: {cardinality:,} unique values.")
                            search = st.text_input(f"{fcol} contains:", key=f"dash_search_{fcol}")
                            if search:
                                filtered_df = filtered_df[
                                    filtered_df[fcol].astype(str).str.contains(search, case=False, na=False, regex=False)
                                ]

        st.caption(f"Showing **{filtered_df.shape[0]:,}** of {rows:,} rows after active filters.")
        st.dataframe(filtered_df.head(15), width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# TAB 3: DISTRIBUTION SNAPSHOTS
# -----------------------------------------------------------------------------
with tab_dist:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        with st.container(border=True):
            section_card_header("Numerical Distribution Previews", "Quick histogram and box-plot checks")
            chart_sample_df, was_sampled, sample_note = sample_for_visualization(df)
            if was_sampled:
                st.caption(sample_note)
            c_pick1, c_pick2 = st.columns(2)
            with c_pick1:
                q_col1 = st.selectbox("Feature Histogram", num_cols, index=0, key="q1")
                fig1 = create_histogram_plot(chart_sample_df, x_col=q_col1, color_discrete_sequence=[chart_palette()[0]])
                st.plotly_chart(fig1, width="stretch", theme="streamlit")

            with c_pick2:
                if len(num_cols) > 1:
                    q_col2 = st.selectbox("Feature Box / Outliers", num_cols, index=1, key="q2")
                    fig2 = create_box_violin_plot(chart_sample_df, y_col=q_col2)
                    st.plotly_chart(fig2, width="stretch", theme="streamlit")
    else:
        st.info("No numeric columns available for distribution charts.")

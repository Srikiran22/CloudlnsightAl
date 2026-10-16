import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Utils.dataset_ui import dataframe_fingerprint, render_sidebar, select_working_dataset
from Utils.quality import quality_metrics
from Utils.Charts import create_histogram_plot, create_box_violin_plot
from Utils.sampling import sample_for_visualization
from Utils.theme import page_header, section_card_header

page_header(
    "Data Overview",
    "Overall data health score, quality indicators, and interactive slice-and-dice filtering."
)

df, selected_file = select_working_dataset("Active Dataset:")
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
        section_card_header("Data Hygiene Index", "Composite health score (completeness & uniqueness)")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(health_index, 1),
            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                'bar': {'color': "#2563EB"},
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 80], 'color': "#FEF3C7"},
                    {'range': [80, 100], 'color': "#D1FAE5"}
                ],
                'threshold': {
                    'line': {'color': "#16A34A", 'width': 3},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(
            height=230,
            margin={"l": 40, "r": 50, "t": 40, "b": 25},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, width="stretch")

with col_metrics_card:
    with st.container(border=True):
        section_card_header("Quality & Hygiene Audit", "Key structural indicators for active dataset")
        kpi_r1_1, kpi_r1_2, kpi_r1_3 = st.columns(3)
        with kpi_r1_1:
            st.metric("Completeness", f"{completeness_score:.1f}%", help="Percentage of non-empty data cells")
        with kpi_r1_2:
            st.metric("Uniqueness", f"{uniqueness_score:.1f}%", help="Percentage of distinct rows")
        with kpi_r1_3:
            st.metric("Total Rows", f"{rows:,}")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        kpi_r2_1, kpi_r2_2, kpi_r2_3 = st.columns(3)
        with kpi_r2_1:
            st.metric("Missing Cells", f"{missing_cells:,}")
        with kpi_r2_2:
            st.metric("Duplicate Rows", f"{dup_rows:,}")
        with kpi_r2_3:
            st.metric("Columns", f"{cols}")

st.markdown("---")

with st.container(border=True):
    section_card_header("Interactive Slice & Dice Filter", "Filter rows by numeric thresholds or discrete categories in real time")
    filter_cols = st.multiselect("Select Columns to Filter On:", df.columns.tolist())
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
                        st.caption(f"{fcol}: {cardinality:,} unique values (high cardinality).")
                        search = st.text_input(f"{fcol} contains:", key=f"dash_search_{fcol}")
                        if search:
                            filtered_df = filtered_df[
                                filtered_df[fcol].astype(str).str.contains(search, case=False, na=False, regex=False)
                            ]

    st.caption(f"Showing **{filtered_df.shape[0]:,}** of {rows:,} rows after active filters.")
    st.dataframe(filtered_df.head(10), width="stretch", hide_index=True)

st.markdown("---")

num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if num_cols:
    with st.container(border=True):
        section_card_header("Quick Numerical Distributions", "Visual shape of filtered numeric distributions")
        chart_sample_df, was_sampled, sample_note = sample_for_visualization(filtered_df)
        if was_sampled:
            st.caption(sample_note)
        c_pick1, c_pick2 = st.columns(2)
        with c_pick1:
            q_col1 = st.selectbox("Metric 1:", num_cols, index=0, key="q1")
            fig1 = create_histogram_plot(chart_sample_df, x_col=q_col1, color_discrete_sequence=["#3B82F6"])
            st.plotly_chart(fig1, width="stretch")

        with c_pick2:
            if len(num_cols) > 1:
                q_col2 = st.selectbox("Metric 2:", num_cols, index=1, key="q2")
                fig2 = create_box_violin_plot(chart_sample_df, y_col=q_col2)
                st.plotly_chart(fig2, width="stretch")
else:
    st.info("No numeric columns available for distribution charts.")


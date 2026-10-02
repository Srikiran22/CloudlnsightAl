import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Utils.dataset_ui import render_sidebar, select_working_dataset
from Utils.quality import quality_metrics
from Utils.Charts import create_histogram_plot, create_box_violin_plot
from Utils.sampling import sample_for_visualization

st.title("Dashboard")
st.markdown("Data health indicators, live filtering, and quick distributions.")

df, selected_file = select_working_dataset("Select Dataset for Dashboard:")
render_sidebar()
st.caption(f"Active: `{selected_file}`")

metrics = quality_metrics(df)
rows, cols = metrics["rows"], metrics["cols"]
missing_cells = metrics["missing_cells"]
completeness_score = metrics["completeness"]
dup_rows = metrics["duplicate_rows"]
uniqueness_score = metrics["uniqueness"]
health_index = metrics["index"]

st.subheader("Data health index")
g1, g2, g3 = st.columns(3)

with g1:
    fig_gauge = go.Figure(go.Indicator(
         mode="gauge+number",
         value=round(health_index, 1),
         title={'text': "Data Hygiene Index", 'font': {'size': 16}},
         gauge={
             'axis': {'range': [0, 100]},
             'bar': {'color': "#2563EB"},
             'steps': [
                 {'range': [0, 50], 'color': "#FEE2E2"},
                 {'range': [50, 80], 'color': "#FEF3C7"},
                 {'range': [80, 100], 'color': "#D1FAE5"}
             ],
             'threshold': {
                 'line': {'color': "green", 'width': 4},
                 'thickness': 0.75,
                 'value': 90
             }
         }
     ))
    fig_gauge.update_layout(height=220, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    st.plotly_chart(fig_gauge, width="stretch")

with g2:
    st.metric("Completeness Rate", f"{completeness_score:.1f}%", help="Percentage of non-empty data cells")
    st.metric("Total Records", f"{rows:,}")
    st.metric("Features / Columns", f"{cols}")

with g3:
    st.metric(
        "Uniqueness Rate", f"{uniqueness_score:.1f}%",
        help="Percentage of distinct rows. Hygiene Index = (completeness + uniqueness) / 2. "
             "Measures structural completeness and uniqueness; does not verify semantic correctness.",
    )
    st.metric("Missing Cells Count", f"{missing_cells:,}")
    st.metric("Duplicate Rows Count", f"{dup_rows:,}")

st.markdown("---")

st.subheader("Filter rows")
st.markdown("Slice the dataset by column conditions; filters combine in real time.")

filter_cols = st.multiselect("Select Columns to Filter On:", df.columns.tolist())
filtered_df = df.copy()

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
st.dataframe(filtered_df.head(10), width="stretch")

st.markdown("---")
st.subheader("Quick distributions")
num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if num_cols:
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

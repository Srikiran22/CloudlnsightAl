import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from Utils.dataset_ui import render_sidebar, select_working_dataset

st.title("🎛️ Executive Data Dashboard")
st.markdown("Monitor data health indicators, key performance metrics, and interactive column filters.")

df, selected_file = select_working_dataset("Select Dataset for Dashboard:")
render_sidebar()
st.caption(f"Active: `{selected_file}`")

rows, cols = df.shape
total_cells = max(rows * cols, 1)
missing_cells = int(df.isnull().sum().sum())
completeness_score = ((total_cells - missing_cells) / total_cells) * 100

dup_rows = int(df.duplicated().sum())
uniqueness_score = ((rows - dup_rows) / max(rows, 1)) * 100

health_index = (completeness_score * 0.6) + (uniqueness_score * 0.4)

st.subheader("1️⃣ Data Health & Integrity Index")
g1, g2, g3 = st.columns(3)

with g1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(health_index, 1),
        title={'text': "Data Quality Index", 'font': {'size': 16}},
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
    fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, width="stretch")

with g2:
    st.metric("Completeness Rate", f"{completeness_score:.1f}%", help="Percentage of non-empty data cells")
    st.metric("Total Records", f"{rows:,}")
    st.metric("Features / Columns", f"{cols}")

with g3:
    st.metric("Uniqueness Rate", f"{uniqueness_score:.1f}%", help="Percentage of distinct rows")
    st.metric("Missing Cells Count", f"{missing_cells:,}")
    st.metric("Duplicate Rows Count", f"{dup_rows:,}")

st.markdown("---")

st.subheader("2️⃣ Dynamic Slice & Dice Filter")
st.markdown("Filter rows in real-time by selecting column conditions.")

filter_cols = st.multiselect("Select Columns to Filter On:", df.columns.tolist())
filtered_df = df.copy()

if filter_cols:
    col_chunks = st.columns(min(len(filter_cols), 4))
    for idx, fcol in enumerate(filter_cols):
        chunk = col_chunks[idx % len(col_chunks)]
        with chunk:
            if pd.api.types.is_numeric_dtype(df[fcol]):
                series = df[fcol].dropna()
                if series.empty:
                    st.caption(f"{fcol}: no numeric values to filter")
                    continue
                min_v = float(series.min())
                max_v = float(series.max())
                if min_v < max_v:
                    selected_range = st.slider(f"{fcol}:", min_value=min_v, max_value=max_v, value=(min_v, max_v))
                    filtered_df = filtered_df[
                        filtered_df[fcol].isna()
                        | ((filtered_df[fcol] >= selected_range[0]) & (filtered_df[fcol] <= selected_range[1]))
                    ]
                else:
                    st.caption(f"{fcol}: constant value {min_v}")
            else:
                unique_vals = df[fcol].dropna().astype(str).unique().tolist()
                if len(unique_vals) <= 50:
                    chosen_vals = st.multiselect(f"{fcol}:", unique_vals, default=unique_vals)
                    if chosen_vals:
                        filtered_df = filtered_df[filtered_df[fcol].astype(str).isin(chosen_vals) | filtered_df[fcol].isna()]
                else:
                    search = st.text_input(f"{fcol} contains:", key=f"dash_search_{fcol}")
                    if search:
                        filtered_df = filtered_df[
                            filtered_df[fcol].astype(str).str.contains(search, case=False, na=False)
                        ]

st.caption(f"Showing **{filtered_df.shape[0]:,}** of {rows:,} rows after active filters.")
st.dataframe(filtered_df.head(10), width="stretch")

st.markdown("---")
st.subheader("3️⃣ Feature Quick-Look Distributions")
num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if num_cols:
    c_pick1, c_pick2 = st.columns(2)
    with c_pick1:
        q_col1 = st.selectbox("Metric 1:", num_cols, index=0, key="q1")
        fig1 = px.histogram(filtered_df, x=q_col1, title=f"Distribution of {q_col1}", template="plotly_white", color_discrete_sequence=["#3B82F6"])
        st.plotly_chart(fig1, width="stretch")

    with c_pick2:
        if len(num_cols) > 1:
            q_col2 = st.selectbox("Metric 2:", num_cols, index=1, key="q2")
            fig2 = px.box(filtered_df, y=q_col2, title=f"Box Plot of {q_col2}", template="plotly_white", color_discrete_sequence=["#10B981"])
            st.plotly_chart(fig2, width="stretch")
else:
    st.info("No numeric columns available for distribution charts.")

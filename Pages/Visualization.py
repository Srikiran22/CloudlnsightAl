import streamlit as st

from Utils.dataset_ui import render_sidebar, select_working_dataset
from Utils.Charts import (
    create_histogram_plot,
    create_box_violin_plot,
    create_scatter_plot,
    create_bar_count_plot,
    create_line_chart,
    create_pie_treemap_plot,
    create_correlation_heatmap
)

st.title("📈 Interactive Visualizations")
st.markdown("Build publication-ready interactive charts to uncover patterns, trends, and distributions.")

df, selected_file = select_working_dataset("Select Dataset for Visualization:")
render_sidebar()
st.caption(f"Visualizing: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} cols)")

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

tab_dist, tab_rel, tab_cat, tab_corr = st.tabs([
    "📊 Distributions",
    "🎯 Relational & Trends",
    "🔤 Categorical & Composition",
    "🔥 Heatmap Matrix"
])

with tab_dist:
    st.subheader("1️⃣ Distribution Analysis")
    if not numeric_cols:
        st.info("No numeric columns available for distribution plots.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            plot_kind = st.selectbox("Plot Type:", ["Histogram", "Box Plot", "Violin Plot"])
        with c2:
            target_col = st.selectbox("Feature to Inspect:", numeric_cols, key="dist_target")
        with c3:
            hue_choice = st.selectbox("Group By (Hue):", ["None"] + cat_cols + numeric_cols, key="dist_hue")
            hue = None if hue_choice == "None" else hue_choice

        if plot_kind == "Histogram":
            c_bins, c_marg = st.columns(2)
            with c_bins:
                bins = st.slider("Number of Bins:", min_value=5, max_value=100, value=30)
            with c_marg:
                marginal = st.selectbox("Marginal Plot:", ["box", "violin", "rug", "none"])
                marginal = None if marginal == "none" else marginal

            fig = create_histogram_plot(df, x_col=target_col, hue_col=hue, nbins=bins, marginal=marginal)
            st.plotly_chart(fig, width="stretch")

        else:
            c_pts, c_grp = st.columns(2)
            with c_pts:
                points_opt = st.selectbox("Points Display:", ["outliers", "all", "suspectedoutliers", "none"])
                points_opt = False if points_opt == "none" else points_opt
            with c_grp:
                group_col = st.selectbox("Categorical Axis (Optional):", ["None"] + cat_cols, key="box_group")
                group_col = None if group_col == "None" else group_col

            fig = create_box_violin_plot(
                df,
                y_col=target_col,
                x_col=group_col,
                hue_col=hue,
                plot_type="Violin" if plot_kind == "Violin Plot" else "Box",
                points=points_opt
            )
            st.plotly_chart(fig, width="stretch")

with tab_rel:
    st.subheader("2️⃣ Relational & Trend Analysis")
    if len(numeric_cols) < 2:
        st.info("At least 2 numeric columns are required for relational scatter plots.")
    else:
        rel_type = st.radio("Chart Style:", ["Scatter / Bubble Plot", "Line / Trend Plot"], horizontal=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            x_options = numeric_cols if rel_type == "Scatter / Bubble Plot" else all_cols
            x_key = "rel_x_scatter" if rel_type == "Scatter / Bubble Plot" else "rel_x_line"
            x_axis = st.selectbox("X-Axis Feature:", x_options, index=0, key=x_key)
        with r2:
            y_axis = st.selectbox("Y-Axis Feature:", numeric_cols, index=min(1, len(numeric_cols) - 1), key="rel_y")
        with r3:
            color_by = st.selectbox("Color By (Hue):", ["None"] + all_cols, key="rel_color")
            color_by = None if color_by == "None" else color_by
        with r4:
            size_by = None
            if rel_type == "Scatter / Bubble Plot":
                size_by = st.selectbox("Size By (Bubble):", ["None"] + numeric_cols, key="rel_size")
                size_by = None if size_by == "None" else size_by

        if rel_type == "Scatter / Bubble Plot":
            add_trend = st.checkbox("Add OLS Trendline (Linear Regression)", value=False)
            if size_by and (df[size_by].dropna() < 0).any():
                st.warning(f"`{size_by}` contains negative values and cannot be used for bubble sizes.")
                size_by = None
            fig = create_scatter_plot(
                df,
                x_col=x_axis,
                y_col=y_axis,
                hue_col=color_by,
                size_col=size_by,
                add_trendline=add_trend
            )
        else:
            show_markers = st.checkbox("Show Data Markers", value=True)
            fig = create_line_chart(
                df,
                x_col=x_axis,
                y_col=y_axis,
                hue_col=color_by,
                markers=show_markers
            )
        st.plotly_chart(fig, width="stretch")

with tab_cat:
    st.subheader("3️⃣ Categorical & Composition Analysis")
    cat_type = st.radio("Chart Type:", ["Bar / Count Chart", "Pie Chart", "Donut Chart", "Treemap"], horizontal=True)

    if cat_type == "Bar / Count Chart":
        agg_options = ["Count"]
        if numeric_cols:
            agg_options.extend(["Sum", "Mean", "Median", "Max", "Min"])

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            cat_x = st.selectbox("Category (X-Axis):", all_cols, key="bar_x")
        with b2:
            agg_type = st.selectbox("Aggregation:", agg_options, key="bar_agg")
        with b3:
            val_y = None
            if agg_type != "Count":
                val_y = st.selectbox("Value Column (Y-Axis):", numeric_cols, key="bar_y")
        with b4:
            orient = st.selectbox("Orientation:", ["Vertical", "Horizontal"], key="bar_orient")

        fig = create_bar_count_plot(
            df,
            x_col=cat_x,
            y_col=val_y,
            agg_func=agg_type,
            orientation="v" if orient == "Vertical" else "h"
        )
        st.plotly_chart(fig, width="stretch")

    else:
        p1, p2 = st.columns(2)
        with p1:
            cat_name = st.selectbox("Category / Labels Column:", all_cols, key="pie_labels")
        with p2:
            val_choice = st.selectbox("Value Column (Optional):", ["Count (Frequency)"] + numeric_cols, key="pie_vals")
            val_col = None if val_choice == "Count (Frequency)" else val_choice

        plot_kind_pie = "Donut" if cat_type == "Donut Chart" else ("Treemap" if cat_type == "Treemap" else "Pie")
        fig = create_pie_treemap_plot(df, names_col=cat_name, values_col=val_col, plot_type=plot_kind_pie)
        st.plotly_chart(fig, width="stretch")

with tab_corr:
    st.subheader("4️⃣ Correlation Matrix Heatmap")
    if len(numeric_cols) < 2:
        st.info("At least 2 numeric columns are required to generate a correlation heatmap.")
    else:
        c_theme = st.selectbox("Heatmap Color Palette:", ["RdBu_r", "Viridis", "Plasma", "Cividis", "Spectral", "Blues"])
        fig = create_correlation_heatmap(df, colorscale=c_theme)
        if fig:
            st.plotly_chart(fig, width="stretch")

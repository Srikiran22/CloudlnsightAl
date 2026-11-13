from __future__ import annotations

import importlib
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from Utils.sampling import sample_for_analysis, sample_for_visualization
from Utils.theme import (
    chart_palette,
    get_semantic_tokens,
    is_dark,
    plot_template,
    style_plotly_fig,
)

MAX_HUE_CATEGORIES = 20


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB token to an rgba() string for translucent fills."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return hex_color
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def create_hygiene_gauge(value: float, height: int = 230) -> go.Figure:
    """Themed composite data-hygiene gauge for the Data Overview page.

    Uses the shared design tokens (not Plotly defaults) so the score and axis
    labels stay legible in both appearances; the surrounding card is painted by
    the CSS layer, so the figure background stays transparent.
    """
    tokens = get_semantic_tokens(is_dark())
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(float(value), 1),
            number={"font": {"size": 38, "color": tokens["text"]}},
            domain={"x": [0.06, 0.94], "y": [0.10, 0.92]},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": tokens["muted"],
                    "tickfont": {"size": 11, "color": tokens["muted"]},
                },
                "bar": {"color": tokens["accent"], "thickness": 0.72},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": _rgba(tokens["danger"], 0.16)},
                    {"range": [50, 80], "color": _rgba(tokens["warning"], 0.16)},
                    {"range": [80, 100], "color": _rgba(tokens["success"], 0.16)},
                ],
                "threshold": {
                    "line": {"color": tokens["success"], "width": 3},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
        )
    )
    # Text colours are intentionally left to Streamlit's native chart theme so
    # the gauge re-colours instantly when the user switches appearance.
    fig.update_layout(height=height, margin={"l": 24, "r": 24, "t": 28, "b": 8})
    return fig


def _cap_hue_column(df, hue_col, max_categories=MAX_HUE_CATEGORIES):
    """Cap high-cardinality hue column to top N categories plus 'Other' to prevent browser freeze."""
    if not hue_col or hue_col not in df.columns:
        return df
    counts = df[hue_col].value_counts(dropna=False)
    if len(counts) > max_categories:
        top_cats = set(counts.head(max_categories).index)
        df_copy = df.copy()
        df_copy[hue_col] = df_copy[hue_col].apply(lambda v: v if v in top_cats else "Other")
        return df_copy
    return df


def create_histogram_plot(
    df: pd.DataFrame,
    x_col: str,
    hue_col: str | None = None,
    nbins: int = 30,
    marginal: str = "box",
    color_discrete_sequence: list[str] | None = None,
) -> go.Figure:
    df_sample, is_sampled, _ = sample_for_visualization(df)
    plot_df = _cap_hue_column(df_sample, hue_col)
    fig = px.histogram(
        plot_df,
        x=x_col,
        color=hue_col,
        nbins=nbins,
        marginal=marginal,
        barmode="overlay" if hue_col else "relative",
        opacity=0.75,
        color_discrete_sequence=color_discrete_sequence or chart_palette(),
        template=plot_template()
    )
    safe_x = escape(str(x_col))
    safe_hue = escape(str(hue_col)) if hue_col else ""
    title_text = f"Distribution of <b>{safe_x}</b>" + (f" by {safe_hue}" if hue_col else "")
    if is_sampled:
        title_text += f" (sampled {len(df_sample):,}/{len(df):,} rows)"
    fig.update_layout(
        title=title_text,
        xaxis_title=str(x_col),
        yaxis_title="Count / Frequency",
        bargap=0.05
    )
    return style_plotly_fig(fig)


def create_box_violin_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: str | None = None,
    hue_col: str | None = None,
    plot_type: str = "Box",
    points: str = "outliers",
) -> go.Figure:
    df_sample, is_sampled, _ = sample_for_visualization(df)
    plot_df = _cap_hue_column(df_sample, hue_col)
    if x_col and x_col in plot_df.columns:
        plot_df = _cap_hue_column(plot_df, x_col, max_categories=MAX_HUE_CATEGORIES)

    safe_y = escape(str(y_col))
    safe_x = escape(str(x_col)) if x_col else ""
    sample_suffix = f" (sampled {len(df_sample):,}/{len(df):,} rows)" if is_sampled else ""
    if plot_type == "Violin":
        fig = px.violin(
            plot_df,
            y=y_col,
            x=x_col,
            color=hue_col or x_col,
            box=True,
            points=points,
            template=plot_template()
        )
        fig.update_layout(title=f"Violin Plot of <b>{safe_y}</b>" + (f" across {safe_x}" if x_col else "") + sample_suffix)
    else:
        fig = px.box(
            plot_df,
            y=y_col,
            x=x_col,
            color=hue_col or x_col,
            points=points,
            notched=False,
            template=plot_template()
        )
        fig.update_layout(title=f"Box Plot of <b>{safe_y}</b>" + (f" across {safe_x}" if x_col else "") + sample_suffix)

    fig.update_layout(yaxis_title=str(y_col), xaxis_title=str(x_col or ""))
    return style_plotly_fig(fig)


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    add_trendline: bool = False,
) -> go.Figure:
    df_sample, is_sampled, _ = sample_for_visualization(df)
    # OLS only makes sense when both axes are numeric and finite
    can_fit_trendline = (
        pd.api.types.is_numeric_dtype(df_sample[x_col])
        and pd.api.types.is_numeric_dtype(df_sample[y_col])
    )
    plot_df = _cap_hue_column(df_sample, hue_col)
    if add_trendline and can_fit_trendline:
        try:
            importlib.import_module("statsmodels.api")
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False

        if has_statsmodels:
            x_num = pd.to_numeric(plot_df[x_col], errors="coerce")
            y_num = pd.to_numeric(plot_df[y_col], errors="coerce")
            finite_mask = np.isfinite(x_num) & np.isfinite(y_num)
            if finite_mask.sum() >= 2:
                plot_df = plot_df[finite_mask]
                trendline = "ols"
            else:
                trendline = None
        else:
            trendline = None
    else:
        trendline = None

    if size_col:
        size_values = pd.to_numeric(plot_df[size_col], errors="coerce")
        if size_values.empty or not np.isfinite(size_values).any() or (size_values.dropna() < 0).any():
            size_col = None

    try:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=hue_col,
            size=size_col,
            trendline=trendline,
            opacity=0.8,
            template=plot_template(),
            hover_data=plot_df.columns[:5].tolist()
        )
    except Exception:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=hue_col,
            size=size_col,
            trendline=None,
            opacity=0.8,
            template=plot_template(),
            hover_data=plot_df.columns[:5].tolist()
        )
    safe_x = escape(str(x_col))
    safe_y = escape(str(y_col))
    fig.update_layout(
        title=f"Relationship: <b>{safe_x}</b> vs <b>{safe_y}</b>",
        xaxis_title=str(x_col),
        yaxis_title=str(y_col)
    )
    return style_plotly_fig(fig)


def create_bar_count_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str | None = None,
    agg_func: str = "Count",
    hue_col: str | None = None,
    orientation: str = "v",
) -> go.Figure:
    df_sample, is_sampled, _ = sample_for_visualization(df)
    # Defend against unhashable elements (e.g. lists/dicts in cells)
    plot_df = _cap_hue_column(df_sample, hue_col)
    if x_col in plot_df.columns:
        if plot_df[x_col].apply(lambda v: isinstance(v, (list, dict, set))).any():
            plot_df = plot_df.copy()
            plot_df[x_col] = plot_df[x_col].astype(str)
        # Cap high-cardinality x_col to top 30 categories plus "Other"
        x_counts = plot_df[x_col].value_counts(dropna=False)
        if len(x_counts) > 30:
            top_cats = set(x_counts.head(30).index)
            if plot_df is df:
                plot_df = df.copy()
            plot_df[x_col] = plot_df[x_col].apply(lambda v: v if v in top_cats else "Other")

    if y_col and agg_func != "Count":
        group_cols = [x_col] if (not hue_col or hue_col == x_col) else [x_col, hue_col]
        val_col = f"{y_col}_{agg_func.lower()}" if y_col in group_cols else y_col
        grouped = plot_df.groupby(group_cols, dropna=False).agg(**{val_col: (y_col, agg_func.lower())}).reset_index()
        fig = px.bar(
            grouped,
            x=x_col if orientation == "v" else val_col,
            y=val_col if orientation == "v" else x_col,
            color=hue_col,
            color_discrete_sequence=[chart_palette()[0]] if not hue_col else None,
            orientation=orientation,
            template=plot_template()
        )
        fig.update_layout(title=f"<b>{agg_func} of {y_col}</b> by {x_col}" + (f" and {hue_col}" if hue_col and hue_col != x_col else ""))
    else:
        fig = px.histogram(
            plot_df,
            x=x_col if orientation == "v" else None,
            y=x_col if orientation == "h" else None,
            color=hue_col,
            color_discrete_sequence=[chart_palette()[0]] if not hue_col else None,
            orientation=orientation,
            template=plot_template()
        )
        fig.update_layout(
            title=f"Frequency Count of <b>{x_col}</b>" + (f" by {hue_col}" if hue_col else ""),
            xaxis_title=x_col if orientation == "v" else "Count",
            yaxis_title="Count" if orientation == "v" else x_col
        )
    if orientation == "v":
        fig.update_xaxes(type="category")
    else:
        fig.update_yaxes(type="category")
    return style_plotly_fig(fig)


def downsample_timeseries(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_points: int = 25000,
    hue_col: str | None = None,
) -> pd.DataFrame:
    """Min-max envelope decimation for time series data.

    Enforces chronological ordering before decimation (datetime coercion -> chronological sort -> decimation).
    Divides the chronological series into N/4 buckets and extracts four representative points per bucket:
    bucket start, bucket end, local minimum, and local maximum. This preserves local extrema and temporal
    envelope boundaries significantly better than naive random sampling or uniform decimation, while bounding
    render points for interactive visualization. Does not guarantee zero aliasing for frequencies above
    the Nyquist limit of the bucket sampling rate.
    """
    if df is None or len(df) <= max_points:
        return df

    def _downsample_single_series(sub_df: pd.DataFrame, target_n: int) -> pd.DataFrame:
        if len(sub_df) <= target_n:
            return sub_df

        if pd.api.types.is_datetime64_any_dtype(sub_df[x_col]):
            s_df = sub_df.sort_values(by=x_col).reset_index(drop=True)
        else:
            try:
                converted_dt = pd.to_datetime(sub_df[x_col], errors="coerce")
                non_null_count = sub_df[x_col].dropna().shape[0]
                if non_null_count > 0 and (converted_dt.notna().sum() / non_null_count) >= 0.8:
                    temp_sub = sub_df.copy()
                    temp_sub["_dt_sort_key"] = converted_dt
                    s_df = temp_sub.sort_values(by="_dt_sort_key").drop(columns=["_dt_sort_key"]).reset_index(drop=True)
                else:
                    s_df = sub_df.sort_values(by=x_col).reset_index(drop=True)
            except Exception:
                s_df = sub_df.sort_values(by=x_col).reset_index(drop=True)

        num_buckets = max(1, target_n // 4)
        bucket_size = len(s_df) / num_buckets

        selected_indices = set()
        y_vals = pd.to_numeric(s_df[y_col], errors="coerce").fillna(0).to_numpy()

        for b in range(num_buckets):
            start_idx = int(b * bucket_size)
            end_idx = int((b + 1) * bucket_size) if b < num_buckets - 1 else len(s_df)
            if start_idx >= end_idx:
                continue

            selected_indices.add(start_idx)
            selected_indices.add(end_idx - 1)

            bucket_slice = y_vals[start_idx:end_idx]
            if len(bucket_slice) > 0:
                min_offset = int(np.argmin(bucket_slice))
                max_offset = int(np.argmax(bucket_slice))
                selected_indices.add(start_idx + min_offset)
                selected_indices.add(start_idx + max_offset)

        return s_df.iloc[sorted(selected_indices)].reset_index(drop=True)

    if hue_col and hue_col in df.columns:
        groups = []
        unique_hues = df[hue_col].dropna().unique()
        if len(unique_hues) > 0:
            points_per_group = max(4, max_points // len(unique_hues))
            for _, grp in df.groupby(hue_col):
                groups.append(_downsample_single_series(grp, points_per_group))
            res = pd.concat(groups, ignore_index=True)
            if len(res) > max_points:
                res = res.iloc[:max_points]
            return res

    return _downsample_single_series(df, max_points)


def create_line_chart(
    df: pd.DataFrame, x_col: str, y_col: str, hue_col: str | None = None, markers: bool = True
) -> go.Figure:
    if df is None or df.empty:
        return style_plotly_fig(px.line(title="Empty Dataset", template=plot_template()))
    sorted_df = _cap_hue_column(df, hue_col)
    if x_col in sorted_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sorted_df[x_col]):
            sorted_df = sorted_df.sort_values(by=x_col)
        else:
            # Check if date-like strings should be chronologically ordered
            try:
                converted_dt = pd.to_datetime(sorted_df[x_col], errors="coerce")
                non_null_count = sorted_df[x_col].dropna().shape[0]
                if non_null_count > 0 and (converted_dt.notna().sum() / non_null_count) >= 0.8:
                    temp_df = sorted_df.copy()
                    temp_df["_sort_key_dt"] = converted_dt
                    sorted_df = temp_df.sort_values(by="_sort_key_dt").drop(columns=["_sort_key_dt"])
                else:
                    sorted_df = sorted_df.sort_values(by=x_col)
            except Exception:
                sorted_df = sorted_df.sort_values(by=x_col)

    if len(sorted_df) > 25_000 and y_col in sorted_df.columns and x_col in sorted_df.columns:
        sorted_df = downsample_timeseries(sorted_df, x_col=x_col, y_col=y_col, max_points=25_000, hue_col=hue_col)

    fig = px.line(
        sorted_df,
        x=x_col,
        y=y_col,
        color=hue_col,
        markers=markers,
        template=plot_template()
    )
    safe_x = escape(str(x_col))
    safe_y = escape(str(y_col))
    fig.update_layout(
        title=f"Trend: <b>{safe_y}</b> over <b>{safe_x}</b>",
        xaxis_title=str(x_col),
        yaxis_title=str(y_col)
    )
    return style_plotly_fig(fig)


def create_pie_treemap_plot(
    df: pd.DataFrame, names_col: str, values_col: str | None = None, plot_type: str = "Pie"
) -> go.Figure:
    df_sample, _, _ = sample_for_visualization(df)
    plot_df = df_sample
    # Defend against unhashable elements in names_col
    if names_col in plot_df.columns and plot_df[names_col].apply(lambda v: isinstance(v, (list, dict, set))).any():
        plot_df = plot_df.copy()
        plot_df[names_col] = plot_df[names_col].astype(str)

    if values_col:
        numeric_vals = pd.to_numeric(plot_df[values_col], errors="coerce").dropna()
        if (numeric_vals < 0).any():
            raise ValueError(
                f"Column '{values_col}' contains negative values. Pie and Treemap charts "
                "require non-negative values to represent proportional parts of a whole."
            )
        value_name = f"{values_col}_sum" if names_col == values_col else values_col
        grouped = plot_df.groupby(names_col, dropna=False).agg(**{value_name: (values_col, "sum")}).reset_index()
        if (grouped[value_name] < 0).any():
            raise ValueError(
                f"Aggregated values for '{values_col}' contain negative sums, "
                "which cannot be represented in a part-to-whole chart."
            )
    else:
        grouped = plot_df[names_col].value_counts(dropna=False).reset_index()
        grouped.columns = [names_col, "count"]
        value_name = "count"

    if len(grouped) > 30:
        top30 = grouped.nlargest(30, value_name)
        remainder_sum = grouped.loc[~grouped.index.isin(top30.index), value_name].sum()
        if remainder_sum > 0:
            other_row = pd.DataFrame([{names_col: "Other", value_name: remainder_sum}])
            grouped = pd.concat([top30, other_row], ignore_index=True)
        else:
            grouped = top30

    safe_names = escape(str(names_col))
    if plot_type == "Treemap":
        fig = px.treemap(
            grouped,
            path=[names_col],
            values=value_name,
            template=plot_template()
        )
        fig.update_layout(title=f"Treemap Distribution of <b>{safe_names}</b>")
    elif plot_type == "Donut":
        fig = px.pie(
            grouped,
            names=names_col,
            values=value_name,
            hole=0.45,
            template=plot_template()
        )
        fig.update_layout(title=f"Donut Chart of <b>{safe_names}</b>")
    else:
        fig = px.pie(
            grouped,
            names=names_col,
            values=value_name,
            template=plot_template()
        )
        fig.update_layout(title=f"Pie Chart of <b>{safe_names}</b>")
    return style_plotly_fig(fig)


def create_correlation_heatmap(df: pd.DataFrame, colorscale: str = "RdBu_r") -> go.Figure | None:
    sample_df, is_sampled, _ = sample_for_analysis(df, max_rows=50_000)
    num_df = sample_df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return None

    # Cap correlation matrix to top 30 numeric columns with highest variance to prevent massive matrix hangs
    MAX_HEATMAP_COLS = 30
    title_suffix = ""
    if num_df.shape[1] > MAX_HEATMAP_COLS:
        variances = num_df.var().sort_values(ascending=False)
        top_cols = variances.head(MAX_HEATMAP_COLS).index.tolist()
        num_df = num_df[top_cols]
        title_suffix = f" (top {MAX_HEATMAP_COLS} numeric columns by variance)"

    corr = num_df.corr().round(3)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont={"size": 11}
        )
    )
    fig.update_layout(
        title=f"Interactive Correlation Matrix Heatmap{title_suffix}",
        template=plot_template(),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed"
    )
    return style_plotly_fig(fig)

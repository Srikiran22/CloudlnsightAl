import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from Utils.theme import plot_template


def create_histogram_plot(df, x_col, hue_col=None, nbins=30, marginal="box",
                          color_discrete_sequence=None):
    fig = px.histogram(
        df,
        x=x_col,
        color=hue_col,
        nbins=nbins,
        marginal=marginal,
        barmode="overlay" if hue_col else "relative",
        opacity=0.75,
        color_discrete_sequence=color_discrete_sequence or px.colors.qualitative.Plotly,
        template=plot_template()
    )
    fig.update_layout(
        title=f"Distribution of <b>{x_col}</b>" + (f" by {hue_col}" if hue_col else ""),
        xaxis_title=x_col,
        yaxis_title="Count / Frequency",
        bargap=0.05
    )
    return fig


def create_box_violin_plot(df, y_col, x_col=None, hue_col=None, plot_type="Box",
                           points="outliers"):
    if plot_type == "Violin":
        fig = px.violin(
            df,
            y=y_col,
            x=x_col,
            color=hue_col or x_col,
            box=True,
            points=points,
            template=plot_template()
        )
        fig.update_layout(title=f"Violin Plot of <b>{y_col}</b>" + (f" across {x_col}" if x_col else ""))
    else:
        fig = px.box(
            df,
            y=y_col,
            x=x_col,
            color=hue_col or x_col,
            points=points,
            notched=False,
            template=plot_template()
        )
        fig.update_layout(title=f"Box Plot of <b>{y_col}</b>" + (f" across {x_col}" if x_col else ""))

    fig.update_layout(yaxis_title=y_col, xaxis_title=x_col or "")
    return fig


def create_scatter_plot(df, x_col, y_col, hue_col=None, size_col=None,
                        add_trendline=False):
    # OLS only makes sense when both axes are numeric and finite
    can_fit_trendline = (
        pd.api.types.is_numeric_dtype(df[x_col])
        and pd.api.types.is_numeric_dtype(df[y_col])
    )
    plot_df = df
    if add_trendline and can_fit_trendline:
        try:
            import statsmodels.api as _sm
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False

        if has_statsmodels:
            import numpy as np
            x_num = pd.to_numeric(df[x_col], errors="coerce")
            y_num = pd.to_numeric(df[y_col], errors="coerce")
            finite_mask = np.isfinite(x_num) & np.isfinite(y_num)
            if finite_mask.sum() >= 2:
                plot_df = df[finite_mask]
                trendline = "ols"
            else:
                trendline = None
        else:
            trendline = None
    else:
        trendline = None

    if size_col:
        import numpy as np
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
            df,
            x=x_col,
            y=y_col,
            color=hue_col,
            size=size_col,
            trendline=None,
            opacity=0.8,
            template=plot_template(),
            hover_data=df.columns[:5].tolist()
        )
    fig.update_layout(
        title=f"Relationship: <b>{x_col}</b> vs <b>{y_col}</b>",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    return fig


def create_bar_count_plot(df, x_col, y_col=None, agg_func="Count", hue_col=None,
                          orientation="v"):
    # Defend against unhashable elements (e.g. lists/dicts in cells)
    if x_col in df.columns and df[x_col].apply(lambda v: isinstance(v, (list, dict, set))).any():
        df = df.copy()
        df[x_col] = df[x_col].astype(str)

    if y_col and agg_func != "Count":
        group_cols = [x_col] if (not hue_col or hue_col == x_col) else [x_col, hue_col]
        val_col = f"{y_col}_{agg_func.lower()}" if y_col in group_cols else y_col
        grouped = df.groupby(group_cols, dropna=False).agg(**{val_col: (y_col, agg_func.lower())}).reset_index()
        fig = px.bar(
            grouped,
            x=x_col if orientation == "v" else val_col,
            y=val_col if orientation == "v" else x_col,
            color=hue_col,
            color_discrete_sequence=[px.colors.qualitative.Plotly[0]] if not hue_col else None,
            orientation=orientation,
            template=plot_template()
        )
        fig.update_layout(title=f"<b>{agg_func} of {y_col}</b> by {x_col}" + (f" and {hue_col}" if hue_col and hue_col != x_col else ""))
    else:
        fig = px.histogram(
            df,
            x=x_col if orientation == "v" else None,
            y=x_col if orientation == "h" else None,
            color=hue_col,
            color_discrete_sequence=[px.colors.qualitative.Plotly[0]] if not hue_col else None,
            orientation=orientation,
            template=plot_template()
        )
        fig.update_layout(
            title=f"Frequency Count of <b>{x_col}</b>" + (f" by {hue_col}" if hue_col else ""),
            xaxis_title=x_col if orientation == "v" else "Count",
            yaxis_title="Count" if orientation == "v" else x_col
        )
    return fig


def create_line_chart(df, x_col, y_col, hue_col=None, markers=True):
    if x_col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            sorted_df = df.sort_values(by=x_col)
        else:
            # Check if date-like strings should be chronologically ordered
            try:
                converted_dt = pd.to_datetime(df[x_col], errors="coerce")
                non_null_count = df[x_col].dropna().shape[0]
                if non_null_count > 0 and (converted_dt.notna().sum() / non_null_count) >= 0.8:
                    temp_df = df.copy()
                    temp_df["_sort_key_dt"] = converted_dt
                    sorted_df = temp_df.sort_values(by="_sort_key_dt").drop(columns=["_sort_key_dt"])
                else:
                    sorted_df = df.sort_values(by=x_col)
            except Exception:
                sorted_df = df.sort_values(by=x_col)
    else:
        sorted_df = df

    fig = px.line(
        sorted_df,
        x=x_col,
        y=y_col,
        color=hue_col,
        markers=markers,
        template=plot_template()
    )
    fig.update_layout(
        title=f"Trend: <b>{y_col}</b> over <b>{x_col}</b>",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    return fig


def create_pie_treemap_plot(df, names_col, values_col=None, plot_type="Pie"):
    # Defend against unhashable elements in names_col
    if names_col in df.columns and df[names_col].apply(lambda v: isinstance(v, (list, dict, set))).any():
        df = df.copy()
        df[names_col] = df[names_col].astype(str)

    if values_col:
        numeric_vals = pd.to_numeric(df[values_col], errors="coerce").dropna()
        if (numeric_vals < 0).any():
            raise ValueError(
                f"Column '{values_col}' contains negative values. Pie and Treemap charts "
                "require non-negative values to represent proportional parts of a whole."
            )
        value_name = f"{values_col}_sum" if names_col == values_col else values_col
        grouped = df.groupby(names_col, dropna=False).agg(**{value_name: (values_col, "sum")}).reset_index()
        if (grouped[value_name] < 0).any():
            raise ValueError(
                f"Aggregated values for '{values_col}' contain negative sums, "
                "which cannot be represented in a part-to-whole chart."
            )
    else:
        grouped = df[names_col].value_counts(dropna=False).reset_index()
        grouped.columns = [names_col, "count"]
        value_name = "count"

    if len(grouped) > 30:
        grouped = grouped.nlargest(30, value_name)

    if plot_type == "Treemap":
        fig = px.treemap(
            grouped,
            path=[names_col],
            values=value_name,
            template=plot_template()
        )
        fig.update_layout(title=f"Treemap Distribution of <b>{names_col}</b>")
    elif plot_type == "Donut":
        fig = px.pie(
            grouped,
            names=names_col,
            values=value_name,
            hole=0.45,
            template=plot_template()
        )
        fig.update_layout(title=f"Donut Chart of <b>{names_col}</b>")
    else:
        fig = px.pie(
            grouped,
            names=names_col,
            values=value_name,
            template=plot_template()
        )
        fig.update_layout(title=f"Pie Chart of <b>{names_col}</b>")
    return fig


def create_correlation_heatmap(df, colorscale="RdBu_r"):
    num_df = df.select_dtypes(include="number")
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
    return fig

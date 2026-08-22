import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
        template="plotly_white"
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
            template="plotly_white"
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
            template="plotly_white"
        )
        fig.update_layout(title=f"Box Plot of <b>{y_col}</b>" + (f" across {x_col}" if x_col else ""))

    fig.update_layout(yaxis_title=y_col, xaxis_title=x_col or "")
    return fig


def create_scatter_plot(df, x_col, y_col, hue_col=None, size_col=None,
                        add_trendline=False):
    # OLS only makes sense when both axes are numeric
    can_fit_trendline = (
        pd.api.types.is_numeric_dtype(df[x_col])
        and pd.api.types.is_numeric_dtype(df[y_col])
    )
    trendline = "ols" if add_trendline and can_fit_trendline else None

    if size_col:
        size_values = pd.to_numeric(df[size_col], errors="coerce").dropna()
        if size_values.empty or (size_values < 0).any():
            size_col = None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=hue_col,
        size=size_col,
        trendline=trendline,
        opacity=0.8,
        template="plotly_white",
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
    if y_col and agg_func != "Count":
        grouped = df.groupby(x_col, dropna=False)[y_col].agg(agg_func.lower()).reset_index()
        fig = px.bar(
            grouped,
            x=x_col if orientation == "v" else y_col,
            y=y_col if orientation == "v" else x_col,
            color=x_col,
            orientation=orientation,
            template="plotly_white"
        )
        fig.update_layout(title=f"<b>{agg_func} of {y_col}</b> by {x_col}")
    else:
        fig = px.histogram(
            df,
            x=x_col if orientation == "v" else None,
            y=x_col if orientation == "h" else None,
            color=hue_col or x_col,
            orientation=orientation,
            template="plotly_white"
        )
        fig.update_layout(
            title=f"Frequency Count of <b>{x_col}</b>",
            xaxis_title=x_col if orientation == "v" else "Count",
            yaxis_title="Count" if orientation == "v" else x_col
        )
    return fig


def create_line_chart(df, x_col, y_col, hue_col=None, markers=True):
    sorted_df = df.sort_values(by=x_col) if x_col in df.columns else df
    fig = px.line(
        sorted_df,
        x=x_col,
        y=y_col,
        color=hue_col,
        markers=markers,
        template="plotly_white"
    )
    fig.update_layout(
        title=f"Trend: <b>{y_col}</b> over <b>{x_col}</b>",
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    return fig


def create_pie_treemap_plot(df, names_col, values_col=None, plot_type="Pie"):
    if values_col:
        grouped = df.groupby(names_col, dropna=False)[values_col].sum().reset_index()
        value_name = values_col
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
            template="plotly_white"
        )
        fig.update_layout(title=f"Treemap Distribution of <b>{names_col}</b>")
    elif plot_type == "Donut":
        fig = px.pie(
            grouped,
            names=names_col,
            values=value_name,
            hole=0.45,
            template="plotly_white"
        )
        fig.update_layout(title=f"Donut Chart of <b>{names_col}</b>")
    else:
        fig = px.pie(
            grouped,
            names=names_col,
            values=value_name,
            template="plotly_white"
        )
        fig.update_layout(title=f"Pie Chart of <b>{names_col}</b>")
    return fig


def create_correlation_heatmap(df, colorscale="RdBu_r"):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return None

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
        title="Interactive Correlation Matrix Heatmap",
        template="plotly_white",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed"
    )
    return fig

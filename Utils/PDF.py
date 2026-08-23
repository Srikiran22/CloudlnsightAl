import io
import datetime

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape

from Utils.logsys import get_logger
from Utils.quality import quality_metrics


logger = get_logger("PDF")

USABLE_WIDTH = 540  # letter minus the 36pt page margins

SECTION_COLORS = {
    "blue": "#2563EB",
    "teal": "#0D9488",
    "purple": "#7C3AED",
    "orange": "#EA580C",
    "rose": "#E11D48",
    "indigo": "#4F46E5",
    "slate": "#475569",
}


def _fmt(value, decimals=2):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _style_table(data, header_color, col_widths=None):
    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _safe_chart(render, *args, default=None, **kwargs):
    """Run one chart renderer; a single chart failure skips that chart only."""
    try:
        return render(*args, **kwargs)
    except Exception as error:
        logger.warning("report chart skipped (%s): %s", getattr(render, "__name__", "chart"), error)
        return default


def _render_histograms(df, max_charts=4):
    plt = _matplotlib()
    if plt is None:
        return []

    numeric_cols = df.select_dtypes(include="number").columns[:max_charts]
    images = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(4, 2.6), dpi=100)
        ax.hist(series, bins=24, color="#2563EB", edgecolor="white", linewidth=0.4)
        ax.set_title(str(col)[:40], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        images.append((str(col), buffer))
    return images


def _render_correlation_heatmap(df):
    plt = _matplotlib()
    if plt is None:
        return None

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) < 2:
        return None
    corr = num_df.corr(numeric_only=True)
    if corr.dropna(how="all").empty:
        return None

    if len(corr.columns) > 10:
        corr = corr.iloc[:10, :10]

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=100)
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels([str(c)[:18] for c in corr.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([str(c)[:18] for c in corr.columns], fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _render_boxplots(df, max_charts=4):
    plt = _matplotlib()
    if plt is None:
        return []

    numeric_cols = df.select_dtypes(include="number").columns[:max_charts]
    images = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(4, 2.2), dpi=100)
        ax.boxplot(series, vert=False, widths=0.55,
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#E11D48"})
        ax.set_title(str(col)[:40], fontsize=9)
        ax.tick_params(labelsize=7, left=False, labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        images.append((str(col), buffer))
    return images


def generate_pdf_report(
    df,
    dataset_name,
    report_title="Executive Data Analytics Report",
    author_name="CloudInsight AI",
    include_ai_insights=None,
    include_charts=True,
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=48,
        title=str(report_title),
        author=str(author_name),
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        fontSize=22, leading=26,
        textColor=colors.HexColor("#1E3A8A"), spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#6B7280"), spaceAfter=14
    )
    heading2_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=13, leading=17,
        textColor=colors.HexColor("#1E40AF"), spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=colors.HexColor("#1F2937")
    )
    ai_heading_style = ParagraphStyle(
        "AIHeading", parent=styles["Heading3"],
        fontSize=11, leading=15,
        textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=4
    )
    cell_style = ParagraphStyle(
        "Cell", parent=styles["Normal"],
        fontSize=7.5, leading=9.5, textColor=colors.HexColor("#1F2937")
    )
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontSize=8, leading=11, textColor=colors.HexColor("#6B7280"),
        spaceBefore=4
    )

    def footer(canvas, doc_ref):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.drawString(36, 24, f"{dataset_name} | {report_title}"[:110])
        canvas.drawRightString(576, 24, f"Page {doc_ref.page}")
        canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
        canvas.line(36, 34, 576, 34)
        canvas.restoreState()

    story = []

    story.append(Paragraph(f"☁️ {escape(str(report_title))}", title_style))
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    story.append(Paragraph(
        f"<b>Dataset:</b> {escape(str(dataset_name))} | "
        f"<b>Generated:</b> {now_str} | "
        f"<b>Prepared by:</b> {escape(str(author_name))}",
        subtitle_style
    ))

    rows, cols = df.shape
    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(exclude="number")
    quality = quality_metrics(df)
    dup_count = quality["duplicate_rows"]
    missing_count = quality["missing_cells"]
    completeness = quality["completeness"]
    uniqueness = quality["uniqueness"]
    quality_index = quality["index"]
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # 1 -- summary
    story.append(Paragraph("1. High-Level Dataset Summary", heading2_style))
    summary_data = [
        ["Total Records (Rows)", f"{rows:,}", "Total Features (Cols)", f"{cols}"],
        ["Duplicate Rows", f"{dup_count:,}", "Total Missing Cells", f"{missing_count:,}"],
        ["Numeric Columns", f"{len(num_df.columns)}", "Categorical Columns", f"{len(cat_df.columns)}"],
        ["Memory Footprint", f"{memory_mb:.2f} MB", "Data Quality Index", f"{quality_index:.1f} / 100"],
    ]
    summary_table = Table(summary_data, colWidths=[150, 120, 150, 120])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Paragraph(
        "Quality Index blends column completeness and row uniqueness equally.",
        note_style
    ))
    story.append(Spacer(1, 10))

    # 2 -- per-column structure, every column listed
    story.append(Paragraph("2. Column Structure & Missing Value Breakdown", heading2_style))
    col_table_data = [["#", "Column Name", "Data Type", "Non-Null", "Missing", "Missing %", "Unique"]]
    for position, col in enumerate(df.columns, start=1):
        m_cnt = int(df[col].isnull().sum())
        m_pct = m_cnt / max(rows, 1) * 100
        col_table_data.append([
            str(position),
            Paragraph(escape(str(col)), cell_style),
            escape(str(df[col].dtype)),
            f"{int(df[col].notnull().sum()):,}",
            f"{m_cnt:,}",
            f"{m_pct:.1f}%",
            f"{int(df[col].nunique(dropna=True)):,}",
        ])
    story.append(_style_table(
        col_table_data, SECTION_COLORS["blue"],
        col_widths=[22, 168, 70, 70, 60, 65, 85]
    ))
    story.append(Spacer(1, 12))

    # 3 -- extended numeric stats
    if not num_df.empty:
        story.append(Paragraph("3. Numerical Descriptive Statistics (Extended)", heading2_style))
        stats_table_data = [[
            "Feature", "Count", "Mean", "Std Dev", "Min",
            "Q1 (25%)", "Median", "Q3 (75%)", "Max", "Skew", "Kurtosis"
        ]]
        for col in num_df.columns:
            s = df[col].dropna()
            if s.empty:
                continue
            stats_table_data.append([
                Paragraph(escape(str(col)), cell_style),
                _fmt(s.count(), 0),
                _fmt(s.mean()),
                _fmt(s.std()),
                _fmt(s.min()),
                _fmt(s.quantile(0.25)),
                _fmt(s.median()),
                _fmt(s.quantile(0.75)),
                _fmt(s.max()),
                _fmt(s.skew()),
                _fmt(s.kurtosis()),
            ])
        story.append(_style_table(
            stats_table_data, SECTION_COLORS["teal"],
            col_widths=[95, 42, 52, 52, 50, 48, 50, 48, 50, 26.5, 26.5]
        ))
        story.append(Spacer(1, 12))

    # 4 -- tukey outliers
    if not num_df.empty:
        story.append(Paragraph("4. Outlier Analysis (Tukey IQR Method)", heading2_style))
        outlier_data = [["Column", "Q1 (25%)", "Q3 (75%)", "IQR", "Lower Bound", "Upper Bound", "Outliers", "Outlier %"]]
        for col in num_df.columns:
            s = df[col].dropna()
            if s.empty:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr
            o_cnt = int(((s < lb) | (s > ub)).sum()) if iqr > 0 else 0
            pct = o_cnt / len(s) * 100
            outlier_data.append([
                Paragraph(escape(str(col)), cell_style),
                _fmt(q1), _fmt(q3), _fmt(iqr), _fmt(lb), _fmt(ub),
                f"{o_cnt:,}", f"{pct:.1f}%"
            ])
        story.append(_style_table(
            outlier_data, SECTION_COLORS["purple"],
            col_widths=[130, 55, 55, 55, 62, 62, 58, 63]
        ))
        story.append(Spacer(1, 12))

    # 5 -- categoricals
    if not cat_df.empty:
        story.append(Paragraph("5. Categorical Column Analysis", heading2_style))
        cat_data = [["Column", "Unique", "Top Value", "Top Frequency", "Top Share", "Missing"]]
        for col in cat_df.columns:
            s = df[col].dropna()
            unique_n = int(s.nunique())
            if s.empty:
                top_value, top_freq, top_share = "-", 0, 0.0
            else:
                top_value = str(s.mode().iloc[0])[:38]
                top_freq = int(s.value_counts().iloc[0])
                top_share = top_freq / len(s) * 100
            cat_data.append([
                Paragraph(escape(str(col)), cell_style),
                f"{unique_n:,}",
                Paragraph(escape(top_value), cell_style),
                f"{top_freq:,}",
                f"{top_share:.1f}%",
                f"{int(df[col].isnull().sum()):,}",
            ])
        story.append(_style_table(
            cat_data, SECTION_COLORS["orange"],
            col_widths=[140, 60, 160, 75, 60, 45]
        ))
        story.append(Spacer(1, 12))

    # 6 -- strongest correlations
    if len(num_df.columns) >= 2:
        corr_matrix = num_df.corr(numeric_only=True)
        pairs = []
        columns = list(corr_matrix.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                r = corr_matrix.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.3:
                    pairs.append((columns[i], columns[j], float(r)))
        pairs.sort(key=lambda item: abs(item[2]), reverse=True)

        if pairs:
            story.append(Paragraph("6. Correlation Insights (|r| ≥ 0.30)", heading2_style))
            corr_data = [["Feature A", "Feature B", "Pearson r", "Strength", "Direction"]]
            for name_a, name_b, r in pairs[:12]:
                strength = "Strong" if abs(r) >= 0.7 else ("Moderate" if abs(r) >= 0.4 else "Weak")
                direction = "Positive" if r > 0 else "Negative"
                corr_data.append([
                    Paragraph(escape(name_a), cell_style),
                    Paragraph(escape(name_b), cell_style),
                    f"{r:+.3f}", strength, direction
                ])
            story.append(_style_table(
                corr_data, SECTION_COLORS["rose"],
                col_widths=[165, 165, 70, 70, 70]
            ))
            story.append(Spacer(1, 12))

    # 7 -- quality flags per column
    story.append(Paragraph("7. Data Quality Assessment", heading2_style))
    quality_data = [["Column", "Completeness %", "Uniqueness %", "Flags"]]
    flags_by_column = {}
    for col in df.columns:
        s = df[col]
        col_missing_pct = s.isnull().mean() * 100
        flags = []
        if col_missing_pct >= 40:
            flags.append("High missingness")
        if s.nunique(dropna=True) <= 1:
            flags.append("Constant")
        if s.dtype == object:
            n_unique = s.nunique(dropna=True)
            if n_unique / max(len(s.dropna()), 1) > 0.95 and n_unique > 20:
                flags.append("Possible ID/high-cardinality")
        if pd.api.types.is_numeric_dtype(s) and not s.dropna().empty:
            q1, q3 = s.dropna().quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_pct = (((s.dropna() < q1 - 1.5 * iqr) | (s.dropna() > q3 + 1.5 * iqr)).mean()) * 100
                if outlier_pct >= 10:
                    flags.append("Heavy outliers")
        flags_by_column[col] = ", ".join(flags) if flags else "OK"
        completeness_col = 100 - col_missing_pct
        uniqueness_col = s.nunique(dropna=True) / max(len(s.dropna()), 1) * 100
        quality_data.append([
            Paragraph(escape(str(col)), cell_style),
            f"{completeness_col:.1f}%",
            f"{uniqueness_col:.1f}%",
            Paragraph(escape(flags_by_column[col]), cell_style),
        ])
    story.append(_style_table(
        quality_data, SECTION_COLORS["indigo"],
        col_widths=[150, 90, 90, 210]
    ))
    flagged = [c for c, f in flags_by_column.items() if f != "OK"]
    if flagged:
        story.append(Paragraph(
            f"⚠️ Columns needing review: {escape(', '.join(str(c) for c in flagged[:15]))}"
            + (" ..." if len(flagged) > 15 else ""),
            note_style
        ))
    story.append(Spacer(1, 12))

    # 8 -- sample records
    if cols == 0:
        story.append(Paragraph("8. Sample Records", heading2_style))
        story.append(Paragraph(
            "The dataset has no columns, so no sample records can be shown.",
            note_style
        ))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("8. Sample Records (First 8 Rows)", heading2_style))
        sample_cols = list(df.columns[:8])
        sample_data = [[escape(str(c)) for c in sample_cols]]
        for _, row in df[sample_cols].head(8).iterrows():
            sample_data.append([
                Paragraph(escape(str(v)[:28] if pd.notna(v) else "-"), cell_style)
                for v in row
            ])
        sample_width = USABLE_WIDTH / len(sample_cols)
        story.append(_style_table(sample_data, SECTION_COLORS["slate"], col_widths=[sample_width] * len(sample_cols)))
        if cols > len(sample_cols):
            story.append(Paragraph(
                f"Showing first {len(sample_cols)} of {cols} columns.", note_style
            ))
        story.append(Spacer(1, 12))

    # 9 -- optional matplotlib charts
    charts_shown = False
    if include_charts and not num_df.empty:
        chart_images = _safe_chart(_render_histograms, df, max_charts=4, default=[])
        heatmap_buffer = _safe_chart(_render_correlation_heatmap, df)
        box_images = _safe_chart(_render_boxplots, df, max_charts=4, default=[])

        if chart_images or heatmap_buffer or box_images:
            charts_shown = True
            story.append(Paragraph("9. Distribution & Relationship Charts", heading2_style))

            def _image_grid(images, image_width=260, image_height=150):
                grid_rows = []
                current_row = []
                for _, img_buffer in images:
                    current_row.append(Image(img_buffer, width=image_width, height=image_height))
                    if len(current_row) == 2:
                        grid_rows.append(current_row)
                        current_row = []
                if current_row:
                    current_row.extend([None] * (2 - len(current_row)))
                    grid_rows.append(current_row)
                for row_cells in grid_rows:
                    cells = [cell if cell is not None else "" for cell in row_cells]
                    row_table = Table([cells], colWidths=[270] * len(cells))
                    row_table.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    story.append(row_table)

            if chart_images:
                story.append(Paragraph("Distributions", body_style))
                _image_grid(chart_images)
            if box_images:
                story.append(Spacer(1, 6))
                story.append(Paragraph("Box Plots (Outlier View)", body_style))
                _image_grid(box_images, image_width=250, image_height=130)
            if heatmap_buffer is not None:
                story.append(Spacer(1, 6))
                story.append(Paragraph("Correlation Heatmap", body_style))
                heat_table = Table([[Image(heatmap_buffer, width=330, height=280)]], colWidths=[540])
                heat_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(heat_table)
            story.append(Spacer(1, 10))

    # 10 -- optional AI insights (markdown-lite rendering); number follows
    # whether a charts section actually exists, so reports never skip a digit
    if include_ai_insights:
        section_no = "10" if charts_shown else "9"
        story.append(Paragraph(f"{section_no}. AI Executive Insights & Strategy", heading2_style))
        for line in include_ai_insights.split("\n"):
            clean_line = line.strip()
            if not clean_line:
                continue
            stripped = clean_line.lstrip("#").strip()
            bold_match = stripped.startswith("**") and "**" in stripped[2:]
            if clean_line.startswith("#") or bold_match:
                text = stripped.replace("**", "").replace("#", "").strip()
                story.append(Paragraph(escape(text), ai_heading_style))
            elif stripped.startswith(("-", "*")):
                story.append(Paragraph(f"• {escape(stripped[1:].strip())}", body_style))
                story.append(Spacer(1, 2))
            else:
                story.append(Paragraph(escape(stripped.replace("*", "")), body_style))
                story.append(Spacer(1, 3))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer.getvalue()

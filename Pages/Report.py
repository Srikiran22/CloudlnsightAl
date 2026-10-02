import datetime
import json
import os
import streamlit as st

from Utils.PDF import generate_pdf_report, validate_report_template
from Utils.paths import (
    REPORTS_DIR, REPORT_TEMPLATES_DIR, atomic_write, get_dataset_row_count, get_unique_filename,
    list_dataset_files, resolve_dataset_path, safe_stem,
)
from Utils.privacy import detect_sensitive_columns
from Utils.dataset_ui import (
    dataframe_fingerprint, dataset_fingerprint, load_dataset_cached, render_sidebar,
    results_match_active, select_working_dataset,
)

st.title("PDF report")
st.markdown("Generate a detailed analytics report for the active dataset, save presets, or batch-generate for every dataset.")

df, selected_file = select_working_dataset("Select Dataset for PDF Report:")
render_sidebar()
st.caption(f"Generating Report for: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} cols)")

# templates: save / load report configurations
def _list_templates() -> list:
    if not REPORT_TEMPLATES_DIR.exists():
        return []
    return sorted(p.stem for p in REPORT_TEMPLATES_DIR.glob("*.json"))


st.subheader("Report settings")

saved_templates = _list_templates()
applied_template = None
if saved_templates:
    chosen_template = st.selectbox(
        "Load template:",
        ["(none)"] + saved_templates,
        help="Templates store title, author, and chart/AI settings."
    )
    if chosen_template != "(none)":
        try:
            raw_tpl = json.loads((REPORT_TEMPLATES_DIR / f"{chosen_template}.json").read_text(encoding="utf-8"))
            applied_template = validate_report_template(raw_tpl)
        except Exception as e:
            st.error(f"Could not read template: {e}")

def _parse_bool(val, default=True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
    return default


default_title = str((applied_template or {}).get("title") or "CloudInsight AI Executive Analytics Report")
default_author = str((applied_template or {}).get("author") or "CloudInsight AI Platform")
default_charts = _parse_bool((applied_template or {}).get("include_charts"), default=True)

c1, c2 = st.columns(2)
with c1:
    rep_title = st.text_input("Report Title:", value=default_title)
with c2:
    author = st.text_input("Prepared By:", value=default_author)

is_file_backed = False
try:
    is_file_backed = resolve_dataset_path(selected_file).is_file()
except Exception:
    is_file_backed = False
curr_fp = dataset_fingerprint(selected_file) if is_file_backed else dataframe_fingerprint(df)

latest_ai = st.session_state.get("latest_ai_insights")
ai_saved = None
if (
    latest_ai
    and latest_ai.get("dataset_name") == selected_file
    and latest_ai.get("dataset_fingerprint") == curr_fp
):
    ai_saved = latest_ai.get("text")
elif st.session_state.get(f"insights_{selected_file}_{curr_fp}"):
    ai_saved = st.session_state.get(f"insights_{selected_file}_{curr_fp}")

template_ai_pref = _parse_bool((applied_template or {}).get("include_ai"), default=True)
include_ai = False
if ai_saved:
    include_ai = st.checkbox("Include Gemini AI Executive Insights section in PDF", value=template_ai_pref)
    if latest_ai and latest_ai.get("model_name"):
        st.caption(f"AI insights model: `{latest_ai.get('model_name')}`")

include_charts = st.checkbox(
    "Include charts: histograms, box plots & correlation heatmap (requires matplotlib)",
    value=default_charts,
    help="Skipped automatically if matplotlib is not installed."
)

sensitive_detected = list(detect_sensitive_columns(df).keys())
excluded_cols = st.multiselect(
    "Exclude columns from PDF statistics, correlations & charts (PII/confidential):",
    options=list(df.columns),
    default=[c for c in sensitive_detected if c in df.columns],
    help="Selected columns will be redacted in summaries and omitted from correlations and visualization charts.",
)

template_name = st.text_input("Save current settings as template (name):", value="")
overwrite_template = st.checkbox("Overwrite existing template with same name", value=False)
if template_name and st.button("Save Template"):
    try:
        # sanitized so a crafted name cannot write outside templates/
        safe_template = safe_stem(template_name, fallback="")
        if not safe_template:
            raise ValueError(
                "Template name must contain letters, numbers, spaces, '-' or '_'."
            )
        template_file = REPORT_TEMPLATES_DIR / f"{safe_template}.json"
        if template_file.exists() and not overwrite_template:
            raise ValueError(
                f"Template `{safe_template}` already exists. Check 'Overwrite existing template' to replace it."
            )
        REPORT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "title": str(rep_title).strip() or default_title,
            "author": str(author).strip() or default_author,
            "include_charts": bool(include_charts),
            "include_ai": bool(include_ai),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write(
            template_file, json.dumps(config, indent=2), mode="w", encoding="utf-8"
        )
        st.success(f"Template `{safe_template}` saved.")
    except Exception as e:
        st.error(f"Template save failed: {str(e)}")

st.markdown("---")


def _build_report(dataset_name, dataframe, title, prepared_by, with_charts, ai_insights=None, source_rows=None, analyzed_rows=None, excluded_columns=None):
    pdf_bytes = generate_pdf_report(
        df=dataframe,
        dataset_name=dataset_name,
        report_title=title,
        author_name=prepared_by,
        include_ai_insights=ai_insights,
        include_charts=with_charts,
        source_rows=source_rows,
        analyzed_rows=analyzed_rows,
        excluded_columns=excluded_columns,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, _ = os.path.splitext(dataset_name)
    target_name = f"Report_{base_name}_{timestamp}.pdf"
    pdf_filename = get_unique_filename(target_name, directory=REPORTS_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(REPORTS_DIR / pdf_filename, pdf_bytes, mode="wb")
    return pdf_filename, pdf_bytes


if st.button("Generate PDF report", type="primary"):
    with st.spinner("Compiling PDF tables, statistics, and metadata..."):
        try:
            total_source = None
            if is_file_backed:
                try:
                    total_source = get_dataset_row_count(selected_file)
                except Exception:
                    total_source = None
            pdf_filename, pdf_bytes = _build_report(
                selected_file, df, rep_title, author, include_charts,
                ai_insights=(ai_saved if include_ai else None),
                source_rows=total_source or df.shape[0],
                analyzed_rows=df.shape[0],
                excluded_columns=excluded_cols,
            )
            st.session_state["last_pdf_report"] = {
                "dataset": selected_file,
                "dataset_fingerprint": curr_fp,
                "filename": pdf_filename,
                "bytes": pdf_bytes,
            }
            st.success(f"Report saved as `Reports/{pdf_filename}`.")
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")

pdf_result = st.session_state.get("last_pdf_report")
if pdf_result and results_match_active(
    {"dataset_name": pdf_result.get("dataset"),
     "dataset_fingerprint": pdf_result.get("dataset_fingerprint")},
    selected_file,
    df=df,
):
    st.download_button(
        label="Download PDF report",
        data=pdf_result["bytes"],
        file_name=pdf_result["filename"],
        mime="application/pdf"
    )

st.markdown("---")
st.subheader("Batch generation")
st.caption(f"Generates a PDF report for each of the {len(list_dataset_files())} datasets in the Datasets/ folder using the settings above.")

max_batch = st.number_input(
    "Row limit per dataset in batch mode:",
    min_value=100, max_value=500000, value=20000, step=1000,
    help="Caps rows loaded per dataset to keep batch runs fast."
)

if st.button("Generate reports for all datasets"):
    all_files = list_dataset_files()
    progress = st.progress(0.0)
    generated, failures = [], []

    for index, file_name in enumerate(all_files):
        try:
            batch_cap = int(max_batch)
            total_source_rows = get_dataset_row_count(file_name)
            batch_df = load_dataset_cached(file_name, max_rows=batch_cap)
            analyzed_rows = len(batch_df)
            if total_source_rows is None:
                total_source_rows = analyzed_rows

            batch_excluded = [c for c in excluded_cols if c in batch_df.columns]
            batch_sensitive = list(detect_sensitive_columns(batch_df).keys())
            for sc in batch_sensitive:
                if sc not in batch_excluded:
                    batch_excluded.append(sc)

            batch_ai = None
            if include_ai:
                if file_name == selected_file:
                    batch_ai = ai_saved
                else:
                    batch_fp = dataframe_fingerprint(batch_df)
                    batch_ai = st.session_state.get(f"insights_{file_name}_{batch_fp}")

            filename_out, _ = _build_report(
                file_name,
                batch_df,
                rep_title,
                author,
                include_charts,
                ai_insights=batch_ai,
                source_rows=total_source_rows,
                analyzed_rows=analyzed_rows,
                excluded_columns=batch_excluded,
            )
            generated.append(filename_out)
        except Exception as e:
            failures.append(f"{file_name}: {str(e)}")
        progress.progress((index + 1) / max(len(all_files), 1))

    progress.empty()
    if generated:
        st.success(f"Generated {len(generated)} report(s) in `Reports/`.")
        with st.expander("View generated files"):
            st.write("\n".join(f"- `{name}`" for name in generated))
    if failures:
        st.error(f"{len(failures)} dataset(s) failed:")
        with st.expander("View failures"):
            st.write("\n".join(f"- {item}" for item in failures))
    if not generated and not failures:
        st.info("No datasets found to process.")

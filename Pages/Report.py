import streamlit as st
import os
import json
import datetime

from Utils.PDF import generate_pdf_report
from Utils.paths import REPORTS_DIR, REPORT_TEMPLATES_DIR, list_dataset_files, read_dataset
from Utils.dataset_ui import render_sidebar, select_working_dataset, load_dataset_cached

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
            applied_template = json.loads((REPORT_TEMPLATES_DIR / f"{chosen_template}.json").read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not read template: {e}")

default_title = (applied_template or {}).get("title", "CloudInsight AI Executive Analytics Report")
default_author = (applied_template or {}).get("author", "CloudInsight AI Platform")
default_charts = bool((applied_template or {}).get("include_charts", True))

c1, c2 = st.columns(2)
with c1:
    rep_title = st.text_input("Report Title:", value=default_title)
with c2:
    author = st.text_input("Prepared By:", value=default_author)

ai_saved = st.session_state.get(f"insights_{selected_file}")
include_ai = False
if ai_saved:
    include_ai = st.checkbox("Include Gemini AI Executive Insights section in PDF", value=True)

include_charts = st.checkbox(
    "Include charts: histograms, box plots & correlation heatmap (requires matplotlib)",
    value=default_charts,
    help="Skipped automatically if matplotlib is not installed."
)

template_name = st.text_input("Save current settings as template (name):", value="")
if template_name and st.button("Save Template"):
    try:
        REPORT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "title": rep_title,
            "author": author,
            "include_charts": include_charts,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        (REPORT_TEMPLATES_DIR / f"{template_name}.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        st.success(f"Template `{template_name}` saved.")
    except Exception as e:
        st.error(f"Template save failed: {str(e)}")

st.markdown("---")


def _build_report(dataset_name, dataframe, title, prepared_by, with_charts, ai_insights=None):
    pdf_bytes = generate_pdf_report(
        df=dataframe,
        dataset_name=dataset_name,
        report_title=title,
        author_name=prepared_by,
        include_ai_insights=ai_insights,
        include_charts=with_charts,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, _ = os.path.splitext(dataset_name)
    pdf_filename = f"Report_{base_name}_{timestamp}.pdf"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / pdf_filename, "wb") as f:
        f.write(pdf_bytes)
    return pdf_filename, pdf_bytes


if st.button("Generate PDF report", type="primary"):
    with st.spinner("Compiling PDF tables, statistics, and metadata..."):
        try:
            pdf_filename, pdf_bytes = _build_report(
                selected_file, df, rep_title, author, include_charts,
                ai_insights=(ai_saved if include_ai else None),
            )
            st.session_state["last_pdf_report"] = {
                "dataset": selected_file,
                "filename": pdf_filename,
                "bytes": pdf_bytes,
            }
            st.success(f"Report saved as `Reports/{pdf_filename}`.")
        except Exception as e:
            st.error(f"PDF generation failed: {str(e)}")

pdf_result = st.session_state.get("last_pdf_report")
if pdf_result and pdf_result.get("dataset") == selected_file:
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
            batch_df = load_dataset_cached(file_name, max_rows=int(max_batch))
            filename_out, _ = _build_report(file_name, batch_df, rep_title, author, include_charts)
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

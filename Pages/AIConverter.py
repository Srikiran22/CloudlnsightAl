from pathlib import Path
import streamlit as st

from Utils.secrets import ask, value_of, keep_box, release
from Utils.Gemini import GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError
from Utils.AIConvert import convert_to_dataframe
from Utils.paths import (
    AIConversionRequired,
    DATASETS_DIR,
    atomic_write,
    get_unique_filename,
    read_tabular,
    SUPPORTED_DATASET_EXTENSIONS,
)
from Utils.dataset_ui import render_sidebar, set_active_dataset, invalidate_dataset_cache
from Utils.privacy import detect_sensitive_text
from Utils.theme import page_header, section_card_header
from Utils.logsys import get_logger

logger = get_logger("AIConverter")

page_header(
    "AI File Converter",
    "Extract structured CSV tables from unstructured documents (PDFs, text notes, logs, or pasted text) using Google Gemini."
)

render_sidebar()

with st.container(border=True):
    section_card_header("How it Works", "3-step automated data conversion pipeline")
    st.markdown(
        """
        1. **Upload or Paste** — Choose a PDF, text log, document, or paste raw unstructured notes.
        2. **Instruct & Extract** — Specify the schema or let Gemini automatically identify tabular records.
        3. **Save & Explore** — Preview the extracted table and save it directly into your active workspace.
        """
    )

with st.sidebar:
    st.markdown('<div class="ci-side-label">Gemini settings</div>', unsafe_allow_html=True)
    ask(
        "gemini",
        "Enter Google Gemini API Key:",
        help_text="Runtime key held in ephemeral memory only."
    )
    keep_box("gemini_keep")

    chosen_model = st.selectbox(
        "Gemini Model:",
        GEMINI_MODELS,
        index=GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL),
        key="converter_gemini_model"
    )

api_key = value_of("gemini")

with st.container(border=True):
    section_card_header("Step 1: Provide Unstructured Source Data", "Upload a document, paste raw text, or load sample log")

    c_key1, c_key2 = st.columns([1.2, 1.8])
    with c_key1:
        if api_key:
            st.success("Gemini API Key: Active in Session")
        else:
            st.warning("Gemini API Key: Required for extraction")
            ask(
                "gemini",
                "Enter Gemini API Key (or set in sidebar):",
                help_text="Key is held in volatile session memory only. Get a free key at https://aistudio.google.com/",
                key="inline_gemini_key_conv",
            )
            api_key = value_of("gemini")
    with c_key2:
        st.caption("*Extract columns directly from PDFs, text files, billing logs, and support transcripts.*")

    SAMPLE_LOG_TEXT = (
        "[2026-03-01 10:14:22] Host: srv-app-01 | Tier: Enterprise | Event: CPU_SPIKE | Usage: 94.2% | Action: Auto-scaled 2 pods\n"
        "[2026-03-01 10:15:08] Host: srv-db-04  | Tier: Enterprise | Event: IO_LATENCY | Latency_ms: 184 | Action: Query throttle enabled\n"
        "[2026-03-01 10:17:33] Host: srv-web-02 | Tier: Pro        | Event: 502_BAD_GATEWAY | Requests_affected: 42 | Action: Worker restarted\n"
        "[2026-03-01 10:19:45] Host: srv-auth-01| Tier: Starter    | Event: AUTH_FAILED | Fail_count: 15 | Action: IP rate-limited\n"
        "[2026-03-01 10:22:10] Host: srv-app-03 | Tier: Pro        | Event: MEMORY_LEAK | RAM_MB: 7680 | Action: Container recycled"
    )

    def _load_sample_log_cb():
        st.session_state["conv_input_mode"] = "Paste Raw Text"
        st.session_state["conv_text_area"] = SAMPLE_LOG_TEXT

    c_mode, c_btn = st.columns([1.8, 1.2])
    with c_mode:
        input_mode = st.radio(
            "Source Type:",
            ["Upload File (.pdf, .txt, .log, .md, .csv, .json)", "Paste Raw Text"],
            horizontal=True,
            key="conv_input_mode"
        )
    with c_btn:
        st.markdown('<div style="height: 1.6rem;"></div>', unsafe_allow_html=True)
        st.button(
            "Load sample text (Server Log)",
            key="btn_load_sample_log",
            use_container_width=True,
            on_click=_load_sample_log_cb
        )

    raw_text = ""
    source_filename = "unstructured_data.txt"

    if input_mode == "Upload File (.pdf, .txt, .log, .md, .csv, .json)":
        accepted_exts = sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)
        uploaded = st.file_uploader(
            "Choose an unstructured or semi-structured file:",
            type=accepted_exts,
            key="conv_file_uploader"
        )
        if uploaded:
            source_filename = uploaded.name
            try:
                # read_tabular extracts text or returns existing frame
                parsed_frame = read_tabular(uploaded, filename=uploaded.name)
                raw_text = parsed_frame.to_csv(index=False)
                st.info(f"File `{uploaded.name}` has existing tabular structure. You can re-structure it or extract sub-tables below.")
            except AIConversionRequired as req:
                raw_text = req.raw_text
            except Exception as e:
                logger.warning("failed to read file: %s", e)
                try:
                    uploaded.seek(0)
                    raw_text = uploaded.read().decode("utf-8", errors="replace")
                except Exception:
                    st.error(f"Could not read uploaded file: {e}")
    else:
        raw_text = st.text_area(
            "Paste raw unstructured text, notes, or log entries here:",
            height=200,
            placeholder="e.g. Server Incident Log:\n2026-03-01 10:14:22 ERROR Host-01 CPU throttle 98%\n2026-03-01 10:16:05 WARN Host-02 Memory pressure 85%",
            key="conv_text_area"
        )
        source_filename = "pasted_notes.txt"

if raw_text.strip():
    with st.container(border=True):
        section_card_header("Step 2: Review Content & Extraction Guidance", "Optional domain instructions and privacy check")
        findings = detect_sensitive_text(raw_text)
        if findings:
            st.warning(f"Sensitive content detected in source: {', '.join(findings)}. Review before transmitting to Gemini.")

        with st.expander("Preview Source Content", expanded=False):
            st.text(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

        hints = st.text_input(
            "Extraction Guidance (Optional):",
            placeholder="e.g. 'Extract columns: timestamp, severity, hostname, event_message'",
            help="Tell Gemini which columns or business entities to prioritize.",
            key="conv_hints_input"
        )

        st.caption("Privacy note: Content is sent securely to Google Gemini for tabular parsing.")

        if st.button("Extract Structured Table with Gemini", type="primary", key="btn_run_conversion"):
            if not api_key:
                st.error("Google Gemini API Key is required. Please enter it above or in the sidebar.")
            else:
                with st.spinner("Gemini is extracting and structuring records into a table..."):
                    try:
                        conv_df = convert_to_dataframe(
                            api_key=api_key,
                            raw_text=raw_text,
                            filename=source_filename,
                            model_name=chosen_model,
                            extra_instructions=hints.strip() if hints else None,
                        )
                        st.session_state["conv_result_df"] = conv_df
                        st.session_state["conv_source_name"] = source_filename
                        st.success(f"Successfully extracted {conv_df.shape[0]:,} rows × {conv_df.shape[1]} columns!")
                    except GeminiError as ge:
                        st.error(f"Gemini conversion error: {ge}")
                    except Exception as e:
                        logger.warning("conversion failed: %s", e)
                        st.error(f"Extraction failed: {e}")
                    finally:
                        if release("gemini", keep_key="gemini_keep"):
                            st.toast("Gemini key released from session state.")

converted_df = st.session_state.get("conv_result_df")
if converted_df is not None and not converted_df.empty:
    with st.container(border=True):
        section_card_header("Step 3: Preview & Save Active Dataset", "Review the generated schema and save to your project")
        st.dataframe(converted_df.head(20), width="stretch", hide_index=True)

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            default_stem = Path(st.session_state.get("conv_source_name", "converted_data.txt")).stem
            save_name_input = st.text_input(
                "Save dataset as:",
                value=f"{default_stem}_structured.csv",
                key="conv_save_name_input"
            )
        with col_s2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Save as Active Dataset", type="primary", key="btn_save_conv_active"):
                clean_target = save_name_input.strip() or f"{default_stem}_structured.csv"
                if not clean_target.lower().endswith(".csv"):
                    clean_target += ".csv"
                target_filename = get_unique_filename(clean_target, directory=DATASETS_DIR)
                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                target_path = DATASETS_DIR / target_filename
                atomic_write(target_path, converted_df.to_csv(index=False), mode="w", encoding="utf-8")
                invalidate_dataset_cache(target_filename)
                set_active_dataset(converted_df, target_filename)
                st.success(f"Saved `{target_filename}` and loaded as active dataset! Navigate to Data Overview or Explore Columns to inspect.")

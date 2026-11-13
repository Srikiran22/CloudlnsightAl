import io
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from Utils.AIConvert import convert_to_dataframe
from Utils.batch import merge_frames
from Utils.dataset_ui import (
    invalidate_dataset_cache,
    load_dataset_cached,
    render_sidebar,
    save_and_activate_dataset,
    set_active_dataset,
)
from Utils.Gemini import DEFAULT_GEMINI_MODEL, GEMINI_MODELS, GeminiError
from Utils.logsys import get_logger
from Utils.paths import (
    DATASETS_DIR,
    MAX_AGGREGATE_UPLOAD_BYTES,
    MAX_COMBINED_ROWS,
    MAX_INGESTION_CELLS,
    MAX_INGESTION_COLUMNS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    SUPPORTED_DATASET_EXTENSIONS,
    AIConversionRequired,
    atomic_write,
    compute_upload_signature,
    delete_dataset,
    get_unique_filename,
    get_valid_conversion,
    list_ready_datasets,
    read_dataset,
    read_tabular,
    record_conversion,
    release_filename_reservation,
    safe_stem,
    sanitize_for_csv_export,
)
from Utils.privacy import detect_sensitive_text
from Utils.secrets import ask, keep_box, release, value_of
from Utils.theme import metric_card, page_header, section_card_header

logger = get_logger("Upload")

page_header(
    "Data Hub & Ingest",
    "Multi-format streaming ingestion (CSV, Parquet, Excel, JSON, SQLite) with pre-parse resource bounding, XXE defense, and collision-safe catalog registration.",
    badge="LOCAL PIPELINE"
)
render_sidebar()

df = None
file_name = None
fresh_ingest = False

tab_files, tab_sqlite, tab_paste, tab_catalog = st.tabs([
    "File Ingestion (CSV / Parquet / Excel / JSON)",
    "SQLite Database Ingest",
    "Quick Paste / TSV",
    "Active Catalog"
])

# -----------------------------------------------------------------------------
# TAB 1: FILE INGESTION
# -----------------------------------------------------------------------------
with tab_files:
    with st.container(border=True):
        section_card_header(
            "Local Tabular File Upload",
            "Upload one or more structured datasets (.csv, .xlsx, .xls, .parquet, .json, .tsv) or document files (.txt, .pdf, .docx)."
        )
        accepted_types = sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)
        uploaded_files = st.file_uploader(
            "Select dataset file(s)",
            type=accepted_types,
            accept_multiple_files=True,
            key="file_uploader_tabular"
        )

    _compute_upload_signature = compute_upload_signature
    parsed_datasets = []
    pending = []
    failures = []
    merge_error = None

    if uploaded_files:
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"Too many files selected ({len(uploaded_files)}). Maximum allowed is {MAX_UPLOAD_FILES} files.")
        else:
            total_bytes = sum(getattr(f, "size", 0) for f in uploaded_files)
            if total_bytes > MAX_AGGREGATE_UPLOAD_BYTES:
                st.error(f"Total upload size ({total_bytes / (1024 * 1024):.1f} MB) exceeds aggregate limit of {MAX_AGGREGATE_UPLOAD_BYTES // (1024 * 1024)} MB.")
            else:
                current_sig = _compute_upload_signature(uploaded_files)
                last_sig = st.session_state.get("last_upload_signature")

                if current_sig == last_sig and "last_upload_state" in st.session_state:
                    saved_state = st.session_state["last_upload_state"]
                    parsed_datasets = saved_state.get("parsed", [])
                    pending = saved_state.get("pending", [])
                    failures = saved_state.get("failures", [])
                    file_name = saved_state.get("file_name")
                    df = saved_state.get("df")
                    merge_error = saved_state.get("merge_error")
                    if merge_error:
                        st.error(merge_error)
                else:
                    seen_names = set()
                    for uploaded in uploaded_files:
                        raw_name = Path(uploaded.name).name
                        target_name = get_unique_filename(raw_name, directory=DATASETS_DIR, extra_names=seen_names)
                        seen_names.add(target_name)

                        try:
                            single_df = read_tabular(uploaded, filename=target_name)
                            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                            if target_name.lower().endswith(".csv"):
                                sanitized_csv = sanitize_for_csv_export(single_df).to_csv(index=False)
                                atomic_write(DATASETS_DIR / target_name, sanitized_csv, mode="w", encoding="utf-8")
                            elif target_name.lower().endswith(".tsv"):
                                sanitized_tsv = sanitize_for_csv_export(single_df).to_csv(sep="\t", index=False)
                                atomic_write(DATASETS_DIR / target_name, sanitized_tsv, mode="w", encoding="utf-8")
                            else:
                                uploaded.seek(0)
                                atomic_write(DATASETS_DIR / target_name, uploaded.getbuffer(), mode="wb")
                            invalidate_dataset_cache(target_name)
                            parsed_datasets.append((target_name, single_df))
                        except AIConversionRequired as needed:
                            needed.__traceback__ = None
                            active_model = st.session_state.get("ai_convert_model")
                            valid_conv = (
                                get_valid_conversion(raw_name, needed.raw_text, model_name=active_model, extra_instructions="")
                                if active_model
                                else None
                            )
                            if valid_conv:
                                st.info(f"Reusing verified conversion `{valid_conv}` for `{target_name}`.")
                                parsed_datasets.append((valid_conv, read_dataset(valid_conv)))
                            else:
                                pending.append((target_name, needed))
                        except Exception as read_error:
                            logger.warning("ingest failed for %s: %s: %s", target_name, type(read_error).__name__, read_error)
                            failures.append((target_name, str(read_error)))
                            release_filename_reservation(target_name, directory=DATASETS_DIR)

                    if parsed_datasets:
                        if len(parsed_datasets) == 1:
                            file_name, df = parsed_datasets[0]
                            fresh_ingest = True
                        else:
                            total_rows = sum(len(f[1]) for f in parsed_datasets)
                            approx_cols = len(set().union(*(set(f[1].columns) for f in parsed_datasets)))
                            if total_rows > MAX_COMBINED_ROWS:
                                merge_error = f"Combined row count ({total_rows:,}) exceeds the maximum supported limit of {MAX_COMBINED_ROWS:,} rows. Merge was aborted."
                                st.error(merge_error)
                                file_name, df = None, None
                            elif approx_cols > MAX_INGESTION_COLUMNS:
                                merge_error = f"Estimated combined column count ({approx_cols:,}) exceeds the maximum supported limit of {MAX_INGESTION_COLUMNS:,} columns. Merge was aborted."
                                st.error(merge_error)
                                file_name, df = None, None
                            elif total_rows * approx_cols > MAX_INGESTION_CELLS:
                                merge_error = f"Estimated combined dataset cell count ({total_rows * approx_cols:,}) exceeds the maximum limit of {MAX_INGESTION_CELLS:,} cells. Merge was aborted."
                                st.error(merge_error)
                                file_name, df = None, None
                            else:
                                combined = merge_frames(parsed_datasets)
                                if combined.empty:
                                    merge_error = "Combined dataset has no records or columns. Merge was aborted."
                                    st.error(merge_error)
                                    file_name, df = None, None
                                else:
                                    sanitized_combined = sanitize_for_csv_export(combined)
                                    csv_bytes = sanitized_combined.to_csv(index=False).encode("utf-8")
                                    if len(csv_bytes) > MAX_UPLOAD_BYTES:
                                        merge_error = f"Combined dataset byte size ({len(csv_bytes)/(1024*1024):.1f} MB) exceeds maximum limit. Merge was aborted."
                                        st.error(merge_error)
                                        file_name, df = None, None
                                    else:
                                        saved_path = save_and_activate_dataset(combined, "combined_dataset.csv", sanitize=True)
                                        file_name = saved_path.name
                                        df = combined
                                        fresh_ingest = True

                    st.session_state["last_upload_signature"] = current_sig
                    st.session_state["last_upload_state"] = {
                        "parsed": parsed_datasets,
                        "pending": pending,
                        "failures": failures,
                        "file_name": file_name,
                        "df": df,
                        "merge_error": merge_error,
                    }
    else:
        st.session_state.pop("last_upload_signature", None)
        st.session_state.pop("last_upload_state", None)

    if failures:
        for fname, reason in failures:
            st.error(f"`{fname}` could not be read: {reason}")

    if pending:
        names_list = ", ".join(f"`{n}`" for n, _ in pending)
        st.info(
            f"{len(pending)} file(s) lack native tabular structure: {names_list}. "
            "Extract structured tabular schema using Google Gemini multimodal parsing."
        )
        hints = st.text_input("Conversion hints (optional)", key="ai_conv_hints")
        ask("gemini", "Google Gemini API Key:")
        keep_box("gemini_keep")

        chosen_model = st.selectbox(
            "Gemini Model:",
            GEMINI_MODELS,
            index=GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL),
            key="ai_convert_model"
        )

        for fname, needed in pending:
            findings = detect_sensitive_text(needed.raw_text)
            if findings:
                st.warning(f"Possible sensitive content detected in `{fname}`: {', '.join(findings)}.")
            with st.expander(f"Preview extracted content — {fname}"):
                preview_text = needed.raw_text[:1500]
                st.text(preview_text + ("..." if len(needed.raw_text) > 1500 else ""))

        if st.button("Convert with Gemini", type="primary"):
            if not value_of("gemini"):
                st.error("Please enter your Google Gemini API key first.")
            else:
                with st.spinner("Reconciling unstructured document text into tabular schema via Gemini API..."):
                    still_pending = []
                    newly_converted = []
                    try:
                        for fname, needed in pending:
                            try:
                                valid_conv = get_valid_conversion(
                                    fname,
                                    needed.raw_text,
                                    model_name=chosen_model,
                                    extra_instructions=(hints or "").strip() or None,
                                )
                                if valid_conv:
                                    st.info(f"Reusing verified conversion `{valid_conv}` for `{fname}`.")
                                    conv_df = read_dataset(valid_conv)
                                    parsed_datasets.append((valid_conv, conv_df))
                                    newly_converted.append((valid_conv, conv_df))
                                    continue
                                converted = convert_to_dataframe(
                                    api_key=value_of("gemini"),
                                    raw_text=needed.raw_text,
                                    filename=fname,
                                    model_name=chosen_model,
                                    extra_instructions=(hints or "").strip() or None,
                                )
                                ext_tag = Path(fname).suffix.lstrip(".").lower() or "txt"
                                target_stem = f"{Path(fname).stem}_{ext_tag}_converted.csv"
                                saved_path = save_and_activate_dataset(converted, target_stem, sanitize=True)
                                converted_name = saved_path.name
                                record_conversion(
                                    converted_name,
                                    fname,
                                    needed.raw_text,
                                    model_name=chosen_model,
                                    extra_instructions=(hints or "").strip() or None,
                                )
                                parsed_datasets.append((converted_name, converted))
                                newly_converted.append((converted_name, converted))
                                st.success(f"Converted `{fname}` to {converted.shape[0]:,} rows × {converted.shape[1]} cols, saved as `{converted_name}`.")
                            except GeminiError as conv_error:
                                still_pending.append((fname, needed))
                                st.error(f"`{fname}` conversion failed — {conv_error}")
                            except Exception as conv_error:
                                still_pending.append((fname, needed))
                                logger.warning("conversion failed for %s: %s: %s", fname, type(conv_error).__name__, conv_error)
                                st.error(f"`{fname}` conversion failed: {conv_error}")
                    finally:
                        if release("gemini", keep_key="gemini_keep"):
                            st.toast("Gemini key released from session state.")
                    pending = still_pending

                    if newly_converted:
                        merge_error = None
                        if len(parsed_datasets) == 1:
                            file_name, df = parsed_datasets[0]
                            fresh_ingest = True
                        else:
                            total_rows = sum(len(f[1]) for f in parsed_datasets)
                            if total_rows > MAX_COMBINED_ROWS:
                                merge_error = (
                                    f"Combined row count ({total_rows:,}) exceeds the maximum supported limit of "
                                    f"{MAX_COMBINED_ROWS:,} rows. Merge was aborted."
                                )
                                st.error(merge_error)
                                file_name, df = None, None
                            else:
                                combined = merge_frames(parsed_datasets)
                                saved_path = save_and_activate_dataset(combined, "combined_dataset.csv", sanitize=True)
                                file_name = saved_path.name
                                df = combined
                                fresh_ingest = True

                        if "last_upload_state" in st.session_state:
                            st.session_state["last_upload_state"]["parsed"] = parsed_datasets
                            st.session_state["last_upload_state"]["pending"] = pending
                            st.session_state["last_upload_state"]["failures"] = failures
                            st.session_state["last_upload_state"]["file_name"] = file_name
                            st.session_state["last_upload_state"]["df"] = df
                            st.session_state["last_upload_state"]["merge_error"] = merge_error

    if parsed_datasets and not merge_error:
        for unique_fname, df_item in parsed_datasets:
            with st.container(border=True):
                section_card_header(f"Dataset: {unique_fname}", "")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    metric_card("Rows", f"{df_item.shape[0]:,}")
                with m2:
                    metric_card("Columns", df_item.shape[1])
                with m3:
                    mem_mb = df_item.memory_usage(deep=True).sum() / (1024 * 1024)
                    metric_card("Memory", f"{mem_mb:.2f} MB")
                with m4:
                    null_pct = (df_item.isnull().sum().sum() / max(df_item.size, 1)) * 100
                    metric_card("Missing", f"{null_pct:.1f}%")
                st.dataframe(df_item.head(10), width="stretch", hide_index=True)

if df is not None and file_name is not None and (fresh_ingest or not st.session_state.get("dataset_name")):
    set_active_dataset(df, file_name)


# -----------------------------------------------------------------------------
# TAB 2: SQLITE DATABASE INGEST
# -----------------------------------------------------------------------------
with tab_sqlite:
    with st.container(border=True):
        section_card_header(
            "SQLite Database Connector",
            "Query or extract tables from a local SQLite database file (.db, .sqlite, .sqlite3)."
        )
        sqlite_file = st.file_uploader(
            "Upload SQLite database file",
            type=["db", "sqlite", "sqlite3"],
            key="sqlite_uploader"
        )

        if sqlite_file:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
                tmp_db.write(sqlite_file.getvalue())
                tmp_db_path = tmp_db.name

            try:
                conn = sqlite3.connect(tmp_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                tables = [r[0] for r in cursor.fetchall()]

                if not tables:
                    st.warning("No user tables found in this SQLite database.")
                else:
                    col_t1, col_t2 = st.columns([1, 2])
                    with col_t1:
                        selected_table = st.selectbox("Select table to extract", tables)
                    with col_t2:
                        custom_query = st.text_input("Optional SQL Query override", value=f"SELECT * FROM `{selected_table}` LIMIT 100000")

                    query_to_run = custom_query.strip() if custom_query.strip() else f"SELECT * FROM `{selected_table}`"

                    if st.button("Run SQL & Ingest Table", type="primary", key="btn_run_sql"):
                        with st.spinner("Executing SQL extraction query..."):
                            sql_df = pd.read_sql_query(query_to_run, conn)
                            if sql_df.empty:
                                st.warning("Query returned zero rows.")
                            else:
                                safe_table = safe_stem(selected_table, fallback="table")
                                saved_path = save_and_activate_dataset(sql_df, f"{safe_table}.csv", sanitize=True)
                                st.success(f"Ingested `{saved_path.name}` ({sql_df.shape[0]:,} rows × {sql_df.shape[1]} columns).")
                                st.rerun()
            except Exception as e:
                st.error(f"SQLite error: {e}")
            finally:
                try:
                    conn.close()
                    Path(tmp_db_path).unlink(missing_ok=True)
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# TAB 3: QUICK PASTE / TSV
# -----------------------------------------------------------------------------
with tab_paste:
    with st.container(border=True):
        section_card_header(
            "Paste Tabular Text",
            "Paste raw CSV, TSV (Tab-separated), or Pipe-separated text directly from spreadsheets or clipboards."
        )
        paste_text = st.text_area(
            "Paste structured data here (first row should be headers)",
            height=220,
            placeholder="feature_a,feature_b,category,label\n1.2,3.4,GroupA,0\n5.6,7.8,GroupB,1\n..."
        )
        col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1])
        with col_p1:
            sep = st.selectbox("Delimiter", [", (Comma)", "\\t (Tab)", "| (Pipe)", "; (Semicolon)"])
            sep_map = {", (Comma)": ",", "\\t (Tab)": "\t", "| (Pipe)": "|", "; (Semicolon)": ";"}
            chosen_sep = sep_map[sep]
        with col_p2:
            pasted_name = st.text_input("Dataset name", value="pasted_data.csv")

        if st.button("Parse & Ingest Text", type="primary", key="btn_parse_pasted"):
            stripped_text = paste_text.strip()
            if not stripped_text:
                st.warning("Please paste some tabular text first.")
            else:
                raw_target = pasted_name.strip() or "pasted_data.csv"
                target_name = raw_target if raw_target.lower().endswith((".csv", ".tsv")) else f"{raw_target}.csv"
                try:
                    pasted_bytes = stripped_text.encode("utf-8")
                    pasted_df = read_tabular(
                        io.BytesIO(pasted_bytes),
                        filename=target_name,
                        sep=chosen_sep,
                    )
                    if pasted_df.empty:
                        st.warning("Parsed dataset contains no records or columns.")
                    else:
                        saved_path = save_and_activate_dataset(pasted_df, target_name, sanitize=True)
                        st.success(
                            f"Parsed `{saved_path.name}` ({pasted_df.shape[0]:,} rows × {pasted_df.shape[1]} cols) "
                            "and registered as active dataset."
                        )
                        st.rerun()
                except ValueError as val_err:
                    st.error(f"Ingestion boundary violation: {val_err}")
                except pd.errors.EmptyDataError:
                    st.warning("Pasted content contains no columns or data.")
                except pd.errors.ParserError as parse_err:
                    st.error(f"Malformed tabular data: Could not parse text with delimiter '{chosen_sep}'. Details: {parse_err}")
                except Exception as exc:
                    logger.warning("Quick-paste ingestion failed for '%s': %s: %s", target_name, type(exc).__name__, exc)
                    st.error(f"Ingestion error: {exc}")


# -----------------------------------------------------------------------------
# TAB 4: ACTIVE CATALOG
# -----------------------------------------------------------------------------
with tab_catalog:
    with st.container(border=True):
        section_card_header(
            "Workspace Dataset Catalog",
            "View and manage tabular datasets stored locally in the Datasets directory."
        )
        ready_files = list_ready_datasets()
        if not ready_files:
            st.info("No datasets currently saved in `Datasets/`. Ingest files above or load sample benchmarks from the Home page.")
        else:
            cat_rows = []
            for rf in ready_files:
                fpath = DATASETS_DIR / rf
                fsize_kb = fpath.stat().st_size / 1024
                mtime_str = pd.to_datetime(fpath.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
                cat_rows.append({
                    "Dataset": rf,
                    "Size (KB)": f"{fsize_kb:.1f} KB",
                    "Last Modified": mtime_str,
                })
            st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)

            c_act1, c_act2 = st.columns(2)
            with c_act1:
                target_to_act = st.selectbox("Select dataset to activate", ready_files, key="select_cat_act")
                if st.button(f"Activate `{target_to_act}`", type="primary", key="btn_cat_activate"):
                    df_to_act = load_dataset_cached(target_to_act)
                    set_active_dataset(df_to_act, target_to_act)
                    st.toast(f"Active dataset set to `{target_to_act}`")
                    st.rerun()
            with c_act2:
                target_to_del = st.selectbox("Select dataset to delete", ready_files, key="select_cat_del")
                if st.button(f"Delete `{target_to_del}`", help="Deletes file from disk", key="btn_cat_delete"):
                    try:
                        delete_dataset(target_to_del, directory=DATASETS_DIR)
                        invalidate_dataset_cache(target_to_del)
                        if st.session_state.get("dataset_name") == target_to_del:
                            set_active_dataset(None, None)
                        st.toast(f"Deleted `{target_to_del}`")
                    except Exception as del_err:
                        st.error(f"Failed to delete `{target_to_del}`: {del_err}")
                    st.rerun()

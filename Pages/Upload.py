import streamlit as st
from pathlib import Path

from Utils.secrets import ask, value_of, keep_box, release, drop
from Utils.S3 import describe_s3_error, get_s3_client, list_s3_datasets, download_s3_dataset, upload_s3_dataset
from Utils.Gemini import GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError
from Utils.logsys import get_logger
from Utils.paths import (
    AIConversionRequired,
    DATASETS_DIR,
    atomic_write,
    get_unique_filename,
    release_filename_reservation,
    get_valid_conversion,
    record_conversion,
    list_ready_datasets,
    read_dataset,
    read_tabular,
    SUPPORTED_DATASET_EXTENSIONS,
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_BYTES,
    MAX_AGGREGATE_UPLOAD_BYTES,
    MAX_COMBINED_ROWS,
    MAX_INGESTION_COLUMNS,
    MAX_INGESTION_CELLS,
    compute_upload_signature,
)
from Utils.AIConvert import convert_to_dataframe
from Utils.batch import merge_frames
from Utils.dataset_ui import render_sidebar, set_active_dataset, invalidate_dataset_cache
from Utils.privacy import detect_sensitive_text
from Utils.theme import page_header, section_card_header

logger = get_logger("Upload")

page_header(
    "Import Files",
    "Load local tabular files (.csv, .xlsx, .parquet, .json) or connect to Amazon S3."
)

upload_mode = st.radio(
    "Source:",
    ["Local file", "Amazon S3"],
    horizontal=True
)

ACCEPTED_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)

df = None
file_name = None
fresh_ingest = False

_compute_upload_signature = compute_upload_signature


if upload_mode == "Local file":
    with st.container(border=True):
        section_card_header(
            "Upload Local Data Files",
            "Choose one or more tabular datasets (.csv, .xlsx, .parquet, .json) to ingest"
        )
        uploaded_files = st.file_uploader(
            "Choose data file(s) — one or many",
            type=ACCEPTED_TYPES,
            accept_multiple_files=True,
        )

    parsed = []    # (name, df) ready to use
    pending = []   # (name, AIConversionRequired) awaiting Gemini conversion
    failures = []  # (name, reason)

    if uploaded_files:
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(
                f"Too many files selected ({len(uploaded_files)}). Maximum allowed is "
                f"{MAX_UPLOAD_FILES} files. Please reduce your selection."
            )
        else:
            total_bytes = sum(getattr(f, "size", 0) for f in uploaded_files)
            if total_bytes > MAX_AGGREGATE_UPLOAD_BYTES:
                st.error(
                    f"Total upload size ({total_bytes / (1024 * 1024):.1f} MB) exceeds aggregate limit of "
                    f"{MAX_AGGREGATE_UPLOAD_BYTES // (1024 * 1024)} MB. Please split your files."
                )
            else:
                current_sig = _compute_upload_signature(uploaded_files)
                last_sig = st.session_state.get("last_upload_signature")
                if current_sig == last_sig and "last_upload_state" in st.session_state:
                    saved_state = st.session_state["last_upload_state"]
                    parsed = saved_state["parsed"]
                    pending = saved_state["pending"]
                    failures = saved_state["failures"]
                    file_name = saved_state["file_name"]
                    df = saved_state["df"]
                    if saved_state.get("merge_error"):
                        st.error(saved_state["merge_error"])
                else:
                    seen_names = set()
                    merge_error = None
                    for uploaded in uploaded_files:
                        raw_name = Path(uploaded.name).name
                        target_name = get_unique_filename(raw_name, directory=DATASETS_DIR, extra_names=seen_names)
                        seen_names.add(target_name)
                        if target_name != raw_name:
                            st.info(f"Existing dataset `{raw_name}` preserved — upload saved as `{target_name}`.")

                        try:
                            single_df = read_tabular(uploaded, filename=target_name)
                            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                            uploaded.seek(0)
                            atomic_write(DATASETS_DIR / target_name, uploaded.getbuffer(), mode="wb")
                            invalidate_dataset_cache(target_name)
                            parsed.append((target_name, single_df))
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
                                parsed.append((valid_conv, read_dataset(valid_conv)))
                            else:
                                pending.append((target_name, needed))
                        except Exception as read_error:
                            logger.warning("ingest failed for %s: %s: %s", target_name, type(read_error).__name__, read_error)
                            failures.append((target_name, str(read_error)))
                            release_filename_reservation(target_name, directory=DATASETS_DIR)

                    if parsed:
                        if len(parsed) == 1:
                            file_name, df = parsed[0]
                            fresh_ingest = True
                        else:
                            total_rows = sum(len(f[1]) for f in parsed)
                            approx_cols = len(set().union(*(set(f[1].columns) for f in parsed)))
                            if total_rows > MAX_COMBINED_ROWS:
                                merge_error = (
                                    f"Combined row count ({total_rows:,}) exceeds the maximum supported limit of "
                                    f"{MAX_COMBINED_ROWS:,} rows. Merge was aborted."
                                )
                                st.error(merge_error)
                                file_name, df = None, None
                            elif approx_cols > MAX_INGESTION_COLUMNS:
                                merge_error = (
                                    f"Estimated combined column count ({approx_cols:,}) exceeds the maximum supported limit of "
                                    f"{MAX_INGESTION_COLUMNS:,} columns. Merge was aborted."
                                )
                                st.error(merge_error)
                                file_name, df = None, None
                            elif total_rows * approx_cols > MAX_INGESTION_CELLS:
                                merge_error = (
                                    f"Estimated combined dataset cell count ({total_rows * approx_cols:,}) exceeds the maximum "
                                    f"limit of {MAX_INGESTION_CELLS:,} cells. Merge was aborted."
                                )
                                st.error(merge_error)
                                file_name, df = None, None
                            else:
                                combined = merge_frames(parsed)
                                if combined.empty:
                                    merge_error = "Combined dataset has no records or columns. Merge was aborted."
                                    st.error(merge_error)
                                    file_name, df = None, None
                                elif combined.shape[1] > MAX_INGESTION_COLUMNS:
                                    merge_error = (
                                        f"Combined column count ({combined.shape[1]:,}) exceeds the maximum supported limit of "
                                        f"{MAX_INGESTION_COLUMNS:,} columns. Merge was aborted."
                                    )
                                    st.error(merge_error)
                                    file_name, df = None, None
                                elif combined.shape[0] * combined.shape[1] > MAX_INGESTION_CELLS:
                                    merge_error = (
                                        f"Combined dataset cell count ({combined.shape[0] * combined.shape[1]:,}) exceeds the maximum "
                                        f"limit of {MAX_INGESTION_CELLS:,} cells. Merge was aborted."
                                    )
                                    st.error(merge_error)
                                    file_name, df = None, None
                                else:
                                    csv_text = combined.to_csv(index=False)
                                    csv_bytes = csv_text.encode("utf-8")
                                    if len(csv_bytes) > MAX_UPLOAD_BYTES:
                                        merge_error = (
                                            f"Combined dataset byte size ({len(csv_bytes)/(1024*1024):.1f} MB) exceeds the maximum "
                                            f"limit of {MAX_UPLOAD_BYTES/(1024*1024):.0f} MB. Merge was aborted."
                                        )
                                        st.error(merge_error)
                                        file_name, df = None, None
                                    else:
                                        file_name = get_unique_filename("combined_dataset.csv", directory=DATASETS_DIR)
                                        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                                        atomic_write(DATASETS_DIR / file_name, csv_text, mode="w", encoding="utf-8")
                                        invalidate_dataset_cache(file_name)
                                        df = combined
                                        fresh_ingest = True

                    st.session_state["last_upload_signature"] = current_sig
                    st.session_state["last_upload_state"] = {
                        "parsed": parsed,
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
            f"{len(pending)} file(s) have no native table structure: {names_list}. "
            "Structured files above are already loaded; convert the rest with Gemini."
        )

        hints = st.text_input(
            "Conversion hints (optional)",
            help=(
                "Tell Gemini what these files contain so it structures them well — "
                "e.g. 'previous year question papers: one row per question with year, "
                "subject, topic, marks'. Leave empty for generic table extraction."
            ),
        )
        ask(
            "gemini",
            "Google Gemini API Key:",
            help_text="Entered at runtime, held in memory only."
        )
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
                st.warning(
                    f"Possible sensitive content detected in `{fname}`: {', '.join(findings)}. "
                    "Review before converting with Gemini."
                )
            with st.expander(f"Preview extracted content — {fname}"):
                preview_text = needed.raw_text[:1500]
                st.text(preview_text + ("..." if len(needed.raw_text) > 1500 else ""))

        st.warning(
            "Privacy note: extracted file content is sent to Google Gemini for structuring. "
            "Avoid sensitive data or use de-identified copies."
        )

        if st.button("Convert with Gemini", type="primary"):
            if not value_of("gemini"):
                st.error("Please enter your Google Gemini API key first.")
            else:
                with st.spinner("Gemini is structuring your files..."):
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
                                    parsed.append((valid_conv, conv_df))
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
                                converted_name = get_unique_filename(target_stem, directory=DATASETS_DIR)
                                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                                atomic_write(DATASETS_DIR / converted_name, converted.to_csv(index=False), mode="w", encoding="utf-8")
                                invalidate_dataset_cache(converted_name)
                                record_conversion(
                                    converted_name,
                                    fname,
                                    needed.raw_text,
                                    model_name=chosen_model,
                                    extra_instructions=(hints or "").strip() or None,
                                )
                                parsed.append((converted_name, converted))
                                newly_converted.append((converted_name, converted))
                                st.success(
                                    f"Converted `{fname}` to {converted.shape[0]:,} rows × "
                                    f"{converted.shape[1]} cols, saved as `{converted_name}`."
                                )
                            except GeminiError as conv_error:
                                still_pending.append((fname, needed))
                                st.error(f"`{fname}` conversion failed — {conv_error}")
                            except Exception as conv_error:
                                still_pending.append((fname, needed))
                                logger.warning("conversion failed for %s: %s: %s",
                                               fname, type(conv_error).__name__, conv_error)
                                st.error(f"`{fname}` conversion failed: {conv_error}")
                    finally:
                        if release("gemini", keep_key="gemini_keep"):
                            st.toast("Gemini key released from session state.")
                    pending = still_pending

                    if newly_converted:
                        merge_error = None
                        if len(parsed) == 1:
                            file_name, df = parsed[0]
                            fresh_ingest = True
                        else:
                            total_rows = sum(len(f[1]) for f in parsed)
                            if total_rows > MAX_COMBINED_ROWS:
                                merge_error = (
                                    f"Combined row count ({total_rows:,}) exceeds the maximum supported limit of "
                                    f"{MAX_COMBINED_ROWS:,} rows. Merge was aborted."
                                )
                                st.error(merge_error)
                                file_name, df = None, None
                            else:
                                combined = merge_frames(parsed)
                                csv_text = combined.to_csv(index=False)
                                csv_bytes = csv_text.encode("utf-8")
                                if combined.empty:
                                    merge_error = "Combined dataset has no records or columns. Merge was aborted."
                                    st.error(merge_error)
                                    file_name, df = None, None
                                elif combined.shape[1] > MAX_INGESTION_COLUMNS:
                                    merge_error = (
                                        f"Combined column count ({combined.shape[1]:,}) exceeds the maximum supported limit of "
                                        f"{MAX_INGESTION_COLUMNS:,} columns. Merge was aborted."
                                    )
                                    st.error(merge_error)
                                    file_name, df = None, None
                                elif len(csv_bytes) > MAX_UPLOAD_BYTES:
                                    merge_error = (
                                        f"Combined dataset byte size ({len(csv_bytes)/(1024*1024):.1f} MB) exceeds the maximum "
                                        f"limit of {MAX_UPLOAD_BYTES/(1024*1024):.0f} MB. Merge was aborted."
                                    )
                                    st.error(merge_error)
                                    file_name, df = None, None
                                else:
                                    file_name = get_unique_filename("combined_dataset.csv", directory=DATASETS_DIR)
                                    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                                    atomic_write(DATASETS_DIR / file_name, csv_text, mode="w", encoding="utf-8")
                                    invalidate_dataset_cache(file_name)
                                    df = combined
                                    fresh_ingest = True

                        if "last_upload_state" in st.session_state:
                            st.session_state["last_upload_state"]["parsed"] = parsed
                            st.session_state["last_upload_state"]["pending"] = pending
                            st.session_state["last_upload_state"]["failures"] = failures
                            st.session_state["last_upload_state"]["file_name"] = file_name
                            st.session_state["last_upload_state"]["df"] = df
                            st.session_state["last_upload_state"]["merge_error"] = merge_error

else:
    with st.container(border=True):
        section_card_header("S3 Connection & IAM Authentication", "Configure AWS bucket credentials or ambient environment roles")

        s3_auth_mode = st.radio(
            "Authentication Method:",
            ["Manual Access Keys", "IAM Role / Environment / AWS CLI (Automatic)"],
            horizontal=True,
            key="s3_auth_mode",
        )

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            bucket = st.text_input(
                "S3 Bucket Name:",
                value=st.session_state.get("s3_bucket", ""),
                placeholder="e.g. my-analytics-bucket",
                key="s3_bucket_input",
            )
        with c_b2:
            region = st.text_input(
                "AWS Region:",
                value=st.session_state.get("s3_region", "us-east-1"),
                placeholder="e.g. us-east-1",
                key="s3_region_input",
            )

        clean_bucket = bucket.strip() if bucket else ""
        clean_region = region.strip() if region else "us-east-1"
        if clean_bucket:
            st.session_state["s3_bucket"] = clean_bucket
        if clean_region:
            st.session_state["s3_region"] = clean_region

        if s3_auth_mode == "Manual Access Keys":
            c1, c2, c3 = st.columns(3)
            with c1:
                aws_key = ask("aws_access", "AWS Access Key ID:")
            with c2:
                aws_secret = ask("aws_secret", "AWS Secret Access Key:")
            with c3:
                aws_session = ask(
                    "aws_session",
                    "AWS Session Token (Optional):",
                    help_text="Required for temporary credentials from STS, AWS SSO, or assumed roles."
                )

            c_keep, c_clear = st.columns([3, 1])
            with c_keep:
                keep_box("aws_keep")
            with c_clear:
                if st.button("Clear keys", help="Purge entered AWS credentials from memory", key="btn_clear_aws_keys"):
                    drop("aws_access", "aws_secret", "aws_session")
                    st.session_state.pop("s3_files", None)
                    st.toast("AWS credentials cleared from memory.")
                    st.rerun()

            auth_configured = bool(clean_bucket and (value_of("aws_access") or aws_key) and (value_of("aws_secret") or aws_secret))
        else:
            st.info(
                "Connecting with ambient credentials from your IAM instance profile, ECS task role, "
                "environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), or AWS CLI (`~/.aws/credentials`)."
            )
            auth_configured = bool(clean_bucket)

    def _create_s3_client():
        if s3_auth_mode == "Manual Access Keys":
            k = value_of("aws_access")
            s = value_of("aws_secret")
            t = value_of("aws_session") or None
            return get_s3_client(k, s, region_name=clean_region, aws_session_token=t)
        return get_s3_client(region_name=clean_region)

    connection_key = (clean_bucket, clean_region, s3_auth_mode)
    if st.session_state.get("s3_connection_key") != connection_key:
        st.session_state["s3_connection_key"] = connection_key
        st.session_state["s3_files"] = []

    with st.container(border=True):
        section_card_header("S3 Bucket Operations", "Browse objects or export datasets")
        s3_tab_import, s3_tab_export = st.tabs(["Import from S3", "Export to S3"])

    with s3_tab_import:
        st.markdown("Browse and download tabular or unstructured datasets from your bucket.")

        if st.button("Fetch datasets", type="primary", disabled=not auth_configured, key="btn_fetch_s3"):
            with st.spinner(f"Scanning bucket `{clean_bucket}` for datasets..."):
                try:
                    s3_client = _create_s3_client()
                    s3_files, s3_meta = list_s3_datasets(clean_bucket, s3_client, return_meta=True)
                    if s3_files:
                        st.session_state["s3_files"] = s3_files
                        st.success(f"Found {len(s3_files)} dataset(s) in S3 bucket `{clean_bucket}`!")
                    else:
                        st.session_state["s3_files"] = []
                        st.info(f"No supported data files found in S3 bucket `{clean_bucket}`.")
                    if s3_meta.get("truncated"):
                        st.warning(
                            f"Notice: Bucket scan stopped after {s3_meta['scanned']:,} objects to protect performance. "
                            "Some datasets in large buckets may not be shown."
                        )
                except Exception as e:
                    logger.warning("s3 listing failed: %s: %s", type(e).__name__, e)
                    st.error(f"S3 connection error: {describe_s3_error(e, action='list', bucket=clean_bucket)}")

        s3_files_avail = st.session_state.get("s3_files", [])
        if s3_files_avail:
            st.markdown("---")
            chosen_s3_file = st.selectbox("Select dataset from bucket:", s3_files_avail, key="select_s3_file")
            if st.button("Download & load", type="primary", key="btn_download_s3"):
                if s3_auth_mode == "Manual Access Keys" and not (value_of("aws_access") and value_of("aws_secret")):
                    st.error("Re-enter your AWS access keys above — they were cleared after the previous operation.")
                else:
                    with st.spinner(f"Downloading `{chosen_s3_file}` from S3..."):
                        try:
                            s3_client = _create_s3_client()
                            s3_stem = chosen_s3_file.replace("/", "_").replace("\\", "_")
                            file_name = get_unique_filename(s3_stem, directory=DATASETS_DIR)

                            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                            local_s3_path = DATASETS_DIR / file_name
                            df, _ = download_s3_dataset(clean_bucket, chosen_s3_file, s3_client, destination_path=local_s3_path)
                            invalidate_dataset_cache(file_name)
                            if df is None:
                                st.warning(
                                    f"`{file_name}` was downloaded to `Datasets/` but has no native table "
                                    "structure. Convert it with Gemini via a **local file upload** "
                                    "(the converted copy is saved automatically)."
                                )
                            else:
                                fresh_ingest = True
                                st.success(f"Downloaded `{file_name}` from Amazon S3 and loaded as active dataset.")
                        except ValueError as ve:
                            st.error(str(ve))
                        except Exception as e:
                            logger.warning("s3 download failed: %s: %s", type(e).__name__, e)
                            st.error(f"S3 download failed: {describe_s3_error(e, action='download', bucket=clean_bucket)}")
                        finally:
                            if s3_auth_mode == "Manual Access Keys":
                                if release("aws_access", "aws_secret", "aws_session", keep_key="aws_keep"):
                                    st.toast("AWS credentials released from session state.")

    with s3_tab_export:
        st.markdown("Upload the currently active dataset or a saved dataset from `Datasets/` to an S3 bucket.")
        curr_df = st.session_state.get("current_df")
        curr_name = st.session_state.get("dataset_name")

        export_df = None
        export_name = None

        if curr_df is not None and curr_name:
            st.info(f"Active dataset ready for export: `{curr_name}` ({curr_df.shape[0]:,} rows × {curr_df.shape[1]} columns)")
            export_df = curr_df
            export_name = curr_name
        else:
            ready_files = list_ready_datasets()
            if ready_files:
                sel_ready = st.selectbox("Select a dataset to export to S3:", ready_files, key="select_export_s3_file")
                try:
                    export_df = read_dataset(sel_ready)
                    export_name = sel_ready
                    st.caption(f"Loaded: `{export_name}` ({export_df.shape[0]:,} rows × {export_df.shape[1]} columns)")
                except Exception as read_err:
                    st.error(f"Could not load `{sel_ready}`: {read_err}")
            else:
                st.warning("No dataset is currently active and no saved datasets found. Ingest or clean a dataset first before exporting to S3.")

        if export_df is not None and export_name:
            default_key = f"exports/{Path(export_name).stem}.csv" if not export_name.endswith(".csv") else f"exports/{export_name}"
            exp_c1, exp_c2 = st.columns([3, 2])
            with exp_c1:
                dest_key = st.text_input("Destination S3 Object Key (Path in Bucket):", value=default_key, help="e.g. exports/customer_churn.csv", key="export_dest_key")
            with exp_c2:
                st.write("")
                st.write("")
                do_sanitize = st.checkbox("Sanitize for CSV export", value=False, help="Neutralize spreadsheet formula injection (=, +, -, @) for external spreadsheets.", key="export_sanitize")

            if st.button("Upload to S3", type="primary", disabled=not auth_configured, key="btn_upload_s3"):
                if not dest_key.strip():
                    st.error("Please enter a valid destination object key.")
                elif s3_auth_mode == "Manual Access Keys" and not (value_of("aws_access") and value_of("aws_secret")):
                    st.error("Please enter your AWS Access Key ID and Secret Access Key above.")
                else:
                    with st.spinner(f"Uploading `{export_name}` to `s3://{clean_bucket}/{dest_key.strip()}`..."):
                        try:
                            s3_client = _create_s3_client()
                            upload_s3_dataset(export_df, clean_bucket, dest_key.strip(), s3_client, sanitize=do_sanitize)
                            st.success(f"Successfully uploaded `{export_name}` to `s3://{clean_bucket}/{dest_key.strip()}`!")
                        except ValueError as ve:
                            st.error(str(ve))
                        except Exception as e:
                            logger.warning("s3 upload failed: %s: %s", type(e).__name__, e)
                            st.error(f"S3 export failed: {describe_s3_error(e, action='upload', bucket=clean_bucket)}")
                        finally:
                            if s3_auth_mode == "Manual Access Keys":
                                if release("aws_access", "aws_secret", "aws_session", keep_key="aws_keep"):
                                    st.toast("AWS credentials released from session state.")

if df is not None and file_name is not None and (fresh_ingest or not st.session_state.get("dataset_name")):
    set_active_dataset(df, file_name)

render_sidebar()

active_df = st.session_state.get("current_df")
active_name = st.session_state.get("dataset_name")
if active_df is not None and active_name:
    st.caption(f"Active dataset: `{active_name}` ({active_df.shape[0]:,} rows × {active_df.shape[1]} columns)")

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{active_df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", active_df.shape[1])
    with col3:
        missing_total = int(active_df.isnull().sum().sum())
        st.metric("Missing Values", f"{missing_total:,}")
    with col4:
        duplicate_total = int(active_df.duplicated().sum())
        st.metric("Duplicate Rows", f"{duplicate_total:,}")

    st.subheader("Preview (first 10 rows)")
    st.dataframe(active_df.head(10), width="stretch", hide_index=True)

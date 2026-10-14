import streamlit as st
from pathlib import Path

from Utils.secrets import ask, value_of, keep_box, release
from Utils.S3 import describe_s3_error, get_s3_client, list_s3_datasets, download_s3_dataset
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

logger = get_logger("Upload")

st.title("Ingest data")
st.markdown(
    "Load a file or connect to S3. Structured formats parse natively; free text and PDFs "
    "are converted into tables with Gemini."
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
    st.subheader("Connect to Amazon S3")
    c1, c2 = st.columns(2)
    with c1:
        aws_key = ask("aws_access", "AWS Access Key ID:")
        bucket = st.text_input("S3 Bucket Name:", value=st.session_state.get("s3_bucket", ""))
    with c2:
        aws_secret = ask("aws_secret", "AWS Secret Access Key:")
        region = st.text_input("AWS Region:", value=st.session_state.get("s3_region", "us-east-1"))

    keep_box("aws_keep")

    if not (aws_key and aws_secret) and st.session_state.get("s3_files"):
        st.info("AWS credentials were cleared after the last operation — re-enter them to keep working with this bucket.")

    if aws_key and aws_secret and bucket:
        st.session_state["s3_bucket"] = bucket
        st.session_state["s3_region"] = region

        connection_key = (bucket, region)
        if st.session_state.get("s3_connection_key") != connection_key:
            st.session_state["s3_connection_key"] = connection_key
            st.session_state["s3_files"] = []

        if st.button("Fetch datasets", disabled=not (aws_key and aws_secret and bucket)):
            try:
                s3_client = get_s3_client(aws_key, aws_secret, region)
                s3_files, s3_meta = list_s3_datasets(bucket, s3_client, return_meta=True)
                if s3_files:
                    st.session_state["s3_files"] = s3_files
                    st.success(f"Found {len(s3_files)} dataset(s) in S3 bucket `{bucket}`!")
                else:
                    st.info(f"No supported data files found in S3 bucket `{bucket}`.")
                if s3_meta.get("truncated"):
                    st.warning(
                        f"Notice: Bucket scan stopped after {s3_meta['scanned']:,} objects to protect performance. "
                        "Some datasets in large buckets may not be shown."
                    )
            except Exception as e:
                logger.warning("s3 listing failed: %s: %s", type(e).__name__, e)
                st.error(f"S3 connection error: {describe_s3_error(e)}")
            finally:
                if release("aws_access", "aws_secret", keep_key="aws_keep"):
                    st.toast("AWS credentials released from session state.")

        s3_files_avail = st.session_state.get("s3_files", [])
        if s3_files_avail:
            chosen_s3_file = st.selectbox("Dataset from bucket:", s3_files_avail)
            if st.button("Download & load"):
                if not (value_of("aws_access") and value_of("aws_secret")):
                    st.error("Re-enter your AWS access keys above — they were cleared after the previous operation.")
                else:
                    try:
                        s3_client = get_s3_client(value_of("aws_access"), value_of("aws_secret"), region)
                        s3_stem = chosen_s3_file.replace("/", "_").replace("\\", "_")
                        file_name = get_unique_filename(s3_stem, directory=DATASETS_DIR)

                        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                        local_s3_path = DATASETS_DIR / file_name
                        df, _ = download_s3_dataset(bucket, chosen_s3_file, s3_client, destination_path=local_s3_path)
                        invalidate_dataset_cache(file_name)
                        if df is None:
                            st.warning(
                                f"`{file_name}` was downloaded to `Datasets/` but has no native table "
                                "structure. Convert it with Gemini via a **local file upload** "
                                "(the converted copy is saved automatically)."
                            )
                        else:
                            fresh_ingest = True
                            st.success(f"Downloaded `{file_name}` from Amazon S3.")
                    except ValueError as ve:
                        st.error(str(ve))
                    except Exception as e:
                        logger.warning("s3 download failed: %s: %s", type(e).__name__, e)
                        st.error(f"S3 download failed: {describe_s3_error(e)}")
                    finally:
                        if release("aws_access", "aws_secret", keep_key="aws_keep"):
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
    st.dataframe(active_df.head(10), width="stretch")

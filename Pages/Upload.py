import streamlit as st
from pathlib import Path

from Utils.secrets import ask, value_of, keep_box, release
from Utils.S3 import describe_s3_error, get_s3_client, list_s3_datasets, download_s3_dataset
from Utils.Gemini import GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError
from Utils.logsys import get_logger
from Utils.paths import (
    AIConversionRequired,
    DATASETS_DIR,
    MAX_UPLOAD_BYTES,
    read_dataset,
    read_tabular,
    SUPPORTED_DATASET_EXTENSIONS,
)
from Utils.AIConvert import convert_to_dataframe
from Utils.batch import merge_frames
from Utils.dataset_ui import render_sidebar

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
        seen_names = set()
        for uploaded in uploaded_files:
            name = Path(uploaded.name).name
            # duplicate upload names would silently overwrite each other on
            # disk and double rows in the merge; disambiguate instead
            if name in seen_names:
                stem, suffix = Path(name).stem, Path(name).suffix
                counter = 2
                while f"{stem}_{counter}{suffix}" in seen_names:
                    counter += 1
                name = f"{stem}_{counter}{suffix}"
                st.warning(f"Duplicate file name — second `{uploaded.name}` stored as `{name}`.")
            seen_names.add(name)

            try:
                if getattr(uploaded, "size", 0) > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"File is {uploaded.size / (1024 * 1024):.0f} MB; the limit is "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Split or trim the file."
                    )
                single_df = read_tabular(uploaded, filename=name)
                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                uploaded.seek(0)
                with open(DATASETS_DIR / name, "wb") as fh:
                    fh.write(uploaded.getbuffer())
                parsed.append((name, single_df))
            except AIConversionRequired as needed:
                converted_name = f"{Path(name).stem}_converted.csv"
                if (DATASETS_DIR / converted_name).exists():
                    parsed.append((converted_name, read_dataset(converted_name)))
                else:
                    pending.append((name, needed))
            except Exception as read_error:
                logger.warning("ingest failed for %s: %s: %s", name, type(read_error).__name__, read_error)
                failures.append((name, str(read_error)))

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
                    for fname, needed in pending:
                        try:
                            converted = convert_to_dataframe(
                                api_key=value_of("gemini"),
                                raw_text=needed.raw_text,
                                filename=fname,
                                model_name=chosen_model,
                                extra_instructions=(hints or "").strip() or None,
                            )
                            converted_name = f"{Path(fname).stem}_converted.csv"
                            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                            converted.to_csv(DATASETS_DIR / converted_name, index=False)
                            parsed.append((converted_name, converted))
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
                    if release("gemini", keep_key="gemini_keep"):
                        st.toast("Gemini key cleared from memory.")
                    pending = still_pending

    if parsed:
        if len(parsed) == 1:
            file_name, df = parsed[0]
        else:
            combined = merge_frames(parsed)
            file_name = "combined_dataset.csv"
            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            combined.to_csv(DATASETS_DIR / file_name, index=False)
            df = combined

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
                s3_files = list_s3_datasets(bucket, s3_client)
                if s3_files:
                    st.session_state["s3_files"] = s3_files
                    st.success(f"Found {len(s3_files)} dataset(s) in S3 bucket `{bucket}`!")
                else:
                    st.info(f"No supported data files found in S3 bucket `{bucket}`.")
                if release("aws_access", "aws_secret", keep_key="aws_keep"):
                    st.toast("AWS credentials cleared from memory.")
            except Exception as e:
                logger.warning("s3 listing failed: %s: %s", type(e).__name__, e)
                st.error(f"S3 connection error: {describe_s3_error(e)}")

        s3_files_avail = st.session_state.get("s3_files", [])
        if s3_files_avail:
            chosen_s3_file = st.selectbox("Dataset from bucket:", s3_files_avail)
            if st.button("Download & load"):
                if not (value_of("aws_access") and value_of("aws_secret")):
                    st.error("Re-enter your AWS access keys above — they were cleared after the previous operation.")
                else:
                    try:
                        s3_client = get_s3_client(value_of("aws_access"), value_of("aws_secret"), region)
                        df, raw_bytes = download_s3_dataset(bucket, chosen_s3_file, s3_client)
                        file_name = Path(chosen_s3_file).name

                        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                        local_s3_path = DATASETS_DIR / file_name
                        local_s3_path.write_bytes(raw_bytes)
                        if release("aws_access", "aws_secret", keep_key="aws_keep"):
                            st.toast("AWS credentials cleared from memory.")
                        if df is None:
                            st.warning(
                                f"`{file_name}` was downloaded to `Datasets/` but has no native table "
                                "structure. Convert it with Gemini via a **local file upload** "
                                "(the converted copy is saved automatically)."
                            )
                        else:
                            st.success(f"Downloaded `{file_name}` from Amazon S3.")
                    except ValueError as ve:
                        st.error(str(ve))
                    except Exception as e:
                        logger.warning("s3 download failed: %s: %s", type(e).__name__, e)
                        st.error(f"S3 download failed: {describe_s3_error(e)}")

if df is not None and file_name is not None:
    st.session_state["current_df"] = df
    st.session_state["dataset_name"] = file_name

render_sidebar()

active_df = st.session_state.get("current_df")
active_name = st.session_state.get("dataset_name")
if active_df is not None and active_name:
    st.success(f"Active dataset: `{active_name}`")

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

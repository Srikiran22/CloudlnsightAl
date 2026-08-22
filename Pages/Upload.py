import streamlit as st
from pathlib import Path

from Utils.secrets import ask, value_of, keep_box, release
from Utils.S3 import get_s3_client, list_s3_datasets, download_s3_dataset
from Utils.paths import (
    AIConversionRequired,
    DATASETS_DIR,
    read_tabular,
    SUPPORTED_DATASET_EXTENSIONS,
)
from Utils.AIConvert import convert_to_dataframe
from Utils.dataset_ui import render_sidebar

st.title("📂 Ingest Dataset")
st.markdown(
    "Upload any data file — CSV, Excel, JSON, TSV, Parquet, XML, HTML tables, or free text/PDF. "
    "Unstructured files are converted into clean tables using Google Gemini AI."
)

upload_mode = st.radio(
    "Select Ingestion Source:",
    ["💻 Local File Upload", "☁️ Amazon S3 Cloud Storage"],
    horizontal=True
)

ACCEPTED_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)

df = None
file_name = None

if upload_mode == "💻 Local File Upload":
    uploaded_file = st.file_uploader(
        "Choose a Data File (any supported format)",
        type=ACCEPTED_TYPES
    )

    if uploaded_file:
        file_name = Path(uploaded_file.name).name
        try:
            df = read_tabular(uploaded_file, filename=file_name)

            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = DATASETS_DIR / Path(file_name).name
            uploaded_file.seek(0)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        except AIConversionRequired as conversion_needed:
            st.info(
                f"ℹ️ `{file_name}` has no native table structure. "
                "Use **Gemini AI** below to convert it into a clean dataset."
            )

            with st.expander("🤖 AI-Powered File Conversion", expanded=True):
                api_key = ask(
                    "gemini",
                    "Google Gemini API Key:",
                    help_text="Entered at runtime, held in memory only."
                )
                keep_box("gemini_keep")

                chosen_model = st.selectbox(
                    "Gemini Model:",
                    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                    index=0,
                    key="ai_convert_model"
                )

                preview_text = conversion_needed.raw_text[:1500]
                with st.expander("👀 Preview extracted content"):
                    st.text(preview_text + ("..." if len(conversion_needed.raw_text) > 1500 else ""))

                st.warning(
                    "Privacy note: the extracted file content is sent to Google Gemini for structuring. "
                    "Avoid sensitive data or use de-identified copies."
                )

                if st.button("✨ Convert with Gemini AI", type="primary"):
                    if not value_of("gemini"):
                        st.error("❌ Please enter your Google Gemini API Key first.")
                    else:
                        with st.spinner("Gemini is extracting structured records from your file..."):
                            try:
                                df = convert_to_dataframe(
                                    api_key=value_of("gemini"),
                                    raw_text=conversion_needed.raw_text,
                                    filename=file_name,
                                    model_name=chosen_model
                                )
                                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                                converted_name = f"{Path(file_name).stem}_converted.csv"
                                df.to_csv(DATASETS_DIR / converted_name, index=False)
                                file_name = converted_name
                                if release("gemini", keep_key="gemini_keep"):
                                    st.toast("Gemini key cleared from memory.")
                                st.success(
                                    f"✅ Converted to {df.shape[0]:,} rows × {df.shape[1]} cols "
                                    f"and saved as `{converted_name}`."
                                )
                            except Exception as e:
                                st.error(f"❌ AI Conversion failed: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

else:
    st.subheader("☁️ Connect to Amazon S3 Bucket")
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

        if st.button("🔄 Fetch Datasets from S3"):
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
                st.error(f"❌ S3 Connection Error: {e}")

        s3_files_avail = st.session_state.get("s3_files", [])
        if s3_files_avail:
            chosen_s3_file = st.selectbox("Select S3 Dataset:", s3_files_avail)
            if st.button("📥 Download & Load from S3"):
                if not (value_of("aws_access") and value_of("aws_secret")):
                    st.error("❌ Re-enter your AWS access keys above — they were cleared after the previous operation.")
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
                        st.success(f"✅ Successfully downloaded `{file_name}` from Amazon S3!")
                    except AIConversionRequired:
                        file_name = Path(chosen_s3_file).name
                        st.warning(
                            f"⚠️ `{file_name}` was downloaded but has no native table structure. "
                            "Re-upload it via **💻 Local File Upload** and use Gemini AI conversion "
                            "(the converted copy is saved automatically)."
                        )
                    except Exception as e:
                        st.error(f"❌ S3 Download Failed: {e}")

if df is not None and file_name is not None:
    st.session_state["current_df"] = df
    st.session_state["dataset_name"] = file_name

render_sidebar()

active_df = st.session_state.get("current_df")
active_name = st.session_state.get("dataset_name")
if active_df is not None and active_name:
    st.success(f"✅ Active Dataset: `{active_name}`")

    st.subheader("📊 Dataset Overview")
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

    st.subheader("🔍 Data Preview (First 10 Rows)")
    st.dataframe(active_df.head(10), width="stretch")

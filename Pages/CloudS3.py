from pathlib import Path
import streamlit as st

from Utils.secrets import ask, value_of, keep_box, release, drop
from Utils.S3 import (
    describe_s3_error,
    get_s3_client,
    list_s3_datasets,
    download_s3_dataset,
    upload_s3_dataset,
)
from Utils.paths import (
    DATASETS_DIR,
    get_unique_filename,
    list_ready_datasets,
    resolve_dataset_path,
)
from Utils.dataset_ui import (
    render_sidebar,
    set_active_dataset,
    invalidate_dataset_cache,
)
from Utils.theme import page_header, section_card_header
from Utils.logsys import get_logger

logger = get_logger("CloudS3")

page_header(
    "Cloud Storage (S3)",
    "Connect to Amazon S3, browse and download bucket datasets, or export cleaned datasets to the cloud."
)

render_sidebar()

with st.container(border=True):
    section_card_header(
        "S3 Connection & Authentication",
        "Configure AWS bucket credentials or use ambient environment roles"
    )

    s3_auth_mode = st.radio(
        "Authentication Method:",
        ["Manual Access Keys", "IAM Role / Environment / AWS CLI (Automatic)"],
        horizontal=True,
        key="dedicated_s3_auth_mode",
    )

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        bucket = st.text_input(
            "S3 Bucket Name:",
            value=st.session_state.get("s3_bucket", ""),
            placeholder="e.g. my-analytics-bucket",
            key="dedicated_s3_bucket_input",
        )
    with c_b2:
        region = st.text_input(
            "AWS Region:",
            value=st.session_state.get("s3_region", "us-east-1"),
            placeholder="e.g. us-east-1",
            key="dedicated_s3_region_input",
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
            if st.button("Clear keys", help="Purge entered AWS credentials from memory", key="dedicated_clear_aws_keys"):
                drop("aws_access", "aws_secret", "aws_session")
                st.session_state.pop("dedicated_s3_files", None)
                st.toast("AWS credentials cleared from memory.")
                st.rerun()

        auth_configured = bool(clean_bucket and (value_of("aws_access") or aws_key) and (value_of("aws_secret") or aws_secret))
    else:
        st.info(
            "Connecting with ambient credentials from your IAM instance profile, ECS task role, "
            "environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), or AWS CLI (~/.aws/credentials)."
        )
        auth_configured = bool(clean_bucket)


def _create_s3_client():
    if s3_auth_mode == "Manual Access Keys":
        k = value_of("aws_access")
        s = value_of("aws_secret")
        t = value_of("aws_session") or None
        return get_s3_client(k, s, region_name=clean_region, aws_session_token=t)
    return get_s3_client(region_name=clean_region)


with st.container(border=True):
    section_card_header("S3 Bucket Operations", "Browse objects or export datasets")
    s3_tab_import, s3_tab_export = st.tabs(["Import from S3", "Export to S3"])

with s3_tab_import:
    st.markdown("Browse and download tabular or unstructured datasets from your configured S3 bucket.")

    if st.button("Fetch datasets", type="primary", disabled=not auth_configured, key="dedicated_fetch_s3"):
        with st.spinner(f"Scanning bucket `{clean_bucket}` for datasets..."):
            try:
                s3_client = _create_s3_client()
                s3_files, s3_meta = list_s3_datasets(clean_bucket, s3_client, return_meta=True)
                if s3_files:
                    st.session_state["dedicated_s3_files"] = s3_files
                    st.success(f"Found {len(s3_files)} dataset(s) in S3 bucket `{clean_bucket}`.")
                else:
                    st.session_state["dedicated_s3_files"] = []
                    st.info(f"No supported data files found in S3 bucket `{clean_bucket}`.")
                if s3_meta.get("truncated"):
                    st.warning(
                        f"Notice: Bucket scan stopped after {s3_meta['scanned']:,} objects to protect performance. "
                        "Some datasets in large buckets may not be shown."
                    )
            except Exception as e:
                logger.warning("s3 listing failed: %s: %s", type(e).__name__, e)
                st.error(f"S3 connection error: {describe_s3_error(e, action='list', bucket=clean_bucket)}")

    s3_files_avail = st.session_state.get("dedicated_s3_files", [])
    if s3_files_avail:
        st.markdown("---")
        chosen_s3_file = st.selectbox("Select dataset from bucket:", s3_files_avail, key="dedicated_select_s3_file")
        if st.button("Download & load", type="primary", key="dedicated_download_s3"):
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
                        downloaded_df, _ = download_s3_dataset(clean_bucket, chosen_s3_file, s3_client, destination_path=local_s3_path)
                        invalidate_dataset_cache(file_name)
                        if downloaded_df is None:
                            st.warning(
                                f"`{file_name}` was downloaded to `Datasets/` but has no native table "
                                "structure. Convert it with Gemini on the **AI File Converter** page."
                            )
                        else:
                            set_active_dataset(downloaded_df, file_name)
                            st.success(f"Downloaded `{file_name}` ({downloaded_df.shape[0]:,} rows × {downloaded_df.shape[1]} cols) from S3 and loaded as active dataset.")
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
        saved_files = list_ready_datasets()
        if saved_files:
            chosen_local = st.selectbox("Select dataset to export:", saved_files, key="dedicated_export_select")
            if chosen_local:
                export_name = chosen_local
        else:
            st.warning("No datasets currently available to export. Ingest or load a dataset first.")

    if export_name:
        default_key = f"exports/{Path(export_name).name}"
        s3_key_input = st.text_input(
            "Destination S3 Key (path in bucket):",
            value=default_key,
            help="S3 object key path, e.g. exports/my_data.csv",
            key="dedicated_export_s3_key"
        )
        if st.button("Upload to S3", type="primary", disabled=not auth_configured, key="dedicated_btn_upload_s3"):
            if not clean_bucket:
                st.error("Please enter a bucket name above.")
            elif s3_auth_mode == "Manual Access Keys" and not (value_of("aws_access") and value_of("aws_secret")):
                st.error("Re-enter your AWS access keys above — they were cleared after the previous operation.")
            else:
                with st.spinner(f"Exporting `{export_name}` to `s3://{clean_bucket}/{s3_key_input.strip()}`..."):
                    try:
                        s3_client = _create_s3_client()
                        local_path = resolve_dataset_path(export_name)
                        upload_s3_dataset(s3_client, clean_bucket, s3_key_input.strip(), local_path)
                        st.success(f"Successfully uploaded `{export_name}` to `s3://{clean_bucket}/{s3_key_input.strip()}`.")
                    except Exception as e:
                        logger.warning("s3 export failed: %s: %s", type(e).__name__, e)
                        st.error(f"S3 export failed: {describe_s3_error(e, action='upload', bucket=clean_bucket)}")
                    finally:
                        if s3_auth_mode == "Manual Access Keys":
                            if release("aws_access", "aws_secret", "aws_session", keep_key="aws_keep"):
                                st.toast("AWS credentials released from session state.")

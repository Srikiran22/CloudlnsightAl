import streamlit as st
from pathlib import Path

from Utils.Preprocessing import remove_duplicates, fill_missing_values, drop_missing_values
from Utils.paths import (
    AIConversionRequired, DATASETS_DIR, atomic_write, get_unique_filename, list_ready_datasets,
    read_tabular, sanitize_for_csv_export, SUPPORTED_DATASET_EXTENSIONS,
)
from Utils.dataset_ui import (
    dataframe_fingerprint,
    dataset_fingerprint,
    invalidate_dataset_cache,
    load_dataset_cached,
    render_sidebar,
    set_active_dataset,
)

from Utils.theme import page_header, section_card_header

page_header(
    "Clean Data",
    "Remove duplicate rows and resolve missing values, then save the result as a new dataset."
)

dataset_folder = DATASETS_DIR
available_files = list_ready_datasets()

with st.container(border=True):
    section_card_header("Source Selection", "Choose an existing dataset or upload a new file")
    source_option = st.radio(
        "Choose Dataset Source:",
        ["Select from Datasets Folder", "Upload New File"],
        horizontal=True
    )

    df = None
    selected_filename = None

    if source_option == "Select from Datasets Folder":
        if not available_files:
            st.info("No datasets found in `Datasets/`. Ingest one on the Ingest data page first.")
        else:
            selected_filename = st.selectbox("Select Dataset to Clean:", available_files)
            try:
                df = load_dataset_cached(selected_filename)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        uploaded_file = st.file_uploader(
            "Upload Dataset to Clean",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)
        )
        if uploaded_file:
            selected_filename = uploaded_file.name
            try:
                df = read_tabular(uploaded_file, filename=selected_filename)
            except AIConversionRequired:
                st.warning(
                    f"`{selected_filename}` has no native table structure. "
                    "Convert it with Gemini on the **Ingest data** page first, then clean the converted copy here."
                )
            except Exception as e:
                st.error(f"Error reading file: {e}")

if df is not None:
    is_stored = (source_option == "Select from Datasets Folder")
    active_fp = dataset_fingerprint(selected_filename) if (is_stored and selected_filename) else dataframe_fingerprint(df)
    metrics_cache_key = f"clean_orig_metrics_{selected_filename}_{active_fp}"
    if metrics_cache_key not in st.session_state:
        st.session_state[metrics_cache_key] = {
            "dup_count": int(df.duplicated().sum()),
            "missing_count": int(df.isnull().sum().sum()),
        }
    cached_metrics = st.session_state[metrics_cache_key]
    dup_count = cached_metrics["dup_count"]
    missing_count = cached_metrics["missing_count"]

    with st.container(border=True):
        section_card_header("Original Dataset Hygiene Audit", f"Initial data profile for `{selected_filename}`")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Duplicate Rows", f"{dup_count:,}", delta="Clean" if dup_count == 0 else f"{dup_count} dups", delta_color="off" if dup_count == 0 else "inverse")
        with col4:
            st.metric("Missing Values", f"{missing_count:,}", delta="Complete" if missing_count == 0 else f"{missing_count} nulls", delta_color="off" if missing_count == 0 else "inverse")

        with st.expander("Preview Original Data"):
            st.dataframe(df.head(10), width="stretch", hide_index=True)

    with st.container(border=True):
        section_card_header("Cleaning Pipeline Rules", "Configure deduplication, imputation strategies, and save destination")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            do_remove_dups = st.checkbox("Remove Duplicate Rows", value=True)
        with col_opt2:
            missing_strategy = st.selectbox(
                "Missing Value Strategy:",
                [
                    "Fill with average (numbers) & most common (text) — Recommended",
                    "Fill missing numbers with median & text with most common value",
                    "Fill missing numbers with zero & text with 'Unknown'",
                    "Drop Rows with Missing Values",
                    "Keep Missing Values As-Is"
                ]
            )

        col_save1, col_save2 = st.columns(2)
        with col_save1:
            save_mode = st.radio(
                "Save Destination:",
                ["Update active session in-place", "Save as new dataset file"],
                horizontal=True,
                key="clean_save_mode"
            )
        with col_save2:
            custom_new_name = ""
            if save_mode == "Save as new dataset file":
                base_name_clean = Path(selected_filename or "dataset.csv").stem
                while base_name_clean.lower().startswith("cleaned_"):
                    base_name_clean = base_name_clean[len("cleaned_"):]
                custom_new_name = st.text_input(
                    "New dataset filename (optional):",
                    value="",
                    placeholder=f"cleaned_{base_name_clean}.csv",
                    key="clean_custom_name"
                )

        if st.button("Apply cleaning", type="primary"):
            cleaned_df = df.copy()

            if do_remove_dups:
                cleaned_df = remove_duplicates(cleaned_df)

            strat_lower = missing_strategy.lower()
            if "average" in strat_lower or "mean" in strat_lower:
                cleaned_df = fill_missing_values(cleaned_df, numeric_strategy="mean", categorical_strategy="mode")
            elif "median" in strat_lower:
                cleaned_df = fill_missing_values(cleaned_df, numeric_strategy="median", categorical_strategy="mode")
            elif "zero" in strat_lower:
                cleaned_df = fill_missing_values(cleaned_df, numeric_strategy="zero", categorical_strategy="unknown")
            elif "drop" in strat_lower:
                cleaned_df = drop_missing_values(cleaned_df)

            is_already_clean = (dup_count == 0 and missing_count == 0)

            if is_already_clean:
                st.info("Dataset already has 0 duplicate rows and 0 missing values. No rows or values needed modification.")

            base_name = Path(selected_filename or "dataset.csv").stem
            while base_name.lower().startswith("cleaned_"):
                base_name = base_name[len("cleaned_"):]

            if save_mode == "Update active session in-place" or (is_already_clean and not custom_new_name.strip()):
                cleaned_name = selected_filename or "session_cleaned.csv"
                set_active_dataset(cleaned_df, cleaned_name)
                if is_already_clean:
                    st.success(f"Dataset `{cleaned_name}` is already clean. Active session retained without duplicating files.")
                else:
                    st.success(f"Active session dataset `{cleaned_name}` updated in-place.")
            else:
                user_stem = custom_new_name.strip() if custom_new_name.strip() else f"cleaned_{base_name}.csv"
                if not user_stem.lower().endswith(".csv"):
                    user_stem += ".csv"
                cleaned_name = get_unique_filename(user_stem, directory=dataset_folder)
                cleaned_path = dataset_folder / cleaned_name
                dataset_folder.mkdir(parents=True, exist_ok=True)
                atomic_write(cleaned_path, cleaned_df.to_csv(index=False), mode="w", encoding="utf-8")
                invalidate_dataset_cache(cleaned_name)
                set_active_dataset(cleaned_df, cleaned_name)
                st.success(f"Cleaned dataset saved as `{cleaned_name}`.")

            is_stored = (source_option == "Select from Datasets Folder")
            src_fp = dataset_fingerprint(selected_filename) if (is_stored and selected_filename) else dataframe_fingerprint(df)
            st.session_state["last_clean_result"] = {
                "source": selected_filename,
                "source_fingerprint": src_fp,
                "name": cleaned_name,
                "df": cleaned_df,
                "orig_rows": df.shape[0],
            }

    result = st.session_state.get("last_clean_result")
    is_stored = (source_option == "Select from Datasets Folder")
    current_fp = dataset_fingerprint(selected_filename) if (is_stored and selected_filename) else dataframe_fingerprint(df)
    if (
        result
        and result.get("source") == selected_filename
        and (not result.get("source_fingerprint") or result.get("source_fingerprint") == current_fp)
    ):
        cleaned_df = result["df"]
        cleaned_name = result["name"]

        with st.container(border=True):
            section_card_header("Cleaned Result & Export", f"Post-cleaning profile for `{cleaned_name}`")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                diff_rows = cleaned_df.shape[0] - result["orig_rows"]
                if diff_rows == 0:
                    delta_label = "0 rows removed"
                    delta_color = "off"
                else:
                    delta_label = f"{abs(diff_rows):,} rows removed"
                    delta_color = "normal"
                st.metric("Rows", f"{cleaned_df.shape[0]:,}", delta=delta_label, delta_color=delta_color)
            with c2:
                st.metric("Columns", cleaned_df.shape[1])
            with c3:
                st.metric("Remaining Duplicates", f"{int(cleaned_df.duplicated().sum()):,}")
            with c4:
                st.metric("Remaining Missing", f"{int(cleaned_df.isnull().sum().sum()):,}")

            st.dataframe(cleaned_df.head(10), width="stretch", hide_index=True)

            c_dl, c_gap = st.columns([2, 3])
            with c_dl:
                csv_bytes = sanitize_for_csv_export(cleaned_df).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download cleaned CSV",
                    data=csv_bytes,
                    file_name=cleaned_name,
                    mime="text/csv",
                    use_container_width=True
                )

            with st.expander("Export cleaned dataset to Amazon S3"):
                c_s3_1, c_s3_2 = st.columns(2)
                with c_s3_1:
                    clean_s3_bucket = st.text_input(
                        "S3 Bucket Name:",
                        value=st.session_state.get("s3_bucket", ""),
                        placeholder="e.g. my-analytics-bucket",
                        key="clean_s3_bucket"
                    )
                with c_s3_2:
                    clean_s3_region = st.text_input(
                        "AWS Region:",
                        value=st.session_state.get("s3_region", "us-east-1"),
                        placeholder="e.g. us-east-1",
                        key="clean_s3_region"
                    )

                clean_s3_key = st.text_input(
                    "Destination S3 Object Key:",
                    value=f"cleaned/{cleaned_name}",
                    key="clean_s3_key"
                )

                if st.button("Upload to S3", type="primary", key="btn_clean_upload_s3"):
                    if not clean_s3_bucket.strip():
                        st.error("Please specify a target S3 bucket name.")
                    else:
                        from Utils.S3 import get_s3_client, upload_s3_dataset, describe_s3_error
                        from Utils.secrets import value_of
                        k = value_of("aws_access")
                        s = value_of("aws_secret")
                        t = value_of("aws_session") or None
                        s3_client = get_s3_client(k, s, region_name=clean_s3_region.strip(), aws_session_token=t)
                        with st.spinner(f"Uploading `{cleaned_name}` to `s3://{clean_s3_bucket.strip()}/{clean_s3_key.strip()}`..."):
                            try:
                                upload_s3_dataset(cleaned_df, clean_s3_bucket.strip(), clean_s3_key.strip(), s3_client, sanitize=True)
                                st.success(f"Successfully uploaded cleaned dataset to `s3://{clean_s3_bucket.strip()}/{clean_s3_key.strip()}`!")
                            except Exception as up_err:
                                st.error(f"S3 export failed: {describe_s3_error(up_err, action='upload', bucket=clean_s3_bucket.strip())}")

render_sidebar()


import io
from pathlib import Path

import streamlit as st

from Utils.dataset_ui import (
    dataframe_fingerprint,
    dataset_fingerprint,
    invalidate_dataset_cache,
    load_dataset_cached,
    render_sidebar,
    save_and_activate_dataset,
    set_active_dataset,
)
from Utils.paths import (
    DATASETS_DIR,
    SUPPORTED_DATASET_EXTENSIONS,
    AIConversionRequired,
    atomic_write,
    get_unique_filename,
    list_ready_datasets,
    read_tabular,
    sanitize_for_csv_export,
)
from Utils.Preprocessing import drop_missing_values, fill_missing_values, remove_duplicates
from Utils.theme import page_header, section_card_header

page_header(
    "Data Cleaning & Imputation",
    "Deduplication, statistical imputation, and sanitized catalog export with atomic filesystem integrity.",
    badge="PREPROCESSING"
)

dataset_folder = DATASETS_DIR
available_files = list_ready_datasets()

with st.container(border=True):
    section_card_header("Source Selection", "Choose an existing dataset or upload a new file")
    source_option = st.radio(
        "Choose dataset source",
        ["Select from datasets folder", "Upload new file"],
        horizontal=True
    )

    df = None
    selected_filename = None

    if source_option == "Select from datasets folder":
        if not available_files:
            st.info("No datasets found in `Datasets/`. Ingest one on the Data Hub & Ingest page first.")
        else:
            selected_filename = st.selectbox("Select dataset to clean", available_files)
            try:
                df = load_dataset_cached(selected_filename)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        uploaded_file = st.file_uploader(
            "Upload dataset to clean",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)
        )
        if uploaded_file:
            selected_filename = uploaded_file.name
            try:
                df = read_tabular(uploaded_file, filename=selected_filename)
            except AIConversionRequired:
                st.warning(
                    f"`{selected_filename}` does not contain a standard tabular schema. "
                    "Please ingest a valid CSV, Parquet, Excel, or JSON dataset on the Data Hub page."
                )
            except Exception as e:
                st.error(f"Error reading file: {e}")

if df is not None:
    is_stored = (source_option == "Select from datasets folder")
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
            st.metric("Duplicate Rows", f"{dup_count:,}")
        with col4:
            st.metric("Missing Values", f"{missing_count:,}")

        with st.expander("Preview original data"):
            st.dataframe(df.head(10), width="stretch", hide_index=True)

    with st.container(border=True):
        section_card_header("Cleaning Pipeline Rules", "Configure deduplication, imputation, and where the result is saved")
        col_opt1, col_opt2 = st.columns([1, 1.8])
        with col_opt1:
            do_remove_dups = st.checkbox("Remove duplicate rows", value=True)
            st.caption("Drops exact duplicates before imputation.")
        with col_opt2:
            missing_strategy = st.selectbox(
                "Missing value strategy",
                [
                    "Mean imputation (numeric) / Mode imputation (categorical)",
                    "Median imputation (numeric) / Mode imputation (categorical)",
                    "Constant imputation (zero for numeric / 'Unknown' for categorical)",
                    "Complete-case analysis (drop incomplete rows)",
                    "Preserve missing values (retain nulls without modification)",
                ],
                help="Applies per column: numeric columns use the chosen statistic, categorical columns fall back to modal or indicator values.",
            )

        st.markdown("")

        col_save1, col_save2 = st.columns([1, 1])
        with col_save1:
            save_mode = st.radio(
                "Save destination",
                ["Update active session in-place", "Save as new dataset file"],
                horizontal=True,
                key="clean_save_mode",
            )
        with col_save2:
            custom_new_name = ""
            if save_mode == "Save as new dataset file":
                base_name_clean = Path(selected_filename or "dataset.csv").stem
                while base_name_clean.lower().startswith("cleaned_"):
                    base_name_clean = base_name_clean[len("cleaned_"):]
                custom_new_name = st.text_input(
                    "New dataset filename (optional)",
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
                st.info("Dataset exhibits 100% completeness and uniqueness; no deduplication or imputation required.")

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
                atomic_write(cleaned_path, sanitize_for_csv_export(cleaned_df).to_csv(index=False), mode="w", encoding="utf-8")
                invalidate_dataset_cache(cleaned_name)
                set_active_dataset(cleaned_df, cleaned_name)
                st.success(f"Cleaned dataset saved as `{cleaned_name}`.")

            is_stored = (source_option == "Select from datasets folder")
            src_fp = dataset_fingerprint(selected_filename) if (is_stored and selected_filename) else dataframe_fingerprint(df)
            st.session_state["last_clean_result"] = {
                "source": selected_filename,
                "source_fingerprint": src_fp,
                "name": cleaned_name,
                "df": cleaned_df,
                "orig_rows": df.shape[0],
            }

    result = st.session_state.get("last_clean_result")
    is_stored = (source_option == "Select from datasets folder")
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
                st.metric(
                    "Rows",
                    f"{cleaned_df.shape[0]:,}",
                    delta=f"{abs(diff_rows):,} rows removed" if diff_rows else None,
                )
            with c2:
                st.metric("Columns", cleaned_df.shape[1])
            with c3:
                st.metric("Remaining Duplicates", f"{int(cleaned_df.duplicated().sum()):,}")
            with c4:
                st.metric("Remaining Missing", f"{int(cleaned_df.isnull().sum().sum()):,}")

            st.dataframe(cleaned_df.head(10), width="stretch", hide_index=True)

            c_dl, c_parquet, c_save = st.columns(3)
            with c_dl:
                csv_bytes = sanitize_for_csv_export(cleaned_df).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=cleaned_name if cleaned_name.endswith(".csv") else f"{cleaned_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c_parquet:
                parquet_buf = io.BytesIO()
                cleaned_df.to_parquet(parquet_buf, index=False)
                st.download_button(
                    label="Download Parquet",
                    data=parquet_buf.getvalue(),
                    file_name=f"{cleaned_name.rsplit('.', 1)[0]}.parquet",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            with c_save:
                if st.button("Save to Workspace Catalog", use_container_width=True):
                    try:
                        saved_path = save_and_activate_dataset(cleaned_df, cleaned_name, sanitize=True)
                        st.success(f"Saved `{saved_path.name}` to Datasets catalog.")
                    except Exception as save_err:
                        st.error(f"Save failed: {save_err}")

render_sidebar()

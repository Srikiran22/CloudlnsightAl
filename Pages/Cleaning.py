import streamlit as st
from pathlib import Path

from Utils.Preprocessing import remove_duplicates, fill_missing_values, drop_missing_values
from Utils.paths import (
    AIConversionRequired, DATASETS_DIR, list_dataset_files, read_dataset,
    read_tabular, SUPPORTED_DATASET_EXTENSIONS,
)
from Utils.dataset_ui import render_sidebar

st.title("Data cleaning")
st.markdown("Remove duplicate rows and resolve missing values, then save the result as a new dataset.")

dataset_folder = DATASETS_DIR
available_files = list_dataset_files()

source_option = st.radio(
    "Choose Dataset Source:",
    ["Select from Datasets Folder", "Upload New File"],
    horizontal=True
)

df = None
selected_filename = None

if source_option == "Select from Datasets Folder":
    if not available_files:
        st.info("No datasets found in `Datasets/`. Ingest one on the Upload page first.")
    else:
        selected_filename = st.selectbox("Select Dataset to Clean:", available_files)
        try:
            df = read_dataset(selected_filename)
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
    st.subheader("Original dataset")
    dup_count = int(df.duplicated().sum())
    missing_count = int(df.isnull().sum().sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Duplicate Rows", f"{dup_count:,}")
    with col4:
        st.metric("Missing Values", f"{missing_count:,}")

    with st.expander("Preview Original Data"):
        st.dataframe(df.head(10), width="stretch")

    st.subheader("Cleaning options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        do_remove_dups = st.checkbox("Remove Duplicate Rows", value=True)
    with col_opt2:
        missing_strategy = st.selectbox(
            "Missing Value Strategy:",
            [
                "Impute (Mean for Numeric, Mode for Categorical)",
                "Impute (Median for Numeric, Mode for Categorical)",
                "Impute (Zero for Numeric, Unknown for Categorical)",
                "Drop Rows with Missing Values",
                "Keep Missing Values As-Is"
            ]
        )

    if st.button("Apply cleaning", type="primary"):
        cleaned_df = df.copy()

        if do_remove_dups:
            cleaned_df = remove_duplicates(cleaned_df)

        if missing_strategy == "Impute (Mean for Numeric, Mode for Categorical)":
            cleaned_df = fill_missing_values(cleaned_df, numeric_strategy="mean", categorical_strategy="mode")
        elif missing_strategy == "Impute (Median for Numeric, Mode for Categorical)":
            cleaned_df = fill_missing_values(cleaned_df, numeric_strategy="median", categorical_strategy="mode")
        elif missing_strategy == "Impute (Zero for Numeric, Unknown for Categorical)":
            cleaned_df = fill_missing_values(cleaned_df, numeric_strategy="zero", categorical_strategy="unknown")
        elif missing_strategy == "Drop Rows with Missing Values":
            cleaned_df = drop_missing_values(cleaned_df)

        base_name = Path(selected_filename or "dataset.csv").stem
        while base_name.lower().startswith("cleaned_"):
            base_name = base_name[len("cleaned_"):]
        cleaned_name = f"cleaned_{base_name}.csv"
        cleaned_path = dataset_folder / cleaned_name
        dataset_folder.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(cleaned_path, index=False)
        st.session_state["current_df"] = cleaned_df
        st.session_state["dataset_name"] = cleaned_name
        st.session_state["last_clean_result"] = {
            "source": selected_filename,
            "name": cleaned_name,
            "df": cleaned_df,
            "orig_rows": df.shape[0],
        }
        st.success(f"Cleaned dataset saved as `{cleaned_name}`.")

    result = st.session_state.get("last_clean_result")
    if result and result.get("source") == selected_filename:
        cleaned_df = result["df"]
        cleaned_name = result["name"]

        st.subheader("Cleaned result")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{cleaned_df.shape[0]:,}", delta=cleaned_df.shape[0] - result["orig_rows"])
        with c2:
            st.metric("Columns", cleaned_df.shape[1])
        with c3:
            st.metric("Remaining Duplicates", f"{int(cleaned_df.duplicated().sum()):,}")
        with c4:
            st.metric("Remaining Missing", f"{int(cleaned_df.isnull().sum().sum()):,}")

        st.dataframe(cleaned_df.head(10), width="stretch")

        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download cleaned CSV",
            data=csv_bytes,
            file_name=cleaned_name,
            mime="text/csv"
        )

render_sidebar()

import streamlit as st
import pandas as pd

from Utils.compare_logic import column_drift_rows, schema_diff
from Utils.paths import list_dataset_files
from Utils.dataset_ui import render_sidebar, load_dataset_cached

st.title("Compare datasets")
st.markdown("Schema differences, missing-value drift, and numeric distribution shifts between two files.")

available = list_dataset_files()
if len(available) < 2:
    st.warning("Need at least two datasets in the `Datasets/` folder to compare. Ingest more data first.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    name_a = st.selectbox("Dataset A:", available, index=0)
with col_b:
    default_b = 1 if len(available) > 1 else 0
    name_b = st.selectbox("Dataset B:", available, index=default_b)

render_sidebar()

if name_a == name_b:
    st.warning("Selected the same dataset twice — pick two different files.")
    st.stop()

try:
    df_a = load_dataset_cached(name_a)
    df_b = load_dataset_cached(name_b)
except Exception as error:
    st.error(f"Failed to load datasets: {error}")
    st.stop()

# overview
st.subheader("Overview")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Rows — A", f"{df_a.shape[0]:,}", delta=f"{df_b.shape[0] - df_a.shape[0]:,} vs B")
with m2:
    st.metric("Rows — B", f"{df_b.shape[0]:,}")
with m3:
    st.metric("Columns — A", df_a.shape[1])
with m4:
    st.metric("Columns — B", df_b.shape[1])

common, only_a, only_b = schema_diff(df_a, df_b)

st.subheader("Schema differences")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"**Common columns** ({len(common)})")
    if common:
        st.caption(", ".join(str(c) for c in common))
    else:
        st.caption("None")
with s2:
    st.markdown(f"**Only in A** ({len(only_a)})")
    st.caption(", ".join(str(c) for c in only_a) if only_a else "None")
with s3:
    st.markdown(f"**Only in B** ({len(only_b)})")
    st.caption(", ".join(str(c) for c in only_b) if only_b else "None")

# per-column drift table
if not common:
    st.info("The two datasets share no columns — nothing to compare at column level.")
    st.stop()

st.subheader("Column-level drift")

drift_df = pd.DataFrame(column_drift_rows(df_a, df_b))
flagged_count = int((drift_df["Flags"] != "OK").sum())
st.dataframe(drift_df, width="stretch", hide_index=True)

if flagged_count:
    st.warning(f"{flagged_count} of {len(common)} common columns show notable drift (dtype change, ≥10% missingness change, or ≥10% mean shift).")
else:
    st.success("No significant drift detected across common columns.")

# duplication profile
st.subheader("Duplication profile")
d1, d2 = st.columns(2)
with d1:
    dup_a = int(df_a.duplicated().sum())
    st.metric("Duplicate Rows — A", f"{dup_a:,}")
with d2:
    dup_b = int(df_b.duplicated().sum())
    st.metric("Duplicate Rows — B", f"{dup_b:,}")

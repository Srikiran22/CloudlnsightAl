import streamlit as st
import pandas as pd

from Utils.compare_logic import column_drift_rows, schema_diff
from Utils.paths import ResourcePolicy, get_dataset_row_count, list_ready_datasets
from Utils.dataset_ui import render_sidebar, load_dataset_cached
from Utils.sampling import sample_for_analysis

st.title("Compare datasets")
st.markdown("Schema differences, missing-value drift, and summary metric shifts between two files.")

available = list_ready_datasets()
if len(available) < 2:
    st.warning("Need at least two ready tabular datasets in the `Datasets/` folder to compare. Ingest more data first.")
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
    row_count_a = get_dataset_row_count(name_a)
    row_count_b = get_dataset_row_count(name_b)
    # P-09: Enforce resource bound at ingestion/load rather than loading full multi-gigabyte datasets first
    df_a = load_dataset_cached(name_a, max_rows=ResourcePolicy.MAX_ANALYSIS_ROWS)
    df_b = load_dataset_cached(name_b, max_rows=ResourcePolicy.MAX_ANALYSIS_ROWS)
    total_a = row_count_a if row_count_a is not None else len(df_a)
    total_b = row_count_b if row_count_b is not None else len(df_b)
except Exception as error:
    st.error(f"Failed to load datasets: {error}")
    st.stop()

# overview
st.subheader("Overview")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Rows — A", f"{total_a:,}", delta=f"{total_b - total_a:,} vs B")
with m2:
    st.metric("Total Rows — B", f"{total_b:,}")
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

# per-column comparison table
if not common:
    st.info("The two datasets share no columns — nothing to compare at column level.")
    st.stop()

st.subheader("Column-level metric comparison & shifts")

if total_a > len(df_a) or total_b > len(df_b):
    st.info(
        f"Analytical scope: Column metric comparison and statistical drift tests are evaluated on a representative sample of "
        f"{len(df_a):,} rows for A (out of {total_a:,}) and {len(df_b):,} rows for B (out of {total_b:,})."
    )

drift_df = pd.DataFrame(column_drift_rows(df_a, df_b))
flagged_count = int((drift_df["Flags"] != "OK").sum())
st.dataframe(drift_df, width="stretch", hide_index=True)
st.caption(
    "Statistical methodology: Numeric distribution drift evaluated via two-sample Kolmogorov-Smirnov tests with "
    "Benjamini-Hochberg False Discovery Rate (FDR) correction (exploratory multi-column screening). "
    "Categorical drift evaluated via Total Variation Distance (TVD >= 0.20)."
)

if flagged_count:
    st.warning(f"{flagged_count} of {len(common)} common columns show notable metric shifts or statistical drift (dtype, missingness, mean, dispersion, KS distribution drift, or categorical TVD drift).")
else:
    st.success("No significant metric shifts or statistical distribution drift detected across common columns.")

# duplication profile
st.subheader("Duplication profile")
d1, d2 = st.columns(2)
with d1:
    sample_dup_a, is_a_sampled, _ = sample_for_analysis(df_a)
    dup_a = int(sample_dup_a.duplicated().sum())
    lbl_a = f"Duplicate Rows — A (Sample of {len(sample_dup_a):,})" if is_a_sampled else "Duplicate Rows — A"
    st.metric(lbl_a, f"{dup_a:,}")
with d2:
    sample_dup_b, is_b_sampled, _ = sample_for_analysis(df_b)
    dup_b = int(sample_dup_b.duplicated().sum())
    lbl_b = f"Duplicate Rows — B (Sample of {len(sample_dup_b):,})" if is_b_sampled else "Duplicate Rows — B"
    st.metric(lbl_b, f"{dup_b:,}")

import streamlit as st
import pandas as pd

from Utils.paths import list_dataset_files
from Utils.dataset_ui import render_sidebar, load_dataset_cached

st.title("⚖️ Dataset Comparison")
st.markdown("Compare two datasets side by side: schema differences, missing-value drift, and numeric distribution shifts.")

available = list_dataset_files()
if len(available) < 2:
    st.warning("⚠️ Need at least two datasets in the `Datasets/` folder to compare. Upload more data first.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    name_a = st.selectbox("Dataset A:", available, index=0)
with col_b:
    default_b = 1 if len(available) > 1 else 0
    name_b = st.selectbox("Dataset B:", available, index=default_b)

render_sidebar()

if name_a == name_b:
    st.warning("⚠️ Selected the same dataset twice — pick two different files.")
    st.stop()

try:
    df_a = load_dataset_cached(name_a)
    df_b = load_dataset_cached(name_b)
except Exception as error:
    st.error(f"❌ Failed to load datasets: {error}")
    st.stop()

# ----------------------------------------------------------------------
# Overview metrics
# ----------------------------------------------------------------------
st.subheader("📐 Overview")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Rows — A", f"{df_a.shape[0]:,}", delta=f"{df_b.shape[0] - df_a.shape[0]:,} vs B")
with m2:
    st.metric("Rows — B", f"{df_b.shape[0]:,}")
with m3:
    st.metric("Columns — A", df_a.shape[1])
with m4:
    st.metric("Columns — B", df_b.shape[1])

cols_a = set(df_a.columns)
cols_b = set(df_b.columns)
common = sorted(cols_a & cols_b)
only_a = sorted(cols_a - cols_b)
only_b = sorted(cols_b - cols_a)

# ----------------------------------------------------------------------
# Schema differences
# ----------------------------------------------------------------------
st.subheader("🧬 Schema Differences")
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

# ----------------------------------------------------------------------
# Column-level drift table
# ----------------------------------------------------------------------
if not common:
    st.info("The two datasets share no columns — nothing to compare at column level.")
    st.stop()

st.subheader("📊 Column-Level Drift")


def _missing_pct(series: pd.Series) -> float:
    return series.isnull().mean() * 100


def _numeric_mean(series: pd.Series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


drift_rows = []
for col in common:
    sa, sb = df_a[col], df_b[col]
    dtype_match = str(sa.dtype) == str(sb.dtype)
    miss_a, miss_b = _missing_pct(sa), _missing_pct(sb)
    mean_a, mean_b = _numeric_mean(sa), _numeric_mean(sb)

    mean_drift = None
    if mean_a is not None and mean_b is not None:
        denominator = abs(mean_a) if abs(mean_a) > 1e-12 else 1.0
        mean_drift = (mean_b - mean_a) / denominator * 100

    flags = []
    if not dtype_match:
        flags.append(f"dtype {sa.dtype}→{sb.dtype}")
    if abs(miss_b - miss_a) >= 10:
        flags.append(f"missingness Δ{miss_b - miss_a:+.1f}%")
    if mean_drift is not None and abs(mean_drift) >= 10:
        flags.append(f"mean shift {mean_drift:+.1f}%")

    drift_rows.append({
        "Column": col,
        "Dtype Match": "✅" if dtype_match else "❌",
        "Missing % A": round(miss_a, 1),
        "Missing % B": round(miss_b, 1),
        "Mean A": round(mean_a, 3) if mean_a is not None else "-",
        "Mean B": round(mean_b, 3) if mean_b is not None else "-",
        "Unique A": int(sa.nunique(dropna=True)),
        "Unique B": int(sb.nunique(dropna=True)),
        "Flags": "; ".join(flags) if flags else "OK",
    })

drift_df = pd.DataFrame(drift_rows)
flagged_count = int((drift_df["Flags"] != "OK").sum())
st.dataframe(drift_df, width="stretch", hide_index=True)

if flagged_count:
    st.warning(f"⚠️ {flagged_count} of {len(common)} common columns show notable drift (dtype change, ≥10% missingness change, or ≥10% mean shift).")
else:
    st.success("✅ No significant drift detected across common columns.")

# ----------------------------------------------------------------------
# Duplicate profile comparison
# ----------------------------------------------------------------------
st.subheader("🔁 Duplication Profile")
d1, d2 = st.columns(2)
with d1:
    dup_a = int(df_a.duplicated().sum())
    st.metric("Duplicate Rows — A", f"{dup_a:,}")
with d2:
    dup_b = int(df_b.duplicated().sum())
    st.metric("Duplicate Rows — B", f"{dup_b:,}")

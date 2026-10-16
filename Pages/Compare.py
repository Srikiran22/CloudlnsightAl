import math
import streamlit as st
import pandas as pd

from Utils.compare_logic import column_drift_rows, schema_diff
from Utils.paths import ResourcePolicy, get_dataset_row_count, list_ready_datasets
from Utils.dataset_ui import render_sidebar, load_dataset_cached
from Utils.sampling import sample_for_analysis
from Utils.theme import page_header, section_card_header
from Utils.ML import is_likely_id_column

page_header(
    "Compare Datasets",
    "Schema differences, missing-value drift, and summary metric shifts between two files."
)

available = list_ready_datasets()
if len(available) < 2:
    st.warning("Need at least two ready tabular datasets in the Datasets folder to compare. Ingest more data first.")
    st.stop()

with st.container(border=True):
    section_card_header("Dataset Pair Selection", "Select baseline (A) and target comparison (B) datasets")
    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.selectbox("Dataset A:", available, index=0)
    with col_b:
        default_b = 0
        for idx, name in enumerate(available):
            if idx != 0:
                if any(token in name.lower() for token in ["drift", "q2", "compare", "target"]):
                    default_b = idx
                    break
        if default_b == 0 and len(available) > 1:
            default_b = 1
        name_b = st.selectbox("Dataset B:", available, index=default_b)

render_sidebar()

if name_a == name_b:
    st.warning("Selected the same dataset twice — pick two different files.")
    st.stop()

try:
    row_count_a = get_dataset_row_count(name_a)
    row_count_b = get_dataset_row_count(name_b)
    df_a = load_dataset_cached(name_a, max_rows=ResourcePolicy.MAX_ANALYSIS_ROWS)
    df_b = load_dataset_cached(name_b, max_rows=ResourcePolicy.MAX_ANALYSIS_ROWS)
    total_a = row_count_a if row_count_a is not None else len(df_a)
    total_b = row_count_b if row_count_b is not None else len(df_b)
except Exception as error:
    st.error(f"Failed to load datasets: {error}")
    st.stop()

common, only_a, only_b = schema_diff(df_a, df_b)
drift_rows = column_drift_rows(df_a, df_b) if common else []
drift_df = pd.DataFrame(drift_rows)

id_cols = {col for col in common if is_likely_id_column(df_a, col) or is_likely_id_column(df_b, col)}
analytical_cols = [c for c in common if c not in id_cols]
flagged_count = sum(1 for r in drift_rows if r["Column"] not in id_cols and r.get("Flags") != "OK")

with st.container(border=True):
    section_card_header("Executive Comparison Summary", f"Comparing `{name_a}` (A) vs `{name_b}` (B)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        row_diff = total_b - total_a
        delta_str = f"{row_diff:+,} vs B" if row_diff != 0 else "0 vs B"
        delta_col = "normal" if row_diff != 0 else "off"
        st.metric("Total Rows — A", f"{total_a:,}", delta=delta_str, delta_color=delta_col)
    with m2:
        st.metric("Total Rows — B", f"{total_b:,}")
    with m3:
        st.metric("Columns Matched", f"{len(common)}/{max(df_a.shape[1], df_b.shape[1])}")
    with m4:
        st.metric(
            "Drifted Features",
            f"{flagged_count} of {len(analytical_cols)}",
            delta="Stable" if flagged_count == 0 else f"{flagged_count} drift",
            delta_color="off" if flagged_count == 0 else "inverse"
        )

if not common:
    st.info("The two datasets share no columns — nothing to compare at column level.")
    st.stop()

tab_drift, tab_schema, tab_dups = st.tabs(["Drift & Changes", "Columns & Missing Data", "Duplicate Check"])

with tab_drift:
    with st.container(border=True):
        section_card_header("Column-Level Metric Comparison & Statistical Drift", "Evaluated via Kolmogorov-Smirnov (KS) tests and Total Variation Distance (TVD)")
        if total_a > len(df_a) or total_b > len(df_b):
            st.info(
                f"Analytical scope: Evaluated on a representative sample of "
                f"{len(df_a):,} rows for A (out of {total_a:,}) and {len(df_b):,} rows for B (out of {total_b:,})."
            )

        if flagged_count > 0:
            st.info(f"Found significant statistical changes in **{flagged_count} of {len(analytical_cols)}** analytical columns between `{name_a}` and `{name_b}`.")
        else:
            st.success(f"All **{len(analytical_cols)}** analytical columns are stable between `{name_a}` and `{name_b}`.")

        display_df = drift_df.copy()

        def _get_status(row):
            col = row.get("Column")
            if col in id_cols:
                return "ID / Key"
            flags = row.get("Flags")
            return "Drift detected" if flags and flags != "OK" else "Stable"

        def _plain_summary(row):
            col = row.get("Column")
            if col in id_cols:
                return "Unique ID / key (omitted from drift test)"
            flags = str(row.get("Flags", "OK"))
            if flags == "OK":
                return "No significant change"
            mean_shift = row.get("Mean Shift %")
            if mean_shift is not None and mean_shift != "—" and not str(mean_shift).startswith("N/A"):
                try:
                    val = float(mean_shift)
                    if not math.isnan(val):
                        return f"Average shifted by {val:+.1f}%"
                except (ValueError, TypeError):
                    pass
            tvd = row.get("Categorical TVD")
            if tvd is not None and tvd != "—":
                try:
                    tvd_val = float(tvd)
                    if not math.isnan(tvd_val) and tvd_val >= 0.20:
                        return "Category mix changed"
                except (ValueError, TypeError):
                    pass
            if "distribution drift" in flags:
                return "Value distribution shifted"
            if "missingness" in flags:
                return "Missing rate changed"
            if "dtype" in flags:
                return "Data type changed"
            return "Distribution shifted"

        display_df.insert(0, "Status", display_df.apply(_get_status, axis=1))
        display_df.insert(2, "Plain-English Summary", display_df.apply(_plain_summary, axis=1))

        show_technical = st.checkbox("Show technical statistical columns (KS test, FDR q-value, TVD)", value=False)

        for col in display_df.columns:
            if col not in ("Status", "Plain-English Summary", "Column"):
                display_df[col] = display_df[col].apply(
                    lambda v: "—" if (v is None or pd.isna(v) or str(v).strip() in {"None", "nan", "NaN"})
                    else (f"{v:.3f}" if isinstance(v, float) else str(v))
                )

        if not show_technical:
            core_cols = [
                "Status", "Column", "Plain-English Summary", "Dtype Match",
                "Missing % A", "Missing % B", "Mean A", "Mean B", "Mean Shift %"
            ]
            display_cols = [c for c in core_cols if c in display_df.columns]
            st.dataframe(display_df[display_cols], width="stretch", hide_index=True)
        else:
            st.dataframe(display_df, width="stretch", hide_index=True)

        st.caption(
            "Statistical methodology: Numeric distribution drift evaluated via two-sample Kolmogorov-Smirnov tests with "
            "Benjamini-Hochberg False Discovery Rate (FDR) correction (exploratory multi-column screening). "
            "Categorical drift evaluated via Total Variation Distance (TVD >= 0.20)."
        )

with tab_schema:
    with st.container(border=True):
        section_card_header("Schema Differences & Column Membership", "Columns present in both or unique to one dataset")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"**Common columns** ({len(common)})")
            st.caption(", ".join(str(c) for c in common) if common else "None")
        with s2:
            st.markdown(f"**Only in A** ({len(only_a)})")
            st.caption(", ".join(str(c) for c in only_a) if only_a else "None")
        with s3:
            st.markdown(f"**Only in B** ({len(only_b)})")
            st.caption(", ".join(str(c) for c in only_b) if only_b else "None")

with tab_dups:
    with st.container(border=True):
        section_card_header("Duplication Profile", "Duplicate row counts across datasets")
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

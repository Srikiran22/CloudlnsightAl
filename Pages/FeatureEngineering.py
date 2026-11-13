import streamlit as st

from Utils.dataset_ui import (
    render_sidebar,
    save_and_activate_dataset,
    select_working_dataset,
)
from Utils.feature_engineering import (
    compute_mutual_information,
    create_interaction_feature,
    encode_features,
    scale_features,
    transform_power,
)
from Utils.ML import detect_problem_type, is_likely_id_column
from Utils.theme import metric_card, page_header, section_card_header

page_header(
    "Feature Engineering Studio",
    "Scale numerical features, encode categorical dimensions, apply non-linear transforms, and screen feature importance.",
    badge="DATA PREPARATION"
)

df, selected_file = select_working_dataset("Active dataset")
render_sidebar()

current_fe_origin = f"{selected_file}:{len(df)}:{list(df.columns)}"
if st.session_state.get("fe_dataset_origin") != current_fe_origin:
    st.session_state["fe_dataset_origin"] = current_fe_origin
    st.session_state["fe_working_df"] = df.copy()

if df.empty or len(df.columns) < 2:
    st.warning("Active dataset must contain at least 2 columns and at least 2 rows for feature engineering.")
    st.stop()

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
id_cols = [c for c in all_cols if is_likely_id_column(df, c)]

with st.container(border=True):
    section_card_header("Feature Space Overview", f"Dimensions and types for `{selected_file}`")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Total Features", len(all_cols))
    with m2:
        metric_card("Numeric Features", len(numeric_cols))
    with m3:
        metric_card("Categorical Features", len(cat_cols))
    with m4:
        metric_card("Likely Identifiers", len(id_cols), subtext="Recommended to exclude")

tab_scale, tab_encode, tab_transform, tab_select, tab_interact = st.tabs([
    "1. Numerical Scaling",
    "2. Categorical Encoding",
    "3. Mathematical Transforms",
    "4. Feature Selection & Importance",
    "5. Interaction Features"
])

# -----------------------------------------------------------------------------
# TAB 1: NUMERICAL SCALING
# -----------------------------------------------------------------------------
with tab_scale:
    with st.container(border=True):
        section_card_header(
            "Numerical Feature Scaling",
            "Normalize or standardize continuous numeric features for distance-based and gradient-descent algorithms."
        )
        if not numeric_cols:
            st.info("No numerical features found in active dataset.")
        else:
            scale_candidates = [c for c in numeric_cols if c not in id_cols]
            def is_binary_or_target(c):
                return (df[c].dropna().nunique() <= 2) or (c.lower() in ["churned", "churn", "target", "label", "is_churned"])
            scale_defaults = [c for c in scale_candidates if not is_binary_or_target(c)]
            selected_scale_cols = st.multiselect("Select columns to scale", scale_candidates, default=scale_defaults)
            scale_method = st.selectbox(
                "Scaling Strategy",
                ["StandardScaler (Mean=0, Std=1)", "MinMaxScaler (Range [0, 1])", "RobustScaler (Median & IQR, outlier robust)"]
            )

            if st.button("Apply Scaling", type="primary", key="btn_apply_scale"):
                if not selected_scale_cols:
                    st.warning("Please select at least one numeric column to scale.")
                else:
                    base_df = st.session_state.get("fe_working_df", df).copy()
                    base_df, new_cols = scale_features(base_df, selected_scale_cols, scale_method)
                    st.session_state["fe_working_df"] = base_df
                    st.success(f"Generated {len(new_cols)} scaled feature(s).")


# -----------------------------------------------------------------------------
# TAB 2: CATEGORICAL ENCODING
# -----------------------------------------------------------------------------
with tab_encode:
    with st.container(border=True):
        section_card_header(
            "Categorical Encoding",
            "Convert discrete categorical columns into numeric indicators for machine learning models."
        )
        if not cat_cols:
            st.info("No categorical features found in active dataset.")
        else:
            cat_candidates = [c for c in cat_cols if c not in id_cols]
            selected_cat_cols = st.multiselect("Select categorical columns to encode", cat_candidates, default=cat_candidates[:5])
            encode_method = st.radio("Encoding Strategy", ["One-Hot Encoding (Dummies)", "Ordinal / Integer Encoding"], horizontal=True)

            if st.button("Apply Categorical Encoding", type="primary", key="btn_apply_encode"):
                if not selected_cat_cols:
                    st.warning("Please select at least one column to encode.")
                else:
                    base_df = st.session_state.get("fe_working_df", df).copy()
                    base_df, new_cols = encode_features(base_df, selected_cat_cols, encode_method)
                    st.session_state["fe_working_df"] = base_df
                    if "One-Hot" in encode_method:
                        st.success(f"Generated {len(new_cols)} one-hot encoded column(s).")
                    else:
                        st.success(f"Generated {len(new_cols)} ordinal encoded feature(s).")


# -----------------------------------------------------------------------------
# TAB 3: MATHEMATICAL TRANSFORMS
# -----------------------------------------------------------------------------
with tab_transform:
    with st.container(border=True):
        section_card_header(
            "Mathematical & Distribution Transforms",
            "Stabilize variance, reduce positive skewness, and normalize feature distributions."
        )
        if not numeric_cols:
            st.info("No numerical features available.")
        else:
            skew_candidates = [c for c in numeric_cols if c not in id_cols]
            selected_tf_cols = st.multiselect("Select columns to transform", skew_candidates)
            transform_kind = st.selectbox(
                "Transformation",
                ["Log1p transform (log(1 + x))", "Square Root transform (sqrt(x))", "Yeo-Johnson power transform (variance stabilization)"]
            )

            if st.button("Apply Transform", type="primary", key="btn_apply_transform"):
                if not selected_tf_cols:
                    st.warning("Select columns to transform.")
                else:
                    base_df = st.session_state.get("fe_working_df", df).copy()
                    base_df, new_cols = transform_power(base_df, selected_tf_cols, transform_kind)
                    st.session_state["fe_working_df"] = base_df
                    st.success(f"Applied {transform_kind} to {len(new_cols)} column(s).")


# -----------------------------------------------------------------------------
# TAB 4: FEATURE SELECTION & IMPORTANCE SCREENING
# -----------------------------------------------------------------------------
with tab_select:
    with st.container(border=True):
        section_card_header(
            "Mutual Information & Feature Screening",
            "Screen non-linear feature dependency relative to a prediction target variable."
        )
        target_col = st.selectbox("Select Target Variable for Importance Ranking", all_cols, index=len(all_cols) - 1, key="fe_target_select")
        predictor_candidates = [c for c in numeric_cols if c != target_col and c not in id_cols]

        if not predictor_candidates:
            st.info("Need numerical predictors and a target to calculate Mutual Information.")
        else:
            prob_type = detect_problem_type(df, target_col)
            st.caption(f"Target problem type auto-detected: **{prob_type}**")

            if st.button("Calculate Mutual Information Scores", type="primary", key="btn_calc_mi"):
                with st.spinner("Computing mutual information dependency scores..."):
                    mi_df = compute_mutual_information(
                        df,
                        target_col=target_col,
                        feature_cols=predictor_candidates,
                        problem_type=prob_type,
                    )
                    st.session_state["fe_mi_scores"] = mi_df

            if "fe_mi_scores" in st.session_state:
                mi_df = st.session_state["fe_mi_scores"]
                st.dataframe(
                    mi_df.style.background_gradient(cmap="Blues", subset=["Mutual Information"]).format({"Mutual Information": "{:.4f}"}),
                    width="stretch",
                    hide_index=True
                )


# -----------------------------------------------------------------------------
# TAB 5: INTERACTION FEATURES
# -----------------------------------------------------------------------------
with tab_interact, st.container(border=True):
    section_card_header(
        "Feature Interactions & Ratios",
        "Compute pairwise interaction terms ($A \\times B$) or ratios ($A / B$) between key numerical features."
    )
    if len(numeric_cols) < 2:
        st.info("Need at least two numerical features to build interactions.")
    else:
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            feat_a = st.selectbox("Feature A", numeric_cols, key="fe_ia")
        with col_i2:
            feat_b = st.selectbox("Feature B", [c for c in numeric_cols if c != feat_a], key="fe_ib")
        with col_i3:
            interact_op = st.selectbox("Operation", ["Multiplication (A * B)", "Ratio (A / B)", "Difference (A - B)"])

        if st.button("Generate Interaction Feature", type="primary", key="btn_gen_interact"):
            base_df = st.session_state.get("fe_working_df", df).copy()
            base_df, new_col_name = create_interaction_feature(base_df, feat_a, feat_b, interact_op)
            st.session_state["fe_working_df"] = base_df
            st.success(f"Created interaction feature: `{new_col_name}`.")


# -----------------------------------------------------------------------------
# FINAL EXPORT & ACTIVATION
# -----------------------------------------------------------------------------
active_fe_df = st.session_state.get("fe_working_df")
if active_fe_df is not None and active_fe_df.shape != df.shape:
    st.markdown("---")
    with st.container(border=True):
        section_card_header(
            "Engineered Dataset Export",
            f"Original dataset: {df.shape[1]} columns → Engineered dataset: {active_fe_df.shape[1]} columns"
        )
        st.dataframe(active_fe_df.head(10), width="stretch", hide_index=True)

        c_save1, c_save2, c_save3 = st.columns([2, 1, 1])
        with c_save1:
            save_name_input = st.text_input(
                "Save engineered dataset as",
                value=f"engineered_{selected_file}"
            )
        with c_save2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Save & Activate Dataset", type="primary", key="btn_save_fe_dataset"):
                try:
                    target_name = save_name_input.strip() or f"engineered_{selected_file}"
                    saved_path = save_and_activate_dataset(active_fe_df, target_name, sanitize=True)
                    st.session_state.pop("fe_working_df", None)
                    st.session_state.pop("fe_dataset_origin", None)
                    st.success(f"Saved `{saved_path.name}` and registered as active workspace dataset.")
                    st.rerun()
                except Exception as save_err:
                    st.error(f"Save failed: {save_err}")
        with c_save3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Discard Changes", key="btn_reset_fe_dataset"):
                st.session_state["fe_working_df"] = df.copy()
                st.rerun()

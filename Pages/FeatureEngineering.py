import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, VarianceThreshold

from Utils.dataset_ui import render_sidebar, select_working_dataset, set_active_dataset, invalidate_dataset_cache
from Utils.paths import DATASETS_DIR, atomic_write, get_unique_filename, sanitize_for_csv_export
from Utils.theme import page_header, section_card_header, metric_card
from Utils.ML import detect_problem_type, is_likely_id_column

page_header(
    "Feature Engineering Studio",
    "Scale numerical features, encode categorical dimensions, apply non-linear transforms, and screen feature importance.",
    badge="DATA PREPARATION"
)

df, selected_file = select_working_dataset("Active dataset")
render_sidebar()

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

transformed_df = df.copy()

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
            is_binary_or_target = lambda c: (df[c].dropna().nunique() <= 2) or (c.lower() in ["churned", "churn", "target", "label", "is_churned"])
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
                    if "StandardScaler" in scale_method:
                        scaler = StandardScaler()
                    elif "MinMaxScaler" in scale_method:
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()

                    clean_vals = df[selected_scale_cols].fillna(df[selected_scale_cols].median())
                    scaled_matrix = scaler.fit_transform(clean_vals)
                    for idx, c in enumerate(selected_scale_cols):
                        transformed_df[f"{c}_scaled"] = scaled_matrix[:, idx]

                    st.session_state["fe_working_df"] = transformed_df
                    st.success(f"Generated {len(selected_scale_cols)} scaled feature(s)!")


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
                    if encode_method == "One-Hot Encoding (Dummies)":
                        dummies = pd.get_dummies(base_df[selected_cat_cols], prefix=selected_cat_cols, drop_first=True, dtype=int)
                        base_df = pd.concat([base_df, dummies], axis=1)
                        st.success(f"Generated {dummies.shape[1]} one-hot encoded columns!")
                    else:
                        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                        ord_matrix = ord_enc.fit_transform(base_df[selected_cat_cols].astype(str))
                        for idx, c in enumerate(selected_cat_cols):
                            base_df[f"{c}_encoded"] = ord_matrix[:, idx]
                        st.success(f"Generated {len(selected_cat_cols)} ordinal encoded features!")

                    st.session_state["fe_working_df"] = base_df


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
                ["Log1p (log(x + 1)) — Best for heavy right-skew", "Square Root (sqrt(x))", "Yeo-Johnson Power Transform — Normalizes distribution"]
            )

            if st.button("Apply Transform", type="primary", key="btn_apply_transform"):
                if not selected_tf_cols:
                    st.warning("Select columns to transform.")
                else:
                    base_df = st.session_state.get("fe_working_df", df).copy()
                    for c in selected_tf_cols:
                        series = base_df[c].fillna(base_df[c].median())
                        if "Log1p" in transform_kind:
                            # Shift minimum to >= 0 if negative
                            min_val = series.min()
                            shift = abs(min_val) if min_val < 0 else 0
                            base_df[f"{c}_log1p"] = np.log1p(series + shift)
                        elif "Square Root" in transform_kind:
                            min_val = series.min()
                            shift = abs(min_val) if min_val < 0 else 0
                            base_df[f"{c}_sqrt"] = np.sqrt(series + shift)
                        else:
                            pt = PowerTransformer(method="yeo-johnson")
                            base_df[f"{c}_yeojohnson"] = pt.fit_transform(series.values.reshape(-1, 1)).ravel()

                    st.session_state["fe_working_df"] = base_df
                    st.success(f"Applied {transform_kind} to {len(selected_tf_cols)} column(s)!")


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
                with st.spinner("Computing information gain scores..."):
                    clean_predictors = df[predictor_candidates].fillna(df[predictor_candidates].median())
                    clean_target = df[target_col].dropna()
                    valid_idx = clean_target.index.intersection(clean_predictors.index)

                    X = clean_predictors.loc[valid_idx]
                    y = clean_target.loc[valid_idx]

                    if prob_type == "Classification":
                        if not pd.api.types.is_numeric_dtype(y):
                            y = pd.factorize(y)[0]
                        mi_scores = mutual_info_classif(X, y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X, y, random_state=42)

                    mi_df = pd.DataFrame({
                        "Feature": predictor_candidates,
                        "Mutual Information": mi_scores
                    }).sort_values(by="Mutual Information", ascending=False)

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
with tab_interact:
    with st.container(border=True):
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
                if "Multiplication" in interact_op:
                    new_col_name = f"{feat_a}_x_{feat_b}"
                    base_df[new_col_name] = base_df[feat_a] * base_df[feat_b]
                elif "Ratio" in interact_op:
                    new_col_name = f"{feat_a}_div_{feat_b}"
                    denom = base_df[feat_b].replace(0, np.nan)
                    base_df[new_col_name] = (base_df[feat_a] / denom).fillna(0)
                else:
                    new_col_name = f"{feat_a}_minus_{feat_b}"
                    base_df[new_col_name] = base_df[feat_a] - base_df[feat_b]

                st.session_state["fe_working_df"] = base_df
                st.success(f"Created interaction feature: `{new_col_name}`!")


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

        c_save1, c_save2 = st.columns([2, 1])
        with c_save1:
            save_name_input = st.text_input(
                "Save engineered dataset as",
                value=f"engineered_{selected_file}"
            )
        with c_save2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Save & Activate Dataset", type="primary", key="btn_save_fe_dataset"):
                clean_name = get_unique_filename(save_name_input.strip() or f"engineered_{selected_file}", directory=DATASETS_DIR)
                dest = DATASETS_DIR / clean_name
                atomic_write(dest, sanitize_for_csv_export(active_fe_df).to_csv(index=False), mode="w", encoding="utf-8")
                invalidate_dataset_cache(clean_name)
                set_active_dataset(active_fe_df, clean_name)
                st.session_state.pop("fe_working_df", None)
                st.success(f"Saved `{clean_name}` and set as active workspace dataset!")
                st.rerun()

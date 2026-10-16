from pathlib import Path
import streamlit as st
import pandas as pd

from Utils.paths import (
    MODELS_DIR,
    read_tabular,
    sanitize_for_csv_export,
    SUPPORTED_DATASET_EXTENSIONS,
)
from Utils.ML import (
    list_saved_models,
    load_trained_model,
    predict_with_model,
)
from Utils.dataset_ui import render_sidebar, select_working_dataset
from Utils.theme import page_header, section_card_header
from Utils.logsys import get_logger

logger = get_logger("Predict")

page_header(
    "Predict & Score",
    "Load HMAC-signed machine learning models, inspect architecture and metrics, and generate batch predictions."
)

render_sidebar()

with st.container(border=True):
    section_card_header(
        "Select Trained Model",
        "Cryptographically verified model artifacts signed with HMAC-SHA256"
    )

    models_with_status = list_saved_models(include_trust_status=True)
    if not models_with_status:
        st.info("No saved models found in `Models/`. Train and save a model on the **Train Model** page first.")
        st.stop()

    display_map = {f"{name} [{status}]": (name, status) for name, status in models_with_status}
    chosen_display = st.selectbox("Select Model Artifact:", list(display_map.keys()), key="predict_model_selector")
    chosen_model_file, chosen_status = display_map[chosen_display]

    if chosen_status != "Trusted":
        st.warning(f"Artifact trust status: **{chosen_status}**. Untrusted or unsigned models will be blocked from loading to prevent code execution.")

    bundle = None
    try:
        bundle = load_trained_model(MODELS_DIR / chosen_model_file)
        if bundle.get("sklearn_version_mismatch"):
            st.warning("This model was saved with a different scikit-learn version; execution may behave unexpectedly.")
    except Exception as e:
        logger.warning("failed to load model %s: %s", chosen_model_file, e)
        st.error(f"Could not load model artifact: {e}")
        st.stop()

    if bundle:
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        with meta_c1:
            st.metric("Algorithm", bundle.get("model_name") or bundle.get("algorithm", "Model"))
        with meta_c2:
            st.metric("Problem Type", bundle.get("problem_type", "Unknown"))
        with meta_c3:
            st.metric("Target Variable", bundle.get("target_column", "Target"))
        with meta_c4:
            primary_metric = "accuracy" if bundle.get("problem_type") == "Classification" else "r2_score"
            metric_val = bundle.get("metrics", {}).get(primary_metric)
            if metric_val is not None:
                lbl = "Test Accuracy" if primary_metric == "accuracy" else "Test R²"
                st.metric(lbl, f"{metric_val * 100:.1f}%" if primary_metric == "accuracy" else f"{metric_val:.3f}")
            else:
                st.metric("Trained Date", bundle.get("created_at", "Saved")[:10])

        req_features = bundle.get("feature_cols", [])
        if req_features:
            st.caption(f"Required predictor features ({len(req_features)}): {', '.join(f'`{f}`' for f in req_features)}")

with st.container(border=True):
    section_card_header(
        "Select Scoring Dataset",
        "Choose an existing dataset or upload a new file to score"
    )

    data_source_mode = st.radio(
        "Scoring Data Source:",
        ["Active Workspace Dataset", "Upload New Scoring File"],
        horizontal=True,
        key="predict_data_source_mode"
    )

    score_df = None
    score_source_name = "active_dataset"

    if data_source_mode == "Active Workspace Dataset":
        active_df, active_name = select_working_dataset("Select active dataset to score:")
        if active_df is not None and not active_df.empty:
            score_df = active_df
            score_source_name = active_name
    else:
        accepted_exts = sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)
        scoring_file = st.file_uploader(
            "Upload file for inference:",
            type=accepted_exts,
            key="predict_file_uploader"
        )
        if scoring_file:
            score_source_name = scoring_file.name
            try:
                score_df = read_tabular(scoring_file, filename=scoring_file.name)
            except Exception as e:
                st.error(f"Failed to read scoring file: {e}")

if score_df is not None and not score_df.empty and bundle is not None:
    req_features = bundle.get("feature_cols", [])
    missing_feats = [f for f in req_features if f not in score_df.columns]

    if missing_feats:
        st.error(
            f"The selected dataset is missing {len(missing_feats)} required feature(s) "
            f"needed by this model: {', '.join(f'`{f}`' for f in missing_feats)}. "
            "Please select or upload a dataset containing all required predictor features."
        )
    else:
        with st.container(border=True):
            section_card_header(
                "Run Prediction",
                f"Score {len(score_df):,} records using `{bundle.get('model_name', 'model')}`"
            )

            if st.button("Generate Predictions", type="primary", key="btn_run_predictions"):
                try:
                    with st.spinner("Generating model predictions..."):
                        preds_df = predict_with_model(bundle, score_df)

                        if "category_drift" in preds_df.attrs:
                            drift_info = preds_df.attrs["category_drift"]
                            drift_desc = ", ".join(f"`{col}` ({len(vals)} unseen)" for col, vals in drift_info.items())
                            st.warning(f"Category drift detected during scoring: {drift_desc}. Unseen categories were mapped to unknown per pipeline policy.")

                        st.session_state["predict_last_preds"] = preds_df
                        st.session_state["predict_last_source"] = score_source_name
                        st.success(f"Successfully generated predictions for {len(preds_df):,} records!")
                except ValueError as ve:
                    st.error(f"Prediction validation error: {ve}")
                except Exception as e:
                    logger.error("prediction error: %s", e, exc_info=True)
                    st.error(f"Prediction failed: {e}")

scored_results = st.session_state.get("predict_last_preds")
if scored_results is not None and not scored_results.empty:
    with st.container(border=True):
        section_card_header("Predictions Preview & Export", "Inspect scored dataset and export results")
        st.dataframe(scored_results.head(25), width="stretch", hide_index=True)

        pred_cols = [c for c in scored_results.columns if c.startswith("prediction")]
        csv_bytes = sanitize_for_csv_export(scored_results).to_csv(index=False).encode("utf-8")
        src_stem = Path(st.session_state.get("predict_last_source", "dataset")).stem
        st.download_button(
            label="Download Scored Predictions (CSV)",
            data=csv_bytes,
            file_name=f"scored_predictions_{src_stem}.csv",
            mime="text/csv",
            key="download_dedicated_predictions"
        )
        st.caption(f"Appended prediction column(s): {', '.join(f'`{c}`' for c in pred_cols)}")

import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from Utils.dataset_ui import (
    dataframe_fingerprint,
    render_sidebar,
    select_working_dataset,
)
from Utils.logsys import get_logger
from Utils.ML import (
    list_saved_models,
    load_trained_model,
    predict_with_model,
)
from Utils.paths import (
    MODELS_DIR,
    SUPPORTED_DATASET_EXTENSIONS,
    read_tabular,
    sanitize_for_csv_export,
)
from Utils.theme import page_header, plot_template, section_card_header, style_plotly_fig

logger = get_logger("Predict")


def _model_fingerprint(model_name: str | None) -> str:
    """Derive a lightweight cache fingerprint for a model file."""
    if not model_name:
        return "none"
    model_path = MODELS_DIR / model_name
    sig_path = model_path.with_suffix(".joblib.sig")
    if sig_path.is_file():
        try:
            sig_data = json.loads(sig_path.read_text(encoding="utf-8"))
            return f"hmac:{sig_data.get('signature', '')}"
        except Exception:
            pass
    if model_path.is_file():
        stat = model_path.stat()
        return f"stat:{stat.st_mtime_ns}:{stat.st_size}"
    return "missing"

page_header(
    "Batch Scoring & Simulation",
    "Load HMAC-signed model artifacts, inspect architecture and validation metrics, and execute batch predictions.",
    badge="INFERENCE"
)

render_sidebar()

with st.container(border=True):
    section_card_header(
        "Select Trained Model",
        "Cryptographically verified model artifacts signed with HMAC-SHA256"
    )

    models_with_status = list_saved_models(include_trust_status=True)
    if not models_with_status:
        st.info("No saved models found in `Models/`. Train and save a model on the **Model Studio & Training** page first.")
        st.stop()

    display_map = {f"{name} [{status}]": (name, status) for name, status in models_with_status}
    display_keys = list(display_map.keys())
    chosen_display = st.selectbox("Model artifact", display_keys, key="predict_model_selector")
    if not chosen_display and display_keys:
        chosen_display = display_keys[0]

    chosen_model_file, chosen_status = display_map.get(chosen_display, (None, None))
    if not chosen_model_file:
        st.stop()

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
            target_name = bundle.get("target_column") or bundle.get("target_col") or "Target"
            st.metric("Target Variable", target_name)
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
        "Scoring data source",
        ["Active workspace dataset", "Upload new scoring file"],
        horizontal=True,
        key="predict_data_source_mode"
    )

    score_df = None
    score_source_name = "active_dataset"

    if data_source_mode == "Active workspace dataset":
        active_df, active_name = select_working_dataset("Dataset to score")
        if active_df is not None and not active_df.empty:
            score_df = active_df
            score_source_name = active_name
    else:
        accepted_exts = sorted(ext.lstrip(".") for ext in SUPPORTED_DATASET_EXTENSIONS)
        scoring_file = st.file_uploader(
            "Upload file for inference",
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
        model_disp_name = bundle.get("model_name", chosen_model_file)
        train_ds_name = bundle.get("dataset_name")
        train_ds_fp = bundle.get("dataset_fingerprint")
        score_df_fp = dataframe_fingerprint(score_df)

        is_exact_overlap = bool(train_ds_fp and score_df_fp and train_ds_fp == score_df_fp)
        is_name_overlap = bool(train_ds_name and score_source_name and train_ds_name == score_source_name)

        if is_exact_overlap:
            st.warning(
                "**Training-data overlap detected.** The selected scoring dataset matches the data "
                "associated with this model's training run. These batch predictions are not an independent "
                "evaluation of performance on unseen data. Consult **Model Studio & Training** for the model's holdout evaluation results."
            )
        elif is_name_overlap and not train_ds_fp:
            st.warning(
                f"**Possible training-data overlap.** The scoring dataset name (`{score_source_name}`) matches "
                "the model's recorded training source. Because cryptographic fingerprint evidence is unavailable "
                "for this model, content equality cannot be verified. Batch scoring may not reflect independent "
                "performance on unseen data."
            )
        elif is_name_overlap and train_ds_fp and train_ds_fp != score_df_fp:
            st.info(
                f"**Dataset version note:** The scoring file name (`{score_source_name}`) matches the model's training "
                "source name, but content fingerprint analysis confirms this dataset has different records or values. "
                "Batch predictions will score this modified dataset version."
            )
        elif not train_ds_fp and not train_ds_name:
            st.caption(
                "Model provenance metadata (training source and fingerprint) is unavailable for this artifact. "
                "Overlap verification cannot be performed."
            )

        with st.container(border=True):
            section_card_header(
                "Run Prediction",
                f"Generate batch predictions for {len(score_df):,} records using `{model_disp_name}`"
            )
            st.caption(
                "Batch prediction applies the selected model to the chosen records. "
                "Model evaluation metrics are calculated separately in Model Studio & Training."
            )

            if st.button("Generate predictions", type="primary", key="btn_run_predictions"):
                st.session_state.pop("predict_last_preds", None)
                st.session_state.pop("predict_last_scoring_id", None)
                st.session_state.pop("predict_last_source", None)
                st.session_state.pop("predict_last_model", None)
                try:
                    with st.spinner("Executing batch inference pipeline across records..."):
                        preds_df = predict_with_model(bundle, score_df)

                        if "category_drift" in preds_df.attrs:
                            drift_info = preds_df.attrs["category_drift"]
                            drift_desc = ", ".join(f"`{col}` ({len(vals)} unseen)" for col, vals in drift_info.items())
                            st.warning(f"Category drift detected during scoring: {drift_desc}. Unseen categories were mapped to unknown per pipeline policy.")

                        model_fp = _model_fingerprint(chosen_model_file)
                        data_fp = dataframe_fingerprint(score_df)
                        current_scoring_id = f"{chosen_model_file}:{model_fp}::{score_source_name}:{data_fp}"
                        st.session_state["predict_last_preds"] = preds_df
                        st.session_state["predict_last_source"] = score_source_name
                        st.session_state["predict_last_model"] = chosen_model_file
                        st.session_state["predict_last_scoring_id"] = current_scoring_id
                        st.success(f"Generated predictions for {len(preds_df):,} records.")
                except ValueError as ve:
                    st.session_state.pop("predict_last_preds", None)
                    st.session_state.pop("predict_last_scoring_id", None)
                    st.error(f"Prediction validation error: {ve}")
                except Exception as e:
                    st.session_state.pop("predict_last_preds", None)
                    st.session_state.pop("predict_last_scoring_id", None)
                    logger.error("prediction error: %s", e, exc_info=True)
                    st.error(f"Prediction failed: {e}")

model_fp = _model_fingerprint(chosen_model_file) if chosen_model_file else "none"
data_fp = dataframe_fingerprint(score_df) if score_df is not None and not score_df.empty else "none"
current_scoring_id = f"{chosen_model_file}:{model_fp}::{score_source_name}:{data_fp}"
scored_results = st.session_state.get("predict_last_preds")
last_scoring_id = st.session_state.get("predict_last_scoring_id")

if chosen_model_file and scored_results is not None and not scored_results.empty:
    if last_scoring_id == current_scoring_id:
        with st.container(border=True):
            section_card_header("Predictions Preview & Export", "Inspect scored dataset and export results")
            st.caption(
                f"**Artifact:** `{st.session_state.get('predict_last_model', chosen_model_file)}` | "
                f"**Source:** `{st.session_state.get('predict_last_source', score_source_name)}` "
                f"({len(scored_results):,} records)"
            )

            pred_cols = [c for c in scored_results.columns if c.startswith("prediction")]
            target_name = bundle.get("target_column") or bundle.get("target_col") or "Target"
            if target_name in scored_results.columns:
                st.info(
                    f"**Observed vs. Predicted:** `{target_name}` contains observed target values from the source dataset. "
                    f"{', '.join(f'`{c}`' for c in pred_cols)} contains values generated by `{model_disp_name}`. "
                    "Batch predictions should not be confused with ground truth or independent evaluation metrics."
                )

            st.dataframe(scored_results.head(15), width="stretch", hide_index=True)

            pipe = bundle.get("pipeline")
            req_features = bundle.get("feature_cols", [])
            if pipe is not None and hasattr(pipe, "predict_proba") and req_features:
                try:
                    probs = pipe.predict_proba(score_df[req_features])
                    if probs.ndim == 2 and probs.shape[1] > 1:
                        pos_probs = probs[:, 1]
                        fig_prob = px.histogram(
                            x=pos_probs,
                            nbins=25,
                            labels={"x": f"Predicted Probability of Positive Class ({target_name})"},
                            title=f"Inference Probability Distribution ({target_name})",
                            color_discrete_sequence=["#38bdf8"],
                        )
                        fig_prob.update_layout(
                            height=250,
                            margin={"l": 20, "r": 20, "t": 35, "b": 20},
                            xaxis_title="Confidence / Probability Score",
                            yaxis_title="Record Count",
                            template=plot_template(),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        style_plotly_fig(fig_prob)
                        st.plotly_chart(fig_prob, width="stretch", theme="streamlit")
                except Exception as pe:
                    logger.debug("predict_proba chart notice: %s", pe)

            csv_bytes = sanitize_for_csv_export(scored_results).to_csv(index=False).encode("utf-8")
            src_stem = Path(st.session_state.get("predict_last_source", "dataset")).stem
            st.download_button(
                label="Download scored predictions (CSV)",
                data=csv_bytes,
                file_name=f"scored_predictions_{src_stem}.csv",
                mime="text/csv",
                key="download_dedicated_predictions"
            )
            st.caption(f"Appended prediction column(s): {', '.join(f'`{c}`' for c in pred_cols)}")
    elif score_df is not None and not score_df.empty:
        st.info("The selected model or scoring dataset has changed. Click **Generate predictions** above to score the updated dataset.")

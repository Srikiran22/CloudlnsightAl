from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from Utils.theme import plot_template
from Utils.paths import MODELS_DIR, resolve_dataset_path, sanitize_for_csv_export
from Utils.ML import (
    detect_problem_type, train_and_evaluate_model,
    save_trained_model, load_trained_model, list_saved_models, predict_with_model
)
from Utils.dataset_ui import (
    dataframe_fingerprint, dataset_fingerprint, render_sidebar, results_match_active,
    select_working_dataset,
)
from Utils.logsys import get_logger

logger = get_logger("ML_Page")

st.title("Machine learning")
st.markdown("Train, evaluate, and persist classification or regression models with automatic preprocessing.")

df, selected_file = select_working_dataset("Select Dataset for Modeling:")
render_sidebar()
st.caption(f"Dataset: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} columns)")

if df.empty or len(df.columns) < 2 or len(df) < 2:
    st.warning("The selected dataset must contain at least 2 columns and at least 2 rows to train machine learning models.")
    st.stop()

st.subheader("Problem setup")
col1, col2 = st.columns(2)

with col1:
    target_column = st.selectbox(
        "Select Target Variable (What do you want to predict?):",
        df.columns.tolist(),
        index=len(df.columns) - 1
    )

if not target_column:
    st.warning("Please select a target variable to continue.")
    st.stop()

auto_type = detect_problem_type(df, target_column)

with col2:
    problem_type = st.radio(
        "Problem Type:",
        ["Classification", "Regression"],
        index=0 if auto_type == "Classification" else 1,
        horizontal=True,
        help=f"Auto-detected as: {auto_type}"
    )

st.subheader("Features & algorithm")
available_features = [col for col in df.columns if col != target_column]
if not available_features:
    st.warning("No input features available to predict the selected target. At least one predictor feature is required.")
    st.stop()

col_feat1, col_feat2 = st.columns([3, 2])
with col_feat1:
    selected_features = st.multiselect(
        "Select Input Features (Predictors):",
        available_features,
        default=available_features
    )

with col_feat2:
    if problem_type == "Classification":
        algorithms = ["Random Forest", "Gradient Boosting", "Decision Tree", "Logistic Regression"]
    else:
        algorithms = ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"]

    chosen_algo = st.selectbox("Choose ML Algorithm:", algorithms)

col_split1, col_split2 = st.columns(2)
with col_split1:
    test_pct = st.slider("Test Set Split Size (%):", min_value=10, max_value=40, value=20, step=5)
with col_split2:
    random_seed = st.number_input("Random State (Seed):", min_value=0, max_value=999, value=42)

has_temporal_col = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns) or isinstance(df.index, pd.DatetimeIndex)
use_time_series_cv = False
if has_temporal_col:
    use_time_series_cv = st.checkbox(
        "Use TimeSeriesSplit for cross-validation (preserves sequential order)",
        value=False,
        help="Evaluates folds forward in time without shuffling to avoid lookahead leakage."
    )

if not selected_features:
    st.warning("Select at least one feature to train the model.")
    st.stop()

if st.button("Train & evaluate", type="primary"):
    with st.spinner(f"Training {chosen_algo} ({problem_type})..."):
        try:
            results = train_and_evaluate_model(
                df=df,
                target_col=target_column,
                feature_cols=selected_features,
                model_name=chosen_algo,
                problem_type=problem_type,
                test_size=test_pct / 100.0,
                random_state=int(random_seed),
                time_series_cv=use_time_series_cv,
            )
            results["dataset_name"] = selected_file
            is_file_backed = False
            try:
                is_file_backed = resolve_dataset_path(selected_file).is_file()
            except Exception:
                is_file_backed = False
            results["dataset_fingerprint"] = dataset_fingerprint(selected_file) if is_file_backed else dataframe_fingerprint(df)
            results["target_col"] = target_column
            results["feature_cols"] = list(selected_features)
            st.session_state["ml_results"] = results
            st.success(
                f"Trained **{chosen_algo}** on {results['train_size']:,} samples "
                f"and evaluated on {results['test_size']:,} test samples."
            )
        except ValueError as e:
            logger.warning("ML training validation error: %s", e)
            st.error(f"Training could not proceed: {e}")
        except Exception as e:
            logger.error("ML training unexpected error: %s", e, exc_info=True)
            st.error(f"Training failed: {type(e).__name__}. Check application logs for technical details.")

results = st.session_state.get("ml_results")
if results_match_active(results, selected_file, df=df):
    st.markdown("---")
    st.subheader("Evaluation metrics")

    if results.get("temporal_warning"):
        st.warning(results["temporal_warning"])

    if results["problem_type"] == "Classification":
        if results.get("stratified_warning"):
            st.info(results["stratified_warning"])
        if results.get("f1_macro") is not None:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
            with m2:
                st.metric("Precision (Weighted)", f"{results['precision'] * 100:.2f}%")
            with m3:
                st.metric("Recall (Weighted)", f"{results['recall'] * 100:.2f}%")
            with m4:
                st.metric("F1-Score (Weighted)", f"{results['f1_score'] * 100:.2f}%")
            with m5:
                st.metric("F1-Score (Macro)", f"{results['f1_macro'] * 100:.2f}%")
        else:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test Accuracy", f"{results['accuracy'] * 100:.2f}%")
            with m2:
                st.metric("Precision (Weighted)", f"{results['precision'] * 100:.2f}%")
            with m3:
                st.metric("Recall (Weighted)", f"{results['recall'] * 100:.2f}%")
            with m4:
                st.metric("F1-Score (Weighted)", f"{results['f1_score'] * 100:.2f}%")

        if results.get("cv_folds"):
            st.markdown("##### Cross-Validation Performance & Baseline Benchmark")
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                st.metric(
                    f"{results['cv_folds']}-Fold CV Accuracy",
                    f"{results['cv_accuracy_mean'] * 100:.2f}%",
                    help=f"Mean across {results['cv_folds']} folds with ±{results['cv_accuracy_std'] * 100:.2f}% standard deviation."
                )
                st.caption(f"Range: {(results['cv_accuracy_mean'] - results['cv_accuracy_std']) * 100:.1f}% – {(results['cv_accuracy_mean'] + results['cv_accuracy_std']) * 100:.1f}%")
            with cv2:
                st.metric(
                    f"{results['cv_folds']}-Fold CV Macro F1",
                    f"{results['cv_f1_macro_mean']:.4f}",
                    help=f"Mean Macro F1 across {results['cv_folds']} folds with ±{results['cv_f1_macro_std']:.4f} standard deviation."
                )
                st.caption(f"Range: {results['cv_f1_macro_mean'] - results['cv_f1_macro_std']:.4f} – {results['cv_f1_macro_mean'] + results['cv_f1_macro_std']:.4f}")
            with cv3:
                st.metric(
                    "Dummy Baseline Accuracy",
                    f"{results.get('baseline_accuracy', 0) * 100:.2f}%",
                    help="Majority-class baseline without feature learning."
                )
                st.caption(f"Baseline Macro F1: {results.get('baseline_f1_macro', 0):.4f}")

        st.subheader("Confusion Matrix")
        cm = results["confusion_matrix"]
        classes = results["classes"]
        fig_cm = px.imshow(
            cm,
            labels={"x": "Predicted Class", "y": "Actual Class", "color": "Count"},
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            text_auto=True,
            color_continuous_scale="Blues",
            template=plot_template()
        )
        fig_cm.update_layout(title="Confusion Matrix Heatmap")
        st.plotly_chart(fig_cm, width="stretch")

    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Test R² Score", f"{results['r2_score']:.4f}")
        with m2:
            st.metric("Test RMSE", f"{results['rmse']:.4f}")
        with m3:
            st.metric("Test MAE", f"{results['mae']:.4f}")
        with m4:
            st.metric("Test MSE", f"{results['mse']:.4f}")

        if results.get("cv_folds"):
            st.markdown("##### Cross-Validation Performance & Baseline Benchmark")
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                st.metric(
                    f"{results['cv_folds']}-Fold CV R² Score",
                    f"{results['cv_r2_mean']:.4f}",
                    help=f"Mean across {results['cv_folds']} folds with ±{results['cv_r2_std']:.4f} standard deviation."
                )
                st.caption(f"Range: {results['cv_r2_mean'] - results['cv_r2_std']:.4f} – {results['cv_r2_mean'] + results['cv_r2_std']:.4f}")
            with cv2:
                st.metric(
                    f"{results['cv_folds']}-Fold CV RMSE",
                    f"{results['cv_rmse_mean']:.4f}",
                    help=f"Mean across {results['cv_folds']} folds with ±{results['cv_rmse_std']:.4f} standard deviation."
                )
                st.caption(f"Range: {results['cv_rmse_mean'] - results['cv_rmse_std']:.4f} – {results['cv_rmse_mean'] + results['cv_rmse_std']:.4f}")
            with cv3:
                st.metric(
                    "Dummy Baseline R²",
                    f"{results.get('baseline_r2', 0):.4f}",
                    help="Mean-target baseline without feature learning."
                )
                st.caption(f"Baseline RMSE: {results.get('baseline_rmse', 0):.4f}")

        pred_df = pd.DataFrame({
            "Actual": results["y_test"],
            "Predicted": results["y_pred"],
            "Residual": results["residuals"]
        })

        plot_pred_df = pred_df.sample(25_000, random_state=42) if len(pred_df) > 25_000 else pred_df
        fig_pred = px.scatter(
            plot_pred_df,
            x="Actual",
            y="Predicted",
            template=plot_template(),
            opacity=0.75,
            title="Actual vs Predicted Values" + (" (Sampled to 25k)" if len(pred_df) > 25_000 else "")
        )
        finite_actual = pred_df["Actual"].dropna()
        finite_actual = finite_actual[np.isfinite(finite_actual)]
        finite_pred = pred_df["Predicted"].dropna()
        finite_pred = finite_pred[np.isfinite(finite_pred)]
        if not finite_actual.empty and not finite_pred.empty:
            min_val = min(float(finite_actual.min()), float(finite_pred.min()))
            max_val = max(float(finite_actual.max()), float(finite_pred.max()))
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction Line",
                line={"color": "red", "dash": "dash"}
            ))
        st.plotly_chart(fig_pred, width="stretch")

    raw_fi = results.get("feature_importances")
    grouped_fi = results.get("grouped_feature_importances")
    if raw_fi or grouped_fi:
        st.subheader("Feature importance")
        imp_metric = results.get("importance_type", "Feature Importance / Relative Weight")
        st.caption(
            f"**Metric:** {imp_metric}. "
            "⚠️ *Note: Feature importance reflects statistical associations within the trained model, "
            "NOT real-world causal drivers.*"
        )

        def _render_fi_chart(fi_dict, title):
            fi_df = pd.DataFrame(
                list(fi_dict.items()),
                columns=["Feature", imp_metric]
            ).sort_values(by=imp_metric, ascending=True)

            fig_fi = px.bar(
                fi_df,
                x=imp_metric,
                y="Feature",
                orientation="h",
                color=imp_metric,
                color_continuous_scale="Viridis",
                template=plot_template(),
                title=title
            )
            st.plotly_chart(fig_fi, width="stretch")

        if grouped_fi and raw_fi and grouped_fi != raw_fi:
            tab_orig, tab_encoded = st.tabs(["Original features (aggregated)", "Encoded levels"])
            with tab_orig:
                _render_fi_chart(grouped_fi, "Top Influential Features (Aggregated)")
            with tab_encoded:
                _render_fi_chart(raw_fi, "Top Influential Encoded Levels")
        else:
            _render_fi_chart(grouped_fi or raw_fi, "Top Influential Features in Model")

st.markdown("---")
st.subheader("Model persistence")

tab_save, tab_load = st.tabs(["Save trained model", "Load saved model & predict"])

with tab_save:
    current_results = st.session_state.get("ml_results")
    if not results_match_active(current_results, selected_file, df=df):
        st.info("Train a model above first, then save it here for reuse.")
    else:
        save_name = st.text_input(
            "Model Name:",
            value=f"{current_results.get('model_name', 'model')}_{current_results.get('problem_type', 'model')}".replace(" ", "_")
        )
        if st.button("Save model to Models/ folder"):
            try:
                path = save_trained_model(current_results, save_name)
                st.success(f"Model saved as `{path.name}` in the Models/ folder.")
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

with tab_load:
    st.caption(
        "Joblib model bundles execute arbitrary Python bytecode upon deserialization. "
        "To prevent arbitrary code execution, only models cryptographically signed with "
        "HMAC-SHA256 by this application instance are permitted to load."
    )
    models_with_status = list_saved_models(include_trust_status=True)
    if not models_with_status:
        st.info("No saved models yet. Train and save one first.")
    else:
        # Build human-readable selector displaying trust state
        display_map = {f"{name} [{status}]": (name, status) for name, status in models_with_status}
        chosen_display = st.selectbox("Select Saved Model:", list(display_map.keys()))
        chosen_model_file, chosen_status = display_map[chosen_display]

        if chosen_status != "Trusted":
            st.warning(f"Artifact trust status: **{chosen_status}**. Untrusted or unsigned models will be blocked.")

        if st.button("Load model & predict"):
            try:
                bundle = load_trained_model(MODELS_DIR / chosen_model_file)
                if bundle.get("sklearn_version_mismatch"):
                    st.warning(
                        "This model was saved with a different scikit-learn version; "
                        "loading may fail or behave unexpectedly."
                    )
                predictions_df = predict_with_model(bundle, df)

                if "category_drift" in predictions_df.attrs:
                    drift_info = predictions_df.attrs["category_drift"]
                    drift_desc = ", ".join(f"`{col}` ({len(vals)} unseen)" for col, vals in drift_info.items())
                    st.warning(f"Category drift detected during inference: {drift_desc}. Unseen categories were ignored per encoding policy.")

                st.success(
                    f"Loaded **{bundle.get('algorithm', 'model')}** "
                    f"({bundle.get('problem_type', '?')}) trained on `{bundle.get('dataset_name', '?')}`."
                )
                if bundle.get("metrics"):
                    metric_items = list(bundle["metrics"].items())
                    metric_cols = st.columns(min(len(metric_items), 4))
                    for i, (metric_name, value) in enumerate(metric_items):
                        with metric_cols[i % 4]:
                            pretty = metric_name.replace("_", " ").title()
                            st.metric(pretty, f"{value:.4f}" if isinstance(value, float) else value)

                st.subheader("Predictions Preview")
                st.dataframe(predictions_df.head(20), width="stretch")

                pred_cols = [c for c in predictions_df.columns if c.startswith("prediction")]
                csv_bytes = sanitize_for_csv_export(predictions_df).to_csv(index=False).encode("utf-8")
                pred_stem = Path(selected_file).stem if selected_file else "dataset"
                st.download_button(
                    label="Download full predictions (CSV)",
                    data=csv_bytes,
                    file_name=f"predictions_{pred_stem}.csv",
                    mime="text/csv",
                    key="download_predictions"
                )
                st.caption(f"Prediction column(s): {', '.join(pred_cols)}")
            except ValueError as e:
                logger.warning("Prediction validation error: %s", e)
                st.error(f"Prediction could not proceed: {e}")
            except Exception as e:
                logger.error("Prediction unexpected failure: %s", e, exc_info=True)
                st.error("Prediction failed. Check application logs for technical details.")

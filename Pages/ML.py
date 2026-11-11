from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Utils.dataset_ui import (
    dataframe_fingerprint,
    dataset_fingerprint,
    render_sidebar,
    results_match_active,
    select_working_dataset,
)
from Utils.logsys import get_logger
from Utils.ML import (
    detect_problem_type,
    is_likely_id_column,
    save_trained_model,
    train_and_evaluate_model,
)
from Utils.paths import resolve_dataset_path
from Utils.theme import page_header, plot_template, section_card_header, style_plotly_fig

logger = get_logger("ML_Page")

page_header(
    "Model Studio & Training",
    "Train, evaluate, and benchmark Scikit-Learn classification and regression models with cross-validation.",
    badge="MACHINE LEARNING"
)

df, selected_file = select_working_dataset("Active dataset")
render_sidebar()

if df.empty or len(df.columns) < 2 or len(df) < 2:
    st.warning("The selected dataset must contain at least 2 columns and at least 2 rows to train machine learning models.")
    st.stop()

with st.container(border=True):
    section_card_header(
        "Model Setup & Architecture",
        f"Configure prediction target, problem framing, and feature space for {selected_file}"
    )
    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox(
            "Target variable (what do you want to predict?)",
            df.columns.tolist(),
            index=len(df.columns) - 1
        )

    if not target_column:
        st.warning("Please select a target variable to continue.")
        st.stop()

    auto_type = detect_problem_type(df, target_column)

    with col2:
        problem_type = st.radio(
            "Problem type",
            ["Classification", "Regression"],
            index=0 if auto_type == "Classification" else 1,
            horizontal=True,
            help=f"Auto-detected as: {auto_type}"
        )

    available_features = [col for col in df.columns if col != target_column]
    if not available_features:
        st.warning("No input features available to predict the selected target. At least one predictor feature is required.")
        st.stop()

    # Auto-detect ID or high-cardinality identifier columns to prevent feature explosion & model collapse
    id_cols = [c for c in available_features if is_likely_id_column(df, c)]
    default_features = [c for c in available_features if c not in id_cols]
    if not default_features:
        default_features = available_features

    col_feat1, col_feat2 = st.columns([3, 2])
    with col_feat1:
        selected_features = st.multiselect(
            "Input features (predictors)",
            available_features,
            default=default_features
        )
        if id_cols:
            st.caption(f"Auto-excluded {len(id_cols)} identifier column(s) from defaults: {', '.join(f'`{c}`' for c in id_cols)}")

    with col_feat2:
        if problem_type == "Classification":
            algorithms = ["Random Forest", "Gradient Boosting", "Decision Tree", "Logistic Regression"]
        else:
            algorithms = ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"]

        chosen_algo = st.selectbox("Algorithm", algorithms)

    col_split1, col_split2 = st.columns(2)
    with col_split1:
        test_pct = st.slider("Test-set size (%)", min_value=10, max_value=40, value=20, step=5)
    with col_split2:
        random_seed = st.number_input("Random seed", min_value=0, max_value=999, value=42)

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

    train_clicked = st.button("Train & evaluate", type="primary")

if train_clicked:
    with st.spinner(f"Fitting {chosen_algo} pipeline ({problem_type}) with cross-validation..."):
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
            try:
                auto_name = f"{chosen_algo}_{target_column}_model".replace(" ", "_").lower()
                auto_path = save_trained_model(results, auto_name)
                st.session_state["last_saved_model_path"] = str(auto_path)
            except Exception as auto_save_err:
                logger.warning("Auto-save model notice: %s", auto_save_err)
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

def _render_fi_chart(fi_dict, title, imp_metric):
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
    style_plotly_fig(fig_fi)
    st.plotly_chart(fig_fi, width="stretch", theme="streamlit")

results = st.session_state.get("ml_results")
if results_match_active(results, selected_file, df=df):
    st.markdown("---")
    with st.container(border=True):
        section_card_header(
            "Evaluation Metrics & Diagnostics",
            f"Performance evaluation for {results.get('model_name', 'model')} ({results.get('problem_type', '')}) on {results.get('test_size', 0):,} holdout samples"
        )

        if results.get("temporal_warning"):
            st.warning(results["temporal_warning"])

        if results["problem_type"] == "Classification":
            if results.get("stratified_warning"):
                st.info(results["stratified_warning"])
            acc_pct = results["accuracy"] * 100
            prec_pct = results.get("precision", 0) * 100
            base_pct = results.get("baseline_accuracy", 0) * 100
            target_name = results.get("target_column") or target_column
            model_disp_name = results.get('model_name', 'model')

            if acc_pct < base_pct:
                diff = base_pct - acc_pct
                st.warning(
                    f"**Model Verdict: Below baseline performance.** This {model_disp_name} achieved "
                    f"**{acc_pct:.1f}% accuracy** on holdout test data, underperforming the naive "
                    f"majority-class baseline ({base_pct:.1f}%) by **{diff:.1f} percentage points**. "
                    "Predictions should not be relied upon for production decisions until feature representation or class imbalance is addressed."
                )
            elif abs(acc_pct - base_pct) <= 3.0 and base_pct > 70.0:
                st.info(
                    f"**Model Verdict: Baseline parity.** Achieved **{acc_pct:.1f}% overall accuracy** and "
                    f"**{prec_pct:.1f}% weighted precision** on unseen test data, comparable to the "
                    f"naive majority baseline of {base_pct:.1f}%."
                )
            else:
                gain = acc_pct - base_pct
                st.success(
                    f"**Model Verdict: Above baseline.** This {model_disp_name} achieved **{acc_pct:.1f}% accuracy** "
                    f"(weighted precision: **{prec_pct:.1f}%**) on holdout test data, outperforming the "
                    f"naive majority baseline ({base_pct:.1f}%) by **+{gain:.1f} percentage points**."
                )
            if results.get("f1_macro") is not None:
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
                with m2:
                    st.metric("Precision (Weighted)", f"{results['precision'] * 100:.2f}%")
                with m3:
                    st.metric("Recall (Weighted)", f"{results['recall'] * 100:.2f}%")
                with m4:
                    st.metric("F1 (Weighted)", f"{results['f1_score'] * 100:.2f}%")
                with m5:
                    st.metric("F1 (Macro)", f"{results['f1_macro'] * 100:.2f}%")
            else:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
                with m2:
                    st.metric("Precision (Weighted)", f"{results['precision'] * 100:.2f}%")
                with m3:
                    st.metric("Recall (Weighted)", f"{results['recall'] * 100:.2f}%")
                with m4:
                    st.metric("F1 (Weighted)", f"{results['f1_score'] * 100:.2f}%")

            if results.get("positive_class") is not None:
                st.markdown(
                    f'<div style="font-size: 0.84rem; color: var(--ci-text-2); margin: 0.6rem 0 0.85rem 0; padding: 8px 14px; background: var(--ci-tint); border-radius: var(--ci-radius-sm); border: 1px solid var(--ci-border);">'
                    f'<strong>Positive class {results["positive_class"]}</strong> &nbsp;|&nbsp; '
                    f'Precision <strong>{results["positive_precision"] * 100:.2f}%</strong> &nbsp;|&nbsp; '
                    f'Recall <strong>{results["positive_recall"] * 100:.2f}%</strong> &nbsp;|&nbsp; '
                    f'F1 <strong>{results["positive_f1"] * 100:.2f}%</strong> &nbsp;|&nbsp; '
                    f'Support <strong>{results["positive_support"]}</strong> samples'
                    f'</div>',
                    unsafe_allow_html=True
                )

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
                        f"{results['cv_f1_macro_mean'] * 100:.2f}%",
                        help=f"Mean Macro F1 across {results['cv_folds']} folds with ±{results['cv_f1_macro_std'] * 100:.2f}% standard deviation."
                    )
                    st.caption(f"Range: {(results['cv_f1_macro_mean'] - results['cv_f1_macro_std']) * 100:.1f}% – {(results['cv_f1_macro_mean'] + results['cv_f1_macro_std']) * 100:.1f}%")
                with cv3:
                    st.metric(
                        "Dummy Baseline Accuracy",
                        f"{results.get('baseline_accuracy', 0) * 100:.2f}%",
                        help="Majority-class baseline without feature learning."
                    )
                    st.caption(f"Baseline Macro F1: {results.get('baseline_f1_macro', 0) * 100:.1f}%")

            tab_cm, tab_fi = st.tabs(["Confusion Matrix Heatmap", "Feature Importance"])
            with tab_cm:
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
                style_plotly_fig(fig_cm)
                st.plotly_chart(fig_cm, width="stretch", theme="streamlit")

            with tab_fi:
                raw_fi = results.get("feature_importances")
                grouped_fi = results.get("grouped_feature_importances")
                if raw_fi or grouped_fi:
                    imp_metric = results.get("importance_type", "Feature Importance / Relative Weight")
                    st.caption(
                        f"**Metric:** {imp_metric}. "
                        "*Note: Feature importance reflects statistical associations within the trained model, "
                        "not real-world causal drivers.*"
                    )
                    if grouped_fi and raw_fi and grouped_fi != raw_fi:
                        tab_orig, tab_encoded = st.tabs(["Original features (aggregated)", "Encoded levels"])
                        with tab_orig:
                            _render_fi_chart(grouped_fi, "Top Influential Features (Aggregated)", imp_metric)
                        with tab_encoded:
                            _render_fi_chart(raw_fi, "Top Influential Encoded Levels", imp_metric)
                    else:
                        _render_fi_chart(grouped_fi or raw_fi, "Top Influential Features in Model", imp_metric)
                else:
                    st.info("Feature importance is not available for this model type.")

        else:
            r2_pct = max(0.0, results["r2_score"] * 100)
            mae = results["mae"]
            target_name = results.get("target_column") or target_column
            st.info(
                f"**Model Verdict:** This {results.get('model_name', 'model')} explains **{r2_pct:.1f}% of the variance** "
                f"in **{target_name}**, with an average prediction error of ±{mae:.2f}."
            )
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("R² Score", f"{results['r2_score']:.4f}")
            with m2:
                st.metric("RMSE", f"{results['rmse']:.4f}")
            with m3:
                st.metric("MAE", f"{results['mae']:.4f}")
            with m4:
                st.metric("MSE", f"{results['mse']:.4f}")

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

            tab_res, tab_fi_reg = st.tabs(["Actual vs Predicted Residuals", "Feature Importance"])
            with tab_res:
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
                style_plotly_fig(fig_pred)
                st.plotly_chart(fig_pred, width="stretch", theme="streamlit")

            with tab_fi_reg:
                raw_fi = results.get("feature_importances")
                grouped_fi = results.get("grouped_feature_importances")
                if raw_fi or grouped_fi:
                    imp_metric = results.get("importance_type", "Feature Importance / Relative Weight")
                    st.caption(
                        f"**Metric:** {imp_metric}. "
                        "*Note: Feature importance reflects statistical associations within the trained model, "
                        "not real-world causal drivers.*"
                    )
                    if grouped_fi and raw_fi and grouped_fi != raw_fi:
                        tab_orig, tab_encoded = st.tabs(["Original features (aggregated)", "Encoded levels"])
                        with tab_orig:
                            _render_fi_chart(grouped_fi, "Top Influential Features (Aggregated)", imp_metric)
                        with tab_encoded:
                            _render_fi_chart(raw_fi, "Top Influential Encoded Levels", imp_metric)
                    else:
                        _render_fi_chart(grouped_fi or raw_fi, "Top Influential Features in Model", imp_metric)
                else:
                    st.info("Feature importance is not available for this model type.")

st.markdown("---")
with st.container(border=True):
    section_card_header(
        "Save Trained Model & Batch Prediction",
        "Export cryptographically signed HMAC-SHA256 model artifacts or generate batch predictions"
    )

    current_results = st.session_state.get("ml_results")
    if not results_match_active(current_results, selected_file, df=df):
        st.info("Train a model above first, then save it here for reuse.")
    else:
        col_save_left, col_save_right = st.columns([1.5, 1])
        with col_save_left:
            last_saved = st.session_state.get("last_saved_model_path")
            if last_saved and Path(last_saved).is_file():
                st.success(f"Model automatically saved and HMAC-signed as `{Path(last_saved).name}` in `Models/`.")
            default_save_name = f"{current_results.get('model_name', 'model')}_{current_results.get('target_col', current_results.get('target_column', 'model'))}".replace(" ", "_").lower()
            save_name = st.text_input(
                "Save copy with custom name",
                value=default_save_name,
                key="ml_save_model_name"
            )
            if st.button("Save additional named copy", key="btn_save_model_explicit"):
                try:
                    path = save_trained_model(current_results, save_name)
                    st.success(f"Model saved and HMAC-signed as `{path.name}` in `Models/`.")
                except Exception as e:
                    st.error(f"Save failed: {str(e)}")
        with col_save_right:
            st.markdown('<div class="ci-side-label" style="margin-top:0.6rem;">Batch Scoring</div>', unsafe_allow_html=True)
            try:
                st.page_link(
                    "Pages/Predict.py",
                    label="Open Batch Scoring & Simulation",
                    icon=":material/batch_prediction:",
                    help="Switch to dedicated batch inference and prediction scoring workspace"
                )
            except Exception as exc:
                logger.debug("suppressed error in %s: %s", "open_batch_scoring_link", exc)
            st.caption("Use saved models to score new datasets with drift checks and CSV export.")

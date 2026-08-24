import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from Utils.paths import MODELS_DIR
from Utils.ML import (
    detect_problem_type, train_and_evaluate_model,
    save_trained_model, load_trained_model, list_saved_models, predict_with_model
)
from Utils.dataset_ui import (
    dataset_fingerprint, render_sidebar, results_match_active,
    select_working_dataset,
)

st.title("Machine learning")
st.markdown("Train, evaluate, and persist classification or regression models with automatic preprocessing.")

df, selected_file = select_working_dataset("Select Dataset for Modeling:")
render_sidebar()
st.caption(f"Dataset: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} columns)")

st.subheader("Problem setup")
col1, col2 = st.columns(2)

with col1:
    target_column = st.selectbox(
        "Select Target Variable (What do you want to predict?):",
        df.columns.tolist(),
        index=len(df.columns) - 1
    )

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
                random_state=int(random_seed)
            )
            results["dataset_name"] = selected_file
            results["dataset_fingerprint"] = dataset_fingerprint(selected_file)
            results["target_col"] = target_column
            results["feature_cols"] = list(selected_features)
            st.session_state["ml_results"] = results
            st.success(
                f"Trained **{chosen_algo}** on {results['train_size']:,} samples "
                f"and evaluated on {results['test_size']:,} test samples."
            )
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

results = st.session_state.get("ml_results")
if results_match_active(results, selected_file):
    st.markdown("---")
    st.subheader("Evaluation metrics")

    if results["problem_type"] == "Classification":
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
        with m2:
            st.metric("Precision (Weighted)", f"{results['precision'] * 100:.2f}%")
        with m3:
            st.metric("Recall (Weighted)", f"{results['recall'] * 100:.2f}%")
        with m4:
            st.metric("F1-Score (Weighted)", f"{results['f1_score'] * 100:.2f}%")

        st.subheader("Confusion Matrix")
        cm = results["confusion_matrix"]
        classes = results["classes"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig_cm.update_layout(title="Confusion Matrix Heatmap")
        st.plotly_chart(fig_cm, width="stretch")

    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("R² Score (Variance Explained)", f"{results['r2_score']:.4f}")
        with m2:
            st.metric("RMSE (Root Mean Sq Error)", f"{results['rmse']:.4f}")
        with m3:
            st.metric("MAE (Mean Absolute Error)", f"{results['mae']:.4f}")
        with m4:
            st.metric("MSE", f"{results['mse']:.4f}")

        pred_df = pd.DataFrame({
            "Actual": results["y_test"],
            "Predicted": results["y_pred"],
            "Residual": results["residuals"]
        })

        fig_pred = px.scatter(
            pred_df,
            x="Actual",
            y="Predicted",
            template="plotly_white",
            opacity=0.75,
            title="Actual vs Predicted Values"
        )
        min_val = min(min(pred_df["Actual"]), min(pred_df["Predicted"]))
        max_val = max(max(pred_df["Actual"]), max(pred_df["Predicted"]))
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction Line",
            line=dict(color="red", dash="dash")
        ))
        st.plotly_chart(fig_pred, width="stretch")

    if results.get("feature_importances"):
        st.subheader("Feature importance")
        fi_df = pd.DataFrame(
            list(results["feature_importances"].items()),
            columns=["Feature", "Importance / Relative Weight"]
        ).sort_values(by="Importance / Relative Weight", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance / Relative Weight",
            y="Feature",
            orientation="h",
            color="Importance / Relative Weight",
            color_continuous_scale="Viridis",
            template="plotly_white",
            title="Top Influential Features in Model"
        )
        st.plotly_chart(fig_fi, width="stretch")

st.markdown("---")
st.subheader("Model persistence")

tab_save, tab_load = st.tabs(["Save trained model", "Load saved model & predict"])

with tab_save:
    current_results = st.session_state.get("ml_results")
    if not results_match_active(current_results, selected_file):
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
        "Only load `.joblib` bundles you trained yourself or received from a trusted source — "
        "model files contain executable code."
    )
    saved_models = list_saved_models()
    if not saved_models:
        st.info("No saved models yet. Train and save one first.")
    else:
        chosen_model_file = st.selectbox("Select Saved Model:", saved_models)

        if st.button("Load model & predict"):
            try:
                bundle = load_trained_model(MODELS_DIR / chosen_model_file)
                if bundle.get("sklearn_version_mismatch"):
                    st.warning(
                        "This model was saved with a different scikit-learn version; "
                        "loading may fail or behave unexpectedly."
                    )
                predictions_df = predict_with_model(bundle, df)

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
                csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download full predictions (CSV)",
                    data=csv_bytes,
                    file_name=f"predictions_{selected_file}",
                    mime="text/csv",
                    key="download_predictions"
                )
                st.caption(f"Prediction column(s): {', '.join(pred_cols)}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

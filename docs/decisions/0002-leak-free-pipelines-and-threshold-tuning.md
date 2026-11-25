# ADR-002: Pipeline-Encapsulated Preprocessing and Out-of-Fold Threshold Tuning

## Status
Accepted

## Date
2026-11-16

## Context
A common source of optimistic bias in tabular ML tools is fitting imputers, scalers, or encoders on the entire dataset prior to train/test splitting, or tuning classification decision thresholds directly against the holdout test set. At the same time, interactive data preparation still benefits from standalone exploration tools (scaling, encoding, interaction features), and users need to inspect precision/recall trade-offs without retraining models on every slider move.

## Decision
1. **Strict pipeline encapsulation during modeling (`utils/ml.py`):** Split the target-filtered dataset first (`train_test_split`, stratified for IID classification when feasible, or chronological `shuffle=False` when `time_series_cv=True`), then fit `ColumnTransformer` (`SimpleImputer` + `StandardScaler` for numeric features; `SimpleImputer` + `OneHotEncoder(handle_unknown="ignore")` for categorical features) strictly inside the scikit-learn `Pipeline` on training folds.
2. **Naive baseline benchmarking:** Fit a `DummyClassifier(strategy="most_frequent")` or `DummyRegressor(strategy="mean")` on the same split so holdout metrics are always contextualized against the trivial baseline.
3. **Out-of-fold threshold calibration:** For binary classification, compute out-of-fold probabilities on `(X_train, y_train)` via `cross_val_predict(..., method="predict_proba")` so slider exploration (`0.05 <= threshold <= 0.95` via `evaluate_threshold_metrics`) evaluates validation metrics without leaking holdout test targets. Persist the chosen `decision_threshold` and `positive_class` in the saved bundle (`save_trained_model`) and apply them in `predict_with_model` and `generate_standalone_inference_code`.
4. **Standalone script export:** Generate a self-contained CLI Python script (`predict_*.py`) that loads the signed `.joblib` bundle, enforces feature presence, and executes batch scoring outside Streamlit.

## Alternatives Considered

### Pre-Split Preprocessing Before `train_test_split`
- Pros: Simpler estimator wrappers.
- Cons: Leaks holdout mean, variance, median, and category frequencies into training.
- Rejected: Invalidates out-of-sample evaluation.

### Automated Hyperparameter & AutoML Search (Optuna / TPOT)
- Pros: Higher peak benchmark accuracy on some datasets.
- Cons: Unbounded CPU time in a synchronous Streamlit request handler; obscures model behavior and risks overfitting small tabular samples.
- Rejected: Deterministic, fast training of transparent linear and tree ensembles (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForest`, `GradientBoosting`, `LinearRegression`, `Ridge`) keeps execution predictable.

## Consequences
- Model evaluation metrics inside `Model Studio & Training` (`pages/ml.py`) avoid preprocessing leakage from the modeling pipeline itself.
- **Known boundary:** Interactive transformations on `pages/feature_engineering.py` (`utils/feature_engineering.py`) fit on the active working DataFrame as a whole; exporting an engineered dataset and feeding it to `Model Studio` leaks distributional parameters of learned transforms (such as `StandardScaler` or `PowerTransformer`) unless done inside the pipeline.
- **Known limitation:** When `time_series_cv=True`, `TimeSeriesSplit` does not produce test folds for the initial training block, so `cross_val_predict` raises and is caught in `utils/ml.py` (leaving `oof_proba = None` and falling back to holdout probabilities in `pages/ml.py` when validation probabilities are unavailable).
- **Known limitation:** `predict_with_model` and `generate_standalone_inference_code` assume model artifacts remain scoped to batch CLI/UI execution rather than an HTTP microservice or model registry.

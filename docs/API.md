# CloudInsight AI — Internal API & Modular Interface Reference

This document details the primary public and internal API contracts across CloudInsight AI's domain modules in `Utils/`.

## 0. Centralized Resource Governance & Limits (`Utils/limits.py`)

The `ResourcePolicy` dataclass and constants enforce deterministic system ceilings:
- `MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024` (200 MB single file upload ceiling)
- `MAX_AGGREGATE_UPLOAD_BYTES: int = 250 * 1024 * 1024` (250 MB batch upload ceiling)
- `MAX_UPLOAD_FILES: int = 20` (Batch file descriptor limit)
- `MAX_INGESTION_ROWS: int = 1_000_000` (Streaming row ceiling)
- `MAX_INGESTION_COLUMNS: int = 200` (Column dimension cap)
- `MAX_INGESTION_CELLS: int = 20_000_000` (Global matrix ceiling: $rows \times cols$)
- `MAX_DATAFRAME_MEMORY_BYTES: int = 400 * 1024 * 1024` (400 MB deep DataFrame RAM budget)
- `MAX_HIGH_CARDINALITY_STRING_CELLS: int = 5_000_000` (CPython pointer indirection ceiling)
- `MAX_MODEL_FILE_SIZE: int = 100 * 1024 * 1024` (100 MB model archive deserialization ceiling)
- `MAX_VISUALIZATION_ROWS: int = 25_000` (Plotly rendering downsampling threshold via LTTB)
- `MAX_ANALYSIS_ROWS: int = 50_000` (`COMPARE_MAX_ANALYSIS_ROWS` for statistical pairwise tests)

---

## 1. Data Ingestion & Storage (`Utils/paths.py`, `Utils/parsing.py`, `Utils/frames.py`)

### `read_tabular(source: Any, filename: str | None = None, max_rows: int | None = None) -> pd.DataFrame`
Reads and normalizes a dataset from a file path, file buffer, or raw bytes.
- **Parameters:**
  - `source`: File path (`str` or `Path`), file-like stream with `.read()`, or raw `bytes`.
  - `filename`: Optional explicit filename to guide parser selection.
  - `max_rows`: Optional integer row cap for streaming chunk reading.
- **Returns:** Cleaned `pd.DataFrame` with normalized column names.
- **Raises:**
  - `AIConversionRequired`: If the source contains unstructured text/PDF without tabular delimiters.
  - `ValueError`: If resource ceilings (200 MB, 1M rows, 200 cols, 20M cells) are exceeded or security checks fail (XXE, zip bomb, path traversal).

### `sanitize_for_csv_export(df: pd.DataFrame) -> pd.DataFrame`
Neutralizes CSV formula injection tokens in user-supplied strings.
- **Parameters:**
  - `df`: Target `pd.DataFrame`.
- **Returns:** Sanitized copy of `pd.DataFrame` with leading single quotes prepended to cells starting with `=`, `+`, `-`, `@`, `\t`, `\r`.

### `get_unique_filename(filename: str, directory: Path | str, extra_names: Iterable[str] | None = None) -> str`
Allocates a deterministic, collision-safe filename on disk using atomic filesystem reservation markers (`.filename.reserve`).
- **Parameters:**
  - `filename`: Proposed base filename (e.g. `data.csv`).
  - `directory`: Target destination directory.
  - `extra_names`: Optional set of in-memory names to avoid colliding with.
- **Returns:** Deterministic unique name (e.g. `data_1.csv`).

---

## 2. Machine Learning Studio (`Utils/ML.py`)

### `detect_problem_type(df: pd.DataFrame | None, target_col: str | None) -> str`
Infers whether a prediction task is `"Classification"` or `"Regression"`.
- **Heuristic:** Non-numeric targets or numeric targets with low unique ratios (<5% distinct values or <=20 unique values) are classified as `"Classification"`. Otherwise `"Regression"`. Resilient against `None` and empty DataFrames.

### `train_and_evaluate_model(...) -> dict[str, Any]`
Trains, validates, benchmarks, and generates explainability metrics for a scikit-learn model pipeline.
- **Parameters:**
  - `df`: Source `pd.DataFrame`.
  - `target_col`: Target column name.
  - `feature_cols`: List of predictor column names.
  - `model_name`: Chosen algorithm (`"Random Forest"`, `"Gradient Boosting"`, `"Decision Tree"`, `"Logistic Regression"`, `"Linear Regression"`, `"Ridge Regression"`).
  - `problem_type`: `"Classification"` or `"Regression"`.
  - `test_size`: Fraction of holdout test set (e.g. `0.2`).
  - `random_state`: Integer seed for reproducibility.
  - `time_series_cv`: Boolean flag to use `TimeSeriesSplit` instead of shuffled K-Fold.
- **Returns:** Dictionary containing trained `pipeline`, evaluation metrics (accuracy, precision, recall, F1, R², RMSE, MAE), confusion matrix, feature importances, cross-validation metrics, and baseline scores.

### `save_trained_model(results: dict, name: str, directory: Path | str = None) -> Path`
Serializes a trained pipeline with Joblib and creates a companion HMAC-SHA256 signature file.
- **Returns:** Path to the saved `.joblib` model file.

### `load_trained_model(model_path: Path, allowed_dir: Path | str = None) -> dict`
Validates cryptographic HMAC-SHA256 authenticity and deserializes the model bundle.
- **Raises:** `ValueError` if the signature is missing or does not match the content digest.

### `predict_with_model(bundle: dict, df: pd.DataFrame) -> pd.DataFrame`
Executes batch inference using a loaded model bundle.
- **Returns:** Copy of `df` with appended `prediction` and optional `probability_*` columns. Proactively flags categorical feature drift in `preds_df.attrs["category_drift"]`.

---

## 3. Privacy & Governance (`Utils/privacy.py`)

### `detect_sensitive_columns(df: pd.DataFrame) -> dict[str, str]`
Scans DataFrame headers and sampled values to identify PII, credentials, and keys.
- **Returns:** Dictionary mapping flagged column names to reason strings (e.g. `{"customer_id": "identifier token", "email_addr": "email-like values"}`).

### `apply_exclusions(df: pd.DataFrame, excluded_cols: list[str]) -> tuple[pd.DataFrame, bool]`
Strips specified columns from a DataFrame before passing it to AI prompts.
- **Returns:** `(filtered_df, success_bool)`.

---

## 4. Dataset Drift & Statistical Comparison (`Utils/compare_logic.py`)

### `column_drift_rows(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[dict[str, Any]]`
Evaluates statistical drift across shared columns of two datasets.
- **Parameters:**
  - `df_a`: Baseline dataset `pd.DataFrame`.
  - `df_b`: Comparison dataset `pd.DataFrame`.
- **Statistical Tests:**
  - Continuous features: Two-sample Kolmogorov-Smirnov test ($D_{n,m} = \sup |F_{1,n}(x) - F_{2,m}(x)|$) evaluated via `scipy.stats.ks_2samp`.
  - Multi-test correction: Monotonic Benjamini-Hochberg FDR control ($Q_{(i)} = \min(1.0, \min(Q_{(i+1)}, \frac{M}{i}P_{(i)}))$). Flags drift when $Q < 0.05$ (or raw $P < 0.01$ for $M=1$).
  - Discrete/categorical features: Total Variation Distance ($\text{TVD} = \frac{1}{2} \sum |P(x) - Q(x)|$). Flags drift when $\text{TVD} \ge 0.20$.
  - Missingness drift: Flags when $|\Delta\%| \ge 10.0\%$.
- **Returns:** List of column drift summary dictionaries.

### `benjamini_hochberg_correction(p_values: list[float | None]) -> list[float | None]`
Applies Benjamini-Hochberg (1995) step-up FDR adjustment to an array of raw p-values with backward monotonicity enforcement.
- **Parameters:**
  - `p_values`: List of raw p-values ($P \in [0, 1]$ or `None` for un-evaluated columns).
- **Returns:** List of adjusted q-values of identical length preserving input order.

### `schema_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, Any]`
Computes structural schema diff between two datasets: common columns, added columns, dropped columns, and data type mismatches.

---

## 5. ReportLab PDF Generation (`Utils/PDF.py`)

### `generate_pdf_report(df: pd.DataFrame, dataset_name: str, report_title: str, ...) -> bytes`
Compiles an executive analytical PDF report.
- **Sections:** Dataset hygiene gauge, descriptive numerical summary, IQR outlier bounds, categorical frequency breakdowns, correlation matrices, and optional high-DPI matplotlib figures.
- **Returns:** Raw binary bytes of the compiled PDF.

---

## 6. Upcoming Computational Interfaces (`Utils/feature_engineering.py` & `Utils/eda_logic.py`)

The platform extracts presentation-decoupled algorithms into pure domain interfaces:

### `Utils.feature_engineering`
- `scale_features(df: pd.DataFrame, columns: list[str], method: str) -> tuple[pd.DataFrame, list[str]]`
- `encode_features(df: pd.DataFrame, columns: list[str], method: str, max_categories: int = 15) -> tuple[pd.DataFrame, list[str]]`
- `transform_power(df: pd.DataFrame, columns: list[str], method: str) -> tuple[pd.DataFrame, list[str]]`
- `create_interaction_feature(df: pd.DataFrame, col_a: str, col_b: str, operation: str) -> tuple[pd.DataFrame, str]`
- `compute_mutual_information(df: pd.DataFrame, target_col: str, feature_cols: list[str], problem_type: str) -> pd.DataFrame`

### `Utils.eda_logic`
- `calculate_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame`
- `detect_tukey_outliers(df: pd.DataFrame, column: str, k: float = 1.5) -> dict[str, Any]`

# TabuLens — Utility API Reference

This document summarizes the public interfaces and behavior of the decoupled computational and security modules in `utils/`. Signatures reflect the current implementation.

---

## 1. Resource Policy & Storage Roots (`utils/limits.py`)

`ResourcePolicy` and module-level constants define storage paths and enforcement bounds:

- `PROJECT_ROOT`, `DATASETS_DIR` (`Datasets/`), `REPORTS_DIR` (`Reports/`), `MODELS_DIR` (`Models/`), `REPORT_TEMPLATES_DIR` (`Reports/templates/`), `CONVERSIONS_MANIFEST` (`Datasets/.conversions.json`)
- `MAX_UPLOAD_BYTES = 200 * 1024 * 1024` (200 MB)
- `MAX_AGGREGATE_UPLOAD_BYTES = 250 * 1024 * 1024` (250 MB)
- `MAX_UPLOAD_FILES = 20`
- `MAX_INGESTION_ROWS = 1_000_000`
- `MAX_COMBINED_ROWS = 1_000_000`
- `MAX_INGESTION_COLUMNS = 200`
- `MAX_INGESTION_CELLS = 20_000_000`
- `ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES = 400 * 1024 * 1024` (400 MB deep memory)
- `ResourcePolicy.MAX_HIGH_CARDINALITY_STRING_CELLS = 5_000_000`
- `ResourcePolicy.MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024` (100 MB)
- `ResourcePolicy.MAX_VISUALIZATION_ROWS = 25_000`
- `ResourcePolicy.MAX_ANALYSIS_ROWS = 50_000`
- `TABULAR_EXTENSIONS`: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.jsonl`, `.ndjson`, `.parquet`, `.xml`, `.html`, `.htm`
- `TEXT_EXTENSIONS`: `.txt`, `.log`, `.md`, `.rst`, `.sql`
- `DOCUMENT_EXTENSIONS`: `.pdf`
- `AIConversionRequired(ValueError)`: Raised when non-tabular text/PDF/HTML input cannot be parsed natively; carries `.raw_text` and `.filename`.

---

## 2. Ingestion, Path Safety & Normalization (`utils/paths.py`, `utils/parsing.py`, `utils/markup_parsing.py`, `utils/frames.py`, `utils/batch.py`)

### `read_tabular(source: Any, filename: str | None = None, max_rows: int | None = None, sep: str | None = None) -> pd.DataFrame` (`utils/paths.py`)
Parses a path, stream, string, or byte payload while enforcing size, row, column, cell, memory, ZIP-bomb, and XML DTD checks.
- **Raises:** `ValueError` on limit/security/format errors; `AIConversionRequired` for unstructured text, PDFs, or HTML lacking simple tables.

### `read_dataset(dataset: str, max_rows: int | None = None) -> pd.DataFrame` (`utils/paths.py`)
Resolves `dataset` strictly inside `DATASETS_DIR` and parses it via `read_tabular()`.

### `resolve_dataset_path(dataset: str) -> Path` (`utils/paths.py`)
Rejects `..`, drive-relative paths, `:` (Windows ADS), and Windows reserved device stems (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`), ensuring the target stays directly inside `DATASETS_DIR`.

### `safe_stem(value: str, fallback: str = "model", max_len: int = 100) -> str` (`utils/paths.py`)
Sanitizes a name to `[A-Za-z0-9-_ ]` with Windows reserved-name and length guards.

### `get_unique_filename(filename: str, directory: Path | str = DATASETS_DIR, extra_names: Iterable[str] | None = None) -> str` (`utils/paths.py`)
Allocates a collision-safe filename using thread locking and atomic `.reserve` marker files (`O_CREAT | O_EXCL`). Pair with `release_filename_reservation()` if ingestion aborts before writing.

### `atomic_write(target_path: Path | str, content: Any, mode: str = "w", encoding: str = "utf-8") -> Path` (`utils/paths.py`)
Writes via a unique temporary file and `os.replace`, clearing any matching filename reservation.

### `cleanup_storage(...) -> list[str]` & `delete_dataset(filename: str, directory: Path | str = DATASETS_DIR) -> bool` (`utils/paths.py`)
Prunes expired temporary files / orphan `.joblib.sig` files or safely deletes a contained dataset file and its manifest entry.

### `cached_load_dataframe(file_payload: bytes | str, file_name: str) -> pd.DataFrame` (`utils/parsing.py`)
`@st.cache_data(ttl=3600, max_entries=50)` wrapper that routes parsing by extension and validates `MAX_DATAFRAME_MEMORY_BYTES`.

### `normalize_and_deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame` (`utils/frames.py`)
Strips column names, replaces empty names with `"unnamed"`, and appends `_2`, `_3`, ... on duplicates.

### `sanitize_for_csv_export(df: pd.DataFrame | None) -> pd.DataFrame | None` (`utils/frames.py`, re-exported in `utils/paths.py`)
Prepends `'` to non-numeric string cells and headers starting with `=`, `+`, `-`, `@`, `\t`, or `\r` (after stripping leading whitespace for `=`, `+`, `-`, `@` checks).

### `merge_frames(frames: Sequence[tuple[str, pd.DataFrame]]) -> pd.DataFrame` (`utils/batch.py`)
Concatenates `(filename, df)` pairs and prepends a collision-free provenance column (`source_file`, `uploaded_file`, or `source_file_N`).

---

## 3. Session Identity & Caching (`utils/dataset_ui.py`)

- `dataset_fingerprint(dataset: str, force_refresh: bool = False) -> str`: Computes SHA-256 of the file on disk (cached by `(path, mtime_ns, size)` unless `force_refresh=True`).
- `dataframe_fingerprint(df: pd.DataFrame | None) -> str`: Computes SHA-256 over DataFrame shape, columns, dtypes, and row contents.
- `results_match_active(results: dict[str, Any] | None, selected_file: str | None, df: pd.DataFrame | None = None) -> bool`: Verifies cached ML/PDF results match the currently active dataset name and cryptographic fingerprint.
- `BoundedDatasetMemoryCache(max_bytes: int = 256 * 1024 * 1024)`: Thread-safe LRU cache with deep-copy-on-read isolation.
- `load_dataset_cached(dataset: str, max_rows: int | None = None) -> pd.DataFrame`
- `save_and_activate_dataset(df: pd.DataFrame, filename: str, sanitize: bool = True) -> Path`

---

## 4. Quality, Privacy & Cleaning (`utils/quality.py`, `utils/privacy.py`, `utils/preprocessing.py`)

- `quality_metrics(df: pd.DataFrame) -> dict[str, Any]` (`utils/quality.py`): Returns `rows`, `cols`, `missing_cells`, `duplicate_rows`, `completeness`, `uniqueness`, `index` (`(completeness + uniqueness) / 2`), and `is_empty`.
- `quality_index(df: pd.DataFrame) -> float`: Convenience accessor returning `quality_metrics(df)["index"]`.
- `detect_sensitive_columns(df: pd.DataFrame) -> dict[str, str]` (`utils/privacy.py`): Maps `{column: reason}` using column-name heuristics and up to `SAMPLE_ROWS = 100` sampled values (email, secret token, IBAN, Luhn-valid credit card, phone number).
- `apply_exclusions(df: pd.DataFrame, excluded_columns: Iterable[str] | None) -> tuple[pd.DataFrame, bool]`: Drops excluded columns; returns `(empty_cols_df, False)` if all columns are excluded.
- `detect_sensitive_text(raw_text: str | None) -> list[str]`: Scans up to 50,000 characters of raw text for sensitive patterns.
- `remove_duplicates(df: pd.DataFrame) -> pd.DataFrame` (`utils/preprocessing.py`)
- `fill_missing_values(df: pd.DataFrame, numeric_strategy: str = "mean", categorical_strategy: str = "mode", datetime_strategy: str = "ffill") -> pd.DataFrame`: Supports numeric (`mean`, `median`, `zero`), categorical (`mode`, `unknown`), and datetime (`ffill`, `ffill_bfill`, `none`); raises `ValueError` on unknown strategies.
- `drop_missing_values(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame`

---

## 5. EDA & Feature Engineering (`utils/eda_logic.py`, `utils/feature_engineering.py`)

- `calculate_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame`: Returns `["Feature", "R² with other features", "VIF", "Multicollinearity Severity"]` sorted descending by `VIF`.
- `detect_tukey_outliers(df: pd.DataFrame, column: str, k: float = 1.5) -> dict[str, Any]`: Returns quartile bounds, IQR fences, outlier counts/indices, and `|Z| >= 3.0` counts.
- `audit_outliers(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame`: Batch outlier table across `columns`.
- `compute_outlier_summary(df: pd.DataFrame, columns: list[str] | None = None, k: float = 1.5) -> pd.DataFrame`: Returns `["Feature", "Q1", "Q3", "IQR", "Lower Fence", "Upper Fence", "IQR Outlier Count", "IQR Outlier %", "Z-Score Count", "Z-Score %", "Skewness", "Kurtosis", "Distribution Shape"]`.
- `calculate_anova(df: pd.DataFrame, group_col: str, num_target: str) -> tuple[float, float, str, pd.DataFrame]`: One-way ANOVA across groups with `>= 3` records (`f_statistic`, `p_value`, `p_formatted`, `grp_summary`).
- `scale_features(df, columns, method) -> tuple[pd.DataFrame, list[str]]` (`standard`, `minmax`, `robust`)
- `encode_features(df, columns, method, max_categories=15) -> tuple[pd.DataFrame, list[str]]` (`onehot`, `ordinal`)
- `transform_power(df, columns, method) -> tuple[pd.DataFrame, list[str]]` (`log1p`, `sqrt`, `yeo-johnson`)
- `create_interaction_feature(df, col_a, col_b, operation) -> tuple[pd.DataFrame, str]` (`multiplication`, `ratio`, `difference`, `sum`)
- `compute_mutual_information(df, target_col, feature_cols, problem_type) -> pd.DataFrame`

---

## 6. Dataset Comparison & Drift (`utils/compare_logic.py`)

### `schema_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[list[str], list[str], list[str]]`
Returns `(common, only_a, only_b)` as sorted column-name lists.

### `benjamini_hochberg_correction(p_values: Sequence[float]) -> list[float]`
Applies Benjamini-Hochberg step-up FDR adjustment returning monotonic $q$-values bounded in $[0.0, 1.0]$.

### `column_drift_rows(df_a: pd.DataFrame, df_b: pd.DataFrame, max_rows: int = COMPARE_MAX_ANALYSIS_ROWS, random_state: int = 42) -> list[dict[str, Any]]`
Evaluates common columns on deterministic samples up to `max_rows` (default `50_000`) and returns per-column dicts containing dtype match, missing percentages, means, relative mean shift, KS statistic, raw KS p-value, BH-adjusted `KS p-adj (FDR)` (drifted when `< 0.05`), `Categorical TVD` (drifted when `>= 0.20`), unique counts, and combined `Flags`.

---

## 7. Machine Learning & Model Security (`utils/ml.py`, `utils/model_security.py`)

### `detect_problem_type(df: pd.DataFrame | None, target_col: str | None) -> Literal["Classification", "Regression"]`
Returns `"Classification"` for non-numeric/boolean targets or integer-like targets with `<= 10` distinct values and `<= 0.5` unique ratio; otherwise `"Regression"`.

### `is_likely_id_column(df: pd.DataFrame | None, col: str | None) -> bool`
Flags common ID column names (`id`, `uuid`, `*_id`, etc.) and string/categorical columns with `>= 20` non-null rows and `> 50%` unique values.

### `train_and_evaluate_model(df, target_col, feature_cols, model_name, problem_type, test_size=0.2, random_state=42, time_series_cv=False) -> dict[str, Any]`
Splits data, fits `ColumnTransformer` + estimator `Pipeline`, evaluates holdout and dummy baseline metrics, runs up to 5-fold CV on `(X_train, y_train)`, computes binary OOF validation probabilities (`y_val_proba`) when applicable, and extracts feature importances.

### `evaluate_threshold_metrics(y_true, y_proba, classes, threshold=0.50, positive_class=None) -> dict[str, Any]`
Computes `accuracy`, weighted `precision`/`recall`/`f1_score`, `f1_macro`, `mcc`, `confusion_matrix`, `positive_class`, `positive_precision`, `positive_recall`, `positive_f1`, `positive_support`, and `decision_threshold`.

### `predict_with_model(bundle, df, feature_cols=None, threshold_override=None) -> pd.DataFrame`
Validates feature presence, numeric coercion, and finiteness; runs inference (applying `decision_threshold` or `threshold_override` for binary classifiers); attaches unseen-category warnings in `output.attrs["category_drift"]`; and appends `prediction` (or `prediction_N`).

### `generate_standalone_inference_code(bundle, sample_df=None, model_rel_path="model.joblib") -> str`
Synthesizes a standalone CLI Python script that verifies `.joblib.sig` HMAC-SHA256 digests, enforces size/schema checks, applies saved/overridden thresholds, and writes sanitized CSV predictions.

### `save_trained_model(res, model_name, directory=None) -> Path` & `load_trained_model(path, require_signature=True, allowed_dir=None, trusted=False) -> ModelArtifactBundle`
Atomic `.joblib` persistence signed via `sign_model_artifact()` and verified in memory via `verify_model_artifact_signature()` (`utils/model_security.py`) before deserialization. Raises `ValueError` on traversal, oversize artifacts, missing signatures, or digest mismatches.

---

## 8. Sampling, Charts, PDF Reporting & Theme (`utils/sampling.py`, `utils/charts.py`, `utils/pdf.py`, `utils/theme.py`, `utils/logsys.py`)

- `sample_for_visualization(df, max_rows=25_000, random_state=42)` & `sample_for_analysis(df, max_rows=50_000, random_state=42)` (`utils/sampling.py`)
- `create_hygiene_gauge`, `create_histogram_plot`, `create_box_violin_plot`, `create_scatter_plot`, `create_bar_count_plot`, `downsample_timeseries`, `create_line_chart`, `create_pie_treemap_plot`, `create_correlation_heatmap` (`utils/charts.py`)
- `generate_pdf_report(df, dataset_name, report_title="Executive Data Analytics Report", author_name="TabuLens", include_ai_insights=None, include_charts=True, source_rows=None, analyzed_rows=None, excluded_columns=None) -> bytes` & `validate_report_template(config)` (`utils/pdf.py`)
- `safe_error_boundary`, `sanitize_error_message`, `page_header`, `section_card_header`, `metric_card`, `style_plotly_fig` (`utils/theme.py`)
- `get_logger(module)` (`utils/logsys.py`): Configures stderr logging at `TABULENS_LOG_LEVEL` (default `WARNING`).

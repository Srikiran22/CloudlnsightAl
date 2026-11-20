# TabuLens — Project Technical Reference

Repository: https://github.com/Srikiran22/CloudlnsightAl  
Branch reviewed: main  
Snapshot: commit 6b64de04f87bf79a9e81226023374e969e22ad12 (2026-11-16)

## 1. What the project is

TabuLens is a local-first tabular data exploration, statistical drift auditing and machine-learning workbench. It is a Python/Streamlit application that takes a user from local dataset ingestion through cleaning, exploratory statistics, feature engineering, supervised model training and evaluation, model export, batch inference, dataset comparison and PDF reporting.

Its defining choices are bounded local ingestion; leakage-aware scikit-learn preprocessing pipelines; cross-validation and dummy baselines; binary decision-threshold tuning; KS tests with Benjamini–Hochberg false-discovery-rate correction; categorical Total Variation Distance; HMAC-SHA256 verification before joblib deserialization; standalone Python inference; and local PDF audit reports.

It is not primarily a chatbot, hosted model API or data warehouse. The most accurate identity is a local tabular data science and applied ML workbench with emphasis on statistical diagnostics, reproducible preprocessing, bounded ingestion and portable inference.

## 2. Repository inspection method

This description was reconstructed from the recursive repository tree, implementation files, test modules, project configuration, architecture/methodology/setup/testing documents and recent commit history. The README was deliberately not used as the source of truth.

The recursive tree returned 89 entries and was not truncated. The review covered the application entry point, all ten workflow pages, core analytical/security utilities, test modules, dependency/build configuration and recent commit history. Implementation and tests were used to cross-check design-document claims. This is a project walkthrough, not a line-by-line defect audit. Tests were inspected, not executed during this review.

## 3. End-to-end user workflow

### Step 1 — Start a local workspace
app.py initializes Streamlit, theme, session state, navigation and the workspace sidebar. It generates deterministic demonstration data with a seeded NumPy generator: a Q1/Q2 SaaS customer pair (Q2 is deliberately shifted) and a housing-price regression dataset. The paired SaaS datasets make drift behavior demonstrable without private customer data.

### Step 2 — Ingest data
pages/upload.py provides file intake, SQLite database/table ingestion, quick paste and a local dataset catalog. The parser and policy layer supports CSV, TSV, XLSX/XLS, Parquet, JSON/JSONL, XML and SQLite paths, with document/text extraction paths also present.

utils/parsing.py, utils/limits.py and utils/paths.py apply byte, row, column, cell and memory ceilings; chunked delimited parsing; format-specific bounded JSON/Parquet/XML parsing; Excel ZIP expansion checks; XML DTD/external-entity rejection; path containment; and atomic writes. Multi-file ingestion/merge is bounded by aggregate and combined dataset limits. The product is designed for workstation-scale data, not unlimited distributed ingestion.

### Step 3 — Inspect and clean
pages/dashboard.py and utils/quality.py expose data hygiene/completeness/uniqueness summaries. pages/cleaning.py supports duplicate handling, missing-value imputation and export/save-to-catalog. utils/frames.py normalizes DataFrame columns and sanitizes spreadsheet formula triggers in CSV exports. Dataset fingerprints and cache invalidation help associate results with the active dataset.

Cleaning is user-directed. It should be presented as a decision with before/after counts and transformations, not as an invisible guarantee that data is now correct.

### Step 4 — Exploratory and statistical analysis
pages/eda.py and utils/eda_logic.py provide univariate summaries, correlation/covariance, VIF multicollinearity screening, Tukey IQR fences, Z-score diagnostics, distribution shape/skewness/kurtosis and group hypothesis testing including ANOVA-related logic.

pages/visualization.py and utils/charts.py provide Plotly histograms, box/violin plots, scatter/bubble plots, line/trend plots, categorical composition and correlation heatmaps. Large chart inputs are sampled or downsampled to maintain browser responsiveness; time-series downsampling preserves chronology and signal envelope.

### Step 5 — Feature engineering
pages/feature_engineering.py and utils/feature_engineering.py provide numeric scaling, categorical encoding, power transforms, mutual-information screening, interaction feature creation and export/activation of engineered datasets.

Important boundary: interactive feature engineering is a data-preparation tool. Any transformation that learns parameters must be fitted inside the model's training pipeline/fold to make evaluation leakage-safe. A manually exported transformed dataset is not automatically leakage-safe.

### Step 6 — Train and evaluate models
pages/ml.py calls utils/ml.py. The engine infers binary classification, multiclass classification or regression; includes logistic regression, decision trees, random forests and gradient boosting variants; and combines preprocessing with estimators using ColumnTransformer and Pipeline.

Numeric preprocessing includes median imputation and scaling; categorical preprocessing includes most-frequent imputation and one-hot encoding. The workflow uses stratified folds for classification, K-fold for regression, and offers a chronological TimeSeriesSplit path. DummyClassifier/DummyRegressor baselines are included. Metrics, cross-validation and holdout results are presented. Feature importance is extracted where supported and encoded categories can be grouped to their originating feature.

The strongest claim is transparent, bounded model development—not that the app automatically finds the universally best model.

### Step 7 — Tune binary decision thresholds
The Model Studio exposes a binary classification cutoff slider. utils/ml.py calculates precision, recall, F1 and Matthews Correlation Coefficient from probability estimates and stores the selected threshold. Out-of-fold validation probabilities are used for threshold exploration; the holdout test metrics are recalculated at the selected threshold.

A threshold is a decision policy, not an intrinsic property of a classifier. Its suitability depends on error costs, prevalence, calibration and the desired operating point.

### Step 8 — Save and verify model artifacts
Model bundles include the fitted pipeline and metadata such as model name, task, metrics, selected threshold, random seed, Python/scikit-learn versions and dataset fingerprint/provenance fields.

utils/model_security.py and utils/ml.py sign model bytes using HMAC-SHA256 and verify the signature before joblib deserialization. The local signing key uses atomic creation/permission handling. This is an integrity/authenticity gate within the local trust model; it does not replace key protection, key rotation or trusted artifact provenance.

### Step 9 — Batch and standalone inference
pages/predict.py loads a model, inspects metadata, accepts scoring data, runs batch predictions and exports results. Input features are validated; unseen categorical levels can be reported as category drift. Dataset fingerprints are compared to warn when scoring data matches the training data, preventing a rescore from being mistaken for independent evaluation.

utils/ml.py generates a standalone Python inference script that verifies the artifact signature, validates input and scores data without the Streamlit UI. This is a meaningful deployment boundary: the model can be used by a separate Python process.

### Step 10 — Compare datasets and drift
pages/compare.py and utils/compare_logic.py compare common columns, schema changes, missingness/mean summaries, duplicate checks and distribution changes.

Implemented methods include two-sample KS tests for numeric features, Benjamini–Hochberg FDR adjustment across numeric tests, categorical TVD with configured 0.20 effect-size threshold and minimum sample-size guard, plus schema comparison.

These are input-distribution/schema diagnostics. They do not establish concept drift, prove predictive accuracy fell, or prove retraining is required. Those claims require labels or a justified proxy.

### Step 11 — Generate audit reports
pages/report.py and utils/pdf.py generate PDF reports with dataset summaries, selected diagnostics, charts and configurable exclusions for sensitive columns. Report presets store reusable metadata/chart settings. Cached report results are checked against active dataset identity.

### Step 12 — Package and verify
pyproject.toml defines package metadata and tooling; requirements files describe dependencies. Dockerfile uses a multi-stage build, non-root runtime user and healthcheck; docker-compose.yml supports container execution. GitHub Actions runs a Python matrix, Ruff and unittest discovery. scripts/healthcheck.py provides container health checks; scripts/verify_pipeline.py provides local verification.

The test suite includes unit, Streamlit AppTest, ingestion, ML/drift, threshold, inference, deployment, offline-integrity and security-boundary modules. The test files demonstrate intended coverage; this review did not execute them.

## 4. Repository map

### Application pages
- app.py — initialization, synthetic benchmark data, home and navigation.
- pages/upload.py — file ingestion, SQLite, paste and catalog.
- pages/dashboard.py — dataset overview and health.
- pages/cleaning.py — deduplication, imputation and exports.
- pages/eda.py — statistical EDA and outlier diagnostics.
- pages/feature_engineering.py — scaling, encoding, transforms and feature screening.
- pages/visualization.py — interactive Plotly analytics.
- pages/ml.py — training, validation, metrics and threshold tuning.
- pages/compare.py — dataset comparison and drift diagnostics.
- pages/predict.py — model inspection and batch inference.
- pages/report.py — PDF audit reports and presets.

### Core utilities
- utils/parsing.py — bounded format-specific parsing.
- utils/limits.py — central resource policy and supported extensions.
- utils/paths.py — storage, path safety, atomic writes and file management.
- utils/frames.py — DataFrame normalization and safe CSV export.
- utils/dataset_ui.py — dataset selection, state, fingerprints and cache.
- utils/quality.py — data hygiene metrics.
- utils/privacy.py — heuristic sensitive-column/text screening and exclusions.
- utils/eda_logic.py — VIF, outlier, distribution and group-test calculations.
- utils/feature_engineering.py — scaling, encoding, transforms, interactions and mutual information.
- utils/compare_logic.py — schema and drift calculations.
- utils/ml.py — training, validation, threshold metrics, artifacts and inference.
- utils/model_security.py — HMAC key handling, signing and verification.
- utils/batch.py — multi-frame merge logic.
- utils/sampling.py — bounded deterministic sampling.
- utils/charts.py — Plotly charts and time-series downsampling.
- utils/pdf.py — PDF generation and template validation.
- utils/theme.py — UI tokens and shared components.
- utils/logsys.py — logging.
- utils/markup_parsing.py — markup/text parsing helper.

### Delivery and verification
- tests/ — 19 test modules covering workflow, statistical logic, parsing, security and packaging.
- scripts/verify_pipeline.py — verification CLI.
- scripts/healthcheck.py — container healthcheck.
- docs/ARCHITECTURE.md — architecture.
- docs/METHODOLOGY.md — statistical/ML methodology.
- docs/API.md — API/utility reference.
- docs/SETUP.md — local and Docker setup.
- docs/TESTING.md — QA strategy.
- docs/superpowers/specs/ and docs/superpowers/plans/ — design specs and implementation plans.
- Dockerfile, docker-compose.yml, .github/workflows/ci.yml — deployment and CI.
- pyproject.toml and requirements files — package/dependencies.
- SECURITY.md, CONTRIBUTING.md, CHANGELOG.md, LICENSE — project governance/history.

## 5. What is distinctive

1. Complete analyst workflow, not a single model demo.
2. KS with multiple-testing control and categorical TVD rather than a generic drift badge.
3. Standalone inference and batch scoring move models beyond the training UI.
4. HMAC verification before joblib loading acknowledges serialized models as a trust boundary.
5. Bounded ingestion treats malformed/resource-exhausting user data as a product concern.
6. Pipeline-based preprocessing, CV and dummy baselines improve evaluation discipline.
7. Dataset fingerprints, cache invalidation and training-overlap warnings preserve data/result context.

## 6. Product boundaries and honest claims

- PII screening is heuristic and may miss sensitive values or raise false positives. It is not anonymization or a compliance certification.
- KS/FDR and TVD detect distribution differences; they do not directly monitor accuracy or prove concept drift.
- HMAC is useful only when the correct signing key is protected. It does not make pickle intrinsically safe or prove model quality.
- joblib/pickle loading must remain restricted to trusted artifacts.
- Standalone inference is not automatically an HTTP serving service, model registry, autoscaler or orchestrator.
- The repository does not demonstrate a hosted registry, scheduled production monitoring service, automatic retraining loop, distributed execution engine or multi-user governance layer.
- Recent refactoring removed external Gemini/GenAI dependencies and the AST sandbox. Do not describe Gemini-powered analysis as a current integrated feature. Residual comments/test fixtures are not evidence of a live integration.

## 7. Product evolution from commit history

Recent commits show the project evolving from a local analytics workspace into a local-first tabular ML studio. Important milestones include statistical EDA/visualization/drift/PDF reporting; scikit-learn pipelines and HMAC persistence; dedicated feature engineering; Docker/CI/verification; decision threshold tuning; standalone inference export; and removal of GenAI/sandbox dependencies.

This is an important product decision: the strongest direction was not adding an LLM to every screen, but narrowing scope around measurable statistical and ML workflow behavior.

## 8. Technical interview demonstration

Use the seeded SaaS churn Q1/Q2 pair:
1. Load both datasets and inspect schema, missingness and hygiene.
2. Inspect distributions and outlier diagnostics.
3. Compare Q1/Q2; explain KS statistic vs adjusted p-value and TVD for categorical proportions.
4. Train a churn classifier; show preprocessing, cross-validation and dummy baseline.
5. Move the decision threshold and explain precision/recall and business costs.
6. Save the signed model.
7. Score a separate dataset and show unseen-category/training-overlap warnings where applicable.
8. Generate standalone inference code and a PDF report.

Explain what each metric means and does not mean. A drift alert is an investigation trigger, not an automatic retraining command.

## 9. Resume positioning

**Title:** TabuLens — Local-First Tabular Data Science & ML Workbench

**Resume description:**

Built a local-first Python/Streamlit workbench for tabular data ingestion, statistical exploration, feature engineering and supervised machine learning. Implemented bounded multi-format parsing, leakage-aware scikit-learn pipelines, cross-validation with dummy baselines, KS drift testing with Benjamini–Hochberg FDR correction, categorical TVD diagnostics, binary decision-threshold tuning, HMAC-verified model artifacts, batch scoring and standalone Python inference export.

Add performance numbers only after benchmarking on documented datasets and hardware.

## 10. Repository references

- Repository tree: https://github.com/Srikiran22/CloudlnsightAl/tree/main
- App entry: https://github.com/Srikiran22/CloudlnsightAl/blob/main/app.py
- Architecture: https://github.com/Srikiran22/CloudlnsightAl/blob/main/docs/ARCHITECTURE.md
- Methodology: https://github.com/Srikiran22/CloudlnsightAl/blob/main/docs/METHODOLOGY.md
- ML engine: https://github.com/Srikiran22/CloudlnsightAl/blob/main/utils/ml.py
- Drift calculations: https://github.com/Srikiran22/CloudlnsightAl/blob/main/utils/compare_logic.py
- Bounded parser: https://github.com/Srikiran22/CloudlnsightAl/blob/main/utils/parsing.py
- Model signing: https://github.com/Srikiran22/CloudlnsightAl/blob/main/utils/model_security.py
- Tests: https://github.com/Srikiran22/CloudlnsightAl/tree/main/tests
- CI: https://github.com/Srikiran22/CloudlnsightAl/blob/main/.github/workflows/ci.yml
- Refactor commit: https://github.com/Srikiran22/CloudlnsightAl/commit/69a1cb3e3e8ede6426e047e45235599a0f17252c
- GenAI removal commit: https://github.com/Srikiran22/CloudlnsightAl/commit/dd769175adb4da109593b0c33f8203d0d2250a30
- Reviewed ML/HMAC/drift commit: https://github.com/Srikiran22/CloudlnsightAl/commit/6b64de04f87bf79a9e81226023374e969e22ad12


## 11. Re-verification addendum — 2026-11-20

This addendum is based on a second pass over the current `main` tree and implementation. The recursive tree returned **91 entries** and was not truncated. The original two documentation files are included in that count.

### 11.1 Additional capabilities that needed to be recorded

- **Preprocessing utility (`utils/preprocessing.py`):** exact duplicate removal; numeric imputation using mean, median or zero; categorical imputation using mode or the literal `Unknown`; datetime fill using forward-fill, forward-fill plus back-fill, or no change; and complete-case row dropping. The cleaning UI exposes mean, median, constant, drop-incomplete and preserve-null choices. The datetime behavior is a library capability, not a separate UI choice on the current cleaning page.
- **Temporal imputation assumption:** forward-fill only has its intended no-lookahead interpretation when rows are already in chronological order. The cleaning page does not itself establish that ordering before calling the default forward-fill helper. Do not describe this as universally time-safe.
- **HTML/XML parsing (`utils/markup_parsing.py`):** XML parsing uses DTD/entity rejection and bounded depth; HTML parsing extracts the largest tabular HTML table, handles column spans, and hands off cases with row spans or no tables as unsupported structured conversions.
- **Text/PDF extraction:** delimited-looking supported text may be parsed as a table. Otherwise a bounded text preview is raised through `AIConversionRequired`. PDF extraction is bounded to 30 pages and 50,000 characters, but PDFs are not natively converted into tables by the current ingestion path.
- **Batch provenance (`utils/batch.py`):** merged upload rows receive a source-file provenance column. The implementation selects a collision-free column name rather than overwriting an existing user column.
- **Dataset state and cache (`utils/dataset_ui.py`):** active dataset identity, file/DataFrame fingerprints, result-to-current-dataset matching, cache invalidation, a bounded in-memory cache with a 256 MiB budget, and cached dataset loading.
- **Structural quality metric (`utils/quality.py`):** the Data Hygiene Index is the average of cell completeness and row uniqueness. It is explicitly a structural heuristic, not a semantic validity, correctness or representativeness score.
- **Logging (`utils/logsys.py`):** standard-library logging to the Streamlit process stderr; documented intent is not to log dataset contents, API keys or raw model responses.
- **Additional delivery files:** `.streamlit/config.toml` configures Streamlit; `start.bat` is a Windows launcher; `.dockerignore` and `.gitignore` govern build/repository inputs.

### 11.2 Corrections to earlier claims and scope

- The repository contains **19** Python test modules, not 17. The current tree names are: `test_app_version.py`, `test_audit_regressions.py`, `test_caching_and_performance.py`, `test_core_pipeline.py`, `test_decision_threshold.py`, `test_deployment_config.py`, `test_inference_script_generation.py`, `test_ingestion_parsers.py`, `test_ml_and_drift.py`, `test_offline_integrity.py`, `test_outlier_diagnostics.py`, `test_package_structure.py`, `test_page_smoke.py`, `test_portfolio_documentation.py`, `test_security_boundaries.py`, `test_theme_and_boundaries.py`, `test_ui_hardening.py`, `test_ui_workflows.py`, and `test_verify_pipeline_cli.py`.
- The existing `docs/TESTING.md` still says 12 modules and 384 passing tests. That count/organization is stale against the current tree. This repository review inspected test files but did **not** execute the suite, so no current pass count is asserted here.
- The interactive Feature Engineering page uses direct fit/transform operations on the currently selected DataFrame. These are useful for exploration/export, but if scaling, power transformation, encoding, or feature selection is applied before model splitting, the learned transformation may see the eventual holdout data. For defensible evaluation, use training-fold-fitted transformations in the model Pipeline or make the transformation itself fold-aware.
- Binary threshold selection is exposed as a user-controlled slider and validation metrics are recalculated from out-of-fold probabilities. It is more accurate to describe this as interactive/manual threshold tuning than as a fully automated business-cost optimizer.
- The time-series training path uses `TimeSeriesSplit` for cross-validation. The same code attempts `cross_val_predict` for binary threshold probabilities and catches failures. Because `TimeSeriesSplit` does not produce a test fold for the initial training block, the standard `cross_val_predict` partition requirement may prevent OOF threshold probabilities in this mode. Treat time-series threshold tuning as a limitation to verify and improve with an explicit rolling-origin aggregation strategy.
- The repository metadata description on GitHub still mentions “Gemini insights”. The current source refactor removed the live Gemini/GenAI integration. The active uploader displays extracted text and asks for structured tabular files; it does not provide a live AI conversion workflow.
- Conversion-era compatibility code remains: `AIConversionRequired`, `.conversions.json`, and helpers such as `record_conversion` / `get_valid_conversion` are still in the utility layer and tests. They are not evidence of a currently integrated Gemini converter. The code also retains legacy comments and documentation that still mention Gemini.
- One upload helper description mentions DOCX, but the central extension set does not include `.docx`; it is not an accepted upload type in the current contract.
- Python packaging declares `requires-python >=3.12` in `pyproject.toml`, while the Dockerfile uses Python 3.11 and CI tests Python 3.11 and 3.12. The setup guide says Python 3.12+. Do not claim a consistent supported Python matrix until these declarations are aligned and verified.

### 11.3 Updated source-to-feature index

| Implementation | What it adds to the product story |
|---|---|
| `utils/preprocessing.py` | User-directed cleaning and datetime-aware fill strategies |
| `utils/markup_parsing.py` | XML and HTML table parsing with structured-document boundaries |
| `utils/dataset_ui.py` | Dataset identity, fingerprints, cache, workspace state and persistence |
| `utils/batch.py` | Multi-file merge with row-to-source provenance |
| `utils/quality.py` | Completeness/uniqueness-based structural hygiene heuristic |
| `utils/logsys.py` | Local diagnostic logging policy |
| `utils/limits.py` | Canonical accepted extension set and resource ceilings |
| `utils/paths.py` | Conversion-provenance compatibility helpers as well as safe storage/path operations |
| `start.bat` | Windows startup convenience |
| `.streamlit/config.toml` | Streamlit app defaults/theme configuration |


# Changelog

All notable changes to CloudInsight AI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Restored AI Subsystem in UI Navigation:** Connected `Pages/AIConverter.py` (AI File Converter), `Pages/AI.py` (AI Summary & Insights), and `Pages/AIChat.py` (Ask AI Chat) into `App.py` under the dedicated "AI & Gemini Assistant" navigation group.
- **Feature Engineering Pipeline State Persistence:** Resolved state reset in `Pages/FeatureEngineering.py` Tab 1 (Numerical Scaling), ensuring that chained transformations (such as prior categorical encodings or polynomial terms) in `st.session_state["fe_working_df"]` are preserved and accumulated rather than overwritten with `df.copy()`.
- **Hardened Catalog Save in Data Cleaning:** Refactored "Save to Workspace Catalog" in `Pages/Cleaning.py` to use `sanitize_for_csv_export()`, `atomic_write()`, `get_unique_filename()`, and `invalidate_dataset_cache()`, preventing CSV formula injection and file corruption during save operations.
- **Harmonized Cross-Page UI Terminology:** Updated stale cross-page textual references ("Import Files" -> "Data Hub & Ingest", "Train Model" -> "Model Studio & Training", "Open Predict & Score page" -> "Open Batch Scoring & Simulation") across `Pages/Cleaning.py`, `Pages/Predict.py`, `Pages/ML.py`, `Pages/AI.py`, and `Utils/dataset_ui.py`.
- **Feature Engineering Dataset Boundary Guard & Discard Action:** Added dataset origin tracking (`fe_dataset_origin`) to purge stale transformed features when switching active datasets in `Pages/FeatureEngineering.py`, and added a "Discard Changes" control.
- **Accessible Primary Button Contrast:** Introduced `--ci-accent-text` token in `Utils/theme.py` to ensure dark-mode primary buttons render with high-contrast text (`#0B0F19` on `#3B82F6`, 4.53:1 ratio), achieving WCAG 2.1 AA/AAA compliance across both themes.

### Added
- **Diátaxis Technical Documentation Architecture:** Authored comprehensive architectural guides in `docs/`: `ARCHITECTURE.md` (C4 models and boundaries), `SETUP.md` (reproducible setup and troubleshooting), `TESTING.md` (test strategy and mock design), `SECURITY.md` (threat model and cryptographic signing), and `API.md` (internal interfaces and contracts).
- **Technical Audit & Walkthrough Briefs:** Added root `AUDIT_REPORT.md` with full P0-P3 classifications, and `PROJECT_WALKTHROUGH.md` for technical review preparation.

## [2.2.0] - 2026-10-18

### Fixed
- **Light-mode consistency:** the CSS palette now follows the resolved Streamlit theme (`st.context.theme`) instead of a session flag, and a zero-height client bridge re-applies the palette instantly when the appearance changes. Dataframes, selectboxes, code chips and charts no longer stay dark on a light page.
- **Card styling on current Streamlit:** bordered containers moved to `stLayoutWrapper` in Streamlit 1.60, so the old `stVerticalBlockBorderWrapper` rules were dead. Cards now use the design tokens again (8 px radius, 20/24 px padding, token border) on every page, with the legacy selector kept for older releases.
- **Sidebar layout:** the brand wordmark is rendered with `st.logo` in the sidebar header (above the navigation) instead of a block that appeared below it; the active dataset panel and appearance note follow the navigation.
- **Sidebar state:** pages that auto-select the working dataset render the sidebar after selection, so the active dataset is never shown as "Nothing loaded yet" on first load.
- **Data hygiene gauge:** the indicator now uses the shared chart theming and design tokens, so the score and tick labels are legible in both appearances.
- **Chart legibility:** axis and legend text in Plotly charts uses the design tokens instead of Streamlit's low-contrast default grey.
- **Disabled buttons:** primary buttons now visibly mute when disabled (the forced primary colours were overriding the disabled state).
- **Cleaning workflow:** option/save controls are arranged in a balanced grid, redundant "Clean"/"Complete" delta pills are gone, and the result card only shows a row-delta when rows were removed.
- **Machine-learning metrics:** CV Macro F1, its range and the baseline F1 are reported as percentages to match the other metrics; the positive-class strip no longer renders `( 1 )` with chip padding.
- **AI pages:** governance cards use a balanced two-column layout (context exclusions next to the PII result, full-width key status and privacy notice), and the converter's sample-log action is contextual to Paste mode instead of floating in the header row.
- **PDF report:** saving a preset is a visible, disabled-until-named action, and the generated report path is shown in a copyable code block instead of wrapping mid-filename.
- **Compare table:** status values are colour-coded (drift / stable / identifier) for faster scanning.
- **Copy drift:** page titles match their navigation labels, "Ingest data" references now say "Import Files", and widget labels, radio options and action buttons use one sentence-case convention.

### Changed
- **Theme ownership:** appearance is now owned by Streamlit (browser preference or **⋮ → Settings → Theme**); both palettes are declared in `.streamlit/config.toml` and `Utils/theme.py`. The custom in-app toggle was removed because it could not re-theme canvas widgets without a full page reload.
- **Chart theming:** all Plotly charts render with `theme="streamlit"` and share `chart_palette()` instead of ad-hoc Plotly defaults.
- **Module structure:** `Utils/paths.py` was split into `limits.py` (resource ceilings, storage roots, supported formats), `frames.py` (column normalisation and export sanitisation) and `parsing.py` (format readers). `paths.py` keeps the public API and ingestion dispatch, so existing imports continue to work.
- **Observability:** best-effort exception handlers now log at DEBUG instead of silently swallowing errors.
- **Typing:** public utility APIs in `paths`, `frames`, `parsing`, `privacy`, `compare_logic`, `Charts`, `sampling`, `quality`, `batch`, `Preprocessing` and `dataset_ui` are annotated.

### Added
- Coverage reporting in CI and `[tool.coverage]` configuration.
- README section documenting the theming architecture.

## [2.1.0] - 2026-10-02

### Added
- **Statistical Drift Engine:** Added Kolmogorov-Smirnov (KS) testing with Benjamini-Hochberg False Discovery Rate (FDR) control and Categorical Total Variation Distance (TVD) for drift analysis across dataset versions.
- **Cryptographic Model Verification:** Enforced HMAC-SHA256 signing for serialized scikit-learn model bundles with atomic cross-process key acquisition and strict directory containment.
- **Central Cell Budget Governance:** Implemented bounded streaming parsers with a 20,000,000 cell ceiling and 200 MB ingestion ceiling across CSV, JSON, Parquet, XML, and multi-file merges.
- **Smart Delimiter & Format Detection:** Added adaptive delimiter inference supporting `,`, `;`, `\t`, and `|` with quoted string protection and safe scalar JSON parsing.
- **Unified Sampling Engine:** Introduced shared peak-preserving downsampling policies for charts (25k points) and analytical operations (50k rows) across EDA, Dashboard, and PDF reports.
- **Filesystem & Device Name Hardening:** Added path sanitization guarding against directory traversal, alternate data streams (ADS), and Windows reserved device stems.

### Changed
- **S3 Streaming Architecture:** Streamlined remote cloud dataset transfers directly to bounded disk temporary storage, eliminating high-memory materialization.
- **Time-Series ML Splitting:** Added strict chronological ordering and holdout isolation when `TimeSeriesSplit` cross-validation is selected.
- **PDF Report Customization:** Propagated sensitive column exclusion masks and AI analytical summaries to automated PDF export batches.

### Fixed
- Fixed timezone-aware datetime parsing to normalize cleanly to UTC epoch timestamps across preprocessing steps.
- Fixed atomic reservation concurrency under multi-threaded Streamlit re-evaluations.
- Fixed category handling for unseen categorical levels during model inference with proactive drift warnings.

## [2.0.0] - 2026-09-15

### Added
- **Gemini Multi-Modal Table Extraction:** Integrated Gemini API with chunked line-aligned extraction to convert unstructured documents (PDF, TXT, markdown, log files) into structured DataFrames.
- **Automated PII Screening:** Added privacy screening rules to detect and mask SSNs, passwords, email addresses, phone numbers, and financial identifiers before analytics or external API dispatch.
- **Multi-Dataset Comparison:** Added side-by-side schema diffing, row/column alignment, and statistical summaries.

### Changed
- Refactored session state to enforce immutable copy-on-read patterns and bounded LRU caching.
- Upgraded report generation with embedded matplotlib/seaborn figures and multi-encoding font resolution.

## [1.0.0] - 2026-08-01

### Added
- **Multi-Format Tabular Ingestion:** Native upload support for CSV, Excel, Parquet, JSON, and XML files.
- **Exploratory Data Analysis:** Automated distribution profiling, missing value imputation, and correlation matrices.
- **Interactive Visualizations:** Configurable Plotly charts including histograms, scatter plots, box plots, and heatmaps.
- **AutoML Baseline Training:** End-to-end model training for classification and regression tasks with cross-validation and feature importance analysis.

# Changelog

All notable changes to CloudInsight AI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

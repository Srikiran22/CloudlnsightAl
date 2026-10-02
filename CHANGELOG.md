# Changelog

All notable changes to CloudInsight AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-10-02

### Added
- **Central Cell Budget Governance:** Implemented `MAX_INGESTION_CELLS` (20,000,000 cells) across tabular readers (`read_tabular`, Parquet, and multi-file merge).
- **Model Deserialization Zero-Trust Default:** `load_trained_model` enforces `require_signature=True` and strict `MODELS_DIR` containment, rejecting unverified pickles prior to `joblib.load()`.
- **Atomic Signing Key Acquisition:** Cross-process atomic key acquisition using `O_CREAT | O_EXCL` flags with retry/backoff.
- **Shared Visualization & Analysis Sampling Policy:** Unified sampling policy (`sample_for_visualization` capped at 25,000 rows, `sample_for_analysis` capped at 50,000 rows) shared across Dashboard, EDA, Compare, and PDF reports.
- **Smart CSV Delimiter Detection:** Adaptive delimiter detection supporting `;`, `\t`, and `|` while preserving legitimate single-column CSVs and quoted commas.
- **Bounded JSON Streaming:** Strict streaming parser that fails closed on large unstreamable JSON files (>2 MB) rather than materializing unbounded memory.
- **Safe Filename Policy:** Sanitization protecting against Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`) and component lengths over 100 characters.
- **Separation of Ready Datasets vs Source Documents:** Dedicated separation between ready tabular datasets and raw unstructured documents (`.pdf`, `.txt`, `.log`, `.md`, `.sql`).
- **AI Session-State Lifecycle Management:** LRU context eviction bounded to 10 conversation contexts.
- **Report Template Schema Validation:** Type-safe schema validation function `validate_report_template` preventing boolean truthiness traps.
- **Model Metadata & Category Drift:** Recording training environment versions, hyperparameters, and dataset dimensions; runtime detection and reporting of unseen categorical levels.
- **Pyproject & Pinned CI Tooling:** Standardized `pyproject.toml` with Ruff configuration and pinned security audit tools.

### Changed
- **S3 Ingestion Memory Pipeline:** Redesigned S3 download to stream directly to bounded disk files with active byte limits, eliminating memory amplification.
- **Batch PDF Generation:** Propagated sensitive/excluded column selections and AI insight flags to batch report runs.
- **Timezone Datetime Normalization:** Robust normalization of timezone-aware timestamps to UTC epoch seconds in ML pipelines.
- **AI Prompts:** Replaced promotional persona text with evidence-based analytic prompt structures.

### Fixed
- Fixed crash when processing timezone-aware timestamps in ML feature extraction.
- Fixed non-atomic signing key creation race condition under concurrent workers.
- Fixed omission of sensitive column filters in batch PDF generation.
- Fixed memory spike during large S3 file downloads.
- Cleaned unused compatibility stubs (`CacheStub`).

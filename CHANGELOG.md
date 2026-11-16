# Changelog

All notable changes to TabuLens are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-11-16

### Refactored
- **Rebrand to TabuLens:** Renamed project to TabuLens (Tabular Data Exploration, Drift Auditing & Leak-Free Modeling) as an authentic, high-quality 4th-year student capstone project.
- **PEP 8 Namespace Normalization:** Normalized all source filenames and directories to standard lowercase `snake_case` (`app.py`, `utils/`, `pages/`).
- **Purged GenAI & Sandbox Dependencies:** Removed external Gemini API dependencies, OCR converters, and AST sandbox hacks in favor of a 100% local, mathematically rigorous tabular data science studio.
- **Streamlined UI Theming:** Replaced brittle 953-line DOM overrides with clean CSS design tokens (<100 LOC) and native Streamlit component styling.

### Added
- **Decision Threshold Optimization:** Implemented candidate threshold evaluation ($0.05 \le \tau \le 0.95$) for binary classification, evaluating precision, recall, F1, and Matthews Correlation Coefficient (MCC) in $O(1)$ from precomputed probabilities.
- **Standalone Inference Script Export:** Implemented automatic generation and export of self-contained Python inference scripts (`predict_standalone.py`) with cryptographic HMAC signature verification and schema enforcement.
- **Outlier & Distribution Diagnostics:** Added Tukey IQR fences with tunable multipliers ($k=1.5, k=3.0$), Z-score outlier detection, skewness, kurtosis, and distribution shape classification.
- **Production 2-Stage Docker:** Created an unprivileged multi-stage `Dockerfile` (`builder` and `runner` with `appuser`), optimizing container size and security.
- **GitHub Actions CI:** Added `.github/workflows/ci.yml` running unit test matrices and lint checks.
- **Pipeline Verification Runner:** Added `scripts/verify_pipeline.py` providing static environment checks (`--check-only`) and test suite verification.
- **Methodology Documentation:** Authored `docs/METHODOLOGY.md` detailing mathematical proofs and algorithms for two-sample KS tests, Benjamini-Hochberg FDR, VIF, Tukey fences, and decision thresholds.

# CloudInsight AI — Testing & Quality Assurance Strategy

This document details the testing architecture, regression verification strategy, mock boundaries, and quality assurance gates implemented in CloudInsight AI.

---

## 1. Testing Philosophy

1. **User-Visible Behavior Over Implementation Details:** Headless UI tests use Streamlit's official `AppTest` framework to simulate page lifecycles, clicks, selectbox mutations, and exception handling without running a physical browser.
2. **Deterministic & Isolated:** Tests use `tempfile.TemporaryDirectory` and monkeypatched storage roots (`paths.DATASETS_DIR`, `paths.MODELS_DIR`) to prevent side effects on working user datasets.
3. **No External Network Dependencies:** AI interactions (`google-genai`), cloud storage, and network requests are mocked using `unittest.mock`. Routine unit tests never touch third-party external services.
4. **Adversarial Security Verification:** Explicit test suites target path traversal, zip bombs, XML entity expansions (XXE), CSV formula injection, and unsigned model pickle deserialization.

---

## 2. Test Suite Organization

The test suite is located in `tests/` and comprises 8 dedicated test modules:

```text
tests/
├── test_page_smoke.py           # Headless AppTest boot checks across all 13 pages
├── test_security_boundaries.py  # XXE, DTD, zip bombs, formula injection, model signatures
├── test_core_pipeline.py        # Cryptographic fingerprints, cache LRU, pipeline preprocessing
├── test_ingestion_parsers.py    # CSV, TSV, JSON, JSONL, Parquet, XML streaming bounds
├── test_ml_and_drift.py         # TimeSeriesSplit, 5-fold CV, KS test FDR, TVD drift
├── test_ui_workflows.py         # File uploads, multi-file merging, PDF fonts, decimation
├── test_audit_regressions.py    # Regression coverage for confirmed audit findings
└── test_ai_conversion.py        # Chunked AI extraction, boundary reconciliation, schema alignment
```

---

## 3. Test Module Details

### 3.1 `test_page_smoke.py` (Headless Boot Checks)
- **Tooling:** `streamlit.testing.v1.AppTest`
- **Scope:** Bootstraps each of the 13 application pages headless:
  - `App.py`
  - `Pages/Upload.py`
  - `Pages/FeatureEngineering.py`
  - `Pages/AIConverter.py`
  - `Pages/Cleaning.py`
  - `Pages/Compare.py`
  - `Pages/EDA.py`
  - `Pages/Visualization.py`
  - `Pages/ML.py`
  - `Pages/Predict.py`
  - `Pages/AI.py`
  - `Pages/AIChat.py`
  - `Pages/Report.py`
  - `Pages/Dashboard.py`
- **Verification:** Asserts `len(at.exception) == 0` for all pages with both empty and populated session state contexts.

### 3.2 `test_security_boundaries.py` (Adversarial Security)
- **Model Deserialization Protection:** Generates unsigned, corrupted, and maliciously crafted pickling payloads (`__reduce__`). Verifies that `load_trained_model()` rejects unverified artifacts with explicit cryptographic signature errors.
- **XXE & Billion Laughs Defense:** Feeds XML payloads containing `<!DOCTYPE>` and `<!ENTITY>` tags; verifies immediate rejection by `_reject_dtd()`.
- **ZIP Bomb Defense:** Creates high-compression zip archives (>100x ratio); asserts `_check_zip_bomb()` raises `ValueError` before decompression.
- **Formula Injection Defense:** Asserts that strings starting with `=`, `+`, `-`, `@`, `\t`, `\r` are neutralized by `sanitize_for_csv_export()`.

### 3.3 `test_ml_and_drift.py` (Statistical & AutoML Correctness)
- **Kolmogorov-Smirnov & TVD Drift:** Compares synthetic baseline vs. drifted distributions, verifying Benjamini-Hochberg False Discovery Rate (FDR) p-value corrections.
- **TimeSeriesSplit Validation:** Ensures forward-chaining chronological splits without lookahead data leakage.
- **Baseline Benchmarking:** Verifies that classification and regression tasks report naive dummy baselines alongside cross-validation metrics.

### 3.4 `test_audit_regressions.py` (Audit Defect Regressions)
- Covers confirmed audit findings:
  - Confusion matrix non-zero counts for integer and string target variables.
  - Positive-class metric calculations (precision, recall, F1, support) for imbalanced binary classification.
  - Outbound AI context stripping of PII/identifier columns.
  - Model provenance cache invalidation upon single-cell dataset modifications.

---

## 4. Test Execution & CLI Commands

### 4.1 Pytest Execution (Recommended Primary Runner)

CloudInsight AI's test harness is fully compatible with `pytest`. All tests execute **completely offline** with zero live network calls:

```bash
# Execute full test suite with verbose reporting
pytest tests/ -v

# Execute full test suite with code coverage
pytest tests/ --cov=Utils --cov=Pages --cov=App --cov-report=term-missing

# Execute individual test modules
pytest tests/test_page_smoke.py -v           # Headless AppTest UI verification
pytest tests/test_security_boundaries.py -v   # Adversarial security checks
pytest tests/test_ml_and_drift.py -v          # Statistical tests, KS/FDR/TVD, AutoML
pytest tests/test_ingestion_parsers.py -v     # Streaming chunk parsers & limits
pytest tests/test_core_pipeline.py -v         # Caching, fingerprints, pipelines
pytest tests/test_audit_regressions.py -v     # Historical defect regressions
pytest tests/test_ai_conversion.py -v         # Chunked table extraction mocks
pytest tests/test_ui_workflows.py -v          # File uploads, PDF rendering
```

### 4.2 Standard Library Unittest Execution (Zero Dependency Fallback)

```bash
# Run entire suite via standard library unittest
python -m unittest discover -s tests -v

# Run individual test files
python -m unittest tests/test_page_smoke.py -v
python -m unittest tests/test_security_boundaries.py -v
```

---

## 5. Offline Isolation & Mock Boundaries

To guarantee deterministic local execution without internet access:
- **External AI Mock Boundary:** In `tests/test_ai_conversion.py` and `tests/test_audit_regressions.py`, calls to `google.genai.Client` are patched using `unittest.mock.patch`. Mock instances intercept `client.models.generate_content` and return structured responses, verifying error handling, chunk boundary reconciliation, and token budgeting without contacting Google servers.
- **Filesystem Isolation:** All test fixtures write to `tempfile.TemporaryDirectory()`, dynamically redirecting `Utils.paths.DATASETS_DIR`, `Utils.paths.MODELS_DIR`, and `Utils.paths.REPORTS_DIR` to prevent disk contamination.

---

## 6. Coverage Configuration (`pyproject.toml`)

Code coverage is configured via `[tool.coverage]`:

```toml
[tool.coverage.run]
source = ["Utils", "Pages", "App.py"]
omit = ["tests/*", "scratch/*"]

[tool.coverage.report]
show_missing = true
skip_covered = true
```

Run test suite with coverage report:
```bash
pytest tests/ --cov=Utils --cov=Pages --cov=App --cov-report=term-missing
# Or using standard coverage tool:
coverage run -m unittest discover -s tests
coverage report
```

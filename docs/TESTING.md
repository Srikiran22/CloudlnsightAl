# TabuLens — Testing & Quality Assurance Strategy

This document details the testing architecture, regression verification strategy, mock boundaries, and quality assurance gates implemented in TabuLens.

---

## 1. Testing Philosophy

1. **User-Visible Behavior Over Implementation Details:** Headless UI tests use Streamlit's official `AppTest` framework to simulate page lifecycles, clicks, selectbox mutations, and exception handling without running a physical browser.
2. **Deterministic & Isolated:** Tests use `tempfile.TemporaryDirectory` and monkeypatched storage roots (`paths.DATASETS_DIR`, `paths.MODELS_DIR`) to prevent side effects on working user datasets.
3. **Zero External Network Dependencies:** All tests run completely offline with no network calls or remote third-party services.
4. **Adversarial Security Verification:** Explicit test suites target path traversal, zip bombs, XML entity expansions (XXE), CSV formula injection, and unsigned model pickle deserialization.

---

## 2. Test Suite Organization

The test harness is located in `tests/` and comprises 12 dedicated test modules (384 passing tests):

```text
tests/
├── test_page_smoke.py                  # Headless AppTest boot checks across studio pages
├── test_security_boundaries.py         # XXE, DTD, zip bombs, formula injection, model signatures
├── test_core_pipeline.py               # Cryptographic fingerprints, cache LRU, pipeline preprocessing
├── test_ingestion_parsers.py           # CSV, TSV, JSON, JSONL, Parquet, XML streaming bounds
├── test_ml_and_drift.py                # TimeSeriesSplit, 5-fold CV, KS test FDR, TVD drift
├── test_ui_workflows.py                # File uploads, multi-file merging, PDF fonts, decimation
├── test_decision_threshold.py          # Decision threshold metrics and MCC optimization
├── test_outlier_diagnostics.py         # Tukey IQR fences, Z-scores, skewness & kurtosis
├── test_inference_script_generation.py # Standalone Python inference script generation
├── test_deployment_config.py           # Dockerfile 2-stage verification & GitHub Actions CI
├── test_portfolio_documentation.py     # Methodology & architecture documentation checks
└── test_verify_pipeline_cli.py         # Static verification runner and CLI exit code checks
```

---

## 3. Test Module Details

### 3.1 `test_page_smoke.py` (Headless Boot Checks)
- **Tooling:** `streamlit.testing.v1.AppTest`
- **Scope:** Bootstraps core studio workflows headless:
  - `app.py`, `pages/upload.py`, `pages/eda.py`, `pages/compare.py`, `pages/ml.py`, `pages/predict.py`.
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

### 3.4 `test_decision_threshold.py` (Threshold Optimization)
- Validates candidate threshold evaluation ($0.05 \le \tau \le 0.95$).
- Tests Matthews Correlation Coefficient (MCC), precision, recall, and F1 calculations from precomputed probabilities.

### 3.5 `test_outlier_diagnostics.py` (Distribution Diagnostics)
- Validates Tukey's IQR fences ($Q_1 - k \times \text{IQR}, Q_3 + k \times \text{IQR}$) across multipliers ($k=1.5, k=3.0$).
- Tests skewness, kurtosis, and distribution classification heuristics.

### 3.6 `test_inference_script_generation.py` (Standalone Code Export)
- Verifies that generated Python inference scripts are syntactically valid and executable independently.
- Confirms signature verification, model unpickling, and schema validation logic in the exported script.

---

## 4. Test Execution & CLI Commands

### 4.1 Pipeline Verification Runner (Recommended)

Run the full end-to-end verification and test harness:

```bash
# Run full verification (environment checks + full test suite)
python scripts/verify_pipeline.py

# Run static environment and cryptographic checks only
python scripts/verify_pipeline.py --check-only
```

### 4.2 Standard Library Unittest Execution

Execute tests using Python's standard `unittest`:

```bash
# Run entire discovery suite
python -m unittest discover -s tests -v

# Run individual test files
python -m unittest tests/test_security_boundaries.py -v
python -m unittest tests/test_ml_and_drift.py -v
python -m unittest tests/test_decision_threshold.py -v
```

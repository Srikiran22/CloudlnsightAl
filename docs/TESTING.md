# TabuLens — Testing & Quality Assurance

This document describes the test suite layout, verification boundaries, and CI checks used in TabuLens.

---

## 1. Testing Principles

1. **Isolated Temporary Storage:** Tests redirect `utils.paths.DATASETS_DIR` and pass temporary directories to `save_trained_model` / `load_trained_model` so user datasets and models on disk are not modified.
2. **Offline Execution:** Every unit, security, and `AppTest` check runs completely offline without network access.
3. **Headless UI Smoke & Regression Coverage:** Streamlit pages are tested with `streamlit.testing.v1.AppTest` (`tests/test_page_smoke.py`, `tests/test_audit_regressions.py`, `tests/test_ui_workflows.py`).
4. **Adversarial Input & Boundary Testing:** Explicit tests cover path traversal, Windows reserved filenames and ADS, XML DTD/XXE declarations, `.xlsx` ZIP bombs, CSV formula injection, memory/row/column limits, and tampered `.joblib` payloads.

---

## 2. Test Modules (`tests/`)

The `tests/` directory contains 19 `unittest` modules:

| Module | Scope |
|---|---|
| `tests/test_app_version.py` | Dynamic package version resolution (`importlib.metadata` and `pyproject.toml` fallback) |
| `tests/test_audit_regressions.py` | Binary confusion matrices, positive-class metrics, metadata preservation, PII/ID screening, CSV formula neutralization, overlap warnings, and HMAC cache keys |
| `tests/test_caching_and_performance.py` | `cached_load_dataframe` bounds and cache attributes |
| `tests/test_core_pipeline.py` | Problem-type detection, model round-trips, multi-format parsing (`JSONL`, `JSON`, `TSV`, `XML`, `Parquet`), `quality_metrics`, preprocessing, `schema_diff`, and PDF generation |
| `tests/test_decision_threshold.py` | Threshold override behavior in `predict_with_model` and `evaluate_threshold_metrics` (accuracy, precision, recall, MCC) |
| `tests/test_deployment_config.py` | Multi-stage `Dockerfile`, `.dockerignore`, `docker-compose.yml`, and `.github/workflows/ci.yml` checks |
| `tests/test_documentation.py` | Presence and structural checks for `README.md`, `docs/METHODOLOGY.md`, and `docs/decisions/` ADRs |
| `tests/test_inference_script_generation.py` | AST syntax validation, subprocess execution, tampered-model rejection, and invalid-threshold rejection for exported inference scripts |
| `tests/test_ingestion_parsers.py` | Delimited sniffing, JSON streaming, Parquet row bounding, HTML table extraction, and `.xlsx` ZIP bomb rejection |
| `tests/test_ml_and_drift.py` | `TimeSeriesSplit`, CV folds, baseline comparisons, KS + BH-FDR drift, categorical TVD, and upload collision naming |
| `tests/test_offline_integrity.py` | Ensures GenAI modules and dependencies (`google-genai`, `pypdf`) remain absent |
| `tests/test_outlier_diagnostics.py` | `compute_outlier_summary` Tukey IQR sensitivity, Z-score counts, skewness/kurtosis shape labels, and edge cases |
| `tests/test_package_structure.py` | Lowercase package/module naming (`app.py`, `utils/*.py`, `pages/*.py`) and core imports |
| `tests/test_page_smoke.py` | Headless `AppTest` boot checks across `app.py` and all 10 `pages/*.py` views plus seeded benchmark generation |
| `tests/test_security_boundaries.py` | Model signature enforcement, `__reduce__` payload blocking, path containment, XXE/DTD rejection, and resource ceilings |
| `tests/test_theme_and_boundaries.py` | Color tokens, error path redaction (`sanitize_error_message`, `safe_error_boundary`), Plotly styling, and SVG logo |
| `tests/test_ui_hardening.py` | `.streamlit/config.toml` error-detail suppression and path redaction |
| `tests/test_ui_workflows.py` | Multi-file merges, row counting, logging level fallback, PDF Unicode handling, and upload page rendering |
| `tests/test_verify_pipeline_cli.py` | Subprocess execution and exit code verification for `scripts/verify_pipeline.py --check-only` |

---

## 3. Running Tests & Checks

```bash
# Static environment, layout, and HMAC tamper checks
python scripts/verify_pipeline.py --check-only

# Full verification runner (static checks + unittest discovery)
python scripts/verify_pipeline.py

# Standard unittest discovery
python -m unittest discover -s tests -v

# Individual module example
python -m unittest tests/test_security_boundaries.py -v

# Lint check
ruff check .
```

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check .` and `coverage run -m unittest discover -s tests -v` on Python `3.11` and `3.12`.

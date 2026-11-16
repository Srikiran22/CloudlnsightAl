# Design Document: TabuLens — Ground-Up Student Capstone Refactoring

**Date:** 2026-11-16  
**Target:** 4th-Year Computer Science / Data Science Capstone Project & Technical Portfolio  
**Status:** Approved by Tri-Audit Swarm (Skeptical Architect, Hostile Design Critic, Technical Copy Editor)  
**Project Name:** **TabuLens** (Tabular Data Exploration, Drift Auditing & Leak-Free Modeling)  

---

## 1. Intent and Motivation

The goal is to restructure the repository into a clean, authentic, and professionally engineered 4th-year data science capstone project that is defensible under viva examination and stands out to technical recruiters.

Industry and academic reviews consistently discount projects that exhibit obvious AI-generated traits:
- Inflated corporate marketing claims (*"Stage 4 Production Maturity"*, *"Enterprise Telemetry Standards"*, *"Algorithmic Proofs"*).
- Inconsistent file naming and bloated boilerplate (mixed `PascalCase.py`, `ALLCAPS.py`, and `snake_case.py`).
- Fragile 950+ line CSS overrides of Streamlit internals.
- Half-baked LLM wrappers and porous in-process AST `exec()` code sandboxes.

**TabuLens** establishes a disciplined, 100% offline, local-first tabular machine learning studio demonstrating core data science and software engineering hygiene:
1. **Clean Architecture:** Standard PEP 8 lowercase packages (`utils/`, `pages/`, `app.py`), modular components, and zero orphaned or unmaintainable code.
2. **Defensible Applied ML:** Leak-free preprocessing pipelines via scikit-learn `Pipeline` and `ColumnTransformer`, rigorous benchmark comparison against naive baselines (`DummyClassifier` and `DummyRegressor`), and practical decision-threshold tuning for imbalanced targets.
3. **Statistical Distribution Drift:** Two-sample Kolmogorov-Smirnov hypothesis tests with Benjamini-Hochberg False Discovery Rate (FDR) control and Total Variation Distance.
4. **Standalone Portability:** 1-click Python inference code export so trained pipelines can be evaluated or served outside the GUI.
5. **Modern CI/CD & Automation:** Fast automated linting (`ruff`), 500+ unit tests, and a genuine multi-stage Dockerfile.

---

## 2. Scope Boundaries: Goals, Non-Goals, and Anti-Goals

### Goals:
- **Pythonic Normalization:** Rename all directories and files to standard PEP 8 `snake_case` (`utils/`, `pages/`, `app.py`). Update all internal import references across the application and test suite.
- **De-AI-ify Copy & Documentation:** Purge synthetic corporate claims, simulated auditor reports, and buzzwords from `README.md`, documentation, and UI labels. Write in direct, measured engineering prose.
- **Purge GenAI & Sandbox Dependencies:** Remove the external Gemini API integration, unstructured document OCR converter, and the AST code sandbox (`ai_sandbox.py`, `ai_agent.py`, `Gemini.py`, `AIConvert.py`). TabuLens becomes 100% offline, self-contained tabular data science.
- **Simplify Theme & CSS:** Replace the 950-line DOM hack in `theme.py` with a clean, lightweight utility (<100 lines) preserving backward-compatible token and styling helper signatures (`SEMANTIC_TOKENS`, `CHART_PALETTE`, `get_semantic_tokens`, `style_plotly_fig`, `safe_error_boundary`, `page_header`, `section_card_header`).
- **Core Capstone Feature 1 — Decision Threshold Optimization:** An interactive cutoff slider (0.05–0.95) for binary classification with dynamic confusion matrix and precision/recall recalculation, persisting the tuned threshold in the `.joblib` bundle and applying it in batch scoring.
- **Core Capstone Feature 2 — Standalone Python Inference Script Export:** Dynamic synthesis of an executable, copy-pasteable Python script tailored to the model's exact feature schema, handling column ordering, dtypes, and the tuned decision threshold.
- **Core Capstone Feature 3 — Outlier & Distribution Diagnostics:** An IQR (Tukey's fences) and Z-score outlier detection summary table in the EDA page.
- **Modern CI/CD Pipeline:** Refactor `.github/workflows/ci.yml` using the `ci-cd-and-automation` skill to execute linting (`ruff`), unit tests with coverage, and container build validation.
- **Small, Human Git Commits:** Enforce Conventional Commits (`feat:`, `fix:`, `refactor:`, `style:`, `test:`, `docs:`) with short, imperative summaries under 60 characters.

### Non-Goals:
- Deep learning frameworks (PyTorch/TensorFlow) or massive black-box models. Tabular tree ensembles and linear baselines are optimal for structured business datasets and ensure fast, reliable demo execution.
- Multi-tenant cloud databases or user authentication. This is an on-device, privacy-preserving analytical workbench.

### Anti-Goals:
- **Synthetic Corporate Artifacts:** No fake "Stage 4" maturity badges, no simulated corporate audit personas, no over-engineered security theater.
- **Silent Breakages:** Existing core tests for tabular parsing, data cleaning, feature engineering, modeling, and statistical drift must remain 100% green.

---

## 3. System Constraints

- **Python Runtime:** Python 3.12+ compatible.
- **Offline Reliability:** Ingestion, profiling, modeling, drift detection, and PDF generation operate 100% offline with zero internet access required.
- **Resource Ceilings:** Strict ingestion limits in `utils/limits.py` (200 MB file upload, 1M rows, 200 columns, 400 MB DataFrame RAM ceiling) prevent browser crashes and local host exhaustion.
- **Case-Sensitive Filesystem Invariance:** All path handling (`datasets/`, `models/`, `reports/`) is standardized in lowercase to prevent container failures on Linux.

---

## 4. Directory Structure & Namespace (PEP 8)

The project adopts a flat, idiomatic Streamlit layout at the repository root:
```
tabulens/
├── app.py                      # Main Streamlit workspace entrypoint (lowercase)
├── pages/                      # Multi-page views (lowercase snake_case)
│   ├── upload.py               # Tabular data ingestion (CSV, TSV, Parquet, Excel, SQLite)
│   ├── dashboard.py            # Dataset overview & health metrics
│   ├── cleaning.py             # Missing values, deduplication, formula sanitization
│   ├── feature_engineering.py  # Encoding, scaling & transformations
│   ├── eda.py                  # Distributions, correlations & outlier diagnostics
│   ├── visualization.py        # Interactive Plotly chart builder
│   ├── ml.py                   # Leak-free training, CV & decision threshold tuning
│   ├── predict.py              # Batch scoring & standalone script export
│   ├── compare.py              # Kolmogorov-Smirnov drift detection (BH-FDR Q=0.05)
│   ├── report.py               # PDF executive audit report generator
│   └── __init__.py
├── utils/                      # Reusable backend modules (lowercase snake_case)
│   ├── charts.py               # Plotly figure builders & styling
│   ├── compare_logic.py        # Two-sample KS test & Benjamini-Hochberg FDR
│   ├── dataset_ui.py           # Session state & active dataset management
│   ├── eda_logic.py            # Normality tests, correlation & outlier bounds (IQR/Z)
│   ├── feature_engineering.py  # Feature encoders and transforms
│   ├── limits.py               # Memory & file size boundaries
│   ├── logsys.py               # Standardized logging
│   ├── ml.py                   # Pipelines, cross-validation, thresholding & code export
│   ├── parsing.py              # Fast, bounded tabular readers (CSV/Parquet/Excel/SQLite)
│   ├── paths.py                # Deterministic path resolution & atomic writes
│   ├── pdf.py                  # Headless ReportLab PDF generator
│   ├── preprocessing.py        # Leak-free transformer helpers
│   ├── privacy.py              # Identifier & PII heuristic detection
│   ├── quality.py              # Missingness & completeness profiling
│   ├── sampling.py             # Deterministic uniform downsampling for UI
│   ├── theme.py                # Lightweight styling & error boundary (<100 lines)
│   └── __init__.py
├── tests/                      # Pytest / unittest discovery test suite
├── scripts/
│   ├── verify_pipeline.py      # Pre-commit & CI verification runner
│   └── healthcheck.py          # Docker healthcheck probe
├── .github/workflows/
│   └── ci.yml                  # Ruff lint + unit tests + Docker build
├── Dockerfile                  # Genuine 2-stage multi-stage build
├── docker-compose.yml
├── pyproject.toml              # Distribution name: tabulens
└── README.md                   # Human-crafted, clear, impressive README
```

---

## 5. Detailed Component Specifications

### 5.1 De-bloating Theme & CSS (`utils/theme.py`)
Replace the 950+ lines of DOM selector overrides with a minimal, robust helper module (<100 lines):
- **Core Helpers:**
  - `page_header(title: str, subtitle: str, badge: str | None = None)`: Clean markdown header with subtle badge styling.
  - `section_card_header(title: str, description: str)`: Clean section title.
  - `safe_error_boundary(title="An error occurred", handler=None)`: Context manager catching unexpected errors, logging details, and displaying sanitized error messages.
  - `plot_template()`: Returns `'plotly_dark'` or `'plotly_white'` based on base theme.
- **Backward-Compatible Shims (preserving existing test assertions):**
  - `SEMANTIC_TOKENS`: Dict mapping tokens (`primary`, `success`, `warning`, `danger`, `surface`, `border`, `text`).
  - `CHART_PALETTE`: Clean categorical color list (`["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]`).
  - `get_semantic_tokens(dark=True)`: Helper returning color tokens.
  - `style_plotly_fig(fig)`: Standardizes layout margins, legend, and background.
  - `sanitize_error_message(msg)`: Strips internal local file paths.

### 5.2 Feature 1: Classification Decision Threshold Optimization (`pages/ml.py`, `utils/ml.py`)
- **UI Interaction:** In `pages/ml.py`, for binary classification models (`len(classes) == 2`):
  - Render an interactive slider: `threshold = st.slider("Decision Cutoff Threshold", 0.05, 0.95, float(results.get("decision_threshold", 0.50)), 0.05)`.
  - Cache holdout prediction probabilities (`y_test_proba`) in `results` during training to compute predictions in $O(1)$:
    $$\hat{y} = \text{classes\_}[1] \quad \text{if} \quad P(y = \text{positive\_class}) \ge \theta \quad \text{else} \quad \text{classes\_}[0]$$
  - Dynamically update the Confusion Matrix, Precision, Recall, and F1 metrics for the selected threshold.
  - Display an operational callout: *"Adjust the threshold when false negatives (e.g. missed churn/fraud) carry higher business costs than false positives."*
- **Operational Persistence:**
  - `st.session_state["ml_results"]["decision_threshold"] = threshold`.
  - When the user saves the model (or updates the slider for an active model), the saved `bundle["decision_threshold"]` is updated atomically.
  - `predict_with_model(bundle, df, feature_cols=None, threshold_override=None)` respects `threshold_override` or `bundle.get("decision_threshold", 0.50)` for binary classification.
  - In `pages/predict.py`, the active threshold is displayed as a metric and can be adjusted prior to running batch predictions.

### 5.3 Feature 2: Standalone Python Inference Script Export (`utils/ml.py`, `pages/predict.py`)
- Function `generate_standalone_inference_code(bundle: dict, sample_df: pd.DataFrame | None = None) -> str` in `utils/ml.py`:
  - Inspects `bundle["feature_cols"]`, `bundle["target_column"]`, `bundle["problem_type"]`, and `bundle.get("decision_threshold", 0.50)`.
  - Generates safe, syntax-valid Python code using `json.dumps()` for sample data (preventing string injection vulnerabilities).
  - Handles column subsetting, dtype coercion, and thresholding cleanly:
    ```python
    import joblib
    import pandas as pd
    import json

    # 1. Load trained pipeline bundle
    bundle = joblib.load("models/saas_churn_model.joblib")
    pipeline = bundle["pipeline"]
    features = bundle["feature_cols"]

    # 2. Prepare sample input data
    sample_records = json.loads("""[...]""")
    df = pd.DataFrame(sample_records)[features]

    # 3. Execute inference with configured decision threshold
    if hasattr(pipeline, "predict_proba") and len(bundle.get("classes", [])) == 2:
        proba = pipeline.predict_proba(df)[:, 1]
        threshold = bundle.get("decision_threshold", 0.50)
        predictions = [bundle["classes"][1] if p >= threshold else bundle["classes"][0] for p in proba]
    else:
        predictions = pipeline.predict(df)

    print("Predictions:", predictions)
    ```
- Rendered in a clean code block with a one-click copy button in `pages/predict.py` and `pages/ml.py`.

### 5.4 Feature 3: Outlier & Distribution Diagnostics (`pages/eda.py`, `utils/eda_logic.py`)
- In `utils/eda_logic.py`, implement `compute_outlier_summary(df: pd.DataFrame) -> pd.DataFrame`:
  - For each numerical column:
    - Calculates Q1, Q3, and $\text{IQR} = Q3 - Q1$.
    - Tukey's fences: $[\text{Lower Fence} = Q1 - 1.5 \times \text{IQR}, \text{Upper Fence} = Q3 + 1.5 \times \text{IQR}]$.
    - Computes count and percentage of IQR outliers.
    - Z-score outliers: count of $|z| > 3$ where $z = (x - \mu) / \sigma$.
  - Returns a structured summary DataFrame.
- In `pages/eda.py`, render an **"Outlier & Distribution Diagnostics"** tab presenting this table alongside box plots for selected columns.

### 5.5 Artifact Verification & Deserialization Security
- Replace the symmetric HMAC key stored next to models (`model_security.py`) with clean SHA-256 integrity digests:
  - When saving a model: compute `sha256 = hashlib.sha256(joblib_bytes).hexdigest()` and record it in `bundle["integrity_sha256"]` and a companion `.sha256` manifest.
  - When loading: verify file digest against the manifest to protect against accidental file corruption or partial writes.

### 5.6 Multi-Stage Dockerfile & Container Hardening
- Implement a genuine 2-stage `Dockerfile`:
  - **Stage 1 (Builder):** Uses `python:3.12-slim`, installs `build-essential`, builds wheels for dependencies in `/wheels`.
  - **Stage 2 (Runner):** Uses clean `python:3.12-slim`, copies `/wheels`, installs without build tools, creates non-root `appuser` (UID 1000).
  - Configures `HEALTHCHECK CMD python scripts/healthcheck.py || exit 1`.
- Update `.dockerignore` to exclude `datasets/`, `models/`, `reports/`, `.git/`, `.venv/`, `__pycache__/`.

### 5.7 CI/CD & Automation (`.github/workflows/ci.yml`)
- Single clean GitHub Actions workflow:
  - Triggers on PR and push to `main`.
  - Jobs:
    1. **Code Hygiene & Linting:** `ruff check .` and `ruff format --check .`.
    2. **Unit Tests & Coverage:** `coverage run -m unittest discover -s tests` with coverage report.
    3. **Container Build Verification:** `docker build -t tabulens:test .`.

---

## 6. Execution Roadmap for Implementation

1. **Phase 1: File Structure & Namespace Normalization**
   - Rename `Utils/` to `utils/` and `Pages/` to `pages/`, `App.py` to `app.py`.
   - Rename modules to lowercase `snake_case`.
   - Update all import statements across `app.py`, `pages/`, `utils/`, and `tests/`.
2. **Phase 2: Purge GenAI, AST Sandbox & AI Slop**
   - Delete `Pages/AI.py`, `Pages/AIChat.py`, `Pages/AIConverter.py`, `Utils/Gemini.py`, `Utils/ai_agent.py`, `Utils/ai_sandbox.py`, `Utils/AIConvert.py`, `Utils/markup_parsing.py`, `Utils/secrets.py`.
   - Remove obsolete synthetic audit docs (`docs/PORTFOLIO_MEMO.md`, `AUDIT_REPORT.md`, `PROJECT_WALKTHROUGH.md`).
3. **Phase 3: Theme De-bloating & Error Boundary**
   - Replace `utils/theme.py` with the minimal (<100 lines) implementation retaining test-compatible shims.
4. **Phase 4: Capstone Features Implementation**
   - Feature 1: Decision threshold slider, $O(1)$ probability cache, and operational persistence in `utils/ml.py` and `pages/ml.py`.
   - Feature 2: Standalone Python script generator `generate_standalone_inference_code` in `utils/ml.py` and `pages/predict.py`.
   - Feature 3: Outlier summary function in `utils/eda_logic.py` and UI tab in `pages/eda.py`.
5. **Phase 5: Docker & CI/CD Pipeline**
   - True 2-stage `Dockerfile`, updated `.dockerignore`, and `.github/workflows/ci.yml`.
6. **Phase 6: Professional Documentation & Verification**
   - Rewrite `README.md` with honest, high-impact project overview, architecture diagram, and baseline benchmarks.
   - Run verification script and full test suite to guarantee 100% green tests.

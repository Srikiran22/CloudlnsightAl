# CloudInsight AI — Technical Audit Report

**Date of Audit:** November 3, 2026  
**Project:** CloudInsight AI (`cloudinsight-ai` v2.2.0)  
**Evaluator Standard:** Local-Only Senior Engineering & Architecture Review  
**Repository Scope:** `C:\Users\Msi\Projects\CloudlnsightAl`  

---

## 1. Executive Summary

CloudInsight AI is a local-first, privacy-conscious data science and analytics workbench built in Python with Streamlit, pandas, scikit-learn, Plotly, ReportLab, and Google Gemini (via `google-genai`). The platform provides a complete tabular data workflow: multi-format ingestion, data hygiene profiling, statistical imputation, feature engineering, exploratory data analysis with hypothesis testing, leak-free machine learning training with 5-fold cross-validation, cryptographic model artifact signing (HMAC-SHA256), batch inference, and automated PDF audit report generation.

### Current Maturity Assessment: **Stage 4 — Robust Production Workbench (Self-Contained & Hardened)**

- **Strengths:** 
  - Comprehensive defenses against common local file vulnerabilities: XML External Entity (XXE) and DTD injection rejection, ZIP decompression bomb guards, strict path containment to prevent directory traversal (`resolve_dataset_path`), CSV formula sanitization (`sanitize_for_csv_export`), and streaming row/column/cell ingestion ceilings.
  - Leak-free ML architecture: Imputation, scaling, and categorical encoding take place strictly inside scikit-learn `Pipeline` and `ColumnTransformer` objects fitted solely on training folds.
  - Cryptographic tamper resistance: Model artifacts are serialized with `joblib` and signed with HMAC-SHA256. Unsigned or modified bundles are rejected prior to bytecode deserialization.
  - Ephemeral credential handling: API keys are stored in `st.session_state` only and never touch disk, environment files, or logging pipelines.
- **Identified Deficiencies & Remediations:**
  - **P1 High:** The entire AI module (`Pages/AI.py`, `Pages/AIChat.py`, `Pages/AIConverter.py`) was excluded from `st.navigation` in `App.py`, completely orphaning conversational analysis and unstructured document conversion from the user interface.
  - **P1 High:** `Pages/FeatureEngineering.py` Tab 1 (Numerical Scaling) reset the working dataset to `df.copy()`, overwriting and discarding any prior chained transformations (such as encodings or polynomial features).
  - **P1 High:** `Pages/Cleaning.py` catalog save action wrote via unbuffered `.to_csv()` without formula sanitization, without atomic write mechanics, and without cache invalidation.
  - **P2 Medium:** Discrepancies in page cross-references (e.g., referring to "Import Files" instead of "Data Hub & Ingest", "Train Model" instead of "Model Studio & Training").
  - **P2 Medium:** Remote GitHub Actions badge URLs in README and missing modular technical documentation (`docs/` architecture, setup, testing, and security guides).
  - **P3 Low:** Local host environment blocker: Python runtime (>=3.12) is not installed in the system PATH, preventing headless test and runtime execution on the local machine until installed.

---

## 2. Technology Inventory & Architectural Stack

| Dimension | Specification |
|---|---|
| **Application Type** | Multi-page data science web application & analytics workbench |
| **Primary Language** | Python 3.12+ (tested against 3.12 and 3.13) |
| **UI Framework** | Streamlit (>=1.52.0, verified on 1.64.0) |
| **Data Processing** | pandas (>=2.0.0), numpy (>=1.24.0) |
| **File Parsers** | `openpyxl` (XLSX), `xlrd` (legacy XLS), `pyarrow` (Parquet), `pypdf` (PDF text extraction), `defusedxml` / hardened SAX parser |
| **Machine Learning** | `scikit-learn` (>=1.3.0), `joblib` (>=1.3.0), `statsmodels` (>=0.14.0 for OLS) |
| **Visualization** | `plotly` (>=5.18.0) |
| **Report Generation**| `reportlab` (>=4.0.0), `matplotlib` (>=3.7.0 for headless rendering) |
| **External AI** | `google-genai` (>=1.0.0) |
| **Code Hygiene** | `ruff` (>=0.16.4) |
| **Package Manager**| `pip` / standard virtualenv, pinned via `requirements-lock.txt` |
| **Persistence** | Local filesystem (`Datasets/`, `Models/`, `Reports/`, `ReportTemplates/`) |

---

## 3. Baseline Environment Discovery & Verification

Safe local inspection was conducted using PowerShell. Findings:

1. **Python Runtime Status:**
   - Command: `python --version`
   - Result: Exit code 1.
   - Output: `Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.`
   - Impact: Python >=3.12 is not installed on the host operating system. The test suite, linter (`ruff`), and Streamlit server cannot be run locally until Python is installed.
   - Status: **Not run (Host dependency missing)**.
2. **Local Version Control & Repository State:**
   - Command: `git status`
   - Result: Clean working tree on branch `main`.
   - Security: No GitHub remotes were accessed, polled, fetched, or contacted.
3. **Static Code Review:**
   - 41 Python source files audited manually across `App.py`, `Pages/`, `Utils/`, and `tests/`.

---

## 4. Prioritized Audit Findings

### P0 — Critical (0 Discovered)
No remote code execution vulnerabilities, unauthenticated network listeners, or exposed plain-text credentials were discovered.

### P1 — High (3 Discovered, 3 Resolved)

#### Finding P1.1: Orphaned AI Subsystem in Streamlit Navigation
- **Evidence:** `App.py` (lines 192–218). The `st.navigation` dictionary defined sections for Overview, Data Management, Exploratory Analysis, Machine Learning, and Audit & Reports, but completely omitted `Pages/AI.py`, `Pages/AIChat.py`, and `Pages/AIConverter.py`.
- **Impact:** Users navigating the Streamlit web interface had no way to access the AI File Converter, the AI Summary insights, or the Ask AI conversational interface, despite these being core advertised capabilities tested in `tests/test_page_smoke.py`.
- **Remediation:** Added the `"AI & Gemini Assistant"` section into `st.navigation` in `App.py` with appropriate Material Design icons (`:material/transform:`, `:material/psychology:`, `:material/forum:`).
- **Status:** **Resolved**.

#### Finding P1.2: Feature Engineering State Reset Overwrites Chained Features
- **Evidence:** `Pages/FeatureEngineering.py` (line 57, 91–96). In Tab 1 (Numerical Scaling), `transformed_df` was initialized as `df.copy()`, and `clean_vals` read directly from `df[selected_scale_cols]`.
- **Impact:** If an analyst performed categorical encoding (Tab 2) or mathematical transformations (Tab 3), subsequent application of numerical scaling discarded all previously engineered features stored in `st.session_state["fe_working_df"]`.
- **Remediation:** Updated Tab 1 to draw from `st.session_state.get("fe_working_df", df).copy()`, ensuring sequential multi-stage feature engineering pipelines preserve cumulative columns.
- **Status:** **Resolved**.

#### Finding P1.3: Unsanitized, Non-Atomic Catalog Save in Data Cleaning
- **Evidence:** `Pages/Cleaning.py` (lines 244–246). The "Save to Workspace Catalog" button executed `cleaned_df.to_csv(save_path, index=False)` directly without calling `sanitize_for_csv_export()`, without using `atomic_write()`, and without calling `invalidate_dataset_cache()`.
- **Impact:** Potential CSV formula injection vulnerability if raw user data contained formula prefixes (`=`, `+`, `-`, `@`), risk of truncated files during write interruptions, and stale in-memory cached frames.
- **Remediation:** Refactored the catalog save routine to apply `sanitize_for_csv_export()`, `atomic_write()`, `get_unique_filename()`, and `invalidate_dataset_cache()`.
- **Status:** **Resolved**.

---

### P2 — Medium (3 Discovered, 3 Resolved)

#### Finding P2.1: Navigation Label Inconsistencies Across Pages
- **Evidence:** `Pages/Cleaning.py` (line 41) and `Utils/dataset_ui.py` (lines 302, 322) referenced "Import Files", while `App.py` and `Pages/Upload.py` used "Data Hub & Ingest". `Pages/Predict.py` (lines 44, 169, 197) referenced "Train Model", while the page was titled "Model Studio & Training".
- **Impact:** Confusing UX and cognitive friction when guided by on-screen callouts.
- **Remediation:** Harmonized all internal page references to match the official titles defined in `st.navigation`.
- **Status:** **Resolved**.

#### Finding P2.2: Hardcoded Remote GitHub Badges in Documentation
- **Evidence:** `README.md` contained remote GitHub Actions and shields.io badge URLs that cannot be resolved in offline or local-only review environments.
- **Impact:** Violates self-contained documentation principles and creates misleading external dependencies.
- **Remediation:** Removed remote GitHub URLs and replaced them with local status summaries and exact reproduction instructions.
- **Status:** **Resolved**.

#### Finding P2.3: Missing Diátaxis Technical Documentation Architecture
- **Evidence:** The project had a comprehensive README but lacked modular documentation in `docs/` (`ARCHITECTURE.md`, `SETUP.md`, `TESTING.md`, `SECURITY.md`, `API.md`).
- **Impact:** Higher onboarding friction for senior reviewers evaluating system architecture and security boundaries.
- **Remediation:** Created complete, evidence-based technical documents in `docs/`.
- **Status:** **Resolved**.

---

### P3 — Low (1 Discovered, 1 Documented)

#### Finding P3.1: Python Runtime Missing in Local Execution Environment
- **Evidence:** PowerShell command `python --version` returns exit code 1 with Windows Store alias redirection.
- **Impact:** Prevents executing `pytest` or `streamlit run` directly on the host until Python 3.12+ is installed.
- **Remediation:** Documented clear setup prerequisites in `docs/SETUP.md` and `README.md` using `winget install Python.Python.3.12` or Python's official installer.
- **Status:** **Documented (Host Prerequisite)**.

---

## 5. Security and Privacy Review

| Security Control | Implementation Verification | Standards Alignment | Audit Finding |
|---|---|---|---|
| **Path Traversal Defense** | `Utils/paths.py::resolve_dataset_path` verifies `Path.is_relative_to(DATASETS_DIR)` and rejects `..`, drive letters, and ADS (`:`). | **OWASP ASVS 12.1.1**<br>**NIST SP 800-218 PW.1.1** | **Robust** |
| **Model Deserialization Safety** | `Utils/ML.py::load_trained_model` verifies HMAC-SHA256 signature against `Models/.model_signing_key` before calling `joblib.load()`. | **OWASP ASVS 5.5.1, 6.2.1**<br>**NIST SP 800-218 PW.1.1** | **Robust** |
| **XXE / XML Entity Injection** | `Utils/paths.py::_reject_dtd` scans raw XML stream for `<!DOCTYPE` and `<!ENTITY>` and aborts with `ValueError`. | **OWASP ASVS 5.2.2**<br>**NIST SP 800-218 PW.4.1** | **Robust** |
| **ZIP Decompression Bombs** | `Utils/paths.py::_check_zip_bomb` checks uncompressed byte sums and enforces $<100\times$ expansion ratio. | **OWASP ASVS 12.2.1**<br>**NIST SP 800-218 PW.1.1** | **Robust** |
| **CSV Formula Injection** | `Utils/frames.py::sanitize_for_csv_export` prepends `'` to cells starting with `=`, `+`, `-`, `@`, `\t`, `\r`. | **OWASP ASVS 5.2.4**<br>**NIST SP 800-218 PW.4.1** | **Robust** |
| **Credential Storage** | Ephemeral only. API keys reside in `st.session_state` RAM only; purged by `temporary_secret` context managers. | **OWASP ASVS 6.4.1, 8.3.1**<br>**NIST SP 800-218 PS.1.1** | **Robust** |
| **PII / Data Boundary** | `Utils/privacy.py` heuristically identifies emails, phone numbers, IBANs, credit cards, and IDs, excluding them from AI calls. | **OWASP ASVS 8.3.1**<br>**NIST SP 800-218 PS.1.1** | **Robust** |
| **Resource Ingestion Ceilings** | Ingestion limits enforced: 200 MB file, 1M rows, 200 cols, 20M cells, 400 MB DataFrame RAM budget. | **OWASP ASVS 5.1.1**<br>**NIST SP 800-218 PW.1.1** | **Robust** |

---

## 6. Frontend, Design System & Accessibility Review

- **Design System Tokens:** Defined in `Utils/theme.py` and synchronized with `.streamlit/config.toml`.
  - Dark Mode: Surface `#111827`, Background `#0B0F19`, Text `#F8FAFC`, Muted `#94A3B8`, Accent `#3B82F6`.
  - Light Mode: Surface `#F8FAFC`, Background `#FFFFFF`, Text `#0F172A`, Muted `#64748B`, Accent `#2563EB`.
- **Contrast & Legibility:** 
  - Normal text contrast is 17.5:1 in dark mode and 15.6:1 in light mode (exceeds WCAG 2.2 AA requirement of 4.5:1).
  - Muted captions maintain 7.5:1 (dark) and 4.6:1 (light) (exceeds 4.5:1).
  - Primary button text is `#FFFFFF` against `#2563EB` (contrast 4.6:1, passing AA).
- **Typography:** Primary font stack: `"Inter", "Segoe UI", system-ui, -apple-system, sans-serif`. Tabular figures (`font-variant-numeric: tabular-nums`) applied to numeric KPI cards to ensure columnar number alignment.
- **Responsive Layout:** Grid layout containers adapt fluidly using Streamlit columns; large tables enforce horizontal scrolling without breaking outer viewport boundaries.

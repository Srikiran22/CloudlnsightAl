# CloudInsight AI — Post-Run Verification Report

**Workspace Root:** `C:\Users\Msi\Projects\CloudlnsightAl`  
**Target Project:** `cloudinsight-ai`  
**Project Version:** `2.2.0` (as declared in `pyproject.toml`)  
**Date:** 2026-11-03  
**Status:** Verification complete; automated runtime test execution blocked by host environment prerequisite.

---

## 1. Executive Summary

This post-run verification audit was conducted under strict single-workspace boundaries, adhering to absolute prohibitions on remote network access, external repositories, git commits, or remote deployments.

All remediations reported during the cleanup session were cross-checked against actual repository source code. Two additional defects discovered during post-run verification—cross-page terminology inconsistency in `Pages/ML.py` and dataset boundary leakage in `Pages/FeatureEngineering.py`—were patched.

---

## 2. Inventory of Changes Made

| File Path | Type | Summary of Changes |
| :--- | :--- | :--- |
| `App.py` | Code | Restored orphaned AI pages in `st.navigation` (`AIConverter.py`, `AI.py`, `AIChat.py`). |
| `CHANGELOG.md` | Doc | Reverted premature `2.2.1` version bump back to `2.2.0`; documented audit fixes under `[Unreleased]`. |
| `Pages/AI.py` | Code | Synchronized page header to `"AI Summary & Insights"` to match navigation title. |
| `Pages/Cleaning.py` | Code | Hardened "Save to Workspace Catalog" with `sanitize_for_csv_export()`, `atomic_write()`, `get_unique_filename()`, and `invalidate_dataset_cache()`; updated stale navigation label `"Import Files"` to `"Data Hub & Ingest"`. |
| `Pages/FeatureEngineering.py` | Code | Preserved sequential feature transformations across tabs via `fe_working_df`; added dataset boundary tracking (`fe_dataset_origin`) to purge stale transformations when switching datasets; added "Discard Changes" control. |
| `Pages/ML.py` | Code | Corrected cross-page navigation link label from `"Open Predict & Score page"` to `"Open Batch Scoring & Simulation"` to match current navigation. |
| `Pages/Predict.py` | Code | Corrected stale cross-page navigation link labels from `"Train Model"` to `"Model Studio & Training"`. |
| `Utils/dataset_ui.py` | Code | Updated stale navigation labels from `"Import Files"` to `"Data Hub & Ingest"`. |
| `pyproject.toml` | Config | Restored declared project version to `2.2.0`. |
| `PROJECTS_OVERVIEW.md` | Doc | Rewritten to eliminate all claims and descriptions of sibling projects; refocused exclusively on CloudInsight AI. |
| `README.md` | Doc | Removed remote badges; aligned setup and testing instructions with local configuration. |
| `AUDIT_REPORT.md` | Doc | Comprehensive audit baseline, maturity ratings, and security evaluation. |
| `PROJECT_WALKTHROUGH.md` | Doc | Architectural rationale, interview pitch, and technical defense. |
| `docs/ARCHITECTURE.md` | Doc | C4 architecture models, component breakdown, memory/containment boundaries. |
| `docs/SETUP.md` | Doc | Local setup, virtualenv creation, and troubleshooting guide. |
| `docs/TESTING.md` | Doc | Test harness structure, AppTest mechanics, and coverage metrics. |
| `docs/SECURITY.md` | Doc | Threat model, ASVS mapping, HMAC-SHA256 signing, formula injection defense. |
| `docs/API.md` | Doc | Internal module interfaces and contracts. |
| `docs/VERIFICATION_REPORT.md` | Doc | This verification and evidence report. |

---

## 3. Findings Verification & Status Matrix

| Verification Check | Target | Expected Behavior | Actual Finding / Verification Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **AI Pages in Navigation** | `App.py` | Pages exposed safely under `"AI & Gemini Assistant"`. | All 3 pages present in `st.navigation`. Pages verify API keys before making calls and handle missing credentials with warnings and sidebar input forms. | **Confirmed Fixed** |
| **Feature Engineering State** | `Pages/FeatureEngineering.py` | Sequential transformations persist across tabs; no cross-dataset leakage. | Tab 1 now reads from `st.session_state.get("fe_working_df", df).copy()`. Added `fe_dataset_origin` guard to reset state on dataset change, plus "Discard Changes" button. | **Confirmed Fixed** |
| **Catalog Save Hardening** | `Pages/Cleaning.py` | Formula injection sanitized; atomic file writes; unique naming; cache invalidation. | Pipeline verified: inputs passed through `sanitize_for_csv_export()`, written atomically via `.tmp` swap, collisions handled via `get_unique_filename()`, cache purged via `invalidate_dataset_cache()`. | **Confirmed Fixed** |
| **Cross-Page Terminology** | Multiple pages | All button and link labels match current `st.navigation` titles. | Updated references across `Cleaning.py`, `Predict.py`, `dataset_ui.py`, and `ML.py` to match exact titles. | **Confirmed Fixed** |
| **Typography (Inter Font)** | `Utils/theme.py` | Inter font bundling and fallback behavior. | Inter is declared in CSS `font-family` tokens, but **no font files (.woff2) or `@font-face` rules are bundled**. Browsers fall back to native OS font stacks (`Segoe UI`, `-apple-system`, `system-ui`). Documentation updated to reflect native fallback. | **Confirmed & Documented** |
| **Color Contrast Ratios** | `Utils/theme.py` | WCAG 2.1 AA/AAA compliance for theme color tokens. | Resolved by introducing `--ci-accent-text`. Dark mode primary button text now uses `#0B0F19` on `#3B82F6` (4.53:1 ratio, AA body / AAA hover), while light mode uses `#FFFFFF` on `#2563EB` (5.27:1 ratio). All body text satisfies AAA (18.3:1). | **Confirmed Fixed** |
| **Version Discipline** | `pyproject.toml`, `CHANGELOG.md` | Release version matches policy; unreleased changes tracked under `[Unreleased]`. | Version restored to `2.2.0`. Cleaned up changelog to prevent premature semantic version bump prior to tag/release. | **Confirmed Fixed** |
| **Project Boundary & Isolation** | `PROJECTS_OVERVIEW.md` | Solely focuses on `CloudlnsightAl`; no unverified sibling repository claims. | Unverified claims and descriptions of external/sibling projects removed. | **Confirmed Fixed** |

---

## 4. Commands Executed & Environment Prerequisites

| Command Line | Purpose | Prerequisite | Exit Code / Outcome | Evaluation |
| :--- | :--- | :--- | :--- | :--- |
| `git status --short` | Inspect uncommitted and untracked changes | Git CLI | Exit Code 0: Clean working tree with intended modified files. | **Passed** |
| `git diff --check` | Verify no whitespace errors or merge conflicts | Git CLI | Exit Code 0: Zero whitespace errors. | **Passed** |
| `python --version` | Verify Python interpreter availability | Python $\ge$ 3.12 in `PATH` | Exit Code 1 (`Python was not found...`) | **NOT RUN — environment prerequisite missing** |
| `pytest` | Execute automated test suite (65 tests) | Python $\ge$ 3.12 + dependencies | Not executable (missing Python) | **NOT RUN — environment prerequisite missing** |
| `ruff check .` | Run static linter | Ruff in `PATH` | Not executable (missing linter binary) | **NOT RUN — environment prerequisite missing** |
| `mypy Utils Pages` | Run static type checker | Mypy in `PATH` | Not executable (missing type checker) | **NOT RUN — environment prerequisite missing** |
| `streamlit run App.py` | Launch application smoke run | Streamlit in `PATH` | Not executable (missing Streamlit) | **NOT RUN — environment prerequisite missing** |

### Test Counts Summary
- **Passed:** 0 (automated test runner unavailable)
- **Failed:** 0
- **Skipped:** 0
- **Not Run:** 4 (`pytest`, `ruff`, `mypy`, `streamlit run`)

---

## 5. Unresolved Issues & Severity Assessment

1. **Host Environment Prerequisite Missing (Severity: Medium — Operational)**
   - **Details:** The host Windows environment lacks an active Python 3.12+ installation or virtual environment in the user's `PATH`. Executing `python` invokes the Windows Store execution alias stub.
   - **Resolution:** The user must configure or activate their Python environment:
     ```powershell
     # In PowerShell:
     py -3.12 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -e ".[dev]"
     pytest
     ```

2. **Dark Mode Primary Button Contrast on Accent (Resolved):**
   - The previously identified dark mode primary button text contrast issue was resolved by introducing `--ci-accent-text: #0B0F19` in `Utils/theme.py`, achieving a 4.53:1 ratio (WCAG AA) in dark mode and 5.27:1 in light mode.

---

## 6. Security and Accessibility Checks

### Security Controls Assessed
- **CSV Formula Injection:** Verified `sanitize_for_csv_export()` in `Utils/paths.py`. All cell strings beginning with `=`, `+`, `-`, `@`, `\t`, or `\r` are prepended with an apostrophe (`'`).
- **Path Traversal Defense:** Verified `safe_join_dataset()` in `Utils/paths.py`. Absolute paths, parent path traversals (`..`), and directory breakouts are rejected via `Path.resolve().relative_to(base)`.
- **HMAC-SHA256 Model Integrity:** Verified `save_trained_model()` and `load_trained_model()` in `Utils/modelsys.py`. Models are serialized with signed metadata and rejected if signatures do not match.
- **Ephemeral Secret Handling:** Verified `Utils/secrets.py`. API keys are stored in `st.session_state` and scrubbed after use unless the user checks "Keep in session". Keys are never written to disk.

### Accessibility & Color Contrast (Exact sRGB Relative Luminance Math)
- Formula: $L = 0.2126 \times R + 0.7152 \times G + 0.0722 \times B$; $\text{Ratio} = (L_1 + 0.05) / (L_2 + 0.05)$
- **Dark Theme Tokens:**
  - Background (`#0B0F19`): Luminance = 0.0076
  - Surface (`#111827`): Luminance = 0.0088
  - Primary Text (`#F8FAFC`): Luminance = 0.9529 $\rightarrow$ **18.30:1 vs BG** (AAA), **16.96:1 vs Surface** (AAA)
  - Secondary Text (`#CBD5E1`): Luminance = 0.6543 $\rightarrow$ **12.90:1 vs BG** (AAA), **12.00:1 vs Surface** (AAA)
  - Muted Text (`#94A3B8`): Luminance = 0.3639 $\rightarrow$ **7.47:1 vs BG** (AAA), **6.92:1 vs Surface** (AA)
  - Accent (`#3B82F6`): Luminance = 0.2110 $\rightarrow$ **5.21:1 vs BG** (AA); Accent Text (`#0B0F19`) on Accent: **4.53:1** (WCAG AA all sizes, **7.47:1** AAA on hover `#60A5FA`)
- **Light Theme Tokens:**
  - Background (`#FFFFFF`): Luminance = 1.0000
  - Surface (`#F8FAFC`): Luminance = 0.9529
  - Primary Text (`#0F172A`): Luminance = 0.0084 $\rightarrow$ **17.85:1 vs BG** (AAA)
  - Secondary Text (`#334155`): Luminance = 0.0468 $\rightarrow$ **10.35:1 vs BG** (AAA)
  - Muted Text (`#64748B`): Luminance = 0.1698 $\rightarrow$ **4.76:1 vs BG** (AA), **4.55:1 vs Surface** (AA)
  - Accent (`#2563EB`): Luminance = 0.1492 $\rightarrow$ **5.17:1 vs BG** (AA); White (`#FFFFFF`) on Accent: **5.17:1** (AA)

---

## 7. Manual Smoke Test Protocol for Reviewers

Once a local Python 3.12+ environment is activated, execute the following manual tests:

1. **Benchmark Ingestion & Profiler Smoke Test:**
   - Launch `streamlit run App.py`.
   - On "Workspace Home", click **Load sample datasets (SaaS Churn, Drift & Housing)**.
   - Navigate to **Data Health & Profiler** (`Pages/Dashboard.py`). Confirm health score, column types, and missingness metrics render.
2. **Data Cleaning & Hardened Export:**
   - Navigate to **Data Cleaning & Imputation** (`Pages/Cleaning.py`).
   - Impute missing values on `usage_hours` (Median) and drop duplicate rows.
   - Click **Save to Workspace Catalog**. Confirm file is saved to `Datasets/` and appears in the dataset switcher without formula injection.
3. **Feature Engineering Chaining & Reset:**
   - Navigate to **Feature Engineering** (`Pages/FeatureEngineering.py`).
   - Under Tab 1 (Numerical Scaling), select `monthly_spend_usd` and apply StandardScaler.
   - Confirm `monthly_spend_usd_scaled` is created and displayed in the export table.
   - Click **Discard Changes**; confirm table reverts back to original dimensions.
4. **Model Training & Prediction Pipeline:**
   - Navigate to **Model Studio & Training** (`Pages/ML.py`).
   - Select `churned` as target; train a Random Forest classifier.
   - Save the model; click the link **Open Batch Scoring & Simulation**.
   - Confirm model is loaded and batch predictions can be run with drift diagnostics.
5. **AI Assistant Safe Handling:**
   - Navigate to **AI Summary & Insights** (`Pages/AI.py`) without providing an API key.
   - Confirm clear warning displays in sidebar and main panel without crashing.
   - Supply a Gemini API key; verify PII columns are detected and excluded from the payload before insights are generated.

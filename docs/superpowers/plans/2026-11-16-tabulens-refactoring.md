# TabuLens Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the project into **TabuLens**, an authentic, PEP-8 standardized, 100% offline tabular data science studio featuring leak-free scikit-learn pipelines, statistical drift detection, decision threshold tuning, and standalone code export.

**Architecture:** A flat Streamlit application at the repository root (`app.py`, `pages/`, `utils/`) stripped of GenAI dependencies and CSS bloat. Enhances ML evaluation with operational binary classification thresholding, dynamic standalone inference script synthesis, and automated CI/CD quality gates.

**Tech Stack:** Python 3.12+, Streamlit 1.52+, scikit-learn 1.3+, pandas 2.0+, numpy, Plotly 5.18+, ReportLab, Docker, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-16-tabulens-refactoring-design.md`

## Global Constraints

- **Python Runtime:** Python 3.12+ compatible.
- **Offline / Local First:** Ingestion, modeling, drift detection, and PDF generation must work 100% offline without external network or API calls.
- **Resource Ceilings:** Strict ingestion limits in `utils/limits.py` (200 MB upload, 1M rows, 200 cols, 400 MB RAM) remain active.
- **Naming & Commits:** All directory and module names must be lowercase `snake_case`. All commit messages must follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `style:`, `test:`, `docs:`) under 60 characters.
- **Zero Broken Tests:** Every task must leave the test suite 100% passing.

## Review Focus

1. **Threshold Auto-Save Desynchronization:** Updating the threshold slider in `pages/ml.py` must atomically update the active model bundle on disk so batch scoring in `pages/predict.py` uses the tuned cutoff.
2. **Binary Positive Class Mapping:** Thresholding must map to `bundle["classes_"][1]` based on `bundle["positive_class"]`, never raw 0/1 integers.
3. **Script Generation Syntax Safety:** `generate_standalone_inference_code` must use `json.dumps()` for sample data, subset `bundle["feature_cols"]`, and coerce dtypes safely without code injection.
4. **Theme Backward Compatibility:** `utils/theme.py` must retain dictionary shims (`SEMANTIC_TOKENS`, `CHART_PALETTE`, `get_semantic_tokens`, `style_plotly_fig`, `safe_error_boundary`) under 100 lines so tests and charts do not break.
5. **Docker Case Sensitivity & Secret Bleed:** Storage paths must be standardized in lowercase (`datasets/`, `models/`, `reports/`) and excluded from Docker images via `.dockerignore`.

---

### Task 1: Pythonic Structure & Namespace Normalization (PEP 8)

**Files:**
- Rename/Move: `App.py` -> `app.py`
- Rename/Move: `Utils/` -> `utils/`, rename internal modules (`Charts.py` -> `charts.py`, `ML.py` -> `ml.py`, `PDF.py` -> `pdf.py`, `Preprocessing.py` -> `preprocessing.py`)
- Rename/Move: `Pages/` -> `pages/`, rename internal modules (`Upload.py` -> `upload.py`, `Dashboard.py` -> `dashboard.py`, `Cleaning.py` -> `cleaning.py`, `FeatureEngineering.py` -> `feature_engineering.py`, `EDA.py` -> `eda.py`, `Visualization.py` -> `visualization.py`, `Compare.py` -> `compare.py`, `ML.py` -> `ml.py`, `Predict.py` -> `predict.py`, `Report.py` -> `report.py`)
- Modify: `pyproject.toml` (package name `tabulens`)
- Test: `tests/test_package_structure.py`

**Interfaces:**
- Consumes: Existing files and directories
- Produces: Standard lowercase Python packages `utils` and `pages`, updated `app.py`

- [ ] **Step 1: Write test verifying PEP 8 module structure and importability**

```python
# tests/test_package_structure.py
import unittest
from pathlib import Path

class TestPackageStructure(unittest.TestCase):
    def test_root_and_package_naming(self):
        self.assertTrue(Path("app.py").is_file(), "app.py must be lowercase")
        self.assertTrue(Path("utils").is_dir(), "utils/ directory must be lowercase")
        self.assertTrue(Path("pages").is_dir(), "pages/ directory must be lowercase")
        for p in Path("utils").glob("*.py"):
            self.assertEqual(p.name, p.name.lower(), f"{p.name} must be lowercase")
        for p in Path("pages").glob("*.py"):
            self.assertEqual(p.name, p.name.lower(), f"{p.name} must be lowercase")

    def test_clean_imports(self):
        from utils import limits, paths, dataset_ui
        self.assertIsNotNone(limits.ResourcePolicy)
        self.assertIsNotNone(paths.DATASETS_DIR)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_package_structure.py -v`  
Expected: FAIL (files currently uppercase)

- [ ] **Step 3: Rename files, directories, and update internal imports**

1. Rename `App.py` to `app.py`.
2. Rename `Utils/` to `utils/` and normalize all filenames to snake_case.
3. Rename `Pages/` to `pages/` and normalize all filenames to snake_case.
4. Update `pyproject.toml` to name the package `tabulens` and include `["utils*", "pages*"]`.
5. Update all import statements across `app.py`, `pages/*.py`, `utils/*.py`, and `tests/*.py` to use `from utils...` and `from pages...`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_package_structure.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py pages utils pyproject.toml tests
git commit -m "refactor: rename modules to pep 8 snake_case"
```

---

### Task 2: Purge GenAI Dependencies, AST Sandbox & AI Bloat

**Files:**
- Delete: `pages/ai.py`, `pages/aichat.py`, `pages/aiconverter.py`
- Delete: `utils/gemini.py`, `utils/ai_agent.py`, `utils/ai_sandbox.py`, `utils/ai_convert.py`, `utils/markup_parsing.py`, `utils/secrets.py`, `utils/model_security.py`
- Delete: `tests/test_ai_agent_and_sandbox.py`, `tests/test_ai_conversion.py`
- Delete: `docs/PORTFOLIO_MEMO.md`, `AUDIT_REPORT.md`, `PROJECT_WALKTHROUGH.md`, `docs/CAPSTONE_THESIS_DEFENSE.md`
- Modify: `app.py` (remove AI navigation group)
- Modify: `requirements.txt`, `requirements-lock.txt`, `pyproject.toml` (remove `google-genai`, `pypdf`, `defusedxml`)

**Interfaces:**
- Consumes: Purged file list from spec § 2
- Produces: 100% offline codebase with zero external LLM or sandbox code

- [ ] **Step 1: Write test verifying zero residual GenAI and sandbox imports**

```python
# tests/test_offline_integrity.py
import unittest
from pathlib import Path

class TestOfflineIntegrity(unittest.TestCase):
    def test_no_genai_files_exist(self):
        forbidden = [
            "utils/gemini.py", "utils/ai_sandbox.py", "utils/ai_agent.py",
            "utils/ai_convert.py", "utils/markup_parsing.py", "utils/secrets.py",
            "pages/ai.py", "pages/aichat.py", "pages/aiconverter.py"
        ]
        for f in forbidden:
            self.assertFalse(Path(f).exists(), f"Forbidden file {f} still exists")

    def test_no_genai_in_requirements(self):
        with open("requirements.txt", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("google-genai", content)
        self.assertNotIn("pypdf", content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_offline_integrity.py -v`  
Expected: FAIL

- [ ] **Step 3: Delete obsolete files and update requirements and app navigation**

1. Delete the specified AI and sandbox files in `pages/`, `utils/`, `tests/`, and `docs/`.
2. Update `app.py` navigation to contain only:
   - Overview: Dashboard (`pages/dashboard.py`)
   - Data Management: Ingest (`pages/upload.py`), Hygiene & Cleaning (`pages/cleaning.py`), Feature Engineering (`pages/feature_engineering.py`)
   - Exploratory Analysis: Statistical EDA (`pages/eda.py`), Chart Studio (`pages/visualization.py`), Drift Audit (`pages/compare.py`)
   - Machine Learning: Model Studio (`pages/ml.py`), Batch Scoring (`pages/predict.py`)
   - Reports: PDF Audit Report (`pages/report.py`)
3. Clean `requirements.txt`, `requirements-lock.txt`, and `pyproject.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_offline_integrity.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor: remove genai dependencies and ast sandbox"
```

---

### Task 3: Streamline Theme & CSS (`utils/theme.py`)

**Files:**
- Modify: `utils/theme.py`
- Test: `tests/test_theme_and_boundaries.py`

**Interfaces:**
- Consumes: Clean styling requirements from spec § 5.1
- Produces: Minimal (<100 lines) `utils/theme.py` with backward-compatible styling tokens and error boundary

- [ ] **Step 1: Write test for minimal theme helpers and backward-compatible shims**

```python
# tests/test_theme_and_boundaries.py
import unittest
from utils.theme import (
    page_header,
    section_card_header,
    safe_error_boundary,
    plot_template,
    sanitize_error_message,
    SEMANTIC_TOKENS,
    CHART_PALETTE,
    get_semantic_tokens,
    style_plotly_fig,
)
import plotly.graph_objects as go

class TestThemeAndBoundaries(unittest.TestCase):
    def test_semantic_tokens_and_palette(self):
        self.assertIn("primary", SEMANTIC_TOKENS)
        self.assertIn("success", SEMANTIC_TOKENS)
        self.assertIsInstance(CHART_PALETTE, list)
        self.assertTrue(len(CHART_PALETTE) >= 4)
        tokens = get_semantic_tokens(dark=True)
        self.assertEqual(tokens, SEMANTIC_TOKENS)

    def test_sanitize_error_message(self):
        raw = "Error at C:\\Users\\secret\\project\\data.csv: failed"
        clean = sanitize_error_message(raw)
        self.assertNotIn("Users", clean)
        self.assertIn("[path redacted]", clean)

    def test_safe_error_boundary_catches(self):
        caught = []
        with safe_error_boundary("Test Failure", handler=lambda msg: caught.append(msg)):
            raise ValueError("Something broke at /home/user/file.py")
        self.assertEqual(len(caught), 1)
        self.assertIn("Test Failure", caught[0])
        self.assertNotIn("/home/user", caught[0])

    def test_style_plotly_fig(self):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        styled = style_plotly_fig(fig)
        self.assertIsNotNone(styled)
```

- [ ] **Step 2: Run test to verify current state**

Run: `python -m unittest tests/test_theme_and_boundaries.py -v`  
Expected: PASS or FAIL depending on module refactoring

- [ ] **Step 3: Implement clean, lightweight `utils/theme.py` (<100 LOC)**

Implement `utils/theme.py`:
- `SEMANTIC_TOKENS` dictionary.
- `CHART_PALETTE` list.
- `get_semantic_tokens(dark=True)`.
- `sanitize_error_message(raw_msg)`.
- `safe_error_boundary(title="An error occurred", handler=None)`.
- `page_header(title, subtitle, badge=None)`.
- `section_card_header(title, description)`.
- `plot_template()`.
- `style_plotly_fig(fig)`.
- Minimal `apply_theme()` applying subtle base styling without DOM overrides.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_theme_and_boundaries.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/theme.py tests/test_theme_and_boundaries.py
git commit -m "style: simplify theme to lightweight utility under 100 lines"
```

---

### Task 4: Capstone Feature 1 — Decision Threshold Optimization (`utils/ml.py`, `pages/ml.py`, `pages/predict.py`)

**Files:**
- Modify: `utils/ml.py`
- Modify: `pages/ml.py`
- Modify: `pages/predict.py`
- Test: `tests/test_decision_threshold.py`

**Interfaces:**
- Consumes: Binary classification probabilities and target classes
- Produces: Operational threshold parameter in `ModelArtifactBundle` and threshold-aware `predict_with_model()`

- [ ] **Step 1: Write test for decision threshold prediction and metric recalculation**

```python
# tests/test_decision_threshold.py
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.ml import predict_with_model, evaluate_threshold_metrics

class TestDecisionThreshold(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        y = np.array(["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"])
        pipe = Pipeline([("clf", LogisticRegression())])
        pipe.fit(X, y)
        self.bundle = {
            "pipeline": pipe,
            "feature_cols": ["feat1"],
            "target_column": "churn",
            "problem_type": "Classification",
            "classes": ["No", "Yes"],
            "positive_class": "Yes",
            "decision_threshold": 0.50
        }
        self.X_test = X

    def test_predict_with_default_threshold(self):
        preds = predict_with_model(self.bundle, self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_predict_with_threshold_override(self):
        # A very high threshold (0.95) should reduce positive predictions
        preds_high = predict_with_model(self.bundle, self.X_test, threshold_override=0.95)
        preds_low = predict_with_model(self.bundle, self.X_test, threshold_override=0.10)
        yes_high = sum(1 for p in preds_high if p == "Yes")
        yes_low = sum(1 for p in preds_low if p == "Yes")
        self.assertLessEqual(yes_high, yes_low)

    def test_evaluate_threshold_metrics(self):
        y_true = np.array(["No", "No", "Yes", "Yes"])
        y_proba = np.array([0.1, 0.4, 0.6, 0.8])
        metrics = evaluate_threshold_metrics(y_true, y_proba, classes=["No", "Yes"], threshold=0.5)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("confusion_matrix", metrics)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_decision_threshold.py -v`  
Expected: FAIL (`evaluate_threshold_metrics` or `threshold_override` not implemented)

- [ ] **Step 3: Implement threshold logic and update UI**

1. In `utils/ml.py`:
   - Implement `evaluate_threshold_metrics(y_true, y_proba, classes, threshold=0.50, positive_class=None)`.
   - Update `predict_with_model(bundle, df, feature_cols=None, threshold_override=None)` to apply $(P(y = \text{positive\_class}) \ge \theta)$ when `bundle["problem_type"] == "Classification"` and `len(classes) == 2`.
   - Cache holdout prediction probabilities `results["y_test_proba"] = y_test_proba` and `results["y_test_true"] = y_test` in `train_and_evaluate_model()`.
2. In `pages/ml.py`:
   - For binary classification, render `threshold = st.slider(...)`.
   - When adjusted, recompute metrics using `evaluate_threshold_metrics()` in $O(1)$ and update confusion matrix and metrics in session state.
   - When saving model, save `bundle["decision_threshold"] = threshold`.
3. In `pages/predict.py`:
   - Display active threshold metric and provide threshold override slider for binary classification batch scoring.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_decision_threshold.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/ml.py pages/ml.py pages/predict.py tests/test_decision_threshold.py
git commit -m "feat: add decision threshold optimization for binary classification"
```

---

### Task 5: Capstone Feature 2 — Standalone Python Inference Script Export (`utils/ml.py`, `pages/predict.py`)

**Files:**
- Modify: `utils/ml.py`
- Modify: `pages/ml.py`
- Modify: `pages/predict.py`
- Test: `tests/test_inference_script_generation.py`

**Interfaces:**
- Consumes: `bundle` dictionary and optional sample dataframe
- Produces: `generate_standalone_inference_code(bundle, sample_df=None) -> str`

- [ ] **Step 1: Write test for script generation and standalone code execution**

```python
# tests/test_inference_script_generation.py
import unittest
import tempfile
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from utils.ml import generate_standalone_inference_code

class TestInferenceScriptGeneration(unittest.TestCase):
    def test_generate_and_execute_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            pipe = Pipeline([("clf", RandomForestClassifier(n_estimators=5, random_state=42))])
            X = pd.DataFrame({"age": [25, 45, 60], "spend": [100.0, 200.0, 300.0]})
            y = ["Standard", "Premium", "Premium"]
            pipe.fit(X, y)
            bundle = {
                "pipeline": pipe,
                "feature_cols": ["age", "spend"],
                "target_column": "tier",
                "problem_type": "Classification",
                "classes": ["Premium", "Standard"],
                "decision_threshold": 0.50
            }
            joblib.dump(bundle, model_path)

            code = generate_standalone_inference_code(bundle, sample_df=X.head(1), model_rel_path=str(model_path))
            self.assertIn("import joblib", code)
            self.assertIn("pipeline.predict", code)

            # Verify generated code executes cleanly
            scope = {}
            exec(code, scope)
            self.assertIn("predictions", scope)
            self.assertEqual(len(scope["predictions"]), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_inference_script_generation.py -v`  
Expected: FAIL (`generate_standalone_inference_code` not implemented)

- [ ] **Step 3: Implement `generate_standalone_inference_code`**

In `utils/ml.py`:
- Implement `generate_standalone_inference_code(bundle, sample_df=None, model_rel_path="model.joblib") -> str`.
- Serialize sample data using `json.dumps()` (safe from code injection).
- Subset feature columns `df[features].copy()`.
- Emit thresholding logic for binary classification with `predict_proba`.
- In `pages/ml.py` and `pages/predict.py`: add an "Export Standalone Inference Script" expander with code block and download button.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_inference_script_generation.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/ml.py pages/ml.py pages/predict.py tests/test_inference_script_generation.py
git commit -m "feat: add standalone python inference code generator"
```

---

### Task 6: Capstone Feature 3 — Outlier & Distribution Diagnostics (`utils/eda_logic.py`, `pages/eda.py`)

**Files:**
- Modify: `utils/eda_logic.py`
- Modify: `pages/eda.py`
- Test: `tests/test_outlier_diagnostics.py`

**Interfaces:**
- Consumes: Numeric pandas DataFrame
- Produces: Structured outlier summary DataFrame and UI diagnostics tab

- [ ] **Step 1: Write test for outlier summary calculation**

```python
# tests/test_outlier_diagnostics.py
import unittest
import pandas as pd
import numpy as np
from utils.eda_logic import compute_outlier_summary

class TestOutlierDiagnostics(unittest.TestCase):
    def test_compute_outlier_summary(self):
        # 100 normal values and 3 extreme outliers
        rng = np.random.default_rng(42)
        vals = list(rng.normal(50, 5, 100)) + [100.0, -20.0, 150.0]
        df = pd.DataFrame({"normal_feat": vals, "text_feat": ["A"] * 103})
        summary = compute_outlier_summary(df)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["Column"], "normal_feat")
        self.assertGreaterEqual(row["IQR Outliers"], 3)
        self.assertIn("Lower Fence", row)
        self.assertIn("Upper Fence", row)
        self.assertIn("Z-Score Outliers", row)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_outlier_diagnostics.py -v`  
Expected: FAIL (`compute_outlier_summary` not implemented)

- [ ] **Step 3: Implement outlier summary in `utils/eda_logic.py` and UI tab in `pages/eda.py`**

1. Implement `compute_outlier_summary(df: pd.DataFrame) -> pd.DataFrame` calculating Tukey's fences and Z-score counts ($|z| > 3$).
2. In `pages/eda.py`, add the tab `"Outlier & Distribution Diagnostics"` displaying this table with box plots and summary statistics.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_outlier_diagnostics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/eda_logic.py pages/eda.py tests/test_outlier_diagnostics.py
git commit -m "feat: add iqr and z-score outlier summary in eda"
```

---

### Task 7: Production 2-Stage Docker & GitHub Actions CI (`Dockerfile`, `ci.yml`)

**Files:**
- Modify: `Dockerfile`
- Modify: `.dockerignore`
- Modify: `.github/workflows/ci.yml` (and delete obsolete `tests.yml`)
- Test: `tests/test_deployment_config.py`

**Interfaces:**
- Consumes: Python runtime, dependencies, and Streamlit config
- Produces: Hardened multi-stage container build and automated CI pipeline

- [ ] **Step 1: Write test verifying 2-stage Dockerfile and strict .dockerignore**

```python
# tests/test_deployment_config.py
import os
import unittest

class TestDeploymentConfig(unittest.TestCase):
    def test_dockerfile_multi_stage_structure(self):
        with open("Dockerfile", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("AS builder", content)
        self.assertIn("AS runner", content)
        self.assertIn("USER appuser", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("streamlit run app.py", content)

    def test_dockerignore_excludes_local_artifacts(self):
        with open(".dockerignore", "r", encoding="utf-8") as f:
            content = f.read()
        for item in ["datasets", "models", "reports", ".git", ".venv", "__pycache__"]:
            self.assertIn(item, content)

    def test_github_actions_workflow_exists(self):
        self.assertTrue(os.path.exists(".github/workflows/ci.yml"))
        with open(".github/workflows/ci.yml", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("ruff check", content)
        self.assertIn("coverage run", content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_deployment_config.py -v`  
Expected: FAIL (single-stage Dockerfile currently)

- [ ] **Step 3: Implement 2-stage Dockerfile, .dockerignore, and .github/workflows/ci.yml**

1. Write multi-stage `Dockerfile`:
   - Stage 1 (`builder`): `python:3.12-slim`, install `build-essential`, wheel build in `/wheels`.
   - Stage 2 (`runner`): clean `python:3.12-slim`, copy `/wheels` and install, create `appuser`, expose 8501, healthcheck, entrypoint `streamlit run app.py`.
2. Update `.dockerignore` to exclude local artifacts (`datasets/`, `models/`, `reports/`, etc.).
3. Configure `.github/workflows/ci.yml` with linting, unit tests + coverage, and container build check. Remove old `.github/workflows/tests.yml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_deployment_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore .github/workflows/ci.yml tests/test_deployment_config.py
git commit -m "ci: configure 2-stage docker build and github actions workflow"
```

---

### Task 8: Authentic Documentation & Pipeline Verification Runner

**Files:**
- Create: `scripts/verify_pipeline.py`
- Modify: `README.md`
- Create: `docs/METHODOLOGY.md`
- Test: `tests/test_verify_pipeline_cli.py`

**Interfaces:**
- Consumes: Complete project state
- Produces: Clean, honest `README.md`, mathematical methodology guide, and CLI verification runner

- [ ] **Step 1: Write test for verify_pipeline runner**

```python
# tests/test_verify_pipeline_cli.py
import unittest
import subprocess
import sys

class TestVerifyPipelineCLI(unittest.TestCase):
    def test_verify_pipeline_runs_cleanly(self):
        res = subprocess.run(
            [sys.executable, "scripts/verify_pipeline.py", "--fast"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0, msg=f"Verification failed:\n{res.stderr}\n{res.stdout}")
        self.assertIn("ALL QUALITY GATES PASSED", res.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_verify_pipeline_cli.py -v`  
Expected: FAIL (`scripts/verify_pipeline.py` not created yet)

- [ ] **Step 3: Implement `scripts/verify_pipeline.py`, `README.md`, and `docs/METHODOLOGY.md`**

1. Implement `scripts/verify_pipeline.py`:
   - Validates Python runtime (>= 3.12).
   - Validates required deployment files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`).
   - Runs unit test discovery.
   - Outputs a clean, colorized terminal summary and returns exit code 0.
2. Rewrite `README.md`:
   - Direct, authentic overview of TabuLens.
   - Architecture & pipeline diagrams.
   - Baseline benchmark tables (SaaS Churn & Housing).
   - Local setup & Docker quickstart.
   - Resource limits table.
3. Write `docs/METHODOLOGY.md` with formulas and rationale for leak-free cross-validation, Kolmogorov-Smirnov drift detection with Benjamini-Hochberg FDR control, and decision threshold optimization.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_verify_pipeline_cli.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_pipeline.py README.md docs/METHODOLOGY.md tests/test_verify_pipeline_cli.py
git commit -m "docs: rewrite readme and add pipeline verification runner"
```

---

## Plan Self-Review Checklist

- [x] **Spec coverage:** All sections of `docs/superpowers/specs/2026-11-16-tabulens-refactoring-design.md` mapped to tasks.
- [x] **Step scan:** Every step contains checkable actions with exact commands, test code, and commit messages.
- [x] **Type consistency:** Function signatures match across `utils/ml.py`, `pages/ml.py`, `pages/predict.py`, and test files.
- [x] **Review Focus:** Addressed threshold race conditions, binary positive class mapping, script generation syntax safety, and theme backward compatibility.

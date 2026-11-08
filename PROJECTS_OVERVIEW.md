# CloudInsight AI — Project Overview & Architecture Summary

**Evaluation Date:** November 3, 2026  
**Project:** CloudInsight AI (`cloudinsight-ai` v2.2.0)  
**Workspace Root:** `C:\Users\Msi\Projects\CloudlnsightAl`  

---

## 1. Project Profile

| Dimension | Specification |
|---|---|
| **Project Name** | **CloudInsight AI** (`cloudinsight-ai` v2.2.0) |
| **Directory** | `C:\Users\Msi\Projects\CloudlnsightAl` |
| **Application Type** | Multi-page data science web application & analytics studio |
| **Current Status** | Production Workbench (Stage 4 Maturity — self-contained, leak-free, cryptographically verified) |
| **Core Technology Stack** | Python 3.12+, Streamlit 1.52+, pandas 2.0+, scikit-learn 1.3+, Plotly 5.18+, ReportLab 4.0+, google-genai 1.0+ |
| **Persistence** | Local filesystem (`Datasets/`, `Models/`, `Reports/`, `ReportTemplates/`) |
| **Security Verification** | HMAC-SHA256 model signing, XXE/DTD rejection, zip bomb guards, CSV formula sanitization, ephemeral credentials |

---

## 2. Strongest Verified Capabilities

1. **Zero-Cloud Local Ingestion & Resource Bounding:**
   - Ingests CSV, TSV, Parquet, Excel (`.xlsx`, `.xls`), JSON, JSONL, XML, and HTML tables completely on-device without cloud dependencies.
   - Enforces streaming row (1M), column (200), cell (20M), and in-memory DataFrame (400 MB) budget ceilings to prevent workstation freeze.
2. **Leak-Free Scikit-Learn Modeling & 5-Fold Cross-Validation:**
   - Preprocessing (scaling, imputation, and categorical one-hot encoding) is strictly encapsulated inside `Pipeline` and `ColumnTransformer` objects fitted solely on training folds.
   - Automatically benchmarks models against naive majority-class or mean-target baselines with standard deviation variability reporting.
3. **Cryptographic Model Tamper Resistance (HMAC-SHA256):**
   - Eliminates arbitrary code execution vulnerabilities inherent to Python's `pickle` and `joblib` serializers by requiring a valid local HMAC signature before loading model artifacts.
4. **Statistical Diagnostics & Dataset Drift Engine:**
   - Evaluates distribution shifts using two-sample Kolmogorov-Smirnov tests with Benjamini-Hochberg False Discovery Rate (FDR) control and Total Variation Distance (TVD) for categorical features.
   - Conducts D'Agostino-Pearson normality tests, Variance Inflation Factor (VIF) multicollinearity screening, and One-Way ANOVA F-tests.
5. **Publication-Ready PDF Audit Reporting:**
   - Compiles comprehensive audit reports via ReportLab with embedded high-DPI figures, Data Hygiene Index gauges, and descriptive statistics.

---

## 3. Key Risks & Known Limitations

- **Local Host Dependency:** Requires Python 3.12+ installed locally (invoking `python.exe` on machines without Python installed resolves to the default Windows Store alias).
- **OCR for Scanned PDFs:** The built-in PDF text parser (`pypdf`) extracts digital text streams; bitmapped image PDFs require external OCR before ingestion.
- **Single-User Workspace:** Designed for local desktop execution (`127.0.0.1:8501`). Multi-tenant deployment across networks requires a reverse proxy with TLS and authentication.

---

## 4. Quickest Safe Way to Evaluate

### Option A: Read Technical Documentation
1. Review the senior engineering audit findings: [AUDIT_REPORT.md](AUDIT_REPORT.md).
2. Inspect system architecture and C4 container diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
3. Review the recruiter and technical interview brief: [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md).

### Option B: Run Locally
1. Ensure Python 3.12+ is installed:
   ```bash
   python --version
   ```
2. Activate virtual environment and install pinned dependencies:
   ```bash
   pip install -r requirements-lock.txt
   ```
3. Run the test suite:
   ```bash
   python -m unittest discover -s tests -v
   ```
4. Start the studio:
   ```bash
   streamlit run App.py
   ```
5. Click **Load sample datasets** on Workspace Home to immediately explore pre-loaded SaaS Churn, Drift, and Housing benchmarks without uploading external files.

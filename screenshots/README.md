# CloudInsight AI — Localhost Visual Tour & Usage Guide

This directory contains comprehensive screenshots captured directly from the live localhost Streamlit application (`http://127.0.0.1:8501`), covering all 10 core pages and external service integrations.

---

## Screenshot Inventory & Quick Reference

| # | Filename | Page / Module | Key Features & Workflow |
| :---: | :--- | :--- | :--- |
| **01** | `01_home_empty.png` | **Home** (`/`) | Architecture badges, step-by-step pipeline overview, and sample dataset bootstrapper. |
| **02** | `02_home_loaded.png` | **Home** (`/`) | Active dataset state (`saas_customers_q1.csv`), row/col counters, and sidebar navigation. |
| **03** | `03_ingest_local_file.png` | **Ingest data** (`Pages/Upload.py`) | Drag-and-drop file ingestion, aggregate size validation, format whitelist, multi-file merging. |
| **04** | `04_ingest_s3_manual_keys.png` | **Ingest data (S3)** | Manual Access Key, Secret Key, and **AWS Session Token (Gotcha 2)** inputs with session control. |
| **05** | `05_ingest_s3_iam_ambient.png` | **Ingest data (S3)** | **IAM Role / Ambient credentials mode (Gotcha 3)** with zero hardcoded keys required. |
| **06** | `06_ingest_s3_export_tab.png` | **Ingest data (S3)** | **Export to S3 tab (Gotcha 8)**: direct upload of datasets to S3 buckets with formula sanitization. |
| **07** | `07_cleaning_options.png` | **Cleaning** (`Pages/Cleaning.py`) | Source selector, original dataset hygiene metrics, deduplication toggle, and imputation strategies. |
| **08** | `08_cleaning_result_and_s3_export.png` | **Cleaning** (`Pages/Cleaning.py`) | Cleaned dataset metrics, row deltas, CSV download, and **S3 Export Expander (Gotcha 8)**. |
| **09** | `09_compare_datasets.png` | **Compare** (`Pages/Compare.py`) | Schema diffing, row differences, Kolmogorov-Smirnov (KS) & Total Variation Distance (TVD) drift. |
| **10** | `10_eda_overview_and_correlations.png` | **EDA** (`Pages/EDA.py`) | Data types, missingness percentages, descriptive stats, correlation matrix, and IQR outliers. |
| **11** | `11_visualize_distributions.png` | **Visualize** (`Pages/Visualization.py`) | Interactive histogram with bin sliders, marginal box/violin plots, and category grouping (Hue). |
| **12** | `12_visualize_relational.png` | **Visualize** (`Pages/Visualization.py`) | Scatter/bubble plots with dynamic sizing, hue encoding, and OLS trendline regression overlays. |
| **13** | `13_visualize_composition.png` | **Visualize** (`Pages/Visualization.py`) | Categorical composition charts: Bar charts with custom aggregations, pie charts, and treemaps. |
| **14** | `14_dashboard_hygiene_index.png` | **Dashboard** (`Pages/Dashboard.py`) | Data Hygiene Index gauge (0-100), completeness/uniqueness rates, live numeric/text filters. |
| **15** | `15_machine_learning_setup.png` | **ML** (`Pages/ML.py`) | Problem setup (auto classification/regression detection), feature selection, and test split slider. |
| **16** | `16_machine_learning_results.png` | **ML** (`Pages/ML.py`) | Model evaluation: Accuracy/F1 scores, ROC curve, confusion matrix, and feature importances. |
| **17** | `17_ai_insights.png` | **AI insights** (`Pages/AI.py`) | Gemini executive insights, automated privacy column screening, and conversational chat. |
| **18** | `18_pdf_report_generator.png` | **PDF report** (`Pages/Report.py`) | Executive PDF report generator with custom title/author, templates, chart toggles, and PII filters. |
| **19** | `19_cloud_storage_s3.png` | **Cloud Storage (S3)** (`Pages/CloudS3.py`) | Dedicated S3 hub: Manual Keys, AWS Session Tokens, Ambient IAM roles, bucket browsing & exports. |
| **20** | `20_ai_file_converter.png` | **AI File Converter** (`Pages/AIConverter.py`) | Dedicated conversion studio: 3-step pipeline converting unstructured PDFs/logs/text into clean CSVs. |
| **21** | `21_ask_ai_chat.png` | **Ask AI (Chat)** (`Pages/AIChat.py`) | Dedicated conversational interface: 1-click starter questions, privacy shield & context controls. |
| **22** | `22_predict_and_score.png` | **Predict & Score** (`Pages/Predict.py`) | Dedicated model scoring studio: HMAC verification, architecture inspect & batch CSV predictions. |

---

## Detailed Page Explanations & Workflow Usage

### 01 & 02: Home Page (`App.py`)
* **File References**: `01_home_empty.png`, `02_home_loaded.png`
* **Purpose**: Application portal and active workspace manager.
* **Usage**:
  * Displays high-level technology badges: Storage (**Amazon S3**), AI (**Gemini 3.8 Flash**), and Modeling (**Scikit-Learn**).
  * Outlines the analytical workflow: **Ingest → Prepare → Explore → Model → Share**.
  * Contains the **"Load sample datasets (Q1 vs Q2 Drift & Churn)"** button, which generates two synthetic SaaS customer datasets (`saas_customers_q1.csv` and `saas_customers_q2_drifted.csv`) with deliberate missingness, duplicates, and statistical drift for instant exploration.
  * Shows the sidebar dataset indicator and appearance toggle (Dark Mode / Light Mode).

---

### 03, 04, 05 & 06: Data Ingestion & AWS S3 (`Pages/Upload.py`)
* **File References**: `03_ingest_local_file.png`, `04_ingest_s3_manual_keys.png`, `05_ingest_s3_iam_ambient.png`, `06_ingest_s3_export_tab.png`
* **Purpose**: Ingest structured tabular files, convert unstructured documents via Gemini, and bidirectionally integrate with Amazon S3.
* **Usage**:
  * **Local File Upload (`03`)**: Upload `.csv`, `.tsv`, `.xlsx`, `.parquet`, `.json`, `.pdf`, `.txt`, etc. Files exceeding limits (200 MB per file, 250 MB aggregate, 20 files max) are rejected cleanly. Unstructured files (PDFs, text) automatically offer Gemini-assisted conversion.
  * **S3 Manual Mode (`04`)**:
    * **Gotcha 2 Implemented**: Added the **AWS Session Token (Optional)** input field alongside Access Key ID and Secret Access Key to support temporary credentials from STS, AWS SSO, and IAM role assumption.
    * **Gotcha 1 Fixed**: Added an explicit **Clear keys** button alongside the **Keep in session state for this session** checkbox, preventing premature credential wiping while browsing bucket files.
  * **S3 IAM Ambient Mode (`05`)**:
    * **Gotcha 3 Implemented**: Select **IAM Role / Environment / AWS CLI (Automatic)** to connect using EC2/ECS/EKS instance profiles, `AWS_ACCESS_KEY_ID` environment variables, or `~/.aws/credentials` without typing manual secrets.
  * **S3 Export Tab (`06`)**:
    * **Gotcha 8 Implemented**: Select the active dataset or any saved dataset from `Datasets/`, specify a destination S3 object key (e.g. `exports/saas_customers_q1.csv`), optionally enable spreadsheet formula injection sanitization, and upload to S3 with a single click.
  * **Gotcha 4 Implemented (IAM Error Diagnostics)**: If permissions are missing, `describe_s3_error()` distinguishes `s3:ListBucket` (on `arn:aws:s3:::bucket`) from `s3:GetObject` or `s3:PutObject` (on `arn:aws:s3:::bucket/*`).

---

### 07 & 08: Data Cleaning (`Pages/Cleaning.py`)
* **File References**: `07_cleaning_options.png`, `08_cleaning_result_and_s3_export.png`
* **Purpose**: Clean tabular data by removing duplicate records and handling missing values.
* **Usage**:
  * **Source Selector**: Choose an active dataset or upload a new file.
  * **Hygiene Metrics**: Displays total rows, columns, duplicate count, and missing value count before cleaning.
  * **Cleaning Options**:
    * Toggle **Remove Duplicate Rows**.
    * Select **Missing Value Strategy**: Mean/Median/Mode imputation, zero/unknown fill, or row deletion.
  * **Results & Export (`08`)**:
    * Displays cleaned metrics with delta indicators (e.g. `-10` rows removed).
    * Provides **Download cleaned CSV** button.
    * **Gotcha 8 Implemented**: Includes an expandable **📤 Export cleaned dataset to Amazon S3** panel to upload the cleaned result directly to an S3 bucket with custom destination keys.

---

### 09: Compare Datasets & Statistical Drift (`Pages/Compare.py`)
* **File References**: `09_compare_datasets.png`
* **Purpose**: Compare two datasets side-by-side to detect schema changes and statistical distribution drift.
* **Usage**:
  * Select **Dataset A** (baseline) and **Dataset B** (comparison).
  * **Schema Diff**: Displays shared columns, columns unique to A, and columns unique to B.
  * **Statistical Drift Screening**:
    * Numeric columns evaluated using two-sample **Kolmogorov-Smirnov (KS) tests** with Benjamini-Hochberg False Discovery Rate (FDR) corrections.
    * Categorical columns evaluated using **Total Variation Distance (TVD >= 0.20)**.
    * Highlights columns showing drift in distribution, variance, or missingness rate.

---

### 10: Exploratory Data Analysis (`Pages/EDA.py`)
* **File References**: `10_eda_overview_and_correlations.png`
* **Purpose**: Deep exploratory analysis into feature distributions, missingness patterns, and outliers.
* **Usage**:
  * **Overview Cards**: Total rows, columns, numeric feature count, categorical feature count.
  * **Column Structure Table**: Column name, data type, non-null count, missing count, missing percentage, and unique cardinality.
  * **Descriptive Statistics**: Detailed pandas describe table (count, mean, std, min, 25%, 50%, 75%, max) for numeric and categorical features.
  * **Correlation Matrix**: Correlation coefficients with a coolwarm gradient heatmap.
  * **Outlier Detection**: Automated Interquartile Range (IQR) method calculating lower/upper bounds, outlier counts, and outlier percentages per numeric column.

---

### 11, 12 & 13: Interactive Visualizations (`Pages/Visualization.py`)
* **File References**: `11_visualize_distributions.png`, `12_visualize_relational.png`, `13_visualize_composition.png`
* **Purpose**: Rich Plotly charts for data distribution, relationship modeling, and categorical composition.
* **Usage**:
  * **Distributions (`11`)**: Histograms, Box plots, and Violin plots with customizable bin counts, marginal plots (box/violin/rug), and hue category grouping.
  * **Relational & Trends (`12`)**: Scatter and bubble plots with X/Y selection, color grouping, bubble sizing, and ordinary least squares (OLS) linear trendlines.
  * **Categorical & Composition (`13`)**: Bar/count charts with aggregations (Sum, Mean, Median, Min, Max), Pie charts, Donut charts, and Treemaps.

---

### 14: Health Dashboard (`Pages/Dashboard.py`)
* **File References**: `14_dashboard_hygiene_index.png`
* **Purpose**: Executive dashboard with high-level data health metrics and real-time slice-and-dice filters.
* **Usage**:
  * **Data Hygiene Index**: Radial gauge score (0–100) calculated from completeness rate and uniqueness rate.
  * **Live Multi-Column Filtering**: Add dynamic range sliders for numeric columns and multi-select tags / search boxes for categorical features. Filters update the active view and metrics in real time.
  * **Quick Distributions**: Side-by-side distribution and box plots for selected key performance indicators.

---

### 15 & 16: Machine Learning (`Pages/ML.py`)
* **File References**: `15_machine_learning_setup.png`, `16_machine_learning_results.png`
* **Purpose**: Train, evaluate, and persist machine learning models with automated preprocessing.
* **Usage**:
  * **Setup (`15`)**: Select the target variable (problem type auto-detected as Classification or Regression). Choose predictors from the available feature checklist, select the algorithm (Random Forest, Gradient Boosting, Decision Tree, Logistic Regression), and configure the test split size slider.
  * **Evaluation Results (`16`)**: Displays Accuracy, Weighted Precision, Recall, and F1 Macro metrics. Renders confusion matrix heatmaps and ROC curves. Models can be saved to `Models/` for inference.

---

### 17: AI Insights & Conversational Q&A (`Pages/AI.py`)
* **File References**: `17_ai_insights.png`
* **Purpose**: Gemini-powered executive summaries, automated privacy screening, and dataset chat.
* **Usage**:
  * **Privacy Screening**: Automatically scans column names and values for sensitive patterns (emails, credit cards, SSNs, phone numbers) and pre-selects them for exclusion from the AI prompt context.
  * **Executive Report**: Generates an AI-written summary covering dataset health, key statistical findings, risks, and strategic business recommendations.
  * **Chat**: Conversational interface enabling natural language queries against dataset structures and distributions.

---

### 18: PDF Report Generator (`Pages/Report.py`)
* **File References**: `18_pdf_report_generator.png`
* **Purpose**: Export publication-quality executive PDF reports.
* **Usage**:
  * Configure report title, author, and organization name (full-width stacked inputs so long titles never truncate).
  * Save and load reusable report configuration templates (`REPORTS_DIR/templates/`).
  * Toggle chart inclusions (histograms, box plots, correlation heatmaps) and Gemini executive summary inclusions.
  * Select columns to redact or exclude for privacy and security.
  * Generate and download the complete PDF report.

---

### 19: Dedicated Cloud Storage S3 Hub (`Pages/CloudS3.py`)
* **File References**: `19_cloud_storage_s3.png`
* **Purpose**: Dedicated, isolated hub for connecting to Amazon S3, browsing bucket objects, downloading datasets, and exporting processed data.
* **Usage**:
  * Authenticate via **Manual Access Keys** (with Session Token support) or **IAM Role / Ambient / AWS CLI (Automatic)**.
  * Clean, emoji-free tabs: `Import from S3` and `Export to S3`.
  * Safe memory governance: credentials remain in memory only and are never saved to disk.

---

### 20: Dedicated AI File Converter Studio (`Pages/AIConverter.py`)
* **File References**: `20_ai_file_converter.png`
* **Purpose**: 3-step beginner-friendly conversion pipeline transforming unstructured or semi-structured files into clean tabular CSVs using Google Gemini.
* **Usage**:
  * Clear 3-step plain-English instructions ("1. Upload or Paste -> 2. Instruct & Extract -> 3. Save & Explore").
  * Supports documents (`.pdf`, `.txt`, `.log`, `.md`, `.html`, `.csv`, `.json`) or directly pasted unstructured text.
  * Previews extracted table with row and column counts before saving to the active workspace.

---

### 21: Dedicated Ask AI Chat (`Pages/AIChat.py`)
* **File References**: `21_ask_ai_chat.png`
* **Purpose**: Conversational natural language Q&A with active datasets, complete with 1-click starter prompts for beginners.
* **Usage**:
  * 3 clickable starter prompt buttons:
    1. *"What are the 3 most important patterns in this dataset?"*
    2. *"Are there any data quality issues or unusual outliers I should fix?"*
    3. *"What are the most important relationships between columns?"*
  * Data privacy shield showing PII risk status and column exclusion multiselect.
  * Conversational chat bubble history with clean tabular answers.

---

### 22: Dedicated Predict & Score Studio (`Pages/Predict.py`)
* **File References**: `22_predict_and_score.png`
* **Purpose**: Production-grade model inference studio for cryptographic HMAC verification, architecture inspection, and batch predictions.
* **Usage**:
  * Select HMAC-SHA256 signed model bundles from `Models/`.
  * Inspect training metadata: target variable, algorithm, required input features, and validation metrics.
  * Score active dataset or uploaded test CSVs, preview predictions with confidence scores, and export results.


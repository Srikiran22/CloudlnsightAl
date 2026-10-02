# CloudInsight AI

CloudInsight AI is a local-first data analytics application built with Streamlit, pandas, scikit-learn, Plotly, ReportLab, and Google Gemini.

It brings common data-analysis tasks into one workspace: ingest a dataset, clean it, compare it with another dataset, explore it, train a machine-learning model, and export an executive PDF report. Google Gemini and Amazon S3 are optional integrations.

## Key Features

### Data ingestion

CloudInsight AI can load:

- CSV and TSV
- Excel (`.xlsx` and legacy `.xls`)
- JSON (including nested records)
- JSONL / NDJSON
- Parquet
- XML (hardened against DTD/entity expansion)
- HTML tables
- Delimited text
- PDF documents with extractable text (via optional `pypdf`)

Structured formats are handled locally. Files without a native table structure can be passed to Google Gemini for conversion into a validated CSV dataset.

Multiple local files can be loaded together and merged with source filename provenance retained.

### Data cleaning

The Cleaning page supports:

- Duplicate removal
- Mean, median, or zero imputation for numeric columns
- Mode or `Unknown` handling for categorical data
- Dropping rows with missing values
- Leaving missing values unchanged
- Saving cleaned datasets as new CSV files with deterministic collision-safe naming

### Dataset comparison

The Compare page compares two datasets at the schema and column level, including:

- Row and column counts
- Common and missing columns
- Missing-value changes
- Summary metric shifts (mean and dispersion / std)
- Distribution drift detection via two-sample Kolmogorov-Smirnov tests (numeric) and Total Variation Distance (categorical)
- Duplicate profiles

### Exploratory analysis and visualization

The EDA and Visualization pages provide:

- Column and data-type summaries
- Descriptive statistics (mean, std, min, quartiles, max)
- Pearson correlations with column caps to prevent layout freeze
- Tukey IQR outlier analysis
- Interactive charts: histograms, box/violin plots, scatter/bubble (with 25k sampling for responsiveness), bar/pie/treemap, heatmaps
- Categorical distribution analysis

### Machine learning

The ML workflow uses scikit-learn pipelines for preprocessing, training, evaluation, persistence, and prediction.

The current implementation includes:

- Logistic Regression
- Linear Regression
- Ridge Regression
- Decision Tree Classifier
- Random Forest Classifier
- Random Forest Regressor
- Gradient Boosting Classifier
- Gradient Boosting Regressor

Features include automatic classification/regression detection, 5-fold cross-validation with metric variability (mean ± std), dummy baseline comparison, leak-free preprocessing inside pipelines, high-dimension guards against dense one-hot encoding explosions, model persistence with Joblib and HMAC-SHA256 signature verification, prediction from saved models, and model provenance metadata with SHA-256 dataset fingerprinting.

A training-cell limit is enforced to prevent oversized jobs from blocking the Streamlit process.

### PDF reports

Reports are generated with ReportLab and can include:

1. Dataset summary and Data Hygiene Index
2. Column structure and missing-value counts
3. Numerical statistics
4. IQR outlier analysis
5. Categorical analysis
6. Correlation analysis
7. Column-level quality flags
8. Sample records
9. Optional charts (requires `matplotlib`; isolated per-chart rendering)
10. Optional Gemini insights

Report templates can be saved with overwrite protection, and batch report generation is supported.

### Gemini integration

Gemini is optional and is used for:

- AI conversion of unstructured text and PDF content into structured CSV tables
- Executive summary generation in PDF reports
- Automated dataset insights and conversational chat

The platform defaults to `gemini-3.8-flash` via the primary `google-genai` SDK, while supporting `gemini-3.5-flash`, `gemini-2.5-flash`, and `gemini-2.5-pro` (retired Gemini 1.x/2.0 models map automatically to supported modern equivalents).

## Requirements

- Python 3.12+ (tested and CI-validated on Ubuntu, Windows, and macOS across Python 3.12 and 3.13)
- Streamlit 1.52.0+
- Linux, Windows, or macOS

A database is not required. Production dependencies are locked in `requirements-lock.txt` for exact CI-matched reproducibility, with loose ranges in `requirements.txt`.

Optional runtime capabilities:

- `pypdf` for PDF text extraction
- `matplotlib` for charts in PDF reports

Gemini and AWS credentials are only required when using those features.

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```powershell
.\venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies (tested lockfile recommended for exact reproducibility matching CI):

```bash
pip install -r requirements-lock.txt
```

Alternatively, to install with unpinned ranges:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run App.py
```

Open:

```
http://localhost:8501
```

## Credentials and external services

### Google Gemini

Enter the Gemini API key in the application when using AI conversion, insights, or chat.

The application keeps credentials in Streamlit session state only and clears them after an operation unless they are explicitly kept for the current session. Nothing touches disk.

Before Gemini analysis, the privacy screener flags columns that look like emails, phone numbers, financial identifiers, credentials, or tokens so they can be excluded from the model context.

### Amazon S3

Enter AWS credentials in the S3 section of the Ingest page.

The S3 client uses chunked streaming with an accumulated byte ceiling to enforce the 200 MB ingest limit even if `ContentLength` is omitted.

## Application pages

| Page | Purpose |
|---|---|
| Ingest data | Local files, S3 browsing, and Gemini conversion |
| Cleaning | Duplicate removal and missing-value handling |
| Compare | Dataset comparison and schema drift |
| EDA | Statistics, correlations, and outliers |
| Visualization | Interactive charts |
| Dashboard | Dataset health gauge and summary metrics |
| Machine learning | Training, evaluation, saving, loading, and prediction |
| AI insights | Gemini analysis and dataset chat |
| PDF report | Report generation, templates, and batch mode |

## Repository structure

```text
CloudInsightAI/
├── .github/workflows/     # CI workflow configurations (tests, linting)
├── .streamlit/            # Streamlit theme and runtime configuration
├── Pages/                 # Multi-page application views
│   ├── AI.py              # AI insights and conversational dataset Q&A
│   ├── Cleaning.py        # Deduplication, missing-value handling, and export
│   ├── Compare.py         # Schema drift and column-level comparison
│   ├── Dashboard.py       # Data hygiene index gauge and summary indicators
│   ├── EDA.py             # Exploratory analysis, correlations, and outliers
│   ├── ML.py              # Model training, evaluation, saving, and prediction
│   ├── Report.py          # PDF report generation and template management
│   ├── Upload.py          # File ingestion (local files and Amazon S3)
│   └── Visualization.py   # Interactive plotting (Plotly)
├── Utils/                 # Domain logic, security guards, and utilities
│   ├── AIConvert.py       # Unstructured data extraction via Gemini
│   ├── Charts.py          # Visualization figure builders
│   ├── Gemini.py          # Google Gemini client and resilience handling
│   ├── ML.py              # Scikit-learn pipelines and model persistence
│   ├── PDF.py             # ReportLab PDF generation
│   ├── Preprocessing.py   # Cleaning and missing-value imputation
│   ├── S3.py              # Amazon S3 client and streaming ingestion
│   ├── batch.py           # Multi-file merging and schema alignment
│   ├── compare_logic.py   # Dataset comparison and drift metrics
│   ├── dataset_ui.py      # Session state and dataset caching helpers
│   ├── logsys.py          # Centralized application logging
│   ├── paths.py           # Safe path handling, format parsing, and limits
│   ├── privacy.py         # Sensitive column screening and PII detection
│   ├── quality.py         # Data hygiene index and completeness/uniqueness scoring
│   ├── sampling.py        # Centralized analytical and visualization sampling policy
│   ├── secrets.py         # Ephemeral in-memory credential management
│   └── theme.py           # UI styling and CSS tokens
├── tests/                 # Unit, security, and regression test suites
├── App.py                 # Application entry point and navigation
├── bug_hunt.py            # Headless page execution sanity check
├── CHANGELOG.md           # Product release notes and remediation history
├── CONTRIBUTING.md        # Contribution guidelines and development checks
├── pyproject.toml         # Ruff, pytest, and project metadata configuration
├── README.md              # Project documentation and setup guide
├── requirements.txt       # Base application dependencies
├── requirements-lock.txt  # Pinned dependency lockfile
└── SECURITY.md            # Security policy and vulnerability disclosure
```

## Development

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run the headless page check:

```bash
python bug_hunt.py
```

Run the CI-level serious-error Ruff check:

```bash
pip install ruff==0.16.4
ruff check --select=E9,F63,F7,F82,F821 --preview .
```

GitHub Actions runs these checks across Python 3.12 and 3.13.

## Security and privacy

The application is designed for local, single-user use.

Key safeguards include:

- Strict dataset path containment preventing directory traversal
- SHA-256 cryptographic dataset content identity with metadata-backed performance caching and fresh byte verification (`force_refresh=True`)
- XML DTD and external entity rejection preventing XXE attacks
- Decompression bomb and size guards across ZIP, Excel (100 MB decompressed limit), and Parquet parsers
- 200 MB upload ceiling, 1,000,000 row limit, 200 column limit, and 20,000,000 cell limit enforced streamingly during ingestion
- 400 MB deep DataFrame memory budget and continuous 5,000,000 high-cardinality string cell observation monitoring
- Bounded AI response parsing with column caps (>200 columns rejected)
- Ephemeral in-memory credential handling (no disk storage or intentional logging of API keys)
- Sensitive-column screening before Gemini analysis
- Provenance metadata, dataset fingerprinting, and HMAC-SHA256 authenticity signing for saved ML models

## Limitations and architectural considerations

- Local-first architecture: Designed for single-user analytical sessions; no multi-tenant authorization or per-user storage isolation
- Memory governance: The 400 MB DataFrame memory budget governs pandas in-memory data structures; total process RSS (working set) is higher due to Python runtime heap, openpyxl/parser objects, and temporary intermediate buffers during multi-chunk ingestion or S3 export
- Excel memory profile: Excel files (.xlsx / .xls) construct in-memory workbook DOMs; while bounded by decompression limits and row/cell limits, process RSS expands significantly compared to flat delimited text
- Temporal decimation: Min-max envelope bucketing enforces chronological ordering and captures local extrema and temporal boundaries, but does not guarantee zero aliasing for frequencies exceeding the bucket sampling rate
- Gemini and S3 require external services when those features are used
- PDF ingestion requires extractable text; scanned image PDFs need a separate OCR workflow
- Joblib model bundles rely on pickle and execute Python bytecode upon deserialization. Models trained in CloudInsight are cryptographically signed with HMAC-SHA256 to ensure integrity; external or untrusted `.joblib` files are blocked from loading
- Production deployment requires an authenticating reverse proxy, storage isolation, and network controls

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and contribution guidance.


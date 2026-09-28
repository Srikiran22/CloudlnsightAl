# CloudInsight AI

CloudInsight AI is a local-first data analytics application built with Streamlit, Pandas, Scikit-Learn, Plotly, ReportLab, and Google Gemini.

It brings common data-analysis tasks into one workspace: ingest a dataset, clean it, compare it with another dataset, explore it, train a machine-learning model, and export an executive PDF report. Google Gemini and Amazon S3 are optional integrations.

## What it does

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
- Numeric distribution shifts and summary metric shifts
- Duplicate profiles

### Exploratory analysis and visualization

The EDA and Visualization pages provide:

- Column and data-type summaries
- Descriptive statistics (quartiles, skewness, kurtosis)
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

Features include automatic classification/regression detection, leak-free preprocessing inside pipelines, high-dimension guards against dense one-hot encoding explosions, model persistence with Joblib, prediction from saved models, and model provenance metadata with SHA-256 dataset fingerprinting.

A training-cell limit is enforced to prevent oversized jobs from blocking the Streamlit process.

### PDF reports

Reports are generated with ReportLab and can include:

1. Dataset summary and Data Quality Index
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

The platform defaults to `gemini-3.8-flash` via the primary `google-genai` SDK, while retaining backward compatibility for `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, and `gemini-1.5-pro`.

## Requirements

- Python 3.10+
- Streamlit 1.52.0+
- Windows, macOS, or Linux

A database is not required. Dependencies are defined in `requirements.txt`.

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

Install dependencies:

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

The application keeps credentials in Streamlit session memory only and releases them after an operation unless they are explicitly kept for the current session. Nothing touches disk.

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

GitHub Actions runs these checks across Python 3.10, 3.11, and 3.12.

## Security and privacy

The application is designed for local, single-user use.

Key safeguards include:

- No built-in authentication
- Dataset path containment (traversal attempts rejected)
- Full 64-character SHA-256 cryptographic dataset content identity
- XML DTD/entity rejection to prevent entity expansion attacks
- 200 MB upload ceiling and parser size limits
- Bounded AI response parsing with column caps (>200 columns rejected)
- Runtime-only credential handling (no disk storage or intentional logging of API keys)
- Sensitive-column screening before Gemini analysis
- Provenance metadata and dataset fingerprinting for saved ML models

## Limitations

- Local-first architecture: No built-in authentication or multi-tenant per-user storage isolation
- Gemini and S3 require external services when those features are used
- PDF ingestion requires extractable text; scanned image PDFs need a separate OCR workflow
- Joblib model bundles execute code on unpickling — load only trusted models
- Production deployment requires additional authenticating reverse proxy, storage isolation, and network controls

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and contribution guidance.

## License

No LICENSE file is currently included in the repository.

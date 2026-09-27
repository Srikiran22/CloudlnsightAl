# CloudInsight AI

CloudInsight AI is a local-first data analytics application built with Streamlit.

It brings common data-analysis tasks into one workspace: ingest a dataset, clean it, compare it with another dataset, explore it, train a machine-learning model, and export a PDF report. Google Gemini and Amazon S3 are optional integrations.

## What it does

### Data ingestion

CloudInsight AI can load:

- CSV and TSV
- Excel
- JSON
- JSONL / NDJSON
- Parquet
- XML
- HTML tables
- Delimited text
- PDF documents with extractable text

Structured formats are handled locally. Files without a native table structure can be passed to Gemini for conversion into a validated CSV dataset.

Multiple local files can be loaded together and merged with their source filename retained.

### Data cleaning

The Cleaning page supports:

- Duplicate removal
- Mean or median imputation for numeric columns
- Mode, zero, or `Unknown` handling for categorical data
- Dropping rows with missing values
- Leaving missing values unchanged
- Saving cleaned datasets as new CSV files

### Dataset comparison

The Compare page compares two datasets at the schema and column level, including:

- Row and column counts
- Common and missing columns
- Missing-value changes
- Numeric distribution shifts
- Duplicate profiles

### Exploratory analysis and visualization

The EDA and Visualization pages provide:

- Column and data-type summaries
- Descriptive statistics
- Correlations
- Tukey IQR outlier analysis
- Distribution plots
- Interactive charts
- Categorical analysis

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

Features include automatic classification/regression detection, preprocessing for numeric and categorical columns, missing-value handling, model persistence with Joblib, prediction from saved models, and model provenance metadata.

A training-cell limit is used to prevent very large jobs from blocking the Streamlit application.

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
9. Optional charts
10. Optional Gemini insights

Report templates can be saved and reused, and batch report generation is supported.

### Gemini

Gemini is optional and is used for:

- Unstructured-file conversion
- Executive dataset insights
- Dataset chat

The conversion flow sends a bounded text sample to Gemini and validates the returned CSV before using it.

Gemini requests use timeouts and bounded retries for transient failures.

### Amazon S3

The application can:

- Connect to an S3 bucket
- List supported dataset objects
- Download datasets into the local `Datasets/` directory

The code also contains an S3 upload helper, but the current Streamlit interface does not expose dataset upload to S3.

## How the application works

```
Local file / S3
       |
       v
Native parser
       |
       +---- tabular file ----> DataFrame
       |
       +---- unstructured ----> Gemini
                                |
                                v
                         validated DataFrame
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
           Cleaning          Compare          EDA / Charts
              |                                   |
              +-----------------+-----------------+
                                |
                         ML / PDF reports
                                |
                                v
                       Local project artifacts
```

## Data Quality Index

The Dashboard and PDF report use the same implementation in `Utils/quality.py`.

```
Quality Index = (completeness + uniqueness) / 2
```

Where:

- **Completeness** is the percentage of non-missing cells.
- **Uniqueness** is the percentage of rows that are not duplicates.

Both components are weighted equally.

## Project structure

```
CloudInsightAI/
├── App.py                 # Application entry point
├── Pages/                 # Streamlit pages
├── Utils/                 # Shared processing and integrations
│   ├── paths.py           # File readers and path safety
│   ├── AIConvert.py       # AI-assisted file conversion
│   ├── Gemini.py          # Gemini integration
│   ├── ML.py              # Model training and persistence
│   ├── PDF.py             # PDF report generation
│   ├── S3.py              # Amazon S3 integration
│   ├── Preprocessing.py   # Cleaning operations
│   ├── compare_logic.py   # Dataset comparison
│   ├── quality.py         # Data Quality Index
│   ├── privacy.py         # Sensitive-column screening
│   ├── secrets.py         # Runtime credential handling
│   └── ...                # Charts, UI, logging, and batching
├── tests/                 # Automated tests
├── bug_hunt.py            # Headless Streamlit page check
├── Datasets/              # Local datasets; gitignored
├── Models/                # Saved models; gitignored
├── Reports/               # Generated reports; gitignored
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.10+
- Windows, macOS, or Linux

A database is not required.

The main dependencies are defined in `requirements.txt`.

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

The application keeps credentials in Streamlit session state and releases them after an operation unless they are explicitly kept for the current session.

Before Gemini analysis, the privacy screen can flag columns that look like emails, phone numbers, financial identifiers, credentials, or tokens so they can be excluded from the model context.

Because Gemini is an external service, do not send confidential data unless your data-handling requirements permit it.

### Amazon S3

Enter AWS credentials in the S3 section of the Ingest page.

The S3 client limits object scanning and rejects objects above the application's 200 MB ingest limit.

## Application pages

| Page | Purpose |
|---|---|
| Ingest data | Local files, S3 browsing, and Gemini conversion |
| Cleaning | Duplicate removal and missing-value handling |
| Compare | Dataset comparison |
| EDA | Statistics, correlations, and outliers |
| Visualization | Interactive charts |
| Dashboard | Dataset health and summary metrics |
| Machine learning | Training, evaluation, saving, loading, and prediction |
| AI insights | Gemini analysis and dataset chat |
| PDF report | Report generation and templates |

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
- Dataset path containment
- XML DTD/entity rejection
- Upload and parser size limits
- Bounded AI response parsing
- Runtime-only credential handling
- Sensitive-column screening before Gemini analysis
- Provenance metadata for saved ML models
- No intentional logging of API keys, dataset contents, or raw model responses

These safeguards are intended for the current local application and are not a replacement for a production multi-user security architecture.

## Limitations

- No built-in authentication or per-user storage isolation
- Gemini and S3 require external services when those features are used
- PDF ingestion requires extractable text; scanned image PDFs need a separate OCR workflow
- Large datasets are limited by upload and training resource guards
- Joblib model bundles can be incompatible across scikit-learn versions
- Production deployment requires additional authentication, storage isolation, monitoring, and network controls

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and repository-specific contribution guidance.

## License

No LICENSE file is currently included in the repository.

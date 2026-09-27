# CloudInsight AI

CloudInsight AI is a local-first data analytics application built with Streamlit.

It provides a single workspace for loading datasets, cleaning and comparing them, exploring the data, training scikit-learn models, generating PDF reports, and using Google Gemini when AI-assisted processing is needed.

The normal analytics workflow runs locally. Gemini and Amazon S3 are optional integrations used by specific features.

## What you can do

### Ingest data

Load datasets from the local filesystem or browse supported objects in Amazon S3.

The native readers handle:

- CSV
- TSV
- Excel
- JSON
- JSONL / NDJSON
- Parquet
- XML
- HTML tables
- Delimited text

PDF and other unstructured text files are passed through the Gemini conversion flow when they do not contain a native tabular representation.

The uploader accepts multiple files and can merge them into one dataset while keeping the source filename.

### Clean datasets

The Cleaning page can:

- Remove duplicate rows
- Impute missing numeric values with mean, median, or zero
- Impute categorical values with mode or `Unknown`
- Drop rows containing missing values
- Leave missing values unchanged
- Save the cleaned result as a new CSV

### Compare datasets

The Compare page reports:

- Row and column counts
- Common and missing columns
- Missing-value drift
- Numeric distribution shifts
- Duplicate profiles
- Column-level comparison results

### Explore data

The EDA and Visualization pages provide descriptive statistics, column summaries, correlations, outlier analysis, and interactive charts.

### Train machine-learning models

The ML workflow uses scikit-learn pipelines for preprocessing and model training.

It includes:

- Automatic classification/regression detection
- Manual problem-type override
- Numeric and categorical preprocessing
- Missing-value imputation
- Scaling and one-hot encoding
- Train/test evaluation
- Model persistence with Joblib
- Prediction from saved models
- Dataset and model provenance metadata

The current model implementation includes:

- Logistic Regression
- Linear Regression
- Ridge Regression
- Decision Tree Classifier
- Random Forest Classifier
- Random Forest Regressor
- Gradient Boosting Classifier
- Gradient Boosting Regressor

A training-cell limit is enforced to prevent very large in-memory jobs from blocking the Streamlit application.

### Generate PDF reports

The report generator uses ReportLab and can include:

1. Dataset summary and Data Quality Index
2. Column structure and missing-value counts
3. Numerical descriptive statistics
4. Tukey IQR outlier analysis
5. Categorical column analysis
6. Correlation analysis
7. Per-column quality flags
8. Sample records
9. Optional charts
10. Optional Gemini insights

Charts are generated with matplotlib when that optional dependency is installed. Individual chart failures are isolated so they do not have to abort the entire report.

Report templates and batch report generation are supported.

### Use Gemini

Gemini is optional.

The application uses it for:

- Unstructured-file to CSV conversion
- Executive dataset insights
- Dataset chat

The conversion path sends a bounded text sample to Gemini and validates the returned CSV before it is accepted. Prompt input and parsing are bounded to keep malformed model responses from causing unbounded processing.

Gemini requests use typed error handling, timeouts, and bounded retries for transient failures.

### Browse Amazon S3

The S3 integration can list supported data objects and download them into the local `Datasets/` directory.

The helper module also contains CSV upload support, although the current Streamlit UI does not expose an S3 upload action.

## Data flow

```
Local file / S3
      |
      v
Native parser
      |
      +---- supported table ----> DataFrame
      |
      +---- unsupported/unstructured
                         |
                         v
                     Gemini
                         |
                         v
                   validated CSV
                         |
                         v
                      DataFrame
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
     Cleaning         Compare          EDA / Charts
        |                                 |
        +----------------+----------------+
                         |
                    ML / Reports
                         |
                         v
                 local artifacts
```

## Data Quality Index

Dashboard and PDF reports use the same implementation from `Utils/quality.py`:

```
Quality Index = (completeness + uniqueness) / 2
```

Where:

- **Completeness** = percentage of non-missing cells
- **Uniqueness** = percentage of rows that are not duplicates

The two components are weighted equally.

## Requirements

- Python 3.10+
- Windows, macOS, or Linux

A database is not required.

The base `requirements.txt` contains the application's Python dependencies. `pypdf` enables PDF text extraction and `matplotlib` enables report charts.

Gemini and AWS credentials are optional and only required for their respective features.

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

The default Streamlit URL is:

```
http://localhost:8501
```

## Credentials and external services

### Google Gemini

Enter the Gemini API key in the application when using AI conversion, insights, or chat.

Credentials are stored in Streamlit session state rather than written to an environment file by the application. They are released after the operation unless the user chooses to keep them in memory for the session.

Only a bounded representation of the dataset is sent to Gemini. The privacy screen can flag columns that look like they contain emails, phone numbers, financial identifiers, credentials, or tokens so they can be excluded from AI context.

### Amazon S3

Enter AWS access and secret keys in the S3 section of the Ingest page.

The application limits S3 object scanning and rejects downloads above the same 200 MB limit used for local uploads.

## Application pages

| Page | Purpose |
|---|---|
| Ingest data | Local file ingestion, S3 browsing, Gemini conversion |
| Cleaning | Duplicate removal and missing-value handling |
| Compare | Dataset and column-level comparison |
| EDA | Statistics, correlations, and outlier analysis |
| Visualization | Interactive charts |
| Dashboard | Dataset health and summary metrics |
| Machine learning | Train, evaluate, save, load, and predict |
| AI insights | Gemini analysis and dataset chat |
| PDF report | Detailed report generation and templates |

## Project structure

```
CloudInsightAI/
├── App.py                 # Application entry point and navigation
├── Pages/                 # Streamlit pages
├── Utils/                 # Shared processing and integration logic
│   ├── paths.py           # File readers and path safety
│   ├── AIConvert.py       # Gemini-to-DataFrame conversion
│   ├── Gemini.py          # Gemini client and error handling
│   ├── ML.py              # Model training, persistence, prediction
│   ├── PDF.py             # PDF report generation
│   ├── S3.py              # Amazon S3 integration
│   ├── Preprocessing.py   # Cleaning operations
│   ├── compare_logic.py   # Dataset comparison calculations
│   ├── quality.py         # Data Quality Index
│   ├── privacy.py         # Sensitive-column screening
│   ├── secrets.py         # Runtime credential handling
│   └── ...                # UI, charts, logging, batching
├── tests/                 # Automated tests
├── bug_hunt.py            # Headless page boot check
├── Datasets/              # Local datasets; gitignored
├── Models/                # Saved models; gitignored
├── Reports/               # Generated reports; gitignored
└── requirements.txt       # Runtime dependencies
```

## Development

Run the unit test suite:

```bash
python -m unittest discover -s tests -v
```

Run the headless page check:

```bash
python bug_hunt.py
```

Run the serious-error Ruff checks:

```bash
pip install ruff==0.16.4
ruff check --select=E9,F63,F7,F82,F821 --preview .
```

The GitHub Actions workflow runs the test suite, the headless page checks, and the same serious-error lint gate on Python 3.10, 3.11, and 3.12.

## Verification status

The latest repository handoff records:

- **122/122 tests passing**
- **9/9 Streamlit pages booting in the headless check**
- Ruff serious-error checks clean

The headless check is not a substitute for a manual browser smoke test. The repository specifically lists manual verification of the theme toggle and secret-wipe user experience as remaining verification.

## Security and privacy

CloudInsight AI is designed for local, single-user use.

Important limitations and controls:

- There is no built-in authentication.
- Keep Streamlit bound to localhost unless an authentication layer is added.
- Dataset paths are contained within the application's `Datasets/` directory.
- XML files containing DTD/entity declarations are rejected before parsing.
- Upload, parser, S3, chat-history, and ML-training limits are enforced.
- AI-generated CSV output is parsed and validated before entering the application.
- `.joblib` model bundles should only be loaded from trusted sources because deserialization executes Python objects.
- Model artifacts record their source dataset and scikit-learn version.
- Sensitive-column screening can remove likely private fields from Gemini context.
- API credentials are not written to disk by the application's secret handling code.
- Application logs are designed not to include API keys, dataset contents, or raw model responses.

These controls are safeguards for the current local application; they do not turn it into a production multi-user security architecture.

## Known limitations

- The application is local and single-user oriented.
- There is no built-in authentication or per-user storage isolation.
- Gemini and S3 depend on external services and credentials.
- PDF ingestion requires extractable text; scanned image PDFs need OCR outside the current workflow.
- Large datasets are bounded by the application's upload and training limits.
- Saved Joblib models can become incompatible across scikit-learn versions.
- Production deployment would require authentication, isolated storage, monitoring, and a deliberate network configuration.

## License

No LICENSE file is currently included in the repository.

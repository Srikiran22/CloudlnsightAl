# CloudInsight AI

CloudInsight AI is a local-first data analytics application built with Streamlit. It provides a single workspace for ingesting datasets, cleaning and comparing them, exploring the data, training machine-learning models, generating PDF reports, and using Google Gemini for optional AI-assisted analysis.

The application is designed to keep the normal analytics workflow local. External services are only used for features that explicitly require them, such as Gemini analysis or the optional Amazon S3 browser.

## Features

### Data ingestion

Supported formats include:

- CSV
- Excel
- JSON and nested JSON
- JSONL / NDJSON
- TSV
- Parquet
- XML
- HTML tables
- Delimited text
- PDF, when `pypdf` is installed

Structured formats are parsed locally first. Gemini is used as a fallback for files that cannot be handled by the native parsers.

### Data preparation

- Duplicate-row removal
- Missing-value handling
- Before/after data-quality summaries
- Cleaned dataset export
- Multi-file ingestion with source tracking

### Dataset comparison

Compare two datasets for:

- Schema differences
- Missing-value changes
- Duplicate profiles
- Numeric mean shifts
- Other basic distribution differences

### Exploratory analysis

The application provides:

- Column and data-type summaries
- Descriptive statistics
- Correlation analysis
- IQR-based outlier detection
- Interactive visualizations
- Distribution plots
- Categorical analysis

### Machine learning

The ML page supports classification and regression workflows using scikit-learn.

The application includes:

- Automatic problem-type detection
- Preprocessing pipelines
- Multiple classification and regression algorithms
- Model evaluation
- Model persistence
- Prediction
- Exportable results
- Model metadata and scikit-learn version tracking

Preprocessing is kept inside the scikit-learn pipeline so transformations are fitted only on the relevant training data.

### Reports

PDF reports are generated with ReportLab and can include:

- Dataset summary
- Data Quality Index
- Column information
- Numerical statistics
- Outlier analysis
- Categorical analysis
- Correlation analysis
- Data-quality flags
- Sample records
- Optional charts
- Optional Gemini-generated insights

Report templates can be saved and reused, and reports can also be generated in batch.

### Gemini integration

Gemini is optional and is used for:

- Converting unsupported or unstructured files into tabular data
- Executive dataset analysis
- Dataset chat

The Gemini integration includes request timeouts, bounded retries for transient failures, model selection, response validation, and limits on the amount of data sent to the model.

### Amazon S3

The application can browse datasets stored in Amazon S3. AWS credentials are requested only when the S3 feature is used.

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- No database is required
- Gemini and AWS credentials are optional and only needed for their respective features

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

Install the dependencies:

```bash
pip install -r requirements.txt
```

`pypdf` enables PDF input and `matplotlib` enables charts in generated PDF reports.

## Run the application

```bash
streamlit run App.py
```

By default, Streamlit starts the application at:

```
http://localhost:8501
```

## Credentials

The application does not require a `.env` file.

Credentials are entered through the application when a feature needs them.

### Google Gemini

A Gemini API key is required only for:

- AI-assisted file conversion
- Executive insights
- Dataset chat

The key is held in server-side session memory and is cleared after completed API tasks unless the user chooses to keep it for the current session.

### Amazon S3

AWS credentials are required only for the S3 browser.

### Privacy

When Gemini is used, the application sends a bounded representation of the dataset to the external model service. Sensitive columns can be detected and excluded before AI analysis.

Do not upload confidential or personally identifiable data to external services unless the data handling requirements of your environment allow it.

## Application pages

| Page | Purpose |
|---|---|
| Ingest data | Load local files or browse S3 datasets |
| Cleaning | Remove duplicates and handle missing values |
| Compare | Compare schemas and dataset characteristics |
| EDA | Explore columns, statistics, correlations, and outliers |
| Visualize | Create interactive charts |
| Dashboard | View dataset health and summary metrics |
| Machine learning | Train, evaluate, save, and reuse models |
| AI insights | Generate Gemini-based analysis and chat with the dataset |
| PDF report | Create detailed reports and reusable report templates |

## Data Quality Index

The Dashboard and PDF report use the same Data Quality Index implementation.

```
Quality Index = (completeness + uniqueness) / 2
```

Where:

- **Completeness** is the percentage of non-empty cells.
- **Uniqueness** is the percentage of distinct rows.

The calculation is intentionally simple so that the score is consistent and easy to interpret.

## Project structure

```
CloudInsightAI/
├── App.py                 # Application entry point
├── Pages/                 # Streamlit pages
├── Utils/                 # Shared application logic
│   ├── paths.py           # File readers and path validation
│   ├── AIConvert.py       # AI-assisted table conversion
│   ├── Gemini.py          # Gemini integration
│   ├── quality.py         # Data Quality Index
│   ├── compare_logic.py   # Dataset comparison logic
│   ├── PDF.py             # PDF report generation
│   ├── S3.py              # Amazon S3 integration
│   ├── ML.py              # ML utilities
│   ├── Preprocessing.py   # Data preprocessing
│   └── ...                # Logging, secrets, UI, charts, and helpers
├── tests/                 # Automated tests
├── Models/                # Saved models; gitignored
├── Datasets/              # Uploaded datasets; gitignored
├── Reports/               # Generated reports; gitignored
├── bug_hunt.py            # Headless application boot check
└── requirements.txt        # Python dependencies
```

## Development

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run the headless page check:

```bash
python bug_hunt.py
```

Run Ruff's serious-error checks:

```bash
pip install ruff
ruff check --select=E9,F63,F7,F82,F821 --preview .
```

The CI workflow runs the test suite and application boot checks across supported Python versions.

## Logging

Logs are written to the terminal that launched Streamlit.

The default level is `WARNING`. To enable more detailed logging:

### Windows

```powershell
$env:CLOUDINSIGHT_LOG_LEVEL="INFO"
```

### macOS / Linux

```bash
export CLOUDINSIGHT_LOG_LEVEL=INFO
```

API keys, dataset contents, and raw model responses are not written to application logs.

## Security considerations

The application is intended primarily for local, single-user use.

Important considerations:

- There is no built-in authentication.
- Keep the application bound to localhost unless an appropriate authentication layer is added.
- Dataset paths are restricted to the application's dataset directory.
- XML files containing DTD/entity declarations are rejected.
- Upload and parser limits prevent unbounded resource use.
- AI responses are validated before converted data enters the application.
- Saved `.joblib` files should only be loaded when their source is trusted.
- ML artifacts record relevant provenance and scikit-learn version information.
- Sensitive columns can be excluded from Gemini context.

These controls improve the application's safety but should not be treated as a substitute for a production security architecture.

## Known limitations

- The application is designed for local single-user use.
- There is no built-in authentication or multi-user isolation.
- Gemini and S3 features depend on external services.
- Very large datasets are constrained by upload and processing limits.
- Model bundles can have compatibility issues across scikit-learn versions.
- Production deployment requires additional storage isolation, authentication, and operational controls.

## License

This repository is currently intended as a private portfolio project and does not include a redistribution license.

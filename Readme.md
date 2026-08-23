# CloudInsight AI

**Any file in. Clean insights out.**

CloudInsight AI is a local-first data analytics studio built with Streamlit, Pandas, Scikit-Learn, Plotly, ReportLab, and Google Gemini. Upload virtually any data file — structured or not — and go from raw content to ML models and publication-ready PDF reports without writing code.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-red)

---

## Highlights

- **Universal ingestion** — CSV, Excel, JSON (nested), JSONL/NDJSON, TSV, Parquet, XML, HTML tables, delimited text, PDFs (via optional `pypdf`)
- **AI-powered structuring** — unstructured files (logs, plain text, PDF content) are converted into clean tables by Google Gemini
- **One-click cleaning** — duplicate removal + smart missing-value imputation with before/after deltas
- **Dataset comparison** — schema diffs, missing-value drift, and numeric mean-shift flags between any two datasets
- **ML studio** — automated classification/regression detection, 8 algorithms, model persistence (save/load/predict), prediction exports
- **Executive PDF reports** — up to 10 detailed sections: quality scoring, extended statistics, outlier bounds, categorical & correlation analysis, charts (optional `matplotlib`), page numbering
- **Report templates & batch mode** — save report configurations; generate reports for every dataset in one click
- **Gemini intelligence** — executive data audits and conversational Q&A over your dataset
- **Fast by default** — cached dataset loading keyed by file path + modification time

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- No database, no cloud account required. Gemini/AWS keys are strictly optional per-feature.

## Getting Started

### 1. Install dependencies

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

All of `requirements.txt` is safe to install; `pypdf` and `matplotlib` unlock PDF ingestion and report charts respectively, and the app degrades gracefully without them.

### 2. Run the application

```bash
streamlit run App.py
```

The app opens at `http://localhost:8501`.

### 3. API keys (entered at runtime — never on disk)

There is **no `.env` file and no key configuration file**. When a feature needs credentials, the app asks for them in the UI:

- A Google Gemini API key ([get one free at AI Studio](https://aistudio.google.com/)) is needed only for file conversion, executive insights, and chat.
- AWS access + secret keys are needed only for the S3 browser.

Keys live in server-side session memory only. After each completed API task they are wiped automatically unless you tick "Keep in memory for this session". You can always clear them with the sidebar button.

> **Privacy note:** AI features send a compact dataset summary (bounded to ~12k characters) to Google Gemini. Use de-identified data for sensitive sources.

## Feature Tour

| Page | What it does |
|------|--------------|
| **Ingest data** | Local files or S3; native parsing first, Gemini conversion as fallback; multi-file merge with source tracking |
| **Cleaning** | Remove duplicates, impute/drop missing values, download cleaned CSVs |
| **Compare** | Side-by-side dataset diff: schema, drift, duplication profiles |
| **EDA** | Column audit, descriptive stats, correlation matrix, IQR outlier detection |
| **Visualize** | Histograms, box/violin plots, scatter/bubble, bar/pie/treemap, heatmaps |
| **Dashboard** | Data-quality gauge, dynamic filtering, quick-look distributions |
| **Machine learning** | Auto problem detection, train/evaluate 8 algorithms, save & reuse models |
| **AI insights** | Automated executive reports + chat with your dataset |
| **PDF report** | Detailed multi-section reports, reusable templates, batch generation |

## The Data Quality Index

Both the Dashboard gauge and the PDF report use ONE shared formula:

```
Quality Index = (completeness + uniqueness) / 2
```

where *completeness* is the share of non-empty cells (%) and *uniqueness* is the share of distinct rows (%). Equal weights are deliberate — symmetric and easy to explain.

## What's inside the generated PDF report?

1. High-level summary with the composite Data Quality Index
2. Complete column structure with unique-value counts
3. Extended numerical statistics (quartiles, skewness, kurtosis)
4. Full Tukey-IQR outlier audit with bounds
5. Categorical column analysis (top values, frequencies)
6. Strongest correlation pairs (|r| ≥ 0.30)
7. Per-column quality flags (high missingness, constants, heavy outliers)
8. Sample records
9. Distribution charts: histograms, box plots, correlation heatmap *(requires matplotlib; each chart fails independently without breaking the report)*
10. Optional Gemini executive insights

Sections renumber automatically when charts or insights are omitted.

## Machine learning notes

- Preprocessing (imputation, scaling, one-hot encoding) lives **inside** the sklearn Pipeline and is fitted on training folds only — no data leakage.
- Problem type is auto-detected but overridable: non-numeric targets are classification; integer-like targets with ≤10 distinct values covering ≤half the rows are treated as class labels.
- Models persist as `.joblib` bundles in `Models/` with metadata (algorithm, target, features, source dataset, metrics, sklearn version, creation time). Loading warns if the saved scikit-learn version differs from the running one.

## Project Structure

```
CloudInsightAI/
├── App.py               # Entry point & navigation
├── Pages/               # One Streamlit page per feature (UI orchestration)
├── Utils/               # Reusable logic
│   ├── paths.py         # Universal format readers + path safety
│   ├── AIConvert.py     # LLM → DataFrame conversion pipeline
│   ├── Gemini.py        # Gemini client wrapper (errors/timeouts/models)
│   ├── quality.py       # Shared Data Quality Index
│   ├── compare_logic.py # Dataset diff math
│   ├── logsys.py        # Logging setup
│   └── ...              # ML, PDF, Charts, S3, secrets, theme, batch
├── tests/               # Unit tests (unittest suite)
├── bug_hunt.py          # Headless boot check of every page
├── Datasets/            # Your uploaded files (gitignored)
├── Models/              # Saved ML models (gitignored)
├── Reports/             # Generated PDFs & templates (gitignored)
└── AI/                  # Agent memory system for AI coding assistants
```

## Development

```bash
# Run the full unit test suite
python -m unittest discover -s tests -v

# Headless boot-check of every app page
python bug_hunt.py

# Optional: serious-error static checks (same gate as CI)
pip install ruff && ruff check --select=E9,F63,F7,F82,F821 --preview .
```

### Logs

The app logs to the terminal where you ran `streamlit run App.py`. Default level is `WARNING`; set more detail with:

```bash
set CLOUDINSIGHT_LOG_LEVEL=INFO      # Windows (export on macOS/Linux)
```

Secrets, dataset contents, and raw model responses are never logged.

### CI

GitHub Actions runs the unit tests plus the headless page-boot check on Python 3.10–3.12, with a ruff pass limited to serious errors (undefined names, syntax-level mistakes).

## Security Notes

- **Local-first by design.** There is no authentication. Anyone who can reach port 8501 can use the app and its stored state — keep it bound to localhost (`streamlit run App.py` defaults to localhost) and do not expose it to a network without adding an authenticating reverse proxy.
- **Secrets are runtime-only**: entered in the UI, held in server-side memory, wiped after each task unless kept for the session. Nothing touches disk; nothing is logged.
- **Path containment**: dataset reads/writes are restricted to the project's `Datasets/` directory (traversal attempts rejected).
- **Model bundles**: `.joblib` files contain executable code — load only bundles you trained or trust. The app displays this warning wherever models are loaded.
- **AI output is sandboxed**: converted CSVs are validated before entering the app; insights/chat render as text and are never executed; prompts wrap untrusted content in guard tags as defense-in-depth.

## Deployment

This app targets local single-user use. If you later deploy it:

1. Put it behind an authenticating proxy (SSO/OAuth) — the app itself has none.
2. Use HTTPS at the proxy.
3. Provide per-user storage isolation; today all users would share one `Datasets/` folder and server-side session state is per-browser-session.
4. Set `server.address` / firewall rules deliberately; Streamlit's default binding is localhost.

See `AI/HANDOFF.md` for architecture context if you plan to extend it.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| PDF upload says pypdf required | `pip install pypdf` |
| PDF reports have no charts | `pip install matplotlib`, regenerate |
| Gemini errors about auth/model | Check the key; try another model from the dropdown |
| "rate limit" messages | Wait a moment; the app already retries transient failures twice |
| S3 listing empty | Verify bucket region matches the AWS Region field and keys allow ListBucket |
| Theme toggle does nothing | Your Streamlit version dropped the runtime hook — switch via Settings menu |
| Slow huge uploads | Files beyond ~200 MB are refused; split or sample first |

---

*Private project — all rights reserved. Not licensed for redistribution.*

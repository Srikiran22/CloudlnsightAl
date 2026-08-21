# ☁️ CloudInsight AI

**Any file in. Clean insights out.**

CloudInsight AI is a local-first data analytics studio built with Streamlit, Pandas, Scikit-Learn, Plotly, ReportLab, and Google Gemini. Upload virtually any data file — structured or not — and go from raw content to ML models and publication-ready PDF reports without writing code.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-red)

---

## ✨ Highlights

- **Universal ingestion** — CSV, Excel, JSON (nested), JSONL, TSV, Parquet, XML, HTML tables, delimited text, PDFs
- **AI-powered structuring** — unstructured files (logs, plain text, PDF content) are converted into clean tables by Google Gemini
- **One-click cleaning** — duplicate removal + smart missing-value imputation with before/after deltas
- **Dataset comparison** — schema diffs, missing-value drift, and numeric mean-shift flags between any two datasets
- **ML studio** — automated classification/regression, 8 algorithms, model persistence (save/load/predict), prediction exports
- **Executive PDF reports** — 10 detailed sections: quality scoring, extended statistics, outlier bounds, categorical & correlation analysis, charts, page numbering
- **Report templates & batch mode** — save report configurations; generate reports for every dataset in one click
- **Gemini intelligence** — executive data audits and conversational Q&A over your dataset
- **Fast by default** — cached dataset loading keyed by file path + modification time

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional extras that unlock more features:

```bash
pip install pypdf matplotlib   # PDF ingestion + report charts
```

### 2. Configure your Gemini key (optional but recommended)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

Get a free key at [Google AI Studio](https://aistudio.google.com/). You can also paste the key directly into the app sidebar at runtime.

> **Privacy note:** AI features send a compact dataset sample (bounded to ~12k characters) to Google Gemini. Use de-identified data for sensitive sources.

### 3. Run the application

```bash
streamlit run App.py
```

The app opens at `http://localhost:8501`.

---

## 🧭 Feature Tour

| Page | What it does |
|------|--------------|
| 📂 **Upload** | Ingest any supported file; native parsing first, Gemini conversion as fallback |
| 🧹 **Cleaning** | Remove duplicates, impute/drop missing values, download cleaned CSVs |
| ⚖️ **Compare** | Side-by-side dataset diff: schema, drift, duplication profiles |
| 📊 **EDA** | Column audit, descriptive stats, correlation matrix, IQR outlier detection |
| 📈 **Visualizations** | Histograms, box/violin plots, scatter/bubble, bar/pie/treemap, heatmaps |
| 🤖 **Machine Learning** | Auto problem detection, train/evaluate 8 algorithms, save & reuse models |
| 💡 **AI Insights** | Automated executive reports + chat with your dataset |
| 📄 **PDF Report** | Detailed multi-section reports, reusable templates, batch generation |
| 🎛️ **Dashboard** | Data-quality gauge, dynamic filtering, quick-look distributions |

## 📄 What's inside the generated PDF report?

1. High-level summary with a composite Data Quality Index
2. Complete column structure with unique-value counts
3. Extended numerical statistics (quartiles, skewness, kurtosis)
4. Full Tukey-IQR outlier audit with bounds
5. Categorical column analysis (top values, frequencies)
6. Strongest correlation pairs (|r| ≥ 0.30)
7. Per-column quality flags (high missingness, constants, heavy outliers)
8. Sample records
9. Distribution charts: histograms, box plots, correlation heatmap
10. Optional Gemini executive insights

---

## 🗂️ Project Structure

```
CloudInsightAI/
├── App.py               # Entry point & navigation
├── Pages/               # One Streamlit page per feature
├── Utils/               # Reusable logic (readers, ML, PDF, Gemini, S3)
│   ├── paths.py         # Universal format readers
│   ├── AIConvert.py     # LLM → DataFrame conversion pipeline
│   └── ...
├── tests/               # Unit tests (pytest-compatible unittest suite)
├── Datasets/            # Your uploaded files (gitignored)
├── Models/              # Saved ML models (gitignored)
├── Reports/             # Generated PDFs & templates (gitignored)
└── AI/                  # Agent memory system for AI coding assistants
```

## 🧪 Development

```bash
# Run the full unit test suite (24 tests)
python -m unittest discover -s tests

# Headless boot-check of every app page
python bug_hunt.py
```

The `AI/` folder implements a vendor-neutral handoff protocol so different AI coding agents (Codex, Cursor, Claude, Gemini CLI, local LLMs…) can collaborate on this repository without losing context. See `AGENTS.md` for the protocol.

## 🔒 Security Notes

- Never commit your `.env` file — it is gitignored by default
- AWS credentials entered in the Upload page live only in session state
- Model bundles (`Models/*.joblib`) should only be loaded from trusted sources

---

*Private project — all rights reserved. Not licensed for redistribution.*

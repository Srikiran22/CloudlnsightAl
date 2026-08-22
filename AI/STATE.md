# Project State

## Current Phase

Feature-rich platform + hardened secret handling: runtime-only Gemini/AWS credentials, de-AI restyled sources.

## Working Features

- Secret lifecycle (`Utils/secrets.py`): runtime password entry, optional per-session keep flag, automatic wipe after each completed API task (conversion / insights / chat / S3 fetch+download), explicit clear controls; nothing written to disk
- Universal file ingestion (`Pages/Upload.py`, `Utils/paths.py`): CSV, Excel, JSON (nested), JSONL/NDJSON, TSV, Parquet, XML, HTML tables, delimited TXT — all native parsers
- Gemini-powered conversion of unstructured files via `Utils/AIConvert.py`; saved as `<stem>_converted.csv`
- PDF input via optional `pypdf`, then AI structuring
- Detailed PDF reports (`Utils/PDF.py`): 10 sections incl. quality scores, extended stats, categorical/correlation analysis, histograms + box plots + heatmap (optional matplotlib), page footers
- Cross-format S3 sync (`Utils/S3.py`)
- Dataset comparison (`Pages/Compare.py`), ML persistence (`Utils/ML.py` + `Pages/ML.py`), report templates/batch (`Pages/Report.py`), cached loading (`Utils/dataset_ui.py`)
- Data cleaning, EDA, visualization studio, ML studio, Gemini chat, executive dashboard

## Incomplete Features

- Readme not yet updated for new formats/features

## Known Bugs

- None known. 24/24 unit tests pass; 9/9 pages verified booting via AppTest.

## Important Files

- `Utils/secrets.py` — runtime credential entry / release / drop (memory-only)
- `Utils/paths.py` — native readers, AIConversionRequired, MODELS_DIR, REPORT_TEMPLATES_DIR
- `Utils/AIConvert.py` — Gemini conversion pipeline
- `Utils/ML.py` — training + save/load/predict helpers
- `Pages/Compare.py` — dataset diff page
- `Utils/PDF.py` — detailed report generator with 3 chart types
- `Pages/Report.py` — templates + batch mode
- `AI/` — agent handoff/memory system

## Recent Changes

- 2026-08-22: Runtime secret lifecycle (DEC-009) wired into Upload + AI pages; env seeding and python-dotenv removed
- 2026-08-22: De-AI restyle of all Python sources with frozen public surface (DEC-010)
- 2026-08-21: Six-feature build; memory system rename; universal ingestion + detailed PDF rewrite

## Tests

- 24 tests in `tests/test_core.py`: ingestion formats, AI CSV parsing (incl. prose-with-commas regression), PDF render, S3 multi-format listing, model save/load/predict round-trip
- Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests`
- Page boot check: `& .\venv\Scripts\python.exe bug_hunt.py` (AppTest boots all 9 pages headlessly)

## Build Status

OK — 24/24 unit tests pass; 9/9 pages boot without exceptions (verified after restyle + secret changes).

## Environment

- **OS:** Windows / PowerShell 5.1
- **Python:** venv at `./venv`
- **Key Dependencies:** Streamlit, Pandas, Scikit-Learn (incl. joblib), Plotly, ReportLab, Google Gemini; pyarrow present; pypdf + matplotlib declared but not yet installed in venv

## Dependencies

See `requirements.txt`.

## Risks

- matplotlib/pypdf not yet pip-installed in venv — charts and PDF input degrade gracefully until installed
- Model bundles are scikit-learn-version-sensitive when loading
- AI conversion sends file content to Google Gemini (privacy notice shown in UI)
- With the keep-box unticked, users must re-enter credentials per task (intentional)

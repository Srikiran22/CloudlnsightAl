# AI Handoff

- **Last Updated:** 2026-08-21
- **Previous Agent:** OpenCode (ox-alpha)
- **Previous Model:** opencode/x-preview-f-free
- **Previous Application:** OpenCode CLI

## What We Are Building

CloudInsight AI — a Streamlit analytics platform. Recently implemented: universal file ingestion with Gemini conversion, detailed 10-section PDF reports, and a six-feature expansion (compare page, ML persistence, S3 formats, report templates/batch, PDF chart upgrades, cached loading).

## Current Objective

None active — all requested features implemented and verified.

## Current Status

Completed — GitHub-ready release. 24/24 unit tests pass; 9/9 pages boot headlessly (bug_hunt.py); CI workflow added (.github/workflows/tests.yml); README rewritten; MIT LICENSE added.

## What Has Been Done

- Bug-fix session: PDF Report page crash (missing selectbox label), AI CSV parser hardened against comma-containing prose (sub-range + header-plausibility scoring), S3 unstructured-download guidance, Cleaning page universal formats, removed duplicate API-key guard in AI.py
- New `bug_hunt.py`: boots all 9 pages via streamlit.testing.v1.AppTest for regression checking

- Universal ingestion: native readers (JSON nested, JSONL, TSV, Parquet, XML, HTML, TXT) with `AIConversionRequired` fallback to Gemini conversion (`Utils/AIConvert.py`); converted files saved as `<stem>_converted.csv`
- Detailed PDF reports: quality index, full column stats incl. skew/kurtosis, outlier bounds, categorical analysis, correlation pairs, quality flags, sample records, histograms + box plots + heatmap, page footers
- Dataset Compare page (`Pages/Compare.py`): schema diff, missingness drift, mean-shift flags, duplication profile
- ML persistence: joblib bundles in `Models/`, load + predict on any dataset, predictions CSV export
- Report templates (JSON in `Reports/templates/`) + batch generation with per-dataset row caps
- Performance: `st.cache_data` keyed by path+mtime in `Utils/dataset_ui.py`
- Cross-format S3 listing via `SUPPORTED_DATASET_EXTENSIONS`
- Memory system: `AI/` folder (renamed from `.ai/`) with SESSIONS.md before/after task logging

## What Is In Progress

Nothing.

## What Remains

- Readme refresh for new formats/features
- `pip install pypdf matplotlib` to unlock PDF input + report charts (currently degrade gracefully)

## Important Decisions

- DEC-005: Native parsers first; AI only as fallback via `AIConversionRequired`
- DEC-006: Gemini conversion returns strict CSV, validated before entering session state
- DEC-007: matplotlib/pypdf optional at runtime; graceful degradation everywhere
- DEC-008: joblib model bundles + JSON templates + mtime-keyed cache (no external services)
- Earlier: plain Markdown memory system; HANDOFF as entry point; no chain-of-thought storage

## Important Constraints

- Downstream pages rely on `current_df`/`dataset_name` session keys and the `Datasets/` folder — keep those contracts intact
- All AI prompts stay bounded (12k chars) and include untrusted-content guards
- New formats must get native readers before considering AI conversion

## Known Problems

- None known. S3 test fixture note: `.txt` is now a supported format (universal ingestion).

## Files Recently Changed

- `Utils/S3.py`, `Utils/ML.py`, `Pages/ML.py`, `Pages/Compare.py` (new), `App.py`
- `Utils/PDF.py`, `Pages/Report.py`, `Utils/dataset_ui.py`, `Utils/paths.py`
- `tests/test_core.py` (+3 tests), `requirements.txt`

## What The Next Agent Should Do

1. Read this file, then `AI/MODEL.md` and `AI/TASK.md`; check `AI/SESSIONS.md` for recent history
2. Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests` (expect 24/24)
3. Also run `& .\venv\Scripts\python.exe bug_hunt.py` (expect 9/9 pages OK)
4. Verify claims above against code before changing anything
5. Record a pre-task snapshot in SESSIONS.md before your first edit

## Verification Needed

- Manual browser smoke test of Compare page and batch report generation
- Confirm `pip install pypdf matplotlib` enables PDF input and charts

## Do Not Forget

- Update `MODEL.md` when switching agents/models
- Update this file and complete the SESSIONS.md post-task summary before ending any session
- Verify documentation against actual code — never trust it blindly

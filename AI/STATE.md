# Project State

## Current Phase

Hardened local-first analytics platform: unified quality index, bounded AI-CSV parsing, classified Gemini errors, structured logging, 71-test suite.

## Working Features

- Secret lifecycle (`Utils/secrets.py`): runtime password entry, optional per-session keep flag, automatic wipe after each completed API task, explicit clear controls; nothing written to disk
- Universal file ingestion (CSV/Excel/JSON/JSONL/TSV/Parquet/XML/HTML/TXT native; PDF via optional pypdf; Gemini fallback) with 200MB guard and duplicate-name disambiguation
- Gemini pipeline (`Utils/Gemini.py`): typed GeminiError kinds, explicit request timeouts on both SDK paths (client-level ms timeout for google-genai; per-request seconds for legacy; loud logged fallback if an SDK version cannot take one), retries only for transient failures, model registry; chat history capped at 100 messages
- AI conversion (`Utils/AIConvert.py`): bounded parser (4k lines / 1MB / 2k candidates), O(n) fast path for clean responses, true column cap
- Shared Data Quality Index (`Utils/quality.py`): equal blend, used by Dashboard AND PDF
- PDF reports (`Utils/PDF.py`): chart-failure isolation, empty-column safety, truthful numbering; charts verified with matplotlib installed
- S3 sync with plain-language error mapping; Compare page math extracted to `Utils/compare_logic.py`
- ML studio: provenance metadata in results + joblib bundles, sklearn-version mismatch warning, trust caption
- Logging (`Utils/logsys.py`): console logs, `CLOUDINSIGHT_LOG_LEVEL`, no secrets/data logged

## Known Bugs

None confirmed. XML entity-expansion remains a theoretical self-upload DoS (stdlib parser, accepted). New-SDK google-genai path is untested live because this venv ships the legacy package only.

## Important Files

- `Utils/logsys.py` — logging setup (get_logger)
- `Utils/quality.py` — canonical Data Quality Index (DEC-011)
- `Utils/compare_logic.py` — testable dataset-diff math
- `Utils/Gemini.py` — model registry + GeminiError taxonomy + retry policy
- `Utils/AIConvert.py` — bounded LLM→CSV parsing (algorithm documented in docstring)
- `Utils/secrets.py` — memory-only credential lifecycle
- `Utils/paths.py` — readers, path containment, MAX_UPLOAD_BYTES
- `tests/test_core.py` — 93 tests; `bug_hunt.py` — 9-page boot check

## Recent Changes

- 2026-08-23: Post-audit targeted fix pass: google-genai client-level ms timeout (legacy keeps per-request seconds; any unsupported-timeout fallback is loud, never silent); API-key redaction in Gemini diagnostics + real leak test; text-ingestion policy rewritten (csv.Sniffer over explicit separators only, ghost-column guard) so prose reaches AI conversion; cache no longer keyed by max_rows; ruff pinned in CI; docs corrected
- 2026-08-23: Comprehensive remediation pass (36-phase audit response): all items in SESSIONS.md post-task summary; DEC-011/DEC-012 recorded
- 2026-08-22: Runtime secret lifecycle wired; de-AI restyle; universal ingestion + detailed PDFs

## Tests

- 93 tests in `tests/test_core.py`: ingestion formats + oversize rejection, AI CSV parsing incl. pathological inputs, Gemini error classification + prompt bounds, quality index math + single-source pinning, preprocessing strategies, compare drift logic, PDF edge cases (0 columns, chart failure), ML persistence/provenance/version flags, S3 error mapping, batch merge, secrets, session-state contract
- Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests`
- Boot check: `& .\venv\Scripts\python.exe bug_hunt.py`
- Static gate: `& .\venv\Scripts\python.exe -m ruff check --select=E9,F63,F7,F82,F821 --preview .`

## Build Status

Verified 2026-08-23: 71/93 tests pass; bug_hunt boots 9/9 pages; ruff clean; e2e workflow script passed all 10 scenarios (CSV/Excel/JSON/text/PDF ingest, merge, cleaning, drift, ML cls+reg+persist+predict, PDF-with-charts 48KB).

## Environment

- **OS:** Windows / PowerShell 5.1
- **Python:** venv at `./venv` (streamlit 1.60, sklearn 1.9, pandas 3.0.5)
- **Key Dependencies:** see requirements.txt (annotated); matplotlib + pypdf NOW INSTALLED locally
- **Note:** venv has legacy google-generativeai, not google-genai (see DEC-012)

## Dependencies

See annotated `requirements.txt`. statsmodels retained deliberately (Plotly OLS trendline feature).

## Risks

- Model bundles execute pickle code on load — trust requirement documented in UI caption + README
- AI conversion sends file content to Google Gemini (privacy notice shown)
- Local-first app has no authentication — network exposure unsafe (documented in README deployment section)

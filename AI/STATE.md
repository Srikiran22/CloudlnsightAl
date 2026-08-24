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
- `tests/test_core.py` — 117 tests; `bug_hunt.py` — 9-page boot check

## Recent Changes

- 2026-08-24 (2nd): Deep review pass: XML DTD guard now scans whole payload (64KB window was bypassable via comment-padded DOCTYPE — mutation-proven); Utils.paths.safe_stem shared sanitizer; Report template names sanitized; S3 downloads capped by MAX_UPLOAD_BYTES via ContentLength; select_working_dataset matches session option positionally (file named "Active Session ..." no longer hijacks); compare_logic docstring corrected. +5 tests -> 122.
- 2026-08-24: CI repair: tests/test_core.py auth-retry test no longer hard-imports google.api_core (absent on CI; google-genai does not provide it) — try-import with stub fallback; workflow gains permissions contents:read, concurrency dedup, timeout-minutes 30. Verified: poisoned-env repro fails-before/passes-after, 117/117, ruff clean, bug_hunt 9/9.
- 2026-08-24: Final hardening pass: XML DTD/entity rejection + deep-nest guard (billion-laughs verified vulnerable then fixed); Gemini SDK-internal retries disabled on google-genai (retry_options attempts=1) with jittered project backoff as sole retry owner; legacy SDK retry-disable attempted with loud degradation; Utils/privacy.py sensitive-column screening wired into AI insights/chat context exclusion; ML training cell-limit + extracted _non_finite_columns; PDF quality flags extracted to testable _quality_flags_for_column; dataset fingerprints (name+size+mtime hash) gate stale ML/report results; cache bounded max_entries=64; DEC-013 records structured-output evaluation
- 2026-08-23: Post-audit targeted fix pass: google-genai client-level ms timeout (legacy keeps per-request seconds; any unsupported-timeout fallback is loud, never silent); API-key redaction in Gemini diagnostics + real leak test; text-ingestion policy rewritten (csv.Sniffer over explicit separators only, ghost-column guard) so prose reaches AI conversion; cache no longer keyed by max_rows; ruff pinned in CI; docs corrected
- 2026-08-23: Comprehensive remediation pass (36-phase audit response): all items in SESSIONS.md post-task summary; DEC-011/DEC-012 recorded
- 2026-08-22: Runtime secret lifecycle wired; de-AI restyle; universal ingestion + detailed PDFs

## Tests

- 122 tests in `tests/test_core.py`: ingestion formats + oversize rejection, AI CSV parsing incl. pathological inputs, Gemini error classification + prompt bounds, quality index math + single-source pinning, preprocessing strategies, compare drift logic, PDF edge cases (0 columns, chart failure), ML persistence/provenance/version flags, S3 error mapping + size cap, batch merge, secrets, session-state contract, XML DTD window-bypass regression
- Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests`
- Boot check: `& .\venv\Scripts\python.exe bug_hunt.py`
- Static gate: `& .\venv\Scripts\python.exe -m ruff check --select=E9,F63,F7,F82,F821 --preview .`

## Build Status

Verified 2026-08-23: 71/117 tests pass; bug_hunt boots 9/9 pages; ruff clean; e2e workflow script passed all 10 scenarios (CSV/Excel/JSON/text/PDF ingest, merge, cleaning, drift, ML cls+reg+persist+predict, PDF-with-charts 48KB).

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

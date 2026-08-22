# AI Handoff

- **Last Updated:** 2026-08-22
- **Previous Agent:** OpenCode (ox-alpha)
- **Previous Model:** opencode/x-preview-f-free
- **Previous Application:** OpenCode CLI

## What We Are Building

CloudInsight AI — a Streamlit analytics platform. Recently implemented: universal file ingestion with Gemini conversion, detailed 10-section PDF reports, six-feature expansion, runtime-only secret handling for Gemini/AWS, and a full de-AI restyle of the Python sources.

## Current Objective

None active — all requested features implemented and verified.

## Current Status

Completed and verified: 24/24 unit tests pass; 9/9 pages boot headlessly (bug_hunt.py). Secrets are now runtime-entered and wiped from memory after each completed API task unless the user opts to keep them.

## What Has Been Done (latest session)

- New `Utils/secrets.py`: `ask()` renders password fields backed by session slots; `release(*names, keep_key=...)` wipes after a finished task unless "Keep in memory for this session" was ticked; `drop()` clears immediately; nothing ever written to disk
- `Pages/AI.py`: key entered in sidebar; wiped after each successful insights/chat call (toast confirmation); explicit clear button
- `Pages/Upload.py`: Gemini key wiped after successful conversion; AWS access/secret wiped after fetch/download; re-entry prompt shown when files are listed but keys were cleared; removed legacy `aws_key`/`aws_secret` session writes that duplicated secrets
- Removed env-var secret seeding (`GEMINI_API_KEY`) and the python-dotenv auto-load + dependency
- De-AI restyle of all Python files per DEC-010: formulaic docstrings/banners stripped, terse lowercase comments only where logic is subtle, type hints thinned, minor internal consolidations (`_datetime_to_epoch`, `_matplotlib`); public APIs, kwargs, prompts, UI strings, error messages untouched
- Earlier sessions: universal ingestion, 10-section PDF reports, Compare page, ML persistence, templates/batch, cached loading, memory system

## What Is In Progress

Nothing.

## What Remains

- Readme refresh for new formats/features
- `pip install pypdf matplotlib` to unlock PDF input + report charts (currently degrade gracefully)

## Important Decisions

- DEC-009: Secrets are runtime-entered, memory-only, wiped after each completed API task unless the user keeps them
- DEC-010: De-AI restyle with frozen public surface (APIs/kwargs/prompts/UI strings/error messages)
- DEC-005: Native parsers first; AI only as fallback via `AIConversionRequired`
- DEC-006: Gemini conversion returns strict CSV, validated before entering session state
- DEC-007: matplotlib/pypdf optional at runtime; graceful degradation everywhere
- DEC-008: joblib model bundles + JSON templates + mtime-keyed cache (no external services)

## Important Constraints

- Downstream pages rely on `current_df`/`dataset_name` session keys and the `Datasets/` folder — keep those contracts intact
- All AI prompts stay bounded (12k chars) and include untrusted-content guards
- New formats must get native readers before considering AI conversion
- Keep the restyle discipline: no formulaic docstrings/banners; check tests/pages before any rename
- Never reintroduce env-file secret seeding or disk persistence of credentials

## Known Problems

- None known. Pre-existing cosmetic Arrow warning on Compare page (mixed-type Mean columns) is auto-fixed by Streamlit.

## Files Recently Changed

- `Utils/secrets.py` (new), `Pages/AI.py`, `Pages/Upload.py`
- All of `Utils/*`, remaining `Pages/*`, `App.py`, `tests/test_core.py` (restyle only), `requirements.txt`

## What The Next Agent Should Do

1. Read this file, then `AI/MODEL.md` and `AI/TASK.md`; check `AI/SESSIONS.md` for recent history
2. Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests` (expect 24/24)
3. Also run `& .\venv\Scripts\python.exe bug_hunt.py` (expect 9/9 pages OK)
4. Verify claims above against code before changing anything
5. Record a pre-task snapshot in SESSIONS.md before your first edit

## Verification Needed

- Manual browser smoke test: enter Gemini/AWS keys with keep-box unticked, run a task, confirm fields clear and toast appears
- Confirm `pip install pypdf matplotlib` enables PDF input and charts

## Do Not Forget

- Update `MODEL.md` when switching agents/models
- Update this file and complete the SESSIONS.md post-task summary before ending any session
- Verify documentation against actual code — never trust it blindly

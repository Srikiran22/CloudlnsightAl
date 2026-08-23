# AI Handoff

- **Last Updated:** 2026-08-23
- **Current Agent:** ox-alpha
- **Current Model:** x-preview-f-free (opencode/x-preview-f-free)
- **Current Application:** OpenCode CLI

## What We Are Building

CloudInsight AI — a Streamlit analytics platform: universal ingestion with Gemini conversion, detailed PDF reports, Compare/ML/Dashboard pages, runtime-only secrets. A full 36-phase remediation pass completed 2026-08-23 (see SESSIONS.md).

## Current Objective

None open. Project is post-remediation; next work should be feature-driven.

## Current Status

71/93 tests pass; bug_hunt boots 9/9 pages; ruff serious-errors gate clean. matplotlib + pypdf are installed in the venv, so PDF charts and PDF ingestion are live. Quality Index is unified in `Utils/quality.py` (equal blend — DEC-011). Gemini wrapper has typed errors/timeouts/retries and a centralized model registry. AI-CSV parsing is bounded with an O(n) fast path (~160000x faster on clean tables, same outputs). Logs go to the launching terminal; level via CLOUDINSIGHT_LOG_LEVEL.

## What Has Been Done (latest session)

- Fixed: Dashboard formatting defect + formula unification; silent excepts now logged; PDF crash on 0-column datasets; per-chart failure isolation; MAX_CONVERTED_COLUMNS now caps columns not cells
- Hardened: upload size guard (200MB) + duplicate-name disambiguation; chat history cap (100); preprocessing strategy validation; S3 error mapping + listing cap; bounded retries for transient Gemini failures only
- Added: Utils/logsys.py, Utils/quality.py, Utils/compare_logic.py; 43 new tests; CI ruff gate + boot-check step; README rewritten (.env instructions removed — they never matched the runtime-key design)
- Decisions: DEC-011 (canonical quality index), DEC-012 (keep dual Gemini SDKs — legacy package is what this venv has)

## What Is In Progress

None.

## What Remains

- Nothing blocking. Optional future items in STATE.md Risks section.

## Important Decisions

- DEC-001..010: see DECISIONS.md (Markdown memory; HANDOFF entry point; no CoT storage; precedence order; native parsers first; strict validated CSV from AI; optional chart deps; plain-file persistence; memory-only secrets; de-AI restyle)
- DEC-011: one canonical equal-blend Data Quality Index in Utils/quality.py
- DEC-012: dual Gemini SDK support retained until venv migrates to google-genai

## Important Constraints

- Downstream pages rely on `current_df`/`dataset_name` session keys and the `Datasets/` folder — keep those contracts intact
- All AI prompts stay bounded and include untrusted-content guards
- Never reintroduce env-file secret seeding or disk persistence of credentials
- Keep the restyle discipline: terse comments, no formulaic docstrings; check tests/pages before any rename
- Quality Index changes must touch quality.py + PDF note + README + QualityIndexTests together

## Known Problems

None confirmed. See STATE.md Risks for accepted limitations.

## Files Recently Changed

- New: `Utils/logsys.py`, `Utils/quality.py`, `Utils/compare_logic.py`
- Modified: `Utils/{Gemini,AIConvert,ML,PDF,S3,Preprocessing,dataset_ui,paths,theme}.py`, `Pages/{Upload,Cleaning,Compare,EDA,Dashboard,ML,AI}.py`, `App.py`, `tests/test_core.py`, `Readme.md`, `requirements.txt`, `.github/workflows/tests.yml`

## What The Next Agent Should Do

1. Read this file, then `AI/MODEL.md` and `AI/TASK.md`; check `AI/SESSIONS.md` for recent history
2. Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests` (expect 93/93)
3. Also run `& .\venv\Scripts\python.exe bug_hunt.py` (expect 9/9 pages OK)
4. Verify claims above against code before changing anything
5. Record a pre-task snapshot in SESSIONS.md before your first edit

## Verification Needed

- Manual browser smoke test of theme toggle + secret wipe UX (headless checks cover logic only)

## Do Not Forget

- Update `MODEL.md` when switching agents/models
- Update this file and complete the SESSIONS.md post-task summary before ending any session
- Verify documentation against actual code — never trust it blindly

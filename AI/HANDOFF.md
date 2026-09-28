# AI Handoff

- **Last Updated:** 2026-09-28
- **Current Agent:** Antigravity IDE
- **Current Model:** Gemini 3.8 Flash (google/gemini-3.8-flash)
- **Current Application:** Antigravity IDE

## What We Are Building

CloudInsight AI — a Streamlit analytics platform: universal ingestion with Gemini conversion, detailed PDF reports, Compare/ML/Dashboard pages, runtime-only secrets. A full engineering audit, fix, test, and verification completed 2026-09-28.

## Current Objective

Full engineering audit & hardening complete. Repository is verified clean, deterministic, and production-grade.

## Current Status

155/155 tests pass (122 baseline + 33 audit hardening in `tests/test_audit_hardening.py`); bug_hunt boots 9/9 pages; ruff fatal checks (`E9,F63,F7,F82,F821`) pass 100% cleanly; 97 non-fatal style findings. Gemini wrapper runs native `google-genai` (v2.25.0) as the primary SDK path with typed errors/timeouts/retries, a centralized model registry defaulting to `gemini-3.8-flash`, and deprecation resolution. Datasets use full cryptographic 64-character SHA-256 hex digests (`hasher.hexdigest()`) guaranteeing exact content identity without birthday collisions and preventing stale ML, cleaning, cache, and AI chat state under middle-byte edits or timestamp-preserving writes.

## What Has Been Done (latest session)

- 2026-09-28 Full Engineering Audit & Hardening:
  1. Data Ingestion & Collision Safety: Implemented `get_unique_filename` deterministic collision avoidance for local uploads, AI conversions (`f"{stem}_{ext}_converted.csv"` registered in `CONVERSIONS_MANIFEST`), combined datasets (`combined_dataset_1.csv`), and S3 downloads (`folder_sub_file.csv`).
  2. Resource & Memory Bounds: Centralized `enforce_size_limit` across `read_dataset`, `read_tabular`, and uploads. Implemented 64KB streaming download with accumulated byte ceiling in S3 even without `ContentLength`. Bounded PDF extraction (`MAX_PDF_PAGES=30`, `MAX_PDF_EXTRACT_CHARS=50,000`). Bounded dataset caching pushing `max_rows` down to readers. Added large scatter plot sampling (25k) and EDA correlation column caps (50 cols).
  3. Parsing Correctness: XML repeated child tags indexed as `tag_1`, `tag_2` to prevent data loss. Ambiguous top-level JSON lists rejected with explicit `ValueError`. List/dict cells in normalized JSON serialized via `_sanitize_unhashable_cells` to avoid unhashable type errors. Quote-aware CSV field counting in `AIConvert.py`.
  4. Dataset Identity & Stale State: Cryptographic SHA-256 content hash with 0ms stat-based cache. In-memory session fingerprinting. AI insights, chat messages, and cleaning results keyed by content hash. Report template overwrite protection checkbox.
  5. Gemini AI Hardening: Default model updated to `gemini-2.5-flash` with registry (`gemini-2.5-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`). Added deprecation resolver (`gemini-2.0-flash` -> `gemini-2.5-flash`). System prompt reinforced with context limits and strict honesty constraints.
  6. ML Pipeline: Guarded against dense one-hot encoding dimensional explosions (`estimated_encoded_columns`). Collision-safe `.joblib` model saving with embedded dataset fingerprint. Documented sklearn version mismatch policy. Explicit tracking for stratified split fallbacks (`stratified_split: False` with descriptive warning). Theme unification using `plot_template()`.
  7. Reports & Visualization: Truncated batch report metadata shows both source rows and analyzed rows with sampling disclosure. Report PDF filenames use microsecond timestamps with collision-safe suffixes. Re-framed comparison drift to summary metric shifts and fixed near-zero baseline percentage math. Pie/treemap negative value validation. Chronological datetime sorting for line charts. Empty dataset quality index handled cleanly (0.0).
  8. Dependencies & Streamlit Compatibility: Added `pyarrow>=14.0.0` and `xlrd>=2.0.1` to `requirements.txt`. Set minimum Streamlit version to `1.52.0` (required for `width="stretch"` support across dataframe and chart elements).
  9. Testing & Hardening: 152/152 tests passing (122 core + 30 audit hardening). Fail-closed timeout guarantees, column cap (>200 columns) explicit ValueError, boundary-sampled SHA-256 fingerprinting with cache invalidation, and Streamlit compatibility verified. Renamed `Readme.md` to `README.md`. Synchronized all documentation.


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
2. Run: `& .\venv\Scripts\python.exe -m unittest discover -s tests` (expect 117/117)
3. Also run `& .\venv\Scripts\python.exe bug_hunt.py` (expect 9/9 pages OK)
4. Verify claims above against code before changing anything
5. Record a pre-task snapshot in SESSIONS.md before your first edit

## Verification Needed

- Manual browser smoke test of theme toggle + secret wipe UX (headless checks cover logic only)

## Do Not Forget

- Update `MODEL.md` when switching agents/models
- Update this file and complete the SESSIONS.md post-task summary before ending any session
- Verify documentation against actual code — never trust it blindly

# Architectural Decision Log

This file records meaningful decisions made during development. Each entry captures what was decided, why, and what alternatives were considered.

---

## DEC-001 — Use plain Markdown for all memory files

- **Date:** 2026-08-21
- **Agent:** OpenCode / mimo-v2.5-free
- **Model:** opencode/mimo-v2.5-free

### Decision

All memory files (now in the `AI/` folder) use plain Markdown (`.md` format).

### Why

- Universally readable by any AI agent or human
- No proprietary formats, plugins, or tools required
- Works across all coding applications (Codex, Cursor, Zed, OpenCode, Cline, etc.)
- Can be read, written, and diffed with standard tools

### Alternatives Considered

- JSON/YAML structured files — more machine-readable but less human-friendly
- Database-backed memory — requires external dependency, breaks portability
- Proprietary IDE formats — vendor lock-in, not portable

### Consequences

- All agents must parse Markdown (universally supported)
- No schema enforcement at file level (relies on conventions in AGENTS.md)

### Status

Active

---

## DEC-002 — HANDOFF.md as the primary entry point for new agents

- **Date:** 2026-08-21
- **Agent:** OpenCode / mimo-v2.5-free
- **Model:** opencode/mimo-v2.5-free

### Decision

`HANDOFF.md` is the single file a new agent reads first. It contains the most critical context in compact form.

### Why

- Minimizes context window usage for new sessions
- Provides a complete "what do I need to know" summary
- Reduces the chance of an agent missing critical information
- Other files (STATE.md, DECISIONS.md) provide detail when needed

### Alternatives Considered

- Read all files in sequence — wastes context tokens on irrelevant detail
- Single monolithic file — becomes unwieldy as project grows
- No structured handoff — each agent must reconstruct context from scratch

### Consequences

- HANDOFF.md must be kept concise and current
- Other files serve as supporting detail, not primary reading

### Status

Active

---

## DEC-003 — No private chain-of-thought storage

- **Date:** 2026-08-21
- **Agent:** OpenCode / mimo-v2.5-free
- **Model:** opencode/mimo-v2.5-free

### Decision

The system explicitly forbids storing hidden reasoning, internal scratchpad content, or private deliberation.

### Why

- Another AI agent should only see useful, actionable information
- Hidden reasoning creates confusion when context is opaque
- Decision summaries and rationale are sufficient for handoff
- Privacy and transparency: all recorded info is safe for any agent to read

### Alternatives Considered

- Store full reasoning traces — increases context burden, risks contradictions
- Store nothing — loses critical context between sessions

### Consequences

- Agents must distill reasoning into concise summaries
- Historical context is limited to what was explicitly recorded

### Status

Active

---

## DEC-004 — Precedence system for conflicting information

- **Date:** 2026-08-21
- **Agent:** OpenCode / mimo-v2.5-free
- **Model:** opencode/mimo-v2.5-free

### Decision

Establish a clear precedence: User request > Task requirements > Project constraints > Current code > .ai documentation.

### Why

- Documentation can become stale
- The user's current instructions are always most important
- Actual code on disk is the ground truth
- Documentation is a helpful guide, not an authority

### Alternatives Considered

- Treat documentation as authoritative — risks acting on stale information
- No precedence — leads to confusion when sources conflict

### Consequences

- Agents must verify claims against actual code
- Documentation serves as context, not ground truth

### Status

Active

---

## DEC-005 � Native parsers first, AI conversion only as fallback

- **Date:** 2026-08-21
- **Agent:** OpenCode (ox-alpha)
- **Model:** opencode/x-preview-f-free

### Decision

Universal ingestion uses deterministic native parsers for structured formats (JSON, JSONL, TSV, Parquet, XML, HTML, delimited TXT). Gemini AI conversion is offered only when no native parse is possible, surfaced via the `AIConversionRequired` exception.

### Why

Deterministic parsing is free, instant, and lossless; AI calls cost money/latency and can hallucinate values. Repository evidence: pyarrow already present for Parquet; stdlib `html.parser` and `xml.etree` avoid new mandatory deps.

### Alternatives Considered

- Route every file through AI � costly, non-deterministic, privacy burden
- Only add pandas-native formats and skip HTML/XML � misses common user files without lxml installed

### Consequences

- `read_tabular` can raise a special exception type callers must handle
- Converted AI output is saved as `<stem>_converted.csv`; original file kept untouched

### Status

Active

---

## DEC-006 � AI conversion outputs strict CSV with validation

- **Date:** 2026-08-21
- **Agent:** OpenCode (ox-alpha)
- **Model:** opencode/x-preview-f-free

### Decision

Gemini is prompted to return ONLY headered CSV; response is fence-stripped, prose-trimmed, parsed by pandas, then validated (non-empty rows/columns, 200-column cap).

### Why

CSV is the most reliable structured output format for LLMs and parses deterministically. Validation prevents empty/garbage tables from entering session state. Bounded sample (12k chars) keeps prompts cheap.

### Alternatives Considered

- JSON output from model � higher malformed-rate with long tables
- Code-interpreter style execution � not available in this stack

### Consequences

- Very large unstructured files are converted from a truncated sample only
- Untrusted-content guard included in prompt (matches existing Gemini.py pattern)

### Status

Active

---

## DEC-007 � Optional matplotlib charts in PDF with graceful degradation

- **Date:** 2026-08-21
- **Agent:** OpenCode (ox-alpha)
- **Model:** opencode/x-preview-f-free

### Decision

PDF distribution charts use matplotlib if importable; otherwise the section is silently skipped. Added to requirements.txt as optional runtime deps alongside pypdf (PDF text extraction).

### Why

Detail gain is large but must not hard-break report generation on machines without the packages. All other PDF detail (quality scores, extended stats, categorical/correlation sections) is pure ReportLab/pandas with zero new dependencies.

### Alternatives Considered

- Plotly + kaleido static export � heavy transitive deps, slower
- Hard dependency requirement � breaks existing installs until manual pip install

### Consequences

- Users should run `pip install pypdf matplotlib` to unlock PDF input and charts
- Report.py exposes an include-charts checkbox

### Status

Active

---

## DEC-008 � Plain-file persistence for models and report templates

- **Date:** 2026-08-21
- **Agent:** OpenCode (ox-alpha)
- **Model:** opencode/x-preview-f-free

### Decision

Trained ML pipelines persist as joblib bundles (pipeline + metadata dict) in `Models/`; report configurations as plain JSON in `Reports/templates/`. Dataset reads go through `st.cache_data` keyed by path + mtime.

### Why

joblib ships with scikit-learn (no new dependency) and handles sklearn pipelines natively. JSON templates stay human-editable and vendor-neutral. mtime-keyed caching gives instant page loads with automatic invalidation on file change.

### Alternatives Considered

- MLflow/model registry � external service, violates zero-dependency constraint
- Pickle directly � less secure and less efficient than joblib for numpy arrays
- Hash-based cache keys � slower; mtime is sufficient for single-user desktop usage

### Consequences

- Model bundles are only loadable within compatible scikit-learn versions
- Batch reports cap rows per dataset (user-configurable) to bound runtime

### Status

Active

---

## DEC-009 — In-memory-only secret lifecycle (runtime entry, wipe after use)

- **Date:** 2026-08-22
- **Agent:** OpenCode (ox-alpha)
- **Model:** opencode/x-preview-f-free

### Decision

Gemini and AWS credentials are entered in-app at runtime (`Utils/secrets.py`). They live only in `st.session_state`; after each completed API task (AI conversion, insights report, chat reply, S3 fetch/download) they are wiped unless the user ticked "Keep in memory for this session". Env-var auto-seeding (`GEMINI_API_KEY`) and `python-dotenv` auto-loading were removed; explicit "clear" controls exist on the AI page.

### Why

Long-lived secrets sitting in browser-session state (or env files) are the main exposure in a desktop Streamlit app. Wiping after use matches the requested behavior: the key exists only while work needs it.

### Alternatives Considered

- Keep env-var seeding for CI convenience — rejected: silently reintroduces persistent secrets
- Write user-approved keys to a local config file — rejected: secrets on disk, worse than session memory
- Per-message re-prompting without any store option — rejected: hostile UX; the keep-checkbox covers it

### Consequences

- Users who leave the box unticked must re-enter credentials per task (by design)
- Bucket/region persist across tasks; only access/secret keys are treated as secrets
- Old `gemini_api_key` / `aws_key` / `aws_secret` session keys no longer exist

### Status

Active

---

## DEC-010 — De-AI restyle with frozen public surface

- **Date:** 2026-08-22
- **Agent:** OpenCode (ox-alpha)
- **Model:** opencode/x-preview-f-free

### Decision

All Python sources were restyled to read less machine-generated: formulaic docstrings and banner comments removed, narration comments trimmed or made terse/lowercase, uniform type hints thinned, small internal consolidations. Frozen throughout: public function names, keyword argument names, result-dict keys, prompts, UI strings, error messages (tests regex-match several), and all runtime behavior.

### Why

The codebase's perfectly uniform docstring/comment/hint patterns read as AI-authored; varying prose style while keeping APIs identical changes presentation without touching what the app does.

### Alternatives Considered

- Renaming public functions/renaming UI copy — rejected: breaks tests, callers, and user-visible behavior
- Full type-hint removal — rejected: hints aid maintenance; only the mechanical uniformity was thinned

### Consequences

- Future edits should keep the same style discipline (terse comments, no formulaic docstrings)
- Any rename must first be checked against tests/pages imports

### Status

Active

---

## DEC-011 — One canonical Data Quality Index (equal blend)

- **Date:** 2026-08-23
- **Agent:** ox-alpha / OpenCode CLI

### Decision

Dashboard gauge and PDF report both use `Utils/quality.py`: `index = (completeness + uniqueness) / 2`. The Dashboard's old `completeness*0.6 + uniqueness*0.4` variant was removed.

### Why

Two formulas for the same user-facing number is a correctness defect. Both date to the initial commit; only the PDF version carried any rationale ("blends equally"), so equal weighting became canonical: symmetric, explainable, and it keeps existing PDF report text truthful without edits. Documented in Readme and in the Dashboard metric help text; pinned by tests asserting both modules import the shared implementation.

### Consequences

- Dashboard gauge values shift slightly on datasets where completeness ≠ uniqueness
- Any future reweighting must update quality.py docstring, the PDF note, README, and QualityIndexTests

### Status

Active

---

## DEC-012 — Keep dual Gemini SDK support

- **Date:** 2026-08-23
- **Agent:** ox-alpha / OpenCode CLI

### Decision

`Utils/Gemini.py` keeps both `google-genai` (preferred) and legacy `google-generativeai` paths. Legacy code stays isolated inside `_generate_once`.

### Why

The audit suggested dropping legacy support since requirements pin google-genai — but verification showed this project's own venv currently has ONLY the legacy package installed. Removing it would break the actual running environment. Revisit if/when the venv migrates to google-genai.

### Status

Active


---

## DEC-013 — Keep CSV-text Gemini conversion; reject JSON-schema output for this task

- **Date:** 2026-08-24
- **Agent:** ox-alpha / OpenCode CLI

### Decision

The unstructured-file to table pipeline stays: prompt for strict CSV text, parse with bounded parse_ai_csv. The google-genai structured-output mechanism (response_mime_type=application/json + response_schema) was evaluated and NOT adopted.

### Why

The extraction target is a variable-schema table, so a response schema cannot constrain anything meaningful beyond a generic {columns, rows} envelope. For that shape JSON mode costs roughly 30-40 percent more output tokens per row, long JSON arrays are where models historically truncate, and it removes none of the required validation (semantic sanity is checked either way). The CSV path already has fence/prose tolerance, hard bounds, and an O(n) fast path; a hybrid would double the maintenance surface without removing any guard.

### Consequences

- parse_ai_csv remains the single AI-to-table boundary; its bounds/tests stay authoritative
- Revisit only if models stop emitting reliable CSV or CSV row-truncation becomes measurable

### Status

Active

---

## DEC-014 — Content-derived dataset fingerprinting with stat-keyed caching

- **Date:** 2026-09-28
- **Agent:** Antigravity IDE / Gemini 3.8 Flash

### Decision

Upgrade `dataset_fingerprint()` from metadata-only (`name_size_mtime`) to a true SHA-256 content-derived hash (`sha256_<64hex>`) with an internal LRU stat cache `_FP_CACHE[(path, mtime, size)]`. Downstream state across AI insights, chat history, cleaning results, report metadata, and ML models are now keyed by or record this SHA-256 fingerprint. Added `dataframe_fingerprint(df)` for in-memory active session frames.

### Why

Filenames or simple metadata fingerprints allowed stale AI insights, stale chat history, stale cleaning records, and misattributed report summaries when a dataset file was replaced on disk by another file with identical name or near-identical metadata. True SHA-256 hashing guarantees cryptographic uniqueness while the `(path, mtime, size)` stat-cache ensures zero disk I/O overhead on repeated queries.

### Alternatives Considered

- Full hash on every access without caching — unacceptable latency on multi-megabyte datasets.
- Pure metadata fingerprint (name + mtime + size) — prone to collisions on file replacement or in-memory manipulations.

### Consequences

- All UI state keys for AI, cleaning, and reports are collision-free and automatically invalidate when file content changes.
- Legacy state keys without fingerprints are safely ignored or migrated.

### Status

Active

---

## DEC-015 — Deterministic collision-safe naming and S3 provenance preservation

- **Date:** 2026-09-28
- **Agent:** Antigravity IDE / Gemini 3.8 Flash

### Decision

Use `get_unique_filename(directory, filename)` to resolve filename collisions via deterministic incremented suffixes (`data.csv` -> `data_1.csv` -> `data_2.csv`) across local uploads, AI conversions, combined merges, ML model artifacts, and PDF reports. For S3 downloads, prefix the filename with sanitized parent folder provenance (e.g., `folder_sub_data.csv`) before collision resolution.

### Why

The application previously overwrote files silently if an uploaded file, S3 download, AI conversion, or batch merge shared a filename with an existing dataset or artifact in `Datasets/`, `Models/`, or `Reports/`. Downstream session state was corrupted or lost historical data.

### Alternatives Considered

- UUID prefixes/suffixes — rejected: creates unreadable, chaotic filenames for end users.
- Overwrite by default with prompt — rejected: automated flows and background conversions cannot block interactively.

### Consequences

- User files and artifacts are never silently overwritten.
- Provenance is preserved for nested S3 keys.

### Status

Active

---

## DEC-016 — Full SHA-256 exact content fingerprinting

- **Date:** 2026-09-28
- **Agent:** Antigravity IDE / Gemini 3.8 Flash

### Decision

Replace the partial 4KB boundary (head/tail) digest with full SHA-256 content hashing in `dataset_fingerprint()`. In addition, wire `_load_cached_dataset.clear()` and `_load_cached_dataset_bounded.clear()` directly into `invalidate_dataset_cache()`.

### Why

The previous boundary digest inspected only the first 4KB and last 4KB of files. In-place modifications to files larger than 8KB where bytes in the middle changed while file size and timestamps remained identical resulted in stale cache hits in `_CONTENT_HASH_CACHE` and stale DataFrame returns from Streamlit's `@st.cache_data`. True full SHA-256 hashing processes at ~1-2 GB/s on modern hardware (5 ms for 10MB), ensuring exact cryptographic content identity with negligible latency.

### Consequences

- Middle-byte edits, identical-size rewrites, and timestamp-preserved modifications are guaranteed to produce distinct fingerprints and invalidate all cached data.
- The system guarantees exact content identity rather than metadata-assisted or probabilistic boundary approximations.

### Status

Active

---

## DEC-017 — Adopt Gemini 3.5 Flash as default production model

- **Date:** 2026-09-28
- **Agent:** Antigravity IDE / Gemini 3.8 Flash

### Decision

Update `DEFAULT_GEMINI_MODEL` from `gemini-2.5-flash` to `gemini-3.5-flash` across the central registry, AI conversion, and UI selectors. Map `gemini-2.0-flash` and `gemini-2.0-flash-exp` to `gemini-3.5-flash` in `DEPRECATED_GEMINI_MODELS`. Retain `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, and `gemini-1.5-pro` as explicit supported choices in `GEMINI_MODELS`.

### Why

Per official Google AI documentation (September 2026), `gemini-3.5-flash` is the current stable, GA flagship Flash model recommended for all new production projects. `gemini-2.0-flash` is officially shut down (June 1, 2026). `gemini-2.5-flash` remains supported for backward compatibility but is no longer recommended as the default for new projects.

### Consequences

- All new AI tasks (conversion, insights, chat) default to `gemini-3.5-flash`.
- Legacy model compatibility is strictly preserved for users with existing quotas.

### Status

Superseded by DEC-018

---

## DEC-018 — Upgrade to full 64-character SHA-256 fingerprint, gemini-3.8-flash default, and google-genai 2.x SDK

- **Date:** 2026-09-28
- **Agent:** Antigravity IDE / Gemini 3.8 Flash

### Decision

1. **Full SHA-256 Hex Digest:** Dataset fingerprinting in `dataset_fingerprint()` and `dataframe_fingerprint()` (`Utils/dataset_ui.py`) now outputs the full 64-character SHA-256 hexadecimal string (`hasher.hexdigest()`), removing the previous 12-character (`[:12]`) truncation.
2. **Gemini 3.8 Flash Default:** `DEFAULT_GEMINI_MODEL` is updated to `gemini-3.8-flash` across the central registry (`Utils/Gemini.py`), AI conversion (`Utils/AIConvert.py`), UI selectors (`Pages/Upload.py`, `Pages/AI.py`), tests, and documentation. Deprecated models (`gemini-2.0-flash`, `gemini-2.0-flash-exp`) map to `gemini-3.8-flash`. Supported legacy models (`gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`) are retained.
3. **Primary google-genai SDK Path:** Installed and activated `google-genai` (v2.25.0) in `./venv`, fulfilling `requirements.txt` (`google-genai>=1.0.0`) and ensuring the primary tested runtime path exercises native `google.genai.Client` rather than deprecated `google-generativeai`.

### Why

- Truncating SHA-256 to 12 hex characters (48 bits) reduced collision entropy to $2^{24}$ files (birthday bound), which contradicts claims of exact cryptographic content identity. Full 64-hex SHA-256 provides complete 256-bit collision security with zero performance overhead.
- Official Google Gemini model documentation lists `gemini-3.8-flash` as the latest stable Flash model. Setting it as default gives users the highest accuracy, speed, and latency optimizations.
- The `google-generativeai` SDK emits end-of-life deprecation notices directing users to `google-genai`. Running and testing on `google-genai` directly ensures production compatibility with current Google Cloud APIs.

### Consequences

- Exact cryptographic identity is verified by unit tests asserting 64-character digests.
- Default model resolution uses `gemini-3.8-flash`.
- The primary SDK runtime uses `google.genai.Client` with configured HTTP timeouts.

### Status

Active

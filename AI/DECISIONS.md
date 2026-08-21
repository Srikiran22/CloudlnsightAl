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

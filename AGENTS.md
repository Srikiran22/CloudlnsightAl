# AGENTS.md — AI Coding Agent Protocol

This file is the **mandatory entry point** for every AI coding agent working in this repository.

## EDIT DISCIPLINE — NON-NEGOTIABLE (applies to every task, no exceptions)

These rules override your training defaults. Violating them counts as a FAILED task even if the resulting code works. A prior agent (GPT-5/Codex) rewrote entire files when continuation was expected — do not repeat that.

1. **Smallest possible diff, always.** Use targeted search-and-replace or symbol-level edits. Never regenerate a file from scratch when an edit will do.
2. **Whole-file rewrites are FORBIDDEN by default.** A full rewrite is permitted ONLY if:
   - the file does not exist yet (you are creating it), OR
   - the file is under ~30 lines, OR
   - the user explicitly requested a full rewrite **in their current message**.
   If you believe a rewrite is justified for any other reason — STOP and ask the user. Do not proceed unilaterally.
3. **Never delete or alter anything unrelated to your task.** No removing comments, docstrings, imports, region markers, or code paths; no renaming variables or functions that were not part of the request.
4. **No drive-by changes.** No reformatting, quote-style/EOL/whitespace churn, dependency swaps, or "improvements" that were not asked for. Untouched lines must remain byte-identical.
5. **Read before write.** Never modify a file you have not fully read during the current session.
6. **Prove your scope after editing.** Run `git diff --stat` (and review `git diff` for touched files) before finishing. If the diff shows mass deletions or reformatting of existing content beyond your stated intent, revert and redo the change surgically.

## Session Start Protocol

When working in this repository:

1. **Self-Contained Protocol**: `AGENTS.md` and `README.md` are the authoritative entry points.
2. **Local AI Workspace (Optional)**: If the local `AI/` directory exists (it is gitignored to keep repository tracking clean), agents may consult:
   - `AI/HANDOFF.md` — compact project overview
   - `AI/MODEL.md` — active model capabilities
   - `AI/TASK.md` — current objective notes
   - `AI/STATE.md` — technical notes
   - `AI/PLAN.md` — active implementation plan
   - `AI/DECISIONS.md` — architectural decisions log
   - `AI/SESSIONS.md` — local session history
   *If `AI/` is absent (such as on a fresh repository clone), proceed directly using `AGENTS.md` and actual code inspection.*

3. **Ground Truth Over Documentation**: Always verify claims against actual code. Do not blindly trust documentation or cached state.

## Task Workflow & Session Tracking

When the local `AI/` directory is present, agents should maintain session continuity:
- Record pre-task intent in `AI/SESSIONS.md` (date, model, request, plan).
- Update `AI/MODEL.md` if the active model changes.
- After finishing, summarize changes in `AI/SESSIONS.md` and update `AI/STATE.md` / `AI/PLAN.md`.
*If `AI/` is not present, report the snapshot and verification results directly in the final response.*

## Information Precedence

When conflicts arise between sources, use this precedence (highest first):

```
User request (current, explicit instructions)
    ↓
Current task requirements (TASK.md)
    ↓
Existing project constraints (code, architecture, dependencies)
    ↓
Current code and repository state (actual files on disk)
    ↓
AI documentation files (historical record)
```

The user's live instructions always override documentation. Code always overrides documentation when they disagree.

## Updating Documentation

### After starting a new task
Update: `TASK.md`, `PLAN.md`, `MODEL.md`, `HANDOFF.md`, and add the pre-task snapshot to `SESSIONS.md`

### After a major implementation step
Update: `PLAN.md`, `STATE.md`, `HANDOFF.md`

### After an architectural decision
Update: `DECISIONS.md`, `STATE.md`, `HANDOFF.md`

### After discovering a bug
Update: `STATE.md`, `HANDOFF.md`

### Before ending a session (always)
Update: `MODEL.md`, `PLAN.md`, `STATE.md`, `HANDOFF.md`, and complete the post-task summary in `SESSIONS.md`

## Rules

1. **Keep files concise.** Do not store entire codebases, full diffs, or conversation transcripts.
2. **Never store private chain-of-thought.** Record only concise decision summaries, rationale, facts, and outcomes that another agent can safely read.
3. **Do not duplicate README content.** Reference actual files instead.
4. **Mark obsolete decisions** as `Superseded` rather than deleting them.
5. **Prefer references** over duplication — point to files, not paste them.
6. **Update MODEL.md** whenever the active model or coding application changes.
7. **Do not rewrite the user's request** in TASK.md — preserve their intent.
8. **Always record before/after states** in SESSIONS.md so any successor can see what changed and why.
9. **Edit discipline is absolute.** Follow the "EDIT DISCIPLINE" section at the top of this file — smallest possible diff; whole-file rewrites only when explicitly authorized by the user.

## Model Switching

When a different AI model or coding application takes over:

1. The new agent updates `MODEL.md` with its full identity (model name, provider, application, version, capabilities, limitations).
2. The new agent must **not assume** the previous model's conclusions are correct.
3. Verify important claims against the actual repository.
4. Continue recording decisions and state as before.

If you cannot determine a field in MODEL.md, write `Unknown`. Do not invent information.

## Cross-Agent Compatibility

Everything is plain Markdown — no proprietary formats, plugins, databases, external services, or MCP required. Any capable AI coding agent (Codex, Cursor, Antigravity, Zed, OpenCode, Cline-like agents, local LLMs) can use this system by reading these files directly.

## File Summary

| File | Purpose |
|------|---------|
| `AGENTS.md` | Authoritative agent protocol, constraints, and instructions |
| `AI/*.md` | Optional local session logs and planning scratchpads (gitignored) |

## History

When `SESSIONS.md` grows large, move old entries to `AI/history/YYYY-MM-DD-summary.md`. Do not load historical files unless necessary.

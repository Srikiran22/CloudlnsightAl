# AGENTS.md — AI Coding Agent Protocol

This file is the **mandatory entry point** for every AI coding agent working in this repository.

## Session Start Protocol

**Every AI agent MUST follow this sequence before doing any work:**

1. Read `AI/HANDOFF.md` — compact project overview; start here
2. Read `AI/MODEL.md` — which model/application was last active, and its capabilities
3. Read `AI/TASK.md` — what the user is asking for
4. Read `AI/STATE.md` — technical state of the repository
5. Read `AI/PLAN.md` — when the task requires planning or multi-step work
6. Read `AI/DECISIONS.md` — when making architectural or implementation decisions
7. Read `AI/SESSIONS.md` — recent session history (before/after task snapshots)

**After reading the documentation, verify against the actual code.** Do not blindly trust old information.

## Before Starting a Task (mandatory)

Before making any change, record a **pre-task snapshot** in `AI/SESSIONS.md`:

- Date, agent identity (model/provider/application)
- The user's request in your own concise words
- Current relevant state (tests passing? build status? files involved?)
- What you intend to do (planned approach in 2–5 bullets)

Also update `AI/MODEL.md` if the active model or application changed.

## After Finishing a Task (mandatory)

Record a **post-task summary** in `AI/SESSIONS.md`:

- What was actually done vs. planned
- Files changed (paths only, no diffs)
- Decisions made (cross-reference DEC IDs from `AI/DECISIONS.md`)
- Verification results (test commands run and outcomes)
- Problems encountered and anything left unresolved
- Updated "after" state (tests passing? new status?)

Then update `AI/HANDOFF.md`, `AI/STATE.md`, and `AI/PLAN.md` so the next agent sees current reality.

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
| `AGENTS.md` | This file — protocol and rules |
| `AI/HANDOFF.md` | Compact project overview for new agents |
| `AI/MODEL.md` | Detailed current AI agent identity |
| `AI/TASK.md` | User's current objective |
| `AI/PLAN.md` | Implementation plan and progress |
| `AI/DECISIONS.md` | Architectural decision log |
| `AI/STATE.md` | Current technical state |
| `AI/SESSIONS.md` | Session history with before/after task snapshots |

## History

When `SESSIONS.md` grows large, move old entries to `AI/history/YYYY-MM-DD-summary.md`. Do not load historical files unless necessary.

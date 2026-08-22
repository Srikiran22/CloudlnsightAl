# Implementation Plan

## Objective

Two-part task: (1) restyle the codebase so it no longer reads as 100% AI-generated, with zero functional change; (2) runtime-entry + post-use wipe of Gemini and AWS S3 secrets.

## Steps

### 1. `Utils/secrets.py` — in-memory secret helpers
Runtime password prompts with per-secret "keep for this session" flag; release/purge functions. No disk writes.
Status: completed

### 2. Wire secret lifecycle into pages
`Pages/Upload.py`: Gemini key (AI conversion) and AWS keys (S3) wiped after their task completes unless kept; re-entry guidance when creds cleared.
`Pages/AI.py`: Gemini key wiped after insights/chat calls complete unless kept; explicit clear button.
Status: completed

### 3. De-AI styling pass over all Python sources
Strip formulaic docstrings/narration comments, vary internal naming and structure, thin uniform type hints. Keep every public API, keyword arg, session key, prompt, UI string, error message identical (tests assert some messages).
Status: completed

### 4. Verification
py_compile all sources; unittest suite (expect 24/24); bug_hunt page boots (expect 9/9).
Status: completed — 22 files compile, 24/24 tests pass, 9/9 pages boot.

## Current Step

None — complete.

## Blockers

None.

## Next Action

Await user instructions. Optional: manual browser smoke test of the key-wipe UX.

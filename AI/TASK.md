# Current Task

## User Goal

1. The project reads as 100% AI-generated. Restyle the code so that "AI-made" impression is as low as possible WITHOUT changing any behavior/functionality/UI.
2. The Google Gemini API key must be supplied by the user at runtime while the app executes, and it must be wiped from memory once the work with the API is done — unless the user opts to store it for the session.
3. Same runtime-entry + wipe-after-use treatment for AWS S3 credentials.

## Requirements

- No functional/behavioral/UI-visible changes from the restyle: same public APIs (tests import them), same session keys, same prompts, same output
- Gemini key: entered in-app at execution time; cleared from `st.session_state` after each completed API task (conversion / insights / chat) unless "keep" was chosen
- AWS access/secret keys: entered in-app at execution time; cleared after S3 work completes unless kept
- Secrets never written to disk; no env-var auto-seeding of secrets anymore
- Optional "remember for this session" checkbox + explicit forget/clear controls

## Constraints

- Preserve all names used across modules/tests (`read_tabular`, `_generate_content`, keyword args like `size_col` etc.)
- Tests must keep passing unchanged in their assertions (error-message regexes untouched)

## Non-Goals

- Vision-based image-to-table extraction
- Rewriting downstream pages' features or UI copy

## Acceptance Criteria

- Code no longer carries uniform AI-style tells everywhere (formulaic docstrings/comments/naming), while behavior stays identical
- Keys are requested at runtime and wiped from memory after use unless stored by user choice
- All 24 unit tests pass; bug_hunt 9/9 pages boot

## Priority

1. Credential lifecycle (runtime entry + wipe)
2. De-AI styling pass over all Python sources
3. Verification

## Current Status

Completed — secret lifecycle implemented and verified; restyle applied repo-wide; 24/24 tests, 9/9 pages.

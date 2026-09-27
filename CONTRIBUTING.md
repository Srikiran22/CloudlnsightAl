# Contributing

CloudInsight AI is primarily a portfolio project. Changes should preserve the application's local-first design and keep external services optional unless a feature specifically requires them.

## Before making changes

- Check the existing page and utility structure before adding new logic.
- Reuse shared utilities instead of duplicating data-processing logic.
- Do not persist API credentials.
- Keep generated datasets, models, reports, and local configuration out of commits.

## Development checks

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run the application boot check:

```bash
python bug_hunt.py
```

Run the serious-error lint checks:

```bash
ruff check --select=E9,F63,F7,F82,F821 --preview .
```

## Pull requests

Describe:

1. What changed.
2. Why it changed.
3. Tests and checks that were run.
4. Any limitations or environment-specific behavior.

Changes involving Gemini, S3, file parsing, credentials, or model persistence should include tests for failure cases where practical.

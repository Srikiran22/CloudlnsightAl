# Contributing

CloudInsight AI is primarily a portfolio project. Changes should keep the application simple, local-first, and consistent with the existing page/utility structure.

## Development setup

Follow the setup instructions in [README.md](README.md).

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the headless page check:

```bash
python bug_hunt.py
```

Run the serious-error lint check:

```bash
ruff check --select=E9,F63,F7,F82,F821 --preview .
```

## Repository conventions

- Put shared processing logic in `Utils/` rather than duplicating it in pages.
- Keep external integrations isolated in their utility modules.
- Do not persist Gemini or AWS credentials to disk.
- Keep generated datasets, models, reports, and local configuration out of commits.
- Add regression tests when fixing parsing, security, ML, or report-generation bugs.
- Keep README claims aligned with the current implementation.

## Pull requests

Describe:

- What changed.
- Why it changed.
- What tests/checks were run.
- Any limitations or environment-specific behavior.

For changes to file parsing, Gemini, S3, model persistence, or credential handling, include relevant failure-case tests where practical.

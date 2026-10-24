# Contributing

Contributions to CloudInsight AI are welcome. Changes should keep the application lightweight, local-first, and consistent with the existing modular structure.

## Development setup

Follow the setup instructions in [README.md](README.md).

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the headless page smoke check:

```bash
python -m unittest tests/test_page_smoke.py -v
```

Run the full lint check used by CI:

```bash
pip install ruff==0.16.4
ruff check .
```

## Repository conventions

- Put shared processing logic in `Utils/` rather than duplicating it in pages.
- Keep resource ceilings in `Utils/limits.py` and format readers in `Utils/parsing.py`; pages must not hard-code limits.
- Keep external integrations isolated in their utility modules.
- Keep the two theme palettes (`Utils/theme.py::_palette` and `.streamlit/config.toml`) in sync when changing colors.
- Do not persist API keys or credentials to disk.
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

# Contributing to TabuLens

Contributions to TabuLens are welcome. Changes should keep the application lightweight, local-first, mathematically sound, and consistent with the modular architecture.

---

## Development Setup

1. Follow the setup instructions in [README.md](README.md) and [docs/SETUP.md](docs/SETUP.md).
2. Run the verification pipeline and test suite:
   ```bash
   python scripts/verify_pipeline.py
   ```
3. Run the unit test discovery suite:
   ```bash
   python -m unittest discover -s tests -v
   ```

---

## Repository Conventions

- **Modular Seams:** Put computational algorithms and data transformations in `utils/` rather than embedding logic in presentation pages (`pages/`).
- **Resource Ceilings:** Keep resource limits in `utils/limits.py` and tabular parsers in `utils/paths.py`. Never hardcode raw limits in UI pages.
- **Leak-Free ML:** Feature transformers must always be encapsulated in Scikit-Learn `Pipeline` and `ColumnTransformer` objects to prevent data leakage.
- **Model Signatures:** Keep HMAC-SHA256 model signing intact for all model serialization.
- **Testing:** Include regression tests in `tests/` for any new calculation, parser, or security check.
- **Conventional Commits:** Keep commits imperative, lowercase, and concise (e.g. `feat: add feature interaction transformer`).

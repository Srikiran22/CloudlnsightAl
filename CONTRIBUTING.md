# Contributing to TabuLens

Contributions are welcome. Keep changes focused, local-first, well-tested, and consistent with existing module boundaries.

---

## Development Setup

1. Follow [docs/SETUP.md](docs/SETUP.md) to set up `.venv` and install `requirements.txt`.
2. Verify the environment and full test suite:
   ```bash
   python scripts/verify_pipeline.py
   ```
3. Check formatting and lint rules:
   ```bash
   ruff check .
   ```

---

## Conventions

- **Modular Seams:** Keep analytical, parsing, ML, and security logic inside `utils/` rather than embedding business logic in Streamlit UI pages (`pages/`).
- **Centralized Limits:** Define file, row, column, and memory bounds in `utils/limits.py` and path guards in `utils/paths.py`.
- **Leak-Free ML:** Keep preprocessing inside scikit-learn `Pipeline` and `ColumnTransformer` estimators rather than fitting transforms before train/test splitting.
- **Model Integrity:** Preserve HMAC-SHA256 signing and in-memory signature verification for all `.joblib` persistence (`utils/model_security.py`, `utils/ml.py`).
- **ADRs:** Document non-trivial architectural changes by adding a numbered record in `docs/decisions/` following [docs/decisions/README.md](docs/decisions/README.md).
- **Tests:** Add or update `unittest` regression coverage in `tests/` for any bug fix, parser change, statistical metric, or security boundary.
- **Commits:** Use concise imperative commit messages (for example, `fix: handle empty groups in anova summary`).

# ADR-001: Local-First Tabular Workbench Over Cloud GenAI Assistant

## Status
Accepted

## Date
2026-11-16

## Context
The repository originated as CloudInsight AI with external `google-genai` integration, LLM-driven unstructured-to-CSV conversion, and an in-process AST execution sandbox (`utils/gemini.py`, `utils/ai_sandbox.py`, `pages/aichat.py`, `pages/aiconverter.py`). Those features introduced external API key requirements, network failure modes, nondeterministic outputs, and an unnecessary code-execution attack surface inside a local Streamlit process.

The core value of the application lies in deterministic tabular workflows: multi-format ingestion, data hygiene checks, statistical diagnostics, drift detection, scikit-learn modeling, and portable batch inference.

## Decision
Remove all external LLM API dependencies (`google-genai`), `pypdf` runtime conversion workflows, and the in-process AST execution sandbox (`dd76917`). Standardize the application as **TabuLens** (`pyproject.toml`, `app.py`, `pages/`, `utils/`), a 100% local, offline-capable tabular analysis and modeling workbench built on Python, Streamlit, pandas, SciPy, scikit-learn, Plotly, and ReportLab.

Legacy compatibility shims (`AIConversionRequired` in `utils/limits.py`, `.conversions.json` provenance helpers in `utils/paths.py`, and the optional `include_ai_insights` argument in `utils/pdf.py`) are retained where internal utilities and regression tests rely on those boundaries, while live UI pages operate solely on structured local data.

## Alternatives Considered

### Keep Gemini API Chat and AST Code Sandbox
- Pros: Conversational data exploration and unstructured PDF/text parsing via external LLMs.
- Cons: Breaks offline execution, requires API credentials, leaks local dataset snippets to external endpoints, and creates maintenance and security exposure around `exec()` sandboxing.
- Rejected: Undermines local-first privacy and deterministic behavior.

### Separate React Frontend + FastAPI Backend
- Pros: Independent frontend/backend deployment and richer async job management.
- Cons: Substantial boilerplate overhead for a single-user local workstation tool; duplicates state management and charting wrappers that Streamlit + Plotly handle natively.
- Rejected: Out of proportion to a single-user local analytics workbench.

## Consequences
- Every workflow in `app.py` and `pages/*.py` executes locally with zero internet access or API keys.
- Non-tabular text and PDF uploads surface a raw text preview via `AIConversionRequired` instead of converting unstructured documents automatically.
- `utils/pdf.py::generate_pdf_report` retains `include_ai_insights: str | None = None` for API compatibility, though `pages/report.py` passes `None`.

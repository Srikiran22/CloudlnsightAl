"""Shared resource limits, storage roots, and the supported-format contract.

One source of truth for the numbers enforced by the ingestion layer and surfaced
in the UI. Import them from here (or from ``Utils.paths``, which re-exports them
for backwards compatibility); never copy a literal limit into a page.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
REPORTS_DIR = PROJECT_ROOT / "Reports"
MODELS_DIR = PROJECT_ROOT / "Models"
REPORT_TEMPLATES_DIR = REPORTS_DIR / "templates"
CONVERSIONS_MANIFEST = DATASETS_DIR / ".conversions.json"

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")

# uploads beyond this are refused before parsing (matches Streamlit's own
# 200 MB default ceiling); raise it if your machine has memory to spare
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_INGESTION_ROWS = 1_000_000
MAX_INGESTION_COLUMNS = 200
MAX_INGESTION_CELLS = 20_000_000
MAX_UPLOAD_FILES = 20
MAX_AGGREGATE_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_COMBINED_ROWS = 1_000_000


class ResourcePolicy:
    """Centralized resource boundaries and memory multipliers for CloudInsight AI."""
    MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES
    MAX_INGESTION_ROWS = MAX_INGESTION_ROWS
    MAX_INGESTION_COLUMNS = MAX_INGESTION_COLUMNS
    MAX_INGESTION_CELLS = MAX_INGESTION_CELLS
    MAX_UPLOAD_FILES = MAX_UPLOAD_FILES
    MAX_AGGREGATE_UPLOAD_BYTES = MAX_AGGREGATE_UPLOAD_BYTES
    MAX_COMBINED_ROWS = MAX_COMBINED_ROWS
    MAX_VISUALIZATION_ROWS = 25_000
    MAX_ANALYSIS_ROWS = 50_000
    MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024
    MAX_DATAFRAME_MEMORY_BYTES = 400 * 1024 * 1024  # 400 MB deep in-memory ceiling
    MAX_HIGH_CARDINALITY_STRING_CELLS = 5_000_000   # 5M high-cardinality string cells early guard


TABULAR_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson",
    ".parquet", ".xml", ".html", ".htm",
}
TEXT_EXTENSIONS = {".txt", ".log", ".md", ".rst", ".sql"}
DOCUMENT_EXTENSIONS = {".pdf"}

SUPPORTED_DATASET_EXTENSIONS = TABULAR_EXTENSIONS | TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS


class AIConversionRequired(ValueError):
    # a file we can't parse natively; raw_text carries what Gemini would see
    def __init__(self, message, raw_text="", filename=""):
        super().__init__(message)
        self.raw_text = raw_text
        self.filename = filename

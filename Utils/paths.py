import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
REPORTS_DIR = PROJECT_ROOT / "Reports"
MODELS_DIR = PROJECT_ROOT / "Models"
REPORT_TEMPLATES_DIR = REPORTS_DIR / "templates"

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")

# uploads beyond this are refused before parsing (matches Streamlit's own
# 200 MB default ceiling); raise it if your machine has memory to spare
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

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


def ensure_project_directories():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def list_dataset_files():
    ensure_project_directories()
    return sorted(
        path.name
        for path in DATASETS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS
    )


def resolve_dataset_path(dataset):
    path = Path(dataset)
    datasets_root = DATASETS_DIR.resolve()
    candidate = path.resolve() if path.is_absolute() else (datasets_root / path).resolve()

    if candidate.parent != datasets_root:
        raise ValueError("Datasets must be read from the project's Datasets directory.")

    return candidate


def _decode_text_buffer(buffer):
    for encoding in CSV_ENCODINGS:
        try:
            buffer.seek(0)
            return buffer.read().decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    buffer.seek(0)
    return buffer.read().decode("utf-8", errors="replace")


def _read_csv_from_buffer(buffer):
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Unable to decode CSV file.")


def _read_delimited_text(text):
    """Native text-table policy.

    A .txt/.log/.md/.rst/.sql file parses natively ONLY when csv.Sniffer
    identifies one of the known field separators (comma, semicolon, tab,
    pipe, colon) used consistently across the sampled lines. Whitespace is
    deliberately NOT a candidate delimiter, so prose never becomes a junk
    wide table -- unparseable text falls through to AIConversionRequired and
    the Gemini path. (A line-per-record prose file with exactly one comma on
    every line is genuinely ambiguous and still parses; that is inherent.)
    """
    sample_lines = [line for line in text.splitlines() if line.strip()][:20]
    if len(sample_lines) < 2:
        return None

    try:
        dialect = csv.Sniffer().sniff("\n".join(sample_lines), delimiters=",;\t|:")
        delimiter = dialect.delimiter
    except csv.Error:
        return None

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
    except Exception:
        return None
    if df.shape[1] < 2:
        return None
    # a separator that only ever appears at end-of-line produces a ghost
    # column of empty cells (e.g. SQL statements ending in ';'); that is
    # structure noise, not a table
    if df.isna().all().any():
        return None
    return df


def _read_json_from_buffer(buffer):
    buffer.seek(0)
    try:
        df = pd.read_json(buffer)
        has_nested = any(
            df[column].map(lambda value: isinstance(value, (dict, list))).any()
            for column in df.columns
        )
        if not has_nested:
            return df
    except ValueError:
        pass
    buffer.seek(0)
    data = json.loads(_decode_text_buffer(buffer))
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("JSON file does not contain a record list.")
    return pd.json_normalize(data)


def _read_jsonl_from_buffer(buffer):
    buffer.seek(0)
    return pd.read_json(buffer, lines=True)


def _flatten_xml_element(element, parent_key=""):
    record = {}
    for attribute, value in element.attrib.items():
        record[f"{parent_key}{element.tag}@{attribute}".strip("_")] = value

    text = (element.text or "").strip()
    children = list(element)
    if text and not children:
        record[f"{parent_key}{element.tag}".strip("_")] = text
    for child in children:
        record.update(_flatten_xml_element(child, parent_key=f"{parent_key}{element.tag}_"))
    return record


# stdlib expat resolves internal entities, so XML input is vulnerable to
# exponential "billion laughs" expansion and quadratic text blowup; external
# entities are never fetched (no network), but a DTD is never needed for
# tabular data -- reject it outright before parsing
MAX_XML_BYTES = MAX_UPLOAD_BYTES


def _reject_dtd(payload):
    if b"<!DOCTYPE" in payload[:65536].upper() or b"<!ENTITY" in payload[:65536].upper():
        raise ValueError(
            "XML datasets must not contain DTD/entity declarations "
            "(they enable resource-exhaustion attacks); export plain elements."
        )


def _read_xml_from_buffer(buffer):
    buffer.seek(0)
    payload = buffer.read()
    _reject_dtd(payload)
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        raise
    except RecursionError as error:
        raise ValueError("XML nesting is too deep to process.") from error

    try:
        records = []
        for element in list(root):
            flat = _flatten_xml_element(element)
            # drop the leading "<roottag>_" from each key
            prefix = f"{element.tag}_"
            flat = {
                (key[len(prefix):] if key.startswith(prefix) else key): value
                for key, value in flat.items()
            }
            records.append(flat)
    except RecursionError as error:
        raise ValueError("XML nesting is too deep to process.") from error
    if not records:
        records.append(_flatten_xml_element(root))
    return pd.json_normalize(records)


class _HTMLTableParser(HTMLParser):
    # grabs the first <table> with at least a header row plus one data row
    def __init__(self):
        super().__init__()
        self.tables = []
        self._table = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self._table is not None:
            if len(self._table) >= 2:
                self.tables.append(self._table)
            self._table = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag in {"td", "th"} and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _read_html_from_buffer(buffer):
    html_text = _decode_text_buffer(buffer)
    parser = _HTMLTableParser()
    parser.feed(html_text)

    if not parser.tables:
        raise AIConversionRequired(
            "No HTML table found; AI conversion required.",
            raw_text=html_text,
        )

    best_table = max(parser.tables, key=len)
    header = best_table[0]
    rows = best_table[1:]
    width = max(len(row) for row in best_table)
    header += [f"column_{i}" for i in range(len(header), width)]
    # repeated <th> text would create duplicate DataFrame columns, and
    # df[name] on a duplicated column returns a DataFrame -- breaking every
    # downstream page; de-duplicate instead (city, city_2, ...)
    seen = {}
    unique_header = []
    for name in header:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_header.append(name if count == 0 else f"{name}_{count + 1}")
    normalized = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(normalized, columns=unique_header)


def _read_parquet_from_buffer(buffer):
    buffer.seek(0)
    return pd.read_parquet(buffer)


def _read_pdf_from_buffer(buffer, filename=""):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ValueError(
            "PDF ingestion requires the optional 'pypdf' package (pip install pypdf)."
        ) from error

    buffer.seek(0)
    reader = PdfReader(buffer)
    pages_text = [(page.extract_text() or "") for page in reader.pages]
    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise ValueError("No extractable text found in the PDF (it may be scanned images).")

    # pdfs never parse into tables directly, always hand off to AI
    raise AIConversionRequired(
        "PDF content requires AI structuring.",
        raw_text=full_text,
        filename=filename,
    )


def read_tabular(source, filename=None):
    """Read any supported dataset from a path, file-like object, or bytes.

    Formats without a native tabular structure raise AIConversionRequired carrying
    the extracted raw text so callers can offer Gemini-powered conversion.
    """
    if isinstance(source, (str, Path)) and filename is None:
        return read_dataset(source)

    name = Path(filename or getattr(source, "name", "dataset.csv")).name
    suffix = Path(name).suffix.lower()

    if isinstance(source, (bytes, bytearray)):
        buffer = io.BytesIO(source)
    else:
        buffer = source
        if hasattr(buffer, "seek"):
            buffer.seek(0)

    if suffix == ".csv" or suffix == "":
        return _read_csv_from_buffer(buffer)

    if suffix == ".tsv":
        return pd.read_csv(buffer, sep="\t")

    if suffix in {".xlsx", ".xls"}:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        return pd.read_excel(buffer)

    if suffix == ".json":
        return _read_json_from_buffer(buffer)

    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_from_buffer(buffer)

    if suffix == ".parquet":
        return _read_parquet_from_buffer(buffer)

    if suffix == ".xml":
        return _read_xml_from_buffer(buffer)

    if suffix in {".html", ".htm"}:
        return _read_html_from_buffer(buffer)

    if suffix in TEXT_EXTENSIONS:
        text = _decode_text_buffer(buffer)
        df = _read_delimited_text(text)
        if df is None:
            raise AIConversionRequired(
                "Plain text could not be parsed as a table; AI conversion required.",
                raw_text=text,
                filename=name,
            )
        return df

    if suffix in DOCUMENT_EXTENSIONS:
        return _read_pdf_from_buffer(buffer, filename=name)

    raise AIConversionRequired(
        f"Unsupported dataset format: {suffix or 'none'}",
        raw_text="",
        filename=name,
    )


def read_dataset(dataset):
    path = resolve_dataset_path(dataset)
    with path.open("rb") as handle:
        return read_tabular(handle, filename=path.name)

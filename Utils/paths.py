import csv
import hashlib
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
CONVERSIONS_MANIFEST = DATASETS_DIR / ".conversions.json"

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


def safe_stem(value, fallback="model"):
    """Reduce a user-supplied name to a single safe path component.

    Keeps letters/digits/space/-/_ only (no separators, no traversal), so a
    crafted name can never escape its target directory. Falls back when
    nothing survives the filter.
    """
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_ ").strip()
    return cleaned or fallback


def get_unique_filename(filename, directory=DATASETS_DIR, extra_names=None):
    """Ensure a filename does not collide with existing files on disk or extra names in memory.

    Generates deterministic names:
      data.csv -> data_1.csv -> data_2.csv
    """
    target_dir = Path(directory)
    extra = set(extra_names or [])
    name = Path(filename).name
    stem = Path(name).stem
    suffix = Path(name).suffix

    candidate = name
    counter = 1
    while (target_dir / candidate).exists() or candidate in extra:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def enforce_size_limit(size_in_bytes, label="File"):
    """Centralized enforcement of the MAX_UPLOAD_BYTES resource limit."""
    if size_in_bytes is not None and size_in_bytes > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{label} is {size_in_bytes / (1024 * 1024):.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Split or trim the file."
        )


def record_conversion(converted_name, source_name, raw_text):
    """Record provenance metadata for an AI-converted file in the manifest."""
    ensure_project_directories()
    manifest = {}
    if CONVERSIONS_MANIFEST.exists():
        try:
            manifest = json.loads(CONVERSIONS_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    manifest[converted_name] = {
        "source_name": source_name,
        "source_hash": source_hash,
        "source_length": len(raw_text),
    }
    CONVERSIONS_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def get_valid_conversion(source_name, raw_text):
    """Return existing converted filename if it matches this source content, else None."""
    if not CONVERSIONS_MANIFEST.exists():
        return None
    try:
        manifest = json.loads(CONVERSIONS_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return None
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    for conv_name, meta in manifest.items():
        if (
            meta.get("source_name") == source_name
            and meta.get("source_hash") == source_hash
            and (DATASETS_DIR / conv_name).exists()
        ):
            return conv_name
    return None


def list_dataset_files():
    ensure_project_directories()
    return sorted(
        path.name
        for path in DATASETS_DIR.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS
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


def _sanitize_unhashable_cells(df):
    """Serialize list/dict cells into deterministic JSON strings so downstream
    hashing, grouping, duplicated, and unique operations work without TypeError."""
    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, (list, dict))).any():
            df[column] = df[column].map(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
            )
    return df


def _read_csv_from_buffer(buffer, nrows=None):
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding=encoding, nrows=nrows)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Unable to decode CSV file.")


def _read_delimited_text(text, nrows=None):
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
        df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python", nrows=nrows)
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


def _read_json_from_buffer(buffer, nrows=None):
    buffer.seek(0)
    try:
        text = _decode_text_buffer(buffer)
        data = json.loads(text)
        if isinstance(data, dict):
            list_keys = [k for k, v in data.items() if isinstance(v, list)]
            if len(list_keys) > 1:
                raise ValueError(
                    f"Ambiguous JSON structure: multiple top-level record lists found ({', '.join(sorted(list_keys))}). "
                    "Extract the target list before ingestion."
                )
            if len(list_keys) == 1:
                data = data[list_keys[0]]
        if isinstance(data, list):
            norm_df = pd.json_normalize(data)
            if nrows is not None and len(norm_df) > nrows:
                norm_df = norm_df.head(nrows)
            return _sanitize_unhashable_cells(norm_df)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    buffer.seek(0)
    try:
        df = pd.read_json(buffer)
        if nrows is not None and len(df) > nrows:
            df = df.head(nrows)
        return _sanitize_unhashable_cells(df)
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON dataset: {exc}") from exc


def _read_jsonl_from_buffer(buffer, nrows=None):
    buffer.seek(0)
    df = pd.read_json(buffer, lines=True, nrows=nrows)
    return _sanitize_unhashable_cells(df)


def _flatten_xml_element(element, parent_key="", index=None):
    record = {}
    tag_suffix = f"_{index}" if index is not None else ""
    current_key = f"{parent_key}{element.tag}{tag_suffix}".strip("_")

    for attribute, value in element.attrib.items():
        record[f"{current_key}@{attribute}"] = value

    text = (element.text or "").strip()
    children = list(element)
    if text and not children:
        record[current_key] = text

    tag_counts = {}
    for child in children:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1

    tag_seen = {}
    for child in children:
        tag = child.tag
        child_index = None
        if tag_counts[tag] > 1:
            tag_seen[tag] = tag_seen.get(tag, 0) + 1
            child_index = tag_seen[tag]
        record.update(_flatten_xml_element(child, parent_key=f"{current_key}_", index=child_index))
    return record


# stdlib expat resolves internal entities, so XML input is vulnerable to
# exponential "billion laughs" expansion and quadratic text blowup; external
# entities are never fetched (no network), but a DTD is never needed for
# tabular data -- reject it outright before parsing


def _reject_dtd(payload):
    # the whole document is scanned: comments/PIs may legally precede the
    # DOCTYPE and push it past any fixed-size inspection window. XML keywords
    # are case-sensitive per spec, so an exact-case search is complete and
    # avoids copying large payloads.
    if b"<!DOCTYPE" in payload or b"<!ENTITY" in payload:
        raise ValueError(
            "XML datasets must not contain DTD/entity declarations "
            "(they enable resource-exhaustion attacks); export plain elements."
        )


def _read_xml_from_buffer(buffer, nrows=None):
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
                key.removeprefix(prefix): value
                for key, value in flat.items()
            }
            records.append(flat)
    except RecursionError as error:
        raise ValueError("XML nesting is too deep to process.") from error
    if not records:
        records.append(_flatten_xml_element(root))
    df = pd.json_normalize(records)
    if nrows is not None and len(df) > nrows:
        df = df.head(nrows)
    return _sanitize_unhashable_cells(df)


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


MAX_PDF_PAGES = 30
MAX_PDF_EXTRACT_CHARS = 50_000


def _read_pdf_from_buffer(buffer, filename=""):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ValueError(
            "PDF ingestion requires the optional 'pypdf' package (pip install pypdf)."
        ) from error

    buffer.seek(0)
    reader = PdfReader(buffer)
    pages_text = []
    total_chars = 0
    for idx, page in enumerate(reader.pages):
        if idx >= MAX_PDF_PAGES or total_chars >= MAX_PDF_EXTRACT_CHARS:
            break
        text = page.extract_text() or ""
        pages_text.append(text)
        total_chars += len(text)

    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise ValueError("No extractable text found in the PDF (it may be scanned images).")

    # pdfs never parse into tables directly, always hand off to AI
    raise AIConversionRequired(
        "PDF content requires AI structuring.",
        raw_text=full_text,
        filename=filename,
    )


def read_tabular(source, filename=None, max_rows=None):
    """Read any supported dataset from a path, file-like object, or bytes.

    Formats without a native tabular structure raise AIConversionRequired carrying
    the extracted raw text so callers can offer Gemini-powered conversion.
    """
    if isinstance(source, (str, Path)) and filename is None:
        return read_dataset(source, max_rows=max_rows)

    name = Path(filename or getattr(source, "name", "dataset.csv")).name
    suffix = Path(name).suffix.lower()

    if hasattr(source, "size"):
        enforce_size_limit(source.size, label=f"File '{name}'")
    elif isinstance(source, (bytes, bytearray)):
        enforce_size_limit(len(source), label=f"File '{name}'")

    if isinstance(source, (bytes, bytearray)):
        buffer = io.BytesIO(source)
    else:
        buffer = source
        if hasattr(buffer, "seek"):
            buffer.seek(0)

    if suffix == ".csv" or suffix == "":
        return _read_csv_from_buffer(buffer, nrows=max_rows)

    if suffix == ".tsv":
        return pd.read_csv(buffer, sep="\t", nrows=max_rows)

    if suffix in {".xlsx", ".xls"}:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        try:
            return pd.read_excel(buffer, nrows=max_rows)
        except ImportError as exc:
            if suffix == ".xls" and "xlrd" in str(exc).lower():
                raise ValueError(
                    "Legacy .xls Excel format requires the 'xlrd' package (pip install xlrd>=2.0.1). "
                    "Alternatively, save or convert the file to modern .xlsx or .csv."
                ) from exc
            raise

    if suffix == ".json":
        return _read_json_from_buffer(buffer, nrows=max_rows)

    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_from_buffer(buffer, nrows=max_rows)

    if suffix == ".parquet":
        buffer.seek(0)
        df = pd.read_parquet(buffer)
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)
        return df

    if suffix == ".xml":
        return _read_xml_from_buffer(buffer, nrows=max_rows)

    if suffix in {".html", ".htm"}:
        df = _read_html_from_buffer(buffer)
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)
        return df

    if suffix in TEXT_EXTENSIONS:
        text = _decode_text_buffer(buffer)
        df = _read_delimited_text(text, nrows=max_rows)
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


def read_dataset(dataset, max_rows=None):
    path = resolve_dataset_path(dataset)
    enforce_size_limit(path.stat().st_size, label=f"Dataset '{path.name}'")
    with path.open("rb") as handle:
        return read_tabular(handle, filename=path.name, max_rows=max_rows)

import csv
import hashlib
import io
import json
import os
import time
import uuid
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree
import xml.parsers.expat
import zipfile

import numpy as np
import pandas as pd


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
MAX_UPLOAD_FILES = 20
MAX_AGGREGATE_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_COMBINED_ROWS = 1_000_000

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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


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


def _check_zip_bomb(buffer, max_uncompressed_bytes=MAX_UPLOAD_BYTES, max_ratio=100.0):
    """Inspect zip archive members (e.g. XLSX) for dangerous expansion ratios."""
    try:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as zf:
            total_compressed = 0
            total_uncompressed = 0
            for info in zf.infolist():
                total_compressed += info.compress_size
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise ValueError(
                        f"Decompressed archive size ({total_uncompressed / (1024 * 1024):.1f} MB) "
                        f"exceeds safety limit of {max_uncompressed_bytes // (1024 * 1024)} MB."
                    )
            if total_compressed > 0 and (total_uncompressed / total_compressed) > max_ratio and total_uncompressed > 50 * 1024 * 1024:
                raise ValueError(
                    f"Decompressed archive expansion ratio ({total_uncompressed / total_compressed:.1f}x) "
                    "exceeds safety limit (possible zip bomb)."
                )
    except (zipfile.BadZipFile, OSError):
        pass
    finally:
        if hasattr(buffer, "seek"):
            buffer.seek(0)


def atomic_write(target_path, content, mode="w", encoding="utf-8"):
    """Atomically write content to target_path via temporary file and os.replace."""
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f"{p.name}.tmp.{uuid.uuid4().hex}")
    try:
        if "b" in mode:
            with open(tmp_path, mode) as f:
                f.write(content)
        else:
            with open(tmp_path, mode, encoding=encoding) as f:
                f.write(content)
        for attempt in range(10):
            try:
                os.replace(tmp_path, p)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.005 * (2 ** attempt))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


CONVERSION_MANIFEST_VERSION = "v1"


def compute_conversion_provenance(raw_text, model_name=None, extra_instructions=None):
    """Deterministic hash incorporating source content, model, instructions, and version."""
    norm_model = (model_name or "").strip().lower()
    norm_instr = (extra_instructions or "").strip()
    hasher = hashlib.sha256()
    hasher.update(raw_text.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(norm_model.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(norm_instr.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(CONVERSION_MANIFEST_VERSION.encode("utf-8"))
    return hasher.hexdigest()[:32]


def record_conversion(
    converted_name,
    source_name,
    raw_text,
    model_name=None,
    extra_instructions=None,
    manifest_path=None,
    datasets_dir=None,
):
    """Record provenance metadata for an AI-converted file in the manifest."""
    ensure_project_directories()
    manifest_file = Path(manifest_path) if manifest_path is not None else CONVERSIONS_MANIFEST
    manifest = {}
    if manifest_file.exists():
        try:
            loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except Exception:
            manifest = {}
    prov_hash = compute_conversion_provenance(raw_text, model_name=model_name, extra_instructions=extra_instructions)
    target_dir = Path(datasets_dir) if datasets_dir is not None else (
        manifest_file.parent if (manifest_file.parent / converted_name).is_file()
        else (manifest_file.parent / "Datasets" if (manifest_file.parent / "Datasets" / converted_name).is_file() else DATASETS_DIR)
    )
    conv_path = target_dir / converted_name
    output_hash = ""
    if conv_path.is_file():
        hasher = hashlib.sha256()
        with conv_path.open("rb") as f:
            while chunk := f.read(64 * 1024):
                hasher.update(chunk)
        output_hash = hasher.hexdigest()

    manifest[converted_name] = {
        "source_name": source_name,
        "provenance_hash": prov_hash,
        "model_name": (model_name or "").strip(),
        "extra_instructions": (extra_instructions or "").strip(),
        "extra_instructions_hash": (
            hashlib.sha256((extra_instructions or "").strip().encode("utf-8")).hexdigest()[:16]
            if extra_instructions
            else ""
        ),
        "output_hash": output_hash,
        "version": CONVERSION_MANIFEST_VERSION,
        "source_length": len(raw_text),
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_file.with_name(f"{manifest_file.name}.tmp.{os.getpid()}")
    try:
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_manifest, manifest_file)
    except Exception:
        if tmp_manifest.exists():
            try:
                tmp_manifest.unlink()
            except OSError:
                pass
        raise


def get_valid_conversion(
    source_name,
    raw_text,
    model_name=None,
    extra_instructions=None,
    manifest_path=None,
    datasets_dir=None,
):
    """Return existing converted filename if it matches source content, model, and instructions, else None."""
    manifest_file = Path(manifest_path) if manifest_path is not None else CONVERSIONS_MANIFEST
    target_dir = Path(datasets_dir) if datasets_dir is not None else DATASETS_DIR
    if not manifest_file.exists():
        return None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
    except Exception:
        return None

    norm_source = Path(source_name).name
    req_instr = (extra_instructions or "").strip()
    req_model = (model_name or "").strip().lower() if model_name is not None else None

    for conv_name, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("version") != CONVERSION_MANIFEST_VERSION:
            continue
        if meta.get("source_name") != norm_source and meta.get("source_name") != source_name:
            continue

        meta_model = (meta.get("model_name") or "").strip().lower()
        if req_model is not None and meta_model != req_model:
            continue

        meta_instr = (meta.get("extra_instructions") or "").strip()
        if meta_instr != req_instr:
            continue

        entry_model = meta.get("model_name", "")
        expected_prov = compute_conversion_provenance(raw_text, model_name=entry_model, extra_instructions=req_instr)
        if meta.get("provenance_hash") != expected_prov:
            continue

        conv_file = target_dir / conv_name
        if not conv_file.is_file():
            continue

        expected_out_hash = meta.get("output_hash")
        if expected_out_hash:
            hasher = hashlib.sha256()
            with conv_file.open("rb") as f:
                while chunk := f.read(64 * 1024):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected_out_hash:
                continue
        return conv_name
    return None


def get_dataset_row_count(dataset):
    """Lightweight determination of exact source row count without full dataset materialization."""
    path = resolve_dataset_path(dataset)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                if header is None:
                    return 0
                return sum(1 for row in reader if row and any(cell.strip() for cell in row))
        except Exception:
            pass

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(path)
            return meta.num_rows
        except Exception:
            pass

    if suffix in {".jsonl", ".ndjson"}:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            pass

    if suffix in {".xlsx", ".xls"}:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True)
            sheet = wb.worksheets[0] if wb.worksheets else wb.active
            rows = max(sheet.max_row - 1, 0) if sheet.max_row is not None else None
            wb.close()
            if rows is not None:
                return rows
        except Exception:
            pass

    try:
        df = read_dataset(dataset)
        return len(df)
    except Exception:
        return None


def list_dataset_files(directory=DATASETS_DIR):
    target_dir = Path(directory)
    if target_dir == DATASETS_DIR:
        ensure_project_directories()
    datasets_root = target_dir.resolve()
    valid_files = []
    for path in target_dir.iterdir():
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS:
            try:
                if path.resolve().parent == datasets_root:
                    valid_files.append(path.name)
            except OSError:
                continue
    return sorted(valid_files)


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


def _serialize_cell(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, set):
        return json.dumps(sorted(list(value)))
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def normalize_and_deduplicate_columns(df):
    """Normalize column names to stripped strings and deduplicate collisions (col, col_2, ...).
    Eliminates mixed-type sorting/escaping crashes and DataFrame-returning duplicate indexing.
    """
    if df is None or not hasattr(df, "columns"):
        return df
    used_names = set()
    counts = {}
    unique_columns = []
    for col in df.columns:
        name = str(col).strip()
        if not name:
            name = "unnamed"
        if name not in used_names:
            candidate = name
            counts[name] = 1
        else:
            count = counts.get(name, 1) + 1
            candidate = f"{name}_{count}"
            while candidate in used_names:
                count += 1
                candidate = f"{name}_{count}"
            counts[name] = count
        used_names.add(candidate)
        unique_columns.append(candidate)
    df.columns = unique_columns
    return df


def _sanitize_unhashable_cells(df):
    """Serialize list/dict/array cells into deterministic JSON strings so downstream
    hashing, grouping, duplicated, and unique operations work without TypeError."""
    for column in df.columns:
        col_data = df[column]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        if col_data.dtype == object or pd.api.types.is_string_dtype(col_data):
            if col_data.map(lambda value: isinstance(value, (list, dict, set, np.ndarray))).any():
                df[column] = col_data.map(_serialize_cell)
    return df


def _sanitize_formula_val(val):
    if not isinstance(val, str):
        return val
    if not val:
        return val
    if val[0] in ("\t", "\r"):
        return f"'{val}"
    s = val.strip()
    if not s:
        return val
    if s[0] in ("=", "+", "-", "@"):
        try:
            float(s)
            return val
        except ValueError:
            return f"'{val}"
    return val


def sanitize_for_csv_export(df):
    """Prepend single quote to string cells and column headers starting with formula triggers
    (=, +, -, @, \\t, \\r) unless the value is a valid numeric literal.
    Neutralizes CSV formula injection (DDE) when downloaded CSV is opened in Excel/Calc.
    """
    if df is None:
        return None
    df_out = df.copy()
    df_out.columns = [_sanitize_formula_val(str(col)) for col in df_out.columns]
    for col in df_out.columns:
        if df_out[col].dtype == object or pd.api.types.is_string_dtype(df_out[col]):
            df_out[col] = df_out[col].map(_sanitize_formula_val)
    return df_out


def _inspect_delimited_header_width(buffer, encoding, sep=","):
    """Quickly inspect header width before full parsing to reject wide files early."""
    buffer.seek(0)
    try:
        hdr = pd.read_csv(buffer, sep=sep, encoding=encoding, nrows=0)
        if len(hdr.columns) > MAX_INGESTION_COLUMNS:
            raise ValueError(
                f"Dataset contains {len(hdr.columns):,} columns, which exceeds the maximum supported limit "
                f"of {MAX_INGESTION_COLUMNS} columns. Filter or transpose the file before ingestion."
            )
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        pass
    finally:
        buffer.seek(0)


def _read_csv_from_buffer(buffer, nrows=None):
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            _inspect_delimited_header_width(buffer, encoding, sep=",")
            buffer.seek(0)
            return pd.read_csv(buffer, encoding=encoding, nrows=nrows)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Unable to decode CSV file.")


def _read_tsv_from_buffer(buffer, nrows=None):
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            _inspect_delimited_header_width(buffer, encoding, sep="\t")
            buffer.seek(0)
            return pd.read_csv(buffer, sep="\t", encoding=encoding, nrows=nrows)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Unable to decode TSV file.")


def _read_delimited_text(text, nrows=None):
    """Native text-table policy.

    A .txt/.log/.md/.rst/.sql file parses natively ONLY when csv.Sniffer
    identifies one of the known field separators (comma, semicolon, tab,
    pipe) used consistently across the sampled lines. Whitespace and colon are
    deliberately NOT candidate delimiters, so logs and prose never become a junk
    wide table -- unparseable text falls through to AIConversionRequired and
    the Gemini path. (A line-per-record prose file with exactly one comma on
    every line is genuinely ambiguous and still parses; that is inherent.)
    """
    sample_lines = [line for line in text.splitlines() if line.strip()][:20]
    if len(sample_lines) < 2:
        return None

    try:
        dialect = csv.Sniffer().sniff("\n".join(sample_lines), delimiters=",;\t|")
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


def _stream_json_array(text, nrows):
    """Incremental decoder for JSON root arrays so only nrows objects are materialized."""
    s_text = text.lstrip()
    if not s_text.startswith("["):
        return None
    decoder = json.JSONDecoder()
    idx = text.find("[") + 1
    length = len(text)
    records = []
    while idx < length and len(records) < nrows:
        while idx < length and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= length or text[idx] == "]":
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
            records.append(obj)
            idx = next_idx
        except ValueError:
            return None
    return records


def _read_json_from_buffer(buffer, nrows=None):
    buffer.seek(0)
    try:
        text = _decode_text_buffer(buffer)
        if nrows is not None and nrows > 0:
            streamed = _stream_json_array(text, nrows)
            if streamed is not None:
                if isinstance(streamed, list):
                    unique_keys = set()
                    for item in streamed:
                        if isinstance(item, dict):
                            unique_keys.update(item.keys())
                            if len(unique_keys) > MAX_INGESTION_COLUMNS:
                                raise ValueError(
                                    f"JSON dataset has {len(unique_keys):,}+ unique column keys, "
                                    f"exceeding the maximum limit of {MAX_INGESTION_COLUMNS} columns."
                                )
                try:
                    norm_df = pd.json_normalize(streamed)
                    if len(streamed) > 0 and norm_df.shape[1] == 0:
                        norm_df = pd.DataFrame(streamed)
                except TypeError:
                    norm_df = pd.DataFrame(streamed)
                return _sanitize_unhashable_cells(norm_df)

        data = json.loads(text)
        if isinstance(data, dict):
            # Check for columnar table: all values are lists of equal length > 0 and contain scalars
            if (
                len(data) > 0
                and all(isinstance(v, list) for v in data.values())
                and len({len(v) for v in data.values()}) == 1
                and all(not any(isinstance(x, dict) for x in v) for v in data.values())
            ):
                if len(data) > MAX_INGESTION_COLUMNS:
                    raise ValueError(
                        f"JSON dataset has {len(data):,} columns, "
                        f"exceeding the maximum limit of {MAX_INGESTION_COLUMNS} columns."
                    )
                df = pd.DataFrame(data)
                if nrows is not None and len(df) > nrows:
                    df = df.head(nrows)
                return _sanitize_unhashable_cells(df)

            # Check for record lists wrapped inside a dict
            record_list_keys = [
                k for k, v in data.items()
                if isinstance(v, list) and (len(v) == 0 or any(isinstance(x, dict) for x in v))
            ]
            if len(record_list_keys) > 1:
                raise ValueError(
                    f"Ambiguous JSON structure: multiple top-level record lists found ({', '.join(sorted(record_list_keys))}). "
                    "Extract the target list before ingestion."
                )
            if len(record_list_keys) == 1:
                data = data[record_list_keys[0]]
            else:
                # Single record object: normalize as a 1-row table
                data = [data]

        if isinstance(data, list):
            unique_keys = set()
            for item in data:
                if isinstance(item, dict):
                    unique_keys.update(item.keys())
                    if len(unique_keys) > MAX_INGESTION_COLUMNS:
                        raise ValueError(
                            f"JSON dataset has {len(unique_keys):,}+ unique column keys, "
                            f"exceeding the maximum limit of {MAX_INGESTION_COLUMNS} columns."
                        )
            if nrows is not None and len(data) > nrows:
                data = data[:nrows]
            try:
                norm_df = pd.json_normalize(data)
                if len(data) > 0 and norm_df.shape[1] == 0:
                    norm_df = pd.DataFrame(data)
            except TypeError:
                norm_df = pd.DataFrame(data)
            return _sanitize_unhashable_cells(norm_df)
    except ValueError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        pass

    buffer.seek(0)
    try:
        df = pd.read_json(buffer)
        if nrows is not None and len(df) > nrows:
            df = df.head(nrows)
        return _sanitize_unhashable_cells(df)
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON dataset: {exc}") from exc


def _inspect_jsonl_header_width(buffer):
    try:
        buffer.seek(0)
        first_line = buffer.readline()
        if first_line:
            first_obj = json.loads(first_line.decode("utf-8", errors="replace"))
            if isinstance(first_obj, dict) and len(first_obj) > MAX_INGESTION_COLUMNS:
                raise ValueError(
                    f"Dataset contains {len(first_obj):,} columns, which exceeds the maximum supported limit "
                    f"of {MAX_INGESTION_COLUMNS} columns. Filter or transpose the file before ingestion."
                )
    except ValueError:
        raise
    except Exception:
        pass
    finally:
        buffer.seek(0)


def _read_jsonl_from_buffer(buffer, nrows=None):
    _inspect_jsonl_header_width(buffer)
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            buffer.seek(0)
            df = pd.read_json(buffer, lines=True, nrows=nrows, encoding=encoding)
            return _sanitize_unhashable_cells(df)
        except UnicodeDecodeError as error:
            last_error = error
        except (ValueError, TypeError):
            break
    if last_error:
        raise last_error
    buffer.seek(0)
    df = pd.read_json(buffer, lines=True, nrows=nrows)
    return _sanitize_unhashable_cells(df)


MAX_XML_DEPTH = 50


def _flatten_xml_element(element, parent_key="", index=None, depth=1, max_depth=MAX_XML_DEPTH):
    if depth > max_depth:
        raise ValueError(f"XML nesting exceeds maximum allowed depth of {max_depth}.")
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
        record.update(
            _flatten_xml_element(
                child,
                parent_key=f"{current_key}_",
                index=child_index,
                depth=depth + 1,
                max_depth=max_depth,
            )
        )
    return record


# stdlib expat resolves internal entities, so XML input is vulnerable to
# exponential "billion laughs" expansion and quadratic text blowup; external
# entities are never fetched (no network), but a DTD is never needed for
# tabular data -- reject it outright before parsing


def _reject_dtd(payload):
    # Fast multi-encoding text and raw-byte scan
    if b"<!DOCTYPE" in payload or b"<!ENTITY" in payload:
        raise ValueError(
            "XML datasets must not contain DTD/entity declarations "
            "(they enable resource-exhaustion attacks); export plain elements."
        )

    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be", "cp1252", "latin1"):
        try:
            decoded = payload.decode(enc)
            upper = decoded.upper()
            if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
                raise ValueError(
                    "XML datasets must not contain DTD/entity declarations "
                    "(they enable resource-exhaustion attacks); export plain elements."
                )
        except (UnicodeDecodeError, LookupError):
            pass

    # Expat parser-level guard fails on any DTD/entity regardless of encoding or padding
    try:
        p = xml.parsers.expat.ParserCreate()
        def _prohibit(*args, **kwargs):
            raise ValueError(
                "XML datasets must not contain DTD/entity declarations "
                "(they enable resource-exhaustion attacks); export plain elements."
            )
        p.StartDoctypeDeclHandler = _prohibit
        p.EntityDeclHandler = _prohibit
        p.Parse(payload, True)
    except ValueError:
        raise
    except Exception:
        pass


def _read_xml_from_buffer(buffer, nrows=None):
    buffer.seek(0)
    payload = buffer.read()
    _reject_dtd(payload)

    records = []
    root = None
    try:
        stream = io.BytesIO(payload)
        context = ElementTree.iterparse(stream, events=("start", "end"))
        depth = 0
        for event, elem in context:
            if event == "start":
                if root is None:
                    root = elem
                depth += 1
                if depth > MAX_XML_DEPTH:
                    raise ValueError(f"XML nesting exceeds maximum allowed depth of {MAX_XML_DEPTH}.")
            elif event == "end":
                depth -= 1
                if depth == 1 and elem != root:
                    flat = _flatten_xml_element(elem, depth=1)
                    prefix = f"{elem.tag}_"
                    flat = {
                        key.removeprefix(prefix): value
                        for key, value in flat.items()
                    }
                    records.append(flat)
                    root.remove(elem)
                    elem.clear()
                    if nrows is not None and len(records) >= nrows:
                        break
    except ElementTree.ParseError as error:
        raise ValueError(f"Failed to parse XML dataset: {error}") from error
    except (RecursionError, ValueError) as error:
        if "nesting" in str(error).lower():
            raise ValueError("XML nesting is too deep to process.") from error
        raise

    if not records and root is not None:
        try:
            records.append(_flatten_xml_element(root, depth=1))
        except (RecursionError, ValueError) as error:
            if "nesting" in str(error).lower():
                raise ValueError("XML nesting is too deep to process.") from error
            raise
    unique_keys = set()
    for item in records:
        unique_keys.update(item.keys())
        if len(unique_keys) > MAX_INGESTION_COLUMNS:
            raise ValueError(
                f"XML dataset contains {len(unique_keys):,}+ unique column tags, "
                f"which exceeds the maximum supported limit of {MAX_INGESTION_COLUMNS} columns."
            )
    df = pd.json_normalize(records)
    if nrows is not None and len(df) > nrows:
        df = df.head(nrows)
    return _sanitize_unhashable_cells(df)


class _HTMLTableParser(HTMLParser):
    # grabs the first <table> with at least a header row plus one data row
    def __init__(self, max_rows=None):
        super().__init__()
        self.tables = []
        self.table_row_counts = []
        self.has_rowspan = False
        self._table = None
        self._table_total_rows = 0
        self._table_stopped = False
        self._row = None
        self._cell = None
        self._current_colspan = 1
        self.max_rows = max_rows

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table = []
            self._table_total_rows = 0
            self._table_stopped = False
            self._row = None
            self._cell = None
        elif tag == "tr" and self._table is not None:
            self._table_total_rows += 1
            if self.max_rows is not None and len(self._table) >= (self.max_rows + 1):
                self._table_stopped = True
            if not self._table_stopped:
                self._row = []
        elif tag in {"td", "th"} and self._row is not None and not self._table_stopped:
            self._cell = []
            attrs_dict = dict(attrs)
            try:
                rspan = int(attrs_dict.get("rowspan", 1))
                if rspan > 1:
                    self.has_rowspan = True
            except (ValueError, TypeError):
                pass
            try:
                cspan = int(attrs_dict.get("colspan", 1))
                self._current_colspan = min(max(cspan, 1), 100)
            except (ValueError, TypeError):
                self._current_colspan = 1

    def handle_endtag(self, tag):
        if tag == "table" and self._table is not None:
            if len(self._table) >= 2:
                self.tables.append(self._table)
                self.table_row_counts.append(self._table_total_rows)
            self._table = None
            self._table_stopped = False
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag in {"td", "th"} and self._cell is not None:
            cell_text = " ".join("".join(self._cell).split())
            self._row.append(cell_text)
            if self._current_colspan > 1:
                for _ in range(self._current_colspan - 1):
                    self._row.append("")
            self._cell = None
            self._current_colspan = 1

    def handle_data(self, data):
        if not self._table_stopped and self._cell is not None:
            self._cell.append(data)


def _read_html_from_buffer(buffer, nrows=None):
    html_text = _decode_text_buffer(buffer)
    parser = _HTMLTableParser(max_rows=nrows)
    parser.feed(html_text)

    if parser.has_rowspan:
        raise AIConversionRequired(
            "HTML table contains complex rowspan spans; AI conversion required.",
            raw_text=html_text[:MAX_TEXT_EXTRACT_CHARS],
        )

    if not parser.tables:
        raise AIConversionRequired(
            "No HTML table found; AI conversion required.",
            raw_text=html_text[:MAX_TEXT_EXTRACT_CHARS],
        )

    best_idx = max(
        range(len(parser.tables)),
        key=lambda i: (
            parser.table_row_counts[i] if i < len(parser.table_row_counts) else len(parser.tables[i]),
            len(parser.tables[i][0]) if parser.tables[i] else 0,
        ),
    )
    best_table = parser.tables[best_idx]
    header = best_table[0]
    rows = best_table[1:]
    if nrows is not None and len(rows) > nrows:
        rows = rows[:nrows]
    width = max(len(row) for row in [header] + rows) if rows else len(header)
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


MAX_TEXT_EXTRACT_CHARS = 50_000


def _read_parquet_from_buffer(buffer, nrows=None):
    buffer.seek(0)
    effective_limit = nrows if (nrows is not None and nrows > 0) else MAX_INGESTION_ROWS
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(buffer)
        total_rows = pf.metadata.num_rows
        if nrows is None and total_rows > MAX_INGESTION_ROWS:
            raise ValueError(
                f"Parquet dataset contains {total_rows:,} rows, which exceeds the maximum supported limit "
                f"of {MAX_INGESTION_ROWS:,} rows. Filter or split the file before ingestion."
            )
        if total_rows == 0 or (nrows is not None and nrows == 0):
            tbl = pf.schema_arrow.empty_table()
            df = tbl.to_pandas()
            return _sanitize_unhashable_cells(df)
        target_rows = min(total_rows, effective_limit)
        batches = []
        count = 0
        batch_size = max(1, min(target_rows, 10000))
        for batch in pf.iter_batches(batch_size=batch_size):
            batches.append(batch)
            count += len(batch)
            if count >= target_rows:
                break
        if batches:
            tbl = pa.Table.from_batches(batches)
            df = tbl.to_pandas().head(target_rows)
            return _sanitize_unhashable_cells(df)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read Parquet dataset safely: {exc}") from exc


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

    limit_check = max_rows is None
    fetch_rows = (MAX_INGESTION_ROWS + 1) if limit_check else min(max_rows, MAX_INGESTION_ROWS)

    df = None
    if suffix == ".csv" or suffix == "":
        df = _read_csv_from_buffer(buffer, nrows=fetch_rows)

    elif suffix == ".tsv":
        df = _read_tsv_from_buffer(buffer, nrows=fetch_rows)

    elif suffix in {".xlsx", ".xls"}:
        if suffix == ".xlsx":
            _check_zip_bomb(buffer)
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        try:
            df = pd.read_excel(buffer, nrows=fetch_rows)
        except ImportError as exc:
            if suffix == ".xls" and "xlrd" in str(exc).lower():
                raise ValueError(
                    "Legacy .xls Excel format requires the 'xlrd' package (pip install xlrd>=2.0.1). "
                    "Alternatively, save or convert the file to modern .xlsx or .csv."
                ) from exc
            raise

    elif suffix == ".json":
        df = _read_json_from_buffer(buffer, nrows=fetch_rows)

    elif suffix in {".jsonl", ".ndjson"}:
        df = _read_jsonl_from_buffer(buffer, nrows=fetch_rows)

    elif suffix == ".parquet":
        df = _read_parquet_from_buffer(buffer, nrows=fetch_rows if not limit_check else None)

    elif suffix == ".xml":
        df = _read_xml_from_buffer(buffer, nrows=fetch_rows)

    elif suffix in {".html", ".htm"}:
        df = _read_html_from_buffer(buffer, nrows=fetch_rows)
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)

    elif suffix in TEXT_EXTENSIONS:
        buffer.seek(0)
        sample_bytes = buffer.read(65536)
        sample_text = ""
        for encoding in CSV_ENCODINGS:
            try:
                sample_text = sample_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not sample_text:
            sample_text = sample_bytes.decode("utf-8", errors="replace")

        delimited_candidate = _read_delimited_text(sample_text, nrows=fetch_rows)
        if delimited_candidate is not None:
            sample_lines = [line for line in sample_text.splitlines() if line.strip()][:20]
            try:
                dialect = csv.Sniffer().sniff("\n".join(sample_lines), delimiters=",;\t|")
                for encoding in CSV_ENCODINGS:
                    try:
                        buffer.seek(0)
                        df = pd.read_csv(buffer, sep=dialect.delimiter, encoding=encoding, nrows=fetch_rows)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, sep=dialect.delimiter, nrows=fetch_rows)
            except Exception:
                pass

        if df is None:
            buffer.seek(0)
            bounded_bytes = buffer.read(MAX_TEXT_EXTRACT_CHARS * 4)
            extracted = ""
            for encoding in CSV_ENCODINGS:
                try:
                    extracted = bounded_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not extracted:
                extracted = bounded_bytes.decode("utf-8", errors="replace")
            bounded_text = extracted[:MAX_TEXT_EXTRACT_CHARS]

            raise AIConversionRequired(
                "Plain text could not be parsed as a table; AI conversion required.",
                raw_text=bounded_text,
                filename=name,
            )

    elif suffix in DOCUMENT_EXTENSIONS:
        return _read_pdf_from_buffer(buffer, filename=name)

    else:
        raise AIConversionRequired(
            f"Unsupported dataset format: {suffix or 'none'}",
            raw_text="",
            filename=name,
        )

    if df is not None:
        if limit_check and len(df) > MAX_INGESTION_ROWS:
            raise ValueError(
                f"Dataset contains over {MAX_INGESTION_ROWS:,} rows, which exceeds the maximum supported limit "
                f"of {MAX_INGESTION_ROWS:,} rows. Filter or split the file before ingestion."
            )
        if not limit_check and max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)
        if df.shape[1] > MAX_INGESTION_COLUMNS:
            raise ValueError(
                f"Dataset contains {df.shape[1]:,} columns, which exceeds the maximum supported limit "
                f"of {MAX_INGESTION_COLUMNS} columns. Filter or transpose the file before ingestion."
            )
        return normalize_and_deduplicate_columns(df)
    return df


def read_dataset(dataset, max_rows=None):
    path = resolve_dataset_path(dataset)
    enforce_size_limit(path.stat().st_size, label=f"Dataset '{path.name}'")
    with path.open("rb") as handle:
        return read_tabular(handle, filename=path.name, max_rows=max_rows)

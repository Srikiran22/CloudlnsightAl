import csv
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
import zipfile
from collections.abc import Iterable
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Public API compatibility layer
# ---------------------------------------------------------------------------
# These modules were split out of this file to keep ingestion, frame hygiene and
# storage concerns separate. They are re-exported here because pages, tests and
# external tooling import them from Utils.paths.
from Utils.limits import (  # noqa: F401  (re-exported)
    CONVERSIONS_MANIFEST,
    CSV_ENCODINGS,
    DATASETS_DIR,
    DOCUMENT_EXTENSIONS,
    MAX_AGGREGATE_UPLOAD_BYTES,
    MAX_COMBINED_ROWS,
    MAX_INGESTION_CELLS,
    MAX_INGESTION_COLUMNS,
    MAX_INGESTION_ROWS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    PROJECT_ROOT,
    REPORTS_DIR,
    REPORT_TEMPLATES_DIR,
    MODELS_DIR,
    SUPPORTED_DATASET_EXTENSIONS,
    TABULAR_EXTENSIONS,
    TEXT_EXTENSIONS,
    AIConversionRequired,
    ResourcePolicy,
)
from Utils.frames import (  # noqa: F401  (re-exported)
    _decode_text_buffer,
    _sanitize_formula_val,
    _sanitize_unhashable_cells,
    _serialize_cell,
    normalize_and_deduplicate_columns,
    sanitize_for_csv_export,
)
from Utils.parsing import (  # noqa: F401  (re-exported)
    MAX_PDF_EXTRACT_CHARS,
    MAX_PDF_PAGES,
    MAX_TEXT_EXTRACT_CHARS,
    MAX_XML_DEPTH,
    _HTMLTableParser,
    _TranscodingStream,
    _check_string_cardinality_amplification,
    _detect_csv_delimiter,
    _detect_xml_encoding,
    _flatten_xml_element,
    _inspect_delimited_header_width,
    _inspect_jsonl_header_width,
    _read_chunked_delimited_buffer,
    _read_csv_from_buffer,
    _read_delimited_text,
    _read_html_from_buffer,
    _read_json_from_buffer,
    _read_jsonl_from_buffer,
    _read_parquet_from_buffer,
    _read_pdf_from_buffer,
    _read_tsv_from_buffer,
    _read_xml_from_buffer,
    _reject_dtd,
    _stream_json_array,
    _stream_json_from_buffer,
)

from Utils.logsys import get_logger  # noqa: E402  (kept below the re-exports)

logger = get_logger("Paths")


def ensure_project_directories() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def safe_stem(value: str, fallback: str = "model", max_len: int = 100) -> str:
    """Reduce a user-supplied name to a single safe path component.

    Keeps letters/digits/space/-/_ only (no separators, no traversal), so a
    crafted name can never escape its target directory. Guards against Windows
    reserved device names and bounds component length. Falls back when
    nothing survives the filter.
    """
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_ ").strip()
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_safe"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-_ ") or fallback
    return cleaned


def compute_upload_signature(files: Iterable[Any]) -> tuple[tuple[str, int, str], ...]:
    """Compute deterministic cryptographic signature tuples for uploaded file buffers."""
    sigs = []
    for f in files:
        if hasattr(f, "seek"):
            f.seek(0)
        hasher = hashlib.sha256()
        total_size = 0
        while True:
            chunk = f.read(64 * 1024)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            hasher.update(chunk)
            total_size += len(chunk)
        if hasattr(f, "seek"):
            f.seek(0)
        size = getattr(f, "size", total_size)
        content_hash = hasher.hexdigest()
        sigs.append((getattr(f, "name", "file"), size, content_hash))
    return tuple(sigs)


_FILENAME_LOCK = threading.Lock()
_RESERVED_FILENAMES = {}  # (dir_str, name): expire_time


def release_filename_reservation(filename: str, directory: Path | str = DATASETS_DIR) -> None:
    """Explicitly release an in-flight reservation."""
    with _FILENAME_LOCK:
        target_dir = Path(directory)
        dir_key = str(target_dir.resolve()) if target_dir.exists() else str(target_dir)
        safe_name = Path(filename).name
        _RESERVED_FILENAMES.pop((dir_key, safe_name), None)
        try:
            (target_dir / f".{safe_name}.reserve").unlink(missing_ok=True)
        except OSError:
            pass


def get_unique_filename(
    filename: str,
    directory: Path | str = DATASETS_DIR,
    extra_names: Iterable[str] | None = None,
) -> str:
    """Ensure a filename does not collide with existing files on disk, extra names, or concurrent allocations.

    Guarantees cross-thread AND cross-process safety via atomic filesystem reservations (O_CREAT | O_EXCL).
    Generates deterministic names:
      data.csv -> data_1.csv -> data_2.csv
    """
    with _FILENAME_LOCK:
        now = time.monotonic()
        epoch_now = time.time()
        # Clean expired in-memory reservations older than 60s
        expired = [k for k, exp in list(_RESERVED_FILENAMES.items()) if now > exp]
        for k in expired:
            _RESERVED_FILENAMES.pop(k, None)

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        dir_key = str(target_dir.resolve()) if target_dir.exists() else str(target_dir)
        extra = set(extra_names or [])
        name = Path(filename).name
        stem = Path(name).stem
        suffix = Path(name).suffix
        if stem.upper() in WINDOWS_RESERVED_NAMES:
            stem = f"{stem}_safe"
            name = f"{stem}{suffix}"

        candidate = name
        counter = 1
        while True:
            if counter > 10_000:
                raise OSError(
                    f"Failed to allocate a unique filename for '{name}' after {counter} attempts. "
                    "Check directory permissions and reservation limits."
                )
            # Check disk, extra names, in-memory registry
            if (
                (target_dir / candidate).exists()
                or candidate in extra
                or (dir_key, candidate) in _RESERVED_FILENAMES
            ):
                candidate = f"{stem}_{counter}{suffix}"
                counter += 1
                continue

            # Atomic cross-process reservation check via .reserve marker file
            res_file = target_dir / f".{candidate}.reserve"
            if res_file.exists():
                try:
                    if (epoch_now - res_file.stat().st_mtime) > 60.0:
                        res_file.unlink(missing_ok=True)
                    else:
                        candidate = f"{stem}_{counter}{suffix}"
                        counter += 1
                        continue
                except OSError:
                    candidate = f"{stem}_{counter}{suffix}"
                    counter += 1
                    continue

            try:
                fd = os.open(str(res_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(epoch_now).encode("ascii"))
                finally:
                    os.close(fd)
                _RESERVED_FILENAMES[(dir_key, candidate)] = now + 60.0
                return candidate
            except (FileExistsError, PermissionError):
                candidate = f"{stem}_{counter}{suffix}"
                counter += 1
                continue


def enforce_size_limit(size_in_bytes: int, label: str = "File") -> None:
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


def atomic_write(
    target_path: Path | str, content: Any, mode: str = "w", encoding: str = "utf-8"
) -> Path:
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
                with _FILENAME_LOCK:
                    dir_key = str(p.parent.resolve()) if p.parent.exists() else str(p.parent)
                    _RESERVED_FILENAMES.pop((dir_key, p.name), None)
                    try:
                        (p.parent / f".{p.name}.reserve").unlink(missing_ok=True)
                    except OSError:
                        pass
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


CONVERSION_MANIFEST_VERSION = "v2"
PARSER_POLICY_VERSION = "2.0"
PROMPT_POLICY_VERSION = "2.0"
RECONCILIATION_POLICY_VERSION = "2.0"


def compute_conversion_provenance(raw_text, model_name=None, extra_instructions=None):
    """Deterministic hash incorporating source content, model, instructions, and policy versions."""
    norm_model = (model_name or "").strip().lower()
    norm_instr = (extra_instructions or "").strip()
    hasher = hashlib.sha256()
    hasher.update(raw_text.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(norm_model.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(norm_instr.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(f"{CONVERSION_MANIFEST_VERSION}:{PARSER_POLICY_VERSION}:{PROMPT_POLICY_VERSION}:{RECONCILIATION_POLICY_VERSION}".encode("utf-8"))
    return hasher.hexdigest()[:32]


_MANIFEST_LOCK = threading.Lock()


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

    with _MANIFEST_LOCK:
        manifest = {}
        if manifest_file.exists():
            try:
                loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except Exception:
                manifest = {}

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
        try:
            for stale in manifest_file.parent.glob(f"{manifest_file.name}.tmp.*"):
                if stale.is_file() and (time.time() - stale.stat().st_mtime) > 60:
                    stale.unlink(missing_ok=True)
        except OSError:
            pass
        tmp_manifest = manifest_file.with_name(f"{manifest_file.name}.tmp.{uuid.uuid4().hex}")
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
    prov_hash = compute_conversion_provenance(raw_text, model_name=model_name, extra_instructions=extra_instructions)

    for conv_name, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("version") != CONVERSION_MANIFEST_VERSION:
            continue
        meta_source = meta.get("source_name", "")
        meta_ext = Path(meta_source).suffix.lower()
        req_ext = Path(source_name).suffix.lower()
        if meta_ext and req_ext and meta_ext != req_ext:
            continue

        is_name_match = (meta_source == norm_source or meta_source == source_name)
        is_prov_match = bool(meta.get("provenance_hash") and meta.get("provenance_hash") == prov_hash)
        if not (is_name_match or is_prov_match):
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


def cleanup_storage(
    directory=DATASETS_DIR,
    max_age_seconds=None,
    max_age_days=None,
    max_files=None,
    preserve_names=None,
    include_temp_only=False,
):
    """Safely prune old or temporary files from a managed directory (retention control)."""
    target_dir = Path(directory)
    if not target_dir.exists():
        return []

    if max_age_days is not None and max_age_seconds is None:
        max_age_seconds = max_age_days * 86400

    preserve = set(preserve_names or [])
    preserve.add(".conversions.json")
    preserve.add(".gitkeep")

    removed_files = []
    now = time.time()
    try:
        files = [p for p in target_dir.iterdir() if p.is_file() and p.name not in preserve]
    except OSError:
        return []

    def _is_temp_artifact(p: Path) -> bool:
        name = p.name
        if re.search(r"\.tmp\.[0-9a-fA-F]+$", name):
            return True
        if name.startswith(".") and name.endswith(".reserve"):
            return True
        return False

    def _safe_unlink(p_to_del: Path):
        try:
            p_to_del.unlink(missing_ok=True)
            removed_files.append(str(p_to_del))
            if p_to_del.name.endswith(".joblib"):
                sig = p_to_del.with_name(f"{p_to_del.name}.sig")
                if sig.is_file():
                    sig.unlink(missing_ok=True)
                    removed_files.append(str(sig))
        except OSError:
            pass

    dir_key = str(target_dir.resolve()) if target_dir.exists() else str(target_dir)
    with _FILENAME_LOCK:
        active_reservations = {cand for (d, cand) in _RESERVED_FILENAMES.keys() if d == dir_key}

    # Clean orphaned temporary files and orphan signatures
    for p in files:
        if _is_temp_artifact(p):
            # Do not delete active filename reservations
            if p.name.startswith(".") and p.name.endswith(".reserve"):
                candidate_name = p.name[1:-8]
                if candidate_name in active_reservations and (now - p.stat().st_mtime) <= 60.0:
                    continue
            _safe_unlink(p)
        elif p.name.endswith(".joblib.sig"):
            companion = p.with_name(p.name[:-4])
            if not companion.exists():
                _safe_unlink(p)

    if include_temp_only:
        return removed_files

    try:
        remaining_files = [p for p in target_dir.iterdir() if p.is_file() and p.name not in preserve and not _is_temp_artifact(p) and not p.name.endswith(".joblib.sig")]
    except OSError:
        return removed_files

    if max_age_seconds is not None:
        for p in remaining_files:
            try:
                if (now - p.stat().st_mtime) > max_age_seconds:
                    _safe_unlink(p)
            except OSError:
                pass

    if max_files is not None and max_files > 0:
        try:
            valid = [p for p in target_dir.iterdir() if p.is_file() and p.name not in preserve and not _is_temp_artifact(p) and not p.name.endswith(".joblib.sig")]
            if len(valid) > max_files:
                valid.sort(key=lambda p: p.stat().st_mtime)
                excess = len(valid) - max_files
                for p in valid[:excess]:
                    _safe_unlink(p)
        except OSError:
            pass

    # Prune orphaned conversion manifest entries
    manifest_file = target_dir / ".conversions.json"
    if manifest_file.is_file():
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(manifest_data, dict):
                current_names = {p.name for p in target_dir.iterdir() if p.is_file()}
                pruned_manifest = {k: v for k, v in manifest_data.items() if k in current_names}
                if len(pruned_manifest) != len(manifest_data):
                    atomic_write(manifest_file, json.dumps(pruned_manifest, indent=2), mode="w", encoding="utf-8")
        except Exception as exc:
            logger.debug("suppressed error in %s: %s", "_safe_unlink", exc)

    return removed_files


def delete_dataset(filename: str, directory: Path | str = DATASETS_DIR) -> bool:
    """Safely delete a dataset file and remove any manifest/provenance records."""
    target_dir = Path(directory).resolve()
    fn_path = Path(filename)
    if ".." in fn_path.parts or fn_path.is_absolute() or bool(fn_path.drive) or ":" in str(filename):
        raise ValueError(f"Security boundary rejection: Path traversal, drive, or Invalid character ':' detected in filename: {filename}")
    safe_name = fn_path.name
    if Path(safe_name).stem.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Security boundary rejection: Windows reserved device name in filename: {filename}")
    target_path = (target_dir / safe_name).resolve()
    if not target_path.is_relative_to(target_dir):
        raise ValueError(f"Filename {filename} escapes target directory {target_dir}.")
    if target_path.is_file():
        target_path.unlink()
        manifest_file = target_dir / ".conversions.json"
        if manifest_file.exists():
            with _MANIFEST_LOCK:
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and safe_name in data:
                        del data[safe_name]
                        tmp_manifest = manifest_file.with_name(f"{manifest_file.name}.tmp.{uuid.uuid4().hex}")
                        try:
                            tmp_manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
                            os.replace(tmp_manifest, manifest_file)
                        except Exception:
                            if tmp_manifest.exists():
                                try:
                                    tmp_manifest.unlink()
                                except OSError:
                                    pass
                            raise
                except Exception as exc:
                    logger.debug("suppressed error in %s: %s", "delete_dataset", exc)
        return True
    return False


_ROW_COUNT_CACHE = {}


def get_dataset_row_count(dataset: str) -> int:
    """Lightweight determination of exact source row count without full dataset materialization."""
    path = resolve_dataset_path(dataset)
    suffix = path.suffix.lower()

    try:
        st_stat = path.stat()
        cache_key = (str(path), st_stat.st_mtime_ns, st_stat.st_size)
        if cache_key in _ROW_COUNT_CACHE:
            return _ROW_COUNT_CACHE[cache_key]
    except OSError:
        cache_key = None

    row_count = None

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                if header is None:
                    row_count = 0
                else:
                    row_count = sum(1 for row in reader if row and any(cell.strip() for cell in row))
        except Exception as exc:
            logger.debug("suppressed error in %s: %s", "get_dataset_row_count", exc)

    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(path)
            row_count = meta.num_rows
        except Exception as exc:
            logger.debug("suppressed error in %s: %s", "get_dataset_row_count", exc)

    elif suffix in {".jsonl", ".ndjson"}:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                row_count = sum(1 for line in f if line.strip())
        except Exception as exc:
            logger.debug("suppressed error in %s: %s", "get_dataset_row_count", exc)

    elif suffix in {".xlsx", ".xls"}:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True)
            sheet = wb.worksheets[0] if wb.worksheets else wb.active
            rows = max(sheet.max_row - 1, 0) if sheet.max_row is not None else None
            wb.close()
            row_count = rows
        except Exception as exc:
            logger.debug("suppressed error in %s: %s", "get_dataset_row_count", exc)

    if cache_key is not None and row_count is not None:
        if len(_ROW_COUNT_CACHE) >= 256:
            _ROW_COUNT_CACHE.pop(next(iter(_ROW_COUNT_CACHE)), None)
        _ROW_COUNT_CACHE[cache_key] = row_count

    return row_count


def list_dataset_files(
    directory: Path | str | None = None, tabular_only: bool = False
) -> list[str]:
    if directory is None:
        ensure_project_directories()
        target_dir = DATASETS_DIR
    else:
        target_dir = Path(directory)
    datasets_root = target_dir.resolve()
    valid_files = []
    allowed_exts = TABULAR_EXTENSIONS if tabular_only else SUPPORTED_DATASET_EXTENSIONS
    if target_dir.exists():
        for path in target_dir.iterdir():
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in allowed_exts:
                try:
                    if path.resolve().parent == datasets_root:
                        valid_files.append(path.name)
                except OSError:
                    continue
    return sorted(valid_files)


def list_ready_datasets(directory: Path | str | None = None) -> list[str]:
    """Return only tabular datasets ready for immediate analytical querying and modeling."""
    return list_dataset_files(directory=directory, tabular_only=True)


def list_source_documents(directory: Path | str | None = None) -> list[str]:
    """Return unstructured or source documents that require AI extraction before analysis."""
    all_files = list_dataset_files(directory=directory, tabular_only=False)
    tabular = set(list_dataset_files(directory=directory, tabular_only=True))
    return [f for f in all_files if f not in tabular]


def resolve_dataset_path(dataset: str) -> Path:
    path = Path(dataset)
    if ".." in path.parts:
        raise ValueError(f"Security boundary rejection: Path traversal detected in dataset path: {dataset}; must stay in Datasets directory")
    if ":" in path.name:
        raise ValueError(f"Security boundary rejection: Alternate Data Stream / Invalid character ':' detected in dataset filename: {dataset}")
    if bool(path.drive) and not path.is_absolute():
        raise ValueError(f"Security boundary rejection: Drive-relative path detected in dataset path: {dataset}")
    stem = Path(path.name).stem.upper()
    if stem in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Security boundary rejection: Windows reserved device name in dataset filename: {dataset}")

    datasets_root = DATASETS_DIR.resolve()
    candidate = path.resolve() if path.is_absolute() else (datasets_root / path).resolve()

    if not candidate.is_relative_to(datasets_root) or candidate == datasets_root or candidate.parent != datasets_root:
        raise ValueError("Datasets must be read from the project's Datasets directory.")

    return candidate


def read_tabular(
    source: Any, filename: str | None = None, max_rows: int | None = None
) -> pd.DataFrame:
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
    elif hasattr(source, "fileno"):
        try:
            enforce_size_limit(os.fstat(source.fileno()).st_size, label=f"File '{name}'")
        except (OSError, io.UnsupportedOperation):
            pass
    elif hasattr(source, "seek") and hasattr(source, "tell"):
        try:
            cur_pos = source.tell()
            source.seek(0, io.SEEK_END)
            stream_len = source.tell()
            source.seek(cur_pos)
            enforce_size_limit(stream_len, label=f"File '{name}'")
        except (OSError, io.UnsupportedOperation, AttributeError):
            pass

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
        num_cols = 1
        for encoding in CSV_ENCODINGS:
            try:
                num_cols = _inspect_delimited_header_width(buffer, encoding, sep=",")
                break
            except UnicodeDecodeError:
                continue
        max_cell_rows = (MAX_INGESTION_CELLS // max(1, num_cols)) + 1
        effective_rows = min(fetch_rows, max_cell_rows) if limit_check else fetch_rows
        df = _read_csv_from_buffer(buffer, nrows=effective_rows)

    elif suffix == ".tsv":
        num_cols = 1
        for encoding in CSV_ENCODINGS:
            try:
                num_cols = _inspect_delimited_header_width(buffer, encoding, sep="\t")
                break
            except UnicodeDecodeError:
                continue
        max_cell_rows = (MAX_INGESTION_CELLS // max(1, num_cols)) + 1
        effective_rows = min(fetch_rows, max_cell_rows) if limit_check else fetch_rows
        df = _read_tsv_from_buffer(buffer, nrows=effective_rows)

    elif suffix in {".xlsx", ".xls"}:
        if suffix == ".xlsx":
            _check_zip_bomb(buffer, max_uncompressed_bytes=100 * 1024 * 1024)
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
            except Exception as exc:
                logger.debug("suppressed error in %s: %s", "read_tabular", exc)

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
        total_cells = len(df) * df.shape[1]
        if limit_check and total_cells > MAX_INGESTION_CELLS:
            raise ValueError(
                f"Dataset contains {total_cells:,} cells ({len(df):,} rows × {df.shape[1]:,} columns), "
                f"which exceeds the maximum supported limit of {MAX_INGESTION_CELLS:,} cells. "
                "Filter, sample, or split the dataset before ingestion."
            )
        df_deep_bytes = int(df.memory_usage(index=True, deep=True).sum())
        if limit_check and df_deep_bytes > ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES:
            raise ValueError(
                f"Dataset in-memory footprint ({df_deep_bytes / (1024 * 1024):.1f} MB) exceeds "
                f"the maximum allowed memory budget of {ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES // (1024 * 1024)} MB. "
                "Filter, sample, or downcast columns before ingestion."
            )
        return normalize_and_deduplicate_columns(df)
    return df


def read_dataset(dataset: str, max_rows: int | None = None) -> pd.DataFrame:
    path = resolve_dataset_path(dataset)
    enforce_size_limit(path.stat().st_size, label=f"Dataset '{path.name}'")
    with path.open("rb") as handle:
        return read_tabular(handle, filename=path.name, max_rows=max_rows)

from __future__ import annotations

import codecs
import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from Utils import limits
from Utils.frames import _decode_text_buffer, _sanitize_unhashable_cells
from Utils.limits import CSV_ENCODINGS, AIConversionRequired, ResourcePolicy
from Utils.logsys import get_logger
from Utils.markup_parsing import (
    _DTD_BYTE_PATTERNS,
    MAX_TEXT_EXTRACT_CHARS,
    MAX_XML_DEPTH,
    _detect_xml_encoding,
    _flatten_xml_element,
    _HTMLTableParser,
    _read_html_from_buffer,
    _read_xml_from_buffer,
    _reject_dtd,
    _TranscodingStream,
)

__all__ = [
    "MAX_TEXT_EXTRACT_CHARS",
    "MAX_XML_DEPTH",
    "_DTD_BYTE_PATTERNS",
    "_HTMLTableParser",
    "_TranscodingStream",
    "_detect_xml_encoding",
    "_flatten_xml_element",
    "_read_html_from_buffer",
    "_read_xml_from_buffer",
    "_reject_dtd",
    "cached_load_dataframe",
]

logger = get_logger("Parsing")


def _inspect_delimited_header_width(
    buffer: io.BytesIO | io.IOBase | Any, encoding: str, sep: str = ","
) -> int:
    """Quickly inspect header width before full parsing to reject wide files early and return column count."""
    buffer.seek(0)
    try:
        hdr = pd.read_csv(buffer, sep=sep, encoding=encoding, nrows=0)
        if isinstance(hdr, list) and len(hdr) > 0 and hasattr(hdr[0], "columns"):
            num_cols = len(hdr[0].columns)
        elif hasattr(hdr, "columns"):
            num_cols = len(hdr.columns)
        elif isinstance(hdr, (list, tuple)):
            num_cols = len(hdr)
        else:
            num_cols = 1
        if num_cols > limits.MAX_INGESTION_COLUMNS:
            raise ValueError(
                f"Dataset contains {num_cols:,} columns, which exceeds the maximum supported limit "
                f"of {limits.MAX_INGESTION_COLUMNS} columns. Filter or transpose the file before ingestion."
            )
        return num_cols
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        return 1
    finally:
        buffer.seek(0)


def _detect_csv_delimiter(sample_text: str) -> str:
    """Detect delimiter for CSV files when default comma parsing produces a single column.
    Checks candidate separators: ';', '\t', '|'.
    Returns detected delimiter if consistently observed across sample rows, else ','."""
    sample_lines = [line for line in sample_text.splitlines() if line.strip()][:25]
    if len(sample_lines) < 1:
        return ","

    candidates = [";", "\t", "|"]
    best_delim = ","
    best_cols = 1

    for delim in candidates:
        try:
            reader = csv.reader(io.StringIO("\n".join(sample_lines)), delimiter=delim)
            counts = [len(row) for row in reader if row]
            if not counts:
                continue
            first_count = counts[0]
            if first_count >= 2 and all(c == first_count for c in counts) and first_count > best_cols:
                best_cols = first_count
                best_delim = delim
        except Exception:
            continue

    return best_delim


def _check_string_cardinality_amplification(
    buffer: io.BytesIO | io.IOBase | Any, sep: str = ",", encoding: str = "utf-8", max_rows: int | None = None
) -> None:
    """Early estimation guard to reject massive high-cardinality string datasets before full materialization."""
    try:
        if not (hasattr(buffer, "seek") and hasattr(buffer, "tell")):
            return
        buffer.seek(0, io.SEEK_END)
        total_bytes = buffer.tell()
        buffer.seek(0)

        # Datasets under 10 MB cannot exceed 5M string cells
        if total_bytes <= 10 * 1024 * 1024:
            return

        sample_df = pd.read_csv(buffer, sep=sep, encoding=encoding, nrows=200)
        buffer.seek(0)
        if sample_df.empty or sample_df.shape[1] == 0 or len(sample_df) < 50:
            return

        string_cols = [
            c for c in sample_df.columns
            if pd.api.types.is_string_dtype(sample_df[c]) or sample_df[c].dtype == object
        ]
        if not string_cols:
            return

        high_card_cols = [
            c for c in string_cols
            if (sample_df[c].nunique() / max(1, len(sample_df))) >= 0.7
        ]
        if high_card_cols:
            sample_bytes = max(1, len(sample_df.to_csv(sep=sep, index=False)))
            est_bytes_per_row = max(1.0, sample_bytes / len(sample_df))
            est_total_rows = min(max_rows or limits.MAX_INGESTION_ROWS, int(total_bytes / est_bytes_per_row))
            est_high_card_cells = est_total_rows * len(high_card_cols)

            if est_high_card_cells > ResourcePolicy.MAX_HIGH_CARDINALITY_STRING_CELLS:
                raise ValueError(
                    f"Dataset contains an estimated {est_high_card_cells:,} memory-risk high-cardinality string cells across "
                    f"{len(high_card_cols)} columns, exceeding the memory-safety limit of "
                    f"{ResourcePolicy.MAX_HIGH_CARDINALITY_STRING_CELLS:,} cells. "
                    "High-cardinality text at this scale requires excessive RAM (>1.5 GB). "
                    "Filter, chunk, or downcast string columns before ingestion."
                )
    except (UnicodeDecodeError, pd.errors.ParserError):
        pass
    finally:
        buffer.seek(0)


def _read_chunked_delimited_buffer(
    buffer: io.BytesIO | io.IOBase | Any, sep: str = ",", encoding: str = "utf-8", nrows: int | None = None
) -> pd.DataFrame:
    """Read delimited buffer in bounded chunks, tracking accumulated deep memory and string cells.

    Tracks accumulated per-chunk high-cardinality string cell observations (not global distinct cardinality),
    which directly reflects Python process heap pressure since each chunk allocates distinct string objects.
    Enforces row, cell, deep memory, and high-cardinality limits streamingly during ingestion to abort before
    expensive full-file materialization. For multi-chunk inputs, intermediate chunk DataFrames are retained
    until concatenation into the final DataFrame.
    """
    if nrows is not None and nrows <= 10_000:
        buffer.seek(0)
        return pd.read_csv(buffer, sep=sep, encoding=encoding, nrows=nrows)

    chunks = []
    accumulated_rows = 0
    accumulated_deep_bytes = 0
    accumulated_high_cardinality_cells = 0
    string_cols = None
    chunk_size = 25_000

    buffer.seek(0)
    reader = pd.read_csv(buffer, sep=sep, encoding=encoding, nrows=nrows, chunksize=chunk_size)
    for chunk in reader:
        chunk_len = len(chunk)
        if chunk_len == 0:
            continue

        accumulated_rows += chunk_len
        if accumulated_rows > limits.MAX_INGESTION_ROWS:
            raise ValueError(
                f"Dataset contains over {limits.MAX_INGESTION_ROWS:,} rows, which exceeds the maximum supported limit "
                f"of {limits.MAX_INGESTION_ROWS:,} rows. Filter or split the file before ingestion."
            )

        num_cols = chunk.shape[1]
        accumulated_cells = accumulated_rows * num_cols
        if accumulated_cells > limits.MAX_INGESTION_CELLS:
            raise ValueError(
                f"Dataset contains over {limits.MAX_INGESTION_CELLS:,} cells ({accumulated_rows:,} rows × {num_cols:,} columns), "
                f"which exceeds the maximum supported limit of {limits.MAX_INGESTION_CELLS:,} cells. "
                "Filter, sample, or split the dataset before ingestion."
            )

        if string_cols is None:
            string_cols = [
                c for c in chunk.columns
                if pd.api.types.is_string_dtype(chunk[c]) or chunk[c].dtype == object
            ]

        chunk_deep_bytes = int(chunk.memory_usage(index=True, deep=True).sum())
        accumulated_deep_bytes += chunk_deep_bytes
        if accumulated_deep_bytes > ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES:
            raise ValueError(
                f"Dataset in-memory footprint exceeds the maximum allowed memory budget "
                f"of {ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES // (1024 * 1024)} MB. "
                "Filter, sample, or downcast columns before ingestion."
            )

        if string_cols and chunk_len >= 10:
            chunk_high_card_cells = sum(
                chunk[c].nunique() for c in string_cols
                if (chunk[c].nunique() / max(1, chunk_len)) >= 0.7
            )
            if chunk_high_card_cells > 0:
                accumulated_high_cardinality_cells += chunk_high_card_cells
                if accumulated_high_cardinality_cells > ResourcePolicy.MAX_HIGH_CARDINALITY_STRING_CELLS:
                    raise ValueError(
                        f"Dataset contains over {ResourcePolicy.MAX_HIGH_CARDINALITY_STRING_CELLS:,} "
                        "high-cardinality string cell observations across chunks, exceeding the memory-safety limit. "
                        "High-cardinality text at this scale requires excessive RAM (>1.5 GB). "
                        "Filter, chunk, or downcast string columns before ingestion."
                    )

        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()
    if len(chunks) == 1:
        return chunks[0]

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    return df


def _read_delimited_buffer(
    buffer: io.BytesIO | io.IOBase | Any,
    sep: str = ",",
    encoding: str | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read delimited buffer using chunked streaming with resource boundary enforcement."""
    encodings_to_try = [encoding] if encoding else CSV_ENCODINGS
    last_error = None
    for enc in encodings_to_try:
        try:
            _inspect_delimited_header_width(buffer, enc, sep=sep)
            _check_string_cardinality_amplification(buffer, sep=sep, encoding=enc, max_rows=nrows)
            buffer.seek(0)
            return _read_chunked_delimited_buffer(buffer, sep=sep, encoding=enc, nrows=nrows)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError(f"Unable to decode delimited data with delimiter '{sep}'.")


def _read_csv_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any,
    nrows: int | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    if sep is not None:
        return _read_delimited_buffer(buffer, sep=sep, nrows=nrows)

    last_error = None
    buffer.seek(0)
    sample_bytes = buffer.read(64 * 1024)
    buffer.seek(0)

    for encoding in CSV_ENCODINGS:
        try:
            sample_text = sample_bytes.decode(encoding, errors="replace")
            _inspect_delimited_header_width(buffer, encoding, sep=",")
            _check_string_cardinality_amplification(buffer, sep=",", encoding=encoding, max_rows=nrows)
            buffer.seek(0)
            df = _read_chunked_delimited_buffer(buffer, sep=",", encoding=encoding, nrows=nrows)
            if df.shape[1] == 1:
                detected_sep = _detect_csv_delimiter(sample_text)
                if detected_sep != ",":
                    buffer.seek(0)
                    _inspect_delimited_header_width(buffer, encoding, sep=detected_sep)
                    _check_string_cardinality_amplification(buffer, sep=detected_sep, encoding=encoding, max_rows=nrows)
                    buffer.seek(0)
                    df = _read_chunked_delimited_buffer(buffer, sep=detected_sep, encoding=encoding, nrows=nrows)
            return df
        except UnicodeDecodeError as error:
            last_error = error
    if last_error:
        raise last_error
    raise ValueError("Unable to decode CSV file.")


def _read_tsv_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int | None = None
) -> pd.DataFrame:
    return _read_delimited_buffer(buffer, sep="\t", nrows=nrows)




def _read_delimited_text(text: str, nrows: int | None = None) -> pd.DataFrame | None:
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


def _stream_json_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int
) -> list[dict[str, Any]] | dict[str, list[Any]] | None:
    """Incremental row-bounded stream decoder for JSON structures directly from a file-like buffer.

    Reads chunk by chunk (64KB) without materializing the full payload into memory as a string.
    Supports root arrays, keyed record arrays, and columnar tables. Stops reading the underlying
    stream immediately once nrows objects/values are parsed, guaranteeing that malformed data
    or gigabytes of unrequested rows after the boundary are never read, decoded, or materialized.
    """
    buffer.seek(0)
    pos = buffer.tell() if hasattr(buffer, "tell") else 0
    raw_head = buffer.read(4)
    if hasattr(buffer, "seek"):
        buffer.seek(pos)

    enc = "utf-8"
    if raw_head.startswith(codecs.BOM_UTF8):
        enc = "utf-8-sig"
    elif raw_head.startswith(codecs.BOM_UTF16_LE):
        enc = "utf-16-le"
    elif raw_head.startswith(codecs.BOM_UTF16_BE):
        enc = "utf-16-be"
    elif raw_head.startswith(codecs.BOM_UTF32_LE):
        enc = "utf-32-le"
    elif raw_head.startswith(codecs.BOM_UTF32_BE):
        enc = "utf-32-be"

    inc_dec = codecs.getincrementaldecoder(enc)(errors="replace")

    def _chunk_gen():
        while chunk := buffer.read(65536):
            text = inc_dec.decode(chunk, False)
            if text:
                yield text
        tail = inc_dec.decode(b"", True)
        if tail:
            yield tail

    it = _chunk_gen()
    buf = ""

    def _need_more():
        nonlocal buf
        try:
            buf += next(it)
            return True
        except StopIteration:
            return False

    while not buf.strip() and _need_more():
        pass
    buf = buf.lstrip()
    if not buf:
        return None

    decoder = json.JSONDecoder()

    # 1. Root-level array
    if buf[0] == "[":
        buf = buf[1:]
        records = []
        while len(records) < nrows:
            while True:
                buf = buf.lstrip(" \t\r\n,")
                if buf:
                    break
                if not _need_more():
                    break
            if not buf or buf[0] == "]":
                break
            while True:
                try:
                    obj, end_idx = decoder.raw_decode(buf)
                    records.append(obj)
                    buf = buf[end_idx:]
                    break
                except json.JSONDecodeError as err:
                    if not _need_more():
                        if buf.strip() and not buf.strip().startswith("]"):
                            raise ValueError(f"Malformed JSON dataset: {err}") from err
                        return records if records else None
                    if len(buf) > 10 * 1024 * 1024:
                        raise ValueError("Individual JSON row object exceeds maximum size limit (10MB).") from err
        return records

    # 2. Outer dict (keyed record array or columnar)
    if buf[0] == "{":
        while "[" not in buf and _need_more():
            if len(buf) > 5 * 1024 * 1024:
                break

        if "[" in buf:
            key_matches = list(re.finditer(r'"((?:\\.|[^"\\])*)"\s*:\s*\[', buf))
            record_array_matches = []
            columnar_matches = []
            for m in key_matches:
                col_name = m.group(1)
                start_bracket = buf.find("[", m.start())
                idx_after = start_bracket + 1
                while idx_after < len(buf) and buf[idx_after] in " \t\r\n":
                    idx_after += 1
                if idx_after >= len(buf) and _need_more():
                    while idx_after < len(buf) and buf[idx_after] in " \t\r\n":
                        idx_after += 1
                if idx_after < len(buf) and buf[idx_after] == "{":
                    record_array_matches.append((col_name, start_bracket + 1))
                else:
                    columnar_matches.append((col_name, start_bracket + 1))

            if len(record_array_matches) > 1:
                keys = sorted([k for k, _ in record_array_matches])
                raise ValueError(f"Ambiguous JSON structure: multiple top-level record lists found ({', '.join(keys)}). Extract the target list before ingestion.")

            if len(record_array_matches) == 1:
                idx = record_array_matches[0][1]
                records = []
                while len(records) < nrows:
                    while idx < len(buf) and buf[idx] in " \t\r\n,":
                        idx += 1
                    if idx >= len(buf) and _need_more():
                        while idx < len(buf) and buf[idx] in " \t\r\n,":
                            idx += 1
                    if idx >= len(buf) or buf[idx] == "]":
                        break
                    while True:
                        try:
                            obj, next_idx = decoder.raw_decode(buf, idx)
                            records.append(obj)
                            idx = next_idx
                            break
                        except json.JSONDecodeError as err:
                            if not _need_more():
                                if buf.strip() and not buf.strip().startswith("]"):
                                    raise ValueError(f"Malformed JSON dataset: {err}") from err
                                return records if records else None
                return records

            has_scalar_keys = bool(re.search(r'"(?:\\.|[^"\\])*"\s*:\s*(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"(?:\\.|[^"\\])*")\s*(?:,|})', buf))
            if columnar_matches and not record_array_matches and not has_scalar_keys:
                columns_data = {}
                for col_name, start_idx in columnar_matches:
                    idx = start_idx
                    col_vals = []
                    while len(col_vals) < nrows:
                        while idx < len(buf) and buf[idx] in " \t\r\n,":
                            idx += 1
                        if idx >= len(buf) and _need_more():
                            while idx < len(buf) and buf[idx] in " \t\r\n,":
                                idx += 1
                        if idx >= len(buf) or buf[idx] == "]":
                            break
                        try:
                            val, next_idx = decoder.raw_decode(buf, idx)
                            if isinstance(val, (dict, list)):
                                col_vals = None
                                break
                            col_vals.append(val)
                            idx = next_idx
                        except json.JSONDecodeError:
                            if not _need_more():
                                break
                            try:
                                val, next_idx = decoder.raw_decode(buf, idx)
                                if isinstance(val, (dict, list)):
                                    col_vals = None
                                    break
                                col_vals.append(val)
                                idx = next_idx
                            except json.JSONDecodeError:
                                col_vals = None
                                break
                    if col_vals is not None and len(col_vals) > 0:
                        columns_data[col_name] = col_vals
                if columns_data:
                    lens = {len(v) for v in columns_data.values()}
                    if len(lens) == 1:
                        return columns_data

    return None


def _stream_json_array(
    text: str, nrows: int
) -> list[dict[str, Any]] | dict[str, list[Any]] | None:
    """Incremental row-bounded decoder for JSON structures:
    - Root arrays: [ {...}, ... ]
    - Keyed record arrays in dicts: {"status": 200, "records": [ {...}, ... ]}
    - Columnar tables in dicts: {"colA": [1, 2, ...], "colB": [3, 4, ...]}
    Streams only nrows objects/values without parsing or materializing full graphs.
    Validates that the requested prefix is well-formed; does not validate unread trailing content.
    """
    s_text = text.lstrip()
    decoder = json.JSONDecoder()
    length = len(text)

    # 1. Root-level array of objects or scalars
    if s_text.startswith("["):
        idx = text.find("[") + 1
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

    # 2. Outer dict: could be wrapped record array or columnar table
    if s_text.startswith("{"):
        key_matches = list(re.finditer(r'"((?:\\.|[^"\\])*)"\s*:\s*\[', text))
        record_array_matches = []
        columnar_matches = []
        for m in key_matches:
            col_name = m.group(1)
            start_bracket = text.find("[", m.start())
            idx_after = start_bracket + 1
            while idx_after < length and text[idx_after] in " \t\r\n":
                idx_after += 1
            if idx_after < length and text[idx_after] == "{":
                record_array_matches.append((col_name, start_bracket + 1))
            else:
                columnar_matches.append((col_name, start_bracket + 1))

        if len(record_array_matches) > 1:
            keys = sorted([k for k, _ in record_array_matches])
            raise ValueError(
                f"Ambiguous JSON structure: multiple top-level record lists found ({', '.join(keys)}). "
                "Extract the target list before ingestion."
            )

        if len(record_array_matches) == 1:
            idx = record_array_matches[0][1]
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

        # 3. Columnar JSON: all keys map to arrays of scalars (no scalar keys)
        has_scalar_keys = bool(re.search(r'"(?:\\.|[^"\\])*"\s*:\s*(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"(?:\\.|[^"\\])*")\s*(?:,|})', text))
        if columnar_matches and not record_array_matches and not has_scalar_keys:
            columns_data = {}
            for col_name, start_idx in columnar_matches:
                idx = start_idx
                col_vals = []
                while idx < length and len(col_vals) < nrows:
                    while idx < length and text[idx] in " \t\r\n,":
                        idx += 1
                    if idx >= length or text[idx] == "]":
                        break
                    try:
                        val, next_idx = decoder.raw_decode(text, idx)
                        if isinstance(val, (dict, list)):
                            col_vals = None
                            break
                        col_vals.append(val)
                        idx = next_idx
                    except ValueError:
                        col_vals = None
                        break
                if col_vals is None or len(col_vals) == 0:
                    columns_data = None
                    break
                columns_data[col_name] = col_vals
            if columns_data and len(columns_data) > 0:
                lens = {len(v) for v in columns_data.values()}
                if len(lens) == 1:
                    return columns_data

    return None


def _read_json_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int | None = None
) -> pd.DataFrame:
    buffer.seek(0)
    try:
        if nrows is not None and nrows > 0:
            streamed = _stream_json_from_buffer(buffer, nrows)
            if streamed is not None:
                if isinstance(streamed, dict):
                    if len(streamed) > limits.MAX_INGESTION_COLUMNS:
                        raise ValueError(
                            f"JSON dataset has {len(streamed):,} columns, "
                            f"exceeding the maximum limit of {limits.MAX_INGESTION_COLUMNS} columns."
                        )
                    df = pd.DataFrame(streamed)
                    return _sanitize_unhashable_cells(df)
                if isinstance(streamed, list):
                    unique_keys = set()
                    for item in streamed:
                        if isinstance(item, dict):
                            unique_keys.update(item.keys())
                            if len(unique_keys) > limits.MAX_INGESTION_COLUMNS:
                                raise ValueError(
                                    f"JSON dataset has {len(unique_keys):,}+ unique column keys, "
                                    f"exceeding the maximum limit of {limits.MAX_INGESTION_COLUMNS} columns."
                                )
                    try:
                        norm_df = pd.json_normalize(streamed)
                        if len(streamed) > 0 and norm_df.shape[1] == 0:
                            norm_df = pd.DataFrame(streamed)
                    except TypeError:
                        norm_df = pd.DataFrame(streamed)
                    return _sanitize_unhashable_cells(norm_df)

        buf_size = 0
        if hasattr(buffer, "seek") and hasattr(buffer, "tell"):
            buffer.seek(0, 2)
            buf_size = buffer.tell()
            buffer.seek(0)

        if buf_size > 2 * 1024 * 1024:
            raise ValueError(
                f"JSON dataset exceeds bounded streaming capacity ({buf_size / (1024 * 1024):.1f} MB) "
                "and structure cannot be parsed incrementally. Maximum allowed for unstreamable JSON is 2 MB. "
                "Ensure the JSON root is an array of objects or columnar dictionary, or convert to JSON Lines / Parquet."
            )

        text = _decode_text_buffer(buffer)
        if nrows is not None and nrows > 0:
            streamed = _stream_json_array(text, nrows)
            if streamed is not None:
                if isinstance(streamed, dict):
                    if len(streamed) > limits.MAX_INGESTION_COLUMNS:
                        raise ValueError(
                            f"JSON dataset has {len(streamed):,} columns, "
                            f"exceeding the maximum limit of {limits.MAX_INGESTION_COLUMNS} columns."
                        )
                    df = pd.DataFrame(streamed)
                    return _sanitize_unhashable_cells(df)
                if isinstance(streamed, list):
                    unique_keys = set()
                    for item in streamed:
                        if isinstance(item, dict):
                            unique_keys.update(item.keys())
                            if len(unique_keys) > limits.MAX_INGESTION_COLUMNS:
                                raise ValueError(
                                    f"JSON dataset has {len(unique_keys):,}+ unique column keys, "
                                    f"exceeding the maximum limit of {limits.MAX_INGESTION_COLUMNS} columns."
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
                if len(data) > limits.MAX_INGESTION_COLUMNS:
                    raise ValueError(
                        f"JSON dataset has {len(data):,} columns, "
                        f"exceeding the maximum limit of {limits.MAX_INGESTION_COLUMNS} columns."
                    )
                if nrows is not None and nrows > 0:
                    data = {k: v[:nrows] for k, v in data.items()}
                df = pd.DataFrame(data)
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
            data = data[record_list_keys[0]] if len(record_list_keys) == 1 else [data]

        if isinstance(data, list):
            if nrows is not None and len(data) > nrows:
                data = data[:nrows]
            unique_keys = set()
            for item in data:
                if isinstance(item, dict):
                    unique_keys.update(item.keys())
                    if len(unique_keys) > limits.MAX_INGESTION_COLUMNS:
                        raise ValueError(
                            f"JSON dataset has {len(unique_keys):,}+ unique column keys, "
                            f"exceeding the maximum limit of {limits.MAX_INGESTION_COLUMNS} columns."
                        )
            try:
                norm_df = pd.json_normalize(data)
                if len(data) > 0 and norm_df.shape[1] == 0:
                    norm_df = pd.DataFrame(data)
            except TypeError:
                norm_df = pd.DataFrame(data)
            return _sanitize_unhashable_cells(norm_df)
    except ValueError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError(f"Failed to parse JSON dataset: {exc}") from exc


def _inspect_jsonl_header_width(buffer: io.BytesIO | io.IOBase | Any) -> None:
    try:
        buffer.seek(0)
        first_line = buffer.readline()
        if first_line:
            first_obj = json.loads(first_line.decode("utf-8", errors="replace"))
            if isinstance(first_obj, dict) and len(first_obj) > limits.MAX_INGESTION_COLUMNS:
                raise ValueError(
                    f"Dataset contains {len(first_obj):,} columns, which exceeds the maximum supported limit "
                    f"of {limits.MAX_INGESTION_COLUMNS} columns. Filter or transpose the file before ingestion."
                )
    except ValueError:
        raise
    except Exception as exc:
        logger.debug("suppressed error in %s: %s", "_inspect_jsonl_header_width", exc)
    finally:
        buffer.seek(0)


def _read_jsonl_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int | None = None
) -> pd.DataFrame:
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


def _read_parquet_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int | None = None
) -> pd.DataFrame:
    buffer.seek(0)
    effective_limit = nrows if (nrows is not None and nrows > 0) else limits.MAX_INGESTION_ROWS
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(buffer)
        total_rows = pf.metadata.num_rows
        total_cols = pf.metadata.num_columns
        if nrows is None and total_rows > limits.MAX_INGESTION_ROWS:
            raise ValueError(
                f"Parquet dataset contains {total_rows:,} rows, which exceeds the maximum supported limit "
                f"of {limits.MAX_INGESTION_ROWS:,} rows. Filter or split the file before ingestion."
            )
        if total_cols > limits.MAX_INGESTION_COLUMNS:
            raise ValueError(
                f"Parquet dataset contains {total_cols:,} columns, which exceeds the maximum supported limit "
                f"of {limits.MAX_INGESTION_COLUMNS} columns. Filter or transpose the file before ingestion."
            )
        if nrows is None and (total_rows * total_cols) > limits.MAX_INGESTION_CELLS:
            raise ValueError(
                f"Parquet dataset contains {total_rows * total_cols:,} cells ({total_rows:,} rows × {total_cols:,} columns), "
                f"which exceeds the maximum supported limit of {limits.MAX_INGESTION_CELLS:,} cells. "
                "Filter, sample, or split the file before ingestion."
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


def _read_pdf_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, filename: str = ""
) -> pd.DataFrame:
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


# Public Survey Aliases for Backwards Compatibility
_read_csv_buffer = _read_csv_from_buffer
_read_json_buffer = _read_json_from_buffer
_read_parquet_buffer = _read_parquet_from_buffer


@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def cached_load_dataframe(file_payload: bytes | str, file_name: str) -> pd.DataFrame:
    """Memory-bounded, TTL-cached DataFrame parser supporting multiple tabular extensions."""
    buf = io.BytesIO(file_payload) if isinstance(file_payload, (bytes, bytearray)) else io.BytesIO(file_payload.encode("utf-8"))
    ext = Path(file_name).suffix.lower()
    if ext == ".tsv":
        df = _read_tsv_from_buffer(buf)
    elif ext in (".json", ".jsonl", ".ndjson"):
        df = _read_json_from_buffer(buf)
    elif ext == ".parquet":
        df = _read_parquet_from_buffer(buf)
    elif ext in (".xml",):
        df = _read_xml_from_buffer(buf)
    elif ext in (".html", ".htm"):
        df = _read_html_from_buffer(buf)
    else:
        df = _read_csv_from_buffer(buf)

    if df is not None:
        mem_usage = df.memory_usage(deep=True).sum()
        if mem_usage > ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES:
            raise ValueError(
                f"Loaded DataFrame exceeds memory budget ({mem_usage} > {ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES} bytes)"
            )
    return df


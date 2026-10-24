"""Frame normalisation helpers shared by ingestion, export, and reporting.

Kept out of the format parsers so CSV/JSON/XML readers and the download layer
share exactly one implementation of column normalisation and spreadsheet-safety
sanitisation.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from Utils.limits import CSV_ENCODINGS

def _decode_text_buffer(buffer: Any) -> str:
    for encoding in CSV_ENCODINGS:
        try:
            buffer.seek(0)
            return buffer.read().decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    buffer.seek(0)
    return buffer.read().decode("utf-8", errors="replace")


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, set):
        return json.dumps(sorted(list(value)))
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def normalize_and_deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _sanitize_unhashable_cells(df: pd.DataFrame) -> pd.DataFrame:
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


def _sanitize_formula_val(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    if not val:
        return val
    lstripped = val.lstrip()
    if not lstripped:
        return val
    if val[0] in ("\t", "\r") or lstripped[0] in ("\t", "\r"):
        return f"'{val}"
    if lstripped[0] in ("=", "+", "-", "@"):
        try:
            float(lstripped)
            return val
        except ValueError:
            return f"'{val}"
    return val


def sanitize_for_csv_export(df: pd.DataFrame | None) -> pd.DataFrame | None:
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

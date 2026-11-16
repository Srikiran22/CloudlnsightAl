"""Markup and structured document parsing (XML, HTML) with security boundaries."""

from __future__ import annotations

import codecs
import io
import xml.parsers.expat
from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree

import pandas as pd

from Utils import limits
from Utils.frames import _decode_text_buffer, _sanitize_unhashable_cells
from Utils.limits import AIConversionRequired
from Utils.logsys import get_logger

logger = get_logger("markup_parsing")

MAX_XML_DEPTH = 50
MAX_TEXT_EXTRACT_CHARS = 50_000


def _flatten_xml_element(
    element: ElementTree.Element,
    parent_key: str = "",
    index: int | None = None,
    depth: int = 1,
    max_depth: int = MAX_XML_DEPTH,
) -> dict[str, Any]:
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


_DTD_BYTE_PATTERNS = [
    b"<!DOCTYPE",
    b"<!ENTITY",
    b"<\x00!\x00D\x00O\x00C\x00T\x00Y\x00P\x00E",
    b"\x00<\x00!\x00D\x00O\x00C\x00T\x00Y\x00P\x00E",
    b"<\x00!\x00E\x00N\x00T\x00I\x00T\x00Y",
    b"\x00<\x00!\x00E\x00N\x00T\x00I\x00T\x00Y",
    b"<\x00\x00\x00!\x00\x00\x00D\x00\x00\x00O\x00\x00\x00C\x00\x00\x00T\x00\x00\x00Y\x00\x00\x00P\x00\x00\x00E",
    b"\x00\x00\x00<\x00\x00\x00!\x00\x00\x00D\x00\x00\x00O\x00\x00\x00C\x00\x00\x00T\x00\x00\x00Y\x00\x00\x00P\x00\x00\x00E",
    b"<\x00\x00\x00!\x00\x00\x00E\x00\x00\x00N\x00\x00\x00T\x00\x00\x00I\x00\x00\x00T\x00\x00\x00Y",
    b"\x00\x00\x00<\x00\x00\x00!\x00\x00\x00E\x00\x00\x00N\x00\x00\x00T\x00\x00\x00I\x00\x00\x00T\x00\x00\x00Y",
]


def _detect_xml_encoding(stream: io.BytesIO | io.IOBase | Any) -> str:
    pos = stream.tell() if hasattr(stream, "tell") else 0
    head = stream.read(4)
    if hasattr(stream, "seek"):
        stream.seek(pos)
    if head.startswith(codecs.BOM_UTF32_LE) or head == b"<\x00\x00\x00":
        return "utf-32-le"
    if head.startswith(codecs.BOM_UTF32_BE) or head == b"\x00\x00\x00<":
        return "utf-32-be"
    if head.startswith(codecs.BOM_UTF16_LE) or head[:2] == b"<\x00":
        return "utf-16-le"
    if head.startswith(codecs.BOM_UTF16_BE) or head[:2] == b"\x00<":
        return "utf-16-be"
    return "utf-8"


class _TranscodingStream:
    """Wraps a multi-byte binary stream (UTF-16, UTF-32) and exposes a UTF-8 chunked read interface."""

    def __init__(self, stream, src_encoding):
        self.stream = stream
        self.inc_dec = codecs.getincrementaldecoder(src_encoding)(errors="replace")
        self.buf = b""

    def read(self, size=-1):
        if size is None or size < 0:
            size = 65536
        while len(self.buf) < size:
            chunk = self.stream.read(65536)
            if not chunk:
                tail = self.inc_dec.decode(b"", True)
                if tail:
                    self.buf += tail.encode("utf-8")
                break
            text = self.inc_dec.decode(chunk, False)
            if text:
                self.buf += text.encode("utf-8")
        out = self.buf[:size]
        self.buf = self.buf[size:]
        return out


def _reject_dtd(payload: bytes | bytearray | io.BytesIO | io.IOBase | Any) -> None:
    def _prohibit(*_args, **_kwargs):
        raise ValueError(
            "XML datasets must not contain DTD/entity declarations "
            "(they enable resource-exhaustion attacks); export plain elements."
        )

    stream = io.BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload
    if hasattr(stream, "read"):
        pos = stream.tell() if hasattr(stream, "tell") else 0
        enc = _detect_xml_encoding(stream)
        try:
            p = xml.parsers.expat.ParserCreate()
            p.StartDoctypeDeclHandler = _prohibit
            p.EntityDeclHandler = _prohibit
            root_started = False

            def _start_element(_name, _attrs):
                nonlocal root_started
                root_started = True

            p.StartElementHandler = _start_element

            overlap = b""
            while not root_started:
                chunk = stream.read(65536)
                if not chunk:
                    if not enc.startswith("utf-32"):
                        try:
                            p.Parse(b"", True)
                        except ValueError:
                            raise
                        except Exception as exc:
                            logger.debug("suppressed error in %s: %s", "_start_element", exc)
                    break
                raw_bytes = chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", errors="ignore")
                data_to_check = overlap + raw_bytes
                overlap = data_to_check[-64:]
                upper = data_to_check.upper()
                for pat in _DTD_BYTE_PATTERNS:
                    if pat in upper:
                        _prohibit()
                if not enc.startswith("utf-32"):
                    try:
                        p.Parse(raw_bytes, False)
                    except ValueError:
                        raise
                    except Exception as exc:
                        logger.debug("suppressed error in %s: %s", "_start_element", exc)
        finally:
            if hasattr(stream, "seek"):
                stream.seek(pos)
        return


def _read_xml_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int | None = None
) -> pd.DataFrame:
    buffer.seek(0)
    enc = _detect_xml_encoding(buffer)
    _reject_dtd(buffer)
    buffer.seek(0)

    xml_stream = _TranscodingStream(buffer, enc) if enc.startswith("utf-32") else buffer
    records = []
    root = None
    try:
        context = ElementTree.iterparse(xml_stream, events=("start", "end"))
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
        if len(unique_keys) > limits.MAX_INGESTION_COLUMNS:
            raise ValueError(
                f"XML dataset contains {len(unique_keys):,}+ unique column tags, "
                f"which exceeds the maximum supported limit of {limits.MAX_INGESTION_COLUMNS} columns."
            )
    df = pd.json_normalize(records)
    if nrows is not None and len(df) > nrows:
        df = df.head(nrows)
    return _sanitize_unhashable_cells(df)


class _HTMLTableParser(HTMLParser):
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


def _read_html_from_buffer(
    buffer: io.BytesIO | io.IOBase | Any, nrows: int | None = None
) -> pd.DataFrame:
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
    seen = {}
    unique_header = []
    for name in header:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_header.append(name if count == 0 else f"{name}_{count + 1}")
    normalized = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(normalized, columns=unique_header)

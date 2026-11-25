# ADR-005: Bounded Local Ingestion and Workstation Memory Ceilings

## Status
Accepted

## Date
2026-11-11

## Context
TabuLens runs in a single-process Python/Streamlit runtime on a workstation or single Docker container. Loading an unbounded CSV, wide JSON object, high-cardinality text dump, compressed `.xlsx` zip bomb, malicious XML DTD, or spreadsheet formula payload can exhaust host RAM, freeze the browser tab, or trigger spreadsheet DDE execution on export.

## Decision
Centralize deterministic resource bounds in `utils/limits.py` (`ResourcePolicy`) and enforce them across parsing, caching, visualization, and export:
1. **Upload and dimensional ceilings (`utils/limits.py`, `utils/parsing.py`, `utils/paths.py`):**
   - `MAX_UPLOAD_BYTES = 200 MB` per file; `MAX_AGGREGATE_UPLOAD_BYTES = 250 MB` across up to `MAX_UPLOAD_FILES = 20` batch files.
   - `MAX_INGESTION_ROWS = 1_000_000`, `MAX_INGESTION_COLUMNS = 200`, `MAX_INGESTION_CELLS = 20_000_000`.
   - `MAX_DATAFRAME_MEMORY_BYTES = 400 MB` deep RAM ceiling (`memory_usage(index=True, deep=True).sum()`), checked per chunk during delimited ingestion (`chunksize=25_000`) and post-parse.
   - `MAX_HIGH_CARDINALITY_STRING_CELLS = 5_000_000` early estimator and streaming guard against CPython string-object heap amplification.
2. **Format-specific parser hardening (`utils/parsing.py`, `utils/markup_parsing.py`, `utils/paths.py`):**
   - Check `.xlsx` archives via `_check_zip_bomb()` (100 MB uncompressed ceiling and >100:1 expansion ratio check above 50 MB).
   - Reject XML `<!DOCTYPE` and `<!ENTITY` declarations via `_reject_dtd()` (expat SAX + multi-byte UTF-16/UTF-32 byte pattern scan) and bound nesting depth (`MAX_XML_DEPTH = 50`).
   - Stream JSON/Parquet/delimited buffers bounded by `nrows` instead of materializing unconstrained objects.
3. **Bounded memory cache and chart downsampling (`utils/dataset_ui.py`, `utils/sampling.py`, `utils/charts.py`):**
   - `BoundedDatasetMemoryCache` caps cached DataFrames at `256 MB` (`MAX_DATASET_CACHE_BYTES`) with LRU eviction and deep-copy-on-read isolation.
   - `sample_for_visualization` caps plots at `25_000` rows (`downsample_timeseries` preserves bucket min/max/endpoints); `sample_for_analysis` caps statistical tests at `50_000` rows.
4. **Export sanitization (`utils/frames.py`):**
   - `sanitize_for_csv_export()` neutralizes spreadsheet formula triggers (`=`, `+`, `-`, `@`, `\t`, `\r`) before writing CSV exports.

## Alternatives Considered

### Relying Solely on Streamlit's Upload Limit
- Pros: No custom parser wrappers required.
- Cons: A 50 MB compressed `.xlsx` or sparse JSON file can still expand into gigabytes of Python objects in RAM, and standard `pd.read_xml` / `pd.read_csv` do not guard against XXE or formula injection.
- Rejected: Fails under adversarial or malformed workstation inputs.

### Distributed Backends (Dask / Spark / Polars Cluster)
- Pros: Scales beyond workstation RAM.
- Cons: Heavy operational footprint and dependency complexity incompatible with a single-container local desktop workbench.
- Rejected: Explicitly out of scope for TabuLens.

## Consequences
- Memory growth during ingestion, caching, and browser rendering stays bounded and predictable on local workstations and containers.
- Datasets exceeding 1,000,000 rows, 200 columns, 20,000,000 cells, or 400 MB deep DataFrame RAM are intentionally rejected with actionable `ValueError` messages rather than swapped to disk.
- Raw values are kept unquoted in memory during analytics (`utils/frames.py` sanitizes on CSV/TSV export) so numerical analysis is not altered by spreadsheet escaping.

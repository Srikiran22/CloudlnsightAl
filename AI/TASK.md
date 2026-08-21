# Current Task

## User Goal

1. Make the project accept ANY file as input and use AI to convert it into a proper tabular format so all existing features (cleaning, EDA, ML, viz, reports) work on it.
2. Make the generated PDF report significantly more detailed.

## Requirements

- Universal ingestion: native parsers for common formats (JSON, JSONL, TSV, Parquet, XML, HTML, TXT) plus Gemini-powered conversion of unstructured content into a DataFrame
- Converted datasets must flow into the existing session state so Cleaning/EDA/Viz/ML/AI/Report pages work unchanged
- PDF report: full column coverage, extended statistics (quartiles, skewness, kurtosis), categorical analysis, correlation insights, data quality scoring, sample rows, optional charts, page footers
- Graceful degradation when optional dependencies are missing

## Constraints

- Reuse existing patterns (`read_tabular`, `_generate_content`, session state keys `current_df` / `dataset_name`)
- No breaking changes to existing CSV/Excel/S3 flows or tests
- Keep AI prompts bounded (truncate raw content samples)

## Non-Goals

- Vision-based image-to-table extraction
- Rewriting downstream pages

## Acceptance Criteria

- Uploading JSON/TSV/Parquet/XML/HTML/TXT produces a working DataFrame without AI
- Unstructured text files can be converted to a DataFrame via Gemini on demand
- PDF contains new detailed sections and renders for wide datasets
- All existing tests still pass; new unit tests cover conversion helpers

## Priority

1. Universal ingestion pipeline
2. Detailed PDF report
3. Tests and verification

## Current Status

In Progress

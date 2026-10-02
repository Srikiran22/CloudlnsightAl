# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | :white_check_mark: |
| < 2.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in CloudInsight AI:
1. Please do not open a public issue.
2. Report the vulnerability privately to the maintainers or via repository security advisories.
3. Include detailed steps to reproduce, sample payloads if applicable, and the affected version.
4. You will receive an initial response within 48 hours.

## Security Architecture & Defenses

CloudInsight AI implements defense-in-depth principles:

### 1. Model Deserialization Boundary
`joblib` bundles unpickle arbitrary Python bytecode. By default:
- All loaded models must reside strictly within `MODELS_DIR` (path traversal is blocked).
- Every model requires a valid HMAC-SHA256 cryptographic signature verified before deserialization begins.
- Tampered, unsigned, or escaping model files are rejected before `joblib.load()` executes.
- An explicit `trusted=True` audit bypass exists only for internal test fixtures and emits loud audit log warnings.

### 2. Ingestion Resource Governance
To prevent memory exhaustion and denial-of-service:
- Hard file size limit: 200 MB (`MAX_INGESTION_BYTES`).
- Global cell budget: 20,000,000 cells (`MAX_INGESTION_CELLS` = rows × columns).
- Hard dimensional ceilings: 1,000,000 rows and 200 columns.
- JSON bounded parsing fails closed when streaming cannot satisfy row limits without unbounded loading.
- S3 datasets stream directly to disk with active byte accounting without in-memory duplication.

### 3. Formula Injection & Spreadsheet Safety
- Canonical datasets in internal storage (`Datasets/`) preserve exact values for lossless data science, ML, and numerical calculations (preserving values such as `=SUM(...)`, `-123`, `+123`, `@user`).
- External spreadsheet downloads (`CSV` / `Excel`) are filtered through `sanitize_for_csv_export()`, prepending a single quote (`'`) to any formula trigger characters (`=`, `+`, `-`, `@`, `\t`, `\r`) to protect client spreadsheet software.

### 4. Filesystem & Path Containment
- All user-supplied and S3-derived filenames are sanitized through `safe_stem()`.
- Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`) are replaced with safe identifiers.
- Component length is bounded to 100 characters.
- Path traversal sequences (`..`, absolute paths, leading slashes) are rejected or normalized.

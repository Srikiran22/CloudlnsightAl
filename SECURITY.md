# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.2.x   | :white_check_mark: |
| < 2.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in CloudInsight AI:
1. Please do not open a public issue.
2. Report the vulnerability privately to the maintainers or via repository security advisories.
3. Include detailed steps to reproduce, sample payloads if applicable, and the affected version.
4. You will receive an initial response within 48 hours.

## Security Architecture, Governance & Compliance Alignment

CloudInsight AI adheres to defense-in-depth engineering principles mapped against **OWASP ASVS v5.0.0** and **NIST SP 800-218 (SSDF v1.1)**:

| Security Vector | Implementation Point | OWASP ASVS v5.0.0 | NIST SP 800-218 | Defense Guarantee |
|---|---|---|---|---|
| **Path Traversal & Sandboxing** | `Utils/paths.py::resolve_dataset_path` | **ASVS 12.1.1** | **PW.1.1** | Enforces `Path.is_relative_to(DATASETS_DIR)`, rejects `..`, absolute drive letters, and Windows ADS (`:`). |
| **Pickle Deserialization RCE** | `Utils/ML.py::load_trained_model` | **ASVS 5.5.1, 6.2.1** | **PW.1.1** | Pre-deserialization HMAC-SHA256 signature validation against `Models/.model_signing_key` (`0600`). |
| **XXE / XML Entity Expansion** | `Utils/paths.py::_reject_dtd` | **ASVS 5.2.2** | **PW.4.1** | Scans raw XML byte streams for `<!DOCTYPE` and `<!ENTITY>`, failing closed prior to parser instantiation. |
| **ZIP Decompression Bombs** | `Utils/paths.py::_check_zip_bomb` | **ASVS 12.2.1** | **PW.1.1** | Inspects archive directory headers; aborts archives $>100\text{ MB}$ uncompressed or ratio $>100:1$. |
| **CSV Formula Injection (DDE)** | `Utils/frames.py::sanitize_for_csv_export` | **ASVS 5.2.4** | **PW.4.1** | Prepends `'` to cells starting with `=`, `+`, `-`, `@`, `\t`, `\r` on external exports. |
| **Ephemeral Secret Isolation** | `App.py`, `Pages/Upload.py` | **ASVS 6.4.1, 8.3.1** | **PS.1.1** | API keys reside in `st.session_state` RAM only; purged by `temporary_secret` context managers. |
| **PII & Identifier Shield** | `Utils/privacy.py` | **ASVS 8.3.1** | **PS.1.1** | Heuristic screening of emails, phones, credit cards (Luhn), IBANs, and IDs prior to AI dispatch. |
| **Resource Ingestion Ceilings** | `Utils/limits.py`, `Utils/parsing.py` | **ASVS 5.1.1** | **PW.1.1** | Enforces 200 MB files, 1M rows, 200 cols, 20M cells, and 400 MB deep DataFrame memory. |

### 1. Model Deserialization Boundary
`joblib` bundles unpickle arbitrary Python bytecode. By default:
- All loaded models must reside strictly within `MODELS_DIR` (path traversal is blocked).
- Every model requires a valid HMAC-SHA256 cryptographic signature verified before deserialization begins.
- Tampered, unsigned, or escaping model files are rejected before `joblib.load()` executes.
- An explicit `trusted=True` audit bypass exists only for internal test fixtures and emits loud audit log warnings.

*Scope note:* The signing key is generated on first use and stored at `Models/.model_signing_key` (mode `0600`, created atomically via `os.O_CREAT | os.O_EXCL`). This provides tamper evidence — corrupted, truncated, or partially rewritten artifacts are rejected. In single-user workstations, this secures against drive-by file contamination. For multi-tenant or cluster environments, the signing key must be injected via a hardware security module (HSM) or dedicated secret vault.

### 2. Ingestion Resource Governance
To prevent memory exhaustion and denial-of-service:
- Hard file size limit: 200 MB (`MAX_UPLOAD_BYTES`).
- Batch upload ceiling: 250 MB (`MAX_AGGREGATE_UPLOAD_BYTES`, max 20 files).
- Global cell budget: 20,000,000 cells (`MAX_INGESTION_CELLS` = rows × columns).
- Dimensional ceilings: 1,000,000 rows and 200 columns.
- Deep DataFrame memory budget: 400 MB (`MAX_DATAFRAME_MEMORY_BYTES`).
- High-cardinality string guard: 5,000,000 cells (`MAX_HIGH_CARDINALITY_STRING_CELLS`).

### 3. Formula Injection & Spreadsheet Safety
- Canonical datasets in internal storage (`Datasets/`) preserve exact values for lossless data science, ML, and numerical calculations (preserving values such as `=SUM(...)`, `-123`, `+123`, `@user`).
- External spreadsheet downloads (`CSV` / `Excel`) are filtered through `sanitize_for_csv_export()`, prepending a single quote (`'`) to any formula trigger characters (`=`, `+`, `-`, `@`, `\t`, `\r`) to protect client spreadsheet software.

### 4. Filesystem & Path Containment
- All user-supplied filenames are sanitized through `safe_stem()`.
- Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`) are replaced with safe identifiers.
- Component length is bounded to 100 characters.
- Path traversal sequences (`..`, absolute paths, leading slashes, Windows ADS colons) are strictly rejected.

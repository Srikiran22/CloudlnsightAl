# ADR-004: HMAC-SHA256 Signing Boundary for Joblib Model Artifacts

## Status
Accepted

## Date
2026-11-11

## Context
Scikit-learn pipelines serialized via `joblib` use Python's `pickle` format internally, which can execute arbitrary Python bytecode during deserialization. Even in a local-first workbench, users may place externally downloaded `.joblib` files into `Models/` or modify saved files between runs, exposing the local host to arbitrary code execution upon loading.

## Decision
Enforce a cryptographic integrity and directory-containment boundary around all model persistence (`utils/model_security.py`, `utils/ml.py`):
1. **Directory containment & filename sanitization:** Sanitize stems via `safe_stem()` and verify resolved paths stay inside `MODELS_DIR` (`Models/`) with `Path.is_relative_to`, rejecting `..`, drive-relative paths, Windows Alternate Data Streams (`:`), and Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`).
2. **Pre-load size cap:** Read bytes up to `MAX_MODEL_FILE_SIZE = 100 MB` before deserializing.
3. **Atomic local signing key:** Create or read a 32-byte key at `Models/.model_signing_key` using `os.open(..., os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)` so concurrent workers converge on a single key without overwrite races.
4. **In-memory HMAC-SHA256 verification:** Write a companion `.joblib.sig` JSON sidecar during `save_trained_model()`, and verify the signature against the in-memory byte buffer before passing `io.BytesIO(content)` to `joblib.load()` (preventing TOCTOU file swaps between verification and load).
5. **Standalone script parity:** Embed the same HMAC verification routine in scripts produced by `generate_standalone_inference_code()`, failing closed unless `--allow-unverified-model` is explicitly passed.

## Alternatives Considered

### Plain `joblib.dump` / `joblib.load`
- Pros: Zero sidecar files or key management.
- Cons: Silently executes arbitrary pickle opcodes from modified or untrusted `.joblib` artifacts.
- Rejected: Unsafe default for model artifacts on disk.

### `skops` or ONNX-Only Serialization
- Pros: Avoids pickle deserialization entirely for supported estimators.
- Cons: Cannot serialize arbitrary scikit-learn `Pipeline` + `ColumnTransformer` + custom bundle metadata dicts uniformly without schema translation gaps or extra runtime dependencies.
- Rejected: HMAC-signed `joblib` preserves full scikit-learn pipeline parity across training, UI scoring, and standalone script export.

## Consequences
- Unsigned, tampered, relocated, or renamed `.joblib` files are rejected with `ValueError` before `joblib.load()` runs (unless an explicit `trusted=True` test override is passed).
- **Trust assumption:** HMAC-SHA256 verifies local integrity and authenticity under the assumption that `Models/.model_signing_key` remains readable only by the local user (`0o600`). It is not a substitute for an external KMS, asymmetric code signing, or key rotation when distributing models across untrusted machines.

"""Model security, cryptographic artifact signing, and HMAC verification."""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

from Utils.logsys import get_logger
from Utils.paths import MODELS_DIR

logger = get_logger("model_security")

MODEL_SIGNING_KEY_FILENAME = ".model_signing_key"
MODEL_SIGNATURE_SUFFIX = ".joblib.sig"

_MODEL_KEY_PATH = MODELS_DIR / MODEL_SIGNING_KEY_FILENAME
_MODEL_KEY_LOCK = threading.Lock()
_CACHED_MODEL_KEY: bytes | None = None


def _get_or_create_model_signing_key(directory: Path | str | None = None) -> bytes:
    """Thread-safe and cross-process atomic acquisition of the HMAC model signing key.

    Uses O_CREAT | O_EXCL kernel semantics so concurrent initialization attempts
    converge deterministically on the first created key without overwrite races.
    """
    global _CACHED_MODEL_KEY
    key_dir = Path(directory) if directory is not None else MODELS_DIR
    key_path = key_dir / MODEL_SIGNING_KEY_FILENAME

    if directory is None and _CACHED_MODEL_KEY is not None and key_path.is_file():
        return _CACHED_MODEL_KEY

    key_dir.mkdir(parents=True, exist_ok=True)
    if key_path.is_file() and key_path.stat().st_size == 32:
        try:
            key = key_path.read_bytes()
            if len(key) == 32:
                if directory is None:
                    with _MODEL_KEY_LOCK:
                        _CACHED_MODEL_KEY = key
                return key
        except OSError:
            pass

    new_key = secrets.token_bytes(32)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        fd = os.open(str(key_path), flags, 0o600)
        try:
            os.write(fd, new_key)
        finally:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.chmod(key_path, 0o600)
        if directory is None:
            with _MODEL_KEY_LOCK:
                _CACHED_MODEL_KEY = new_key
        return new_key
    except (FileExistsError, OSError) as err:
        # A concurrent worker created the key first; wait for bytes to flush
        for attempt in range(25):
            if key_path.is_file() and key_path.stat().st_size == 32:
                try:
                    key = key_path.read_bytes()
                    if len(key) == 32:
                        if directory is None:
                            with _MODEL_KEY_LOCK:
                                _CACHED_MODEL_KEY = key
                        return key
                except OSError:
                    pass
            time.sleep(0.005 * (1.5 ** min(attempt, 8)))
        if key_path.is_file():
            return key_path.read_bytes()
        raise RuntimeError(f"Failed to initialize or read model signing key at {key_path}") from err


def sign_model_artifact(model_path: Path | str) -> Path:
    """Compute and store a cryptographic HMAC-SHA256 signature for a saved model file."""
    p = Path(model_path).resolve()
    if not p.is_file():
        raise ValueError(f"Model file not found to sign: {model_path}")
    key_dir = p.parent if (p.parent / MODEL_SIGNING_KEY_FILENAME).is_file() else MODELS_DIR
    key = _get_or_create_model_signing_key(directory=key_dir)
    content = p.read_bytes()
    sig = hmac.new(key, content, hashlib.sha256).hexdigest()
    sig_path = p.with_suffix(MODEL_SIGNATURE_SUFFIX)
    sig_data = {
        "model_file": p.name,
        "algo": "hmac-sha256",
        "signature": sig,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    sig_path.write_text(json.dumps(sig_data, indent=2), encoding="utf-8")
    return sig_path


def verify_model_artifact_signature(model_path: Path | str, content: bytes | None = None) -> bool:
    """Verify cryptographic authenticity and integrity of a model file before deserialization."""
    p = Path(model_path).resolve()
    sig_path = p.with_suffix(MODEL_SIGNATURE_SUFFIX)
    if not sig_path.is_file():
        return False
    try:
        sig_data = json.loads(sig_path.read_text(encoding="utf-8"))
        if not isinstance(sig_data, dict):
            return False
        if sig_data.get("algo") != "hmac-sha256":
            return False
        if sig_data.get("model_file") != p.name:
            return False
        expected_sig = sig_data.get("signature", "")
        if not isinstance(expected_sig, str) or not expected_sig:
            return False
        key_dir = p.parent if (p.parent / MODEL_SIGNING_KEY_FILENAME).is_file() else MODELS_DIR
        key = _get_or_create_model_signing_key(directory=key_dir)
        if content is None:
            content = p.read_bytes()
        actual_sig = hmac.new(key, content, hashlib.sha256).hexdigest()
        return hmac.compare_digest(actual_sig, expected_sig)
    except Exception as exc:
        logger.warning("Signature verification error for %s: %s", p.name, exc)
        return False

# Pre-Gemini privacy screening: flag likely-sensitive columns so the user
# can exclude them from AI context with one click. Detection is heuristic by
# design -- it should produce useful warnings with few false positives, not
# replace human judgment.

import re

import pandas as pd


SAMPLE_ROWS = 100

_EXACT_OR_TOKEN_HINTS = {"tel", "cell", "pwd", "ssn", "cvv"}
_SUBSTRING_HINTS = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "social_security", "credit_card", "card_number", "iban",
    "phone", "mobile", "email",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PHONE_RE = re.compile(r"\+?\d[\d\t .()-]{7,}\d")
_TOKEN_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bey[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}"),  # JWT shape
)


def _luhn_ok(digits):
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _scan_numeric_phone_values(series):
    """Detect numeric columns storing 10-15 digit phone numbers."""
    s = series.dropna().head(SAMPLE_ROWS)
    if s.empty:
        return None
    try:
        numeric_vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(numeric_vals) == 0:
            return None
        if (numeric_vals.astype("int64") == numeric_vals).all():
            int_vals = numeric_vals.astype("int64")
            # Filter out 10-digit Unix epoch timestamps (seconds between 2001 and 2038)
            if ((int_vals >= 1_000_000_000) & (int_vals <= 2_150_000_000)).all():
                return None
            str_vals = int_vals.abs().astype(str)
            if ((str_vals.str.len() >= 10) & (str_vals.str.len() <= 15)).mean() >= 0.8:
                return "phone-number-like numeric values"
    except Exception:
        pass
    return None


def _scan_values(series):
    """Return a reason string when sampled values look sensitive."""
    text = series.dropna().astype(str).head(SAMPLE_ROWS)
    joined = "\n".join(text.tolist())
    if not joined.strip():
        return None
    if _EMAIL_RE.search(joined):
        return "email-like values"
    for pattern in _TOKEN_RES:
        if pattern.search(joined):
            return "credential/token-like values"
    if _IBAN_RE.search(joined):
        return "IBAN-like values"
    for match in _CARD_RE.finditer(joined):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) in range(13, 20) and _luhn_ok(digits):
            return "credit-card-like values"
    for m in _PHONE_RE.finditer(joined):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= 9:
            if len(digits) == 10 and digits.isdigit():
                val = int(digits)
                if 1_000_000_000 <= val <= 2_150_000_000:
                    continue
            return "phone-number-like values"
    return None


def detect_sensitive_columns(df):
    """Map {column: reason} for columns that look like they hold sensitive data.

    A column is flagged when its NAME suggests secrets/identifiers, or when a
    sample of its string values matches email / token / IBAN / credit-card
    (Luhn-checked) / phone patterns. Numeric-only measurement columns are
    never flagged from values alone.
    """
    flags = {}
    for column in df.columns:
        col_str = str(column)
        lowered = col_str.lower()
        tokens = set(re.split(r"[^a-z0-9]+", lowered))
        camel_tokens = {m.lower() for m in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", col_str)}
        all_tokens = (tokens | camel_tokens) - {""}
        hint = next((h for h in _EXACT_OR_TOKEN_HINTS if h in all_tokens), None)
        if not hint:
            hint = next((h for h in _SUBSTRING_HINTS if h in lowered), None)
        if hint:
            flags[column] = f"column name suggests '{hint}'"
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if not pd.api.types.is_bool_dtype(df[column]):
                phone_reason = _scan_numeric_phone_values(df[column])
                if phone_reason:
                    flags[column] = phone_reason
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        reason = _scan_values(df[column])
        if reason:
            flags[column] = reason
    return flags


def apply_exclusions(df, excluded_columns):
    """Drop excluded columns. Fails closed with an empty frame if all columns are excluded."""
    excluded_set = set(excluded_columns or [])
    remaining = [c for c in df.columns if c not in excluded_set]
    if not remaining:
        return df.iloc[:, 0:0].copy(), False
    return df[remaining], True


def detect_sensitive_text(raw_text):
    """Scan raw text for likely-sensitive patterns (emails, tokens, cards, phones)."""
    if not raw_text or not isinstance(raw_text, str):
        return []
    findings = []
    sample = raw_text[:50_000]
    if _EMAIL_RE.search(sample):
        findings.append("email-like values")
    for pattern in _TOKEN_RES:
        if pattern.search(sample):
            findings.append("credential/token-like values")
            break
    if _IBAN_RE.search(sample):
        findings.append("IBAN-like values")
    for match in _CARD_RE.finditer(sample):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) in range(13, 20) and _luhn_ok(digits):
            findings.append("credit-card-like values")
            break
    if any(len(re.sub(r"\D", "", m.group(0))) >= 9 for m in _PHONE_RE.finditer(sample)):
        findings.append("phone-number-like values")
    return findings

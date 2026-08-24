# Pre-Gemini privacy screening: flag likely-sensitive columns so the user
# can exclude them from AI context with one click. Detection is heuristic by
# design -- it should produce useful warnings with few false positives, not
# replace human judgment.

import re

import pandas as pd


SAMPLE_ROWS = 100

_NAME_HINTS = (
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey",
    "ssn", "social_security", "credit_card", "card_number", "cvv", "iban",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
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
    if any(len(re.sub(r"\D", "", m.group(0))) >= 9 for m in _PHONE_RE.finditer(joined)):
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
        lowered = str(column).lower()
        hint = next((h for h in _NAME_HINTS if h in lowered), None)
        if hint:
            flags[column] = f"column name suggests '{hint}'"
            continue
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        reason = _scan_values(df[column])
        if reason:
            flags[column] = reason
    return flags


def apply_exclusions(df, excluded_columns):
    """Drop excluded columns, but refuse to strip the dataset bare."""
    remaining = [c for c in df.columns if c not in set(excluded_columns)]
    if not remaining:
        return df, False
    return df[remaining], True

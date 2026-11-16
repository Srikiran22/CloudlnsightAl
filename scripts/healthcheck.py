#!/usr/bin/env python3
"""Lightweight health probe for TabuLens Streamlit runtime."""

import sys
import urllib.error
import urllib.request

HEALTH_URL = "http://127.0.0.1:8501/_stcore/health"
TIMEOUT_SECONDS = 3.0


def check_health() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    sys.exit(1)

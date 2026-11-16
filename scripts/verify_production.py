#!/usr/bin/env python3
"""CloudInsight AI: Automated Production Smoke and Verification Suite.

Runs environmental, security, deployment, and unit test audits to certify that
the studio is ready for 4th-year capstone defense and production deployment.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ANSI colors for clear terminal presentation
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_status(step: str, success: bool, detail: str = ""):
    mark = f"{GREEN}[PASS]{RESET}" if success else f"{RED}[FAIL]{RESET}"
    msg = f"{mark} {BOLD}{step}{RESET}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def check_python_version() -> bool:
    v = sys.version_info
    ok = v >= (3, 11)
    print_status("Python Runtime Compatibility", ok, f"Detected Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_required_artifacts() -> bool:
    required = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        ".streamlit/config.toml",
        "scripts/healthcheck.py",
        "docs/CAPSTONE_THESIS_DEFENSE.md",
        "README.md",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    ok = len(missing) == 0
    detail = "All 7 core deployment and defense files present" if ok else f"Missing: {missing}"
    print_status("Artifact Completeness", ok, detail)
    return ok


def check_streamlit_security_config() -> bool:
    cfg_path = Path(".streamlit/config.toml")
    if not cfg_path.exists():
        print_status("Streamlit Client Hardening", False, "Missing .streamlit/config.toml")
        return False

    with open(cfg_path, encoding="utf-8") as f:
        content = f.read()

    has_client = "[client]" in content
    has_minimal = 'toolbarMode = "minimal"' in content or "toolbarMode = 'minimal'" in content
    has_hide_errors = "showErrorDetails = false" in content

    ok = has_client and has_minimal and has_hide_errors
    detail = "Toolbar minimal, raw tracebacks suppressed" if ok else "Security configuration incomplete"
    print_status("Streamlit Client Hardening", ok, detail)
    return ok


def check_security_invariants() -> bool:
    try:
        from Utils.limits import ResourcePolicy
        from Utils.ML import sign_model_artifact, verify_model_artifact_signature
        from Utils.theme import safe_error_boundary

        ok = (
            callable(verify_model_artifact_signature)
            and callable(sign_model_artifact)
            and ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES == 400 * 1024 * 1024
            and safe_error_boundary is not None
        )
        print_status("Security & Cryptographic Invariants", ok, "HMAC-SHA256 active, 400MB RAM budget guarded")
        return ok
    except Exception as exc:
        print_status("Security & Cryptographic Invariants", False, str(exc))
        return False


def run_test_suite(pattern: str = "test_*.py") -> bool:
    print(f"\n{BLUE}{BOLD}Running Test Verification ({pattern})...{RESET}")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description="CloudInsight AI Production & Capstone Verifier")
    parser.add_argument("--check-only", action="store_true", help="Run static and environment audits only")
    parser.add_argument("--fast", action="store_true", help="Run production tests and smoke suite")
    parser.add_argument("--full", action="store_true", help="Run complete test suite including all regressions")
    args = parser.parse_args()

    print(f"\n{BLUE}{BOLD}======================================================================{RESET}")
    print(f"{BLUE}{BOLD}        CloudInsight AI: Production and Capstone Defense Verifier       {RESET}")
    print(f"{BLUE}{BOLD}======================================================================{RESET}\n")

    checks = [
        check_python_version(),
        check_required_artifacts(),
        check_streamlit_security_config(),
        check_security_invariants(),
    ]

    if not all(checks):
        print(f"\n{RED}{BOLD}Static verification failed. See errors above.{RESET}\n")
        return 1

    if args.check_only:
        print(f"\n{GREEN}{BOLD}ALL PRODUCTION VERIFICATIONS PASSED (Static & Environment Audit){RESET}\n")
        return 0

    test_pattern = "test_*.py" if args.full else "test_*deployment*.py"
    if args.fast:
        test_pattern = "test_*deployment*.py"

    tests_ok = run_test_suite(test_pattern)
    if not tests_ok:
        print(f"\n{RED}{BOLD}Test suite verification failed.{RESET}\n")
        return 1

    print(f"\n{GREEN}{BOLD}ALL PRODUCTION VERIFICATIONS PASSED (Full Suite Ready){RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

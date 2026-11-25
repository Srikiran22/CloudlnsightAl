#!/usr/bin/env python3
"""TabuLens: Automated Pipeline & Verification Runner.

Verifies environmental integrity, directory layout, cryptographic signing invariants,
and executes the complete unit and integration test suite.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_status(step: str, success: bool, detail: str = ""):
    mark = f"{GREEN}[PASS]{RESET}" if success else f"{RED}[FAIL]{RESET}"
    msg = f"{mark} {BOLD}{step}{RESET}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def check_python_runtime() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    print_status("Python Runtime", ok, f"Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_core_dependencies() -> bool:
    deps = ["pandas", "numpy", "sklearn", "streamlit", "joblib", "plotly"]
    missing = []
    for mod in deps:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    ok = len(missing) == 0
    detail = "All tabular and modeling dependencies installed" if ok else f"Missing: {missing}"
    print_status("Core Dependencies", ok, detail)
    return ok


def check_directory_structure() -> bool:
    from utils.paths import ensure_project_directories

    ensure_project_directories()
    required_dirs = ["Datasets", "Models", "Reports", "pages", "utils", "scripts", "tests", "docs"]
    missing = [d for d in required_dirs if not (PROJECT_ROOT / d).is_dir()]
    ok = len(missing) == 0
    detail = "All workspace directories present" if ok else f"Missing directories: {missing}"
    print_status("Workspace Layout", ok, detail)
    return ok


def check_core_artifacts() -> bool:
    required_files = [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/METHODOLOGY.md",
        "docs/decisions/README.md",
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        ".github/workflows/ci.yml",
        "scripts/healthcheck.py",
        "app.py",
    ]
    missing = [f for f in required_files if not (PROJECT_ROOT / f).is_file()]
    ok = len(missing) == 0
    detail = f"All {len(required_files)} core configuration and documentation files present" if ok else f"Missing: {missing}"
    print_status("Artifact Completeness", ok, detail)
    return ok


def check_cryptographic_signing() -> bool:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from utils.ml import load_trained_model, save_trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            X = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0]})
            y = np.array(["A", "B", "A", "B"])
            pipe = Pipeline([("clf", LogisticRegression())])
            pipe.fit(X, y)
            bundle = {
                "pipeline": pipe,
                "feature_cols": ["feat"],
                "target_column": "target",
                "problem_type": "Classification",
                "classes": ["A", "B"],
            }
            saved_file = save_trained_model(bundle, "verification_probe", directory=tmp_path)
            loaded = load_trained_model(saved_file, allowed_dir=tmp_path)
            if "pipeline" not in loaded:
                print_status("Model Cryptographic HMAC Invariant", False, "Loaded bundle corrupted")
                return False

            # Tampering test: flip one byte in the artifact
            content = bytearray(saved_file.read_bytes())
            content[-1] ^= 0xFF
            saved_file.write_bytes(content)

            tamper_caught = False
            try:
                load_trained_model(saved_file, allowed_dir=tmp_path)
            except ValueError:
                tamper_caught = True

            print_status("Model Cryptographic HMAC Invariant", tamper_caught, "HMAC-SHA256 signature and tamper rejection verified")
            return tamper_caught
    except Exception as e:
        print_status("Model Cryptographic HMAC Invariant", False, f"Exception: {e}")
        return False


def run_unit_tests() -> bool:
    print(f"\n{BOLD}Running TabuLens Unit & Integration Test Suite:{RESET}")
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="TabuLens automated pipeline verification suite.")
    parser.add_argument("--check-only", action="store_true", help="Run static configuration and crypto checks without executing unit tests")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("           TabuLens: Pipeline & Architecture Verification           ")
    print("=" * 70 + "\n")

    checks = [
        check_python_runtime(),
        check_core_dependencies(),
        check_directory_structure(),
        check_core_artifacts(),
        check_cryptographic_signing(),
    ]

    all_passed = all(checks)
    if not all_passed:
        print(f"\n{RED}[ERROR] Static verification checks failed.{RESET}")
        sys.exit(1)

    if args.check_only:
        print(f"\n{GREEN}[SUCCESS] ALL STATIC VERIFICATION CHECKS PASSED.{RESET}\n")
        sys.exit(0)

    test_success = run_unit_tests()
    if test_success:
        print(f"\n{GREEN}[SUCCESS] ALL PIPELINE VERIFICATIONS AND TESTS PASSED.{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}[ERROR] Unit test suite execution failed.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

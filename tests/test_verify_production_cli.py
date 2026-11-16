import os
import subprocess
import sys
import unittest


class TestVerifyProductionCLI(unittest.TestCase):
    def test_verify_production_cli_script_exists(self):
        self.assertTrue(
            os.path.exists("scripts/verify_production.py"),
            "scripts/verify_production.py should exist"
        )

    def test_verify_production_cli_runs_and_passes(self):
        result = subprocess.run(
            [sys.executable, "scripts/verify_production.py", "--check-only"],
            capture_output=True,
            text=True
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"CLI failed with returncode {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        self.assertIn("ALL PRODUCTION VERIFICATIONS PASSED", result.stdout)

if __name__ == "__main__":
    unittest.main()

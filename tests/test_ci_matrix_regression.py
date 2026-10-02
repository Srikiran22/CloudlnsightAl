import tempfile
import unittest
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

from Utils import paths
from Utils.ML import detect_problem_type


class TestCIMatrixRegression(unittest.TestCase):
    """Regression tests for GitHub Actions CI matrix failures and cross-platform headless page boot."""

    def test_detect_problem_type_resilient_to_none_and_invalid_targets(self):
        """detect_problem_type must never raise KeyError on None, invalid target names, or empty frames."""
        # 1. df is None
        self.assertEqual(detect_problem_type(None, "col"), "Classification")

        # 2. target_col is None
        df = pd.DataFrame({"target": [1, 2, 3, 4], "feat": [10, 20, 30, 40]})
        self.assertEqual(detect_problem_type(df, None), "Classification")

        # 3. target_col is empty string or missing
        self.assertEqual(detect_problem_type(df, ""), "Classification")
        self.assertEqual(detect_problem_type(df, "nonexistent"), "Classification")

        # 4. Empty dataframe with no columns
        empty_df = pd.DataFrame()
        self.assertEqual(detect_problem_type(empty_df, None), "Classification")
        self.assertEqual(detect_problem_type(empty_df, "col"), "Classification")

        # 5. Empty series (column exists but has only nulls)
        null_df = pd.DataFrame({"col": [None, None]})
        self.assertEqual(detect_problem_type(null_df, "col"), "Regression")

    def test_ml_page_handles_empty_dataframe_without_exception(self):
        """Pages/ML.py must display a warning and stop gracefully on empty DataFrame without KeyError: None."""
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        at = AppTest.from_file(ml_script, default_timeout=20)
        at.session_state["current_df"] = pd.DataFrame()
        at.session_state["dataset_name"] = "empty.csv"
        at.run()

        self.assertEqual(len(at.exception), 0, f"Pages/ML.py raised unhandled exceptions on empty df: {[e.message for e in at.exception]}")
        self.assertTrue(
            any("at least 2 columns and at least 2 rows" in w.value for w in at.warning),
            f"Expected 2-column / 2-row warning, got warnings: {[w.value for w in at.warning]}"
        )

    def test_ml_page_handles_single_column_dataframe_without_exception(self):
        """Pages/ML.py must handle single-column dataframe safely without crashing."""
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        at = AppTest.from_file(ml_script, default_timeout=20)
        at.session_state["current_df"] = pd.DataFrame({"only_col": [1, 2, 3, 4]})
        at.session_state["dataset_name"] = "single.csv"
        at.run()

        self.assertEqual(len(at.exception), 0, f"Pages/ML.py raised unhandled exceptions on single-col df: {[e.message for e in at.exception]}")
        self.assertTrue(
            any("at least 2 columns and at least 2 rows" in w.value for w in at.warning),
            f"Expected warning on single-column df, got: {[w.value for w in at.warning]}"
        )

    def test_ml_page_headless_boot_with_non_tabular_disk_file(self):
        """Pages/ML.py must boot cleanly even if Datasets/ contains an unparseable or 0-column CSV."""
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a non-tabular file similar to what caused the CI failure
            (tmp_path / "corrupted.csv").write_text("dummy content 2", encoding="utf-8")

            orig_dir = paths.DATASETS_DIR
            paths.DATASETS_DIR = tmp_path
            try:
                at = AppTest.from_file(ml_script, default_timeout=20)
                at.run()
                self.assertEqual(len(at.exception), 0, f"Headless ML boot failed on corrupted file: {[e.message for e in at.exception]}")
            finally:
                paths.DATASETS_DIR = orig_dir

    def test_oversized_merge_test_does_not_leak_to_project_datasets_dir(self):
        """Running the oversized merge test must not write any files into project DATASETS_DIR."""
        from tests.test_audit_zero_trust_remediation import OversizedMergeRerunAppTest

        before_files = set(paths.DATASETS_DIR.glob("*")) if paths.DATASETS_DIR.exists() else set()

        suite = unittest.TestSuite()
        suite.addTest(OversizedMergeRerunAppTest("test_oversized_merge_error_survives_reruns"))
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        self.assertTrue(result.wasSuccessful())

        after_files = set(paths.DATASETS_DIR.glob("*")) if paths.DATASETS_DIR.exists() else set()
        new_files = after_files - before_files
        self.assertEqual(new_files, set(), f"OversizedMergeRerunAppTest leaked files to project DATASETS_DIR: {new_files}")


if __name__ == "__main__":
    unittest.main()

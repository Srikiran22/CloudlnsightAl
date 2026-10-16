"""Page smoke and headless execution tests: AppTest boot checks and CI matrix regressions."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

from Utils import paths
from Utils.paths import ensure_project_directories
from Utils.ML import detect_problem_type


class TestPageSmoke(unittest.TestCase):
    """Headless AppTest boot checks verifying that App.py and all Pages boot cleanly without exceptions."""

    PAGES = [
        "App.py",
        "Pages/Upload.py",
        "Pages/CloudS3.py",
        "Pages/AIConverter.py",
        "Pages/Cleaning.py",
        "Pages/Compare.py",
        "Pages/EDA.py",
        "Pages/Visualization.py",
        "Pages/ML.py",
        "Pages/Predict.py",
        "Pages/AI.py",
        "Pages/AIChat.py",
        "Pages/Report.py",
        "Pages/Dashboard.py",
    ]

    @classmethod
    def setUpClass(cls):
        ensure_project_directories()

    def test_all_pages_boot_without_exceptions(self):
        root = Path(__file__).resolve().parent.parent
        for page_rel in self.PAGES:
            with self.subTest(page=page_rel):
                page_path = str(root / page_rel)
                at = AppTest.from_file(page_path, default_timeout=60)
                at.run()
                self.assertEqual(
                    len(at.exception),
                    0,
                    f"Page {page_rel} raised unhandled exceptions: {[getattr(e, 'value', str(e)) for e in at.exception]}",
                )

    def test_load_sample_datasets_in_demo_workspace(self):
        """Demo workspace button creates Q1 and Q2 sample datasets and sets active dataset."""
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            orig_dir = paths.DATASETS_DIR
            paths.DATASETS_DIR = tmp_path
            try:
                at = AppTest.from_file(str(root / "App.py"), default_timeout=30)
                at.run()
                self.assertEqual(len(at.exception), 0)
                buttons = [b for b in at.button if "Load sample datasets" in getattr(b, "label", "")]
                self.assertTrue(len(buttons) > 0, "Demo workspace button not found")
                buttons[0].click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertTrue((tmp_path / "saas_customers_q1.csv").exists())
                self.assertTrue((tmp_path / "saas_customers_q2_drifted.csv").exists())
                self.assertEqual(at.session_state["dataset_name"], "saas_customers_q1.csv")
            finally:
                paths.DATASETS_DIR = orig_dir


class TestPageRegression(unittest.TestCase):
    """Regression tests for cross-platform headless page boot and edge case session states."""

    def test_detect_problem_type_resilient_to_none_and_invalid_targets(self):
        """detect_problem_type must never raise KeyError on None, invalid target names, or empty frames."""
        self.assertEqual(detect_problem_type(None, "col"), "Classification")
        df = pd.DataFrame({"target": [1, 2, 3, 4], "feat": [10, 20, 30, 40]})
        self.assertEqual(detect_problem_type(df, None), "Classification")
        self.assertEqual(detect_problem_type(df, ""), "Classification")
        self.assertEqual(detect_problem_type(df, "nonexistent"), "Classification")
        empty_df = pd.DataFrame()
        self.assertEqual(detect_problem_type(empty_df, None), "Classification")
        self.assertEqual(detect_problem_type(empty_df, "col"), "Classification")
        null_df = pd.DataFrame({"col": [None, None]})
        self.assertEqual(detect_problem_type(null_df, "col"), "Regression")

    def test_ml_page_handles_empty_dataframe_without_exception(self):
        """Pages/ML.py must display a warning and stop gracefully on empty DataFrame without KeyError: None."""
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        at = AppTest.from_file(ml_script, default_timeout=20)
        at.session_state["current_df"] = pd.DataFrame()
        at.session_state["dataset_name"] = "empty.csv"
        at.run()

        self.assertEqual(
            len(at.exception),
            0,
            f"Pages/ML.py raised unhandled exceptions on empty df: {[e.message for e in at.exception]}",
        )
        self.assertTrue(
            any("at least 2 columns and at least 2 rows" in w.value for w in at.warning),
            f"Expected 2-column / 2-row warning, got warnings: {[w.value for w in at.warning]}",
        )

    def test_ml_page_handles_single_column_dataframe_without_exception(self):
        """Pages/ML.py must handle single-column dataframe safely without crashing."""
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        at = AppTest.from_file(ml_script, default_timeout=20)
        at.session_state["current_df"] = pd.DataFrame({"only_col": [1, 2, 3, 4]})
        at.session_state["dataset_name"] = "single.csv"
        at.run()

        self.assertEqual(
            len(at.exception),
            0,
            f"Pages/ML.py raised unhandled exceptions on single-col df: {[e.message for e in at.exception]}",
        )
        self.assertTrue(
            any("at least 2 columns and at least 2 rows" in w.value for w in at.warning),
            f"Expected warning on single-column df, got: {[w.value for w in at.warning]}",
        )

    def test_ml_page_headless_boot_with_non_tabular_disk_file(self):
        """Pages/ML.py must boot cleanly even if Datasets/ contains an unparseable or 0-column CSV."""
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "corrupted.csv").write_text("dummy content 2", encoding="utf-8")
            orig_dir = paths.DATASETS_DIR
            paths.DATASETS_DIR = tmp_path
            try:
                at = AppTest.from_file(ml_script, default_timeout=20)
                at.run()
                self.assertEqual(
                    len(at.exception),
                    0,
                    f"Headless ML boot failed on corrupted file: {[e.message for e in at.exception]}",
                )
            finally:
                paths.DATASETS_DIR = orig_dir

    def test_oversized_merge_test_does_not_leak_to_project_datasets_dir(self):
        """Running the oversized merge test must not write any files into project DATASETS_DIR."""
        from tests.test_ai_conversion import OversizedMergeRerunAppTest

        before_files = set(paths.DATASETS_DIR.glob("*")) if paths.DATASETS_DIR.exists() else set()

        suite = unittest.TestSuite()
        suite.addTest(OversizedMergeRerunAppTest("test_oversized_merge_error_survives_reruns"))
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        self.assertTrue(result.wasSuccessful())

        after_files = set(paths.DATASETS_DIR.glob("*")) if paths.DATASETS_DIR.exists() else set()
        new_files = after_files - before_files
        self.assertEqual(
            new_files,
            set(),
            f"OversizedMergeRerunAppTest leaked files to project DATASETS_DIR: {new_files}",
        )


if __name__ == "__main__":
    unittest.main()

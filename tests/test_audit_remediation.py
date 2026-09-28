"""Regression tests for forensic audit remediation across P0, P1, P2, and P3 findings."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from Utils.dataset_ui import (
    dataframe_fingerprint,
    select_working_dataset,
)
from Utils.Gemini import get_dataset_summary_context
from Utils.ML import train_and_evaluate_model
from Utils.paths import (
    MAX_TEXT_EXTRACT_CHARS,
    MAX_XML_DEPTH,
    AIConversionRequired,
    get_dataset_row_count,
    get_unique_filename,
    read_tabular,
)
from Utils.privacy import apply_exclusions
from Utils.quality import quality_metrics
from Utils.secrets import temporary_secret


class PrivacyFailClosedTests(unittest.TestCase):
    def test_single_excluded_column_never_in_result(self):
        df = pd.DataFrame({"ssn": ["000-00-0000"], "name": ["Alice"], "score": [95]})
        reduced, applied = apply_exclusions(df, ["ssn"])
        self.assertTrue(applied)
        self.assertNotIn("ssn", reduced.columns)
        self.assertIn("name", reduced.columns)
        self.assertIn("score", reduced.columns)

    def test_all_columns_excluded_fails_closed(self):
        df = pd.DataFrame({"secret_a": [1, 2], "secret_b": [3, 4]})
        reduced, applied = apply_exclusions(df, ["secret_a", "secret_b"])
        self.assertFalse(applied)
        self.assertEqual(reduced.shape[1], 0)
        self.assertEqual(list(reduced.columns), [])
        # Crucial: original DataFrame columns are NEVER substituted back in
        self.assertNotIn("secret_a", reduced.columns)
        self.assertNotIn("secret_b", reduced.columns)

    def test_all_excluded_blocks_gemini_context(self):
        df = pd.DataFrame({"secret_a": [1, 2], "secret_b": [3, 4]})
        reduced, applied = apply_exclusions(df, ["secret_a", "secret_b"])
        with self.assertRaisesRegex(ValueError, "has no columns or all columns were excluded"):
            get_dataset_summary_context(reduced, "test.csv")


class SecretCleanupTests(unittest.TestCase):
    def test_temporary_secret_cleans_up_on_success(self):
        fake_state = {"my_key_secret": "secret123", "my_key_keep": False}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=fake_state)):
            with temporary_secret("my_key", keep_key="my_key_keep"):
                self.assertEqual(fake_state["my_key_secret"], "secret123")
            self.assertNotIn("my_key_secret", fake_state)

    def test_temporary_secret_cleans_up_on_exception(self):
        fake_state = {"api_token_secret": "tokenABC", "api_token_keep": False}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=fake_state)):
            try:
                with temporary_secret("api_token", keep_key="api_token_keep"):
                    raise RuntimeError("simulated API explosion")
            except RuntimeError:
                pass
            self.assertNotIn("api_token_secret", fake_state)

    def test_temporary_secret_preserves_when_keep_flag_set(self):
        fake_state = {"api_token_secret": "tokenABC", "api_token_keep": True}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=fake_state)):
            with temporary_secret("api_token", keep_key="api_token_keep"):
                pass
            self.assertEqual(fake_state.get("api_token_secret"), "tokenABC")


class UploadFilenameDisambiguationTests(unittest.TestCase):
    def test_filename_uniqueness_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seen = set()

            # First upload of sales.csv -> natural name preserved
            f1 = get_unique_filename("sales.csv", directory=tmp_path, extra_names=seen)
            seen.add(f1)
            (tmp_path / f1).write_text("x", encoding="utf-8")
            self.assertEqual(f1, "sales.csv")

            # Second upload of same name -> sales_1.csv
            f2 = get_unique_filename("sales.csv", directory=tmp_path, extra_names=seen)
            seen.add(f2)
            (tmp_path / f2).write_text("x", encoding="utf-8")
            self.assertEqual(f2, "sales_1.csv")

            # Third upload of same name -> sales_2.csv (NOT sales_2_1.csv)
            f3 = get_unique_filename("sales.csv", directory=tmp_path, extra_names=seen)
            seen.add(f3)
            (tmp_path / f3).write_text("x", encoding="utf-8")
            self.assertEqual(f3, "sales_2.csv")


class DataFrameFingerprintTests(unittest.TestCase):
    def test_equal_dataframes_produce_identical_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self.assertEqual(dataframe_fingerprint(df1), dataframe_fingerprint(df2))
        self.assertEqual(len(dataframe_fingerprint(df1)), 64)

    def test_row_beyond_50_changes_fingerprint(self):
        # 100 rows; modify row index 75
        df1 = pd.DataFrame({"val": list(range(100))})
        df2 = pd.DataFrame({"val": list(range(100))})
        df2.iloc[75, 0] = 999999
        self.assertNotEqual(dataframe_fingerprint(df1), dataframe_fingerprint(df2))

    def test_column_rename_changes_fingerprint(self):
        df1 = pd.DataFrame({"col_a": [1, 2]})
        df2 = pd.DataFrame({"col_b": [1, 2]})
        self.assertNotEqual(dataframe_fingerprint(df1), dataframe_fingerprint(df2))

    def test_dtype_change_changes_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1.0, 2.0]})
        self.assertNotEqual(dataframe_fingerprint(df1), dataframe_fingerprint(df2))

    def test_handles_nested_unhashable_cells_without_error(self):
        df = pd.DataFrame({"nested": [{"key": "val"}, {"key": "val2"}]})
        fp = dataframe_fingerprint(df)
        self.assertEqual(len(fp), 64)


class ParquetNestedCellSafetyTests(unittest.TestCase):
    def test_parquet_nested_cells_sanitized_and_safe_downstream(self):
        # Create a parquet buffer with a column of dicts/lists
        df_nested = pd.DataFrame({
            "id": [1, 2, 3],
            "details": [{"city": "Paris"}, {"city": "Berlin"}, {"city": "Tokyo"}],
            "tags": [["a", "b"], ["c"], ["d", "e"]],
        })
        buffer = io.BytesIO()
        df_nested.to_parquet(buffer, index=False)
        buffer.seek(0)

        ingested = read_tabular(buffer, filename="nested.parquet")
        # Cells should be serialized strings
        self.assertIsInstance(ingested["details"].iloc[0], str)
        self.assertIsInstance(ingested["tags"].iloc[0], str)

        # Downstream operations must NOT raise TypeError: unhashable type
        dup_count = int(ingested.duplicated().sum())
        self.assertEqual(dup_count, 0)

        metrics = quality_metrics(ingested)
        self.assertGreater(metrics["index"], 0.0)

        fp = dataframe_fingerprint(ingested)
        self.assertEqual(len(fp), 64)


class MLTargetFeatureSeparationTests(unittest.TestCase):
    def test_target_in_features_is_rejected(self):
        df = pd.DataFrame({
            "target": [0, 0, 1, 1],
            "feat1": [1.0, 2.0, 3.0, 4.0],
        })
        with self.assertRaisesRegex(ValueError, "cannot be included in feature columns"):
            train_and_evaluate_model(
                df,
                target_col="target",
                feature_cols=["feat1", "target"],
                model_name="Logistic Regression",
                problem_type="Classification",
                test_size=0.5,
            )

    def test_valid_separation_trains_successfully(self):
        df = pd.DataFrame({
            "target": [0, 0, 1, 1],
            "feat1": [1.0, 2.0, 3.0, 4.0],
        })
        res = train_and_evaluate_model(
            df,
            target_col="target",
            feature_cols=["feat1"],
            model_name="Logistic Regression",
            problem_type="Classification",
            test_size=0.5,
        )
        self.assertIn("accuracy", res)


class PreserveUnsavedWorkingDataTests(unittest.TestCase):
    def test_browsing_stored_file_preserves_active_session_dataset(self):
        session_df = pd.DataFrame({"active_data": [10, 20]})
        stored_df = pd.DataFrame({"stored_data": [1, 2]})

        fake_session_state = {
            "current_df": session_df,
            "dataset_name": "working_clean.csv",
            "session_fingerprint": "fake_fp",
        }

        fake_st = SimpleNamespace(
            session_state=fake_session_state,
            selectbox=lambda label, options: "stored.csv",
            warning=lambda *a, **k: None,
            info=lambda *a, **k: None,
            button=lambda *a, **k: False,
            stop=lambda: None,
        )

        with patch("Utils.dataset_ui.st", fake_st), \
             patch("Utils.dataset_ui.list_dataset_files", return_value=["stored.csv"]), \
             patch("Utils.dataset_ui.load_dataset_cached", return_value=stored_df):
            loaded_df, chosen_name = select_working_dataset("Choose:")

            # The page receives the stored dataset to view
            self.assertIs(loaded_df, stored_df)
            self.assertEqual(chosen_name, "stored.csv")

            # Crucial: the active session DataFrame was NOT clobbered!
            self.assertIs(fake_session_state["current_df"], session_df)
            self.assertEqual(fake_session_state["dataset_name"], "working_clean.csv")


class BoundedBatchReportTests(unittest.TestCase):
    def test_get_dataset_row_count_and_bounded_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "large.csv"
            # Write 500 rows
            lines = ["id,val\n"] + [f"{i},{i*2}\n" for i in range(500)]
            csv_path.write_text("".join(lines), encoding="utf-8")

            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                total_rows = get_dataset_row_count("large.csv")
                self.assertEqual(total_rows, 500)

                bounded_df = read_tabular(tmp_path / "large.csv", max_rows=100)
                self.assertEqual(len(bounded_df), 100)


class BoundedTextReadingTests(unittest.TestCase):
    def test_unstructured_text_bounded_to_max_chars(self):
        huge_text = "This is non-tabular prose narrative. " * 3000
        payload = huge_text.encode("utf-8")
        self.assertGreater(len(huge_text), MAX_TEXT_EXTRACT_CHARS)

        with self.assertRaises(AIConversionRequired) as ctx:
            read_tabular(payload, filename="big_prose.txt")

        extracted = ctx.exception.raw_text
        self.assertLessEqual(len(extracted), MAX_TEXT_EXTRACT_CHARS)


class XMLResourceBoundaryTests(unittest.TestCase):
    def test_excessively_deep_xml_is_rejected_proactively(self):
        # Build XML nested 60 levels deep (> MAX_XML_DEPTH of 50)
        deep_xml = "<root>"
        for i in range(MAX_XML_DEPTH + 10):
            deep_xml += f"<nested_{i}>"
        deep_xml += "payload"
        for i in reversed(range(MAX_XML_DEPTH + 10)):
            deep_xml += f"</nested_{i}>"
        deep_xml += "</root>"

        with self.assertRaises(ValueError) as ctx:
            read_tabular(deep_xml.encode("utf-8"), filename="deep.xml")
        self.assertIn("nesting", str(ctx.exception).lower())

    def test_normal_xml_parses_successfully(self):
        xml = "<records><row><id>1</id><name>Alice</name></row></records>"
        df = read_tabular(xml.encode("utf-8"), filename="normal.xml")
        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df["name"].iloc[0], "Alice")


if __name__ == "__main__":
    unittest.main()

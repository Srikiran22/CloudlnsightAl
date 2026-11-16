"""UI workflow tests: Upload deduplication, multi-file merge, PDF generation, and chart downsampling."""

import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.charts import create_bar_count_plot
from utils.ml import predict_with_model, train_and_evaluate_model
from utils.paths import read_tabular
from utils.pdf import _resolve_pdf_fonts, generate_pdf_report
from utils.privacy import detect_sensitive_columns


class UIRemediationRegressionTests(unittest.TestCase):
    """Regression tests covering edge-case UI, ingestion, and inference workflows."""

    def test_csv_pre_parse_column_limit_enforced_on_header(self):
        """CSV files exceeding MAX_INGESTION_COLUMNS fail fast via header inspection."""
        wide_header = ",".join(f"col_{i}" for i in range(250)) + "\n"
        data_row = ",".join("1" for _ in range(250)) + "\n"
        payload = (wide_header + data_row).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "exceeds the maximum supported limit of 200 columns"):
            read_tabular(payload, filename="wide_dataset.csv")

    def test_tsv_pre_parse_column_limit_enforced_on_header(self):
        """TSV files exceeding MAX_INGESTION_COLUMNS fail fast via header inspection."""
        wide_header = "\t".join(f"col_{i}" for i in range(210)) + "\n"
        data_row = "\t".join("val" for _ in range(210)) + "\n"
        payload = (wide_header + data_row).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "exceeds the maximum supported limit of 200 columns"):
            read_tabular(payload, filename="wide_dataset.tsv")

    def test_jsonl_pre_parse_column_limit_enforced_on_first_line(self):
        """JSONL files with > 200 keys on line 1 fail fast via first line inspection."""
        import json
        wide_obj = {f"k_{i}": i for i in range(205)}
        payload = (json.dumps(wide_obj) + "\n").encode("utf-8")
        with self.assertRaisesRegex(ValueError, "exceeds the maximum supported limit of 200 columns"):
            read_tabular(payload, filename="wide.jsonl")

    def test_privacy_detects_camel_case_identifiers(self):
        """Privacy screening detects camelCase identifiers (CustomerSSN, UserPassword, etc.)."""
        df = pd.DataFrame({
            "CustomerSSN": ["123", "456"],
            "UserPassword": ["hash1", "hash2"],
            "AccountCell": ["555", "666"],
            "ClientPwd": ["p1", "p2"],
            "NormalMetric": [10.5, 20.2],
        })
        flags = detect_sensitive_columns(df)
        self.assertIn("CustomerSSN", flags)
        self.assertIn("UserPassword", flags)
        self.assertIn("AccountCell", flags)
        self.assertIn("ClientPwd", flags)
        self.assertNotIn("NormalMetric", flags)

    def test_ml_predict_with_model_rejects_infinite_inputs(self):
        """predict_with_model validates non-finite features and raises a clear error."""
        train_df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        bundle = train_and_evaluate_model(
            train_df, "target", ["feature1", "feature2"], "Logistic Regression",
            problem_type="Classification", test_size=0.33
        )
        inf_df = pd.DataFrame({
            "feature1": [1.0, np.inf],
            "feature2": [10.0, 20.0],
        })
        with self.assertRaisesRegex(ValueError, "Inference features contain infinity"):
            predict_with_model(bundle, inf_df)

    def test_bar_chart_cardinality_capping_to_top_30(self):
        """create_bar_count_plot groups categories beyond top 30 into 'Other'."""
        data = {f"cat_{i}": 1 for i in range(100)}
        data["dominant"] = 500
        # Expand into a DataFrame with 101 unique categories
        rows = []
        for k, count in data.items():
            rows.extend([k] * count)
        df = pd.DataFrame({"category": rows})
        fig = create_bar_count_plot(df, "category", agg_func="Count")
        # In Plotly figure, check distinct values plotted
        plotted_categories = set(fig.data[0].x)
        self.assertLessEqual(len(plotted_categories), 31)
        self.assertIn("Other", plotted_categories)
        self.assertIn("dominant", plotted_categories)

    def test_pdf_report_unicode_font_resolution(self):
        """generate_pdf_report resolves a TrueType font and renders non-Latin characters."""
        reg, bold = _resolve_pdf_fonts()
        self.assertNotEqual(reg, "")
        self.assertNotEqual(bold, "")
        df = pd.DataFrame({
            "用户ID": ["张伟", "李娜"],
            "Score": [100, 200]
        })
        pdf_bytes = generate_pdf_report(
            df=df,
            dataset_name="测试.csv",
            report_title="测试报告",
            author_name="审计员",
            include_ai_insights="分析完成",
            include_charts=False
        )
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


# ======================================================================
# Additional domain tests
# ======================================================================

"""Comprehensive test suite for audit remediation (H-1 through H-5 and M-1 through M-9).

Includes:
- Streamlit AppTest lifecycle workflows (Upload deduplication)
- Resource amplification guards (Parquet decompression bombs, aggregate upload limits)
- Data integrity & parsing regression tests (JSON structures, CSV injection)
- Visualization, ML feature importance, PDF boolean handling, privacy screening, and manifest robustness.
"""

import logging
import os
import shutil
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

from utils.charts import create_scatter_plot
from utils.logsys import get_logger
from utils.paths import (
    DATASETS_DIR,
    MAX_INGESTION_ROWS,
    get_dataset_row_count,
    get_valid_conversion,
    record_conversion,
    sanitize_for_csv_export,
)
from utils.privacy import (
    detect_sensitive_text,
)


class StreamlitUploadWorkflowTests(unittest.TestCase):
    """Workflow and AppTest tests for pages/upload.py (H-1, H-5)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_datasets_dir = DATASETS_DIR
        import utils.paths as paths
        paths.DATASETS_DIR = Path(self.temp_dir)

    def tearDown(self):
        import utils.paths as paths
        paths.DATASETS_DIR = self.orig_datasets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upload_rerun_deduplication(self):
        """H-1: Uploading a file and triggering a rerun does NOT write duplicate files to disk."""
        upload_script = str(Path(__file__).resolve().parent.parent / "pages" / "upload.py")
        at = AppTest.from_file(upload_script, default_timeout=15)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # 1. Upload initial file
        csv_bytes = b"col_a,col_b\n1,2\n3,4\n"
        at.file_uploader[0].upload("data.csv", csv_bytes)
        at.run()
        self.assertEqual(len(at.exception), 0)

        written_files_run1 = sorted([f.name for f in Path(self.temp_dir).glob("*.csv")])
        self.assertEqual(written_files_run1, ["data.csv"])

        # 2. Trigger rerun with unchanged upload widget
        at.run()
        self.assertEqual(len(at.exception), 0)
        written_files_run2 = sorted([f.name for f in Path(self.temp_dir).glob("*.csv")])
        # Crucial check: rerun did NOT generate data_1.csv
        self.assertEqual(written_files_run2, ["data.csv"])

        # 3. Upload genuinely changed content under the same filename
        at.file_uploader[0].clear()
        changed_bytes = b"col_a,col_b\n10,20\n30,40\n"
        at.file_uploader[0].upload("data.csv", changed_bytes)
        at.run()
        self.assertEqual(len(at.exception), 0)

        written_files_run3 = sorted([f.name for f in Path(self.temp_dir).glob("*.csv")])
        # Changed content correctly recognized and saved with unique filename
        self.assertIn("data.csv", written_files_run3)
        self.assertIn("data_1.csv", written_files_run3)

    def test_upload_aggregate_file_count_limit(self):
        """H-5: Exceeding MAX_UPLOAD_FILES triggers error and rejects upload."""
        from pages.upload import MAX_UPLOAD_FILES

        upload_script = str(Path(__file__).resolve().parent.parent / "pages" / "upload.py")
        at = AppTest.from_file(upload_script, default_timeout=15)
        at.run()

        # Simulate 21 uploaded files
        for i in range(MAX_UPLOAD_FILES + 1):
            at.file_uploader[0].upload(f"file_{i}.csv", b"a,b\n1,2\n")
        at.run()

        # Should display error about too many files
        error_messages = [e.value for e in at.error]
        self.assertTrue(
            any("Maximum allowed is 20 files" in msg for msg in error_messages),
            f"Expected max upload files error, got: {error_messages}"
        )
        # Disk should have zero files written
        self.assertEqual(len(list(Path(self.temp_dir).glob("*.csv"))), 0)

    def test_upload_aggregate_byte_limit(self):
        """H-5: Exceeding MAX_AGGREGATE_UPLOAD_BYTES triggers error."""
        with patch("utils.paths.MAX_AGGREGATE_UPLOAD_BYTES", 5):
            upload_script = str(Path(__file__).resolve().parent.parent / "pages" / "upload.py")
            at = AppTest.from_file(upload_script, default_timeout=15)
            at.run()
            at.file_uploader[0].upload("data.csv", b"col_a,col_b\n1,2\n3,4\n")
            at.run()

            error_messages = [e.value for e in at.error]
            self.assertTrue(
                any("exceeds aggregate limit" in msg for msg in error_messages),
                f"Expected aggregate size error, got: {error_messages}"
            )
            # Disk should have zero files written
            self.assertEqual(len(list(Path(self.temp_dir).glob("*.csv"))), 0)


class PDFGenerationRobustnessTests(unittest.TestCase):
    """H-3: Boolean columns must not crash PDF report generation."""

    def test_pdf_report_with_boolean_column(self):
        df = pd.DataFrame({
            "is_active": [True, False, True],
            "value": [1, 2, 3]
        })
        pdf_bytes = generate_pdf_report(df, "bool_test")
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_pdf_report_with_all_boolean_and_mixed_types(self):
        df = pd.DataFrame({
            "flag1": [True, False, False, True],
            "flag2": [False, False, False, False],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [95.5, 82.0, 78.4, 91.2],
        })
        pdf_bytes = generate_pdf_report(df, "mixed_test")
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


class ParquetIngestionSafetyTests(unittest.TestCase):
    """H-4: Parquet decompression bomb and excessive row count rejection."""

    def test_parquet_excessive_row_count_rejected_before_decompression(self):
        """A small compressed Parquet file (> 1,000,000 rows) is rejected in milliseconds."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_arrays([pa.array(["sample"] * 1_000_001)], names=["col"])
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        payload = buf.getvalue()

        # Serialized payload is tiny (< 10 KB)
        self.assertLess(len(payload), 50_000)

        with self.assertRaisesRegex(ValueError, f"exceeds the maximum supported limit of {MAX_INGESTION_ROWS:,} rows"):
            read_tabular(payload, filename="bomb.parquet")

    def test_valid_parquet_within_limit_succeeds(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_arrays([pa.array([1, 2, 3]), pa.array(["a", "b", "c"])], names=["x", "y"])
        buf = io.BytesIO()
        pq.write_table(table, buf)
        df = read_tabular(buf.getvalue(), filename="safe.parquet")
        self.assertEqual(df.shape, (3, 2))


class JSONStructureHandlingTests(unittest.TestCase):
    """M-2: Robust JSON tabular parsing across columnar, object, and list structures."""

    def test_case_a_columnar_json(self):
        """Case A: Columnar JSON format -> DataFrame with 2 rows."""
        payload = b'{"id": [1, 2], "score": [10, 20]}'
        df = read_tabular(payload, filename="columnar.json")
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(list(df.columns), ["id", "score"])
        self.assertEqual(df["score"].tolist(), [10, 20])

    def test_case_b_object_with_list_field(self):
        """Case B: Single object with list property -> 1 row without discarding scalar keys."""
        payload = b'{"id": 7, "name": "Alice", "tags": ["admin", "user"]}'
        df = read_tabular(payload, filename="single_obj.json")
        self.assertEqual(df.shape, (1, 3))
        self.assertEqual(df.iloc[0]["id"], 7)
        self.assertEqual(df.iloc[0]["name"], "Alice")
        # List field preserved (serialized or as list)
        self.assertTrue(isinstance(df.iloc[0]["tags"], (list, str)))

    def test_case_c_list_of_records(self):
        """Case C: Standard list of records -> tabular dataset."""
        payload = b'[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        df = read_tabular(payload, filename="records.json")
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df["name"].tolist(), ["Alice", "Bob"])

    def test_ambiguous_nested_json_raises_value_error(self):
        """Ambiguous nested structures raise clear ValueError."""
        payload = b'{"users": [{"id": 1}], "orders": [{"id": 2}]}'
        with self.assertRaisesRegex(ValueError, r"(?i)ambiguous.*structure"):
            read_tabular(payload, filename="nested.json")


class CSVHeaderInjectionSanitizationTests(unittest.TestCase):
    """M-3: Sanitization against CSV formula injection on column headers."""

    def test_formula_headers_prepended_with_single_quote(self):
        df = pd.DataFrame({
            "=HYPERLINK('http://evil.com')": [1],
            "+CMD('calc')": [2],
            "@SUM(A1:A10)": [3],
            "-HARMFUL(123)": [4],
            "normal_header": [5],
        })
        sanitized = sanitize_for_csv_export(df)
        cols = list(sanitized.columns)
        self.assertEqual(cols[0], "'=HYPERLINK('http://evil.com')")
        self.assertEqual(cols[1], "'+CMD('calc')")
        self.assertEqual(cols[2], "'@SUM(A1:A10)")
        self.assertEqual(cols[3], "'-HARMFUL(123)")
        self.assertEqual(cols[4], "normal_header")


class PrivacyScreeningTests(unittest.TestCase):
    """M-4: Privacy screening heuristics and detection coverage."""

    def test_column_name_hints_detected(self):
        df = pd.DataFrame({
            "user_phone": ["123"],
            "mobile_no": ["456"],
            "work_email": ["test@example.com"],
            "tel_primary": ["789"],
            "cell_num": ["101"],
            "safe_metric": [1.0],
        })
        sensitive = detect_sensitive_columns(df)
        for col in ["user_phone", "mobile_no", "work_email", "tel_primary", "cell_num"]:
            self.assertIn(col, sensitive)
        self.assertNotIn("safe_metric", sensitive)

    def test_phone_regex_newline_boundary(self):
        """Phone regex must not match numbers split across newline boundaries."""
        df = pd.DataFrame({
            "notes": ["line 1 1234\n5678 line 2", "no numbers here"]
        })
        sensitive = detect_sensitive_columns(df)
        self.assertNotIn("notes", sensitive)

    def test_numeric_phone_values_detected(self):
        """Integer column containing 10-digit phone numbers detected."""
        df = pd.DataFrame({
            "contact": [9876543210, 9123456789, 9988776655, 9000000001] * 25
        })
        sensitive = detect_sensitive_columns(df)
        self.assertIn("contact", sensitive)
        self.assertIn("phone-number-like numeric values", sensitive["contact"])

    def test_detect_sensitive_text_in_raw_text(self):
        text_with_creds = "Here is my secret token: sk-12345678901234567890123456 and email me@domain.com"
        findings = detect_sensitive_text(text_with_creds)
        self.assertIn("credential/token-like values", findings)
        self.assertIn("email-like values", findings)


class VisualizationRobustnessTests(unittest.TestCase):
    """M-5: Visualization safety on edge cases."""

    def test_bar_count_plot_same_x_and_y(self):
        """create_bar_count_plot succeeds when x_col == y_col."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C", "B"]})
        fig = create_bar_count_plot(df, "category", "category")
        self.assertIsNotNone(fig)

    def test_scatter_plot_non_finite_values(self):
        """create_scatter_plot safely filters out NaN, inf, and -inf."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0, np.inf, 6.0],
            "y": [2.0, np.nan, 6.0, 8.0, -np.inf, 12.0],
        })
        fig = create_scatter_plot(df, "x", "y")
        self.assertIsNotNone(fig)


class MLFeatureImportanceAlignmentTests(unittest.TestCase):
    """M-6: Feature importance mapping to preprocessed feature names."""

    def test_feature_importance_aligned_with_preprocessing(self):
        df = pd.DataFrame({
            "category": ["cat", "dog", "bird", "cat", "dog", "cat", "bird", "dog", "cat", "cat"],
            "numeric1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "numeric2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        res = train_and_evaluate_model(
            df,
            target_col="target",
            feature_cols=["category", "numeric1", "numeric2"],
            model_name="Random Forest",
            problem_type="Classification",
        )
        self.assertIn("feature_importances", res)
        importances = res["feature_importances"]
        # Transformed features include numeric1, numeric2, and one-hot category columns
        self.assertGreater(len(importances), 3)
        self.assertTrue(any("category_" in k for k in importances))
        self.assertIn("numeric1", importances)
        self.assertIn("numeric2", importances)


class ConversionManifestRobustnessTests(unittest.TestCase):
    """M-7: Corrupted manifest recovery and atomic manifest writes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_datasets_dir = DATASETS_DIR
        import utils.paths as paths
        paths.DATASETS_DIR = Path(self.temp_dir)

    def tearDown(self):
        import utils.paths as paths
        paths.DATASETS_DIR = self.orig_datasets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_non_dict_manifest_recovery(self):
        manifest_path = Path(self.temp_dir) / ".conversions.json"
        manifest_path.write_text("[]", encoding="utf-8")  # Non-dict JSON

        res = get_valid_conversion("source.txt", "raw text")
        self.assertIsNone(res)

    def test_atomic_manifest_recording(self):
        target_path = Path(self.temp_dir) / "target.csv"
        target_path.write_text("a,b\n1,2\n", encoding="utf-8")
        record_conversion("target.csv", "source.txt", "raw text", model_name="test-model")
        res = get_valid_conversion("source.txt", "raw text", model_name="test-model")
        self.assertEqual(res, "target.csv")


class DatasetRowCountConsistencyTests(unittest.TestCase):
    """M-8: Report row count consistency ignoring trailing/blank records."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_datasets_dir = DATASETS_DIR
        import utils.paths as paths
        paths.DATASETS_DIR = Path(self.temp_dir)

    def tearDown(self):
        import utils.paths as paths
        paths.DATASETS_DIR = self.orig_datasets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_row_count_ignores_trailing_blanks(self):
        csv_file = Path(self.temp_dir) / "trailing.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n\n\n\n", encoding="utf-8")
        count = get_dataset_row_count("trailing.csv")
        self.assertEqual(count, 2)

    def test_row_count_multiline_fields(self):
        csv_file = Path(self.temp_dir) / "multiline.csv"
        csv_file.write_text('id,desc\n1,"line1\nline2"\n2,"normal"\n', encoding="utf-8")
        count = get_dataset_row_count("multiline.csv")
        self.assertEqual(count, 2)


class LoggingLevelValidationTests(unittest.TestCase):
    """M-9: Logger level validation falls back safely on invalid values."""

    def test_invalid_logging_string_falls_back_to_warning(self):
        import utils.logsys as logsys
        logsys._configured = False
        with patch.dict(os.environ, {"CLOUDINSIGHT_LOG_LEVEL": "BASIC_FORMAT"}):
            logger = get_logger("TestLoggerInvalid")
            self.assertEqual(logger.getEffectiveLevel(), logging.WARNING)

    def test_valid_logging_string_accepted(self):
        import utils.logsys as logsys
        logsys._configured = False
        with patch.dict(os.environ, {"CLOUDINSIGHT_LOG_LEVEL": "DEBUG"}):
            logger = get_logger("TestLoggerValid")
            self.assertEqual(logger.getEffectiveLevel(), logging.DEBUG)


class UploadPageStudioAppTests(unittest.TestCase):
    """AppTest regression tests for Data Hub & Ingest page tabs and catalog."""

    def test_upload_page_renders_tabs_and_runs_cleanly(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent.parent
        at = AppTest.from_file(str(root / "pages" / "upload.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.tabs), 4)


if __name__ == '__main__':
    unittest.main()


"""Comprehensive regression tests for final hardening and forensic re-audit remediation."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from Utils.AIConvert import parse_ai_csv
from Utils.Charts import create_bar_count_plot
from Utils.compare_logic import column_drift_rows, schema_diff
from Utils.paths import normalize_and_deduplicate_columns, read_tabular
from Utils.PDF import generate_pdf_report
from Utils.secrets import drop, release


class XMLMultiEncodingHardeningTests(unittest.TestCase):
    def test_utf16_internal_entity_rejected(self):
        payload = '''<?xml version="1.0" encoding="utf-16"?>
<!DOCTYPE lolz [
  <!ENTITY a "EXPANDED_ENTITY">
]>
<root><row><val>&a;</val></row></root>'''.encode("utf-16")
        with self.assertRaisesRegex(ValueError, "DTD/entity"):
            read_tabular(payload, filename="exploit.xml")

    def test_utf16le_doctype_rejected(self):
        payload = '''<?xml version="1.0" encoding="utf-16"?>
<!DOCTYPE root SYSTEM "evil.dtd">
<root><row><val>1</val></row></root>'''.encode("utf-16le")
        with self.assertRaisesRegex(ValueError, "DTD/entity"):
            read_tabular(payload, filename="exploit_le.xml")

    def test_utf16be_doctype_rejected(self):
        payload = '''<?xml version="1.0" encoding="utf-16"?>
<!DOCTYPE root [ <!ENTITY test "x"> ]>
<root><row><val>1</val></row></root>'''.encode("utf-16be")
        with self.assertRaisesRegex(ValueError, "DTD/entity"):
            read_tabular(payload, filename="exploit_be.xml")

    def test_utf32_doctype_rejected(self):
        payload = '''<?xml version="1.0" encoding="utf-32"?>
<!DOCTYPE root [ <!ENTITY x "32bit"> ]>
<root><row><val>&x;</val></row></root>'''.encode("utf-32")
        with self.assertRaisesRegex(ValueError, "DTD/entity"):
            read_tabular(payload, filename="exploit_32.xml")

    def test_padded_utf16_doctype_rejected(self):
        pad = "<!--" + "A" * 70000 + "-->"
        payload = f'''<?xml version="1.0" encoding="utf-16"?>{pad}<!DOCTYPE root><root><row><val>1</val></row></root>'''.encode("utf-16")
        with self.assertRaisesRegex(ValueError, "DTD/entity"):
            read_tabular(payload, filename="padded_utf16.xml")

    def test_valid_utf16_xml_parses_successfully(self):
        payload = '''<?xml version="1.0" encoding="utf-16"?>
<root>
  <row><id>1</id><name>Alice</name></row>
  <row><id>2</id><name>Bob</name></row>
</root>'''.encode("utf-16")
        df = read_tabular(payload, filename="valid_utf16.xml")
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["id", "name"])
        self.assertEqual(df["name"].tolist(), ["Alice", "Bob"])


class IngestionBoundingTests(unittest.TestCase):
    def test_parquet_row_bounding(self):
        df_large = pd.DataFrame({"id": range(100), "val": [f"val_{i}" for i in range(100)]})
        buf = io.BytesIO()
        df_large.to_parquet(buf, index=False)
        buf.seek(0)
        bounded = read_tabular(buf, filename="data.parquet", max_rows=7)
        self.assertEqual(len(bounded), 7)
        self.assertEqual(bounded["id"].tolist(), list(range(7)))

    def test_json_row_bounding(self):
        records = [{"a": i, "b": f"str_{i}"} for i in range(50)]
        import json
        payload = json.dumps(records).encode("utf-8")
        bounded = read_tabular(payload, filename="data.json", max_rows=4)
        self.assertEqual(len(bounded), 4)
        self.assertEqual(bounded["a"].tolist(), [0, 1, 2, 3])

    def test_xml_row_bounding(self):
        rows = "".join(f"<row><id>{i}</id><val>{i*2}</val></row>" for i in range(50))
        payload = f"<root>{rows}</root>".encode("utf-8")
        bounded = read_tabular(payload, filename="data.xml", max_rows=5)
        self.assertEqual(len(bounded), 5)
        self.assertEqual(bounded["id"].tolist(), ["0", "1", "2", "3", "4"])

    def test_html_row_bounding(self):
        rows = "".join(f"<tr><td>{i}</td><td>val_{i}</td></tr>" for i in range(50))
        payload = f"<table><tr><th>ID</th><th>Val</th></tr>{rows}</table>".encode("utf-8")
        bounded = read_tabular(payload, filename="data.html", max_rows=6)
        self.assertEqual(len(bounded), 6)

    def test_json_streaming_bounds_materialization(self):
        # Items beyond max_rows contain invalid JSON syntax. Incremental streaming must
        # succeed and materialize only the requested rows without parsing trailing invalid syntax.
        payload = b'[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, INVALID_TRAILING_PAYLOAD'
        bounded = read_tabular(payload, filename="stream.json", max_rows=2)
        self.assertEqual(len(bounded), 2)
        self.assertEqual(bounded["id"].tolist(), [1, 2])
        self.assertEqual(bounded["name"].tolist(), ["A", "B"])
        with self.assertRaises(ValueError):
            read_tabular(payload, filename="stream.json", max_rows=5)

    def test_xml_streaming_bounds_materialization(self):
        # Elements beyond max_rows contain malformed XML syntax. iterparse early-stop must
        # succeed and materialize only the requested rows without parsing trailing broken elements.
        payload = b"<root><row><id>1</id><name>A</name></row><row><id>2</id><name>B</name></row><unclosed_tag>"
        bounded = read_tabular(payload, filename="stream.xml", max_rows=2)
        self.assertEqual(len(bounded), 2)
        self.assertEqual(bounded["id"].tolist(), ["1", "2"])
        self.assertEqual(bounded["name"].tolist(), ["A", "B"])
        with self.assertRaises(ValueError):
            read_tabular(payload, filename="stream.xml", max_rows=5)


class MixedTypeAndDuplicateColumnTests(unittest.TestCase):
    def test_schema_diff_mixed_type_columns(self):
        df_a = pd.DataFrame([[1, 2, 3]], columns=[1, "2", "name"])
        df_b = pd.DataFrame([[4, 5, 6]], columns=["1", 2, "name"])
        common, _only_a, _only_b = schema_diff(df_a, df_b)
        # Should compare cleanly without TypeError
        self.assertIn("name", common)
        self.assertIn("1", common)
        self.assertIn("2", common)

    def test_column_drift_rows_duplicate_columns_no_crash(self):
        df_a = pd.DataFrame([[1, 2]], columns=["x", "x"])
        df_b = pd.DataFrame([[3, 4]], columns=["x", "x"])
        # Duplicate columns must not raise AttributeError: 'DataFrame' object has no attribute 'dtype'
        rows = column_drift_rows(df_a, df_b)
        self.assertTrue(len(rows) >= 1)
        self.assertEqual(rows[0]["Column"], "x")

    def test_pdf_report_with_mixed_type_and_integer_headers(self):
        df = pd.DataFrame({
            1: [10.0, 20.0, 30.0, 40.0],
            "feature_b": [1.0, 2.0, 3.0, 4.0],
            3: [100.0, 90.0, 80.0, 70.0],
        })
        # Integer column names must not crash escape() with AttributeError
        pdf_bytes = generate_pdf_report(
            df=df,
            dataset_name="int_headers.csv",
            report_title="Integer Columns Report",
            author_name="Tester",
        )
        self.assertTrue(len(pdf_bytes) > 1000)

    def test_normalize_and_deduplicate_columns(self):
        df = pd.DataFrame([[1, 2, 3, 4]], columns=[100, " 100 ", "name", "name"])
        normalized = normalize_and_deduplicate_columns(df)
        self.assertEqual(list(normalized.columns), ["100", "100_2", "name", "name_2"])
        for col in normalized.columns:
            self.assertIsInstance(col, str)

    def test_normalize_and_deduplicate_columns_chained_suffix_collisions(self):
        # Chained collisions: ['a', 'a_2', 'a'] must produce unique columns (['a', 'a_2', 'a_3'])
        df1 = pd.DataFrame([[1, 2, 3]], columns=["a", "a_2", "a"])
        norm1 = normalize_and_deduplicate_columns(df1)
        self.assertEqual(list(norm1.columns), ["a", "a_2", "a_3"])
        self.assertEqual(len(set(norm1.columns)), len(norm1.columns))

        # Higher-order chained collisions: ['a', 'a_2', 'a_3', 'a', 'a']
        df2 = pd.DataFrame([[1, 2, 3, 4, 5]], columns=["a", "a_2", "a_3", "a", "a"])
        norm2 = normalize_and_deduplicate_columns(df2)
        self.assertEqual(list(norm2.columns), ["a", "a_2", "a_3", "a_4", "a_5"])
        self.assertEqual(len(set(norm2.columns)), len(norm2.columns))

        # Collision with empty string converting to 'unnamed'
        df3 = pd.DataFrame([[1, 2, 3, 4]], columns=["", " ", "unnamed", "unnamed_2"])
        norm3 = normalize_and_deduplicate_columns(df3)
        self.assertEqual(list(norm3.columns), ["unnamed", "unnamed_2", "unnamed_3", "unnamed_2_2"])
        self.assertEqual(len(set(norm3.columns)), len(norm3.columns))

    def test_ai_convert_deduplicates_headers(self):
        # AI response with duplicate column names (e.g. whitespace-variant duplicates like 'col ' and 'col')
        # must be normalized and deduplicated by normalize_and_deduplicate_columns
        csv_text = "col ,col\n10,20\n40,50"
        df = parse_ai_csv(csv_text)
        self.assertEqual(len(set(df.columns)), len(df.columns))
        self.assertEqual(list(df.columns), ["col", "col_2"])
        self.assertEqual(len(df), 2)

        # Chained suffix collision in AI response
        csv_chained = "col ,col_2,col\n10,20,30\n40,50,60"
        df_chained = parse_ai_csv(csv_chained)
        self.assertEqual(len(set(df_chained.columns)), len(df_chained.columns))
        self.assertEqual(list(df_chained.columns), ["col", "col_2", "col_3"])


class AnalyticalAndVisualizationHardeningTests(unittest.TestCase):
    def test_plotly_bar_count_chart_single_trace_when_no_hue(self):
        df = pd.DataFrame({"category": [f"cat_{i % 50}" for i in range(500)]})
        fig = create_bar_count_plot(df, x_col="category")
        # px.histogram with hue=None must not have 50 traces
        self.assertEqual(len(fig.data), 1)

    def test_pdf_correlation_caps_to_top_50_columns(self):
        # 60 numeric columns -> must cap without calculating full 60x60 matrix
        data = {f"col_{i}": np.random.randn(20) for i in range(60)}
        df = pd.DataFrame(data)
        pdf_bytes = generate_pdf_report(
            df=df,
            dataset_name="wide_data.csv",
            report_title="Wide Data Report",
            author_name="Tester",
        )
        self.assertTrue(len(pdf_bytes) > 500)


class SecretStateResetTests(unittest.TestCase):
    def test_release_rotates_widget_version(self):
        state = {"gemini_secret": "my-secret-key", "gemini_ver": 0, "gemini_keep": False}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=state)):
            cleared = release("gemini")
            self.assertIn("gemini", cleared)
            self.assertNotIn("gemini_secret", state)
            # Widget version must be bumped to force Streamlit widget reset
            self.assertEqual(state.get("gemini_ver"), 1)

    def test_drop_rotates_widget_version(self):
        state = {"gemini_secret": "my-secret-key", "gemini_ver": 2}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=state)):
            drop("gemini")
            self.assertNotIn("gemini_secret", state)
            self.assertEqual(state.get("gemini_ver"), 3)


class NonFiniteValueHandlingTests(unittest.TestCase):
    def test_dashboard_slider_finite_range_filtering(self):
        s = pd.Series([np.nan, -np.inf, 10.0, 50.0, np.inf, 100.0])
        finite_s = s[np.isfinite(s)]
        self.assertFalse(finite_s.empty)
        min_v = float(finite_s.min())
        max_v = float(finite_s.max())
        self.assertEqual(min_v, 10.0)
        self.assertEqual(max_v, 100.0)
        self.assertTrue(np.isfinite(min_v))
        self.assertTrue(np.isfinite(max_v))

    def test_ml_prediction_line_finite_filtering(self):
        pred_df = pd.DataFrame({
            "Actual": [1.0, 2.0, np.inf, 4.0],
            "Predicted": [-np.inf, 2.1, 3.1, np.nan],
        })
        valid_actual = pred_df["Actual"].dropna()
        valid_actual = valid_actual[np.isfinite(valid_actual)]
        valid_pred = pred_df["Predicted"].dropna()
        valid_pred = valid_pred[np.isfinite(valid_pred)]
        self.assertTrue(len(valid_actual) > 0 and len(valid_pred) > 0)
        min_val = min(valid_actual.min(), valid_pred.min())
        max_val = max(valid_actual.max(), valid_pred.max())
        self.assertTrue(np.isfinite(min_val))
        self.assertTrue(np.isfinite(max_val))
        self.assertEqual(min_val, 1.0)
        self.assertEqual(max_val, 4.0)


class EDAOutlierDecouplingTests(unittest.TestCase):
    def test_outlier_detection_not_capped_by_correlation(self):
        # 60 numeric columns: columns 0..49 have huge variance so they take up all 50 slots in corr_df
        data = {}
        for i in range(50):
            data[f"c_{i}"] = [float(j * 10000 + i) for j in range(100)]
        for i in range(50, 60):
            # c_50..c_59 have smaller variance
            data[f"c_{i}"] = [float(j) for j in range(100)]
        # c_55 has a clear outlier for a 0..99 distribution
        data["c_55"][0] = 500.0
        df = pd.DataFrame(data)
        numeric_df = df.select_dtypes(include=[np.number])
        self.assertEqual(len(numeric_df.columns), 60)

        # Correlated df caps to top 50 by variance, excluding c_55
        variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
        corr_cols = variances.head(50).index.tolist() if len(variances) > 50 else numeric_df.columns.tolist()
        corr_df = numeric_df[corr_cols]
        self.assertEqual(len(corr_df.columns), 50)
        self.assertNotIn("c_55", corr_df.columns)
        self.assertEqual(len(numeric_df.columns), 60)

        # Outlier detection runs on full numeric_df, finding c_55
        outlier_counts = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                count = int(((series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))).sum())
                if count > 0:
                    outlier_counts[col] = count
        self.assertIn("c_55", outlier_counts)


class FingerprintFreshnessAndCacheInvalidationTests(unittest.TestCase):
    def test_results_match_active_detects_content_replacement(self):
        import os
        from Utils.paths import DATASETS_DIR
        from Utils.dataset_ui import (
            dataset_fingerprint,
            results_match_active,
            invalidate_dataset_cache,
        )
        test_file = DATASETS_DIR / "temp_fresh_probe.csv"
        test_file.write_text("x,y\n1,2", encoding="utf-8")
        try:
            stat1 = test_file.stat()
            fp1 = dataset_fingerprint("temp_fresh_probe.csv")
            results = {"dataset_name": "temp_fresh_probe.csv", "dataset_fingerprint": fp1}
            self.assertTrue(results_match_active(results, "temp_fresh_probe.csv"))

            # Replace content while preserving file size and mtime
            test_file.write_text("x,y\n2,1", encoding="utf-8")
            os.utime(test_file, ns=(stat1.st_atime_ns, stat1.st_mtime_ns))

            # results_match_active uses force_refresh=True, so it catches the change!
            self.assertFalse(results_match_active(results, "temp_fresh_probe.csv"))
        finally:
            if test_file.exists():
                test_file.unlink()
            invalidate_dataset_cache("temp_fresh_probe.csv")


class UnhashableCellOptimizationTests(unittest.TestCase):
    def test_pure_numeric_columns_fast_path(self):
        from Utils.paths import _sanitize_unhashable_cells
        df = pd.DataFrame({"ints": range(10), "floats": [1.1 * i for i in range(10)]})
        # Should return quickly without modifying dtypes
        sanitized = _sanitize_unhashable_cells(df)
        self.assertTrue(pd.api.types.is_integer_dtype(sanitized["ints"]))
        self.assertTrue(pd.api.types.is_float_dtype(sanitized["floats"]))

    def test_compound_structures_in_object_columns_serialized(self):
        from Utils.paths import _sanitize_unhashable_cells
        df = pd.DataFrame({"mixed": [{"a": 1}, [2, 3], "plain_str", 42]})
        sanitized = _sanitize_unhashable_cells(df)
        self.assertEqual(sanitized["mixed"].iloc[0], '{"a": 1}')
        self.assertEqual(sanitized["mixed"].iloc[1], "[2, 3]")
        self.assertEqual(sanitized["mixed"].iloc[2], "plain_str")


class AuditAdversarialFindingsTests(unittest.TestCase):
    def test_json_scalar_array_without_max_rows(self):
        # JSON arrays of scalar primitives (ints, strings, floats)
        df_int = read_tabular(b"[1, 2, 3]", filename="ints.json")
        self.assertEqual(len(df_int), 3)
        self.assertEqual(list(df_int.columns), ["0"])
        self.assertEqual(df_int["0"].tolist(), [1, 2, 3])

        df_str = read_tabular(b'["alpha", "beta", "gamma"]', filename="strings.json")
        self.assertEqual(len(df_str), 3)
        self.assertEqual(list(df_str.columns), ["0"])
        self.assertEqual(df_str["0"].tolist(), ["alpha", "beta", "gamma"])

    def test_json_scalar_array_with_max_rows(self):
        # Scalar primitives with max_rows streaming early stop
        df_bounded = read_tabular(b"[10, 20, 30, 40, 50]", filename="ints_stream.json", max_rows=3)
        self.assertEqual(len(df_bounded), 3)
        self.assertEqual(list(df_bounded.columns), ["0"])
        self.assertEqual(df_bounded["0"].tolist(), [10, 20, 30])

    def test_html_multi_table_with_max_rows_selects_main_table(self):
        # Nav table has 3 rows; main data table has 10 rows.
        # With max_rows=2, the main table must be selected (not hijacked by nav table)
        # and bounded to 2 data rows.
        html_payload = (
            b"<html><body>"
            b"<table><tr><td>Nav 1</td></tr><tr><td>Nav 2</td></tr><tr><td>Nav 3</td></tr></table>"
            b"<table><tr><th>ID</th><th>Name</th><th>Score</th></tr>"
            + b"".join(f"<tr><td>{i}</td><td>User_{i}</td><td>{i*10}</td></tr>".encode() for i in range(1, 11))
            + b"</table></body></html>"
        )
        df = read_tabular(html_payload, filename="report.html", max_rows=2)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["ID", "Name", "Score"])
        self.assertEqual(df["ID"].tolist(), ["1", "2"])

    def test_html_multi_table_without_max_rows_selects_main_table(self):
        html_payload = (
            b"<html><body>"
            b"<table><tr><td>Nav 1</td></tr><tr><td>Nav 2</td></tr></table>"
            b"<table><tr><th>ID</th><th>Name</th></tr>"
            b"<tr><td>1</td><td>Alice</td></tr><tr><td>2</td><td>Bob</td></tr><tr><td>3</td><td>Charlie</td></tr>"
            b"</table></body></html>"
        )
        df = read_tabular(html_payload, filename="report.html")
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ["ID", "Name"])

    def test_column_drift_rows_unhashable_cells_accurate_uniqueness(self):
        # In-memory DataFrames with unhashable structures (lists, dicts, arrays)
        df_a = pd.DataFrame({
            "lists": [[1, 2], [1, 2], [3, 4], None],
            "dicts": [{"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 2}, None],
            "arrays": [np.array([1, 2]), np.array([1, 2]), np.array([3, 4]), None],
        })
        df_b = pd.DataFrame({
            "lists": [[1, 2], [5, 6], [7, 8], [9, 10]],
            "dicts": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}],
            "arrays": [np.array([1, 2]), np.array([5, 6]), np.array([7, 8]), np.array([9, 10])],
        })
        rows = column_drift_rows(df_a, df_b)
        self.assertEqual(len(rows), 3)
        row_map = {r["Column"]: r for r in rows}

        # Accurate uniqueness counts: [1, 2] duplicated in A -> 2 unique; 4 distinct in B -> 4 unique
        self.assertEqual(row_map["lists"]["Unique A"], 2)
        self.assertEqual(row_map["lists"]["Unique B"], 4)

        # Dict key order normalization: {"a": 1, "b": 2} == {"b": 2, "a": 1} -> 2 unique in A; 4 in B
        self.assertEqual(row_map["dicts"]["Unique A"], 2)
        self.assertEqual(row_map["dicts"]["Unique B"], 4)

        # NumPy arrays: [1, 2] duplicated in A -> 2 unique; 4 in B
        self.assertEqual(row_map["arrays"]["Unique A"], 2)
        self.assertEqual(row_map["arrays"]["Unique B"], 4)


class ConcurrencyAndManifestHardeningTests(unittest.TestCase):
    def test_concurrent_get_unique_filename_reservations(self):
        import concurrent.futures
        import tempfile
        from pathlib import Path
        from Utils.paths import get_unique_filename

        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            num_threads = 10
            names = []

            def worker():
                return get_unique_filename("data.csv", directory=target_dir)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads)]
                for f in concurrent.futures.as_completed(futures):
                    names.append(f.result())

            self.assertEqual(len(names), num_threads)
            self.assertEqual(len(set(names)), num_threads, "Concurrent reservations produced duplicate names!")

    def test_concurrent_manifest_record_conversion(self):
        import concurrent.futures
        import json
        import tempfile
        from pathlib import Path
        from Utils.paths import record_conversion

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / ".conversions.json"
            num_threads = 10

            def worker(i):
                record_conversion(
                    f"dataset_{i}.csv", f"source_{i}.txt", "raw text", model_name="test", manifest_path=manifest_path
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)

            self.assertTrue(manifest_path.exists())
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), num_threads, f"Expected {num_threads} entries in manifest, got {len(data)}")


class StorageCleanupAndRetentionTests(unittest.TestCase):
    def test_cleanup_storage_prunes_orphans_and_exceeding_files(self):
        import tempfile
        import time
        from pathlib import Path
        from Utils.paths import cleanup_storage

        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            # Create a tmp file
            tmp_file = target_dir / "test.csv.tmp.12345"
            tmp_file.write_text("orphan", encoding="utf-8")

            # Create 5 old files
            for i in range(5):
                f = target_dir / f"dataset_{i}.csv"
                f.write_text(f"val_{i}", encoding="utf-8")

            time.sleep(0.01)
            # Run cleanup with max_files=3
            pruned = cleanup_storage(directory=target_dir, max_files=3, max_age_days=30)
            self.assertTrue(
                any(Path(p).resolve() == tmp_file.resolve() for p in pruned)
                or str(tmp_file) in pruned,
                f"Expected {tmp_file} in pruned list: {pruned}",
            )
            # Check remaining regular files <= 3
            csvs = list(target_dir.glob("*.csv"))
            self.assertLessEqual(len(csvs), 3)

    def test_cleanup_storage_cross_platform_symlink_and_resolution(self):
        import tempfile
        from pathlib import Path
        from Utils.paths import cleanup_storage

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            resolved_dir = raw_dir.resolve()
            # Test cleanup works identically whether invoked with raw or resolved directory
            orphan = raw_dir / "orphan.csv.tmp.999"
            orphan.write_text("data", encoding="utf-8")
            pruned_raw = cleanup_storage(directory=raw_dir, include_temp_only=True)
            self.assertTrue(
                any(Path(p).resolve() == orphan.resolve() for p in pruned_raw),
                f"Raw directory invocation failed to match: {pruned_raw}",
            )

            # Test resolved directory invocation
            orphan2 = resolved_dir / "orphan2.csv.tmp.888"
            orphan2.write_text("data", encoding="utf-8")
            pruned_resolved = cleanup_storage(directory=resolved_dir, include_temp_only=True)
            self.assertTrue(
                any(Path(p).resolve() == orphan2.resolve() for p in pruned_resolved),
                f"Resolved directory invocation failed to match: {pruned_resolved}",
            )

    def test_delete_dataset_traversal_protection(self):
        from Utils.paths import delete_dataset
        with self.assertRaises(ValueError):
            delete_dataset("../../etc/passwd")


class PDFPIIRedactionTests(unittest.TestCase):
    def test_pdf_report_redacts_sensitive_modal_values(self):
        from Utils.PDF import generate_pdf_report
        # Create dataset with sensitive SSN column
        df = pd.DataFrame({
            "user_ssn": ["123-45-6789", "123-45-6789", "987-65-4321"],
            "normal_cat": ["A", "B", "A"],
            "normal_num": [10, 20, 30]
        })
        pdf_bytes = generate_pdf_report(df, dataset_name="pii_test.csv")
        self.assertGreater(len(pdf_bytes), 0)
        # Extract text from generated PDF and verify real SSN is redacted
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertNotIn("123-45-6789", full_text)
        self.assertIn("[REDACTED SENSITIVE]", full_text)


class MLInputValidationAndFeatureImportanceTests(unittest.TestCase):
    def test_predict_with_model_validates_numeric_features(self):
        from Utils.ML import train_and_evaluate_model, predict_with_model

        train_df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50],
            "salary": [50000, 60000, 70000, 80000, 90000, 100000],
            "target": [0, 0, 0, 1, 1, 1]
        })
        res = train_and_evaluate_model(
            train_df, "target", ["age", "salary"], "Logistic Regression", "Classification"
        )
        # Inference with invalid string data in numeric feature
        bad_df = pd.DataFrame({
            "age": ["twenty-five", "thirty"],
            "salary": [50000, 60000]
        })
        with self.assertRaisesRegex(ValueError, "was trained as a numeric feature"):
            predict_with_model(res, bad_df)

    def test_grouped_feature_importances_aggregates_encoded_levels(self):
        from Utils.ML import train_and_evaluate_model

        train_df = pd.DataFrame({
            "country": ["US", "UK", "DE", "FR", "US", "DE", "UK", "FR"],
            "age": [20, 30, 40, 50, 25, 35, 45, 55],
            "target": [100, 200, 300, 400, 150, 250, 350, 450]
        })
        res = train_and_evaluate_model(
            train_df, "target", ["country", "age"], "Linear Regression", "Regression"
        )
        self.assertIn("grouped_feature_importances", res)
        grouped = res["grouped_feature_importances"]
        self.assertIn("country", grouped)
        self.assertIn("age", grouped)
        # Grouped should not have one-hot encoded suffixes like country_US
        for key in grouped.keys():
            self.assertIn(key, ["country", "age"])


class AIConversionHardeningTests(unittest.TestCase):
    def test_ai_continuation_prompt_includes_preceding_context(self):
        from Utils.AIConvert import build_continuation_prompt

        prompt = build_continuation_prompt(
            "chunk 2 text", "test.txt", ["col1", "col2"], 2, 2,
            preceding_context="trailing data from chunk 1"
        )
        self.assertIn("<preceding_context_for_continuity>", prompt)
        self.assertIn("trailing data from chunk 1", prompt)


class DataHygieneReframingTests(unittest.TestCase):
    def test_hygiene_metrics_returns_heuristic_scores(self):
        from Utils.quality import hygiene_metrics, hygiene_index, quality_metrics

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        m = hygiene_metrics(df)
        self.assertEqual(m["index"], 100.0)
        self.assertEqual(hygiene_index(df), 100.0)
        self.assertEqual(m["index"], quality_metrics(df)["index"])


if __name__ == "__main__":
    unittest.main()

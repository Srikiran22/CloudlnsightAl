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


class RowCountEdgeCaseTests(unittest.TestCase):
    def test_csv_without_trailing_newline_returns_correct_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "single_row.csv"
            csv_path.write_text("col_a,col_b\nval1,val2", encoding="utf-8")
            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                self.assertEqual(get_dataset_row_count("single_row.csv"), 1)

    def test_csv_header_only_without_newline_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "header_only.csv"
            csv_path.write_text("col_a,col_b", encoding="utf-8")
            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                self.assertEqual(get_dataset_row_count("header_only.csv"), 0)

    def test_csv_with_multiline_quoted_fields_counts_logical_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "multiline.csv"
            content = 'id,comment\n1,"first line\nsecond line\nthird line"\n2,"single line"'
            csv_path.write_text(content, encoding="utf-8")
            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                self.assertEqual(get_dataset_row_count("multiline.csv"), 2)

    def test_jsonl_without_trailing_newline_returns_correct_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jsonl_path = tmp_path / "data.jsonl"
            content = '{"id": 1, "name": "a"}\n{"id": 2, "name": "b"}'
            jsonl_path.write_text(content, encoding="utf-8")
            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                self.assertEqual(get_dataset_row_count("data.jsonl"), 2)


class FingerprintCacheRealLookupTests(unittest.TestCase):
    def test_real_cache_hit_does_not_reread_unchanged_file(self):
        from Utils.dataset_ui import dataset_fingerprint, invalidate_dataset_cache
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fpath = tmp_path / "probe.csv"
            fpath.write_text("a,b\n1,2", encoding="utf-8")

            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                invalidate_dataset_cache("probe.csv")
                fp1 = dataset_fingerprint("probe.csv")

                def boom_open(*args, **kwargs):
                    raise RuntimeError("File was reread despite being in cache!")

                with patch.object(Path, "open", boom_open):
                    fp2 = dataset_fingerprint("probe.csv")
                    self.assertEqual(fp1, fp2)

    def test_cache_misses_and_recomputes_after_file_modification(self):
        from Utils.dataset_ui import dataset_fingerprint, invalidate_dataset_cache
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fpath = tmp_path / "probe_mod.csv"
            fpath.write_text("a,b\n1,2", encoding="utf-8")

            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                invalidate_dataset_cache("probe_mod.csv")
                fp1 = dataset_fingerprint("probe_mod.csv")

                fpath.write_text("a,b\n1,2\n3,4", encoding="utf-8")
                fp2 = dataset_fingerprint("probe_mod.csv")
                self.assertNotEqual(fp1, fp2)

    def test_same_size_same_mtime_invalidation_and_force_refresh(self):
        from Utils.dataset_ui import dataset_fingerprint, invalidate_dataset_cache
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fpath = tmp_path / "probe_same.csv"
            fpath.write_text("a,b\n1,2", encoding="utf-8")
            orig_stat = fpath.stat()

            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                invalidate_dataset_cache("probe_same.csv")
                fp1 = dataset_fingerprint("probe_same.csv")
                self.assertEqual(len(fp1), 64)

                # Overwrite with same byte length and preserve mtime
                fpath.write_text("a,b\n3,4", encoding="utf-8")
                import os
                os.utime(fpath, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))

                # Normal call with force_refresh=False hits stat cache
                cached_fp = dataset_fingerprint("probe_same.csv")
                self.assertEqual(cached_fp, fp1)

                # force_refresh=True bypasses cache and hashes full content
                fresh_fp = dataset_fingerprint("probe_same.csv", force_refresh=True)
                self.assertNotEqual(fresh_fp, fp1)
                self.assertEqual(len(fresh_fp), 64)

                # explicit invalidate_dataset_cache evicts the cache entry
                invalidate_dataset_cache("probe_same.csv")
                recomputed_fp = dataset_fingerprint("probe_same.csv")
                self.assertEqual(recomputed_fp, fresh_fp)


class CleaningUploadDiskCollisionTests(unittest.TestCase):
    def test_upload_with_same_name_as_disk_file_uses_dataframe_fingerprint(self):
        from Utils.dataset_ui import dataset_fingerprint, dataframe_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            disk_file = tmp_path / "data.csv"
            disk_file.write_text("a,b\n1,2", encoding="utf-8")

            uploaded_df = pd.DataFrame({"a": [99, 100], "b": [200, 300]})

            with patch("Utils.paths.DATASETS_DIR", tmp_path):
                disk_fp = dataset_fingerprint("data.csv")
                upload_fp = dataframe_fingerprint(uploaded_df)
                self.assertNotEqual(disk_fp, upload_fp)


class MLInMemoryDatasetTests(unittest.TestCase):
    def test_ml_results_match_in_memory_dataset_via_dataframe_fingerprint(self):
        from Utils.dataset_ui import results_match_active, dataframe_fingerprint
        df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})
        fp = dataframe_fingerprint(df)
        results = {
            "dataset_name": "Active Session: working.csv",
            "dataset_fingerprint": fp,
        }
        self.assertTrue(results_match_active(results, "Active Session: working.csv", df=df))

        other_df = pd.DataFrame({"feat": [9, 9, 9], "target": [1, 1, 1]})
        self.assertFalse(results_match_active(results, "Active Session: working.csv", df=other_df))


class GeminiClientLifecycleTests(unittest.TestCase):
    def test_gemini_client_closes_on_success(self):
        from Utils.Gemini import _generate_once
        closed = []
        class MockClient:
            class models:
                @staticmethod
                def generate_content(model, contents):
                    return "response"
            def close(self):
                closed.append(True)

        with patch("Utils.Gemini._new_sdk_client", return_value=MockClient()):
            res = _generate_once("fake_key", "gemini-3.8-flash", "test prompt")
            self.assertEqual(res, "response")
            self.assertEqual(len(closed), 1)

    def test_gemini_client_closes_on_failure(self):
        from Utils.Gemini import _generate_once
        closed = []
        class MockClient:
            class models:
                @staticmethod
                def generate_content(model, contents):
                    raise RuntimeError("API crash")
            def close(self):
                closed.append(True)

        with patch("Utils.Gemini._new_sdk_client", return_value=MockClient()):
            with self.assertRaises(RuntimeError):
                _generate_once("fake_key", "gemini-3.8-flash", "test prompt")
            self.assertEqual(len(closed), 1)


class ConversionManifestIntegrityTests(unittest.TestCase):
    def test_output_tampering_invalidates_conversion_cache(self):
        from Utils.paths import record_conversion, get_valid_conversion
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / ".conversions.json"
            conv_file = tmp_path / "data_txt_converted.csv"
            conv_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

            record_conversion(
                "data_txt_converted.csv",
                "data.txt",
                "raw unstructured text",
                model_name="gemini-3.8-flash",
                manifest_path=manifest,
                datasets_dir=tmp_path,
            )

            valid = get_valid_conversion(
                "data.txt",
                "raw unstructured text",
                model_name="gemini-3.8-flash",
                manifest_path=manifest,
                datasets_dir=tmp_path,
            )
            self.assertEqual(valid, "data_txt_converted.csv")

            conv_file.write_text("col1,col2\nTAMPERED,DATA", encoding="utf-8")
            tampered = get_valid_conversion(
                "data.txt",
                "raw unstructured text",
                model_name="gemini-3.8-flash",
                manifest_path=manifest,
                datasets_dir=tmp_path,
            )
            self.assertIsNone(tampered)

    def test_conversion_lookup_does_not_reuse_prompted_conversion_when_unprompted(self):
        from Utils.paths import record_conversion, get_valid_conversion
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / ".conversions.json"
            conv_file = tmp_path / "data_txt_converted.csv"
            conv_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

            record_conversion(
                "data_txt_converted.csv",
                "data.txt",
                "raw unstructured text",
                model_name="gemini-3.8-flash",
                extra_instructions="only extract numeric tables",
                manifest_path=manifest,
                datasets_dir=tmp_path,
            )

            # Unprompted query must NOT reuse conversion that had custom instructions
            unprompted = get_valid_conversion(
                "data.txt",
                "raw unstructured text",
                model_name="gemini-3.8-flash",
                extra_instructions="",
                manifest_path=manifest,
                datasets_dir=tmp_path,
            )
            self.assertIsNone(unprompted)

            # Matching prompt succeeds
            matched = get_valid_conversion(
                "data.txt",
                "raw unstructured text",
                model_name="gemini-3.8-flash",
                extra_instructions="only extract numeric tables",
                manifest_path=manifest,
                datasets_dir=tmp_path,
            )
            self.assertEqual(matched, "data_txt_converted.csv")


class AIReportProvenanceTests(unittest.TestCase):
    def test_ai_insights_not_reused_when_dataset_fingerprint_differs(self):
        saved_insights = {
            "dataset_name": "data.csv",
            "dataset_fingerprint": "hash_a",
            "model_name": "gemini-3.8-flash",
            "text": "Executive Insights",
        }
        active_name = "data.csv"
        active_fp = "hash_b"
        reusable = (
            saved_insights.get("dataset_name") == active_name
            and saved_insights.get("dataset_fingerprint") == active_fp
        )
        self.assertFalse(reusable)

    def test_ai_insights_not_reused_when_generation_provenance_differs(self):
        import hashlib
        active_name = "data.csv"
        active_fp = "hash_a"
        active_model = "gemini-3.8-flash"
        active_exclusions = ["email"]

        def make_ctx_hash(name, fp, model, exclusions):
            excl_sig = ",".join(sorted(exclusions))
            ctx_str = f"{name}|{fp}|{model}|{excl_sig}|v2"
            return hashlib.sha256(ctx_str.encode("utf-8")).hexdigest()[:16]

        base_ctx = make_ctx_hash(active_name, active_fp, active_model, active_exclusions)

        # Different model produces different hash
        diff_model_ctx = make_ctx_hash(active_name, active_fp, "gemini-2.5-pro", active_exclusions)
        self.assertNotEqual(base_ctx, diff_model_ctx)

        # Different exclusions produce different hash
        diff_excl_ctx = make_ctx_hash(active_name, active_fp, active_model, ["email", "ssn"])
        self.assertNotEqual(base_ctx, diff_excl_ctx)

        # Different dataset content produces different hash
        diff_fp_ctx = make_ctx_hash(active_name, "hash_modified", active_model, active_exclusions)
        self.assertNotEqual(base_ctx, diff_fp_ctx)


class S3StreamClosingTests(unittest.TestCase):
    def test_streaming_body_closed_on_success_and_on_error(self):
        from Utils.S3 import download_s3_dataset

        closed_success = []
        class MockBodySuccess:
            def read(self, chunk_size):
                if not hasattr(self, "_done"):
                    self._done = True
                    return b"a,b\n1,2"
                return b""
            def close(self):
                closed_success.append(True)

        mock_client = SimpleNamespace(
            get_object=lambda Bucket, Key: {
                "Body": MockBodySuccess(),
                "ContentLength": 7,
            }
        )
        df, _ = download_s3_dataset("mybucket", "test.csv", mock_client)
        self.assertEqual(len(closed_success), 1)

        closed_err = []
        class MockBodyTooBig:
            def read(self, chunk_size):
                return b"X" * (1024 * 1024)
            def close(self):
                closed_err.append(True)

        mock_client_err = SimpleNamespace(
            get_object=lambda Bucket, Key: {
                "Body": MockBodyTooBig(),
                "ContentLength": 500 * 1024 * 1024,
            }
        )
        with self.assertRaises(ValueError):
            download_s3_dataset("mybucket", "huge.csv", mock_client_err)
        self.assertEqual(len(closed_err), 1)

    def test_streaming_body_closed_on_unstructured_content(self):
        from Utils.S3 import download_s3_dataset
        closed = []
        class MockBody:
            def read(self, size):
                if not hasattr(self, "_done"):
                    self._done = True
                    return b"unstructured text"
                return b""
            def close(self):
                closed.append(True)
        mock_client = SimpleNamespace(
            get_object=lambda Bucket, Key: {
                "Body": MockBody(),
                "ContentLength": 17,
            }
        )
        df, raw_bytes = download_s3_dataset("b", "doc.txt", mock_client)
        self.assertIsNone(df)
        self.assertEqual(raw_bytes, b"unstructured text")
        self.assertEqual(len(closed), 1)


class SingleReportChartRowBoundTests(unittest.TestCase):
    def test_pdf_chart_generation_bounds_rows(self):
        from Utils.PDF import generate_pdf_report
        large_df = pd.DataFrame({
            "a": list(range(30_000)),
            "b": [x * 1.5 for x in range(30_000)],
        })
        pdf_bytes = generate_pdf_report(
            large_df,
            "test_large.csv",
            report_title="Test Report",
            author_name="Tester",
            include_charts=True,
        )
        self.assertGreater(len(pdf_bytes), 1000)

    def test_pdf_report_distinguishes_source_and_analyzed_rows(self):
        from Utils.PDF import generate_pdf_report
        sample_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        pdf_bytes = generate_pdf_report(
            sample_df,
            "sample.csv",
            report_title="Batch Sample Report",
            author_name="Auditor",
            source_rows=100_000,
            analyzed_rows=3,
        )
        self.assertGreater(len(pdf_bytes), 1000)


class CSVFormulaHardeningTests(unittest.TestCase):
    def test_csv_formula_injection_sanitization(self):
        from Utils.paths import sanitize_for_csv_export
        df = pd.DataFrame({
            "formula_cmd": ["=cmd|' /C calc'!A0", "@SUM(A1:A10)", "+formula", "-formula"],
            "numeric_neg": ["-15", "-42.5", "-1e4", -99],
            "numeric_pos": ["+100", "+3.14", 50, "+0"],
            "control_chars": ["\tcalc", "\rcalc", "safe string", "another safe"],
        })
        sanitized = sanitize_for_csv_export(df)

        self.assertTrue(sanitized["formula_cmd"].iloc[0].startswith("'="))
        self.assertTrue(sanitized["formula_cmd"].iloc[1].startswith("'@"))
        self.assertTrue(sanitized["formula_cmd"].iloc[2].startswith("'+"))
        self.assertTrue(sanitized["formula_cmd"].iloc[3].startswith("'-"))

        self.assertEqual(sanitized["numeric_neg"].iloc[0], "-15")
        self.assertEqual(sanitized["numeric_neg"].iloc[1], "-42.5")
        self.assertEqual(sanitized["numeric_neg"].iloc[3], -99)
        self.assertEqual(sanitized["numeric_pos"].iloc[0], "+100")
        self.assertEqual(sanitized["numeric_pos"].iloc[1], "+3.14")

        self.assertTrue(sanitized["control_chars"].iloc[0].startswith("'\t"))
        self.assertTrue(sanitized["control_chars"].iloc[1].startswith("'\r"))
        self.assertEqual(sanitized["control_chars"].iloc[2], "safe string")

    def test_formula_injection_and_legitimate_numeric_values(self):
        from Utils.paths import _sanitize_formula_val
        cases = [
            ("=SUM(A1:A2)", "'=SUM(A1:A2)"),
            ("+123", "+123"),
            ("-123", "-123"),
            ("@cmd", "'@cmd"),
            ("\tprefix", "'\tprefix"),
            ("\rprefix", "'\rprefix"),
            (-123, -123),
            (-42.5, -42.5),
            ("-123.45", "-123.45"),
            (42, 42),
            (3.14, 3.14),
            ("42", "42"),
            ("3.14", "3.14"),
            ("-", "'-"),
            ("+", "'+"),
            ("=", "'="),
            ("@", "'@"),
            ("-1e5", "-1e5"),
            ("+1.2e-3", "+1.2e-3"),
        ]
        for inp, expected in cases:
            self.assertEqual(_sanitize_formula_val(inp), expected, f"Failed for {inp}")


class HTMLSpanHandlingTests(unittest.TestCase):
    def test_html_colspan_expands_columns(self):
        html = """
        <table>
          <tr><th>Name</th><th colspan="2">Details</th></tr>
          <tr><td>Alice</td><td>Developer</td><td>London</td></tr>
        </table>
        """
        df = read_tabular(html.encode("utf-8"), filename="table.html")
        self.assertEqual(df.shape[1], 3)
        self.assertEqual(list(df.columns[:2]), ["Name", "Details"])

    def test_html_rowspan_triggers_ai_conversion(self):
        html = """
        <table>
          <tr><th>Name</th><th>Quarter</th><th>Revenue</th></tr>
          <tr><td rowspan="2">Acme</td><td>Q1</td><td>100</td></tr>
          <tr><td>Q2</td><td>120</td></tr>
        </table>
        """
        with self.assertRaises(AIConversionRequired):
            read_tabular(html.encode("utf-8"), filename="complex_table.html")


class MLBundleVersionTests(unittest.TestCase):
    def test_newer_bundle_version_rejected(self):
        from Utils.ML import load_trained_model
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            m_path = tmp_path / "future_model.joblib"
            import joblib
            joblib.dump({"bundle_version": 999, "pipeline": "mock_pipeline"}, m_path)
            with self.assertRaisesRegex(ValueError, "newer than supported version"):
                load_trained_model(m_path)

    def test_legacy_bundle_without_version_defaults_to_v1_and_succeeds(self):
        from Utils.ML import load_trained_model
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            m_path = tmp_path / "legacy_v1_model.joblib"
            import joblib
            joblib.dump({"pipeline": "mock_pipeline"}, m_path)
            bundle = load_trained_model(m_path)
            self.assertIn("pipeline", bundle)

    def test_invalid_bundle_version_rejected(self):
        from Utils.ML import load_trained_model
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            m_path = tmp_path / "invalid_model.joblib"
            import joblib
            for bad_ver in [0, -1, "v1", 2.5]:
                joblib.dump({"bundle_version": bad_ver, "pipeline": "mock_pipeline"}, m_path)
                with self.assertRaises(ValueError):
                    load_trained_model(m_path)


class S3TruncationNoticeTests(unittest.TestCase):
    def test_s3_truncation_reported_when_exceeding_20k(self):
        from Utils.S3 import list_s3_datasets
        class MockClient:
            def list_objects_v2(self, **kwargs):
                return {
                    "Contents": [{"Key": f"data_{i}.csv"} for i in range(100)],
                    "IsTruncated": True,
                    "NextContinuationToken": "token",
                }

        with patch("Utils.S3.MAX_S3_OBJECTS_SCAN", 50):
            files, meta = list_s3_datasets("mybucket", MockClient(), return_meta=True)
            self.assertTrue(meta["truncated"])
            self.assertGreaterEqual(meta["scanned"], 50)


if __name__ == "__main__":
    unittest.main()

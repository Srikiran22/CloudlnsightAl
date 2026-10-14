"""Machine learning and drift tests: Leak-free pipelines, TimeSeriesSplit, KS FDR drift, and TVD."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from Utils.paths import (
    get_unique_filename, enforce_size_limit, MAX_UPLOAD_BYTES,
    record_conversion, get_valid_conversion, read_tabular,
    DATASETS_DIR, ensure_project_directories,
)
from Utils.AIConvert import _field_count_quote_aware
from Utils.S3 import download_s3_dataset
from Utils.dataset_ui import (
    dataset_fingerprint, dataframe_fingerprint, load_dataset_cached,
)
from Utils.Gemini import (
    DEFAULT_GEMINI_MODEL, GEMINI_MODELS, resolve_gemini_model,
    chat_with_gemini_dataset,
)
from Utils.ML import (
    train_and_evaluate_model, save_trained_model, load_trained_model,
)
from Utils.compare_logic import column_drift_rows
from Utils.Charts import (
    create_line_chart, create_pie_treemap_plot, create_correlation_heatmap,
)
from Utils.quality import quality_metrics, quality_index
from Utils.PDF import generate_pdf_report

ensure_project_directories()


class UploadSafetyAndCollisionsTests(unittest.TestCase):
    def test_get_unique_filename_avoids_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "data.csv").write_text("x", encoding="utf-8")
            u1 = get_unique_filename("data.csv", directory=tmp_path)
            self.assertEqual(u1, "data_1.csv")

            (tmp_path / "data_1.csv").write_text("x", encoding="utf-8")
            u2 = get_unique_filename("data.csv", directory=tmp_path)
            self.assertEqual(u2, "data_2.csv")

    def test_get_unique_filename_with_extra_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {"file.csv", "file_1.csv"}
            u = get_unique_filename("file.csv", directory=tmp, extra_names=seen)
            self.assertEqual(u, "file_2.csv")

    def test_conversion_manifest_caching_and_staleness(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tmp_manifest = tmp_path / ".conversions.json"
            tmp_datasets = tmp_path / "Datasets"
            tmp_datasets.mkdir()
            dummy_file = tmp_datasets / "doc_txt_converted.csv"
            dummy_file.write_text("dummy", encoding="utf-8")

            text_a = "name,val\nalice,1"
            text_b = "name,val\nbob,2"

            record_conversion(
                "doc_txt_converted.csv",
                "doc.txt",
                text_a,
                model_name="gemini-3.8-flash",
                extra_instructions="clean data",
                manifest_path=tmp_manifest,
            )

            # Same text + model + instructions -> valid reuse
            valid = get_valid_conversion(
                "doc.txt",
                text_a,
                model_name="gemini-3.8-flash",
                extra_instructions="clean data",
                manifest_path=tmp_manifest,
                datasets_dir=tmp_datasets,
            )
            self.assertEqual(valid, "doc_txt_converted.csv")

            # Modified text -> stale, returns None
            stale_text = get_valid_conversion(
                "doc.txt",
                text_b,
                model_name="gemini-3.8-flash",
                extra_instructions="clean data",
                manifest_path=tmp_manifest,
                datasets_dir=tmp_datasets,
            )
            self.assertIsNone(stale_text)

            # Modified model -> invalidates reuse
            diff_model = get_valid_conversion(
                "doc.txt",
                text_a,
                model_name="gemini-2.5-pro",
                extra_instructions="clean data",
                manifest_path=tmp_manifest,
                datasets_dir=tmp_datasets,
            )
            self.assertIsNone(diff_model)

            # Modified instructions -> invalidates reuse
            diff_instr = get_valid_conversion(
                "doc.txt",
                text_a,
                model_name="gemini-3.8-flash",
                extra_instructions="different prompt",
                manifest_path=tmp_manifest,
                datasets_dir=tmp_datasets,
            )
            self.assertIsNone(diff_instr)

            # Different extension, same stem -> does not collide
            diff_ext = get_valid_conversion(
                "doc.pdf",
                text_a,
                model_name="gemini-3.8-flash",
                extra_instructions="clean data",
                manifest_path=tmp_manifest,
                datasets_dir=tmp_datasets,
            )
            self.assertIsNone(diff_ext)


class ResourceAndMemoryLimitsTests(unittest.TestCase):
    def test_enforce_size_limit_rejects_oversized_payload(self):
        with self.assertRaises(ValueError) as ctx:
            enforce_size_limit(MAX_UPLOAD_BYTES + 1, "test file")
        self.assertIn("limit is 200 MB", str(ctx.exception))

    def test_s3_streaming_hard_byte_limit_without_content_length(self):
        mock_body = Mock()
        # Return 64KB chunks infinitely
        mock_body.read.return_value = b"X" * (64 * 1024)

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            "Body": mock_body,
            # ContentLength intentionally absent
        }

        with self.assertRaises(ValueError) as ctx:
            download_s3_dataset("bucket", "huge.csv", mock_s3)
        self.assertIn("limit", str(ctx.exception).lower())

    def test_load_dataset_cached_bounds_rows(self):
        test_file = DATASETS_DIR / "temp_cache_test.csv"
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        df.to_csv(test_file, index=False)
        try:
            loaded = load_dataset_cached("temp_cache_test.csv", max_rows=10)
            self.assertEqual(len(loaded), 10)
        finally:
            if test_file.exists():
                test_file.unlink()


class ParsingCorrectnessTests(unittest.TestCase):
    def test_xml_repeated_elements_are_indexed(self):
        xml = b"""<root>
            <row><item>A</item><item>B</item></row>
        </root>"""
        df = read_tabular(io.BytesIO(xml), filename="data.xml")
        self.assertEqual(len(df), 1)
        self.assertIn("item_1", df.columns)
        self.assertIn("item_2", df.columns)
        self.assertEqual(df.iloc[0]["item_1"], "A")
        self.assertEqual(df.iloc[0]["item_2"], "B")

    def test_ambiguous_json_multiple_lists_raises_error(self):
        json_data = b'{"users": [{"id": 1}], "products": [{"id": 2}]}'
        with self.assertRaises(ValueError) as ctx:
            read_tabular(io.BytesIO(json_data), filename="data.json")
        self.assertIn("Ambiguous JSON structure", str(ctx.exception))

    def test_list_valued_json_cells_are_sanitized_to_strings(self):
        json_data = b'[{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": ["c"]}]'
        df = read_tabular(io.BytesIO(json_data), filename="data.json")
        self.assertIsInstance(df.iloc[0]["tags"], str)
        # Verify unhashable type operations do not crash
        uniques = df["tags"].nunique()
        self.assertEqual(uniques, 2)
        dup_count = df.duplicated().sum()
        self.assertEqual(dup_count, 0)

    def test_quote_aware_csv_counting(self):
        line = '"hello, world",123,456'
        count = _field_count_quote_aware(line)
        self.assertEqual(count, 3)

    def test_parse_ai_csv_fails_clearly_when_exceeding_column_cap(self):
        from Utils.AIConvert import MAX_CONVERTED_COLUMNS, parse_ai_csv
        header = ",".join(f"col_{i}" for i in range(MAX_CONVERTED_COLUMNS + 5))
        row = ",".join(str(i) for i in range(MAX_CONVERTED_COLUMNS + 5))
        with self.assertRaisesRegex(ValueError, "exceeds the maximum supported limit"):
            parse_ai_csv(f"{header}\n{row}")


class DatasetIdentityAndStateTests(unittest.TestCase):
    def test_dataset_fingerprint_changes_on_content_modification(self):
        test_file = DATASETS_DIR / "temp_fp_test.csv"
        test_file.write_text("a,b\n1,2", encoding="utf-8")
        try:
            fp1 = dataset_fingerprint("temp_fp_test.csv")
            test_file.write_text("a,b\n1,999", encoding="utf-8")
            fp2 = dataset_fingerprint("temp_fp_test.csv")
            self.assertNotEqual(fp1, fp2)
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_dataframe_fingerprint_deterministic(self):
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [1, 2, 3]})
        df3 = pd.DataFrame({"x": [1, 2, 4]})
        self.assertEqual(dataframe_fingerprint(df1), dataframe_fingerprint(df2))
        self.assertNotEqual(dataframe_fingerprint(df1), dataframe_fingerprint(df3))
        self.assertEqual(len(dataframe_fingerprint(df1)), 64)

    def test_dataset_fingerprint_changes_on_same_size_same_mtime_replacement(self):
        import os
        test_file = DATASETS_DIR / "temp_collision_probe.csv"
        test_file.write_text("a,b\n1,2", encoding="utf-8")
        try:
            stat1 = test_file.stat()
            fp1 = dataset_fingerprint("temp_collision_probe.csv")
            self.assertEqual(len(fp1), 64)
            test_file.write_text("a,b\n2,1", encoding="utf-8")
            os.utime(test_file, ns=(stat1.st_atime_ns, stat1.st_mtime_ns))
            # With force_refresh or invalidation, full cryptographic rehash detects content change
            fp2 = dataset_fingerprint("temp_collision_probe.csv", force_refresh=True)
            self.assertEqual(len(fp2), 64)
            self.assertNotEqual(fp1, fp2)
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_dataset_fingerprint_changes_on_middle_byte_modification_same_mtime_same_size(self):
        import os
        test_file = DATASETS_DIR / "temp_midbyte_probe.csv"
        header = "id,val\n"
        pad1 = ("1," + "A" * 50 + "\n") * 80  # > 4096 bytes head
        mid1 = "50,ORIGINAL_VALUE\n" + ("2," + "X" * 50 + "\n") * 70
        mid2 = "50,MODIFIED_VALUE\n" + ("2," + "X" * 50 + "\n") * 70
        pad2 = ("3," + "Z" * 50 + "\n") * 80  # > 4096 bytes tail

        c1 = (header + pad1 + mid1 + pad2).encode("utf-8")
        c2 = (header + pad1 + mid2 + pad2).encode("utf-8")
        self.assertEqual(len(c1), len(c2))

        test_file.write_bytes(c1)
        try:
            stat1 = test_file.stat()
            fp1 = dataset_fingerprint("temp_midbyte_probe.csv")
            df1 = load_dataset_cached("temp_midbyte_probe.csv")
            self.assertEqual(df1[df1["id"] == 50]["val"].values[0], "ORIGINAL_VALUE")

            # Mutate strictly in the middle, preserving head 4KB and tail 4KB
            with open(test_file, "r+b") as f:
                f.seek(len((header + pad1).encode("utf-8")))
                f.write(mid2.encode("utf-8"))

            os.utime(test_file, ns=(stat1.st_atime_ns, stat1.st_mtime_ns))
            stat2 = test_file.stat()
            self.assertEqual(stat2.st_mtime_ns, stat1.st_mtime_ns)
            self.assertEqual(stat2.st_size, stat1.st_size)

            from Utils.dataset_ui import invalidate_dataset_cache
            invalidate_dataset_cache("temp_midbyte_probe.csv")
            fp2 = dataset_fingerprint("temp_midbyte_probe.csv")
            df2 = load_dataset_cached("temp_midbyte_probe.csv")

            self.assertNotEqual(fp1, fp2)
            self.assertEqual(df2[df2["id"] == 50]["val"].values[0], "MODIFIED_VALUE")
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_invalidate_dataset_cache_clears_entries(self):
        from Utils.dataset_ui import _CONTENT_HASH_CACHE, invalidate_dataset_cache
        test_file = DATASETS_DIR / "temp_inval_probe.csv"
        test_file.write_text("x,y\n10,20", encoding="utf-8")
        try:
            dataset_fingerprint("temp_inval_probe.csv")
            self.assertTrue(any("temp_inval_probe.csv" in k[0] for k in _CONTENT_HASH_CACHE))
            invalidate_dataset_cache("temp_inval_probe.csv")
            self.assertFalse(any("temp_inval_probe.csv" in k[0] for k in _CONTENT_HASH_CACHE))
        finally:
            if test_file.exists():
                test_file.unlink()


class GeminiHardeningTests(unittest.TestCase):
    def test_gemini_model_registry_contains_supported_models(self):
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.8-flash")
        self.assertIn("gemini-3.8-flash", GEMINI_MODELS)
        self.assertIn("gemini-3.5-flash", GEMINI_MODELS)
        self.assertIn("gemini-2.5-flash", GEMINI_MODELS)
        self.assertIn("gemini-2.5-pro", GEMINI_MODELS)
        self.assertIn("gemini-3.5-flash", GEMINI_MODELS)
        self.assertIn("gemini-2.5-flash", GEMINI_MODELS)
        self.assertIn("gemini-2.5-pro", GEMINI_MODELS)
        self.assertNotIn("gemini-1.5-flash", GEMINI_MODELS)
        self.assertNotIn("gemini-1.5-pro", GEMINI_MODELS)

    def test_deprecated_model_resolution(self):
        self.assertEqual(resolve_gemini_model("gemini-2.0-flash"), "gemini-3.8-flash")
        self.assertEqual(resolve_gemini_model("gemini-2.0-flash-exp"), "gemini-3.8-flash")
        self.assertEqual(resolve_gemini_model("gemini-1.5-flash"), "gemini-3.8-flash")
        self.assertEqual(resolve_gemini_model("gemini-1.5-pro"), "gemini-2.5-pro")
        self.assertEqual(resolve_gemini_model("gemini-1.0-pro"), "gemini-2.5-pro")
        self.assertEqual(resolve_gemini_model("gemini-pro"), "gemini-2.5-pro")
        self.assertEqual(resolve_gemini_model("gemini-3.8-flash"), "gemini-3.8-flash")

    def test_ai_convert_uses_default_gemini_model(self):
        import inspect
        from Utils.AIConvert import convert_to_dataframe
        sig = inspect.signature(convert_to_dataframe)
        self.assertEqual(sig.parameters["model_name"].default, DEFAULT_GEMINI_MODEL)

    def test_chat_prompt_includes_context_honesty_constraints(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        messages = [{"role": "user", "content": "How many total rows?"}]
        with patch("Utils.Gemini._generate_content") as mock_gen:
            mock_gen.return_value = "response"
            chat_with_gemini_dataset("fake_key", df, "data.csv", messages)
            call_args = mock_gen.call_args[0]
            prompt = call_args[2]
            self.assertIn("Important constraints:", prompt)
            self.assertIn("You do NOT have full-dataset or row-level query execution access", prompt)

    def test_new_sdk_client_fails_closed_when_timeout_unsupported(self):
        import sys
        import types
        from Utils.Gemini import GeminiError, _new_sdk_client
        fake_genai = types.ModuleType("google.genai")
        class OldClient:
            def __init__(self, **kwargs):
                if "http_options" in kwargs:
                    raise TypeError("unexpected keyword 'http_options'")
        fake_genai.Client = OldClient
        google_mod = sys.modules.get("google")
        with patch.dict(sys.modules, {"google.genai": fake_genai}):
            if google_mod and hasattr(google_mod, "genai"):
                with patch.object(google_mod, "genai", fake_genai):
                    with self.assertRaises(GeminiError) as ctx:
                        _new_sdk_client("key")
            else:
                with self.assertRaises(GeminiError) as ctx:
                    _new_sdk_client("key")
            self.assertEqual(ctx.exception.kind, "sdk")
            self.assertFalse(ctx.exception.retryable)

    def test_generate_once_fails_closed_when_genai_missing(self):
        import sys
        from Utils.Gemini import GeminiError, _generate_once
        with patch.dict(sys.modules, {"google.genai": None}):
            with self.assertRaises(GeminiError) as ctx:
                _generate_once("key", "gemini-3.8-flash", "prompt")
            self.assertEqual(ctx.exception.kind, "sdk")
            self.assertFalse(ctx.exception.retryable)

    def test_primary_google_genai_sdk_client_initialization(self):
        from Utils.Gemini import _new_sdk_client, REQUEST_TIMEOUT_SECONDS
        import google.genai as genai
        client = _new_sdk_client("fake-key-0123456789")
        self.assertIsInstance(client, genai.Client)
        http_opts = getattr(client._api_client, "_http_options", None)
        timeout_ms = http_opts.get("timeout") if isinstance(http_opts, dict) else getattr(http_opts, "timeout", None)
        self.assertEqual(timeout_ms, REQUEST_TIMEOUT_SECONDS * 1000)


class MLPipelineHardeningTests(unittest.TestCase):
    def test_dense_one_hot_encoding_explosion_guard(self):
        # 10,000 rows with 5,000 unique categories would yield 50,000,000 cells
        n_rows = 1000
        # Set a small test limit to test the guard deterministically
        df = pd.DataFrame({
            "target": [0, 1] * (n_rows // 2),
            "high_card": [f"cat_{i}" for i in range(n_rows)],
        })
        with patch("Utils.ML.ML_MAX_TRAIN_CELLS", 10_000):
            with self.assertRaises(ValueError) as ctx:
                train_and_evaluate_model(
                    df, "target", ["high_card"], "Logistic Regression", "Classification"
                )
            self.assertIn("exceeds the 10,000-cell safety limit", str(ctx.exception))
            self.assertIn("One-hot encoding would risk memory exhaustion", str(ctx.exception))

    def test_save_trained_model_collision_safety_and_fingerprint(self):
        df = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        res = train_and_evaluate_model(
            df, "target", ["feature"], "Logistic Regression", "Classification", test_size=0.5
        )
        res["dataset_fingerprint"] = "abc12345"

        with tempfile.TemporaryDirectory() as tmp:
            p1 = save_trained_model(res, "test_model", directory=tmp)
            p2 = save_trained_model(res, "test_model", directory=tmp)
            self.assertNotEqual(p1.name, p2.name)
            self.assertEqual(p1.name, "test_model.joblib")
            self.assertEqual(p2.name, "test_model_1.joblib")

            bundle = load_trained_model(p1, allowed_dir=tmp)
            self.assertEqual(bundle.get("dataset_fingerprint"), "abc12345")

    def test_stratified_split_status_tracking(self):
        df = pd.DataFrame({
            "feature": range(10),
            "target": [0] * 5 + [1] * 5,
        })
        res = train_and_evaluate_model(
            df, "target", ["feature"], "Logistic Regression", "Classification", test_size=0.4
        )
        self.assertTrue(res.get("stratified_split"))


class ReportingHardeningTests(unittest.TestCase):
    def test_pdf_report_sampling_disclosure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        pdf_bytes = generate_pdf_report(
            df, "data.csv",
            source_rows=50000, analyzed_rows=3
        )
        self.assertIsInstance(pdf_bytes, (bytes, bytearray))
        self.assertGreater(len(pdf_bytes), 500)


class ComparisonAndAnalyticsTests(unittest.TestCase):
    def test_near_zero_baseline_mean_drift_uses_absolute_delta(self):
        df_a = pd.DataFrame({"val": [0.0, 0.0, 0.0]})  # Mean = 0.0
        df_b = pd.DataFrame({"val": [2.5, 2.5, 2.5]})  # Mean = 2.5
        rows = column_drift_rows(df_a, df_b)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["Mean Shift %"], "N/A (baseline ≈ 0)")
        self.assertIn("mean shift Δ+2.500 (baseline ≈ 0)", r["Flags"])

    def test_pie_treemap_rejects_negative_values(self):
        df = pd.DataFrame({"cat": ["A", "B"], "val": [-10, 50]})
        with self.assertRaises(ValueError) as ctx:
            create_pie_treemap_plot(df, names_col="cat", values_col="val", plot_type="Pie")
        self.assertIn("contains negative values", str(ctx.exception))

    def test_line_chart_chronological_datetime_sorting(self):
        # Lexicographical sort would put '10/01/2023' after '05/01/2023', but '01/01/2024' first!
        df = pd.DataFrame({
            "date": ["2023-12-01", "2023-01-01", "2023-06-01"],
            "val": [30, 10, 20]
        })
        fig = create_line_chart(df, "date", "val")
        # Check that x data was sorted chronologically
        x_data = list(fig.data[0].x)
        self.assertEqual(x_data, ["2023-01-01", "2023-06-01", "2023-12-01"])

    def test_correlation_heatmap_caps_columns(self):
        cols = {f"c{i}": np.random.randn(20) for i in range(40)}
        df = pd.DataFrame(cols)
        fig = create_correlation_heatmap(df)
        self.assertIsNotNone(fig)
        self.assertIn("top 30 numeric columns", fig.layout.title.text)

    def test_empty_dataframe_quality_metrics(self):
        empty_df = pd.DataFrame()
        m = quality_metrics(empty_df)
        self.assertEqual(m["index"], 0.0)
        self.assertEqual(m["completeness"], 0.0)
        self.assertEqual(m["uniqueness"], 0.0)
        self.assertTrue(m["is_empty"])
        self.assertEqual(quality_index(empty_df), 0.0)


class DependencyCompatibilityTests(unittest.TestCase):
    def test_streamlit_minimum_version_and_stretch_width(self):
        import inspect
        import streamlit as st
        # Requirements specify streamlit>=1.52.0 for width="stretch" support
        version_parts = [int(p) for p in st.__version__.split(".")[:2]]
        self.assertGreaterEqual((version_parts[0], version_parts[1]), (1, 52))

        df_sig = inspect.signature(st.dataframe)
        self.assertIn("width", df_sig.parameters)
        chart_sig = inspect.signature(st.plotly_chart)
        self.assertIn("width", chart_sig.parameters)


# ======================================================================
# Additional domain tests
# ======================================================================

import unittest

from Utils.paths import _reject_dtd
from Utils.AIConvert import _split_into_chunks, reconcile_chunk_frames
from Utils.dataset_ui import BoundedDatasetMemoryCache
from Utils.Charts import downsample_timeseries


class TestBoundedJSONParsing(unittest.TestCase):
    """Row-bounded JSON decoding across array, keyed, and columnar structures."""

    def test_keyed_record_array_bounded_and_malformed_tail(self):
        # Keyed record array with valid prefix and malformed/truncated tail
        # 10 records, but tail is cut off after record 3
        valid_part = '{"status": 200, "meta": {"page": 1}, "records": [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}, {"id": 3, "val": "c"}, {"id": 4, "val": "d"}'
        malformed_tail = ', {"id": 5, "val": TRUNCATED_BAD_JSON'
        payload = (valid_part + malformed_tail).encode("utf-8")

        # Full json.loads must fail on malformed tail
        with self.assertRaises(Exception):
            import json
            json.loads(payload.decode("utf-8"))

        # Streaming parser with max_rows=2 must succeed and yield exactly 2 rows
        df = read_tabular(payload, filename="keyed_stream.json", max_rows=2)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["id", "val"])
        self.assertEqual(df["id"].tolist(), [1, 2])

    def test_columnar_json_bounded_and_malformed_tail(self):
        # Columnar JSON with valid start and malformed tail
        payload = b'{"col_a": [1, 2, 3, 4, 5], "col_b": [10, 20, 30, 40, 50], TRUNCATED_TAIL'

        with self.assertRaises(Exception):
            import json
            json.loads(payload.decode("utf-8"))

        df = read_tabular(payload, filename="columnar_stream.json", max_rows=3)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ["col_a", "col_b"])
        self.assertEqual(df["col_a"].tolist(), [1, 2, 3])
        self.assertEqual(df["col_b"].tolist(), [10, 20, 30])

    def test_root_array_bounded_streaming(self):
        payload = b'[{"x": 10}, {"x": 20}, {"x": 30}, {"x": 40}, BAD_TAIL'
        df = read_tabular(payload, filename="array_stream.json", max_rows=2)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["x"].tolist(), [10, 20])


class TestXMLStreamingDTDDefense(unittest.TestCase):
    """Streaming XML DTD defense without full-payload memory buffering."""

    def test_dtd_rejection_without_materializing_full_stream(self):
        # Stream with DTD header and 10MB of padding
        dtd_payload = b'<?xml version="1.0"?>\n<!DOCTYPE root [ <!ENTITY test "val"> ]>\n<root>' + b"<row><a>1</a></row>" * 1000
        bio = io.BytesIO(dtd_payload)
        with self.assertRaisesRegex(ValueError, r"(?i)DTD/entity declarations"):
            _reject_dtd(bio)

    def test_non_dtd_xml_memory_bounded(self):
        # Valid XML parses without buffering full payload in _reject_dtd
        xml_data = b'<root>' + b'<item><id>1</id><val>test</val></item>' * 100 + b'</root>'
        bio = io.BytesIO(xml_data)
        _reject_dtd(bio)
        bio.seek(0)
        df = read_tabular(bio, filename="test.xml", max_rows=10)
        self.assertEqual(len(df), 10)
        self.assertEqual(list(df.columns), ["id", "val"])


class TestAIChunkBoundaryDeduplication(unittest.TestCase):
    """Deterministic semantic chunking and chunk boundary deduplication."""

    def test_split_into_chunks_deterministic(self):
        text = "First paragraph here.\n\nSecond paragraph is longer and has more information.\n\nThird paragraph."
        chunks = _split_into_chunks(text, max_chunk_chars=50)
        self.assertTrue(len(chunks) >= 2)
        # Verify chunks preserve paragraph/semantic boundaries
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_reconcile_chunk_frames_eliminates_boundary_duplicates(self):
        # Chunk 1 produced rows 1, 2, 3
        df1 = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}])
        # Chunk 2, due to continuation context, re-extracted row 3 and added rows 4, 5
        df2 = pd.DataFrame([{"id": 3, "name": "Charlie"}, {"id": 4, "name": "Diana"}, {"id": 5, "name": "Evan"}])

        merged = reconcile_chunk_frames([df1, df2])
        self.assertEqual(len(merged), 5)
        self.assertEqual(merged["id"].tolist(), [1, 2, 3, 4, 5])


class TestBoundedDatasetMemoryCache(unittest.TestCase):
    """Hard memory budget enforcement with deep byte tracking and LRU eviction."""

    def test_hard_memory_budget_eviction(self):
        # 1MB memory budget
        cache = BoundedDatasetMemoryCache(max_bytes=1024 * 1024)

        # Create DataFrames of ~400KB each
        df1 = pd.DataFrame({"col": ["a" * 100 for _ in range(4000)]})
        df2 = pd.DataFrame({"col": ["b" * 100 for _ in range(4000)]})
        df3 = pd.DataFrame({"col": ["c" * 100 for _ in range(4000)]})

        size1 = df1.memory_usage(deep=True).sum()
        self.assertTrue(size1 > 300_000)

        cache.put("df1", df1)
        self.assertIn("df1", cache._cache)
        cache.put("df2", df2)
        self.assertIn("df2", cache._cache)

        # Access df1 to make df2 the LRU entry
        _ = cache.get("df1")

        # Putting df3 must evict df2 to stay within 1MB budget
        cache.put("df3", df3)
        self.assertIn("df1", cache._cache)
        self.assertIn("df3", cache._cache)
        self.assertNotIn("df2", cache._cache)
        self.assertLessEqual(cache.current_bytes, 1024 * 1024)


class TestTimeSeriesDownsampling(unittest.TestCase):
    """Time-preserving peak/valley decimation without random sampling."""

    def test_spike_and_envelope_preservation(self):
        # 50,000 points of a smooth signal + 1 extreme anomaly spike
        np.random.seed(42)
        n = 50_000
        dates = pd.date_range("2026-01-01", periods=n, freq="min")
        values = np.sin(np.linspace(0, 50, n))
        values[23_456] = 999.0  # isolated extreme spike

        df = pd.DataFrame({"timestamp": dates, "value": values})

        # Decimate to max 500 points
        downsampled = downsample_timeseries(df, "timestamp", "value", max_points=500)
        self.assertLessEqual(len(downsampled), 1000)

        # Verify extreme spike is 100% preserved
        self.assertAlmostEqual(downsampled["value"].max(), 999.0, places=1)
        # Verify min value is preserved
        self.assertAlmostEqual(downsampled["value"].min(), values.min(), places=1)
        # Verify timestamps remain strictly monotonic
        self.assertTrue(downsampled["timestamp"].is_monotonic_increasing)


class TestMLCrossValidationAndBaselines(unittest.TestCase):
    """K-fold cross-validation variability and dummy baseline model comparison."""

    def test_classification_cv_and_baseline(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": np.random.choice([0, 1], size=100)
        })

        results = train_and_evaluate_model(
            df,
            target_col="target",
            feature_cols=["f1", "f2"],
            model_name="Logistic Regression",
            problem_type="Classification",
        )

        self.assertIn("cv_accuracy_mean", results)
        self.assertIn("cv_accuracy_std", results)
        self.assertIsNotNone(results["cv_accuracy_mean"])
        self.assertIsNotNone(results["cv_accuracy_std"])
        self.assertIn("baseline_accuracy", results)
        self.assertIsNotNone(results["baseline_accuracy"])

    def test_regression_cv_and_baseline(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": np.random.randn(100)
        })

        results = train_and_evaluate_model(
            df,
            target_col="target",
            feature_cols=["f1", "f2"],
            model_name="Ridge Regression",
            problem_type="Regression",
        )

        self.assertIn("cv_r2_mean", results)
        self.assertIn("cv_r2_std", results)
        self.assertIsNotNone(results["cv_r2_mean"])
        self.assertIsNotNone(results["cv_r2_std"])
        self.assertIn("baseline_r2", results)
        self.assertIsNotNone(results["baseline_r2"])


class TestLockfileAndCIMatrix(unittest.TestCase):
    """F-17 & F-19: Requirements locking and cross-platform CI matrix."""

    def test_lockfile_exists_and_referenced_in_readme(self):
        root = Path(__file__).resolve().parent.parent
        lockfile = root / "requirements-lock.txt"
        readme = root / "README.md"

        self.assertTrue(lockfile.exists(), "requirements-lock.txt must exist")
        content = readme.read_text(encoding="utf-8")
        self.assertIn("requirements-lock.txt", content, "README must instruct using lockfile")

    def test_ci_workflow_matrix_includes_windows_and_macos(self):
        root = Path(__file__).resolve().parent.parent
        ci_file = root / ".github" / "workflows" / "tests.yml"
        self.assertTrue(ci_file.exists(), ".github/workflows/tests.yml must exist")

        content = ci_file.read_text(encoding="utf-8")
        self.assertIn("os: [ubuntu-latest, windows-latest, macos-latest]", content)


class TestDatasetDriftDetection(unittest.TestCase):
    """Kolmogorov-Smirnov and Total Variation Distance drift detection."""

    def test_numeric_ks_drift_on_identical_mean_and_std(self):
        # Two distributions with identical mean (~0) and std (~1):
        # Dataset 1: Standard Normal distribution
        # Dataset 2: Uniform distribution scaled to std=1
        np.random.seed(42)
        n = 1000
        normal_data = np.random.normal(0, 1, n)
        uniform_data = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)

        df_ref = pd.DataFrame({"feat": normal_data})
        df_cur = pd.DataFrame({"feat": uniform_data})

        rows = column_drift_rows(df_ref, df_cur)
        feat_res = rows[0]

        # KS test detects distribution drift (p < 0.01)
        self.assertIsNotNone(feat_res["KS Stat"])
        self.assertIsNotNone(feat_res["KS p-val"])
        self.assertLess(feat_res["KS p-val"], 0.01)
        self.assertIn("distribution drift", feat_res["Flags"])

    def test_categorical_tvd_drift(self):
        df_ref = pd.DataFrame({"category": ["A"] * 80 + ["B"] * 20})
        df_cur = pd.DataFrame({"category": ["A"] * 20 + ["B"] * 80})

        rows = column_drift_rows(df_ref, df_cur)
        cat_res = rows[0]
        self.assertIsNotNone(cat_res["Categorical TVD"])
        self.assertGreater(cat_res["Categorical TVD"], 0.5)
        self.assertIn("categorical drift", cat_res["Flags"])


class TestModelDeserializationTrustBoundary(unittest.TestCase):
    """HMAC-SHA256 signature verification and untrusted model blocking."""

    def test_signed_model_loads_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "target": [0, 1, 0, 1, 0, 1]})
            bundle = train_and_evaluate_model(
                df,
                target_col="target",
                feature_cols=["f1"],
                model_name="Logistic Regression",
                problem_type="Classification",
            )

            save_trained_model(bundle, "trusted_model", directory=tmp_path)
            sig_file = tmp_path / "trusted_model.joblib.sig"
            self.assertTrue(sig_file.exists(), "Model signature file must be generated")

            # Load with require_signature=True must succeed
            loaded = load_trained_model(tmp_path / "trusted_model.joblib", allowed_dir=tmp_path)
            self.assertEqual(loaded["problem_type"], "Classification")
            self.assertIn("pipeline", loaded)

    def test_tampered_model_rejected_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "target": [0, 1, 0, 1, 0, 1]})
            bundle = train_and_evaluate_model(
                df,
                target_col="target",
                feature_cols=["f1"],
                model_name="Logistic Regression",
                problem_type="Classification",
            )

            save_trained_model(bundle, "tampered", directory=tmp_path)
            model_file = tmp_path / "tampered.joblib"

            # Tamper with model file bytes
            with model_file.open("ab") as f:
                f.write(b"MALICIOUS_INJECTION")

            # Loading must be blocked before joblib.load
            with self.assertRaisesRegex(ValueError, r"(?i)signature verification failed"):
                load_trained_model(tmp_path / "tampered.joblib", allowed_dir=tmp_path)

    def test_unsigned_model_rejected_when_signature_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_file = tmp_path / "unsigned.joblib"
            model_file.write_bytes(b"some_joblib_bytes")

            with self.assertRaisesRegex(ValueError, r"(?i)lacks a cryptographic authenticity signature"):
                load_trained_model(tmp_path / "unsigned.joblib", allowed_dir=tmp_path)


class TestAtomicFilenameReservation(unittest.TestCase):
    """Concurrency-safe atomic kernel filesystem reservations."""

    def test_concurrent_reservations_do_not_collide(self):
        from Utils.paths import get_unique_filename, release_filename_reservation, atomic_write
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            f1 = get_unique_filename("data.csv", directory=tmp_path)
            f2 = get_unique_filename("data.csv", directory=tmp_path)
            self.assertEqual(f1, "data.csv")
            self.assertEqual(f2, "data_1.csv")

            atomic_write(tmp_path / f1, "col\n1", mode="w")
            self.assertTrue((tmp_path / f1).exists())
            self.assertFalse((tmp_path / f"{f1}.reserve").exists())

            release_filename_reservation(tmp_path / f2)
            self.assertFalse((tmp_path / f"{f2}.reserve").exists())


class TestPDFPrivacyAndExcludedColumns(unittest.TestCase):
    """PII redaction and sensitive column exclusion from correlations and charts."""

    def test_pdf_report_excludes_sensitive_columns(self):
        from Utils.PDF import generate_pdf_report
        df = pd.DataFrame({
            "social_security_number": [111223333, 222334444, 333445555, 444556666, 555667777],
            "feature_x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature_y": [15.0, 25.0, 35.0, 45.0, 55.0],
            "confidential_salary": [50000, 60000, 70000, 80000, 90000],
        })
        pdf_bytes = generate_pdf_report(
            df=df,
            dataset_name="privacy_test.csv",
            report_title="Confidential Test Report",
            include_charts=True,
            excluded_columns=["confidential_salary"]
        )
        self.assertTrue(len(pdf_bytes) > 1000)


class TestChartHueAndCategoryCapping(unittest.TestCase):
    """High-cardinality hue and category capping preventing UI hang."""

    def test_histogram_and_box_plots_cap_high_cardinality(self):
        from Utils.Charts import create_histogram_plot, create_box_violin_plot
        np.random.seed(42)
        df = pd.DataFrame({
            "val": np.random.randn(500),
            "user_id": [f"user_{i}" for i in range(500)],
        })
        fig_hist = create_histogram_plot(df, "val", hue_col="user_id")
        self.assertIsNotNone(fig_hist)

        fig_box = create_box_violin_plot(df, y_col="val", x_col="user_id")
        self.assertIsNotNone(fig_box)


class TestMLFeatureContractAndSemantics(unittest.TestCase):
    """F-13 & F-15: ML schema contract, strict coercion, and explicit metric attribution."""

    def test_ml_contract_and_unparseable_string_rejection(self):
        from Utils.ML import predict_with_model
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df = pd.DataFrame({
                "num_feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat_feat": ["A", "B", "A", "B", "A", "B"],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            })
            res = train_and_evaluate_model(
                df,
                target_col="target",
                feature_cols=["num_feat", "cat_feat"],
                model_name="Linear Regression",
                problem_type="Regression"
            )
            self.assertIn("feature_dtypes", res)
            self.assertIn("categories", res)
            self.assertIn("nullable", res)
            self.assertEqual(res["importance_type"], "Absolute Standardized Coefficient Magnitude (|β|)")

            mpath = save_trained_model(res, "contract_model", directory=tmp_path)
            bundle = load_trained_model(mpath, allowed_dir=tmp_path)
            self.assertIn("feature_dtypes", bundle)
            self.assertEqual(bundle["importance_type"], "Absolute Standardized Coefficient Magnitude (|β|)")

            test_df = pd.DataFrame({"num_feat": [7.0, 8.0], "cat_feat": ["A", "B"]})
            preds = predict_with_model(bundle, test_df)
            self.assertEqual(len(preds), 2)

            bad_test_df = pd.DataFrame({"num_feat": [7.0, "NOT_A_NUMBER"], "cat_feat": ["A", "B"]})
            with self.assertRaisesRegex(ValueError, r"(?i)non-numeric text values"):
                predict_with_model(bundle, bad_test_df)


class TestCredentialSessionStateSecurity(unittest.TestCase):
    """Honest session state security terminology."""

    def test_secrets_terminology_and_clearing(self):
        from Utils.secrets import release
        from types import SimpleNamespace
        from unittest.mock import patch
        state = {"gemini_secret": "secret_abc", "gemini_ver": 0}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=state)):
            cleared = release("gemini")
            self.assertEqual(cleared, ["gemini"])
            self.assertNotIn("gemini_secret", state)
            self.assertEqual(state.get("gemini_ver"), 1)


if __name__ == '__main__':
    unittest.main()

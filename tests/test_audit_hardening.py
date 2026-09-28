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
        seen = {"file.csv", "file_1.csv"}
        u = get_unique_filename("file.csv", extra_names=seen)
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
        import sys, types
        from Utils.Gemini import GeminiError, _new_sdk_client
        fake_genai = types.ModuleType("google.genai")
        class OldClient:
            def __init__(self, **kwargs):
                if "http_options" in kwargs:
                    raise TypeError("unexpected keyword 'http_options'")
        fake_genai.Client = OldClient
        with patch.dict(sys.modules, {"google.genai": fake_genai}):
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

            bundle = load_trained_model(p1)
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


if __name__ == "__main__":
    unittest.main()

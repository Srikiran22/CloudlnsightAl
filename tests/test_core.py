import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Utils.AIConvert import build_conversion_prompt, parse_ai_csv
from Utils.Charts import create_scatter_plot
from Utils.Gemini import get_dataset_summary_context
from Utils.ML import (
    detect_problem_type, train_and_evaluate_model,
    save_trained_model, load_trained_model, predict_with_model,
)
from Utils.PDF import generate_pdf_report
from Utils.S3 import list_s3_datasets, upload_s3_dataset
from Utils.paths import AIConversionRequired, resolve_dataset_path, read_tabular


class FakeS3Client:
    def __init__(self):
        self.pages = iter([
            {
                "Contents": [{"Key": "first.csv"}, {"Key": "skip.zip"}],
                "IsTruncated": True,
                "NextContinuationToken": "next-page",
            },
            {"Contents": [{"Key": "archive/second.xlsx"}], "IsTruncated": False},
        ])
        self.upload = None

    def list_objects_v2(self, **kwargs):
        return next(self.pages)

    def put_object(self, **kwargs):
        self.upload = kwargs


class CoreUtilityTests(unittest.TestCase):
    def test_problem_type_preserves_continuous_numeric_targets(self):
        df = pd.DataFrame({"target": [1, 2, 3, 4, 5, 6]})
        self.assertEqual(detect_problem_type(df, "target"), "Regression")

    def test_problem_type_identifies_repeated_numeric_labels(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
        self.assertEqual(detect_problem_type(df, "target"), "Classification")

    def test_regression_requires_numeric_target_values(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "target": ["one", "two", "three", "four"],
        })
        with self.assertRaisesRegex(ValueError, "numeric values"):
            train_and_evaluate_model(
                df, "target", ["feature"], "Linear Regression", "Regression", test_size=0.5
            )

    def test_scatter_plot_safely_ignores_invalid_optional_features(self):
        df = pd.DataFrame({
            "label": ["A", "B", "C"],
            "value": [1, 2, 3],
            "size": [-1, 2, 3],
        })
        figure = create_scatter_plot(df, "label", "value", size_col="size", add_trendline=True)
        self.assertEqual(len(figure.data), 1)

    def test_gemini_context_truncates_long_text(self):
        df = pd.DataFrame({"note": ["x" * 500], "value": [1]})
        context = get_dataset_summary_context(df, "demo.csv")
        self.assertIn("...", context)
        self.assertLess(len(context), 1_000)

    def test_dataset_paths_cannot_escape_the_workspace_data_directory(self):
        with self.assertRaisesRegex(ValueError, "Datasets"):
            resolve_dataset_path("../App.py")

    def test_s3_listing_paginates_and_uploads_bytes(self):
        client = FakeS3Client()
        self.assertEqual(
            list_s3_datasets("bucket", client),
            ["first.csv", "archive/second.xlsx"],
        )

        upload_s3_dataset(pd.DataFrame({"value": [1]}), "bucket", "result.csv", client)
        self.assertEqual(client.upload["Bucket"], "bucket")
        self.assertEqual(client.upload["Key"], "result.csv")
        self.assertIsInstance(client.upload["Body"], bytes)

    def test_s3_listing_includes_all_supported_formats(self):
        class MultiFormatClient:
            def __init__(self):
                self.calls = 0

            def list_objects_v2(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "Contents": [
                            {"Key": "events.jsonl"}, {"Key": "table.tsv"},
                            {"Key": "notes.txt"}, {"Key": "skip.pdf.exe"},
                        ],
                        "IsTruncated": False,
                    }
                return {}

        keys = list_s3_datasets("bucket", MultiFormatClient())
        self.assertIn("events.jsonl", keys)
        self.assertIn("table.tsv", keys)
        self.assertNotIn("skip.pdf.exe", keys)


class ModelPersistenceTests(unittest.TestCase):
    def _train(self):
        df = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        return train_and_evaluate_model(
            df, "target", ["feature"], "Logistic Regression", "Classification", test_size=0.5
        )

    def test_model_save_load_predict_round_trip(self):
        results = self._train()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_trained_model(results, "round trip model", directory=tmp)
            self.assertTrue(Path(path).exists())

            bundle = load_trained_model(path)
            self.assertEqual(bundle["problem_type"], "Classification")
            self.assertEqual(bundle["target_col"], "target")

            scoring_df = pd.DataFrame({"feature": [2.5, 6.5], "extra": ["x", "y"]})
            predictions = predict_with_model(bundle, scoring_df)
            self.assertIn("prediction", predictions.columns)
            self.assertEqual(len(predictions), 2)

    def test_predict_rejects_missing_features(self):
        results = self._train()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_trained_model(results, "strict", directory=tmp)
            bundle = load_trained_model(path)
            with self.assertRaisesRegex(ValueError, "missing required features"):
                predict_with_model(bundle, pd.DataFrame({"unrelated": [1]}))


class UniversalIngestionTests(unittest.TestCase):
    def test_jsonl_lines_become_rows(self):
        payload = b'{"name": "a", "value": 1}\n{"name": "b", "value": 2}\n'
        df = read_tabular(payload, filename="events.jsonl")
        self.assertEqual(list(df.columns), ["name", "value"])
        self.assertEqual(len(df), 2)

    def test_nested_json_is_flattened(self):
        payload = b'[{"user": {"name": "ann", "age": 30}, "score": 9}]'
        df = read_tabular(payload, filename="users.json")
        self.assertIn("user.name", df.columns)
        self.assertEqual(df.iloc[0]["score"], 9)

    def test_tsv_is_parsed_with_tab_separator(self):
        payload = b"a\tb\n1\t2\n3\t4\n"
        df = read_tabular(payload, filename="table.tsv")
        self.assertEqual(df.shape, (2, 2))

    def test_xml_records_are_flattened(self):
        payload = (
            b"<records><record><id>1</id><name>alpha</name></record>"
            b"<record><id>2</id><name>beta</name></record></records>"
        )
        df = read_tabular(payload, filename="data.xml")
        self.assertEqual(len(df), 2)
        self.assertIn("id", df.columns)

    def test_html_table_is_extracted_without_external_parsers(self):
        payload = (
            b"<html><body><table>"
            b"<tr><th>city</th><th>pop</th></tr>"
            b"<tr><td>paris</td><td>11</td></tr>"
            b"<tr><td>rome</td><td>4</td></tr>"
            b"</table></body></html>"
        )
        df = read_tabular(payload, filename="page.html")
        self.assertEqual(list(df.columns), ["city", "pop"])
        self.assertEqual(len(df), 2)

    def test_delimited_plain_text_parses_natively(self):
        payload = "x|y\n1|2\n3|4\n".encode("utf-8")
        df = read_tabular(payload, filename="pipe.txt")
        self.assertEqual(df.shape, (2, 2))

    def test_unstructured_text_requests_ai_conversion(self):
        payload = "The quick brown fox jumps over the lazy dog.\n".encode("utf-8")
        with self.assertRaises(AIConversionRequired) as ctx:
            read_tabular(payload, filename="notes.txt")
        self.assertIn("fox", ctx.exception.raw_text)

    def test_parquet_round_trip(self):
        source = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        buffer = io.BytesIO()
        source.to_parquet(buffer, index=False)
        df = read_tabular(buffer.getvalue(), filename="data.parquet")
        self.assertEqual(list(df.columns), ["a", "b"])


class AIConversionTests(unittest.TestCase):
    def test_parse_ai_csv_strips_markdown_fences(self):
        response = "```csv\nname,value\na,1\nb,2\n```"
        df = parse_ai_csv(response)
        self.assertEqual(list(df.columns), ["name", "value"])
        self.assertEqual(len(df), 2)

    def test_parse_ai_csv_ignores_surrounding_prose(self):
        response = "Here is the CSV you requested:\nname,value\na,1\nHope this helps!"
        df = parse_ai_csv(response)
        self.assertEqual(list(df.columns), ["name", "value"])
        self.assertEqual(len(df), 1)

    def test_parse_ai_csv_survives_prose_containing_commas(self):
        response = (
            "Sure, here is the extracted table:\n"
            "city,population\n"
            "paris,11\n"
            "rome,4\n"
            "Note: values, as reported, are in millions."
        )
        df = parse_ai_csv(response)
        self.assertEqual(list(df.columns), ["city", "population"])
        self.assertEqual(len(df), 2)

    def test_parse_ai_csv_rejects_empty_output(self):
        with self.assertRaisesRegex(ValueError, "CSV"):
            parse_ai_csv("no data here at all")

    def test_conversion_prompt_bounds_sample_and_warns_on_untrusted_content(self):
        prompt = build_conversion_prompt("x" * 20000, "log.txt")
        self.assertLess(len(prompt), 14000)
        self.assertIn("untrusted", prompt)


class PDFReportTests(unittest.TestCase):
    def test_detailed_report_renders_all_sections(self):
        df = pd.DataFrame({
            "value": [10, 12, 11, 40, 13, 12, 11, 10],
            "category": ["a", "b", "a", "c", "b", "a", "c", "b"],
        })
        pdf_bytes = generate_pdf_report(
            df,
            "unit_test.csv",
            include_ai_insights="## Summary\n- Insight one",
            include_charts=False,
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 3000)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import io
import pandas as pd
import numpy as np

from Utils.AIConvert import (
    MAX_SAMPLE_CHARS, MAX_CONVERSION_CHARS, _split_into_chunks,
    build_continuation_prompt, convert_to_dataframe
)
from Utils.paths import read_tabular, atomic_write
from Utils.privacy import detect_sensitive_columns
from Utils.ML import predict_with_model, train_and_evaluate_model
from Utils.Charts import create_bar_count_plot
from Utils.PDF import generate_pdf_report, _resolve_pdf_fonts


class RemediationAuditRegressionTests(unittest.TestCase):
    """Adversarial and regression tests covering the 10 audit remediation fixes."""

    def test_ai_convert_chunking_splits_line_aligned(self):
        """Chunking splits along lines without splitting words or mid-line records."""
        lines = [f"record_id: {i}, metric: {i * 10}\n" for i in range(1000)]
        raw_text = "".join(lines)
        chunks = _split_into_chunks(raw_text, max_chunk_chars=4000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4000 + 100)
            self.assertTrue(chunk.endswith("\n"))
        self.assertEqual("".join(chunks), raw_text)

    def test_ai_convert_chunked_extraction_combines_frames(self):
        """Documents between 12,000 and 60,000 chars are extracted across chunks without data loss."""
        # ~20,000 chars (exceeds MAX_SAMPLE_CHARS of 12,000)
        lines = [f"detailed_row_{i}_transaction_identifier,metric_value_{i * 2}\n" for i in range(400)]
        raw_text = "name,score\n" + "".join(lines)
        self.assertGreater(len(raw_text), MAX_SAMPLE_CHARS)
        self.assertLessEqual(len(raw_text), MAX_CONVERSION_CHARS)

        call_count = 0
        def fake_generate(api_key, model_name, prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "```csv\nname,score\nrow_1,2\nrow_2,4\n```"
            else:
                return "```csv\nname,score\nrow_3,6\nrow_4,8\n```"

        with patch("Utils.AIConvert._generate_content", side_effect=fake_generate):
            df = convert_to_dataframe("fake-key", raw_text, "long_notes.txt")
        self.assertGreater(call_count, 1)
        self.assertEqual(len(df), 4)
        self.assertListEqual(list(df["name"]), ["row_1", "row_2", "row_3", "row_4"])

    def test_ai_convert_rejects_oversized_documents_clearly(self):
        """Documents exceeding MAX_CONVERSION_CHARS fail clearly rather than losing records."""
        oversized = "a" * (MAX_CONVERSION_CHARS + 100)
        with self.assertRaisesRegex(ValueError, "exceeds maximum supported AI conversion limit"):
            convert_to_dataframe("fake-key", oversized, "massive.txt")

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
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        first_page = reader.pages[0].extract_text()
        # Verify black square replacement character is not present in title / header
        self.assertNotIn("\u25a0", first_page)


if __name__ == "__main__":
    unittest.main()

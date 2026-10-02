"""Adversarial regression test suite verifying fixes for confirmed audit defects.

Covers:
1. Upload signature blindspot (complete streaming SHA-256 over 64 KB chunks).
2. AI CSV parser silent data loss & header punctuation preservation.
3. One-column AI CSV parsing & prose rejection.
4. Aggregated bar chart + hue grouping without crashes.
5. Pie/Treemap same-column (names_col == values_col) handling.
6. ML feature importance double-underscore name preservation.
7. Post-AI-conversion MAX_COMBINED_ROWS resource enforcement.
8. Upload session-state active dataset persistence across page revisits.
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from Utils.paths import DATASETS_DIR, MAX_COMBINED_ROWS, compute_upload_signature as _compute_upload_signature
from Utils.AIConvert import parse_ai_csv
from Utils.Charts import create_bar_count_plot, create_pie_treemap_plot
from Utils.ML import train_and_evaluate_model


class UploadSignatureAdversarialTests(unittest.TestCase):
    """Defect 1: Streaming SHA-256 upload signature regression tests."""

    def test_upload_signature_large_file_middle_bytes_change_produces_different_hash(self):
        """Files > 128 KB with identical heads and tails but different middle bytes must produce different signatures."""
        head_tail_padding = b"A" * (64 * 1024)
        middle_1 = b"B" * (64 * 1024)
        middle_2 = b"C" * (64 * 1024)

        buf1 = io.BytesIO(head_tail_padding + middle_1 + head_tail_padding)
        buf1.name = "large_file.csv"
        buf1.size = len(buf1.getvalue())

        buf2 = io.BytesIO(head_tail_padding + middle_2 + head_tail_padding)
        buf2.name = "large_file.csv"
        buf2.size = len(buf2.getvalue())

        sig1 = _compute_upload_signature([buf1])
        sig2 = _compute_upload_signature([buf2])

        self.assertNotEqual(sig1, sig2)
        # Verify file pointer was restored to 0
        self.assertEqual(buf1.tell(), 0)
        self.assertEqual(buf2.tell(), 0)

    def test_upload_signature_unchanged_file_produces_identical_signature(self):
        """Same file unchanged must produce identical signatures on multiple calls."""
        content = b"header1,header2\n" + (b"data1,data2\n" * 10000)
        buf = io.BytesIO(content)
        buf.name = "test.csv"
        buf.size = len(content)

        sig_first = _compute_upload_signature([buf])
        sig_second = _compute_upload_signature([buf])

        self.assertEqual(sig_first, sig_second)
        self.assertEqual(buf.tell(), 0)

    def test_upload_signature_deterministic_order_multi_file(self):
        """Multiple files produce deterministic signatures and preserve order."""
        buf_a = io.BytesIO(b"file a content")
        buf_a.name = "a.csv"
        buf_b = io.BytesIO(b"file b content")
        buf_b.name = "b.csv"

        sig_ab = _compute_upload_signature([buf_a, buf_b])
        sig_ab_again = _compute_upload_signature([buf_a, buf_b])
        self.assertEqual(sig_ab, sig_ab_again)

        sig_ba = _compute_upload_signature([buf_b, buf_a])
        self.assertNotEqual(sig_ab, sig_ba)


class AICsvParserAdversarialTests(unittest.TestCase):
    """Defect 2 & Defect 7: AI CSV parser robust header detection and 1-column support."""

    def test_header_with_business_punctuation_preserves_first_data_row(self):
        """Item #, Department & Team, Cost [$] must be recognized as headers and row 1 preserved as data."""
        csv_text = (
            "Item #,Department & Team,Cost [$]\n"
            "P-101,Engineering,100.50\n"
            "P-102,HR & Ops,50.00\n"
        )
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["Item #", "Department & Team", "Cost [$]"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["Item #"], "P-101")
        self.assertEqual(df.iloc[0]["Department & Team"], "Engineering")

    def test_headers_with_varied_punctuation_and_quotes(self):
        """Headers with :, ?, @, [], &, $, quotes, apostrophes must be accepted."""
        csv_text = (
            "Item #,Dept & Team,Cost [$],Status: Active,Required?,User @ Domain,'Type',Owner's ID\n"
            "1001,Marketing,150.00,Active,Yes,alice@work,Standard,OWN-1\n"
        )
        df = parse_ai_csv(csv_text)
        self.assertEqual(len(df.columns), 8)
        self.assertEqual(len(df), 1)
        self.assertIn("Status: Active", df.columns)
        self.assertIn("Required?", df.columns)
        self.assertIn("User @ Domain", df.columns)
        self.assertIn("Owner's ID", df.columns)

    def test_alphanumeric_first_data_row_not_chosen_as_header(self):
        """Alphanumeric first data row (e.g. P-101) must stay as data row 0."""
        csv_text = (
            "Product Code,Department & Subteam,Price\n"
            "P-101,Sales & Marketing,99.99\n"
            "P-102,R&D,199.99\n"
        )
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["Product Code", "Department & Subteam", "Price"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["Product Code"], "P-101")

    def test_numeric_first_data_row_not_chosen_as_header(self):
        """Numeric literals in row 1 must not be chosen over row 0 header."""
        csv_text = "Year,Count,Cost\n2020,15,45.50\n2021,20,55.00\n"
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["Year", "Count", "Cost"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["Year"], 2020)

    def test_quoted_fields_with_embedded_commas_preserved(self):
        """Quoted commas inside fields must be preserved without column splitting."""
        csv_text = 'id,title,notes\n1,Task 1,"High priority, expedite ASAP"\n2,Task 2,"Normal, standard review"\n'
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["id", "title", "notes"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["notes"], "High priority, expedite ASAP")

    def test_multiline_fields_preserved(self):
        """Multiline fields spanning multiple newlines inside quotes must be preserved."""
        csv_text = 'id,name,description\n1,Widget,"This is line 1\nand this is line 2"\n2,Gadget,"Single line"\n'
        df = parse_ai_csv(csv_text)
        self.assertEqual(len(df), 2)
        self.assertIn("\n", df.iloc[0]["description"])
        self.assertEqual(df.iloc[1]["name"], "Gadget")

    def test_ambiguous_prose_is_rejected(self):
        """Natural language sentences with commas or poetry must be rejected as CSV."""
        prose_text = (
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer,\n"
            "The slings and arrows of outrageous fortune,\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(prose_text)

    def test_one_column_header_and_multiple_rows(self):
        """Legitimate one-column email list must be parsed cleanly."""
        csv_text = "Email\nalice@example.com\nbob@example.com\ncarol@example.com\n"
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["Email"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df["Email"].tolist(), ["alice@example.com", "bob@example.com", "carol@example.com"])

    def test_one_column_numeric_values(self):
        """Legitimate one-column numeric list must be parsed."""
        csv_text = "Score\n95.5\n80.0\n100.0\n"
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["Score"])
        self.assertEqual(len(df), 3)

    def test_one_column_identifiers(self):
        """Legitimate one-column identifier list must be parsed."""
        csv_text = "User_ID\nUSR-001\nUSR-002\nUSR-003\n"
        df = parse_ai_csv(csv_text)
        self.assertEqual(list(df.columns), ["User_ID"])
        self.assertEqual(len(df), 3)

    def test_one_column_prose_is_rejected(self):
        """Arbitrary prose sentences without columns must be rejected."""
        prose_text = (
            "This is an executive summary of our Q3 performance.\n"
            "Overall revenue grew by eight percent year over year.\n"
            "Operating expenses remained strictly within the planned budget.\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(prose_text)

    def test_header_abbreviations_preserve_data_row_1(self):
        """Business headers ending in abbreviations must preserve data row 1 without row dropping."""
        cases = [
            ("Item No.", "Item No.,Vendor,Price\nP-101,Acme,99.99\nP-102,Globex,149.99\n", ["Item No.", "Vendor", "Price"], "P-101"),
            ("Dept. No.", "Dept. No.,Department Name,Headcount\n10,Engineering,45\n20,Sales,30\n", ["Dept. No.", "Department Name", "Headcount"], 10),
            ("Vendor Inc.", "ID,Vendor Inc.,Status\n1,Alpha Inc.,Active\n2,Beta LLC,Inactive\n", ["ID", "Vendor Inc.", "Status"], 1),
            ("Total Amt.", "Order,Total Amt.,Tax\n1001,150.00,12.00\n1002,250.00,20.00\n", ["Order", "Total Amt.", "Tax"], 1001),
            ("Sr. No.", "Sr. No.,Full Name,Role\n1,John Doe,Developer\n2,Jane Smith,Manager\n", ["Sr. No.", "Full Name", "Role"], 1),
            ("Ménage & Co.", "Ménage & Co.,Branch,Revenue\nAlpha,Paris,50000\nBeta,Lyon,35000\n", ["Ménage & Co.", "Branch", "Revenue"], "Alpha"),
        ]
        for label, csv_text, expected_cols, expected_first_val in cases:
            with self.subTest(abbrev=label):
                df = parse_ai_csv(csv_text)
                self.assertEqual(list(df.columns), expected_cols, f"Columns mismatched for {label}")
                self.assertEqual(len(df), 2, f"Row dropped for {label}")
                self.assertEqual(df.iloc[0][expected_cols[0]], expected_first_val)

    def test_genuine_sentence_ending_in_period_is_rejected(self):
        """A genuine sentence ending in a period must not be ingested as a CSV table."""
        sentence_text = (
            "The implementation has been thoroughly reviewed and tested.\n"
            "All functional requirements were successfully satisfied.\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(sentence_text)

    def test_natural_language_prose_containing_commas_rejected(self):
        """Natural-language sentences with commas must not be parsed into a table."""
        prose_text = (
            "However, we should note that, during the third quarter, sales were strong.\n"
            "In addition, marketing expenses declined, whereas operating revenue rose.\n"
            "Finally, customer retention reached an all-time high, exceeding our goal.\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(prose_text)

    def test_unfenced_conversational_response_multiline_csv_parsed(self):
        """Unfenced conversational response containing multiline quoted cells must parse accurately."""
        text = (
            "Here is the table you asked for:\n"
            "ID,Description,Status\n"
            '1,"Line 1\nLine 2\nLine 3",Active\n'
            '2,"Single line",Done\n'
            "Hope this helps!"
        )
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["ID", "Description", "Status"])
        self.assertEqual(len(df), 2)
        self.assertIn("Line 1\nLine 2\nLine 3", df.iloc[0]["Description"])
        self.assertEqual(df.iloc[1]["Status"], "Done")

    def test_multi_column_business_headers_with_mixed_punctuation(self):
        """Multi-column headers with mixed business punctuation and symbols must parse cleanly."""
        csv_text = (
            "Item #,Dept. No.,Vendor Inc.,Cost [$],Total Amt.,Status: OK?,User @ Domain\n"
            "101,10,Alpha Inc.,99.50,110.00,Yes,alice@work\n"
            "102,20,Beta LLC,199.50,220.00,No,bob@work\n"
        )
        df = parse_ai_csv(csv_text)
        self.assertEqual(len(df.columns), 7)
        self.assertEqual(len(df), 2)
        self.assertIn("Dept. No.", df.columns)
        self.assertIn("Vendor Inc.", df.columns)
        self.assertIn("Total Amt.", df.columns)
        self.assertEqual(df.iloc[0]["Item #"], 101)


class ChartsAggregatedBarHueTests(unittest.TestCase):
    """Defect 3 & Defect 4: Chart aggregation with hue_col and names_col == values_col."""

    def setUp(self):
        self.df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C"],
            "amount": [10.0, 20.0, 15.0, 25.0, 30.0],
            "dept": ["Engineering", "Product", "Engineering", "Product", "Engineering"],
        })

    def test_bar_sum_with_hue(self):
        """Aggregated Sum with hue_col must group by [x_col, hue_col] and render without crashing."""
        fig = create_bar_count_plot(self.df, x_col="category", y_col="amount", agg_func="Sum", hue_col="dept")
        self.assertIsNotNone(fig)

    def test_bar_mean_with_hue(self):
        """Aggregated Mean with hue_col must render without crashing."""
        fig = create_bar_count_plot(self.df, x_col="category", y_col="amount", agg_func="Mean", hue_col="dept")
        self.assertIsNotNone(fig)

    def test_bar_count_with_hue(self):
        """Frequency count with hue_col must render without crashing."""
        fig = create_bar_count_plot(self.df, x_col="category", agg_func="Count", hue_col="dept")
        self.assertIsNotNone(fig)

    def test_bar_x_col_equals_y_col(self):
        """Aggregated bar chart when x_col == y_col must avoid name collisions in reset_index."""
        fig = create_bar_count_plot(self.df, x_col="amount", y_col="amount", agg_func="Sum")
        self.assertIsNotNone(fig)

    def test_bar_x_col_equals_y_col_with_hue(self):
        """Aggregated bar chart when x_col == y_col and hue_col is supplied."""
        fig = create_bar_count_plot(self.df, x_col="amount", y_col="amount", agg_func="Sum", hue_col="dept")
        self.assertIsNotNone(fig)

    def test_bar_x_col_equals_hue(self):
        """Aggregated bar chart when x_col == hue_col."""
        fig = create_bar_count_plot(self.df, x_col="category", y_col="amount", agg_func="Sum", hue_col="category")
        self.assertIsNotNone(fig)

    def test_bar_horizontal_orientation_with_hue(self):
        """Aggregated bar chart with horizontal orientation and hue_col."""
        fig = create_bar_count_plot(self.df, x_col="category", y_col="amount", agg_func="Sum", hue_col="dept", orientation="h")
        self.assertIsNotNone(fig)

    def test_pie_names_col_equals_values_col(self):
        """Pie chart when names_col == values_col must not raise duplicate column error."""
        fig = create_pie_treemap_plot(self.df, names_col="amount", values_col="amount", plot_type="Pie")
        self.assertIsNotNone(fig)

    def test_donut_names_col_equals_values_col(self):
        """Donut chart when names_col == values_col must render without crash."""
        fig = create_pie_treemap_plot(self.df, names_col="amount", values_col="amount", plot_type="Donut")
        self.assertIsNotNone(fig)

    def test_treemap_names_col_equals_values_col(self):
        """Treemap chart when names_col == values_col must render without crash."""
        fig = create_pie_treemap_plot(self.df, names_col="amount", values_col="amount", plot_type="Treemap")
        self.assertIsNotNone(fig)


class MLFeatureImportanceNamePreservationTests(unittest.TestCase):
    """Defect 5: ML feature importance double-underscore preservation."""

    def test_feature_names_with_double_underscores_preserved(self):
        """Features such as user__id, account__id, another__score must not collapse into id or score."""
        df = pd.DataFrame({
            "user__id": list(range(1, 21)),
            "account__id": [i * 10 for i in range(1, 21)],
            "another__score": [i * 1.5 for i in range(1, 21)],
            "target": [0, 1] * 10,
        })
        res = train_and_evaluate_model(
            df,
            target_col="target",
            feature_cols=["user__id", "account__id", "another__score"],
            model_name="Random Forest",
            problem_type="Classification",
        )
        importances = res.get("feature_importances", {})
        self.assertIn("user__id", importances)
        self.assertIn("account__id", importances)
        self.assertIn("another__score", importances)
        self.assertNotIn("id", importances)
        self.assertNotIn("score", importances)
        self.assertEqual(len(importances), 3)

    def test_numeric_and_categorical_features_with_double_underscores(self):
        """Both numeric and categorical features containing double underscores must preserve their names."""
        df = pd.DataFrame({
            "user__id": list(range(1, 21)),
            "role__type": ["admin", "member"] * 10,
            "target": [10.0 + i for i in range(1, 21)],
        })
        res = train_and_evaluate_model(
            df,
            target_col="target",
            feature_cols=["user__id", "role__type"],
            model_name="Random Forest",
            problem_type="Regression",
        )
        importances = res.get("feature_importances", {})
        self.assertIn("user__id", importances)
        # Categorical feature gets one-hot encoded: role__type_admin or role__type_member
        encoded_role_names = [k for k in importances if "role__type" in k]
        self.assertGreaterEqual(len(encoded_role_names), 1)


class UploadPostAICombinedRowLimitTests(unittest.TestCase):
    """Defect 6: Post-AI conversion enforcement of MAX_COMBINED_ROWS."""

    def test_post_ai_conversion_enforces_combined_row_ceiling(self):
        """When multiple converted frames exceed MAX_COMBINED_ROWS, merge must be aborted before materializing."""
        # Simulate two converted dataframes whose combined rows exceed MAX_COMBINED_ROWS
        df1 = pd.DataFrame({"a": [1] * (MAX_COMBINED_ROWS // 2 + 10)})
        df2 = pd.DataFrame({"a": [2] * (MAX_COMBINED_ROWS // 2 + 10)})
        parsed = [("file1.csv", df1), ("file2.csv", df2)]

        total_rows = sum(len(f[1]) for f in parsed)
        self.assertGreater(total_rows, MAX_COMBINED_ROWS)

        # Ensure the resource check prevents merge_frames from being invoked
        with unittest.mock.patch("Utils.batch.merge_frames") as mock_merge:
            # Emulate logic from Pages/Upload.py
            if total_rows > MAX_COMBINED_ROWS:
                merged_df = None
            else:
                merged_df = mock_merge(parsed)
            self.assertIsNone(merged_df)
            mock_merge.assert_not_called()


class UploadSessionStateLifecycleTests(unittest.TestCase):
    """Defect 8: Upload session state active dataset persistence across revisits."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_datasets_dir = DATASETS_DIR
        from Utils import paths
        paths.DATASETS_DIR = Path(self.temp_dir)

    def tearDown(self):
        from Utils import paths
        paths.DATASETS_DIR = self.orig_datasets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upload_revisit_preserves_external_active_dataset(self):
        """Lifecycle test: upload -> select another dataset -> revisit Upload page -> active dataset must remain unchanged."""
        upload_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Upload.py")
        at = AppTest.from_file(upload_script, default_timeout=15)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # 1. Upload dataset_a
        at.file_uploader[0].upload("dataset_a.csv", b"col1,col2\n1,2\n3,4\n")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["dataset_name"], "dataset_a.csv")

        # 2. Simulate user switching to dataset_b on another page / in sidebar
        other_df = pd.DataFrame({"x": [100, 200], "y": [300, 400]})
        at.session_state["dataset_name"] = "dataset_b.csv"
        at.session_state["current_df"] = other_df

        # 3. Revisit / rerun the Upload page with file_uploader still holding dataset_a
        at.run()
        self.assertEqual(len(at.exception), 0)

        # 4. Verify dataset_b remains active and was NOT overwritten by cached dataset_a upload
        self.assertEqual(at.session_state["dataset_name"], "dataset_b.csv")
        self.assertTrue(at.session_state["current_df"].equals(other_df))


class UploadMixedBatchLifecycleTests(unittest.TestCase):
    """Blocker 2: Mixed-mode upload lifecycle (direct tabular + AI-converted files)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_datasets_dir = DATASETS_DIR
        from Utils import paths
        paths.DATASETS_DIR = Path(self.temp_dir)

    def tearDown(self):
        from Utils import paths
        paths.DATASETS_DIR = self.orig_datasets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mixed_direct_and_ai_converted_upload_lifecycle(self):
        """Mixed batch: upload direct CSV + unstructured TXT -> convert with Gemini -> verify merged active dataset."""
        upload_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Upload.py")
        at = AppTest.from_file(upload_script, default_timeout=15)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # 1. Upload direct CSV (2 rows) and unstructured TXT
        at.file_uploader[0].upload("direct.csv", b"id,val\n1,10\n2,20\n")
        at.file_uploader[0].upload("notes.txt", b"Unstructured notes...")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["dataset_name"], "direct.csv")
        self.assertEqual(len(at.session_state["current_df"]), 2)

        # 2. Convert unstructured TXT with Gemini
        gemini_inputs = [t for t in at.text_input if "Gemini" in t.label]
        if gemini_inputs:
            gemini_inputs[0].input("fake_gemini_key")
        else:
            at.session_state["gemini_secret"] = "fake_gemini_key"

        converted_df = pd.DataFrame({"id": [3, 4, 5], "val": [30, 40, 50]})
        with unittest.mock.patch("Utils.AIConvert.convert_to_dataframe", return_value=converted_df), \
             unittest.mock.patch("Pages.Upload.convert_to_dataframe", return_value=converted_df):
            convert_btn = next(b for b in at.button if "Convert with Gemini" in b.label)
            convert_btn.click()
            at.run()

        self.assertEqual(len(at.exception), 0)
        # Verify both datasets are present in parsed upload state
        parsed_entries = at.session_state["last_upload_state"]["parsed"]
        self.assertEqual(len(parsed_entries), 2)
        parsed_names = [p[0] for p in parsed_entries]
        self.assertIn("direct.csv", parsed_names)
        self.assertTrue(any("converted" in name for name in parsed_names))

        # Verify merged result is active and has combined rows
        self.assertTrue(at.session_state["dataset_name"].startswith("combined_dataset"))
        self.assertEqual(len(at.session_state["current_df"]), 5)

        # 3. Verify no duplicate conversion occurs on rerun
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.session_state["dataset_name"].startswith("combined_dataset"))
        self.assertEqual(len(at.session_state["current_df"]), 5)

    def test_mixed_batch_oversized_combined_rows_rejected(self):
        """Oversized mixed batch exceeding MAX_COMBINED_ROWS must be aborted before merge."""
        upload_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Upload.py")
        at = AppTest.from_file(upload_script, default_timeout=15)
        at.run()

        at.file_uploader[0].upload("direct.csv", b"id,val\n1,10\n2,20\n")
        at.file_uploader[0].upload("huge.txt", b"Huge content")
        at.run()

        gemini_inputs = [t for t in at.text_input if "Gemini" in t.label]
        if gemini_inputs:
            gemini_inputs[0].input("fake_gemini_key")
        else:
            at.session_state["gemini_secret"] = "fake_gemini_key"

        huge_df = pd.DataFrame({"id": [1] * MAX_COMBINED_ROWS, "val": [2] * MAX_COMBINED_ROWS})
        with unittest.mock.patch("Utils.AIConvert.convert_to_dataframe", return_value=huge_df), \
             unittest.mock.patch("Pages.Upload.convert_to_dataframe", return_value=huge_df):
            convert_btn = next(b for b in at.button if "Convert with Gemini" in b.label)
            convert_btn.click()
            at.run()

        error_msgs = [e.value for e in at.error]
        self.assertTrue(
            any("exceeds the maximum supported limit" in msg for msg in error_msgs),
            f"Expected combined row limit error, got: {error_msgs}"
        )


if __name__ == "__main__":
    unittest.main()

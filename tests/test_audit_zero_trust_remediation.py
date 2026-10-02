import io
from pathlib import Path
import unittest
import unittest.mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from Utils.AIConvert import parse_ai_csv
from Utils.paths import (
    MAX_COMBINED_ROWS,
    MAX_INGESTION_ROWS,
    read_tabular,
)


class ZeroTrustAICsvParserTests(unittest.TestCase):
    """Defect 1, 2, 3: Robust AI CSV parsing, conversational prose rejection, empty headers, long headers."""

    # Defect 1: Conversational prose falsely accepted
    def test_single_sentence_ending_in_short_word_rejected(self):
        """Conversational sentence ending in 'risk.' must not become a CSV dataset."""
        sentence = "I checked the log, found no risk."
        with self.assertRaises(ValueError):
            parse_ai_csv(sentence)

    def test_multi_sentence_conversational_prose_ending_in_short_words_rejected(self):
        """Multiple sentences ending in normal words like good., fine., risk., stop. must be rejected."""
        prose_cases = [
            ("risk and good", "I checked the log, found no risk.\nInspected system files, looks good."),
            ("test and fine", "We ran the test, everything is fine.\nAll systems go, no errors found."),
            ("time and plan", "Deployment completed, on time.\nFollowed the migration, plan."),
            ("data and stop", "Received input data, check data.\nEmergency alert, stop."),
            ("here and test", "Look at the log, check here.\nRunning verification, test."),
        ]
        for label, text in prose_cases:
            with self.subTest(prose=label), self.assertRaises(ValueError, msg=f"Prose '{label}' should have been rejected"):
                parse_ai_csv(text)

    def test_ordinary_comma_heavy_prose_rejected(self):
        """Ordinary prose with commas must not be parsed into a table."""
        prose = (
            "However, we should note that, during the third quarter, sales were strong.\n"
            "In addition, marketing expenses declined, whereas operating revenue rose.\n"
            "Finally, customer retention reached an all-time high, exceeding our goal.\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(prose)

    def test_poetry_and_literary_prose_rejected(self):
        """Poetic verse or literary lines with commas must be rejected."""
        prose = (
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer,\n"
            "The slings and arrows of outrageous fortune,\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(prose)

    def test_malformed_and_ambiguous_csv_rejected(self):
        """Ambiguous non-tabular text without tabular structure must be rejected."""
        ambiguous = (
            "Note: the following items were observed.\n"
            "Item 1 was fine.\n"
            "Item 2 had an issue.\n"
        )
        with self.assertRaises(ValueError):
            parse_ai_csv(ambiguous)

    # Legitimate abbreviations and punctuation preserved
    def test_legitimate_abbreviations_and_business_headers(self):
        """Headers ending in abbreviations (No., Inc., Amt., Sr., Co.) and snake_case periods (cost_approx.) must work."""
        cases = [
            ("Item No.", "Item No.,Vendor,Price\nP-101,Acme,99.99\nP-102,Globex,149.99\n", ["Item No.", "Vendor", "Price"], "P-101"),
            ("Dept. No.", "Dept. No.,Department Name,Headcount\n10,Engineering,45\n20,Sales,30\n", ["Dept. No.", "Department Name", "Headcount"], 10),
            ("Vendor Inc.", "ID,Vendor Inc.,Status\n1,Alpha Inc.,Active\n2,Beta LLC,Inactive\n", ["ID", "Vendor Inc.", "Status"], 1),
            ("Total Amt.", "Order,Total Amt.,Tax\n1001,150.00,12.00\n1002,250.00,20.00\n", ["Order", "Total Amt.", "Tax"], 1001),
            ("Sr. No.", "Sr. No.,Full Name,Role\n1,John Doe,Developer\n2,Jane Smith,Manager\n", ["Sr. No.", "Full Name", "Role"], 1),
            ("Ménage & Co.", "Ménage & Co.,Branch,Revenue\nAlpha,Paris,50000\nBeta,Lyon,35000\n", ["Ménage & Co.", "Branch", "Revenue"], "Alpha"),
            ("cost_approx.", "cost_approx.,margin_pct,category\n12.5,0.25,electronics\n45.0,0.30,hardware\n", ["cost_approx.", "margin_pct", "category"], 12.5),
        ]
        for label, text, expected_cols, expected_val in cases:
            with self.subTest(abbrev=label):
                df = parse_ai_csv(text)
                self.assertEqual(list(df.columns), expected_cols)
                self.assertEqual(len(df), 2, f"Data row dropped for {label}")
                self.assertEqual(df.iloc[0][expected_cols[0]], expected_val)

    def test_headers_with_varied_punctuation_and_symbols(self):
        """Preserve legitimate headers containing #, &, $, [], :, ?, @, quotes, apostrophes, Unicode."""
        text = (
            "Item #,Dept & Team,Cost [$],Status: Active,Required?,User @ Domain,'Type',Owner's ID,Métrique €\n"
            "1001,Marketing,150.00,Active,Yes,alice@work,Standard,OWN-1,42.50\n"
            "1002,Sales,250.00,Pending,No,bob@work,Custom,OWN-2,85.00\n"
        )
        df = parse_ai_csv(text)
        self.assertEqual(len(df.columns), 9)
        self.assertEqual(len(df), 2)
        self.assertIn("Item #", df.columns)
        self.assertIn("Dept & Team", df.columns)
        self.assertIn("Cost [$]", df.columns)
        self.assertIn("Status: Active", df.columns)
        self.assertIn("Required?", df.columns)
        self.assertIn("User @ Domain", df.columns)
        self.assertIn("Owner's ID", df.columns)
        self.assertIn("Métrique €", df.columns)

    def test_alphanumeric_and_numeric_first_data_rows_preserved(self):
        """Alphanumeric (e.g. P-101) and numeric (e.g. 2020) first data rows must remain data, not swapped."""
        text_alphanumeric = (
            "Product Code,Department & Subteam,Price\n"
            "P-101,Sales & Marketing,99.99\n"
            "P-102,R&D,199.99\n"
        )
        df_alpha = parse_ai_csv(text_alphanumeric)
        self.assertEqual(list(df_alpha.columns), ["Product Code", "Department & Subteam", "Price"])
        self.assertEqual(len(df_alpha), 2)
        self.assertEqual(df_alpha.iloc[0]["Product Code"], "P-101")

        text_numeric = (
            "Year,Count,Cost\n"
            "2020,15,45.50\n"
            "2021,20,55.00\n"
        )
        df_num = parse_ai_csv(text_numeric)
        self.assertEqual(list(df_num.columns), ["Year", "Count", "Cost"])
        self.assertEqual(len(df_num), 2)
        self.assertEqual(df_num.iloc[0]["Year"], 2020)

    def test_preamble_and_signoff_removed_with_unfenced_multiline(self):
        """Conversational preamble and sign-off must be stripped, and multiline quoted fields preserved."""
        text = (
            "Here is the table you asked for:\n"
            "ID,Description,Status\n"
            '1,"Line 1\nLine 2\nLine 3",Active\n'
            '2,"Single line",Done\n'
            "Hope this helps! Let me know if you need anything else."
        )
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["ID", "Description", "Status"])
        self.assertEqual(len(df), 2)
        self.assertIn("Line 1\nLine 2\nLine 3", df.iloc[0]["Description"])
        self.assertEqual(df.iloc[1]["Status"], "Done")

    def test_quoted_commas_preserved(self):
        """Quoted commas inside cells must not split into extra columns."""
        text = (
            'id,title,notes\n'
            '1,Task 1,"High priority, expedite ASAP"\n'
            '2,Task 2,"Normal, standard review"\n'
        )
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["id", "title", "notes"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["notes"], "High priority, expedite ASAP")

    def test_one_column_csv_supported(self):
        """One-column tables for emails, scores, identifiers must parse."""
        email_csv = "Email\nalice@example.com\nbob@example.com\ncarol@example.com\n"
        df_email = parse_ai_csv(email_csv)
        self.assertEqual(list(df_email.columns), ["Email"])
        self.assertEqual(len(df_email), 3)

        score_csv = "Score\n95.5\n80.0\n100.0\n"
        df_score = parse_ai_csv(score_csv)
        self.assertEqual(list(df_score.columns), ["Score"])
        self.assertEqual(len(df_score), 3)

        id_csv = "User_ID\nUSR-001\nUSR-002\nUSR-003\n"
        df_id = parse_ai_csv(id_csv)
        self.assertEqual(list(df_id.columns), ["User_ID"])
        self.assertEqual(len(df_id), 3)

    # Defect 2: Empty header fields cause row deletion
    def test_empty_header_id_empty_status(self):
        """Header 'id,,status' must preserve row 0 and assign synthetic column name 'column_2'."""
        text = "id,,status\n1,foo,bar\n2,baz,qux\n"
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["id", "column_2", "status"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["id"], 1)
        self.assertEqual(df.iloc[0]["column_2"], "foo")
        self.assertEqual(df.iloc[0]["status"], "bar")
        self.assertEqual(df.iloc[1]["id"], 2)

    def test_empty_header_leading_comma(self):
        """Header ',name,value' must preserve row 0 and assign synthetic column name 'column_1'."""
        text = ",name,value\n101,widget,29.99\n102,gadget,49.99\n"
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["column_1", "name", "value"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["column_1"], 101)
        self.assertEqual(df.iloc[0]["name"], "widget")

    def test_empty_header_consecutive_empty(self):
        """Header 'id,,,status' must assign 'column_2' and 'column_3' without row dropping."""
        text = "id,,,status\n1,alpha,beta,active\n2,gamma,delta,inactive\n"
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["id", "column_2", "column_3", "status"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["id"], 1)
        self.assertEqual(df.iloc[0]["column_2"], "alpha")
        self.assertEqual(df.iloc[0]["column_3"], "beta")

    def test_quoted_empty_header(self):
        """Header with quoted empty field 'id,"",status' must synthesize column_2 and preserve rows."""
        text = 'id,"",status\n1,foo,bar\n2,baz,qux\n'
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["id", "column_2", "status"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["column_2"], "foo")

    def test_header_with_mixed_empty_and_non_empty_names(self):
        """Header with mixed empty/non-empty fields '\"\",foo,\"\",bar' must preserve both rows."""
        text = '"",foo,"",bar\n10,alpha,20,beta\n30,gamma,40,delta\n'
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["column_1", "foo", "column_3", "bar"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["column_1"], 10)
        self.assertEqual(df.iloc[0]["foo"], "alpha")

    # Defect 3: Long descriptive headers rejected
    def test_long_descriptive_business_metric_header(self):
        """Multi-word descriptive headers like 'average monthly revenue per active user' must parse."""
        text = (
            "user_id,average monthly revenue per active user,status\n"
            "101,54.20,active\n"
            "102,68.10,churned\n"
        )
        df = parse_ai_csv(text)
        self.assertEqual(list(df.columns), ["user_id", "average monthly revenue per active user", "status"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["user_id"], 101)
        self.assertEqual(df.iloc[0]["average monthly revenue per active user"], 54.20)

    def test_long_survey_question_headers(self):
        """Survey questions as column headers must not be rejected merely for length."""
        text = (
            "respondent_id,how often do you use cloud infrastructure in your organization?,primary cloud provider,annual cloud spend in usd\n"
            "R-001,daily,AWS,50000\n"
            "R-002,weekly,GCP,25000\n"
        )
        df = parse_ai_csv(text)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual(len(df), 2)
        self.assertIn("how often do you use cloud infrastructure in your organization?", df.columns)
        self.assertEqual(df.iloc[0]["primary cloud provider"], "AWS")

    def test_long_technical_performance_headers(self):
        """Technical headers with 8+ words must parse without error."""
        text = (
            "model_id,mean squared error on cross validation holdout set,inference latency in milliseconds at p99,production ready\n"
            "M-1,0.0142,12.5,yes\n"
            "M-2,0.0098,18.2,yes\n"
        )
        df = parse_ai_csv(text)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual(len(df), 2)
        self.assertIn("mean squared error on cross validation holdout set", df.columns)
        self.assertEqual(df.iloc[0]["production ready"], "yes")


class ZeroRowParquetTests(unittest.TestCase):
    """Defect 4: Zero-row Parquet loading without batch_size=0 crash."""

    def test_schema_only_zero_row_parquet_loads_successfully(self):
        """Zero-row Parquet must load without error and preserve schema column names and dtypes."""
        df_empty = pd.DataFrame({
            "account_id": pd.Series(dtype="int64"),
            "username": pd.Series(dtype="str"),
            "balance": pd.Series(dtype="float64"),
        })
        buf = io.BytesIO()
        df_empty.to_parquet(buf)
        buf.seek(0)

        df_loaded = read_tabular(buf, filename="empty_test.parquet")
        self.assertEqual(len(df_loaded), 0)
        self.assertEqual(list(df_loaded.columns), ["account_id", "username", "balance"])
        self.assertEqual(str(df_loaded["account_id"].dtype), "int64")

    def test_oversized_parquet_still_rejected(self):
        """Parquet metadata reporting > MAX_INGESTION_ROWS must still be rejected."""
        buf = io.BytesIO()
        pd.DataFrame({"x": [1]}).to_parquet(buf)
        buf.seek(0)

        with unittest.mock.patch("pyarrow.parquet.ParquetFile.metadata") as mock_meta:
            mock_meta.num_rows = MAX_INGESTION_ROWS + 10
            with self.assertRaises(ValueError) as ctx:
                read_tabular(buf, filename="huge.parquet")
            self.assertIn("exceeds the maximum supported limit", str(ctx.exception))

    def test_one_row_parquet_loads_normally(self):
        """1-row normal Parquet file loads unchanged."""
        df_single = pd.DataFrame({"id": [42], "tag": ["alpha"]})
        buf = io.BytesIO()
        df_single.to_parquet(buf)
        buf.seek(0)

        df_loaded = read_tabular(buf, filename="single.parquet")
        self.assertEqual(len(df_loaded), 1)
        self.assertEqual(list(df_loaded.columns), ["id", "tag"])
        self.assertEqual(df_loaded.iloc[0]["id"], 42)


class OversizedMergeRerunAppTest(unittest.TestCase):
    """Defect 5: Oversized merge error persistence across Streamlit reruns."""

    def setUp(self):
        import tempfile
        from Utils import paths
        self.orig_datasets_dir = paths.DATASETS_DIR
        self.temp_dir = tempfile.mkdtemp()
        paths.DATASETS_DIR = Path(self.temp_dir)
        try:
            import Pages.Upload
            self.orig_upload_datasets_dir = getattr(Pages.Upload, "DATASETS_DIR", self.orig_datasets_dir)
            Pages.Upload.DATASETS_DIR = Path(self.temp_dir)
        except Exception:
            self.orig_upload_datasets_dir = None

    def tearDown(self):
        import shutil
        from Utils import paths
        paths.DATASETS_DIR = self.orig_datasets_dir
        if self.orig_upload_datasets_dir is not None:
            try:
                import Pages.Upload
                Pages.Upload.DATASETS_DIR = self.orig_upload_datasets_dir
            except Exception:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_oversized_merge_error_survives_reruns(self):
        """When MAX_COMBINED_ROWS is exceeded, error must remain visible on subsequent rerun."""
        upload_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Upload.py")
        at = AppTest.from_file(upload_script, default_timeout=15)
        at.run()

        # Simulate two files whose combined row count exceeds MAX_COMBINED_ROWS
        frame1 = pd.DataFrame({"id": [1] * (MAX_COMBINED_ROWS // 2 + 10), "v": [10] * (MAX_COMBINED_ROWS // 2 + 10)})
        frame2 = pd.DataFrame({"id": [2] * (MAX_COMBINED_ROWS // 2 + 10), "v": [20] * (MAX_COMBINED_ROWS // 2 + 10)})

        at.file_uploader[0].upload("file1.csv", b"dummy content 1")
        at.file_uploader[0].upload("file2.csv", b"dummy content 2")

        with unittest.mock.patch("Pages.Upload.read_tabular", side_effect=[frame1, frame2]), \
             unittest.mock.patch("Utils.paths.read_tabular", side_effect=[frame1, frame2]):
            at.run()

        # Check initial error message
        error_msgs = [e.value for e in at.error]
        self.assertTrue(
            any("exceeds the maximum supported limit" in msg for msg in error_msgs),
            f"Expected limit error on initial upload, got: {error_msgs}"
        )

        # Trigger a rerun with the same upload signature (simulating user interaction or page rerun)
        at.run()

        # Error must STILL be visible on rerun!
        rerun_error_msgs = [e.value for e in at.error]
        self.assertTrue(
            any("exceeds the maximum supported limit" in msg for msg in rerun_error_msgs),
            f"Error disappeared on rerun! Got: {rerun_error_msgs}"
        )


if __name__ == "__main__":
    unittest.main()

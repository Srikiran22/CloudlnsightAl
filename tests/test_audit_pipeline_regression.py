import concurrent.futures
import html
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from Utils.paths import (
    cleanup_storage,
    get_dataset_row_count,
    list_dataset_files,
    list_ready_datasets,
    list_source_documents,
    read_tabular,
    safe_stem,
    sanitize_for_csv_export,
    _detect_csv_delimiter,
)
from Utils.ML import (
    _datetime_to_epoch,
    _get_or_create_model_signing_key,
    load_trained_model,
    predict_with_model,
    save_trained_model,
    sign_model_artifact,
    train_and_evaluate_model,
)
from Utils.PDF import validate_report_template, generate_pdf_report
from Utils.sampling import sample_for_visualization, sample_for_analysis
from Utils.Charts import create_histogram_plot
from Utils.Gemini import generate_executive_insights
from Utils.S3 import upload_s3_dataset
from Utils.Preprocessing import fill_missing_values
from Utils.compare_logic import column_drift_rows, benjamini_hochberg_correction
from Utils.theme import get_semantic_tokens, SEMANTIC_TOKENS, CHART_PALETTE


def _malicious_trigger():
    raise RuntimeError("EXPLOIT EXECUTED!")


class Exploit:
    def __reduce__(self):
        return (_malicious_trigger, ())


class TestAuditPipelineRegression(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Finding A: Model deserialization trust boundary
    # -------------------------------------------------------------------------
    def test_finding_a_unsigned_model_rejected(self):
        """Unsigned model in allowed_dir is rejected before loading."""
        model_file = self.test_path / "model.joblib"
        import joblib
        joblib.dump({"pipeline": None, "bundle_version": 1}, model_file)

        with self.assertRaises(ValueError) as ctx:
            load_trained_model(model_file, allowed_dir=self.test_path)
        self.assertIn("lacks a cryptographic authenticity signature", str(ctx.exception).lower())

    def test_finding_a_valid_signed_model_accepted(self):
        """Valid signed model in allowed_dir loads successfully."""
        model_file = self.test_path / "model.joblib"
        import joblib
        bundle = {
            "pipeline": "mock_pipeline",
            "bundle_version": 1,
            "feature_cols": ["a"],
            "problem_type": "Regression",
        }
        joblib.dump(bundle, model_file)
        sign_model_artifact(model_file)

        loaded = load_trained_model(model_file, allowed_dir=self.test_path)
        self.assertEqual(loaded["pipeline"], "mock_pipeline")

    def test_finding_a_tampered_model_rejected(self):
        """Tampered model file invalidates signature and is rejected."""
        model_file = self.test_path / "model.joblib"
        import joblib
        joblib.dump({"pipeline": "clean", "bundle_version": 1}, model_file)
        sign_model_artifact(model_file)

        # Tamper with file content
        with open(model_file, "ab") as f:
            f.write(b"\x00CORRUPT")

        with self.assertRaises(ValueError) as ctx:
            load_trained_model(model_file, allowed_dir=self.test_path)
        self.assertIn("tampered with", str(ctx.exception).lower())

    def test_finding_a_model_outside_allowed_dir_rejected(self):
        """Model outside allowed_dir is rejected unless trusted=True."""
        outside_dir = tempfile.mkdtemp()
        try:
            model_file = Path(outside_dir) / "model.joblib"
            import joblib
            joblib.dump({"pipeline": "clean", "bundle_version": 1}, model_file)
            sign_model_artifact(model_file)

            with self.assertRaises(ValueError) as ctx:
                load_trained_model(model_file, allowed_dir=self.test_path)
            self.assertIn("security boundary rejection", str(ctx.exception).lower())

            # Explicit trusted override with audit log succeeds
            loaded = load_trained_model(model_file, allowed_dir=self.test_path, trusted=True)
            self.assertEqual(loaded["pipeline"], "clean")
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_finding_a_path_traversal_rejected(self):
        """Path traversal patterns are rejected before filesystem access."""
        with self.assertRaises(ValueError) as ctx:
            load_trained_model(self.test_path / ".." / "evil.joblib", allowed_dir=self.test_path)
        self.assertIn("path traversal detected", str(ctx.exception).lower())

    def test_finding_a_malicious_pickle_blocked_before_load(self):
        """Untrusted pickle execution payload is rejected BEFORE joblib.load executes."""
        model_file = self.test_path / "exploit.joblib"
        import pickle
        with open(model_file, "wb") as f:
            pickle.dump(Exploit(), f)

        # Loading must fail on signature verification, never reaching __reduce__
        with self.assertRaises(ValueError) as ctx:
            load_trained_model(model_file, allowed_dir=self.test_path)
        self.assertNotIn("EXPLOIT EXECUTED", str(ctx.exception))
        self.assertIn("authenticity signature", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Finding B: Model-signing key race (Atomic create-if-absent)
    # -------------------------------------------------------------------------
    def test_finding_b_signing_key_concurrency(self):
        """Concurrent workers converge on the exact same key without overwriting."""
        key_dir = self.test_path / "keys"
        key_dir.mkdir()

        def worker(_):
            return _get_or_create_model_signing_key(directory=key_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            keys = list(executor.map(worker, range(20)))

        self.assertEqual(len(set(keys)), 1, "All workers must observe the identical key")
        self.assertEqual(len(keys[0]), 32, "Key must be 32 bytes")

    # -------------------------------------------------------------------------
    # Finding C: Batch PDF report privacy & AI propagation
    # -------------------------------------------------------------------------
    def test_finding_c_batch_report_privacy_propagation(self):
        """Batch report generation excludes PII columns and handles AI insights."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 75000, 90000, 110000],
        })

        # Exclude 'name'
        pdf_bytes = generate_pdf_report(
            df=df,
            dataset_name="users.csv",
            report_title="Privacy Test",
            author_name="Auditor",
            include_charts=False,
            include_ai_insights="Summary findings here.",
            excluded_columns=["name"],
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    # -------------------------------------------------------------------------
    # Finding D: Timezone-aware datetime ML handling
    # -------------------------------------------------------------------------
    def test_finding_d_timezone_aware_normalization(self):
        """Timezone-aware timestamps are normalized to UTC epoch seconds."""
        # 2026-01-01 12:00:00 UTC == 2026-01-01 07:00:00 US/Eastern
        utc_ts = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
        est_ts = pd.Timestamp("2026-01-01 07:00:00", tz="US/Eastern")

        df = pd.DataFrame({
            "utc_date": [utc_ts, pd.NaT],
            "est_date": [est_ts, pd.NaT],
        })

        _datetime_to_epoch(df)

        # Both equivalent instants must yield identical epoch seconds
        self.assertEqual(df["utc_date"].iloc[0], df["est_date"].iloc[0])
        self.assertTrue(np.isnan(df["utc_date"].iloc[1]))
        self.assertTrue(np.isnan(df["est_date"].iloc[1]))

    # -------------------------------------------------------------------------
    # Finding E: S3 streaming memory isolation
    # -------------------------------------------------------------------------
    def test_finding_e_s3_streaming_bounds_and_cleanup(self):
        """S3 download enforces byte boundaries and cleans up files on failure."""
        from Utils.S3 import download_s3_dataset

        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.side_effect = [b"col1,col2\n", b"1,2\n" * 10, b""]
        mock_s3.get_object.return_value = {
            "ContentLength": 50,
            "Body": mock_body,
        }

        # Successful bounded download
        df, _ = download_s3_dataset("bucket", "key.csv", mock_s3)
        self.assertEqual(len(df), 10)

        # Oversized ContentLength rejected
        mock_s3.get_object.return_value = {
            "ContentLength": 300 * 1024 * 1024,
            "Body": mock_body,
        }
        with self.assertRaises(ValueError) as ctx:
            download_s3_dataset("bucket", "large.csv", mock_s3)
        self.assertIn("ingest limit is 200 mb", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Finding F: Global ingestion cell budget
    # -------------------------------------------------------------------------
    def test_finding_f_cell_budget_enforced(self):
        """read_tabular rejects files exceeding MAX_INGESTION_CELLS."""
        # 1 row × 3 columns = 3 cells > 2 cells
        buf = io.BytesIO(b"a,b,c\n1,2,3\n")
        with patch("Utils.paths.MAX_INGESTION_CELLS", 2):
            with self.assertRaises(ValueError) as ctx:
                read_tabular(buf, filename="test.csv")
            self.assertIn("exceeds the maximum supported limit", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Finding M: CSV Delimiter detection
    # -------------------------------------------------------------------------
    def test_finding_m_csv_delimiter_detection(self):
        """Delimiter detection supports ;, tab, pipe, while preserving 1-col CSVs."""
        # Semicolon
        self.assertEqual(_detect_csv_delimiter("col1;col2;col3\n1;2;3"), ";")
        # Tab
        self.assertEqual(_detect_csv_delimiter("col1\tcol2\tcol3\n1\t2\t3"), "\t")
        # Pipe
        self.assertEqual(_detect_csv_delimiter("col1|col2|col3\n1|2|3"), "|")
        # Standard comma
        self.assertEqual(_detect_csv_delimiter("col1,col2,col3\n1,2,3"), ",")
        # Legitimate 1-column comma CSV
        self.assertEqual(_detect_csv_delimiter("single_column\nval1\nval2"), ",")
        # Quoted comma within single column
        self.assertEqual(_detect_csv_delimiter('"quoted, comma, value"\n"another, one"'), ",")

    # -------------------------------------------------------------------------
    # Finding J: Bounded JSON streaming
    # -------------------------------------------------------------------------
    def test_finding_j_bounded_json_streaming(self):
        """JSON reader reads bounded rows and fails closed on unstreamable large payloads."""
        # Valid JSON array
        valid_json = io.BytesIO(json.dumps([{"a": i, "b": i * 2} for i in range(100)]).encode("utf-8"))
        df = read_tabular(valid_json, filename="data.json", max_rows=10)
        self.assertEqual(len(df), 10)

    # -------------------------------------------------------------------------
    # Finding K: get_dataset_row_count behavior
    # -------------------------------------------------------------------------
    def test_finding_k_get_dataset_row_count_cheap(self):
        """get_dataset_row_count returns cheap line count for CSV, None if not cheap."""
        with patch("Utils.paths.DATASETS_DIR", self.test_path):
            csv_path = self.test_path / "test.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            self.assertEqual(get_dataset_row_count("test.csv"), 2)

    # -------------------------------------------------------------------------
    # Finding L: Source documents vs Ready datasets
    # -------------------------------------------------------------------------
    def test_finding_l_source_documents_vs_ready_datasets(self):
        """list_ready_datasets separates tabular data from source documents."""
        datasets_dir = self.test_path / "Datasets"
        datasets_dir.mkdir()
        (datasets_dir / "table.csv").write_text("a\n1", encoding="utf-8")
        (datasets_dir / "document.pdf").write_bytes(b"%PDF-1.4")
        (datasets_dir / "notes.txt").write_text("notes", encoding="utf-8")

        with patch("Utils.paths.DATASETS_DIR", datasets_dir):
            ready = list_ready_datasets()
            sources = list_source_documents()
            tabular_files = list_dataset_files(tabular_only=True)

            self.assertIn("table.csv", ready)
            self.assertNotIn("document.pdf", ready)
            self.assertNotIn("notes.txt", ready)

            self.assertIn("document.pdf", sources)
            self.assertIn("notes.txt", sources)
            self.assertNotIn("table.csv", sources)

            self.assertEqual(ready, tabular_files)

    # -------------------------------------------------------------------------
    # Finding N: AI session state LRU eviction
    # -------------------------------------------------------------------------
    def test_finding_n_ai_context_lru_eviction(self):
        """AI chat context evicts oldest conversation when exceeding MAX_AI_CONTEXTS."""
        from Utils.Gemini import prune_ai_contexts, MAX_AI_CONTEXTS

        state = {}
        # Add 15 contexts
        for i in range(15):
            prune_ai_contexts(state, f"ctx_{i}")

        lru = state.get("_ai_context_lru", [])
        self.assertEqual(len(lru), MAX_AI_CONTEXTS)
        self.assertNotIn("ctx_0", lru)
        self.assertIn("ctx_14", lru)

    # -------------------------------------------------------------------------
    # Finding O: Storage cleanup contract
    # -------------------------------------------------------------------------
    def test_finding_o_storage_cleanup_contract(self):
        """cleanup_storage preserves files containing .tmp. and purges true tmp files."""
        storage_dir = self.test_path / "storage"
        storage_dir.mkdir()

        # Legitimate file containing .tmp.
        legit = storage_dir / "report.tmp.data.csv"
        legit.write_text("data", encoding="utf-8")

        # True temporary files matching regex contracts
        temp1 = storage_dir / "dataset.csv.tmp.1234abcd"
        temp1.write_text("temp", encoding="utf-8")
        temp2 = storage_dir / ".file.csv.reserve"
        temp2.write_text("lock", encoding="utf-8")

        cleanup_storage(directory=storage_dir, include_temp_only=True)

        self.assertTrue(legit.exists(), "Legitimate filename containing .tmp. must be preserved")
        self.assertFalse(temp1.exists(), "True temporary file must be purged")
        self.assertFalse(temp2.exists(), "True reserve lock file must be purged")

    # -------------------------------------------------------------------------
    # Finding P: Safe filename stem
    # -------------------------------------------------------------------------
    def test_finding_p_safe_stem_windows_reserved_and_length(self):
        """safe_stem handles Windows reserved names, component lengths, and path traversal."""
        # Windows reserved names
        self.assertEqual(safe_stem("CON"), "CON_safe")
        self.assertEqual(safe_stem("prn"), "prn_safe")
        self.assertEqual(safe_stem("aux"), "aux_safe")
        self.assertEqual(safe_stem("nul"), "nul_safe")
        self.assertEqual(safe_stem("COM1"), "COM1_safe")
        self.assertEqual(safe_stem("lpt1"), "lpt1_safe")

        # Path traversal stripped
        self.assertEqual(safe_stem("../../evil_name"), "evil_name")

        # Long names bounded to 100 characters
        long_stem = safe_stem("a" * 200)
        self.assertLessEqual(len(long_stem), 100)

    # -------------------------------------------------------------------------
    # Finding Q: Report template schema validation
    # -------------------------------------------------------------------------
    def test_finding_q_template_schema_validation(self):
        """validate_report_template parses string booleans and rejects invalid types."""
        # String 'false' should parse to False, not bool('false') == True
        config = {
            "title": "Quarterly Report",
            "author": "Analyst",
            "include_charts": "false",
            "include_ai": "0",
        }
        validated = validate_report_template(config)
        self.assertFalse(validated["include_charts"])
        self.assertFalse(validated["include_ai"])

        # True boolean values
        config_true = {
            "title": "Quarterly Report",
            "author": "Analyst",
            "include_charts": "true",
            "include_ai": 1,
        }
        validated_true = validate_report_template(config_true)
        self.assertTrue(validated_true["include_charts"])
        self.assertTrue(validated_true["include_ai"])

        # Non-dict rejected
        with self.assertRaises(ValueError):
            validate_report_template(["not", "a", "dict"])

    # -------------------------------------------------------------------------
    # Findings R & S: Model metadata & Category drift
    # -------------------------------------------------------------------------
    def test_findings_r_and_s_metadata_and_category_drift(self):
        """Model training captures metadata and inference detects unseen category drift."""
        df_train = pd.DataFrame({
            "category": ["A", "B", "A", "B", "A", "B"],
            "numeric": [10, 20, 15, 25, 12, 22],
            "target": [0, 1, 0, 1, 0, 1],
        })

        res = train_and_evaluate_model(
            df=df_train,
            target_col="target",
            feature_cols=["category", "numeric"],
            model_name="Logistic Regression",
            problem_type="Classification",
            random_state=42,
        )

        # Check metadata fields in training result
        self.assertIn("hyperparameters", res)
        self.assertIn("created_at", res)

        saved_path = save_trained_model(res, "drift_test_model", directory=self.test_path)
        bundle = load_trained_model(saved_path, allowed_dir=self.test_path)

        self.assertIn("training_config", bundle)
        self.assertIn("python_version", bundle)
        self.assertIn("pandas_version", bundle)
        self.assertIn("numpy_version", bundle)

        # Inference on unseen category 'C'
        df_test = pd.DataFrame({
            "category": ["A", "C"],
            "numeric": [14, 28],
        })

        pred_df = predict_with_model(bundle, df_test)
        self.assertIn("prediction", pred_df.columns)
        self.assertIn("category_drift", pred_df.attrs)
        self.assertIn("category", pred_df.attrs["category_drift"])
        self.assertIn("C", pred_df.attrs["category_drift"]["category"])

    # -------------------------------------------------------------------------
    # Rule 3: Formula sanitization boundary
    # -------------------------------------------------------------------------
    def test_formula_sanitization_boundary(self):
        """Internal analytical data preserves formulas; export sanitizes them."""
        df = pd.DataFrame({
            "code": ["=SUM(A1:A2)", "-123", "+456", "@user", "-calc()", "regular"],
        })

        # Internal analytical dataset must preserve exact raw strings
        self.assertEqual(df["code"].iloc[0], "=SUM(A1:A2)")
        self.assertEqual(df["code"].iloc[1], "-123")
        self.assertEqual(df["code"].iloc[4], "-calc()")

        # Export boundary escapes non-numeric formula triggers with leading quote
        exported_df = sanitize_for_csv_export(df)
        self.assertEqual(exported_df["code"].iloc[0], "'=SUM(A1:A2)")
        self.assertEqual(exported_df["code"].iloc[1], "-123")  # Numeric literal preserved
        self.assertEqual(exported_df["code"].iloc[2], "+456")  # Numeric literal preserved
        self.assertEqual(exported_df["code"].iloc[3], "'@user")
        self.assertEqual(exported_df["code"].iloc[4], "'-calc()")
        self.assertEqual(exported_df["code"].iloc[5], "regular")

    # -------------------------------------------------------------------------
    # Finding X: Plotly HTML escaping
    # -------------------------------------------------------------------------
    def test_finding_x_plotly_html_escaping(self):
        """Malicious column names are safely escaped in Plotly chart titles."""
        malicious_col = "<script>alert('xss')</script>"
        df = pd.DataFrame({malicious_col: [1, 2, 3, 4, 5]})
        fig = create_histogram_plot(df, malicious_col)
        self.assertNotIn("<script>", fig.layout.title.text)
        self.assertIn(html.escape(malicious_col), fig.layout.title.text)

    # -------------------------------------------------------------------------
    # Shared Sampling Policy (Findings G, H, I, Y)
    # -------------------------------------------------------------------------
    def test_shared_sampling_policy(self):
        """sample_for_visualization and sample_for_analysis bound large dataframes."""
        large_df = pd.DataFrame({"x": range(100_000)})
        vis_sampled, is_vis_sampled, _ = sample_for_visualization(large_df, max_rows=25_000)
        self.assertTrue(is_vis_sampled)
        self.assertEqual(len(vis_sampled), 25_000)

        analysis_sampled, is_ana_sampled, _ = sample_for_analysis(large_df, max_rows=50_000)
        self.assertTrue(is_ana_sampled)
        self.assertEqual(len(analysis_sampled), 50_000)

        # Small dataframe preserved exactly
        small_df = pd.DataFrame({"x": range(100)})
        small_sampled, is_small_sampled, _ = sample_for_visualization(small_df)
        self.assertFalse(is_small_sampled)
        self.assertEqual(len(small_sampled), 100)

    @patch("Utils.Gemini._generate_content")
    def test_rule_4_prompt_professionalism(self, mock_gen):
        """AI prompt generation excludes persona fluff and follows evidence-based structure."""
        df = pd.DataFrame({"metric": [1, 2, 3], "target": [10, 20, 30]})
        generate_executive_insights(api_key="mock_key", df=df, dataset_name="test.csv")
        self.assertTrue(mock_gen.called)
        prompt = mock_gen.call_args[0][2]
        self.assertNotIn("Chief Data Scientist", prompt)
        self.assertNotIn("elite AI", prompt)
        self.assertNotIn("Executive Intelligence Report", prompt)
        self.assertIn("Observed Facts", prompt)
        self.assertIn("Derived Statistics", prompt)

    # -------------------------------------------------------------------------
    # Finding 2.1 & 2.2: Alternate data stream filename rejection
    # -------------------------------------------------------------------------
    def test_finding_2_1_alternate_stream_filename_rejected(self):
        """Model loader rejects filenames with alternate data stream colons."""
        with self.assertRaises(ValueError) as ctx:
            load_trained_model(self.test_path / "model.joblib:stream", allowed_dir=self.test_path)
        self.assertIn("Invalid character ':'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Finding 2.5: S3 upload limits & export boundary
    # -------------------------------------------------------------------------
    def test_finding_2_5_s3_upload_limits_and_export_boundary(self):
        """upload_s3_dataset bounds memory/upload size and separates canonical vs export sanitization."""
        mock_client = MagicMock()
        df = pd.DataFrame({"formula": ["=SUM(A1:A2)", "-123"], "val": [10, 20]})

        # Canonical internal upload preserves exact values
        upload_s3_dataset(df, "mybucket", "canonical.csv", mock_client, sanitize=False)
        self.assertTrue(mock_client.put_object.called)
        uploaded_bytes = mock_client.put_object.call_args[1]["Body"]
        self.assertIn(b"=SUM(A1:A2)", uploaded_bytes)

        # External spreadsheet export sanitizes formula injection triggers
        mock_client.reset_mock()
        upload_s3_dataset(df, "mybucket", "export.csv", mock_client, sanitize=True)
        uploaded_bytes_sanitized = mock_client.put_object.call_args[1]["Body"]
        self.assertIn(b"'=SUM(A1:A2)", uploaded_bytes_sanitized)

        # Exceeding size ceiling fails closed before put_object
        mock_client.reset_mock()
        with self.assertRaises(ValueError) as ctx:
            upload_s3_dataset(df, "mybucket", "huge.csv", mock_client, max_bytes=10)
        self.assertIn("exceeds the maximum allowed upload limit", str(ctx.exception))
        self.assertFalse(mock_client.put_object.called)

    # -------------------------------------------------------------------------
    # Finding 2.13: Temporal preprocessing (no lookahead bias by default)
    # -------------------------------------------------------------------------
    def test_finding_2_13_temporal_preprocessing_no_lookahead(self):
        """fill_missing_values defaults to ffill for temporal data to avoid lookahead bias."""
        ts_df = pd.DataFrame({
            "ts": pd.to_datetime([None, "2026-01-02", None, "2026-01-04"]),
            "val": [1.0, 2.0, None, 4.0]
        })

        # Default "ffill" does NOT back-fill index 0 from future index 1
        filled_ffill = fill_missing_values(ts_df, datetime_strategy="ffill")
        self.assertTrue(pd.isna(filled_ffill["ts"].iloc[0]), "Initial timestamp must not be backfilled from the future")
        self.assertEqual(filled_ffill["ts"].iloc[2], pd.Timestamp("2026-01-02"), "Intermediate timestamp forward-filled")

        # Explicit "ffill_bfill" fills all when requested
        filled_bfill = fill_missing_values(ts_df, datetime_strategy="ffill_bfill")
        self.assertFalse(filled_bfill["ts"].isna().any())

        # "none" leaves datetime missingness untouched
        filled_none = fill_missing_values(ts_df, datetime_strategy="none")
        self.assertTrue(pd.isna(filled_none["ts"].iloc[0]))
        self.assertTrue(pd.isna(filled_none["ts"].iloc[2]))

        # Invalid strategy raises ValueError
        with self.assertRaises(ValueError):
            fill_missing_values(ts_df, datetime_strategy="invalid_strat")

    # -------------------------------------------------------------------------
    # Finding 2.17: Template version in schema
    # -------------------------------------------------------------------------
    def test_finding_2_17_template_version_in_schema(self):
        """validate_report_template includes explicit template_version."""
        validated = validate_report_template({"title": "Test Report"})
        self.assertEqual(validated.get("template_version"), 1)

    # -------------------------------------------------------------------------
    # Finding 2.25: Benjamini-Hochberg FDR correction in drift analysis
    # -------------------------------------------------------------------------
    def test_finding_2_25_benjamini_hochberg_correction_math(self):
        """benjamini_hochberg_correction produces valid monotonic FDR adjusted p-values."""
        pvals = [0.001, 0.04, 0.02, 0.50]
        adj = benjamini_hochberg_correction(pvals)
        self.assertEqual(len(adj), 4)
        self.assertTrue(all(0.0 <= q <= 1.0 for q in adj))
        # Monotonicity test: sorted raw p-values must have sorted adjusted q-values
        sorted_pairs = sorted(zip(pvals, adj))
        sorted_q = [q for _, q in sorted_pairs]
        self.assertTrue(all(sorted_q[i] <= sorted_q[i + 1] for i in range(len(sorted_q) - 1)))

    def test_finding_2_25_compare_drift_rows_includes_fdr_qval(self):
        """column_drift_rows includes KS p-adj (FDR) field."""
        df_a = pd.DataFrame({"feat": np.random.RandomState(42).normal(0, 1, 50)})
        df_b = pd.DataFrame({"feat": np.random.RandomState(42).normal(5, 1, 50)})
        rows = column_drift_rows(df_a, df_b)
        self.assertIn("KS p-adj (FDR)", rows[0])
        self.assertIsNotNone(rows[0]["KS p-adj (FDR)"])

    # -------------------------------------------------------------------------
    # Finding 2.26: ML Time-Series Split & Temporal Warning
    # -------------------------------------------------------------------------
    def test_finding_2_26_ml_cross_validation_temporal_awareness(self):
        """train_and_evaluate_model detects temporal structure and supports TimeSeriesSplit."""
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "feature": np.arange(30, dtype=float),
            "target": np.arange(30, dtype=float) * 2 + 1,
        })

        # Shuffled CV on temporal data surfaces warning
        res_shuffled = train_and_evaluate_model(
            df=df,
            target_col="target",
            feature_cols=["date", "feature"],
            model_name="Linear Regression",
            problem_type="Regression",
            time_series_cv=False,
        )
        self.assertIsNotNone(res_shuffled.get("temporal_warning"))
        self.assertIn("Temporal data detected", res_shuffled["temporal_warning"])
        self.assertEqual(res_shuffled.get("cv_strategy"), "KFold")

        # TimeSeriesSplit suppresses IID warning and uses TimeSeriesSplit
        res_ts = train_and_evaluate_model(
            df=df,
            target_col="target",
            feature_cols=["date", "feature"],
            model_name="Linear Regression",
            problem_type="Regression",
            time_series_cv=True,
        )
        self.assertIsNone(res_ts.get("temporal_warning"))
        self.assertEqual(res_ts.get("cv_strategy"), "TimeSeriesSplit")

    # -------------------------------------------------------------------------
    # Finding 3.11: Centralized semantic design tokens
    # -------------------------------------------------------------------------
    def test_finding_3_11_semantic_design_tokens(self):
        """Utils.theme exposes structured semantic color tokens and chart palettes."""
        tokens_light = get_semantic_tokens(dark=False)
        tokens_dark = get_semantic_tokens(dark=True)
        self.assertIn("accent", tokens_light)
        self.assertIn("success", tokens_light)
        self.assertIn("danger", tokens_light)
        self.assertIn("accent", tokens_dark)
        self.assertEqual(len(CHART_PALETTE), 8)
        self.assertEqual(SEMANTIC_TOKENS["accent"], "#2563EB")


if __name__ == "__main__":
    unittest.main()

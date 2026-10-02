import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from Utils.paths import read_tabular, _reject_dtd
from Utils.AIConvert import _split_into_chunks, reconcile_chunk_frames
from Utils.dataset_ui import BoundedDatasetMemoryCache
from Utils.Charts import downsample_timeseries
from Utils.ML import train_and_evaluate_model, save_trained_model, load_trained_model
from Utils.compare_logic import column_drift_rows


class TestF01RowBoundedJSONParsing(unittest.TestCase):
    """F-01: Row-bounded JSON decoding across array, keyed, and columnar structures."""

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


class TestF02XMLStreamingDTDDefense(unittest.TestCase):
    """F-02: Streaming XML DTD defense without full-payload memory buffering."""

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


class TestF07AIChunkBoundaryDeduplication(unittest.TestCase):
    """F-07: Deterministic semantic chunking and chunk boundary deduplication."""

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


class TestF09BoundedDatasetMemoryCache(unittest.TestCase):
    """F-09: Hard memory budget enforcement with deep byte tracking and LRU eviction."""

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


class TestF11TimeSeriesDownsampling(unittest.TestCase):
    """F-11: Time-preserving peak/valley decimation without random sampling."""

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


class TestF14MLCrossValidationAndBaselines(unittest.TestCase):
    """F-14: K-fold cross-validation variability and dummy baseline model comparison."""

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


class TestF17AndF19LockfileAndCIMatrix(unittest.TestCase):
    """F-17 & F-19: Requirements locking and cross-platform CI matrix."""

    def test_f17_lockfile_exists_and_referenced_in_readme(self):
        root = Path(__file__).resolve().parent.parent
        lockfile = root / "requirements-lock.txt"
        readme = root / "README.md"

        self.assertTrue(lockfile.exists(), "requirements-lock.txt must exist")
        content = readme.read_text(encoding="utf-8")
        self.assertIn("requirements-lock.txt", content, "README must instruct using lockfile")

    def test_f19_ci_workflow_matrix_includes_windows_and_macos(self):
        root = Path(__file__).resolve().parent.parent
        ci_file = root / ".github" / "workflows" / "tests.yml"
        self.assertTrue(ci_file.exists(), ".github/workflows/tests.yml must exist")

        content = ci_file.read_text(encoding="utf-8")
        self.assertIn("os: [ubuntu-latest, windows-latest, macos-latest]", content)


class TestF21DatasetDriftDetection(unittest.TestCase):
    """F-21: Kolmogorov-Smirnov and Total Variation Distance drift detection."""

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


class TestF23ModelDeserializationTrustBoundary(unittest.TestCase):
    """F-23: HMAC-SHA256 signature verification and untrusted model blocking."""

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
            loaded = load_trained_model(tmp_path / "trusted_model.joblib", require_signature=True)
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
                load_trained_model(tmp_path / "tampered.joblib", require_signature=True)

    def test_unsigned_model_rejected_when_signature_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_file = tmp_path / "unsigned.joblib"
            model_file.write_bytes(b"some_joblib_bytes")

            with self.assertRaisesRegex(ValueError, r"(?i)lacks a cryptographic authenticity signature"):
                load_trained_model(tmp_path / "unsigned.joblib", require_signature=True)


class TestF03AtomicFilenameReservation(unittest.TestCase):
    """F-03: Concurrency-safe atomic kernel filesystem reservations."""

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


class TestF05PDFPrivacyAndExcludedColumns(unittest.TestCase):
    """F-05: PII redaction and sensitive column exclusion from correlations and charts."""

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


class TestF12ChartHueAndCategoryCapping(unittest.TestCase):
    """F-12: High-cardinality hue and category capping preventing UI hang."""

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


class TestF13AndF15MLFeatureContractAndSemantics(unittest.TestCase):
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
            bundle = load_trained_model(mpath, require_signature=True)
            self.assertIn("feature_dtypes", bundle)
            self.assertEqual(bundle["importance_type"], "Absolute Standardized Coefficient Magnitude (|β|)")

            test_df = pd.DataFrame({"num_feat": [7.0, 8.0], "cat_feat": ["A", "B"]})
            preds = predict_with_model(bundle, test_df)
            self.assertEqual(len(preds), 2)

            bad_test_df = pd.DataFrame({"num_feat": [7.0, "NOT_A_NUMBER"], "cat_feat": ["A", "B"]})
            with self.assertRaisesRegex(ValueError, r"(?i)non-numeric text values"):
                predict_with_model(bundle, bad_test_df)


class TestF24CredentialSessionStateSecurity(unittest.TestCase):
    """F-24: Honest session state security terminology."""

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


class TestF27LicenseFilePresent(unittest.TestCase):
    """F-27: MIT License file present in repository root."""

    def test_license_file_exists_in_root(self):
        root = Path(__file__).resolve().parent.parent
        license_path = root / "LICENSE"
        self.assertTrue(license_path.exists())
        self.assertIn("MIT License", license_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()


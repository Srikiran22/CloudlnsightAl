"""Regression tests for audit findings and implementation fixes."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import pandas as pd

from Utils.ML import (
    train_and_evaluate_model,
    save_trained_model,
    load_trained_model,
)
from Utils.privacy import (
    detect_sensitive_columns,
    detect_sensitive_text,
)
from Utils.secrets import drop
from Utils.Gemini import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    resolve_gemini_model,
)


class TestAuditRegressions(unittest.TestCase):
    """Regression test cases verifying audit claims and fixes."""

    def test_numeric_target_confusion_matrix_counts(self):
        """P1: Confusion matrix must count predictions for integer targets (0/1), not all zeros."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.random.normal(0, 1, n),
            "feat2": np.random.normal(2, 1, n),
            "churned": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        })
        results = train_and_evaluate_model(
            df=df,
            target_col="churned",
            feature_cols=["feat1", "feat2"],
            model_name="Logistic Regression",
            problem_type="Classification",
            random_state=42,
        )
        cm = results["confusion_matrix"]
        self.assertIsNotNone(cm)
        self.assertEqual(cm.sum(), len(results["y_test"]))
        self.assertGreater(cm.sum(), 0)
        # Confusion matrix must not be all zeros
        self.assertTrue((cm > 0).any(), "Confusion matrix for integer targets must contain non-zero counts")

    def test_string_target_confusion_matrix_counts(self):
        """P1: Confusion matrix counts predictions for string target labels."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.random.normal(0, 1, n),
            "feat2": np.random.normal(2, 1, n),
            "status": np.random.choice(["active", "churned"], size=n, p=[0.7, 0.3]),
        })
        results = train_and_evaluate_model(
            df=df,
            target_col="status",
            feature_cols=["feat1", "feat2"],
            model_name="Logistic Regression",
            problem_type="Classification",
            random_state=42,
        )
        cm = results["confusion_matrix"]
        self.assertEqual(cm.sum(), len(results["y_test"]))
        self.assertTrue((cm > 0).any())

    def test_positive_class_metrics_computed_for_binary_classification(self):
        """P1: Binary classification returns positive-class precision, recall, F1, and support."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "feat1": np.random.normal(0, 1, n),
            "churned": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        })
        results = train_and_evaluate_model(
            df=df,
            target_col="churned",
            feature_cols=["feat1"],
            model_name="Logistic Regression",
            problem_type="Classification",
            random_state=42,
        )
        self.assertIn("positive_class", results)
        self.assertEqual(results["positive_class"], "1")
        self.assertIn("positive_precision", results)
        self.assertIn("positive_recall", results)
        self.assertIn("positive_f1", results)
        self.assertIn("positive_support", results)
        self.assertGreaterEqual(results["positive_support"], 0)

    def test_saved_model_preserves_target_column_metadata(self):
        """P1: Target variable metadata is saved as both target_col and target_column."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "feat": np.random.normal(0, 1, n),
            "churned": np.random.choice([0, 1], size=n),
        })
        results = train_and_evaluate_model(
            df=df,
            target_col="churned",
            feature_cols=["feat"],
            model_name="Decision Tree",
            problem_type="Classification",
            random_state=42,
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = save_trained_model(results, "test_churn_model", directory=tmpdir)
            bundle = load_trained_model(model_path, allowed_dir=tmpdir)
            self.assertEqual(bundle.get("target_column"), "churned")
            self.assertEqual(bundle.get("target_col"), "churned")

    def test_privacy_flags_customer_id_and_other_identifiers(self):
        """P1: Privacy screening detects customer_id, user_id, uuid, etc."""
        df = pd.DataFrame({
            "customer_id": ["CUST-001", "CUST-002"],
            "user_id": [101, 102],
            "client_uuid": ["abc-123", "def-456"],
            "monthly_spend": [100.5, 200.0],
        })
        flags = detect_sensitive_columns(df)
        self.assertIn("customer_id", flags)
        self.assertIn("user_id", flags)
        self.assertIn("client_uuid", flags)
        self.assertNotIn("monthly_spend", flags)

    def test_privacy_ignores_timestamps_in_free_text(self):
        """P1: Free-text phone detector excludes timestamps from triggering false positives."""
        sample_log = (
            "[2026-03-01 10:14:22] Host: srv-app-01 | Tier: Enterprise | Event: CPU_SPIKE\n"
            "[2026-03-01 10:15:08] Host: srv-db-04  | Tier: Enterprise | Event: IO_LATENCY\n"
        )
        findings = detect_sensitive_text(sample_log)
        self.assertEqual(findings, [], "Timestamps in server logs must not trigger phone number warnings")

        text_with_phone = "Contact customer support at +1 (555) 234-5678 immediately."
        findings_phone = detect_sensitive_text(text_with_phone)
        self.assertIn("phone-number-like values", findings_phone)

    def test_secret_clearing_rotates_inline_widget_key(self):
        """P1: drop() and release() rotate widget key so subsequent ask() renders with clean version."""
        state = {"gemini_secret": "secret_api_key_123", "gemini_ver": 0}
        with patch("Utils.secrets.st", SimpleNamespace(session_state=state)):
            drop("gemini")
            self.assertNotIn("gemini_secret", state)
            self.assertEqual(state.get("gemini_ver"), 1)

    def test_csv_export_sanitization(self):
        """P1 & P2: sanitize_for_csv_export neutralizes formula injection for exports."""
        from Utils.paths import sanitize_for_csv_export
        df = pd.DataFrame({"col": ["=SUM(A1:A2)", "clean_val"]})
        sanitized = sanitize_for_csv_export(df)
        csv_text = sanitized.to_csv(index=False)
        self.assertIn("'=SUM(A1:A2)", csv_text)

    def test_gemini_registry_and_deprecation(self):
        """Gemini registry maintains gemini-3.8-flash, gemini-3.6-flash, and maps gemini-3.5-flash."""
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.8-flash")
        self.assertIn("gemini-3.8-flash", GEMINI_MODELS)
        self.assertIn("gemini-3.6-flash", GEMINI_MODELS)
        self.assertEqual(resolve_gemini_model("gemini-3.5-flash"), "gemini-3.6-flash")

    def test_below_baseline_verdict_rendered_in_ml_page(self):
        """P1: Pages/ML.py must display Below baseline warning when accuracy < majority class baseline."""
        from streamlit.testing.v1 import AppTest
        from Utils.dataset_ui import dataframe_fingerprint
        ml_script = str(Path(__file__).resolve().parent.parent / "Pages" / "ML.py")
        at = AppTest.from_file(ml_script, default_timeout=30)
        df = pd.DataFrame({
            "feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "churned": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        })
        at.session_state["dataset_name"] = "test_churn.csv"
        at.session_state["current_df"] = df
        at.session_state["ml_results"] = {
            "model_name": "Random Forest",
            "problem_type": "Classification",
            "target_column": "churned",
            "dataset_name": "test_churn.csv",
            "dataset_fingerprint": dataframe_fingerprint(df),
            "source_type": "session",
            "feature_cols": ["feat"],
            "accuracy": 0.65,
            "baseline_accuracy": 0.80,
            "precision": 0.60,
            "recall": 0.65,
            "f1_score": 0.62,
            "classes": ["0", "1"],
            "confusion_matrix": np.array([[6, 2], [1, 1]]),
        }
        at.run()
        self.assertEqual(len(at.exception), 0)
        warning_texts = [w.value for w in at.warning]
        self.assertTrue(
            any("Below baseline performance" in t for t in warning_texts),
            f"Expected 'Below baseline performance' warning, got: {warning_texts}",
        )

    @patch("Pages.Predict.list_saved_models")
    @patch("Pages.Predict.load_trained_model")
    def test_predict_displays_actual_target_column(self, mock_load, mock_list):
        """P1: Pages/Predict.py must display actual target name instead of generic 'Target'."""
        from streamlit.testing.v1 import AppTest
        mock_list.return_value = [("test_model.joblib", "Trusted")]
        mock_load.return_value = {
            "model_name": "Random Forest",
            "problem_type": "Classification",
            "target_col": "churned",
            "target_column": "churned",
            "feature_cols": ["feat"],
            "metrics": {"accuracy": 0.85},
            "created_at": "2026-10-16",
        }
        predict_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Predict.py")
        at = AppTest.from_file(predict_script, default_timeout=30)
        at.session_state["dataset_name"] = "test.csv"
        at.session_state["current_df"] = pd.DataFrame({"feat": [1, 2], "churned": [0, 1]})
        at.run()
        self.assertEqual(len(at.exception), 0)
        target_metrics = [m for m in at.metric if getattr(m, "label", "") == "Target Variable"]
        self.assertTrue(len(target_metrics) > 0, "Target Variable metric not found")
        self.assertEqual(target_metrics[0].value, "churned")

    def test_prediction_invalidation_on_content_change_without_shape_change(self):
        """P1: Altering a single cell value without changing shape or filename must change the prediction fingerprint and invalidate cached predictions."""
        from Utils.dataset_ui import dataframe_fingerprint
        df_orig = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]})
        fp_orig = dataframe_fingerprint(df_orig)

        # Mutate a single value without changing shape, columns, or dtypes
        df_mutated = df_orig.copy()
        df_mutated.loc[0, "feat"] = 999.0
        self.assertEqual(df_orig.shape, df_mutated.shape)
        self.assertEqual(list(df_orig.columns), list(df_mutated.columns))

        fp_mutated = dataframe_fingerprint(df_mutated)
        self.assertNotEqual(
            fp_orig,
            fp_mutated,
            "Changing cell value in dataframe must produce a different cryptographic fingerprint",
        )

        # Simulate Predict.py scoring ID matching
        model_name = "test_model.joblib"
        source_name = "active_dataset"
        scoring_id_orig = f"{model_name}:fp::{source_name}:{fp_orig}"
        scoring_id_mutated = f"{model_name}:fp::{source_name}:{fp_mutated}"
        self.assertNotEqual(scoring_id_orig, scoring_id_mutated)

    def test_style_plotly_fig_dark_theme(self):
        """P1: style_plotly_fig applies dark template and transparent background in dark mode."""
        import plotly.express as px
        from Utils.theme import style_plotly_fig
        fig = px.scatter(x=[1, 2], y=[3, 4])
        styled = style_plotly_fig(fig)
        self.assertEqual(styled.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(styled.layout.plot_bgcolor, "rgba(0,0,0,0)")
        self.assertIsNotNone(styled.layout.template)

    def test_theme_defaults_to_dark_base(self):
        """P1: Default theme base is dark mode."""
        from Utils.theme import current_theme_base, is_dark
        self.assertEqual(current_theme_base(), "dark")
        self.assertTrue(is_dark())

    def test_ai_converter_privacy_notice_wording(self):
        """P2: AIConverter.py contains explicit Google Gemini transmission notice and heuristic screening disclaimer."""
        p = Path(__file__).resolve().parent.parent / "Pages" / "AIConverter.py"
        content = p.read_text(encoding="utf-8")
        self.assertIn("Selected source content is sent to Google Gemini for extraction", content)
        self.assertIn("Privacy screening uses heuristics and cannot guarantee that all sensitive data will be detected", content)
        self.assertNotIn("Privacy note: Content is sent securely to Google Gemini for tabular parsing.", content)

    def test_semantic_alert_styling_rules(self):
        """P2: theme.py defines targeted semantic colors for amber warning, teal success, blue info, and red error."""
        p = Path(__file__).resolve().parent.parent / "Utils" / "theme.py"
        content = p.read_text(encoding="utf-8")
        self.assertIn("stAlertContentWarning", content)
        self.assertIn("#F59E0B", content)  # Amber warning
        self.assertIn("stAlertContentSuccess", content)
        self.assertIn("#14B8A6", content)  # Teal success
        self.assertIn("stAlertContentInfo", content)
        self.assertIn("#3B82F6", content)  # Sky blue info
        self.assertIn("stAlertContentError", content)
        self.assertIn("#EF4444", content)  # Crimson error

    @patch("Utils.ML.list_saved_models")
    @patch("Utils.ML.load_trained_model")
    def test_predict_apptest_renders_overlap_warning(self, mock_load, mock_list):
        """P1: Predict.py displays training-data overlap warning when scoring dataset matches training fingerprint."""
        from streamlit.testing.v1 import AppTest
        from Utils.dataset_ui import dataframe_fingerprint

        mock_list.return_value = [("test_model.joblib", "Trusted")]
        scoring_df = pd.DataFrame({"feat": [1, 2, 3], "churned": [0, 1, 0]})
        df_fp = dataframe_fingerprint(scoring_df)

        mock_load.return_value = {
            "model_name": "Random Forest",
            "problem_type": "Classification",
            "target_col": "churned",
            "target_column": "churned",
            "feature_cols": ["feat"],
            "dataset_name": "train_data.csv",
            "dataset_fingerprint": df_fp,
            "metrics": {"accuracy": 0.85},
            "created_at": "2026-10-16",
        }
        predict_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Predict.py")
        at = AppTest.from_file(predict_script, default_timeout=30)
        at.session_state["dataset_name"] = "train_data.csv"
        at.session_state["current_df"] = scoring_df
        at.run()
        self.assertEqual(len(at.exception), 0)
        warning_texts = [w.value for w in at.warning]
        self.assertTrue(
            any("Training-data overlap detected" in wt for wt in warning_texts),
            f"Expected overlap warning in at.warning, got: {warning_texts}"
        )

    @patch("Utils.ML.list_saved_models")
    @patch("Utils.ML.load_trained_model")
    def test_predict_apptest_different_fingerprint_same_filename_renders_version_note(self, mock_load, mock_list):
        """Phase 2: Same filename with different content does not incorrectly imply identical training data."""
        from streamlit.testing.v1 import AppTest

        mock_list.return_value = [("test_model.joblib", "Trusted")]
        scoring_df = pd.DataFrame({"feat": [10, 20, 30], "churned": [1, 0, 1]})

        mock_load.return_value = {
            "model_name": "Random Forest",
            "problem_type": "Classification",
            "target_col": "churned",
            "target_column": "churned",
            "feature_cols": ["feat"],
            "dataset_name": "train_data.csv",
            "dataset_fingerprint": "different_hash_from_original_training_run",
            "metrics": {"accuracy": 0.85},
            "created_at": "2026-10-16",
        }
        predict_script = str(Path(__file__).resolve().parent.parent / "Pages" / "Predict.py")
        at = AppTest.from_file(predict_script, default_timeout=30)
        at.session_state["dataset_name"] = "train_data.csv"
        at.session_state["current_df"] = scoring_df
        at.run()
        self.assertEqual(len(at.exception), 0)
        # Must not claim exact training-data overlap
        warning_texts = [w.value for w in at.warning]
        self.assertFalse(any("Training-data overlap detected" in wt for wt in warning_texts))
        # Must display dataset version note
        info_texts = [i.value for i in at.info]
        self.assertTrue(
            any("Dataset version note" in it for it in info_texts),
            f"Expected 'Dataset version note' in at.info, got: {info_texts}"
        )

    def test_ai_request_boundary_excludes_sensitive_columns_and_values(self):
        """Phase 3: Outbound prompt construction strips excluded columns and values from request context."""
        from Utils.Gemini import get_dataset_summary_context
        from Utils.privacy import apply_exclusions

        df = pd.DataFrame({
            "customer_id": ["CUST_SECRET_999888", "CUST_SECRET_777666"],
            "ssn": ["000-12-3456", "000-65-4321"],
            "monthly_spend": [120.50, 240.00],
            "churned": [0, 1],
        })

        excluded = ["customer_id", "ssn"]
        filtered_df, ok = apply_exclusions(df, excluded)
        self.assertTrue(ok)
        self.assertNotIn("customer_id", filtered_df.columns)
        self.assertNotIn("ssn", filtered_df.columns)

        context = get_dataset_summary_context(filtered_df, "test_dataset.csv")

        # Confirm excluded column names do not appear in context
        self.assertNotIn("customer_id", context)
        self.assertNotIn("ssn", context)
        # Confirm distinctive sensitive values do not appear in context
        self.assertNotIn("CUST_SECRET_999888", context)
        self.assertNotIn("CUST_SECRET_777666", context)
        self.assertNotIn("000-12-3456", context)
        self.assertNotIn("000-65-4321", context)

        # Confirm non-excluded columns and values ARE present
        self.assertIn("monthly_spend", context)
        self.assertIn("churned", context)
        self.assertIn("120.5", context)

    def test_model_fingerprint_incorporates_hmac_signature(self):
        """Phase 2: Model cache fingerprint binds to HMAC-SHA256 signature when present."""
        import tempfile
        import Pages.Predict as predict_mod

        # Test with a mock signature file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "test_model.joblib"
            model_file.write_bytes(b"dummy_model_bytes")
            sig_file = model_file.with_suffix(".joblib.sig")
            sig_file.write_text('{"signature": "mock_hmac_abc123"}', encoding="utf-8")

            with patch.object(predict_mod, "MODELS_DIR", Path(tmpdir)):
                fp = predict_mod._model_fingerprint("test_model.joblib")
                self.assertEqual(fp, "hmac:mock_hmac_abc123")


if __name__ == "__main__":
    unittest.main()




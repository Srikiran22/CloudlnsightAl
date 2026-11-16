import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.ml import generate_standalone_inference_code, save_trained_model


class TestInferenceScriptGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name)

        X = pd.DataFrame({
            "age": [25, 45, 35, 50, 23, 40],
            "balance": [1000.0, 5000.0, 1500.0, 7000.0, 800.0, 3200.0]
        })
        y = np.array(["No", "Yes", "No", "Yes", "No", "Yes"])
        pipe = Pipeline([("clf", LogisticRegression())])
        pipe.fit(X, y)

        self.bundle = {
            "pipeline": pipe,
            "feature_cols": ["age", "balance"],
            "target_column": "churn",
            "problem_type": "Classification",
            "model_name": "Logistic Regression",
            "classes": ["No", "Yes"],
            "positive_class": "Yes",
            "decision_threshold": 0.60,
        }
        self.test_df = X

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_script_syntax_is_valid_python(self):
        code = generate_standalone_inference_code(self.bundle, model_rel_path="model.joblib")
        tree = ast.parse(code)
        self.assertIsNotNone(tree)
        self.assertIn("argparse", code)
        self.assertIn("joblib", code)
        self.assertIn("--input", code)
        self.assertIn("--output", code)
        self.assertIn("0.6", code)

    def test_standalone_script_executes_successfully(self):
        # Save model bundle
        model_path = save_trained_model(self.bundle, "test_churn_model", directory=self.path)

        # Save input data
        input_csv = self.path / "input.csv"
        self.test_df.to_csv(input_csv, index=False)
        output_csv = self.path / "output.csv"

        # Generate script pointing to saved model
        code = generate_standalone_inference_code(
            self.bundle,
            model_rel_path=model_path.name
        )
        script_path = self.path / "predict_standalone.py"
        script_path.write_text(code, encoding="utf-8")

        # Run script via subprocess
        res = subprocess.run(
            [sys.executable, str(script_path), "--input", str(input_csv), "--output", str(output_csv)],
            cwd=str(self.path),
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0, f"Script failed with: {res.stderr}\nOutput: {res.stdout}")
        self.assertTrue(output_csv.exists(), "Output CSV should have been created")

        scored_df = pd.read_csv(output_csv)
        self.assertEqual(len(scored_df), len(self.test_df))
        self.assertIn("prediction", scored_df.columns)

    def test_standalone_script_rejects_tampered_model(self):
        model_path = save_trained_model(self.bundle, "test_tamper_model", directory=self.path)
        input_csv = self.path / "input.csv"
        self.test_df.to_csv(input_csv, index=False)
        output_csv = self.path / "output_tamper.csv"

        code = generate_standalone_inference_code(self.bundle, model_rel_path=model_path.name)
        script_path = self.path / "predict_tamper.py"
        script_path.write_text(code, encoding="utf-8")

        # Tamper with the model file bytes
        with open(model_path, "ab") as f:
            f.write(b"CORRUPTED_BYTES")

        res = subprocess.run(
            [sys.executable, str(script_path), "--input", str(input_csv), "--output", str(output_csv)],
            cwd=str(self.path),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0, "Script should fail on tampered model")
        self.assertIn("verification failed", res.stderr.lower())

    def test_standalone_script_rejects_invalid_threshold(self):
        model_path = save_trained_model(self.bundle, "test_thresh_model", directory=self.path)
        input_csv = self.path / "input.csv"
        self.test_df.to_csv(input_csv, index=False)
        output_csv = self.path / "output_thresh.csv"

        code = generate_standalone_inference_code(self.bundle, model_rel_path=model_path.name)
        script_path = self.path / "predict_thresh.py"
        script_path.write_text(code, encoding="utf-8")

        res = subprocess.run(
            [sys.executable, str(script_path), "--input", str(input_csv), "--output", str(output_csv), "--threshold", "1.5"],
            cwd=str(self.path),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0, "Script should fail on threshold > 1.0")
        self.assertIn("must be strictly between", res.stderr)

if __name__ == "__main__":
    unittest.main()

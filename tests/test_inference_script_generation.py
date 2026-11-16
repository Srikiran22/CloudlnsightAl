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

if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.ml import evaluate_threshold_metrics, predict_with_model


class TestDecisionThreshold(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        y = np.array(["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"])
        pipe = Pipeline([("clf", LogisticRegression())])
        pipe.fit(X, y)
        self.bundle = {
            "pipeline": pipe,
            "feature_cols": ["feat1"],
            "target_column": "churn",
            "problem_type": "Classification",
            "classes": ["No", "Yes"],
            "positive_class": "Yes",
            "decision_threshold": 0.50
        }
        self.X_test = X

    def test_predict_with_default_threshold(self):
        preds = predict_with_model(self.bundle, self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_predict_with_threshold_override(self):
        preds_high = predict_with_model(self.bundle, self.X_test, threshold_override=0.95)
        preds_low = predict_with_model(self.bundle, self.X_test, threshold_override=0.10)
        yes_high = sum(1 for p in preds_high if p == "Yes")
        yes_low = sum(1 for p in preds_low if p == "Yes")
        self.assertLessEqual(yes_high, yes_low)

    def test_evaluate_threshold_metrics(self):
        y_true = np.array(["No", "No", "Yes", "Yes"])
        y_proba = np.array([0.1, 0.4, 0.6, 0.8])
        metrics = evaluate_threshold_metrics(y_true, y_proba, classes=["No", "Yes"], threshold=0.5, positive_class="Yes")
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("mcc", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mcc"], 1.0)

    def test_threshold_metrics_imbalanced_mcc(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
        metrics_high = evaluate_threshold_metrics(y_true, y_proba, classes=[0, 1], threshold=0.8, positive_class=1)
        metrics_low = evaluate_threshold_metrics(y_true, y_proba, classes=[0, 1], threshold=0.2, positive_class=1)
        self.assertIn("mcc", metrics_high)
        self.assertIn("mcc", metrics_low)
        self.assertIsInstance(metrics_high["mcc"], float)

import unittest

import numpy as np
import pandas as pd

from utils.eda_logic import compute_outlier_summary


class TestOutlierDiagnostics(unittest.TestCase):
    def setUp(self):
        # 100 normal values + 5 high outliers
        np.random.seed(42)
        normal_data = np.random.normal(loc=50.0, scale=5.0, size=100)
        outliers = np.array([120.0, 130.0, 140.0, 150.0, 160.0])
        combined = np.concatenate([normal_data, outliers])
        self.df = pd.DataFrame({
            "injected_feature": combined,
            "constant_feature": [10.0] * 105,
            "small_feature": [1.0, 2.0, float("nan")] + [float("nan")] * 102,
        })

    def test_compute_outlier_summary_identifies_injected_outliers(self):
        summary = compute_outlier_summary(self.df, columns=["injected_feature"])
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 1)

        row = summary.iloc[0]
        self.assertEqual(row["Feature"], "injected_feature")
        self.assertGreaterEqual(row["IQR Outlier Count"], 5)
        self.assertGreaterEqual(row["Z-Score Count"], 5)
        self.assertGreater(row["Skewness"], 1.0)
        self.assertIn("Right-skewed", row["Distribution Shape"])

    def test_compute_outlier_summary_handles_edge_cases(self):
        # Constant and sparse columns should not raise exceptions
        summary = compute_outlier_summary(self.df)
        self.assertIsInstance(summary, pd.DataFrame)
        # Empty dataframe
        empty_summary = compute_outlier_summary(pd.DataFrame())
        self.assertTrue(empty_summary.empty)

    def test_k_multiplier_sensitivity(self):
        summary_standard = compute_outlier_summary(self.df, columns=["injected_feature"], k=1.5)
        summary_extreme = compute_outlier_summary(self.df, columns=["injected_feature"], k=3.0)
        count_standard = summary_standard.iloc[0]["IQR Outlier Count"]
        count_extreme = summary_extreme.iloc[0]["IQR Outlier Count"]
        self.assertGreaterEqual(count_standard, count_extreme)

if __name__ == "__main__":
    unittest.main()

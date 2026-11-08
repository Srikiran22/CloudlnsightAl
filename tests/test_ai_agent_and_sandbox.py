"""Unit and regression tests for Sandboxed Code Execution and Autonomous Data Agent."""

import unittest

import numpy as np
import pandas as pd

from Utils.ai_agent import (
    apply_remediation_blueprint,
    generate_autonomous_remediation_blueprint,
    run_local_deterministic_query,
)
from Utils.ai_sandbox import execute_sandboxed_code


class TestAISandbox(unittest.TestCase):
    """Verify AST security boundaries and sandboxed execution."""

    def setUp(self):
        self.df = pd.DataFrame({
            "age": [25, 30, 35, 40, 100],  # 100 is outlier
            "salary": [50000.0, 60000.0, 75000.0, 90000.0, 150000.0],
            "dept": ["Eng", "HR", "Eng", "Sales", "Eng"],
        })

    def test_benign_aggregation_execution(self):
        code = "result = df.groupby('dept')['salary'].mean().to_dict()"
        res = execute_sandboxed_code(code, self.df)
        self.assertTrue(res.success)
        self.assertIsNone(res.error)
        self.assertIn("Eng", res.result)
        self.assertAlmostEqual(res.result["Eng"], 91666.666, places=1)

    def test_plotly_figure_creation(self):
        code = "fig = px.scatter(df, x='age', y='salary', color='dept')"
        res = execute_sandboxed_code(code, self.df)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.fig)
        self.assertTrue(hasattr(res.fig, "to_dict"))

    def test_security_import_rejected(self):
        code = "import os\nos.system('dir')"
        res = execute_sandboxed_code(code, self.df)
        self.assertFalse(res.success)
        self.assertIn("Security Policy Violation", res.error)

    def test_security_from_import_rejected(self):
        code = "from subprocess import Popen\nPopen(['dir'])"
        res = execute_sandboxed_code(code, self.df)
        self.assertFalse(res.success)
        self.assertIn("Security Policy Violation", res.error)

    def test_security_eval_exec_rejected(self):
        code = "eval('1 + 1')"
        res = execute_sandboxed_code(code, self.df)
        self.assertFalse(res.success)
        self.assertIn("Security Policy Violation", res.error)

    def test_security_dunder_lookup_rejected(self):
        code = "x = ().__class__.__subclasses__()"
        res = execute_sandboxed_code(code, self.df)
        self.assertFalse(res.success)
        self.assertIn("Security Policy Violation", res.error)

    def test_immutability_of_original_dataframe(self):
        code = "df['hacked'] = 123"
        res = execute_sandboxed_code(code, self.df)
        self.assertTrue(res.success)
        # Original DataFrame must remain untouched
        self.assertNotIn("hacked", self.df.columns)


class TestAIAgent(unittest.TestCase):
    """Verify autonomous deterministic data agent and remediation blueprint."""

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "age": [20, 22, 25, 27, 29, 30, 31, 35, 36, 120],  # 120 is outlier
            "income": [30000.0, 35000.0, 40000.0, 50000.0, 55000.0, 60000.0, 70000.0, 80000.0, 90000.0, 250000.0],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", None],  # Missing category
        })

    def test_local_query_correlation(self):
        res = run_local_deterministic_query(self.df, "Show correlation matrix")
        self.assertTrue(res.is_offline)
        self.assertIsNotNone(res.fig)
        self.assertIsNotNone(res.result_df)
        self.assertIn("Pearson Correlation", res.insights)

    def test_local_query_outliers(self):
        res = run_local_deterministic_query(self.df, "Find outliers in age")
        self.assertTrue(res.is_offline)
        self.assertIsNotNone(res.fig)
        self.assertIn("Outlier", res.insights)

    def test_local_query_distribution(self):
        res = run_local_deterministic_query(self.df, "What is the distribution of income?")
        self.assertTrue(res.is_offline)
        self.assertIsNotNone(res.fig)
        self.assertIn("Distribution", res.insights)

    def test_local_query_missingness(self):
        res = run_local_deterministic_query(self.df, "Analyze missing data")
        self.assertTrue(res.is_offline)
        self.assertIn("Missing", res.insights)

    def test_autonomous_remediation_blueprint(self):
        bp = generate_autonomous_remediation_blueprint(self.df, "test.csv")
        self.assertIn("category", bp.imputation_plan)
        self.assertEqual(bp.imputation_plan["category"], "mode")

        cleaned = apply_remediation_blueprint(self.df, bp)
        self.assertEqual(cleaned["category"].isna().sum(), 0)
        # Verify age outlier was clipped
        self.assertLess(cleaned["age"].max(), 120)


if __name__ == "__main__":
    unittest.main()

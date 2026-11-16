import os
import unittest


class TestPortfolioDocumentation(unittest.TestCase):
    def test_readme_contains_recruiter_and_viva_essentials(self):
        self.assertTrue(os.path.exists("README.md"), "README.md should exist")
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("TabuLens", content)
        self.assertIn("System Architecture", content)
        self.assertIn("Docker Quickstart", content)
        self.assertIn("METHODOLOGY.md", content)

    def test_methodology_doc_exists_and_detailed(self):
        self.assertTrue(os.path.exists("docs/METHODOLOGY.md"), "docs/METHODOLOGY.md should exist")
        with open("docs/METHODOLOGY.md", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Statistical & Machine Learning Methodology", content)
        self.assertIn("Kolmogorov-Smirnov", content)
        self.assertIn("Benjamini-Hochberg", content)
        self.assertIn("Variance Inflation Factor", content)
        self.assertIn("Decision Threshold Optimization", content)


if __name__ == "__main__":
    unittest.main()

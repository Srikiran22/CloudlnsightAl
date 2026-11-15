import os
import unittest

class TestPortfolioDocumentation(unittest.TestCase):
    def test_readme_contains_recruiter_and_viva_essentials(self):
        self.assertTrue(os.path.exists("README.md"), "README.md should exist")
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Business Impact", content)
        self.assertIn("System Architecture", content)
        self.assertIn("Baseline Benchmark", content)
        self.assertIn("Docker Quickstart", content)
        self.assertIn("CAPSTONE_THESIS_DEFENSE.md", content)

    def test_capstone_thesis_defense_doc_exists_and_detailed(self):
        self.assertTrue(os.path.exists("docs/CAPSTONE_THESIS_DEFENSE.md"), "docs/CAPSTONE_THESIS_DEFENSE.md should exist")
        with open("docs/CAPSTONE_THESIS_DEFENSE.md", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Capstone Defense Specification", content)
        self.assertIn("Mathematical Foundations", content)
        self.assertIn("Kolmogorov-Smirnov", content)
        self.assertIn("Benjamini-Hochberg", content)
        self.assertIn("Viva Defense Cheat Sheet", content)

if __name__ == "__main__":
    unittest.main()

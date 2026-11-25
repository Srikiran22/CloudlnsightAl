import os
import unittest
from pathlib import Path


class TestRepositoryDocumentation(unittest.TestCase):
    def test_readme_covers_core_sections_and_links(self):
        self.assertTrue(os.path.exists("README.md"), "README.md should exist")
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("TabuLens", content)
        self.assertIn("System Architecture", content)
        self.assertIn("Docker Quickstart", content)
        self.assertIn("METHODOLOGY.md", content)
        self.assertIn("docs/decisions/README.md", content)

    def test_methodology_doc_exists_and_covers_core_methods(self):
        self.assertTrue(os.path.exists("docs/METHODOLOGY.md"), "docs/METHODOLOGY.md should exist")
        with open("docs/METHODOLOGY.md", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Statistical & Machine Learning Methodology", content)
        self.assertIn("Kolmogorov-Smirnov", content)
        self.assertIn("Benjamini-Hochberg", content)
        self.assertIn("Variance Inflation Factor", content)
        self.assertIn("Decision Threshold Optimization", content)

    def test_architecture_decision_records_present(self):
        decisions_dir = Path("docs/decisions")
        self.assertTrue((decisions_dir / "README.md").is_file(), "docs/decisions/README.md should exist")
        adrs = sorted(p for p in decisions_dir.glob("000*.md"))
        self.assertGreaterEqual(len(adrs), 5, "Expected at least 5 ADRs in docs/decisions/")
        for adr in adrs:
            text = adr.read_text(encoding="utf-8")
            self.assertIn("## Status", text, f"{adr.name} missing Status section")
            self.assertIn("## Context", text, f"{adr.name} missing Context section")
            self.assertIn("## Decision", text, f"{adr.name} missing Decision section")
            self.assertIn("## Consequences", text, f"{adr.name} missing Consequences section")

    def test_no_scratch_planning_directories_in_docs(self):
        self.assertFalse(Path("docs/superpowers").exists(), "docs/superpowers should not be tracked")
        self.assertFalse(Path("docs/PROJECT.md").exists(), "docs/PROJECT.md should not remain")
        self.assertFalse(Path("docs/RESEARCH_FINDINGS.md").exists(), "docs/RESEARCH_FINDINGS.md should not remain")


if __name__ == "__main__":
    unittest.main()

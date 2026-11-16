import unittest
from pathlib import Path

class TestPackageStructure(unittest.TestCase):
    def test_root_and_package_naming(self):
        self.assertTrue(Path("app.py").is_file(), "app.py must be lowercase")
        self.assertTrue(Path("utils").is_dir(), "utils/ directory must be lowercase")
        self.assertTrue(Path("pages").is_dir(), "pages/ directory must be lowercase")
        for p in Path("utils").glob("*.py"):
            self.assertEqual(p.name, p.name.lower(), f"{p.name} must be lowercase")
        for p in Path("pages").glob("*.py"):
            self.assertEqual(p.name, p.name.lower(), f"{p.name} must be lowercase")

    def test_clean_imports(self):
        from utils import limits, paths, dataset_ui
        self.assertIsNotNone(limits.ResourcePolicy)
        self.assertIsNotNone(paths.DATASETS_DIR)

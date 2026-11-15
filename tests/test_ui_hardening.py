import sys
import unittest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    from Utils.theme import safe_error_boundary
except ImportError:
    safe_error_boundary = None

class TestUIHardening(unittest.TestCase):
    def test_client_configuration_security(self):
        with open(".streamlit/config.toml", "rb") as f:
            cfg = tomllib.load(f)
        self.assertIn("client", cfg, "config.toml must define [client] block")
        self.assertEqual(cfg["client"].get("toolbarMode"), "minimal")
        self.assertFalse(cfg["client"].get("showErrorDetails", True))

    def test_safe_error_boundary_redacts_internal_paths(self):
        self.assertIsNotNone(safe_error_boundary, "safe_error_boundary must be defined in Utils.theme")
        captured = []
        with safe_error_boundary(title="Dataset Ingestion Error", handler=lambda msg: captured.append(msg)):
            raise FileNotFoundError("C:/Users/Internal/Secret/data.csv not found on system")
        
        self.assertEqual(len(captured), 1)
        self.assertIn("Dataset Ingestion Error", captured[0])
        self.assertNotIn("C:/Users/Internal/Secret", captured[0])

if __name__ == "__main__":
    unittest.main()

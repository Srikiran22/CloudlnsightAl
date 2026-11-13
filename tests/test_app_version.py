"""Tests for dynamic package version resolution in App.py."""

import unittest
from unittest import mock

import App


class TestAppVersion(unittest.TestCase):
    """Verify version resolution across packaging environments."""

    def test_version_resolved_from_installed_metadata(self):
        """When package metadata is installed, version is resolved from metadata directly."""
        with mock.patch.object(App.metadata, "version", return_value="2.2.0"):
            self.assertEqual(App._get_version(), "2.2.0")

    def test_version_resolved_from_pyproject_when_metadata_missing(self):
        """When package metadata is missing in source checkout, version is resolved from pyproject.toml."""
        with mock.patch.object(App.metadata, "version", side_effect=App.metadata.PackageNotFoundError):
            self.assertEqual(App._get_version(), "2.2.0")

    def test_version_raises_runtime_error_when_both_missing(self):
        """When neither package metadata nor pyproject.toml is available, RuntimeError is raised."""
        orig_is_file = App.Path.is_file

        def fake_is_file(path_obj):
            if "pyproject.toml" in str(path_obj):
                return False
            return orig_is_file(path_obj)

        with mock.patch.object(App.metadata, "version", side_effect=App.metadata.PackageNotFoundError):
            with mock.patch.object(App.Path, "is_file", fake_is_file):
                with self.assertRaises(RuntimeError) as ctx:
                    App._get_version()
                self.assertIn("Could not resolve package version", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

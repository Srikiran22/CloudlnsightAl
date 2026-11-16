import unittest
from pathlib import Path


class TestDeploymentConfig(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent

    def test_dockerfile_is_genuine_multistage_and_non_root(self):
        df_path = self.root / "Dockerfile"
        self.assertTrue(df_path.is_file(), "Dockerfile should exist")
        content = df_path.read_text(encoding="utf-8")
        self.assertIn("AS builder", content)
        self.assertIn("AS runner", content)
        self.assertIn("USER appuser", content)
        self.assertIn("app.py", content)
        self.assertNotIn("App.py", content)
        self.assertIn("HEALTHCHECK", content)

    def test_dockerignore_excludes_data_and_artifacts(self):
        di_path = self.root / ".dockerignore"
        self.assertTrue(di_path.is_file(), ".dockerignore should exist")
        content = di_path.read_text(encoding="utf-8")
        self.assertIn(".venv", content)
        self.assertIn(".git", content)
        self.assertIn("datasets", content)
        self.assertIn("models", content)
        self.assertIn("reports", content)

    def test_docker_compose_references_tabulens(self):
        dc_path = self.root / "docker-compose.yml"
        self.assertTrue(dc_path.is_file(), "docker-compose.yml should exist")
        content = dc_path.read_text(encoding="utf-8")
        self.assertIn("tabulens", content)
        self.assertIn("healthcheck", content)

    def test_github_actions_ci_workflow_configured(self):
        ci_path = self.root / ".github" / "workflows" / "ci.yml"
        self.assertTrue(ci_path.is_file(), ".github/workflows/ci.yml should exist")
        content = ci_path.read_text(encoding="utf-8")
        self.assertIn("ruff check", content)
        self.assertIn("unittest", content)
        self.assertIn("3.11", content)

if __name__ == "__main__":
    unittest.main()

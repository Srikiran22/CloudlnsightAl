import os
import unittest

class TestDeploymentConfig(unittest.TestCase):
    def test_dockerfile_best_practices(self):
        self.assertTrue(os.path.exists("Dockerfile"), "Dockerfile should exist")
        with open("Dockerfile", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("FROM python:3.12-slim", content)
        self.assertIn("useradd", content)
        self.assertIn("USER appuser", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("8501", content)

    def test_dockerignore_entries(self):
        self.assertTrue(os.path.exists(".dockerignore"), ".dockerignore should exist")
        with open(".dockerignore", "r", encoding="utf-8") as f:
            content = f.read()
        for item in [".venv", "__pycache__", ".git", ".ruff_cache", "*.egg-info"]:
            self.assertIn(item, content)

    def test_docker_compose_structure(self):
        self.assertTrue(os.path.exists("docker-compose.yml"), "docker-compose.yml should exist")
        with open("docker-compose.yml", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("version:", content)
        self.assertIn("cloudinsight-ai:", content)
        self.assertIn("8501:8501", content)

    def test_healthcheck_script_executable(self):
        self.assertTrue(os.path.exists("scripts/healthcheck.py"), "scripts/healthcheck.py should exist")
        with open("scripts/healthcheck.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("/_stcore/health", content)

if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


class TestOfflineIntegrity(unittest.TestCase):
    def test_no_genai_files_exist(self):
        forbidden = [
            "utils/gemini.py", "utils/ai_sandbox.py", "utils/ai_agent.py",
            "utils/ai_convert.py", "utils/secrets.py",
            "pages/ai.py", "pages/aichat.py", "pages/aiconverter.py"
        ]
        for f in forbidden:
            self.assertFalse(Path(f).exists(), f"Forbidden file {f} still exists")

    def test_no_genai_in_requirements(self):
        with open("requirements.txt", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("google-genai", content)
        self.assertNotIn("pypdf", content)

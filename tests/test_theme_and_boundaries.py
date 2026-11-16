import unittest

import plotly.graph_objects as go

from utils.theme import (
    CHART_PALETTE,
    SEMANTIC_TOKENS,
    brand_logo_svg,
    current_theme_base,
    get_semantic_tokens,
    is_dark,
    safe_error_boundary,
    sanitize_error_message,
    style_plotly_fig,
)


class TestThemeAndBoundaries(unittest.TestCase):
    def test_semantic_tokens_and_palette(self):
        self.assertIn("primary", SEMANTIC_TOKENS)
        self.assertIn("success", SEMANTIC_TOKENS)
        self.assertIsInstance(CHART_PALETTE, list)
        self.assertTrue(len(CHART_PALETTE) >= 4)
        tokens_light = get_semantic_tokens(dark=False)
        self.assertEqual(tokens_light, SEMANTIC_TOKENS)
        tokens_dark = get_semantic_tokens(dark=True)
        self.assertIn("accent", tokens_dark)

    def test_sanitize_error_message(self):
        raw = "Error at C:\\Users\\secret\\project\\data.csv: failed"
        clean = sanitize_error_message(raw)
        self.assertNotIn("Users", clean)
        self.assertIn("[path redacted]", clean)

    def test_safe_error_boundary_catches(self):
        caught = []
        with safe_error_boundary("Test Failure", handler=lambda msg: caught.append(msg)):
            raise ValueError("Something broke at /home/user/file.py")
        self.assertEqual(len(caught), 1)
        self.assertIn("Test Failure", caught[0])
        self.assertNotIn("/home/user", caught[0])

    def test_style_plotly_fig(self):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        styled = style_plotly_fig(fig)
        self.assertIsNotNone(styled)
        self.assertEqual(styled.layout.paper_bgcolor, "rgba(0,0,0,0)")

    def test_theme_helpers(self):
        self.assertIn("<svg", brand_logo_svg())
        self.assertTrue(is_dark())
        self.assertEqual(current_theme_base(), "dark")

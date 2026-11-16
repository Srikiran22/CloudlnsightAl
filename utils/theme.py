"""TabuLens UI styling and theme utilities.

Provides lightweight brand elements, Plotly figure styling, color tokens,
and safety boundaries without brittle DOM selector overrides.
"""

from __future__ import annotations

import contextlib
import html
import re

import streamlit as st

from utils.logsys import get_logger

logger = get_logger("theme")

SEMANTIC_TOKENS = {
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_text": "#FFFFFF",
    "primary": "#2563EB",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "info": "#0284C7",
    "text": "#0F172A",
    "muted": "#64748B",
    "background": "#FFFFFF",
    "surface": "#F8FAFC",
    "border": "#E2E8F0",
}

DARK_SEMANTIC_TOKENS = {
    "accent": "#3B82F6",
    "accent_hover": "#60A5FA",
    "accent_text": "#0B0F19",
    "primary": "#3B82F6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#38BDF8",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "background": "#0B0F19",
    "surface": "#111827",
    "border": "#263042",
}

CHART_PALETTE = [
    "#2563EB",  # Primary Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber Gold
    "#8B5CF6",  # Violet Purple
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#F97316",  # Orange
    "#64748B",  # Slate
]


def sanitize_error_message(raw_msg: object) -> str:
    """Redact internal file system paths (Windows and Unix-style) from error messages."""
    sanitized = re.sub(r"[A-Za-z]:[\\/][^:\s]+", "[path redacted]", str(raw_msg))
    sanitized = re.sub(r"/(?:usr|home|etc|var|tmp|app)[^\s]*", "[path redacted]", sanitized)
    if not sanitized.strip():
        sanitized = "An unexpected error occurred."
    return sanitized


@contextlib.contextmanager
def safe_error_boundary(title="An error occurred", handler=None):
    """Context manager to intercept exceptions and display sanitized user-facing alerts."""
    try:
        yield
    except Exception as exc:
        raw_msg = str(exc)
        sanitized = sanitize_error_message(raw_msg)
        display_msg = f"**{title}**: {sanitized}"
        logger.error(f"UI error boundary caught: {raw_msg}", exc_info=True)
        if handler is not None:
            handler(display_msg)
        else:
            with contextlib.suppress(Exception):
                st.error(display_msg)


def get_semantic_tokens(dark: bool = False) -> dict:
    """Return dictionary of semantic color tokens based on active mode."""
    return DARK_SEMANTIC_TOKENS.copy() if dark else SEMANTIC_TOKENS.copy()


def chart_palette() -> list[str]:
    """The categorical palette every figure in the app should share."""
    return list(CHART_PALETTE)


def brand_logo_svg() -> str:
    """Two-tone TabuLens wordmark for `st.logo`."""
    accent = "#2563EB"
    text = "#475569"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="220" height="30" viewBox="0 0 220 30">
  <rect y="2" width="26" height="26" rx="6" fill="{accent}"/>
  <path d="M6 18 L10 13 L15 16 L20 8" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="20" cy="8" r="1.8" fill="#FFFFFF"/>
  <text x="34" y="20" font-family="'Inter', -apple-system, Segoe UI, sans-serif" font-size="15"
        font-weight="750" letter-spacing="-0.02em" fill="{text}">Tabu<tspan fill="{accent}">Lens</tspan> <tspan font-size="10" font-weight="650" fill="#64748B" letter-spacing="0.08em">STUDIO</tspan></text>
</svg>"""


def page_header(title: str, subtitle: str, badge: str | None = None):
    """Renders a clean page header row with optional status badge."""
    badge_html = (
        f'<span style="background: rgba(37,99,235,0.12); border: 1px solid rgba(37,99,235,0.3); color: #3B82F6; '
        f'padding: 0.2rem 0.65rem; border-radius: 9999px; font-size: 0.76rem; font-weight: 600; '
        f'letter-spacing: 0.04em;">{html.escape(badge)}</span>'
        if badge
        else ""
    )
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.2rem;">
          <h1 style="margin: 0; padding: 0;">{html.escape(title)}</h1>
          {badge_html}
        </div>
        <p style="margin-top: 0.15rem; margin-bottom: 1.1rem; opacity: 0.8; font-size: 0.92rem;">{html.escape(subtitle)}</p>
        """,
        unsafe_allow_html=True,
    )


def section_card_header(title: str, subtitle: str | None = None):
    """Renders a clean card section header inside bordered containers."""
    sub_html = (
        f'<p style="margin: 0.2rem 0 0.85rem 0; opacity: 0.75; font-size: 0.85rem; line-height: 1.45;">{html.escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 0.65rem;">
          <h3 style="margin: 0; font-size: 1.05rem; font-weight: 650; letter-spacing: -0.01em;">{html.escape(title)}</h3>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str | int | float, subtext: str | None = None, badge: str | None = None) -> None:
    """Render a structured KPI card."""
    badge_html = f'<span style="font-size:0.72rem; padding:2px 7px; border-radius:999px; background:rgba(37,99,235,0.1); border:1px solid rgba(37,99,235,0.25); color:#3B82F6; font-weight:600;">{html.escape(badge)}</span>' if badge else ''
    sub_html = f'<div style="font-size:0.8rem; opacity:0.7; margin-top:0.25rem;">{html.escape(subtext)}</div>' if subtext else ''
    st.markdown(
        f"""
        <div style="border: 1px solid rgba(148,163,184,0.2); border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size: 0.78rem; text-transform: uppercase; font-weight: 650; letter-spacing: 0.05em; opacity: 0.75;">{html.escape(label)}</span>
            {badge_html}
          </div>
          <div style="font-size: 1.55rem; font-weight: 700; margin-top: 0.25rem; font-family: monospace;">{html.escape(str(value))}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_theme_base() -> str:
    """The configured base theme ("light" or "dark")."""
    try:
        return (st.get_option("theme.base") or "dark").lower()
    except Exception:
        return "dark"


def resolved_theme_type() -> str | None:
    """The theme the client is actively rendering, or None when unavailable."""
    try:
        value = getattr(st.context.theme, "type", None)
    except Exception:
        return None
    return value.lower() if isinstance(value, str) and value else None


def is_dark() -> bool:
    """True when rendering in dark mode."""
    resolved = resolved_theme_type()
    if resolved in {"light", "dark"}:
        return resolved == "dark"
    return current_theme_base() == "dark"


def plot_template():
    """Plotly template matching the active theme."""
    return "plotly_dark" if is_dark() else "plotly_white"


def style_plotly_fig(fig):
    """Apply consistent styling across Plotly figures."""
    dark = is_dark()
    tpl = "plotly_dark" if dark else "plotly_white"
    bg_color = "rgba(0,0,0,0)"
    text_color = "#F8FAFC" if dark else "#0F172A"
    grid_color = "rgba(148, 163, 184, 0.12)" if dark else "rgba(100, 116, 139, 0.12)"

    fig.update_layout(
        template=tpl,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font={"color": text_color, "family": '"Inter", "Segoe UI", system-ui, sans-serif'},
    )
    with contextlib.suppress(Exception):
        fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
        fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    return fig


def apply_theme() -> None:
    """Apply base custom CSS styles for clean layout spacing."""
    css = """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      [data-testid="stSidebar"] { padding-top: 1rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_appearance_note() -> None:
    """Sidebar appearance note."""
    st.caption("Theme follows your system preference. You can change it in Settings -> Theme.")

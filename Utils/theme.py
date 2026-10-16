# CloudInsight AI design system.
# Single source of truth for the visual language: color tokens, typography,
# spacing, and component styling. All DOM-scoped selectors live here so page
# code stays free of styling hacks.
#
# Selector rationale (Streamlit 1.60):
# - .stApp h1/h2/h3, hr, a: plain elements rendered by Streamlit blocks — stable.
# - [data-testid="stMainBlockContainer"], section[data-testid="stSidebar"],
#   [data-testid="stMetric*"], [data-testid="stAlert"], .stButton > button,
#   .stTabs [data-baseweb="tab*"]: Streamlit's public test ids / BaseWeb hooks;
#   standard theming practice, isolated in this file only.
# - Tokens are BAKED per active base theme (light/dark) at injection time using
#   concrete values — no reliance on Streamlit runtime CSS variable names, so
#   contrast is correct in both modes.

import html
import re
import streamlit as st


def _palette(dark):
    if dark:
        return {
            "__BG__": "#0B0F19",
            "__ACCENT__": "#3B82F6",
            "__ACCENT_HOVER__": "#60A5FA",
            "__TEXT__": "#F8FAFC",
            "__TEXT2__": "#CBD5E1",
            "__MUTED__": "#94A3B8",
            "__BORDER__": "rgba(148, 163, 184, .26)",
            "__TINT__": "rgba(148, 163, 184, .09)",
            "__SURFACE__": "#111827",
            "__SHADOW__": "none",
        }
    return {
        "__BG__": "#FFFFFF",
        "__ACCENT__": "#2563EB",
        "__ACCENT_HOVER__": "#1D4ED8",
        "__TEXT__": "#0F172A",
        "__TEXT2__": "#334155",
        "__MUTED__": "#64748B",
        "__BORDER__": "rgba(100, 116, 139, .22)",
        "__TINT__": "rgba(100, 116, 139, .06)",
        "__SURFACE__": "#F8FAFC",
        "__SHADOW__": "0 1px 2px rgba(15, 23, 42, .06)",
    }


SEMANTIC_TOKENS = {
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
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


def get_semantic_tokens(dark: bool = False) -> dict:
    """Return dictionary of semantic color tokens based on active mode."""
    return DARK_SEMANTIC_TOKENS.copy() if dark else SEMANTIC_TOKENS.copy()


_CSS = """
:root {
  --ci-bg: __BG__;
  --ci-accent: __ACCENT__;
  --ci-accent-hover: __ACCENT_HOVER__;
  --ci-text: __TEXT__;
  --ci-text-2: __TEXT2__;
  --ci-muted: __MUTED__;
  --ci-border: __BORDER__;
  --ci-tint: __TINT__;
  --ci-surface: __SURFACE__;
  --ci-shadow-sm: __SHADOW__;
  --ci-radius: 8px;
  --ci-radius-sm: 6px;
  --ci-font: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}

/* ---- global app container & background consistency ----------------- */
.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--ci-bg) !important;
  color: var(--ci-text) !important;
  font-family: var(--ci-font);
}

/* ---- typography scale ------------------------------------------------ */
.stApp, .stApp p, .stApp li {
  font-family: var(--ci-font);
}
.stApp h1 {                       /* page titles via st.title */
  font-size: 1.45rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.25;
  color: var(--ci-text) !important;
  margin-bottom: 0;
}
.stApp h2 {                       /* section headings via st.subheader */
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: var(--ci-text) !important;
  padding-bottom: .35rem;
  border-bottom: 1px solid var(--ci-border);
}
.stApp h3 {
  font-size: .95rem;
  font-weight: 650;
  color: var(--ci-text) !important;
}
.stApp h4 {
  font-size: .9rem;
  font-weight: 650;
  color: var(--ci-text) !important;
}
.stApp p, .stApp li {
  font-size: .9rem;
  line-height: 1.55;
  color: var(--ci-text-2);
}
.stApp caption, .stApp [data-testid="stCaptionContainer"] p {
  font-size: .82rem;
  color: var(--ci-muted);
}
.stApp strong { color: var(--ci-text); font-weight: 650; }

/* ---- form labels & radios & checkboxes ------------------------------- */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
  color: var(--ci-text) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
}
[data-testid="stCheckbox"] label p,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  color: var(--ci-text) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
}

/* ---- layout ------------------------------------------------------------ */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1180px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  background-color: transparent !important;
}
hr {
  border: none;
  border-top: 1px solid var(--ci-border);
  margin: 1.4rem 0 !important;
}

/* ---- header bar -------------------------------------------------------- */
[data-testid="stHeader"] {
  background: transparent;
  height: 2.6rem;
}

/* ---- sidebar ------------------------------------------------------------ */
section[data-testid="stSidebar"] {
  background: var(--ci-surface);
  border-right: 1px solid var(--ci-border);
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 0.6rem !important;
  padding-bottom: 0.6rem !important;
}
section[data-testid="stSidebar"] hr { margin: .5rem 0; }

/* ---- compact sidebar navigation ----------------------------------------- */
[data-testid="stSidebarNav"] {
  padding-top: 0.15rem !important;
  padding-bottom: 0.15rem !important;
}
[data-testid="stSidebarNavItems"] {
  gap: 1px !important;
}
[data-testid="stSidebarNavItems"] li {
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebarNavItems"] a,
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
  font-size: 0.83rem !important;
  border-radius: var(--ci-radius-sm);
  padding: 2.5px 8px !important;
  min-height: 1.7rem !important;
  line-height: 1.25 !important;
}
[data-testid="stSidebarNavItems"] a:hover,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
  background: var(--ci-tint);
}
[data-testid="stSidebarNav"] [data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarNav"] span[data-testid="stHeader"] {
  margin-top: 0.35rem !important;
  margin-bottom: 0.1rem !important;
  font-size: 0.70rem !important;
  letter-spacing: 0.06em !important;
}
[data-testid="stSidebarUserContent"] {
  padding-top: 0.15rem !important;
}

.ci-brand { margin-bottom: .2rem; user-select: none; }
.ci-brand-name {
  font-family: var(--ci-font);
  font-size: .95rem;
  font-weight: 750;
  letter-spacing: -0.02em;
  color: var(--ci-text);
}
.ci-brand-name span { color: var(--ci-accent); }
.ci-brand-tag {
  font-size: .72rem;
  color: var(--ci-muted);
  line-height: 1.35;
}
.ci-side-label {
  font-size: .65rem;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--ci-muted);
  margin-top: 0.3rem;
  margin-bottom: .1rem;
}
.ci-dataset {
  background: var(--ci-surface);
  border: 1px solid var(--ci-border);
  border-radius: var(--ci-radius);
  padding: .4rem .6rem;
  margin-top: .1rem;
  margin-bottom: .25rem;
}
.ci-dataset-name {
  font-size: .82rem;
  font-weight: 600;
  color: var(--ci-text);
  word-break: break-all;
}
.ci-dataset-meta {
  font-size: .72rem;
  color: var(--ci-muted);
  margin-top: .05rem;
}

/* ---- buttons: primary / secondary / tertiary hierarchy ------------------ */
.stButton > button {
  border-radius: var(--ci-radius-sm);
  border: 1px solid var(--ci-border);
  font-weight: 600;
  font-size: .88rem;
  height: 2.35rem;
  box-shadow: none;
  transition: all .15s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.stButton > button[kind="primary"] {
  background: var(--ci-accent) !important;
  border-color: transparent !important;
  color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--ci-accent-hover) !important;
  border-color: transparent !important;
  color: #FFFFFF !important;
  box-shadow: 0 2px 4px rgba(37, 99, 235, .25);
}
.stButton > button:not([kind="primary"]) {
  background: var(--ci-surface);
  color: var(--ci-text);
  border: 1px solid var(--ci-border);
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--ci-accent);
  color: var(--ci-accent);
  background: var(--ci-tint);
}
.stButton > button:focus-visible {
  outline: 2px solid var(--ci-accent);
  outline-offset: 2px;
}

/* ---- metrics: structured KPI cards --------------------------------------- */
[data-testid="stMetric"] {
  background: var(--ci-surface) !important;
  border: 1px solid var(--ci-border) !important;
  border-radius: 10px !important;
  padding: 12px 16px !important;
  box-shadow: var(--ci-shadow-sm);
  transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover {
  border-color: var(--ci-accent) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
  font-size: .74rem !important;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--ci-muted) !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.25 !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.35rem !important;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums !important;
  color: var(--ci-text);
  margin-top: 0.1rem;
}
[data-testid="stMetricDelta"] > div[data-testid="stMetricDelta"] {
  font-size: .78rem !important;
  font-weight: 550;
}
/* Hide fake upward/downward arrow SVG on neutral delta status pills */
[data-testid*="stMetricDelta"] svg,
[data-testid*="stMetricDeltaIcon-Up"],
[data-testid*="stMetricDeltaIcon-Down"],
[data-testid*="stMetricDeltaIcon"],
[data-testid*="stMetricDelta"] span:has(svg) {
  display: none !important;
}

/* ---- containers & borders ------------------------------------------------ */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--ci-radius) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: var(--ci-border) !important;
  border-radius: var(--ci-radius) !important;
}

/* ---- tabs: underline navigation, not pills -------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 1.1rem;
  border-bottom: 1px solid var(--ci-border);
}
.stTabs [data-baseweb="tab"] {
  padding: .45rem .1rem;
  border-radius: 0;
  color: var(--ci-text-2);
  font-weight: 550;
}
.stTabs [aria-selected="true"] {
  color: var(--ci-text) !important;
  box-shadow: inset 0 -2px 0 var(--ci-accent);
}

/* ---- inputs ----------------------------------------------------------------- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div,
[data-testid="stTextAreaTextarea"] {
  border-radius: var(--ci-radius-sm) !important;
  font-size: .9rem;
}
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {
  box-shadow: none !important;
}

/* ---- alerts: quiet left-rule notes ------------------------------------------- */
div[data-testid="stAlert"] {
  border-left: 3px solid rgba(120, 120, 135, .5);
  border-radius: 0 var(--ci-radius) var(--ci-radius) 0;
  background-color: var(--ci-tint);
}
div[data-testid="stAlert"] p { font-size: .86rem; }

/* ---- expanders / chat / dataframes ---------------------------------------------- */
[data-testid="stExpander"] {
  border: 1px solid var(--ci-border);
  border-radius: var(--ci-radius);
  background: transparent;
}
[data-testid="stExpander"] details { border: none !important; }
[data-testid="stChatMessage"] {
  background: var(--ci-surface);
  border-radius: var(--ci-radius);
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--ci-border);
  border-radius: var(--ci-radius);
}

/* ---- misc -------------------------------------------------------------------------- */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb {
  background: rgba(120, 120, 135, .35);
  border-radius: 8px;
}
::-webkit-scrollbar-track { background: transparent; }

/* Streamlit decor that competes with the app's own chrome (community-standard
   polish; the Settings menu stays reachable via the header toolbar). */
[data-testid="stAppDeployButton"], #MainMenu {
  display: none !important;
}
[data-testid="stSidebarCollapseButton"] {
  display: flex !important;
  visibility: visible !important;
}
footer [data-testid="stStatusWidget"], footer { display: none !important; }
"""


def _format_inline_code(text: str) -> str:
    """Format inline backtick code snippets into clean HTML code tags after escaping."""
    if not text:
        return ""
    escaped = html.escape(str(text))
    pattern = r"`([^`]+)`"
    def _repl(match):
        inner = match.group(1)
        return (
            f'<code style="background:var(--ci-tint, rgba(148,163,184,0.15)); '
            f'padding:2px 6px; border-radius:4px; font-family:monospace; '
            f'font-size:0.86em; color:var(--ci-text);">{inner}</code>'
        )
    return re.sub(pattern, _repl, escaped)


def page_header(title: str, subtitle: str, badge: str | None = None):
    """Renders a standardized, professional page header row with optional status badge."""
    badge_html = (
        f'<span style="background: var(--ci-tint); border: 1px solid var(--ci-border); color: var(--ci-accent); '
        f'padding: 0.2rem 0.65rem; border-radius: 9999px; font-size: 0.76rem; font-weight: 600; '
        f'letter-spacing: 0.04em;">{html.escape(badge)}</span>'
        if badge
        else ""
    )
    formatted_sub = _format_inline_code(subtitle)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.2rem;">
          <h1 style="margin: 0; padding: 0;">{html.escape(title)}</h1>
          {badge_html}
        </div>
        <p style="margin-top: 0.15rem; margin-bottom: 1.1rem; color: var(--ci-text-2); font-size: 0.92rem;">{formatted_sub}</p>
        """,
        unsafe_allow_html=True,
    )


def section_card_header(title: str, subtitle: str | None = None):
    """Renders a clean card section header inside bordered containers."""
    sub_html = (
        f'<p style="margin: 0.1rem 0 0.65rem 0; color: var(--ci-muted); font-size: 0.82rem;">{_format_inline_code(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 0.45rem;">
          <h3 style="margin: 0; font-size: 0.98rem; font-weight: 650; color: var(--ci-text);">{html.escape(title)}</h3>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_theme():
    """Inject the design system once per page run, themed to the active base."""
    css = _CSS
    for token, value in _palette(is_dark()).items():
        css = css.replace(token, value)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --- runtime theme switching -------------------------------------------------
# Streamlit has no public runtime theming API yet (github.com/streamlit/streamlit
# issue #14172). The community-standard solution is st._config.set_option on the
# theme options followed by a rerun; it is isolated here so the rest of the app
# never touches private APIs. If the private hook disappears, users can still
# switch via Settings (both light and dark themes are defined in config.toml).

def current_theme_base():
    try:
        return (st.get_option("theme.base") or "light").lower()
    except Exception:
        return "light"


def is_dark():
    if hasattr(st, "session_state") and "theme_mode" in st.session_state:
        return st.session_state["theme_mode"] == "dark"
    return current_theme_base() == "dark"


def plot_template():
    """Plotly template matching the active app theme."""
    return "plotly_dark" if is_dark() else "plotly_white"


def toggle_theme_button():
    dark = is_dark()
    btn_label = "Light mode" if dark else "Dark mode"
    if st.button(btn_label, key="ci_theme_toggle", use_container_width=True):
        new_mode = "light" if dark else "dark"
        if hasattr(st, "session_state"):
            st.session_state["theme_mode"] = new_mode
        try:
            st._config.set_option("theme.base", new_mode)
        except Exception as error:
            # private-API risk made explicit: if the hook ever vanishes the
            # user still gets a hint instead of a button that does nothing
            from Utils.logsys import get_logger
            get_logger("theme").warning("runtime theme switch unavailable: %s", error)
            st.toast("Theme switching is unavailable in this Streamlit version; "
                     "use the Settings menu instead.")
        # rerun re-renders with the new base theme (st.rerun halts this script)
        st.rerun()

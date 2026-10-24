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

from Utils.logsys import get_logger

logger = get_logger("theme")


def _palette(dark):
    if dark:
        return {
            "__BG__": "#0B0F19",
            "__ACCENT__": "#3B82F6",
            "__ACCENT_HOVER__": "#60A5FA",
            "__TEXT__": "#F8FAFC",
            "__TEXT2__": "#CBD5E1",
            "__MUTED__": "#94A3B8",
            "__BORDER__": "#263042",
            "__TINT__": "rgba(59, 130, 246, .09)",
            "__SURFACE__": "#111827",
            "__INPUT_BG__": "#131C2E",
            "__SHADOW__": "none",
        }
    return {
        "__BG__": "#FFFFFF",
        "__ACCENT__": "#2563EB",
        "__ACCENT_HOVER__": "#1D4ED8",
        "__TEXT__": "#0F172A",
        "__TEXT2__": "#334155",
        "__MUTED__": "#64748B",
        "__BORDER__": "#E2E8F0",
        "__TINT__": "rgba(100, 116, 139, .06)",
        "__SURFACE__": "#F8FAFC",
        "__INPUT_BG__": "#FFFFFF",
        "__SHADOW__": "0 1px 2px rgba(15, 23, 42, .06)",
    }


# Maps palette keys onto the CSS custom properties the stylesheet consumes, so
# the Python palette stays the single source of truth for both appearances.
_TOKEN_VARS = {
    "__BG__": "--ci-bg",
    "__ACCENT__": "--ci-accent",
    "__ACCENT_HOVER__": "--ci-accent-hover",
    "__TEXT__": "--ci-text",
    "__TEXT2__": "--ci-text-2",
    "__MUTED__": "--ci-muted",
    "__BORDER__": "--ci-border",
    "__TINT__": "--ci-tint",
    "__SURFACE__": "--ci-surface",
    "__INPUT_BG__": "--ci-input-bg",
    "__SHADOW__": "--ci-shadow-sm",
}

_STATIC_TOKENS = """  --ci-radius: 8px;
  --ci-radius-sm: 6px;
  --ci-font: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;"""


def _token_block(dark: bool) -> str:
    """Render one palette as a CSS declaration block."""
    palette = _palette(dark)
    declarations = [f"  {_TOKEN_VARS[key]}: {palette[key]};" for key in _TOKEN_VARS]
    return "\n".join(declarations + [_STATIC_TOKENS])


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


def chart_palette() -> list[str]:
    """The categorical palette every figure in the app should share."""
    return list(CHART_PALETTE)


_CSS = """
/* The palette is resolved on the client (see _sync_client_theme): Streamlit's
   own theme is the source of truth, and the data-ci-theme attribute mirrors it.
   :root keeps the server-known theme so the first paint never flashes. */
:root { __INITIAL_TOKENS__ }
html[data-ci-theme="dark"] { __DARK_TOKENS__ }
html[data-ci-theme="light"] { __LIGHT_TOKENS__ }

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
[data-testid="stWidgetLabel"] {
  margin-bottom: 6px !important;
}
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

.ci-side-label {
  font-size: .65rem;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--ci-muted);
  margin-top: 0.3rem;
  margin-bottom: .1rem;
}
.ci-theme-note {
  font-size: .74rem;
  line-height: 1.4;
  color: var(--ci-muted);
  padding: .1rem .1rem .3rem .1rem;
}
.ci-theme-note b { color: var(--ci-text-2); font-weight: 650; }
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
/* Streamlit disables buttons via the native attribute; the primary colours above
   are forced with !important, so mute disabled controls explicitly. */
.stButton > button:disabled,
.stButton > button[disabled] {
  opacity: .45 !important;
  cursor: not-allowed !important;
  box-shadow: none !important;
}

/* ---- metrics: structured KPI cards --------------------------------------- */
[data-testid="stMetric"] {
  background: var(--ci-surface) !important;
  border: 1px solid var(--ci-border) !important;
  border-radius: 10px !important;
  padding: 16px 20px !important;
  min-height: 94px !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  box-shadow: var(--ci-shadow-sm);
  transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover {
  border-color: var(--ci-accent) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
  font-size: .80rem !important;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--ci-muted) !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.35 !important;
  margin-bottom: 4px !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.45rem !important;
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
/* Streamlit 1.60 wraps `st.container(border=True)` in a stLayoutWrapper and
   paints the border on the child stVerticalBlock; older releases used
   stVerticalBlockBorderWrapper. Both are supported so the design survives
   Streamlit upgrades. */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
  border-radius: var(--ci-radius) !important;
  border: 1px solid var(--ci-border) !important;
  padding: 20px 24px !important;
  margin-bottom: 1.25rem !important;
  background: transparent !important;
}

/* Cards size to their content. Stretching them to the column height leaves
   awkward empty space inside short cards when the sibling column is taller. */

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

/* ---- form inputs, text areas, number inputs & selects --------------------- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stTextAreaTextarea"],
[data-baseweb="input"] input,
[data-baseweb="base-input"] input,
[data-baseweb="select"] > div,
[data-baseweb="base-input"] {
  background-color: var(--ci-input-bg) !important;
  color: var(--ci-text) !important;
  border: 1px solid var(--ci-border) !important;
  border-radius: var(--ci-radius-sm) !important;
  font-size: .9rem !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextAreaTextarea"]::placeholder {
  color: var(--ci-muted) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextAreaTextarea"]:focus,
[data-baseweb="select"] > div:focus-within {
  border-color: var(--ci-accent) !important;
  box-shadow: 0 0 0 1px var(--ci-accent) !important;
}
[data-baseweb="select"] svg {
  fill: var(--ci-text-2) !important;
}
[data-baseweb="select"] span {
  color: var(--ci-text) !important;
}
[data-baseweb="menu"], [data-baseweb="popover"] > div {
  background-color: var(--ci-surface) !important;
  border: 1px solid var(--ci-border) !important;
  color: var(--ci-text) !important;
}
[data-baseweb="menu"] li {
  color: var(--ci-text) !important;
}
[data-baseweb="menu"] li:hover {
  background-color: var(--ci-tint) !important;
}

/* ---- file uploader ---------------------------------------------------------- */
[data-testid="stFileUploader"] {
  background-color: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
  background-color: var(--ci-input-bg) !important;
  border: 1px dashed var(--ci-border) !important;
  border-radius: var(--ci-radius) !important;
  color: var(--ci-text) !important;
  padding: 1.2rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--ci-accent) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
  color: var(--ci-muted) !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background-color: var(--ci-surface) !important;
  border: 1px solid var(--ci-border) !important;
  color: var(--ci-text) !important;
}

/* ---- chat input & bottom pinned bar ----------------------------------------- */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
  background-color: var(--ci-bg) !important;
  border-top: 1px solid var(--ci-border) !important;
}
[data-testid="stChatInput"] {
  background-color: var(--ci-input-bg) !important;
  border: 1px solid var(--ci-border) !important;
  border-radius: var(--ci-radius) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--ci-accent) !important;
  box-shadow: 0 0 0 1px var(--ci-accent) !important;
}
[data-testid="stChatInput"] textarea {
  background-color: transparent !important;
  color: var(--ci-text) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--ci-muted) !important;
}
[data-testid="stChatInput"] button {
  color: var(--ci-accent) !important;
}

/* ---- alerts: semantic styling (amber warning, teal success, blue info, red error) ---- */
div[data-testid="stAlert"] {
  border-radius: 0 var(--ci-radius) var(--ci-radius) 0 !important;
  padding: 0 !important;
  border: none !important;
  overflow: hidden;
  margin: 0.65rem 0 !important;
  background-color: var(--ci-tint) !important;
  border-left: 3px solid var(--ci-accent) !important;
}
div[data-testid="stAlert"] [data-testid="stAlertContainer"] {
  background-color: transparent !important;
  padding: 14px 18px !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
  font-size: .88rem;
  line-height: 1.5;
  color: var(--ci-text) !important;
}

/* Warning alert: restrained amber */
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
  background-color: rgba(245, 158, 11, 0.12) !important;
  border-left: 3px solid #F59E0B !important;
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) svg {
  fill: #F59E0B !important;
  color: #F59E0B !important;
}

/* Success alert: crisp teal/emerald */
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
  background-color: rgba(20, 184, 166, 0.12) !important;
  border-left: 3px solid #14B8A6 !important;
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) svg {
  fill: #14B8A6 !important;
  color: #14B8A6 !important;
}

/* Info alert: slate/sky blue */
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
  background-color: rgba(59, 130, 246, 0.12) !important;
  border-left: 3px solid #3B82F6 !important;
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) svg {
  fill: #3B82F6 !important;
  color: #3B82F6 !important;
}

/* Error alert: crimson */
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
  background-color: rgba(239, 68, 68, 0.12) !important;
  border-left: 3px solid #EF4444 !important;
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) svg {
  fill: #EF4444 !important;
  color: #EF4444 !important;
}

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
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--ci-border) !important;
  border-radius: var(--ci-radius) !important;
  background-color: var(--ci-surface) !important;
}

/* ---- chart typography ---------------------------------------------------- */
/* Streamlit's default chart axis colour is ~#808495, which is below AA contrast
   on a white card. Align axis and legend text with the design tokens instead. */
html[data-ci-theme="light"] [data-testid="stPlotlyChart"] .xtick text,
html[data-ci-theme="light"] [data-testid="stPlotlyChart"] .ytick text,
html[data-ci-theme="light"] [data-testid="stPlotlyChart"] .legendtext {
  fill: var(--ci-text-2) !important;
}
html[data-ci-theme="dark"] [data-testid="stPlotlyChart"] .xtick text,
html[data-ci-theme="dark"] [data-testid="stPlotlyChart"] .ytick text,
html[data-ci-theme="dark"] [data-testid="stPlotlyChart"] .legendtext {
  fill: var(--ci-muted) !important;
}

/* ---- misc -------------------------------------------------------------------------- */
/* The theme bridge runs in a 1px frame (st.iframe rejects height 0); collapse it. */
iframe[title="st.iframe"] {
  height: 0 !important;
  min-height: 0 !important;
  border: 0 !important;
  display: block !important;
}
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb {
  background: rgba(120, 120, 135, .35);
  border-radius: 8px;
}
::-webkit-scrollbar-track { background: transparent; }

/* Inline code chips: single source of truth for the small labelled values
   (dataset names, column names) rendered through markdown backticks. */
.stApp :not(pre) > code {
  background: var(--ci-tint);
  border: none;
  border-radius: 4px;
  padding: 1px 5px;
  font-size: .85em;
  color: var(--ci-text) !important;
}

/* Streamlit decor that competes with the app's own chrome. The Deploy button
   is hidden, but the main menu stays available because it hosts Appearance
   (Settings → Theme), print and screencast controls. */
[data-testid="stAppDeployButton"] {
  display: none !important;
}
[data-testid="stMainMenuButton"] {
  color: var(--ci-muted) !important;
}
[data-testid="stMainMenuButton"]:hover {
  color: var(--ci-text) !important;
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


def brand_logo_svg() -> str:
    """Two-tone Data Science Studio wordmark for `st.logo`, drawn as an inline SVG."""
    accent = "#2563EB"
    text = "#475569"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="220" height="30" viewBox="0 0 220 30">
  <rect y="2" width="26" height="26" rx="6" fill="{accent}"/>
  <path d="M6 18 L10 13 L15 16 L20 8" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="20" cy="8" r="1.8" fill="#FFFFFF"/>
  <text x="34" y="20" font-family="'Inter', -apple-system, Segoe UI, sans-serif" font-size="15"
        font-weight="750" letter-spacing="-0.02em" fill="{text}">Cloud<tspan fill="{accent}">Insight</tspan> <tspan font-size="10" font-weight="650" fill="#64748B" letter-spacing="0.08em">STUDIO</tspan></text>
</svg>"""


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
        f'<p style="margin: 0.2rem 0 0.85rem 0; color: var(--ci-muted); font-size: 0.85rem; line-height: 1.45;">{_format_inline_code(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 0.65rem;">
          <h3 style="margin: 0; font-size: 1.05rem; font-weight: 650; color: var(--ci-text); letter-spacing: -0.01em;">{html.escape(title)}</h3>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str | int | float, subtext: str | None = None, badge: str | None = None) -> None:
    """Render a structured, high-legibility Data Science KPI card."""
    badge_html = f'<span style="font-size:0.72rem; padding:2px 7px; border-radius:999px; background:var(--ci-tint); border:1px solid var(--ci-border); color:var(--ci-accent); font-weight:600;">{html.escape(badge)}</span>' if badge else ''
    sub_html = f'<div style="font-size:0.8rem; color:var(--ci-muted); margin-top:0.25rem;">{html.escape(subtext)}</div>' if subtext else ''
    st.markdown(
        f"""
        <div style="border: 1px solid var(--ci-border); background: var(--ci-surface); border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size: 0.78rem; text-transform: uppercase; font-weight: 650; letter-spacing: 0.05em; color: var(--ci-muted);">{html.escape(label)}</span>
            {badge_html}
          </div>
          <div style="font-size: 1.55rem; font-weight: 700; color: var(--ci-text); margin-top: 0.25rem; font-family: monospace, var(--ci-font);">{html.escape(str(value))}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


_THEME_BRIDGE = r"""
<script>
(function () {
  var doc = window.parent.document;
  var root = doc.documentElement;

  // Streamlit resolves the theme on the client, so the only reliable signal is
  // the rendered background colour. Body is deliberately used: the app's own
  // stylesheet only paints .stApp, so this value always reflects the live theme.
  function currentMode() {
    var background = getComputedStyle(doc.body).backgroundColor;
    var match = background.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (!match) { return null; }
    var luminance = (0.2126 * +match[1] + 0.7152 * +match[2] + 0.0722 * +match[3]) / 255;
    return luminance < 0.45 ? "dark" : "light";
  }

  function sync() {
    var mode = currentMode();
    if (mode && root.getAttribute("data-ci-theme") !== mode) {
      root.setAttribute("data-ci-theme", mode);
    }
  }

  sync();
  if (window.MutationObserver && doc.body) {
    new MutationObserver(sync).observe(doc.body, {
      attributes: true,
      subtree: true,
      attributeFilter: ["style", "class"]
    });
  }
  window.setInterval(sync, 500);
})();
</script>
"""


def _sync_client_theme() -> None:
    """Mirror Streamlit's live theme onto <html data-ci-theme> for instant switching.

    Streamlit re-themes its widgets in the browser without re-running the
    script, so a server-rendered palette would lag behind. This zero-height
    frame watches the resolved background colour and flips the palette
    attribute; the CSS then applies the matching token block immediately.

    ``st.iframe`` is the supported replacement for ``st.components.v1.html``
    (which is deprecated); older Streamlit releases fall back to the legacy
    component API. The frame is one pixel tall: ``st.iframe`` rejects a zero
    height, and the stylesheet collapses the frame visually.
    """
    try:
        st.iframe(_THEME_BRIDGE, height=1)
    except AttributeError:  # Streamlit < 1.6x has no st.iframe
        try:
            import streamlit.components.v1 as components

            components.html(_THEME_BRIDGE, height=0)
        except Exception as exc:  # pragma: no cover - defensive for exotic runtimes
            logger.debug("theme bridge unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive for exotic runtimes
        logger.debug("theme bridge unavailable: %s", exc)


def apply_theme() -> None:
    """Inject the design system and keep its palette in sync with the live theme."""
    initial_dark = is_dark()
    css = (
        _CSS.replace("__INITIAL_TOKENS__", _token_block(initial_dark))
        .replace("__DARK_TOKENS__", _token_block(True))
        .replace("__LIGHT_TOKENS__", _token_block(False))
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _sync_client_theme()


# --- theme resolution ---------------------------------------------------------
# Appearance belongs to Streamlit, not to this app: the browser preference
# (system default, or a choice pinned in the ⋮ menu → Settings → Theme) decides
# which theme the canvas-rendered widgets use. Server-side CSS must therefore
# read the *resolved* mode from st.context.theme rather than guessing from
# config.toml, otherwise the hand-styled page can disagree with st.dataframe,
# st.selectbox and st.code (the mismatch is what breaks light mode).
#
# st.context.theme is documented as read-only and may momentarily report the
# previous mode during a theme transition, so it is always treated as a hint
# with a config-based fallback.

def current_theme_base() -> str:
    """The configured base theme ("light" or "dark"), used as a fallback."""
    try:
        return (st.get_option("theme.base") or "dark").lower()
    except Exception:
        return "dark"


def resolved_theme_type() -> str | None:
    """The theme the client is actually rendering, or None when unavailable.

    Headless/AppTest runs have no browser context, so this returns None and
    callers fall back to the configured base theme.
    """
    try:
        value = getattr(st.context.theme, "type", None)
    except Exception:
        return None
    return value.lower() if isinstance(value, str) and value else None


def is_dark() -> bool:
    """True when the UI the user is looking at is in dark mode."""
    resolved = resolved_theme_type()
    if resolved in {"light", "dark"}:
        return resolved == "dark"
    return current_theme_base() == "dark"


def plot_template():
    """Plotly template matching the active app theme."""
    return "plotly_dark" if is_dark() else "plotly_white"


def style_plotly_fig(fig):
    """Ensure consistent dark/light styling on all Plotly figures."""
    dark = is_dark()
    tpl = "plotly_dark" if dark else "plotly_white"
    bg_color = "rgba(0,0,0,0)"
    text_color = "#F8FAFC" if dark else "#0F172A"
    grid_color = "rgba(148, 163, 184, 0.12)" if dark else "rgba(100, 116, 139, 0.12)"

    fig.update_layout(
        template=tpl,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family='"Inter", "Segoe UI", system-ui, sans-serif'),
    )
    try:
        fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
        fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    except Exception:
        pass
    return fig


def render_appearance_note() -> None:
    """Sidebar appearance hint pointing at Streamlit's native theme control."""
    st.markdown(
        '<div class="ci-side-label">Appearance</div>'
        '<div class="ci-theme-note">Follows your system theme unless pinned. '
        'Change it from the <b>⋮</b> menu → Settings → Theme.</div>',
        unsafe_allow_html=True,
    )

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

import streamlit as st


def _palette(dark):
    if dark:
        return {
            "__ACCENT__": "#3B82F6",
            "__ACCENT_HOVER__": "#60A5FA",
            "__TEXT__": "#E5E7EB",
            "__TEXT2__": "rgba(226, 232, 240, .80)",
            "__MUTED__": "rgba(148, 163, 184, .88)",
            "__BORDER__": "rgba(148, 163, 184, .26)",
            "__TINT__": "rgba(148, 163, 184, .09)",
            "__SURFACE__": "#111827",
            "__SHADOW__": "none",
        }
    return {
        "__ACCENT__": "#2563EB",
        "__ACCENT_HOVER__": "#1D4ED8",
        "__TEXT__": "#101828",
        "__TEXT2__": "rgba(52, 64, 84, .92)",
        "__MUTED__": "rgba(100, 116, 139, .95)",
        "__BORDER__": "rgba(120, 120, 135, .24)",
        "__TINT__": "rgba(128, 128, 140, .06)",
        "__SURFACE__": "#F7F8FA",
        "__SHADOW__": "0 1px 2px rgba(16, 24, 40, .06)",
    }


_CSS = """
:root {
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

/* ---- typography scale ------------------------------------------------ */
.stApp, .stApp p, .stApp li {
  font-family: var(--ci-font);
}
.stApp h1 {                       /* page titles via st.title */
  font-size: 1.45rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.25;
  margin-bottom: 0;
}
.stApp h2 {                       /* section headings via st.subheader */
  font-size: 1.02rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  padding-bottom: .35rem;
  border-bottom: 1px solid var(--ci-border);
}
.stApp h3 {
  font-size: .93rem;
  font-weight: 650;
}
.stApp p, .stApp li {
  font-size: .9rem;
  line-height: 1.55;
  color: var(--ci-text-2);
}
.stApp caption, .stApp [data-testid="stCaptionContainer"] p {
  font-size: .8rem;
  color: var(--ci-muted);
}
.stApp strong { color: var(--ci-text); }

/* ---- layout ------------------------------------------------------------ */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1180px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
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
  padding-top: 1rem;
}
section[data-testid="stSidebar"] hr { margin: .8rem 0; }
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
  font-size: .88rem;
  border-radius: var(--ci-radius-sm);
  padding: .3rem .55rem;
}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
  background: var(--ci-tint);
}
.ci-brand { margin-bottom: .25rem; user-select: none; }
.ci-brand-name {
  font-family: var(--ci-font);
  font-size: .98rem;
  font-weight: 750;
  letter-spacing: -0.02em;
  color: var(--ci-text);
}
.ci-brand-name span { color: var(--ci-accent); }
.ci-brand-tag {
  font-size: .74rem;
  color: var(--ci-muted);
  line-height: 1.4;
}
.ci-side-label {
  font-size: .68rem;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--ci-muted);
  margin-bottom: .15rem;
}
.ci-dataset {
  background: var(--ci-surface);
  border: 1px solid var(--ci-border);
  border-radius: var(--ci-radius);
  padding: .6rem .7rem;
  margin-top: .15rem;
}
.ci-dataset-name {
  font-size: .85rem;
  font-weight: 600;
  color: var(--ci-text);
  word-break: break-all;
}
.ci-dataset-meta {
  font-size: .76rem;
  color: var(--ci-muted);
  margin-top: .1rem;
}

/* ---- buttons: primary / secondary / tertiary hierarchy ------------------ */
.stButton > button {
  border-radius: var(--ci-radius-sm);
  border: 1px solid var(--ci-border);
  font-weight: 600;
  font-size: .87rem;
  height: 2.2rem;
  box-shadow: none;
  transition: background-color .12s ease, border-color .12s ease;
}
.stButton > button[kind="primary"] {
  background: var(--primary-color, var(--ci-accent));
  border-color: transparent;
  color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
  background: var(--ci-accent-hover);
  border-color: transparent;
  color: #FFFFFF;
}
.stButton > button:hover {
  border-color: rgba(120, 120, 135, .45);
  color: var(--ci-text);
}
.stButton > button:focus-visible {
  outline: 2px solid var(--ci-accent);
  outline-offset: 2px;
}

/* ---- metrics: quiet numbers instead of cards ----------------------------- */
[data-testid="stMetric"] {
  background: none;
  border: none;
  padding: .15rem 0 .5rem 0;
}
[data-testid="stMetricLabel"] > div {
  font-size: .72rem !important;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--ci-muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.32rem !important;
  font-weight: 700;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: var(--ci-text);
}
[data-testid="stMetricDelta"] > div[data-testid="stMetricDelta"] {
  font-size: .78rem !important;
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
footer [data-testid="stStatusWidget"], footer { display: none; }
"""


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
    return current_theme_base() == "dark"


def plot_template():
    """Plotly template matching the active app theme."""
    return "plotly_dark" if is_dark() else "plotly_white"


def toggle_theme_button():
    dark = is_dark()
    if st.button("Light mode" if dark else "Dark mode", key="ci_theme_toggle",
                 use_container_width=True):
        try:
            st._config.set_option("theme.base", "light" if dark else "dark")
        except Exception as error:
            # private-API risk made explicit: if the hook ever vanishes the
            # user still gets a hint instead of a button that does nothing
            from Utils.logsys import get_logger
            get_logger("theme").warning("runtime theme switch unavailable: %s", error)
            st.toast("Theme switching is unavailable in this Streamlit version; "
                     "use the Settings menu instead.")
        # rerun re-renders with the new base theme (st.rerun halts this script)
        st.rerun()

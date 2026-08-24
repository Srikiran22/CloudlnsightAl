import hashlib
import streamlit as st
from html import escape

from Utils.paths import list_dataset_files, read_dataset, resolve_dataset_path
from Utils.theme import toggle_theme_button


_BRAND_HTML = """
<div class="ci-brand">
  <div class="ci-brand-name">Cloud<span>Insight</span> AI</div>
  <div class="ci-brand-tag">Analytics workspace</div>
</div>
"""


def set_active_dataset(df, name):
    st.session_state["current_df"] = df
    st.session_state["dataset_name"] = name


def init_session_state():
    """Guarantee the app's core data contract keys exist with safe defaults.

    `current_df` / `dataset_name` are THE shared dataset state every page
    reads; all other session keys are page-local and initialize themselves.
    """
    if "current_df" not in st.session_state:
        st.session_state["current_df"] = None
    if "dataset_name" not in st.session_state:
        st.session_state["dataset_name"] = None


def dataset_fingerprint(dataset):
    """Cheap stable identity for a dataset FILE: name+size+mtime hash.

    Filenames alone cannot tell a rewritten file apart from the original;
    this 12-hex fingerprint can. It reads no file bytes, so it costs one
    stat call and is safe to compute on every rerun.
    """
    path = resolve_dataset_path(dataset)
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def results_match_active(results, selected_file):
    """True when stored analysis results still describe the active dataset.

    Guards against the same filename being replaced on disk between training
    and viewing. Results saved before fingerprints existed keep working on a
    name-only basis.
    """
    if not results or results.get("dataset_name") != selected_file:
        return False
    expected = results.get("dataset_fingerprint")
    if expected is None:
        return True
    try:
        return dataset_fingerprint(selected_file) == expected
    except OSError:
        return False


@st.cache_data(show_spinner="Loading dataset...", max_entries=64)
def _load_cached_dataset(path_str, mtime_ns):
    # mtime_ns is only here so the cache busts when the file changes on disk.
    # max_rows is deliberately NOT part of the key: the full frame is cached
    # once per file version and row limits are applied by callers afterwards.
    # max_entries bounds memory across many files/edits in one long session.
    return read_dataset(path_str)


def load_dataset_cached(dataset, max_rows=None):
    path = resolve_dataset_path(dataset)
    df = _load_cached_dataset(str(path), path.stat().st_mtime_ns)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df


def render_sidebar():
    with st.sidebar:
        st.markdown(_BRAND_HTML, unsafe_allow_html=True)
        name = st.session_state.get("dataset_name")
        df = st.session_state.get("current_df")
        st.markdown('<div class="ci-side-label">Active dataset</div>', unsafe_allow_html=True)
        if name:
            meta = f"{df.shape[0]:,} rows × {df.shape[1]} cols" if df is not None else "shape unavailable"
            st.markdown(
                f'<div class="ci-dataset">'
                f'<div class="ci-dataset-name">{escape(str(name))}</div>'
                f'<div class="ci-dataset-meta">{meta}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ci-dataset"><div class="ci-dataset-meta">Nothing loaded yet — start at '
                '<b>Ingest data</b>.</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="ci-side-label">Appearance</div>', unsafe_allow_html=True)
        toggle_theme_button()


def select_working_dataset(selectbox_label, max_rows=None):
    files = list_dataset_files()
    df_session = st.session_state.get("current_df")
    name_session = st.session_state.get("dataset_name")

    options = []
    if df_session is not None:
        label = f"Active Session: {name_session}" if name_session else "Active Session"
        options.append(label)
    options.extend(file for file in files if file != name_session)

    if not options:
        st.warning("No dataset loaded. Ingest one on the **Ingest data** page first.")
        st.stop()

    selected_option = st.selectbox(selectbox_label, options)
    if selected_option.startswith("Active Session"):
        return df_session, name_session or "Session Dataset"

    try:
        loaded_df = load_dataset_cached(selected_option, max_rows=max_rows)
        set_active_dataset(loaded_df, selected_option)
        return loaded_df, selected_option
    except Exception as error:
        st.error(f"Failed to load `{selected_option}`: {error}")
        st.stop()

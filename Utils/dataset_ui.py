import hashlib
import pandas as pd
import streamlit as st
from html import escape

from Utils.paths import (
    _sanitize_unhashable_cells,
    list_dataset_files,
    read_dataset,
    resolve_dataset_path,
)
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
    try:
        st.session_state["session_fingerprint"] = dataset_fingerprint(name) if name else None
    except Exception:
        st.session_state["session_fingerprint"] = None


def init_session_state():
    """Guarantee the app's core data contract keys exist with safe defaults.

    `current_df` / `dataset_name` are THE shared dataset state every page
    reads; all other session keys are page-local and initialize themselves.
    """
    if "current_df" not in st.session_state:
        st.session_state["current_df"] = None
    if "dataset_name" not in st.session_state:
        st.session_state["dataset_name"] = None


_CONTENT_HASH_CACHE = {}


def invalidate_dataset_cache(dataset=None):
    """Clear cached content fingerprints.

    If dataset is provided, clears entries for that specific dataset path.
    If None, clears the entire fingerprint cache. Also clears Streamlit data cache.
    """
    if dataset is None:
        _CONTENT_HASH_CACHE.clear()
        try:
            _load_cached_dataset.clear()
            _load_cached_dataset_bounded.clear()
        except Exception:
            pass
    else:
        try:
            target_path_str = str(resolve_dataset_path(dataset))
            for key in list(_CONTENT_HASH_CACHE.keys()):
                if key[0] == target_path_str:
                    _CONTENT_HASH_CACHE.pop(key, None)
            try:
                _load_cached_dataset.clear()
                _load_cached_dataset_bounded.clear()
            except Exception:
                pass
        except Exception:
            pass


def dataset_fingerprint(dataset, force_refresh=False):
    """Cryptographic content-derived fingerprint for a dataset file.

    Computes a full SHA-256 hash over the actual file content to guarantee
    exact content identity, with metadata-based caching (mtime_ns, size)
    to avoid redundant disk reads when the file is unchanged.
    """
    path = resolve_dataset_path(dataset)
    path_str = str(path)
    stat = path.stat()
    cache_key = (path_str, stat.st_mtime_ns, stat.st_size)

    if not force_refresh and cache_key in _CONTENT_HASH_CACHE:
        return _CONTENT_HASH_CACHE[cache_key]

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(64 * 1024):
            hasher.update(chunk)

    fp = hasher.hexdigest()

    for key in list(_CONTENT_HASH_CACHE.keys()):
        if key[0] == path_str:
            _CONTENT_HASH_CACHE.pop(key, None)

    if len(_CONTENT_HASH_CACHE) >= 128:
        _CONTENT_HASH_CACHE.pop(next(iter(_CONTENT_HASH_CACHE)), None)

    _CONTENT_HASH_CACHE[cache_key] = fp
    return fp


def dataframe_fingerprint(df):
    """Deterministic cryptographic content fingerprint for an in-memory DataFrame.

    Incorporates shape, column names, column order, data types, and all row values.
    """
    if df is None:
        return ""
    meta = f"{df.shape}|{list(df.columns)}|{list(df.dtypes.astype(str))}"
    hasher = hashlib.sha256(meta.encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False)
        hasher.update(row_hashes.values.tobytes())
    except TypeError:
        try:
            sanitized = _sanitize_unhashable_cells(df.copy(deep=False))
            row_hashes = pd.util.hash_pandas_object(sanitized, index=False)
            hasher.update(row_hashes.values.tobytes())
        except Exception:
            for col in df.columns:
                hasher.update(df[col].astype(str).str.cat(sep=",").encode("utf-8", errors="replace"))
    except Exception:
        for col in df.columns:
            hasher.update(df[col].astype(str).str.cat(sep=",").encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def results_match_active(results, selected_file, df=None):
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
        path = resolve_dataset_path(selected_file)
        if path.is_file():
            return dataset_fingerprint(selected_file) == expected
    except (ValueError, OSError):
        pass
    if df is not None:
        return dataframe_fingerprint(df) == expected
    return False


@st.cache_data(show_spinner="Loading dataset...", max_entries=64)
def _load_cached_dataset(path_str, fingerprint):
    return read_dataset(path_str)


@st.cache_data(show_spinner="Loading dataset subset...", max_entries=64)
def _load_cached_dataset_bounded(path_str, fingerprint, max_rows):
    return read_dataset(path_str, max_rows=max_rows)


def load_dataset_cached(dataset, max_rows=None):
    path = resolve_dataset_path(dataset)
    fp = dataset_fingerprint(dataset)
    if max_rows is None:
        return _load_cached_dataset(str(path), fp)
    return _load_cached_dataset_bounded(str(path), fp, max_rows)


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
    options.extend(files)

    if not options:
        st.warning("No dataset loaded. Ingest one on the **Ingest data** page first.")
        st.stop()

    selected_option = st.selectbox(selectbox_label, options)
    # positional comparison: a stored FILE may itself be named like the
    # session label, so startswith() would misroute it and yield no dataframe
    if df_session is not None and selected_option == options[0]:
        if name_session and name_session in files:
            try:
                disk_fp = dataset_fingerprint(name_session)
                session_fp = st.session_state.get("session_fingerprint")
                if session_fp and disk_fp != session_fp:
                    st.info(
                        f"Notice: `{name_session}` on disk was modified since it was loaded into this session. "
                        "Pick the file from the dropdown to reload fresh data from disk."
                    )
            except Exception:
                pass
        return df_session, name_session or "Session Dataset"

    try:
        loaded_df = load_dataset_cached(selected_option, max_rows=max_rows)
        if df_session is None:
            set_active_dataset(loaded_df, selected_option)
        else:
            if hasattr(st, "info"):
                st.info(
                    f"Viewing stored dataset `{selected_option}`. Active session dataset (`{name_session}`) is preserved."
                )
            if hasattr(st, "button") and st.button(
                f"Set `{selected_option}` as active working dataset",
                key=f"set_active_{selected_option}",
            ):
                set_active_dataset(loaded_df, selected_option)
                if hasattr(st, "rerun"):
                    st.rerun()
        return loaded_df, selected_option
    except Exception as error:
        st.error(f"Failed to load `{selected_option}`: {error}")
        st.stop()

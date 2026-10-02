import hashlib
import threading
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


class DatasetIdentity:
    """Unified dataset identity binding source type, name, and cryptographic fingerprint."""

    def __init__(self, dataset_name, fingerprint=None, source_type="disk", shape=None, path=None):
        self.dataset_name = dataset_name
        self.fingerprint = fingerprint
        self.source_type = source_type  # "session" or "disk"
        self.shape = shape
        self.path = path

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.fingerprint,
            "source_type": self.source_type,
            "shape": list(self.shape) if self.shape else None,
        }


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
    """Clear cached content fingerprints and memory-cached DataFrames.

    If dataset is provided, clears entries for that specific dataset path.
    If None, clears the entire fingerprint cache and bounded memory cache.
    """
    if dataset is None:
        _CONTENT_HASH_CACHE.clear()
        try:
            _DATASET_MEMORY_CACHE.clear()
        except Exception:
            pass
    else:
        try:
            target_path_str = str(resolve_dataset_path(dataset))
            for key in list(_CONTENT_HASH_CACHE.keys()):
                if key[0] == target_path_str:
                    _CONTENT_HASH_CACHE.pop(key, None)
            _DATASET_MEMORY_CACHE.invalidate(target_path_str)
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
                col_bytes = b"".join(f"{len(s)}:{s}".encode("utf-8", errors="replace") for s in df[col].astype(str))
                hasher.update(col_bytes)
    except Exception:
        for col in df.columns:
            col_bytes = b"".join(f"{len(s)}:{s}".encode("utf-8", errors="replace") for s in df[col].astype(str))
            hasher.update(col_bytes)
    return hasher.hexdigest()


def results_match_active(results, selected_file, df=None):
    """True when stored analysis results still describe the active dataset.

    Guards against the same filename being replaced on disk between training
    and viewing. Prioritizes active in-memory session datasets when present.
    """
    if not results:
        return False
    res_name = results.get("dataset_name")
    if res_name != selected_file and not (selected_file and selected_file.startswith("Active Session") and res_name in selected_file):
        return False
    expected = results.get("dataset_fingerprint")
    if expected is None:
        return True

    source_type = results.get("source_type")
    if df is not None:
        actual_df_fp = dataframe_fingerprint(df)
        if actual_df_fp == expected:
            return True
        if source_type == "session":
            return False

    try:
        path = resolve_dataset_path(selected_file)
        if path.is_file():
            return dataset_fingerprint(selected_file, force_refresh=True) == expected
    except (ValueError, OSError):
        pass

    return False


MAX_DATASET_CACHE_BYTES = 256 * 1024 * 1024  # 256 MB hard RAM budget for cached datasets


class BoundedDatasetMemoryCache:
    """LRU dataset cache enforcing a hard total byte memory budget.

    Tracks deep in-memory DataFrame size via memory_usage(deep=True) and evicts
    oldest datasets when total memory exceeds the budget. Single datasets that
    exceed the budget are not cached to prevent memory exhaustion.
    """

    def __init__(self, max_bytes=MAX_DATASET_CACHE_BYTES):
        import collections
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._cache = collections.OrderedDict()
        self._lock = threading.RLock()

    def _estimate_size(self, df: pd.DataFrame) -> int:
        try:
            return int(df.memory_usage(index=True, deep=True).sum())
        except Exception:
            return int(df.shape[0] * df.shape[1] * 8)

    def get(self, key):
        with self._lock:
            if key in self._cache:
                df, size = self._cache.pop(key)
                self._cache[key] = (df, size)
                # Copy-on-read contract: callers receive isolated copy, preserving fingerprint identity
                return df.copy(deep=True)
            return None

    def put(self, key, df: pd.DataFrame):
        with self._lock:
            size = self._estimate_size(df)
            if size > self.max_bytes:
                return

            cached_df = df.copy(deep=True)
            if key in self._cache:
                _, old_size = self._cache.pop(key)
                self.current_bytes -= old_size

            while self.current_bytes + size > self.max_bytes and self._cache:
                _, (_, evicted_size) = self._cache.popitem(last=False)
                self.current_bytes -= evicted_size

            self._cache[key] = (cached_df, size)
            self.current_bytes += size

    def invalidate(self, path_str):
        with self._lock:
            for key in list(self._cache.keys()):
                if key[0] == path_str:
                    _, size = self._cache.pop(key)
                    self.current_bytes -= size

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.current_bytes = 0

    def stats(self):
        with self._lock:
            return {
                "current_bytes": self.current_bytes,
                "max_bytes": self.max_bytes,
                "entry_count": len(self._cache),
            }


_DATASET_MEMORY_CACHE = BoundedDatasetMemoryCache()


def get_cache_memory_stats():
    return _DATASET_MEMORY_CACHE.stats()


def clear_dataset_memory_cache():
    _DATASET_MEMORY_CACHE.clear()


def load_dataset_cached(dataset, max_rows=None):
    path = resolve_dataset_path(dataset)
    fp = dataset_fingerprint(dataset)
    cache_key = (str(path), fp, max_rows)
    cached = _DATASET_MEMORY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    loaded = read_dataset(path, max_rows=max_rows)
    _DATASET_MEMORY_CACHE.put(cache_key, loaded)
    return loaded



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
    files = list_dataset_files(tabular_only=True)
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
        ret_df = df_session
        if max_rows is not None and len(ret_df) > max_rows:
            ret_df = ret_df.iloc[:max_rows].copy()
        return ret_df, name_session or "Session Dataset"

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

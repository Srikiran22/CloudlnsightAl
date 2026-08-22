import streamlit as st

from Utils.paths import list_dataset_files, read_dataset, resolve_dataset_path


def set_active_dataset(df, name):
    st.session_state["current_df"] = df
    st.session_state["dataset_name"] = name


@st.cache_data(show_spinner="Loading dataset...")
def _load_cached_dataset(path_str, mtime_ns, max_rows):
    # mtime_ns is only here so the cache busts when the file changes on disk
    del mtime_ns
    return read_dataset(path_str)


def load_dataset_cached(dataset, max_rows=None):
    path = resolve_dataset_path(dataset)
    df = _load_cached_dataset(str(path), path.stat().st_mtime_ns, max_rows)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df


def render_sidebar():
    with st.sidebar:
        st.markdown("### ☁️ CloudInsight AI")
        name = st.session_state.get("dataset_name")
        df = st.session_state.get("current_df")
        if name:
            st.success(f"📂 **Active Dataset:**\n`{name}`")
            if df is not None:
                st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
        else:
            st.info("ℹ️ No dataset loaded yet. Go to **Upload** to begin.")


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
        st.warning("⚠️ No dataset loaded. Please upload a dataset in the **Upload** page first.")
        st.stop()

    selected_option = st.selectbox(selectbox_label, options)
    if selected_option.startswith("Active Session"):
        return df_session, name_session or "Session Dataset"

    try:
        loaded_df = load_dataset_cached(selected_option, max_rows=max_rows)
        set_active_dataset(loaded_df, selected_option)
        return loaded_df, selected_option
    except Exception as error:
        st.error(f"❌ Failed to load `{selected_option}`: {error}")
        st.stop()

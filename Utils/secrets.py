# Secrets live in st.session_state only. Nothing here ever touches disk.
#
# ask() renders a password field and mirrors what the user types back into
# the session slot, so the value survives reruns while the page needs it.
# release() gets called once the work that needed the secret has finished;
# it wipes the slot(s) again unless the user ticked the keep box.

import streamlit as st


def _slot(name):
    return f"{name}_secret"


def ask(name, label, help_text=""):
    slot = _slot(name)
    if slot not in st.session_state:
        st.session_state[slot] = ""
    value = st.text_input(label, value=st.session_state[slot], type="password", help=help_text)
    st.session_state[slot] = value
    return value


def value_of(name):
    return st.session_state.get(_slot(name)) or ""


def keep_box(session_key):
    return st.checkbox(
        "Keep in memory for this session",
        key=session_key,
        help="Unticked: the secret is wiped from memory as soon as the current task finishes.",
    )


def release(*names, **opts):
    keep_key = opts.get("keep_key")
    if keep_key and st.session_state.get(keep_key):
        return []

    wiped = []
    for name in names:
        old = st.session_state.pop(_slot(name), None)
        if old not in (None, ""):
            wiped.append(name)
    return wiped


def drop(*names):
    for name in names:
        st.session_state.pop(_slot(name), None)

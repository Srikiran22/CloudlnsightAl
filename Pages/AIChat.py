import hashlib

import streamlit as st

from Utils.ai_agent import run_local_deterministic_query, run_neuro_symbolic_query
from Utils.dataset_ui import dataframe_fingerprint, dataset_fingerprint, render_sidebar, select_working_dataset
from Utils.Gemini import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    MAX_CHAT_HISTORY,
    MAX_MESSAGE_CHARS,
    prune_ai_contexts,
)
from Utils.logsys import get_logger
from Utils.privacy import apply_exclusions, detect_sensitive_columns
from Utils.secrets import ask, keep_box, release, value_of
from Utils.theme import page_header, section_card_header

logger = get_logger("AIChat")

MAX_TOTAL_CHAT_CHARS = 100_000

page_header(
    "Ask AI (Chat)",
    "Statistical natural language dialogue and diagnostic queries over screened dataset context.",
    badge="GOVERNED CHAT"
)

df, selected_file = select_working_dataset("Active dataset")
render_sidebar()

try:
    active_fp = dataset_fingerprint(selected_file)
except Exception:
    active_fp = dataframe_fingerprint(df)

with st.sidebar:
    st.markdown('<div class="ci-side-label">Gemini settings</div>', unsafe_allow_html=True)
    ask(
        "gemini",
        "Google Gemini API key",
        help_text="Runtime key held in ephemeral memory only."
    )
    keep_box("gemini_keep")

    chosen_model = st.selectbox(
        "Gemini model",
        GEMINI_MODELS,
        index=GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL),
        key="aichat_model_select"
    )

api_key = value_of("gemini")

# Privacy screening
sensitive = detect_sensitive_columns(df)
all_cols = list(df.columns)
default_exclusions = [c for c in all_cols if c in sensitive]

with st.container(border=True):
    section_card_header(
        "Data Privacy & Context Protection",
        f"PII detection and column context boundary for {selected_file}"
    )
    if api_key:
        st.success("Gemini API key active — Neuro-Symbolic Safe Code Execution engine enabled.")
    else:
        st.info("Local Deterministic Engine active (Air-Gapped). Queries execute locally with zero cloud dependencies. Enter Gemini API key in sidebar to enable neuro-symbolic code generation.")

    gov_col1, gov_col2 = st.columns([1.5, 1])
    with gov_col1:
        excluded_cols = st.multiselect(
            "Columns to exclude from chat context",
            options=all_cols,
            default=default_exclusions,
            help="Excluded columns are stripped from anything sent to Gemini.",
            key="aichat_excluded_cols"
        )
    with gov_col2:
        if sensitive:
            reasons = ", ".join(f"`{col}`" for col in sorted(sensitive.keys()))
            st.info(f"PII screening: {len(sensitive)} sensitive/identifier column(s) detected ({reasons}) and pre-excluded.")
        else:
            st.success("PII screening: no identifiers or sensitive patterns matched the current rules.")

    st.caption(
        "**Privacy notice:** Only statistical summaries and non-excluded sample records are transmitted to Google Gemini. "
        "Privacy screening uses heuristics and cannot guarantee that all sensitive data will be detected."
    )

ai_df, exclusions_applied = apply_exclusions(df, excluded_cols)
if not exclusions_applied and excluded_cols:
    st.error(
        "All columns were selected for exclusion. Deselect some non-sensitive columns to proceed."
    )
    st.stop()

excl_sig = ",".join(sorted(excluded_cols))
ctx_str = f"chat_{selected_file}|{active_fp}|{chosen_model}|{excl_sig}|v2"
ctx_hash = hashlib.sha256(ctx_str.encode("utf-8")).hexdigest()[:16]
chat_key = f"dedicated_chat_messages_{ctx_hash}"

prune_ai_contexts(st.session_state, ctx_hash)

if chat_key not in st.session_state:
    st.session_state[chat_key] = []

with st.container(border=True):
    section_card_header("Guided Diagnostic Inquiries", "Select a structured analytical inquiry to evaluate dataset properties")
    btn_q1, btn_q2, btn_q3 = st.columns(3)
    clicked_question = None
    with btn_q1:
        if st.button("Analyze feature covariance structures and identify potential collinearity bottlenecks.", key="q1_btn", use_container_width=True):
            clicked_question = "Analyze feature covariance structures and identify potential collinearity bottlenecks."
    with btn_q2:
        if st.button("Evaluate missingness mechanisms, sparsity patterns, and extreme value outliers.", key="q2_btn", use_container_width=True):
            clicked_question = "Evaluate missingness mechanisms, sparsity patterns, and extreme value outliers."
    with btn_q3:
        if st.button("Assess target distribution, class balance, and candidate predictor signal.", key="q3_btn", use_container_width=True):
            clicked_question = "Assess target distribution, class balance, and candidate predictor signal."

with st.container(border=True):
    section_card_header("Conversation History", f"Chat with {selected_file}")
    if not st.session_state[chat_key]:
        st.info("Session initialized. Submit a query below or select a diagnostic question above.")
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            if msg.get("engine"):
                st.caption(f"**Execution Engine:** `{msg['engine']}`")
            st.markdown(msg["content"])
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)
            if msg.get("result_df") is not None and not msg["result_df"].empty:
                st.dataframe(msg["result_df"], use_container_width=True)
            if msg.get("code"):
                with st.expander("Inspected & Verified Python Code"):
                    st.code(msg["code"], language="python")

user_prompt = st.chat_input("Enter analytical query or statistical hypothesis...")
prompt_to_run = clicked_question or user_prompt

if prompt_to_run:
    if len(prompt_to_run) > MAX_MESSAGE_CHARS:
        prompt_to_run = f"{prompt_to_run[:MAX_MESSAGE_CHARS]}... [truncated]"

    st.session_state[chat_key].append({"role": "user", "content": prompt_to_run})
    del st.session_state[chat_key][:-MAX_CHAT_HISTORY]

    with st.chat_message("user"):
        st.markdown(prompt_to_run)

    with st.chat_message("assistant"):
        spinner_msg = (
            "Generating sandboxed code & evaluating query via Gemini..."
            if api_key
            else "Executing local statistical diagnostics (Air-Gapped)..."
        )
        with st.spinner(spinner_msg):
            try:
                if api_key:
                    res = run_neuro_symbolic_query(
                        df=ai_df,
                        query=prompt_to_run,
                        api_key=api_key,
                        model_name=chosen_model,
                        dataset_name=selected_file,
                    )
                else:
                    res = run_local_deterministic_query(df=ai_df, query=prompt_to_run)

                st.caption(f"**Execution Engine:** `{res.engine}`")
                st.markdown(res.insights)
                if res.fig is not None:
                    st.plotly_chart(res.fig, use_container_width=True)
                if res.result_df is not None and not res.result_df.empty:
                    st.dataframe(res.result_df, use_container_width=True)
                if res.code:
                    with st.expander("Inspected & Verified Python Code"):
                        st.code(res.code, language="python")

                assistant_msg = {
                    "role": "assistant",
                    "content": res.insights,
                    "engine": res.engine,
                    "code": res.code,
                    "fig": res.fig,
                    "result_df": res.result_df,
                    "result_metric": res.result_metric,
                }
                st.session_state[chat_key].append(assistant_msg)
                del st.session_state[chat_key][:-MAX_CHAT_HISTORY]
            except Exception as e:
                logger.warning("Query execution failed: %s: %s", type(e).__name__, e)
                st.error(f"Execution Error: {str(e)}")
            finally:
                if api_key and release("gemini", keep_key="gemini_keep"):
                    st.toast("Gemini key released from session state.")

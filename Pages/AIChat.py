import hashlib
import streamlit as st

from Utils.Gemini import (
    chat_with_gemini_dataset,
    GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError, MAX_CHAT_HISTORY,
    MAX_MESSAGE_CHARS, prune_ai_contexts,
)
from Utils.privacy import apply_exclusions, detect_sensitive_columns
from Utils.secrets import ask, value_of, keep_box, release
from Utils.logsys import get_logger
from Utils.dataset_ui import dataframe_fingerprint, dataset_fingerprint, render_sidebar, select_working_dataset
from Utils.theme import page_header, section_card_header

logger = get_logger("AIChat")

MAX_TOTAL_CHAT_CHARS = 100_000

page_header(
    "Ask AI (Chat)",
    "Interactive conversational Q&A with your dataset in plain English, powered by Google Gemini."
)

df, selected_file = select_working_dataset("Active Dataset:")
render_sidebar()

try:
    active_fp = dataset_fingerprint(selected_file)
except Exception:
    active_fp = dataframe_fingerprint(df)

with st.sidebar:
    st.markdown('<div class="ci-side-label">Gemini settings</div>', unsafe_allow_html=True)
    ask(
        "gemini",
        "Enter Google Gemini API Key:",
        help_text="Runtime key held in ephemeral memory only."
    )
    keep_box("gemini_keep")

    chosen_model = st.selectbox(
        "Gemini Model:",
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
    gov_col1, gov_col2 = st.columns([1.2, 1.8])
    with gov_col1:
        if api_key:
            st.success("Gemini API Key: Active in Session")
        else:
            st.warning("Gemini API Key: Required to ask questions")
            ask(
                "gemini",
                "Enter Gemini API Key (or set in sidebar):",
                help_text="Key is held in volatile session memory only. Get a free key at https://aistudio.google.com/",
                key="inline_gemini_key_chat",
            )
            api_key = value_of("gemini")

        if sensitive:
            reasons = ", ".join(f"`{col}`" for col in sorted(sensitive.keys()))
            st.info(f"PII Protection: {len(sensitive)} sensitive column(s) detected ({reasons}) and pre-excluded.")
        else:
            st.success("PII Protection: No high-risk identifiers detected in active schema.")

        st.caption("*Only statistical summaries and non-excluded sample records are transmitted to Google Gemini.*")

    with gov_col2:
        excluded_cols = st.multiselect(
            "Columns to EXCLUDE from chat context:",
            options=all_cols,
            default=default_exclusions,
            help="Excluded columns are stripped from anything sent to Gemini.",
            key="aichat_excluded_cols"
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
    section_card_header("Starter Questions for Beginners", "Click any prompt to ask immediately")
    btn_q1, btn_q2, btn_q3 = st.columns(3)
    clicked_question = None
    with btn_q1:
        if st.button("What are the 3 most important patterns in this dataset?", key="q1_btn", use_container_width=True):
            clicked_question = "What are the 3 most important patterns in this dataset?"
    with btn_q2:
        if st.button("Are there any data quality issues or unusual outliers I should fix?", key="q2_btn", use_container_width=True):
            clicked_question = "Are there any data quality issues or unusual outliers I should fix?"
    with btn_q3:
        if st.button("What are the most important relationships between columns?", key="q3_btn", use_container_width=True):
            clicked_question = "What are the most important relationships between columns?"

with st.container(border=True):
    section_card_header("Conversation History", f"Chat with {selected_file}")
    if not st.session_state[chat_key]:
        st.info("No messages yet. Ask a question below or click one of the starter questions above to begin!")
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_prompt = st.chat_input("Ask a question about your dataset...")
prompt_to_run = clicked_question or user_prompt

if prompt_to_run:
    if not api_key:
        st.error("Google Gemini API key is required to chat. Enter your key in the sidebar.")
    else:
        if len(prompt_to_run) > MAX_MESSAGE_CHARS:
            prompt_to_run = f"{prompt_to_run[:MAX_MESSAGE_CHARS]}... [truncated]"

        st.session_state[chat_key].append({"role": "user", "content": prompt_to_run})
        del st.session_state[chat_key][:-MAX_CHAT_HISTORY]

        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Gemini is analyzing your data..."):
                try:
                    reply = chat_with_gemini_dataset(
                        api_key=api_key,
                        df=ai_df,
                        dataset_name=selected_file,
                        messages=st.session_state[chat_key],
                        model_name=chosen_model
                    )
                    st.markdown(reply)
                    st.session_state[chat_key].append({"role": "assistant", "content": reply})
                    del st.session_state[chat_key][:-MAX_CHAT_HISTORY]
                except GeminiError as e:
                    st.error(f"Gemini error — {e}")
                except Exception as e:
                    logger.warning("chat failed: %s: %s", type(e).__name__, e)
                    st.error(f"Error: {str(e)}")
                finally:
                    if release("gemini", keep_key="gemini_keep"):
                        st.toast("Gemini key released from session state.")

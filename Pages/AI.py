import hashlib
import streamlit as st

from Utils.Gemini import (
    generate_executive_insights, chat_with_gemini_dataset,
    GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError, MAX_CHAT_HISTORY,
    MAX_MESSAGE_CHARS, prune_ai_contexts,
)

from Utils.privacy import apply_exclusions, detect_sensitive_columns
from Utils.secrets import ask, value_of, keep_box, release, drop
from Utils.logsys import get_logger
from Utils.dataset_ui import dataframe_fingerprint, dataset_fingerprint, render_sidebar, select_working_dataset
from Utils.theme import page_header, section_card_header

logger = get_logger("AI")

MAX_TOTAL_CHAT_CHARS = 100_000

page_header(
    "AI Summary & Insights",
    "Executive summaries, natural language diagnostics, and statistical insights over the active dataset."
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
        help_text="Get your key at https://aistudio.google.com/"
    )
    keep_box("gemini_keep")
    if st.button("Clear key from session state"):
        drop("gemini")
        st.rerun()

    chosen_model = st.selectbox(
        "Gemini Model:",
        GEMINI_MODELS,
        index=GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL)
    )

api_key = value_of("gemini")

# Privacy screening: detect sensitive columns
sensitive = detect_sensitive_columns(df)
all_cols = list(df.columns)
default_exclusions = [c for c in all_cols if c in sensitive]

# Consolidated AI Governance & Privacy Shield
with st.container(border=True):
    section_card_header(
        "AI Governance & Privacy Shield",
        f"Data governance, PII protection, and context boundary for {selected_file}"
    )
    gov_col1, gov_col2 = st.columns([1.2, 1.8])
    with gov_col1:
        if api_key:
            st.success("Gemini API Key: Active in Session")
        else:
            st.warning("Gemini API Key: Required to generate insights")
            ask(
                "gemini",
                "Enter Gemini API Key (or set in sidebar):",
                help_text="Key is held in volatile session memory only. Get a free key at https://aistudio.google.com/",
                key="inline_gemini_key_ai",
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
            "Columns to EXCLUDE from AI context:",
            options=all_cols,
            default=default_exclusions,
            help="Excluded columns are stripped from everything sent to Gemini. Detected sensitive columns are preselected automatically.",
        )

ai_df, exclusions_applied = apply_exclusions(df, excluded_cols)
if not exclusions_applied and excluded_cols:
    st.error(
        "All columns were selected for exclusion. Sending zero columns makes AI analysis "
        "impossible, and sending the original data would violate privacy exclusions. "
        "Deselect some non-sensitive columns to proceed."
    )
    st.stop()
if exclusions_applied:
    logger.info("AI context excludes %d flagged column(s)", len(excluded_cols))

excl_sig = ",".join(sorted(excluded_cols))
ctx_str = f"{selected_file}|{active_fp}|{chosen_model}|{excl_sig}|v2"
ctx_hash = hashlib.sha256(ctx_str.encode("utf-8")).hexdigest()[:16]

insights_key = f"insights_{ctx_hash}"
chat_key = f"chat_messages_{ctx_hash}"

prune_ai_contexts(st.session_state, ctx_hash)


def render_executive_summary():
    with st.container(border=True):
        section_card_header(
            "Executive Data Summary",
            "Generate an automated data health audit, pattern discovery, and actionable findings"
        )

        if st.button("Generate insights report", type="primary"):
            if not api_key:
                st.error("Google Gemini API key is required to generate new insights. Enter your key in the sidebar.")
            else:
                with st.spinner("Gemini is analyzing your dataset structure, metrics, and distributions..."):
                    try:
                        insights_text = generate_executive_insights(
                            api_key=api_key,
                            df=ai_df,
                            dataset_name=selected_file,
                            model_name=chosen_model
                        )
                        st.session_state[insights_key] = insights_text
                        st.session_state["latest_ai_insights"] = {
                            "dataset_name": selected_file,
                            "dataset_fingerprint": active_fp,
                            "model_name": chosen_model,
                            "excluded_cols": sorted(list(excluded_cols)),
                            "ctx_hash": ctx_hash,
                            "text": insights_text,
                        }
                        st.session_state[f"insights_{selected_file}_{active_fp}"] = insights_text
                    except GeminiError as e:
                        st.error(f"Gemini error — {e}")
                    except Exception as e:
                        logger.warning("insights generation failed: %s: %s", type(e).__name__, e)
                        st.error(f"Gemini error: {str(e)}")
                    finally:
                        if release("gemini", keep_key="gemini_keep"):
                            st.toast("Gemini key released from session state.")

        saved_insights = st.session_state.get(insights_key)
        if saved_insights:
            st.markdown(saved_insights)
            st.download_button(
                label="Download insights (Markdown)",
                data=saved_insights.encode("utf-8"),
                file_name=f"ai_insights_{selected_file}.md",
                mime="text/markdown"
            )


def render_chat_tab():
    st.markdown(
        "Ask questions about your dataset based on column structures, summary metrics, "
        "and sample records."
    )

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a question about your dataset..."):
        if not api_key:
            st.error("Google Gemini API key is required to chat. Enter your key in the sidebar.")
        else:
            if len(user_prompt) > MAX_MESSAGE_CHARS:
                user_prompt = f"{user_prompt[:MAX_MESSAGE_CHARS]}... [truncated]"
            st.session_state[chat_key].append({"role": "user", "content": user_prompt})
            del st.session_state[chat_key][:-MAX_CHAT_HISTORY]
            total_chars = sum(len(m.get("content", "")) for m in st.session_state[chat_key])
            while total_chars > MAX_TOTAL_CHAT_CHARS and len(st.session_state[chat_key]) > 1:
                evicted = st.session_state[chat_key].pop(0)
                total_chars -= len(evicted.get("content", ""))
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
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
                        total_chars = sum(len(m.get("content", "")) for m in st.session_state[chat_key])
                        while total_chars > MAX_TOTAL_CHAT_CHARS and len(st.session_state[chat_key]) > 1:
                            evicted = st.session_state[chat_key].pop(0)
                            total_chars -= len(evicted.get("content", ""))
                    except GeminiError as e:
                        st.error(f"Gemini error — {e}")
                    except Exception as e:
                        logger.warning("chat failed: %s: %s", type(e).__name__, e)
                        st.error(f"Error: {str(e)}")
                    finally:
                        if release("gemini", keep_key="gemini_keep"):
                            st.toast("Gemini key released from session state.")


has_chat_history = bool(st.session_state.get(chat_key))
if has_chat_history:
    tab_insights, tab_chat = st.tabs(["Executive summary", "Chat"])
    with tab_insights:
        render_executive_summary()
    with tab_chat:
        render_chat_tab()
else:
    render_executive_summary()

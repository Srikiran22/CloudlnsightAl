import hashlib
import streamlit as st

from Utils.Gemini import (
    generate_executive_insights, chat_with_gemini_dataset,
    GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError, MAX_CHAT_HISTORY,
    prune_ai_contexts,
)
from Utils.privacy import apply_exclusions, detect_sensitive_columns
from Utils.secrets import ask, value_of, keep_box, release, drop
from Utils.logsys import get_logger
from Utils.dataset_ui import dataframe_fingerprint, dataset_fingerprint, render_sidebar, select_working_dataset

logger = get_logger("AI")

st.title("AI insights")
st.markdown("Gemini-powered executive summaries and conversational Q&A over the active dataset.")

df, selected_file = select_working_dataset("Select Dataset for AI Analysis:")
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

if not api_key:
    st.info("Enter your **Google Gemini API key** in the sidebar to generate new AI insights or send chat messages.")

st.caption(f"Analyzing: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} cols)")
st.warning(
    "Privacy note: generating insights or asking a question sends a compact dataset summary and sample rows "
    "to Google Gemini. Use de-identified data when it contains sensitive information."
)

# privacy screening: let the user exclude likely-sensitive columns from the
# AI context entirely; everything below uses ai_df instead of df
st.caption(
    "Automated privacy screening: uses pattern heuristics to detect likely-sensitive columns "
    "(emails, tokens, credit cards, phones, IBANs). This does not replace human data classification."
)

sensitive = detect_sensitive_columns(df)
if sensitive:
    reasons = ", ".join(f"`{col}` ({reason})" for col, reason in sorted(sensitive.items()))
    st.warning(f"Possible sensitive columns detected: {reasons}.")

all_cols = list(df.columns)
default_exclusions = [c for c in all_cols if c in sensitive]
excluded_cols = st.multiselect(
    "Columns to EXCLUDE from AI context:",
    options=all_cols,
    default=default_exclusions,
    help="Excluded columns are removed from everything sent to Gemini. Detected sensitive columns are preselected automatically.",
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

tab_insights, tab_chat = st.tabs(["Executive report", "Chat"])

excl_sig = ",".join(sorted(excluded_cols))
ctx_str = f"{selected_file}|{active_fp}|{chosen_model}|{excl_sig}|v2"
ctx_hash = hashlib.sha256(ctx_str.encode("utf-8")).hexdigest()[:16]

insights_key = f"insights_{ctx_hash}"
chat_key = f"chat_messages_{ctx_hash}"

prune_ai_contexts(st.session_state, ctx_hash)

with tab_insights:
    st.markdown("Generate a data health audit, pattern discovery, and business recommendations.")

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

with tab_chat:
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
            st.session_state[chat_key].append({"role": "user", "content": user_prompt})
            del st.session_state[chat_key][:-MAX_CHAT_HISTORY]
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
                    except GeminiError as e:
                        st.error(f"Gemini error — {e}")
                    except Exception as e:
                        logger.warning("chat failed: %s: %s", type(e).__name__, e)
                        st.error(f"Error: {str(e)}")
                    finally:
                        if release("gemini", keep_key="gemini_keep"):
                            st.toast("Gemini key released from session state.")

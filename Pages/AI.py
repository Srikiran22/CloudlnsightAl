import streamlit as st

from Utils.Gemini import (
    generate_executive_insights, chat_with_gemini_dataset,
    GEMINI_MODELS, DEFAULT_GEMINI_MODEL, GeminiError, MAX_CHAT_HISTORY,
)
from Utils.privacy import apply_exclusions, detect_sensitive_columns
from Utils.secrets import ask, value_of, keep_box, release, drop
from Utils.logsys import get_logger
from Utils.dataset_ui import render_sidebar, select_working_dataset

logger = get_logger("AI")

st.title("AI insights")
st.markdown("Gemini-powered executive summaries and conversational Q&A over the active dataset.")

df, selected_file = select_working_dataset("Select Dataset for AI Analysis:")
render_sidebar()

with st.sidebar:
    st.markdown('<div class="ci-side-label">Gemini settings</div>', unsafe_allow_html=True)
    ask(
        "gemini",
        "Enter Google Gemini API Key:",
        help_text="Get your key at https://aistudio.google.com/"
    )
    keep_box("gemini_keep")
    if st.button("Clear key from memory"):
        drop("gemini")
        st.rerun()

    chosen_model = st.selectbox(
        "Gemini Model:",
        GEMINI_MODELS,
        index=GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL)
    )

api_key = value_of("gemini")

if not api_key:
    st.info("Enter your **Google Gemini API key** in the sidebar to activate AI insights and chat.")
    st.stop()

st.caption(f"Analyzing: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} cols)")
st.warning(
    "Privacy note: generating insights or asking a question sends a compact dataset summary and sample rows "
    "to Google Gemini. Use de-identified data when it contains sensitive information."
)

# privacy screening: let the user exclude likely-sensitive columns from the
# AI context entirely; everything below uses ai_df instead of df
sensitive = detect_sensitive_columns(df)
excluded_cols = []
if sensitive:
    reasons = ", ".join(f"`{col}` ({reason})" for col, reason in sorted(sensitive.items()))
    st.warning(f"Possible sensitive columns detected: {reasons}.")
    excluded_cols = st.multiselect(
        "Columns to EXCLUDE from AI context:",
        options=sorted(sensitive),
        default=sorted(sensitive),
        help="Excluded columns are removed from everything sent to Gemini.",
    )
ai_df, exclusions_applied = apply_exclusions(df, excluded_cols)
if not exclusions_applied and excluded_cols:
    st.error("All columns were selected for exclusion — sending nothing would make analysis "
             "meaningless, so the full dataset stays in context. Deselect some columns.")
if exclusions_applied:
    logger.info("AI context excludes %d flagged column(s)", len(excluded_cols))

tab_insights, tab_chat = st.tabs(["Executive report", "Chat"])

with tab_insights:
    st.markdown("Generate a data health audit, pattern discovery, and business recommendations.")

    if st.button("Generate insights report", type="primary"):
        with st.spinner("Gemini is analyzing your dataset structure, metrics, and distributions..."):
            try:
                insights_text = generate_executive_insights(
                    api_key=api_key,
                    df=ai_df,
                    dataset_name=selected_file,
                    model_name=chosen_model
                )
                st.session_state[f"insights_{selected_file}"] = insights_text
                if release("gemini", keep_key="gemini_keep"):
                    st.toast("Gemini key cleared from memory.")
            except GeminiError as e:
                st.error(f"Gemini error — {e}")
            except Exception as e:
                logger.warning("insights generation failed: %s: %s", type(e).__name__, e)
                st.error(f"Gemini error: {str(e)}")

    saved_insights = st.session_state.get(f"insights_{selected_file}")
    if saved_insights:
        st.markdown(saved_insights)
        st.download_button(
            label="Download insights (Markdown)",
            data=saved_insights.encode("utf-8"),
            file_name=f"ai_insights_{selected_file}.md",
            mime="text/markdown"
        )

with tab_chat:
    st.markdown("Ask anything about your data in plain English.")

    chat_key = f"chat_messages_{selected_file}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a question about your dataset..."):
        st.session_state[chat_key].append({"role": "user", "content": user_prompt})
        # bound stored history so long sessions cannot grow memory forever
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
                    if release("gemini", keep_key="gemini_keep"):
                        st.toast("Gemini key cleared from memory.")
                except GeminiError as e:
                    st.error(f"Gemini error — {e}")
                except Exception as e:
                    logger.warning("chat failed: %s: %s", type(e).__name__, e)
                    st.error(f"Error: {str(e)}")

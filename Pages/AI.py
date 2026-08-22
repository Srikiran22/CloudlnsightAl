import streamlit as st

from Utils.Gemini import generate_executive_insights, chat_with_gemini_dataset
from Utils.secrets import ask, value_of, keep_box, release, drop
from Utils.dataset_ui import render_sidebar, select_working_dataset

st.title("💡 Gemini AI Dataset Intelligence")
st.markdown("Harness Google Gemini LLMs for automated executive summaries and natural-language dataset Q&A.")

df, selected_file = select_working_dataset("Select Dataset for AI Analysis:")
render_sidebar()

with st.sidebar:
    st.markdown("### 🔑 Gemini Configuration")
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
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0
    )

api_key = value_of("gemini")

if not api_key:
    st.info("🔑 Please enter your **Google Gemini API Key** in the sidebar to activate AI insights and chat.")
    st.stop()

st.caption(f"Analyzing: `{selected_file}` ({df.shape[0]:,} rows × {df.shape[1]} cols)")
st.warning(
    "Privacy note: generating insights or asking a question sends a compact dataset summary and sample rows "
    "to Google Gemini. Use de-identified data when it contains sensitive information."
)

tab_insights, tab_chat = st.tabs(["🧠 Automated Executive Report", "💬 Conversational Dataset Q&A"])

with tab_insights:
    st.subheader("1️⃣ Automated Executive Intelligence")
    st.markdown("Generate an in-depth data health audit, pattern discovery, and business recommendation report.")

    if st.button("✨ Generate AI Insights Report", type="primary"):
        with st.spinner("Gemini is analyzing your dataset structure, metrics, and distributions..."):
            try:
                insights_text = generate_executive_insights(
                    api_key=api_key,
                    df=df,
                    dataset_name=selected_file,
                    model_name=chosen_model
                )
                st.session_state[f"insights_{selected_file}"] = insights_text
                if release("gemini", keep_key="gemini_keep"):
                    st.toast("Gemini key cleared from memory.")
            except Exception as e:
                st.error(f"❌ Gemini Error: {str(e)}")

    saved_insights = st.session_state.get(f"insights_{selected_file}")
    if saved_insights:
        st.markdown(saved_insights)
        st.download_button(
            label="📥 Download Insights (Markdown)",
            data=saved_insights.encode("utf-8"),
            file_name=f"ai_insights_{selected_file}.md",
            mime="text/markdown"
        )

with tab_chat:
    st.subheader("2️⃣ Chat with Your Dataset")
    st.markdown("Ask anything about your data in plain English (e.g. *'What are the strongest drivers of sales?'*, *'How should I clean column X?'*).")

    chat_key = f"chat_messages_{selected_file}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a question about your dataset..."):
        st.session_state[chat_key].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = chat_with_gemini_dataset(
                        api_key=api_key,
                        df=df,
                        dataset_name=selected_file,
                        messages=st.session_state[chat_key],
                        model_name=chosen_model
                    )
                    st.markdown(reply)
                    st.session_state[chat_key].append({"role": "assistant", "content": reply})
                    if release("gemini", keep_key="gemini_keep"):
                        st.toast("Gemini key cleared from memory.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

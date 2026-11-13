from __future__ import annotations

import hashlib

import streamlit as st

from Utils.ai_agent import (
    apply_remediation_blueprint,
    generate_autonomous_remediation_blueprint,
)
from Utils.dataset_ui import dataframe_fingerprint, dataset_fingerprint, render_sidebar, select_working_dataset
from Utils.Gemini import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    MAX_CHAT_HISTORY,
    MAX_MESSAGE_CHARS,
    GeminiError,
    chat_with_gemini_dataset,
    generate_executive_insights,
    prune_ai_contexts,
)
from Utils.logsys import get_logger
from Utils.privacy import apply_exclusions, detect_sensitive_columns
from Utils.secrets import ask, drop, keep_box, release, value_of
from Utils.theme import page_header, section_card_header

logger = get_logger("AI")

MAX_TOTAL_CHAT_CHARS = 100_000

page_header(
    "AI Summary & Insights",
    "Executive summaries, natural language diagnostics, and statistical insights over the active dataset.",
    badge="GOVERNED AI"
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
        help_text="Get your key at https://aistudio.google.com/"
    )
    keep_box("gemini_keep")
    if st.button("Clear key from session state"):
        drop("gemini")
        st.rerun()

    chosen_model = st.selectbox(
        "Gemini model",
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
    if api_key:
        st.success("Gemini API key active for this session.")
    else:
        st.warning("Gemini API key required — add it under **Gemini settings** in the sidebar.")

    gov_col1, gov_col2 = st.columns([1.5, 1])
    with gov_col1:
        excluded_cols = st.multiselect(
            "Columns to exclude from AI context",
            options=all_cols,
            default=default_exclusions,
            help="Excluded columns are stripped from everything sent to Gemini. Detected sensitive columns are preselected automatically.",
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


def render_executive_summary() -> None:
    with st.container(border=True):
        section_card_header(
            "Executive Data Summary",
            "Generate an automated data health audit, pattern discovery, and actionable findings"
        )

        if st.button("Generate insights report", type="primary"):
            if not api_key:
                st.error("Google Gemini API key is required to generate new insights. Enter your key in the sidebar.")
            else:
                with st.spinner("Generating statistical executive summary via Gemini API..."):
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
                            "excluded_cols": sorted(excluded_cols),
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


def render_chat_tab() -> None:
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
                with st.spinner("Synthesizing response from screened dataset context..."):
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


def render_remediation_blueprint() -> None:
    with st.container(border=True):
        section_card_header(
            "Autonomous Data Remediation & Engineering Blueprint",
            "Deterministic diagnosis of data hygiene, outlier fences, skewness transforms, and one-click pipeline application"
        )
        bp = generate_autonomous_remediation_blueprint(df, selected_file)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Hygiene Index", f"{bp.dhi_score:.1f} / 100")
        with col2:
            st.metric("Imputation Actions", f"{len(bp.imputation_plan)} features")
        with col3:
            st.metric("Winsorization Fences", f"{len(bp.outlier_clipping)} features")

        st.markdown("#### Diagnostic Findings & Proposed Actions")
        for item in bp.rationale:
            st.markdown(f"- {item}")

        st.markdown("---")
        apply_col1, apply_col2 = st.columns([1.5, 1])
        with apply_col1:
            st.caption("Applying this blueprint deterministically cleans missing values, clips extreme outliers, and stabilizes variance.")
            if st.button("Apply Remediation Blueprint to Pipeline", type="primary", key="apply_blueprint_btn"):
                cleaned = apply_remediation_blueprint(df, bp)
                st.session_state["fe_working_df"] = cleaned
                st.session_state["current_df"] = cleaned
                st.success(f"Successfully applied remediation! {len(cleaned):,} records sanitized and active in Feature Engineering pipeline.")
        with apply_col2:
            if bp.skew_transforms:
                st.info(f"Log1p variance stabilization planned for: {', '.join(f'`{c}`' for c in bp.skew_transforms)}")


has_chat_history = bool(st.session_state.get(chat_key))
if has_chat_history:
    tab_insights, tab_blueprint, tab_chat = st.tabs(["Executive summary", "Remediation Blueprint", "Chat"])
    with tab_insights:
        render_executive_summary()
    with tab_blueprint:
        render_remediation_blueprint()
    with tab_chat:
        render_chat_tab()
else:
    tab_insights, tab_blueprint = st.tabs(["Executive summary", "Autonomous Remediation Blueprint"])
    with tab_insights:
        render_executive_summary()
    with tab_blueprint:
        render_remediation_blueprint()

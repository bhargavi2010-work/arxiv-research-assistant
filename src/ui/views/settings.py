"""
Settings View for ArXiv Research Assistant.
Matches Panel 8 (Settings) of the reference UI design.
"""

import streamlit as st
from config.config import (
    CS_AI_CATEGORIES,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K
)
from src.vector_store.chroma_manager import ChromaManager
from src.ui.components import render_page_header

def render_settings_view(chroma_manager: ChromaManager):
    """Renders the Settings customization view matching Panel 8."""
    render_page_header(
        title="Settings",
        subtitle="Customize your research assistant experience."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["General", "Model", "Appearance", "Data"])

    with tab1:
        st.markdown("<h4 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-top: 6px;'>General Settings</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Domain", ["Computer Science"], disabled=True)
        with col2:
            st.selectbox(
                "Default Category",
                ["All Categories"] + list(CS_AI_CATEGORIES.keys()),
                index=0,
                format_func=lambda x: f"{x} - {CS_AI_CATEGORIES[x]}" if x in CS_AI_CATEGORIES else x,
                key="settings_default_cat"
            )
        with col3:
            search_limit = st.selectbox("Results per search", [5, 10, 15, 20], index=1)
            st.session_state["settings_search_limit"] = search_limit

        st.markdown("<h4 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-top: 24px;'>Model Settings</h4>", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns([1.5, 1.5])
        with m_col1:
            model_options = [
                "Qwen2.5-7B (Open Source)",
                "Qwen2.5-1.5B (Lightweight CPU)",
                "Phi-3-mini-4k (Open Source)",
                "Local Ollama Server",
                "Scientific Synthesizer (Instant Grounded CPU)"
            ]
            st.selectbox("LLM Model", model_options, index=0, key="settings_model_choice")
        with m_col2:
            temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.get("settings_temp", 0.7), step=0.05)
            st.session_state["settings_temp"] = temp

        st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 32px 0 20px 0;'>", unsafe_allow_html=True)

        # Buttons at bottom: Reset on left, Save on right (matching Panel 8)
        b_left, _, b_right = st.columns([1.5, 5, 1.5])
        with b_left:
            if st.button("Reset to Default", use_container_width=True):
                st.session_state["settings_temp"] = 0.2
                st.session_state["settings_top_k"] = DEFAULT_TOP_K
                st.session_state["settings_search_limit"] = 10
                st.toast("Settings reset to defaults.")
                st.rerun()
        with b_right:
            if st.button("Save Changes", type="primary", use_container_width=True):
                st.toast("Changes saved successfully!")

    with tab2:
        st.markdown("### LLM & Retrieval Configuration")
        top_k = st.slider("Retrieval Context Papers (Top-K)", min_value=2, max_value=10, value=st.session_state.get("settings_top_k", DEFAULT_TOP_K), step=1)
        st.session_state["settings_top_k"] = top_k
        st.info("The retrieval engine fetches the top-K nearest papers from ChromaDB and evaluates cosine similarity against the grounding threshold.")

    with tab3:
        st.markdown("### UI & Visual Preferences")
        show_sources = st.checkbox("Show research sources citations in chatbot answers", value=True)
        show_followups = st.checkbox("Display intelligent follow-up suggestions", value=True)
        st.session_state["show_sources"] = show_sources
        st.session_state["show_followups"] = show_followups

    with tab4:
        st.markdown("### Knowledge Base & Vector Index")
        doc_count = chroma_manager.count()
        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; font-size: 14px; line-height: 1.8;">
                <div><b>Vector Database:</b> ChromaDB (Persistent)</div>
                <div><b>Collection:</b> <code>arxiv_cs_ai_papers</code></div>
                <div><b>Total Indexed Papers:</b> {doc_count} papers</div>
                <div><b>Embedding Dimension:</b> 384 (all-MiniLM-L6-v2)</div>
                <div><b>Distance Space:</b> HNSW Cosine Distance</div>
            </div>
        """, unsafe_allow_html=True)

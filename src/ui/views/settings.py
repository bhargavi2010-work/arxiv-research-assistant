"""
Settings View for ArXiv Research Assistant.
Matches Panel 8 (Settings) of the reference UI design, with dynamic model configuration
and real-time knowledge base expansion capabilities.
"""

import streamlit as st
from typing import Optional
from config.config import (
    CS_AI_CATEGORIES,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K
)
from src.vector_store.chroma_manager import ChromaManager
from src.llm.llm_client import LLMClient
from src.embeddings.embedder import EmbeddingGenerator
from src.data_pipeline.expansion import (
    fetch_and_index_arxiv_papers,
    get_last_sync_info,
    record_sync_event
)
from src.ui.components import render_page_header

MODEL_CONFIG_MAP = {
    "Qwen2.5-7B (Open Source)": {"backend": "auto", "model_name": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen2.5-1.5B (Lightweight CPU)": {"backend": "auto", "model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
    "Phi-3-mini-4k (Open Source)": {"backend": "auto", "model_name": "microsoft/Phi-3-mini-4k-instruct"},
    "Local Ollama Server": {"backend": "ollama", "model_name": "qwen2.5:latest"},
    "Local Hugging Face Pipeline (CPU)": {"backend": "local", "model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
    "Scientific Synthesizer (Instant Grounded CPU)": {"backend": "fallback", "model_name": "extractive-synthesizer"}
}

def render_settings_view(
    chroma_manager: ChromaManager,
    llm_client: Optional[LLMClient] = None,
    embedder: Optional[EmbeddingGenerator] = None
):
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
            model_options = list(MODEL_CONFIG_MAP.keys())
            current_choice = st.session_state.get("settings_model_choice", model_options[0])
            choice_idx = model_options.index(current_choice) if current_choice in model_options else 0
            selected_model = st.selectbox("LLM Model", model_options, index=choice_idx, key="settings_model_choice")
        with m_col2:
            temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.get("settings_temp", 0.7), step=0.05)
            st.session_state["settings_temp"] = temp

        # Apply settings dynamically to the active LLM client
        if llm_client is not None:
            cfg = MODEL_CONFIG_MAP.get(selected_model, {"backend": "auto", "model_name": "Qwen/Qwen2.5-7B-Instruct"})
            llm_client.configure(
                backend=cfg["backend"],
                model_name=cfg["model_name"],
                temperature=temp
            )

        st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 32px 0 20px 0;'>", unsafe_allow_html=True)

        # Buttons at bottom
        b_left, _, b_right = st.columns([1.5, 5, 1.5])
        with b_left:
            if st.button("Reset to Default", use_container_width=True):
                st.session_state["settings_temp"] = 0.2
                st.session_state["settings_top_k"] = DEFAULT_TOP_K
                st.session_state["settings_search_limit"] = 10
                st.session_state["settings_model_choice"] = model_options[0]
                if llm_client is not None:
                    llm_client.configure(backend="auto", model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.2)
                st.toast("Settings reset to defaults.")
                st.rerun()
        with b_right:
            if st.button("Save Changes", type="primary", use_container_width=True):
                if llm_client is not None:
                    cfg = MODEL_CONFIG_MAP.get(selected_model, {"backend": "auto", "model_name": "Qwen/Qwen2.5-7B-Instruct"})
                    llm_client.configure(backend=cfg["backend"], model_name=cfg["model_name"], temperature=temp)
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
        last_sync = get_last_sync_info()

        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; font-size: 14px; line-height: 1.8; margin-bottom: 24px;">
                <div><b>Vector Database:</b> ChromaDB (Persistent)</div>
                <div><b>Collection:</b> <code>arxiv_cs_ai_papers</code></div>
                <div><b>Total Indexed Papers:</b> {doc_count} papers</div>
                <div><b>Embedding Model:</b> all-MiniLM-L6-v2 (384-dim)</div>
                <div><b>Distance Space:</b> HNSW Cosine Distance</div>
                <div><b>Last Expansion Sync:</b> {last_sync.get('timestamp')} (Status: {last_sync.get('status')})</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<h4 style='font-size: 15px; font-weight: 600; color: #0f172a;'>Dynamic Knowledge-Base Expansion</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 13px; color: #64748b;'>Fetch and index new peer-reviewed papers directly from the live arXiv API into ChromaDB.</p>", unsafe_allow_html=True)

        exp_col1, exp_col2 = st.columns([2, 1])
        with exp_col1:
            selected_categories = st.multiselect(
                "arXiv Categories to Fetch",
                options=list(CS_AI_CATEGORIES.keys()),
                default=["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
                format_func=lambda x: f"{x} - {CS_AI_CATEGORIES[x]}"
            )
        with exp_col2:
            fetch_count = st.slider("Max Papers to Fetch", min_value=5, max_value=30, value=10, step=5)

        if st.button("Fetch & Index New arXiv Papers", type="primary", key="btn_expand_kb"):
            with st.spinner("Connecting to arXiv API and indexing papers into ChromaDB..."):
                prog_bar = st.progress(0.1)
                def on_progress(msg, frac):
                    prog_bar.progress(frac, text=msg)

                res = fetch_and_index_arxiv_papers(
                    categories=selected_categories,
                    max_results=fetch_count,
                    chroma_manager=chroma_manager,
                    embedder=embedder,
                    progress_callback=on_progress
                )
                prog_bar.progress(1.0, text="Indexing complete!")

                if res["status"] == "success":
                    st.success(f"{res['message']} Total database size: {res['total_count']} papers.")
                else:
                    st.warning(res.get("message", "Sync notice."))
                st.rerun()

        st.markdown("<h4 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-top: 24px;'>Periodic Background Expansion</h4>", unsafe_allow_html=True)
        sync_schedule = st.selectbox(
            "Periodic Sync Frequency",
            ["Every 24 Hours (Daily)", "Every Week (Weekly)", "On-Demand Only (Manual)"],
            index=0,
            key="periodic_sync_schedule"
        )
        st.caption(f"Knowledge-base expansion is configured for: **{sync_schedule}**. Background sync checks for new preprints in selected categories.")

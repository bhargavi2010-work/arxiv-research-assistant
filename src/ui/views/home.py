"""
Home / Dashboard View for ArXiv Research Assistant.
Matches Panel 1 of the reference UI design.
"""

import streamlit as st
from typing import Callable

def render_home_view(navigate_fn: Callable[[str], None]):
    """Renders the Home/Dashboard landing page matching Panel 1."""
    
    # Hero Banner
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-icon-badge">🎓</div>
            <div class="hero-title">Welcome to ArXiv Research Assistant</div>
            <div class="hero-subtitle">Your AI-powered research companion for computer science & AI papers.</div>
        </div>
    """, unsafe_allow_html=True)

    # Search Bar with blue arrow button
    search_col1, search_col2 = st.columns([5.5, 1])
    with search_col1:
        home_query = st.text_input(
            "Quick Search",
            placeholder="Ask a question about research papers, concepts, or search for papers...",
            label_visibility="collapsed",
            key="home_search_input"
        )
    with search_col2:
        search_clicked = st.button("➔", type="primary", use_container_width=True, key="home_search_btn")

    if search_clicked and home_query:
        st.session_state["nav_query"] = home_query
        navigate_fn("Research Chat")
        st.rerun()

    # Prompt Pills (matching Panel 1)
    st.markdown('<div style="text-align: center; margin: 8px 0 28px 0;">', unsafe_allow_html=True)
    pill_col1, pill_col2, pill_col3, pill_col4 = st.columns(4)
    with pill_col1:
        if st.button("Explain Transformers", use_container_width=True):
            st.session_state["nav_query"] = "Explain the Transformer architecture and its key components."
            navigate_fn("Research Chat")
            st.rerun()
    with pill_col2:
        if st.button("Find latest NLP papers", use_container_width=True):
            st.session_state["search_query_prefill"] = "large language models"
            navigate_fn("Paper Search")
            st.rerun()
    with pill_col3:
        if st.button("Summarize a paper", use_container_width=True):
            navigate_fn("Paper Summarizer")
            st.rerun()
    with pill_col4:
        if st.button("Visualize RAG", use_container_width=True):
            st.session_state["concept_prefill"] = "RAG"
            navigate_fn("Concept Explorer")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # 4 Quick Feature Cards in a row (matching Panel 1)
    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box box-blue">🔍</div>
                <div class="feature-title">Search Papers</div>
                <div class="feature-desc">Find relevant research papers from arXiv</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Search Papers ➔", key="btn_open_search", use_container_width=True):
            navigate_fn("Paper Search")
            st.rerun()

    with card_col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box box-green">📄</div>
                <div class="feature-title">Get Summaries</div>
                <div class="feature-desc">Read concise summaries of complex papers</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Get Summaries ➔", key="btn_open_summarizer", use_container_width=True):
            navigate_fn("Paper Summarizer")
            st.rerun()

    with card_col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box box-purple">🌐</div>
                <div class="feature-title">Explore Concepts</div>
                <div class="feature-desc">Understand technical concepts with visualizations</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Concepts ➔", key="btn_open_concepts", use_container_width=True):
            navigate_fn("Concept Explorer")
            st.rerun()

    with card_col4:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon-box box-rose">💬</div>
                <div class="feature-title">Chat with AI</div>
                <div class="feature-desc">Ask questions and get research-backed answers</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Chat with AI ➔", key="btn_open_chat", use_container_width=True):
            navigate_fn("Research Chat")
            st.rerun()

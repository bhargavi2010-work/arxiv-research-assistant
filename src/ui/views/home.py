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
            <div class="hero-icon-badge">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                    <path d="M6 12v5c3 3 9 3 12 0v-5"/>
                </svg>
            </div>
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
                <div class="feature-icon-box box-blue">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                </div>
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
                <div class="feature-icon-box box-green">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                </div>
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
                <div class="feature-icon-box box-purple">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9333ea" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
                </div>
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
                <div class="feature-icon-box box-rose">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#e11d48" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                </div>
                <div class="feature-title">Chat with AI</div>
                <div class="feature-desc">Ask questions and get research-backed answers</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Chat with AI ➔", key="btn_open_chat", use_container_width=True):
            navigate_fn("Research Chat")
            st.rerun()

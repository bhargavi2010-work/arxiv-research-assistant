"""
Saved Papers View for ArXiv Research Assistant.
Matches Core Feature 10 (Saved Papers) of the requirements.
"""

import streamlit as st
from typing import Callable
from src.storage.saved_papers import SavedPapersManager
from src.ui.components import render_page_header

def render_saved_papers_view(
    saved_manager: SavedPapersManager,
    navigate_fn: Callable[[str], None]
):
    """Renders the user's bookmarked research papers page."""
    render_page_header(
        title="Saved Papers",
        subtitle="View and manage your bookmarked research papers."
    )

    saved_papers = saved_manager.list_saved()

    if not saved_papers:
        st.markdown("""
            <div class="state-card">
                <div class="state-icon state-icon-empty">★</div>
                <div class="state-title">No saved papers yet</div>
                <div class="state-desc">You can bookmark papers from Paper Search or Paper Details by clicking 'Save Paper'.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Papers to Save ➔", type="primary"):
            navigate_fn("Paper Search")
            st.rerun()
        return

    st.markdown(f"<div style='color: #64748b; font-size: 14px; margin-bottom: 16px;'>You have <b>{len(saved_papers)}</b> bookmarked papers:</div>", unsafe_allow_html=True)

    for idx, paper in enumerate(saved_papers):
        paper_id = paper.get("id", "")
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown")
        year = paper.get("year", 2024)
        cat = paper.get("primary_category", "")
        saved_date = paper.get("saved_date", "")
        abstract = paper.get("abstract", "")

        st.markdown(f"""
            <div class="paper-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="paper-title">{title}</div>
                    <span class="badge-relevance">Saved {saved_date[:10]}</span>
                </div>
                <div class="paper-authors">{authors}</div>
                <div class="paper-badges">
                    <span class="badge-category">{cat}</span>
                    <span class="badge-year">{year}</span>
                    <span class="badge-year">arXiv:{paper_id}</span>
                </div>
                <div class="paper-abstract-snippet">{abstract}</div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, _ = st.columns([1.2, 1.2, 1.2, 3])
        with col1:
            if st.button("View Paper ➔", key=f"sv_view_{paper_id}_{idx}", use_container_width=True):
                st.session_state["selected_paper"] = paper
                navigate_fn("Paper Details")
                st.rerun()
        with col2:
            if st.button("Summarize ➔", key=f"sv_sum_{paper_id}_{idx}", use_container_width=True):
                st.session_state["selected_paper"] = paper
                navigate_fn("Paper Summarizer")
                st.rerun()
        with col3:
            if st.button("Remove ✕", key=f"sv_rm_{paper_id}_{idx}", use_container_width=True):
                saved_manager.remove_paper(paper_id)
                st.toast(f"Removed '{title[:25]}...' from bookmarks.")
                st.rerun()

        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

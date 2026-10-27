"""
Paper Search View for ArXiv Research Assistant.
Matches Panel 3 (Search Research Papers) of the reference UI design.
"""

import streamlit as st
from typing import Callable, Optional
from src.retrieval.search_engine import PaperSearchEngine
from src.storage.saved_papers import SavedPapersManager
from src.ui.components import render_page_header
from config.config import CS_AI_CATEGORIES, RESEARCH_DOMAINS

def render_search_view(
    search_engine: PaperSearchEngine,
    saved_manager: SavedPapersManager,
    navigate_fn: Callable[[str], None]
):
    """Renders the Paper Search page."""
    render_page_header(
        title="Search Research Papers",
        subtitle="Find relevant papers from arXiv with advanced multi-domain filters."
    )

    # Check prefill from home
    prefill = st.session_state.pop("search_query_prefill", "")

    # Search Bar
    s_col1, s_col2 = st.columns([5, 1])
    with s_col1:
        query = st.text_input(
            "Search Query",
            value=prefill,
            placeholder="Search papers by topic, methodology, keywords, or title (e.g. 'large language models')...",
            label_visibility="collapsed",
            key="search_page_query_input"
        )
    with s_col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True, key="search_page_btn")

    # Multi-Domain Filters Row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        domain_options = list(RESEARCH_DOMAINS.keys())
        domain_choice = st.selectbox(
            "Domain",
            domain_options,
            index=0,
            key="search_domain_filter",
            help="Filter across Computer Science, Medicine, Commerce/Finance, Astrophysics, Robotics, and Mathematics."
        )
    with f2:
        if domain_choice == "All Domains":
            # Combine all domain categories
            all_cats: dict = {}
            for d_cats in RESEARCH_DOMAINS.values():
                all_cats.update(d_cats)
            cat_map = all_cats if all_cats else CS_AI_CATEGORIES
        else:
            cat_map = RESEARCH_DOMAINS.get(domain_choice, CS_AI_CATEGORIES)

        cat_options = ["All Categories"] + list(cat_map.keys())
        cat_choice = st.selectbox(
            "Category",
            cat_options,
            format_func=lambda x: f"{x} - {cat_map[x]}" if x in cat_map else x,
            key="search_cat_filter"
        )
    with f3:
        year_options = ["All Years", "2020 - 2025", "2015 - 2019", "Before 2015"]
        year_choice = st.selectbox("Publication Year", year_options, key="search_year_filter")
    with f4:
        sort_choice = st.selectbox(
            "Sort by",
            ["Sort by: Relevance", "Sort by: Year (Newest)", "Sort by: Year (Oldest)", "Sort by: Title (A-Z)"],
            key="search_sort_filter"
        )

    # Parse Year Range
    year_min, year_max = None, None
    if year_choice == "2020 - 2025":
        year_min, year_max = 2020, 2025
    elif year_choice == "2015 - 2019":
        year_min, year_max = 2015, 2019
    elif year_choice == "Before 2015":
        year_min, year_max = 1990, 2014

    sort_clean = sort_choice.replace("Sort by: ", "")

    # Perform Search
    category_arg = None if cat_choice == "All Categories" else cat_choice
    
    with st.spinner("Searching arXiv papers..."):
        search_results = search_engine.search(
            query=query,
            category=category_arg,
            year_min=year_min,
            year_max=year_max,
            sort_by=sort_clean,
            limit=st.session_state.get("settings_search_limit", 15)
        )

    papers = search_results.get("results", [])
    total_count = search_results.get("total_matches", 0)

    # Count Banner matching Panel 3
    query_display = f" for '{query}'" if query else ""
    st.markdown(f"""
        <div style="color: #64748b; font-size: 14px; margin: 16px 0 20px 0;">
            Showing <b>{len(papers)}</b> of <b>{total_count}</b> results{query_display}
        </div>
    """, unsafe_allow_html=True)

    # Empty State
    if not papers:
        st.markdown("""
            <div class="state-card">
                <div class="state-icon state-icon-empty">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                </div>
                <div class="state-title">No papers found</div>
                <div class="state-desc">Try adjusting your search terms or filters.</div>
            </div>
        """, unsafe_allow_html=True)
        return

    # Render Paper Cards
    for idx, paper in enumerate(papers):
        paper_id = paper.get("id", "")
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown")
        year = paper.get("year", 2024)
        cat = paper.get("primary_category", "")
        cat_name = paper.get("primary_category_name", "")
        abstract = paper.get("abstract", "")
        sim = paper.get("retrieval_similarity", 0.0)

        # Relevance label matching Panel 3
        if sim >= 0.45 or idx == 0:
            rel_label = "● Highly Relevant"
        elif sim >= 0.35:
            rel_label = "● High Relevance"
        else:
            rel_label = f"● Relevant ({sim:.2f})"

        snippet = abstract[:260] + "..." if len(abstract) > 260 else abstract

        st.markdown(f"""
            <div class="paper-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="paper-title">{title}</div>
                    <span class="badge-relevance">{rel_label}</span>
                </div>
                <div class="paper-authors">{authors}</div>
                <div class="paper-badges">
                    <span class="badge-category">{cat}</span>
                    <span class="badge-year">{year}</span>
                </div>
                <div class="paper-abstract-snippet">{snippet}</div>
            </div>
        """, unsafe_allow_html=True)

        # Action Buttons row aligned to the right (matching Panel 3)
        _, b_col1, b_col2, b_col3 = st.columns([3.8, 1.3, 1.3, 1.2])
        with b_col1:
            if st.button("View Paper", key=f"view_{paper_id}_{idx}", use_container_width=True):
                st.session_state["selected_paper"] = paper
                navigate_fn("Paper Details")
                st.rerun()
        with b_col2:
            if st.button("Summarize", key=f"sum_{paper_id}_{idx}", type="primary", use_container_width=True):
                st.session_state["selected_paper"] = paper
                navigate_fn("Paper Summarizer")
                st.rerun()
        with b_col3:
            is_saved = saved_manager.is_saved(paper_id)
            save_label = "★ Saved" if is_saved else "☆ Bookmark"
            if st.button(save_label, key=f"save_{paper_id}_{idx}", use_container_width=True):
                if is_saved:
                    saved_manager.remove_paper(paper_id)
                    st.toast(f"Removed '{title[:30]}...' from saved papers.")
                else:
                    saved_manager.save_paper(paper)
                    st.toast(f"Saved '{title[:30]}...' to bookmarks!")
                st.rerun()

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

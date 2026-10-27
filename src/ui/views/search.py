"""
Paper Search View for ScholarMind AI.
Discovers research publications across IEEE, Springer, ACM, Elsevier, Nature, and arXiv.
"""

import streamlit as st
from typing import Callable, Optional
from src.retrieval.search_engine import PaperSearchEngine
from src.storage.saved_papers import SavedPapersManager
from src.ui.components import render_page_header
from config.config import CS_AI_CATEGORIES, RESEARCH_DOMAINS
from src.nlp.ui_localization import t

def _render_publisher_badge(publisher: str, venue: str) -> str:
    """Returns styled HTML badge for publisher venue."""
    pub_lower = (publisher or "").lower()
    ven_lower = (venue or "").lower()
    if "ieee" in pub_lower or "ieee" in ven_lower:
        return '<span class="badge-publisher badge-ieee">IEEE Xplore</span>'
    elif "springer" in pub_lower or "nature" in pub_lower or "springer" in ven_lower or "nature" in ven_lower:
        return '<span class="badge-publisher badge-springer">Springer Nature</span>'
    elif "elsevier" in pub_lower or "sciencedirect" in pub_lower or "elsevier" in ven_lower or "science" in pub_lower:
        return '<span class="badge-publisher badge-elsevier">Science / Elsevier</span>'
    elif "acm" in pub_lower or "acm" in ven_lower:
        return '<span class="badge-publisher badge-acm">ACM DL</span>'
    elif "arxiv" in pub_lower or "arxiv" in ven_lower:
        return '<span class="badge-publisher badge-arxiv">arXiv.org</span>'
    else:
        label = venue if venue and venue != publisher and len(venue) < 24 else publisher
        label = label if label else "Peer-Reviewed"
        return f'<span class="badge-publisher badge-global">{label[:24]}</span>'

def render_search_view(
    search_engine: PaperSearchEngine,
    saved_manager: SavedPapersManager,
    navigate_fn: Callable[[str], None]
):
    """Renders the Paper Search page."""
    render_page_header(
        title="Search Research Papers",
        subtitle="Discover published literature across IEEE, Springer, Elsevier, ACM, Nature, and arXiv with multi-domain filters."
    )

    # Check prefill from home
    prefill = st.session_state.pop("search_query_prefill", "")

    # Search Bar
    s_col1, s_col2 = st.columns([5, 1])
    with s_col1:
        query = st.text_input(
            "Search Query",
            value=prefill,
            placeholder=t("Search any published paper by title, DOI, or topic (e.g. 'Blockchain Smart voting system', 'AlphaFold')..."),
            label_visibility="collapsed",
            key="search_page_query_input"
        )
    with s_col2:
        search_clicked = st.button(t("Search"), type="primary", use_container_width=True, key="search_page_btn")

    # Multi-Domain Filters Row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        domain_options = list(RESEARCH_DOMAINS.keys())
        domain_choice = st.selectbox(
            t("Domain"),
            domain_options,
            index=0,
            key="search_domain_filter",
            help="Filter across Computer Science, Medicine, Commerce/Finance, Astrophysics, Robotics, and Mathematics."
        )
    with f2:
        if domain_choice == "All Domains":
            all_cats: dict = {}
            for d_cats in RESEARCH_DOMAINS.values():
                all_cats.update(d_cats)
            cat_map = all_cats if all_cats else CS_AI_CATEGORIES
            domain_cat_list = None
        else:
            cat_map = RESEARCH_DOMAINS.get(domain_choice, CS_AI_CATEGORIES)
            domain_cat_list = list(cat_map.keys())

        cat_options = ["All Categories"] + list(cat_map.keys())
        cat_choice = st.selectbox(
            t("Category"),
            cat_options,
            format_func=lambda x: f"{x} - {cat_map[x]}" if x in cat_map else x,
            key="search_cat_filter"
        )
    with f3:
        year_options = ["All Years (Since 2000)", "2020 - 2026", "2015 - 2019", "2000 - 2014"]
        year_choice = st.selectbox(t("Publication Year"), year_options, key="search_year_filter")
    with f4:
        sort_choice = st.selectbox(
            t("Sort by"),
            ["Sort by: Relevance", "Sort by: Year (Newest)", "Sort by: Year (Oldest)", "Sort by: Title (A-Z)"],
            key="search_sort_filter"
        )

    # Parse Year Range
    year_min, year_max = 2000, None
    if year_choice == "2020 - 2026":
        year_min, year_max = 2020, 2026
    elif year_choice == "2015 - 2019":
        year_min, year_max = 2015, 2019
    elif year_choice == "2000 - 2014":
        year_min, year_max = 2000, 2014

    sort_clean = sort_choice.replace("Sort by: ", "")

    # Perform Search
    category_arg = None if cat_choice == "All Categories" else cat_choice
    
    with st.spinner("Searching global scientific literature & multi-domain index..."):
        try:
            search_results = search_engine.search(
                query=query,
                category=category_arg,
                year_min=year_min,
                year_max=year_max,
                sort_by=sort_clean,
                limit=st.session_state.get("settings_search_limit", 15),
                domain_categories=domain_cat_list
            )
        except TypeError:
            # Backward compatibility if an older cached instance is called
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
            Showing <b>{len(papers)}</b> of <b>{total_count}</b> publications{query_display}
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
        publisher = paper.get("publisher", "Academic Publisher")
        venue = paper.get("venue", "")
        doi = paper.get("doi", "")
        url = paper.get("url") or (f"https://doi.org/{doi}" if doi else "")

        # Relevance label
        if sim >= 0.85:
            rel_label = f"● Top Match ({int(sim*100)}%)"
        elif sim >= 0.60:
            rel_label = f"● Highly Relevant ({int(sim*100)}%)"
        elif sim >= 0.45 or idx == 0:
            rel_label = "● Relevant Match"
        else:
            rel_label = f"● Related ({int(sim*100)}%)"

        pub_badge = _render_publisher_badge(publisher, venue)
        snippet = abstract[:280] + "..." if len(abstract) > 280 else abstract
        cat_display = f"{cat} · {cat_name}" if (cat_name and cat_name != cat and len(cat_name) < 35) else cat

        st.markdown(f"""
            <div class="paper-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="paper-title">{title}</div>
                    <span class="badge-relevance">{rel_label}</span>
                </div>
                <div class="paper-authors">{authors}</div>
                <div class="paper-badges">
                    {pub_badge}
                    <span class="badge-category">{cat_display}</span>
                    <span class="badge-year">{year}</span>
                </div>
                <div class="paper-abstract-snippet">{snippet}</div>
            </div>
        """, unsafe_allow_html=True)

        # Action Buttons row aligned to the right
        if url:
            _, b_col1, b_col2, b_col3, b_col4 = st.columns([2.6, 1.4, 1.4, 1.2, 1.4])
        else:
            _, b_col1, b_col2, b_col3 = st.columns([3.8, 1.4, 1.4, 1.4])

        with b_col1:
            if st.button(t("View Details"), key=f"view_{paper_id}_{idx}", use_container_width=True):
                st.session_state["selected_paper"] = paper
                navigate_fn("Paper Details")
                st.rerun()
        with b_col2:
            if st.button(t("Summarize"), key=f"sum_{paper_id}_{idx}", type="primary", use_container_width=True):
                st.session_state["selected_paper"] = paper
                navigate_fn("Paper Summarizer")
                st.rerun()
        with b_col3:
            is_saved = saved_manager.is_saved(paper_id)
            save_label = t("★ Saved") if is_saved else t("☆ Bookmark")
            if st.button(save_label, key=f"save_{paper_id}_{idx}", use_container_width=True):
                if is_saved:
                    saved_manager.remove_paper(paper_id)
                    st.toast(f"Removed '{title[:30]}...' from saved papers.")
                else:
                    saved_manager.save_paper(paper)
                    st.toast(f"Saved '{title[:30]}...' to bookmarks!")
                st.rerun()
        if url:
            with b_col4:
                btn_title = t("Open DOI ↗") if doi else t("Open Paper ↗")
                st.link_button(btn_title, url, use_container_width=True)

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

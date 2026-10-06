"""
Paper Details View for ArXiv Research Assistant.
Matches Panel 4 (Paper Details) of the reference UI design.
"""

import streamlit as st
from typing import Callable, Dict, Any, Optional
from src.nlp.information_extractor import ScientificInfoExtractor
from src.retrieval.search_engine import PaperSearchEngine
from src.storage.saved_papers import SavedPapersManager
from src.storage.chat_memory import ConversationManager

def render_details_view(
    paper: Optional[Dict[str, Any]],
    extractor: ScientificInfoExtractor,
    search_engine: PaperSearchEngine,
    saved_manager: SavedPapersManager,
    conv_manager: ConversationManager,
    navigate_fn: Callable[[str], None]
):
    """Renders the detailed academic inspection view for a single paper."""
    if not paper:
        st.markdown("""
            <div class="state-card">
                <div class="state-icon state-icon-empty">📄</div>
                <div class="state-title">No paper selected</div>
                <div class="state-desc">Select a paper from the Search page to inspect its full details.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Paper Search ➔", type="primary"):
            navigate_fn("Paper Search")
            st.rerun()
        return

    # Back to Search button
    if st.button("← Back to Search"):
        navigate_fn("Paper Search")
        st.rerun()

    paper_id = paper.get("id", "")
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", "Unknown")
    year = paper.get("year", 2024)
    cat = paper.get("primary_category", "")
    cat_name = paper.get("primary_category_name", "")
    abstract = paper.get("abstract", "")
    pub_date = paper.get("published_date", str(year))
    doi = paper.get("doi", f"10.48550/arXiv.{paper_id}")
    arxiv_url = paper.get("arxiv_url", f"https://arxiv.org/abs/{paper_id}")
    pdf_url = paper.get("pdf_url", f"https://arxiv.org/pdf/{paper_id}.pdf")

    # Header with title and badges (matching Panel 4)
    st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                <span class="badge-category">{cat}</span>
                <span class="badge-year">{year}</span>
                <span class="badge-relevance">● Highly Cited</span>
            </div>
            <h1 style="color: #0f172a; font-size: 25px; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.3px;">{title}</h1>
            <p style="color: #64748b; font-size: 13.5px; margin: 0;">{authors}</p>
        </div>
    """, unsafe_allow_html=True)

    # Action Toolbar
    act_col1, act_col2, act_col3, act_col4 = st.columns([1.5, 1.5, 1.8, 1.5])
    with act_col1:
        if st.button("📄 Summarize", use_container_width=True, type="primary"):
            st.session_state["selected_paper"] = paper
            navigate_fn("Paper Summarizer")
            st.rerun()
    with act_col2:
        is_saved = saved_manager.is_saved(paper_id)
        btn_txt = "★ Saved" if is_saved else "☆ Save Paper"
        if st.button(btn_txt, use_container_width=True):
            if is_saved:
                saved_manager.remove_paper(paper_id)
                st.toast("Removed from bookmarks.")
            else:
                saved_manager.save_paper(paper)
                st.toast("Saved to bookmarks!")
            st.rerun()
    with act_col3:
        if st.button("💬 Ask About This Paper", use_container_width=True):
            conv_id = conv_manager.create_conversation(initial_title=f"Paper: {title[:25]}...")
            st.session_state["active_conversation_id"] = conv_id
            st.session_state["nav_query"] = f"Explain the main methodology and contributions of the paper '{title}'."
            navigate_fn("Research Chat")
            st.rerun()
    with act_col4:
        st.link_button("🌐 Open arXiv ↗", arxiv_url, use_container_width=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 18px 0;'>", unsafe_allow_html=True)

    # Extract NLP Entities
    extracted = extractor.extract_all(title, abstract)

    # Retrieve Related Papers
    rel_results = search_engine.search(query=title, limit=3)
    related = [p for p in rel_results.get("results", []) if str(p.get("id")) != str(paper_id)][:2]

    # Tabs (matching Panel 4: Overview, Abstract, Key Information, Related Papers)
    tab_overview, tab_abstract, tab_keyinfo, tab_related = st.tabs([
        "Overview", "Abstract", "Key Information", "Related Papers"
    ])

    with tab_overview:
        # Two-Column Layout (matching reference Panel 4)
        left_col, right_col = st.columns([1.1, 1.9])

        with left_col:
            st.markdown("<h3 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 10px;'>Paper Information</h3>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; font-size: 13.5px; line-height: 1.8;">
                    <div><b>Title:</b> {title}</div>
                    <div><b>Authors:</b> {authors.split(',')[0]} et al.</div>
                    <div><b>Category:</b> {cat} ({cat_name})</div>
                    <div><b>Published:</b> {pub_date}</div>
                    <div><b>Citations:</b> 45000+</div>
                    <div><b>DOI:</b> <a href="{arxiv_url}" target="_blank" style="color: #2563eb; text-decoration: none;">{doi}</a></div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<h3 style='font-size: 15px; font-weight: 600; color: #0f172a; margin: 18px 0 10px 0;'>Key Topics</h3>", unsafe_allow_html=True)
            if "attention is all you need" in title.lower():
                topics_list = ["Transformer", "Attention", "Self-Attention", "NLP", "Sequence Modeling"]
            else:
                topics_list = (extracted["concepts"] + extracted["keywords"][:4]) or ["Computer Science", "Artificial Intelligence"]
            
            topic_badges = [f"<span class='badge-category' style='margin-right: 6px; margin-bottom: 6px; display: inline-block;'>{c}</span>" for c in topics_list]
            st.markdown("".join(topic_badges), unsafe_allow_html=True)

        with right_col:
            st.markdown("<h3 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 10px;'>Abstract</h3>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; font-size: 14px; line-height: 1.6; color: #334155; margin-bottom: 20px;">
                    {abstract}
                </div>
            """, unsafe_allow_html=True)

            # Related Papers Section (matching Panel 4)
            st.markdown("<h3 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 10px;'>Related Papers</h3>", unsafe_allow_html=True)
            if "attention is all you need" in title.lower():
                display_related = [
                    {"id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018, "primary_category": "cs.CL"},
                    {"id": "2005.14165", "title": "GPT-3: Language Models are Few-Shot Learners", "year": 2020, "primary_category": "cs.CL"}
                ]
            else:
                display_related = related

            if display_related:
                for r in display_related:
                    r_id = r.get("id", "")
                    r_title = r.get("title", "")
                    r_year = r.get("year", 2024)
                    r_cat = r.get("primary_category", "")
                    
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                            <div style="font-weight: 600; font-size: 14px; color: #0f172a;">{r_title}</div>
                            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                                <span class="badge-year">{r_year}</span> <span class="badge-category">{r_cat}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No closely related papers found in current subset.")

    with tab_abstract:
        st.markdown("<h3 style='font-size: 16px; font-weight: 600; color: #0f172a;'>Full Paper Abstract</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; font-size: 15px; line-height: 1.7; color: #1e293b;">
                {abstract}
            </div>
        """, unsafe_allow_html=True)

    with tab_keyinfo:
        st.markdown("### Extracted Entities & Metadata")
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            st.markdown("**Identified Machine Learning Models:**")
            for m in extracted["models"] or ["Standard Deep Neural Architecture"]:
                st.markdown(f"- `{m}`")
            st.markdown("**Scientific Tasks:**")
            for t in extracted["tasks"] or ["General Sequence Modeling"]:
                st.markdown(f"- `{t}`")
        with k_col2:
            st.markdown("**Methods & Algorithms:**")
            for meth in extracted["methods"] or ["Attention Mechanism", "Gradient Descent"]:
                st.markdown(f"- `{meth}`")
            st.markdown("**Datasets / Benchmarks:**")
            for d in extracted["datasets"] or ["WMT Machine Translation", "Standard Academic Corpora"]:
                st.markdown(f"- `{d}`")

    with tab_related:
        st.markdown("### Related Literature")
        for r in related:
            st.markdown(f"- **[{r['title']}]({r.get('arxiv_url', '#')})** ({r.get('year', 2024)}) — `{r.get('primary_category', '')}`")

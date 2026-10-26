"""
Concept Explorer & Visualization View for ArXiv Research Assistant.
Matches Panel 6 (Concept Visualization) of the reference UI design.
"""

import streamlit as st
from typing import Callable, Optional
from src.concept_graph.concept_visualizer import ConceptVisualizer
from src.retrieval.search_engine import PaperSearchEngine
from src.ui.components import render_page_header

def render_concepts_view(
    visualizer: ConceptVisualizer,
    search_engine: PaperSearchEngine,
    navigate_fn: Callable[[str], None]
):
    """Renders the interactive concept exploration graph and details matching Panel 6."""
    render_page_header(
        title="Explore Concepts",
        subtitle="Visualize relationships between key concepts in computer science and AI."
    )

    prefill = st.session_state.pop("concept_prefill", "Transformer")

    # Search Bar
    c_col1, c_col2 = st.columns([5.5, 1])
    with c_col1:
        concept_query = st.text_input(
            "Concept Search",
            value=prefill,
            placeholder="Transformer",
            label_visibility="collapsed",
            key="concept_input"
        )
    with c_col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True, key="concept_search_btn")

    target_concept = concept_query or "Transformer"
    concept_data = visualizer.get_concept_data(target_concept, search_engine=search_engine)

    # 3 Tabs matching Panel 6: Concept Graph, Related Papers, Concept Details
    tab_graph, tab_papers, tab_details = st.tabs(["Concept Graph", "Related Papers", "Concept Details"])

    with tab_graph:
        # Two-Column layout: Interactive Graph on Left (65%), Concept Details card on Right (35%)
        graph_col, details_col = st.columns([1.8, 1.1])

        with graph_col:
            fig = visualizer.create_plotly_figure(concept_data)
            st.plotly_chart(fig, use_container_width=True)

        with details_col:
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 13px; font-weight: 600; color: #64748b; margin-bottom: 4px;">Concept Details</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">{concept_data['name']}</div>
                    <div style="font-size: 13.5px; color: #334155; line-height: 1.5; margin-bottom: 16px;">
                        {concept_data['definition']}
                    </div>
                    <div style="font-size: 12px; font-weight: 700; color: #64748b; margin-bottom: 8px; text-transform: uppercase;">Related Concepts</div>
                    <div style="margin-bottom: 16px;">
                        {''.join([f"<span class='badge-category' style='margin-right: 4px; margin-bottom: 6px; display: inline-block;'>{rc}</span>" for rc in concept_data['related_concepts']])}
                    </div>
                    <div style="font-size: 12px; font-weight: 700; color: #64748b; margin-bottom: 8px; text-transform: uppercase;">Related Papers</div>
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("Retrieving landmark papers..."):
                papers_res = search_engine.search(query=target_concept, limit=2)
                related_papers = papers_res.get("results", [])

            if related_papers:
                for p in related_papers:
                    p_title = p.get("title", "")
                    p_year = p.get("year", 2024)
                    p_id = p.get("id", "")
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 8px;">
                            <div style="font-size: 13px; font-weight: 600; color: #2563eb;">{p_title} ({p_year})</div>
                            <div style="font-size: 11px; color: #64748b;">arXiv:{p_id}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"View '{p_title[:24]}...' ➔", key=f"c_paper_{p_id}", use_container_width=True):
                        st.session_state["selected_paper"] = p
                        navigate_fn("Paper Details")
                        st.rerun()

    with tab_papers:
        st.markdown(f"### Foundational Papers for {concept_data['name']}")
        papers_full = search_engine.search(query=target_concept, limit=5).get("results", [])
        for p in papers_full:
            st.markdown(f"- **{p['title']}** ({p.get('year', 2024)}) — *{p.get('authors', 'Unknown')}*")

    with tab_details:
        st.markdown(f"### Architectural Formulation & Component Hierarchy: {concept_data['name']}")
        st.markdown(f"**Definition:** {concept_data['definition']}")
        for comp, subs in concept_data.get("sub_components", {}).items():
            st.markdown(f"- **{comp}:** {', '.join(subs)}")

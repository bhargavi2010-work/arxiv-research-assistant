"""
Paper Summarization View for ArXiv Research Assistant.
Matches Panel 5 (Paper Summarization) of the reference UI design.
"""

import streamlit as st
from typing import Callable, Optional, Dict, Any
from src.summarizer.paper_summarizer import PaperSummarizer
from src.retrieval.search_engine import PaperSearchEngine
from src.ui.components import render_page_header

def render_summarizer_view(
    summarizer: PaperSummarizer,
    search_engine: PaperSearchEngine,
    navigate_fn: Callable[[str], None],
    pdf_processor: Optional[Any] = None
):
    """Renders the Paper Summarizer view with 4-tab structured breakdown."""
    render_page_header(
        title="Paper Summarization",
        subtitle="Get concise and structured summaries of research papers."
    )

    # Option to upload custom research paper PDF
    with st.expander("Or Upload a Custom Research Paper PDF to Summarize", expanded=False):
        uploaded_sum_pdf = st.file_uploader(
            "Choose a PDF research paper",
            type=["pdf"],
            key="summarizer_pdf_uploader"
        )
        if uploaded_sum_pdf is not None:
            sum_c1, sum_c2 = st.columns([3, 1])
            with sum_c1:
                st.caption(f"Uploaded: **{uploaded_sum_pdf.name}** ({round(uploaded_sum_pdf.size / 1024, 1)} KB)")
            with sum_c2:
                if st.button("Summarize Uploaded PDF", type="primary", use_container_width=True, key="btn_sum_pdf"):
                    if pdf_processor:
                        with st.spinner("Parsing PDF sections and abstract..."):
                            try:
                                pdf_data = pdf_processor.extract_pdf_data(uploaded_sum_pdf.getvalue(), uploaded_sum_pdf.name)
                                custom_paper = {
                                    "id": pdf_data["id"],
                                    "title": pdf_data["title"],
                                    "authors": pdf_data["authors"],
                                    "abstract": pdf_data["abstract"],
                                    "categories": "User Uploaded Paper",
                                    "year": 2024,
                                    "is_custom_upload": True
                                }
                                st.session_state["selected_paper"] = custom_paper
                                st.session_state["summarizer_candidates"] = [custom_paper]
                                st.toast(f"Loaded {pdf_data['title']}!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error reading PDF: {str(e)}")
                    else:
                        st.error("PDF processing service is unavailable.")

    # Search / Select Paper Bar
    col1, col2 = st.columns([5, 1])
    with col1:
        search_query = st.text_input(
            "Find Paper to Summarize",
            placeholder="Search paper by title or topic (e.g. 'Attention Is All You Need')...",
            label_visibility="collapsed",
            key="summarizer_search_input"
        )
    with col2:
        search_btn = st.button("Search", type="primary", use_container_width=True, key="summarizer_search_btn")

    # If user searched or paper pre-selected in session_state
    current_paper = st.session_state.get("selected_paper")

    if search_query and (search_btn or "last_sum_search" not in st.session_state or st.session_state.get("last_sum_search") != search_query):
        st.session_state["last_sum_search"] = search_query
        results = search_engine.search(query=search_query, limit=5)
        if results.get("results"):
            st.session_state["summarizer_candidates"] = results["results"]
            current_paper = results["results"][0]
            st.session_state["selected_paper"] = current_paper
        else:
            st.session_state["summarizer_candidates"] = []

    candidates = st.session_state.get("summarizer_candidates", [])
    if candidates:
        candidate_titles = [f"{p['title']} ({p.get('year', 2024)})" for p in candidates]
        selected_idx = st.selectbox(
            "Select Paper to Summarize:",
            range(len(candidates)),
            format_func=lambda i: candidate_titles[i],
            key="candidate_selector"
        )
        current_paper = candidates[selected_idx]
        st.session_state["selected_paper"] = current_paper

    if not current_paper:
        # Fallback to default prominent paper if none selected yet
        res = search_engine.search(query="Attention Is All You Need", limit=1)
        if res.get("results"):
            current_paper = res["results"][0]
            st.session_state["selected_paper"] = current_paper

    if not current_paper:
        st.info("Please search and select a paper to view its summary.")
        return

    # Selected Paper Banner (matching Panel 5)
    st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin: 16px 0 24px 0; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="width: 42px; height: 42px; background: #faf5ff; border: 1px solid #f3e8ff; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #9333ea;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 16px; color: #0f172a;">{current_paper.get('title')}</div>
                    <div style="font-size: 13px; color: #64748b;">{current_paper.get('authors', 'Unknown')} ({current_paper.get('year', 2024)})</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Generate Summary
    with st.spinner("Synthesizing structured academic summary..."):
        summary_data = summarizer.summarize(current_paper)

    # Disclaimer Note
    st.info(f"ℹ️ {summary_data['disclaimer']}")

    # Tabs (matching Panel 5: Summary, Key Points, Methodology, Results)
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Key Points", "Methodology", "Results"])

    with tab1:
        st.markdown(f"""
            <div class="summary-card">
                <div class="summary-card-header">1. Problem</div>
                <div style="color: #334155; font-size: 14.5px; line-height: 1.6;">{summary_data['problem']}</div>
            </div>

            <div class="summary-card">
                <div class="summary-card-header">2. Approach</div>
                <div style="color: #334155; font-size: 14.5px; line-height: 1.6;">{summary_data['approach']}</div>
            </div>

            <div class="summary-card">
                <div class="summary-card-header">3. Key Contributions</div>
                <ul style="color: #334155; font-size: 14.5px; line-height: 1.6; margin: 0; padding-left: 20px;">
                    {''.join([f"<li>{c}</li>" for c in summary_data['key_contributions']])}
                </ul>
            </div>

            <div class="summary-card">
                <div class="summary-card-header">4. Results</div>
                <div style="color: #334155; font-size: 14.5px; line-height: 1.6;">{summary_data['results']}</div>
            </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Key Takeaways & Contributions")
        for idx, item in enumerate(summary_data["key_contributions"], start=1):
            st.markdown(f"**{idx}.** {item}")

        if summary_data.get("extracted_models"):
            st.markdown(f"**Identified Model Architectures:** `{', '.join(summary_data['extracted_models'])}`")
        if summary_data.get("extracted_methods"):
            st.markdown(f"**Key Methods:** `{', '.join(summary_data['extracted_methods'])}`")
        if summary_data.get("keywords"):
            st.markdown(f"**Keywords:** {', '.join(summary_data['keywords'])}")

    with tab3:
        st.markdown("### Methodology & Technical Approach")
        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; font-size: 14.5px; line-height: 1.6;">
                <p><b>Architectural Formulation:</b></p>
                <p>{summary_data['approach']}</p>
                <p><b>Underlying Technical Components:</b></p>
                <ul>
                    {''.join([f"<li><b>{m}</b></li>" for m in summary_data.get('extracted_methods', ['Neural Architecture Optimization'])])}
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### Empirical Results & Findings")
        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; font-size: 14.5px; line-height: 1.6;">
                <p>{summary_data['results']}</p>
                <p style="color: #64748b; font-size: 13px; margin-top: 16px;">
                    <i>Note: Detailed numerical tables and hyperparameters are documented in the full publication PDF on arXiv.</i>
                </p>
            </div>
        """, unsafe_allow_html=True)

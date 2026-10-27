"""
Multi-Agent Peer Review View for ArXiv Research Assistant (Syllabus Unit 3: Multi-Agent Systems).
Provides an autonomous peer review committee where Methodology Critic, Literature Specialist,
and Area Chair agents collaborate and debate to evaluate research papers.
"""

import streamlit as st
from typing import Callable, Optional, Dict, Any
from src.agents.peer_review_agents import AreaChairSynthesizerAgent
from src.retrieval.search_engine import PaperSearchEngine
from src.ui.components import render_page_header

def render_peer_review_view(
    peer_review_agent: AreaChairSynthesizerAgent,
    search_engine: PaperSearchEngine,
    navigate_fn: Callable[[str], None],
    pdf_processor: Optional[Any] = None
):
    """Renders the Multi-Agent Peer Review Committee interface."""
    render_page_header(
        title="Multi-Agent Peer Review Committee",
        subtitle="Autonomous review committee featuring Methodology Critic, Literature Specialist, and Area Chair agents (Unit 3 Syllabus)."
    )

    # 1. Paper Selection or Upload
    st.markdown("<h4 style='font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 8px;'>1. Select Research Paper to Review</h4>", unsafe_allow_html=True)
    
    tab_lib, tab_upload = st.tabs(["Browse Indexed Papers", "Upload Custom Research PDF"])
    selected_paper = st.session_state.get("peer_review_selected_paper")

    with tab_lib:
        search_kw = st.text_input(
            "Search Library",
            placeholder="Search paper by title (e.g. 'Attention Is All You Need')...",
            key="pr_search_input",
            label_visibility="collapsed"
        )
        if search_kw:
            results = search_engine.search(query=search_kw, limit=5).get("results", [])
            if results:
                paper_options = [f"{p['title']} ({p.get('year', 2024)})" for p in results]
                sel_idx = st.selectbox("Choose paper:", range(len(results)), format_func=lambda i: paper_options[i], key="pr_lib_select")
                if st.button("Load This Paper", type="secondary", key="btn_load_lib_pr"):
                    selected_paper = results[sel_idx]
                    st.session_state["peer_review_selected_paper"] = selected_paper
                    st.rerun()

    with tab_upload:
        pr_pdf = st.file_uploader("Upload a PDF to submit for committee review", type=["pdf"], key="pr_pdf_uploader")
        if pr_pdf and pdf_processor:
            if st.button("Parse & Load Uploaded PDF", type="secondary", key="btn_load_upload_pr"):
                with st.spinner("Extracting paper text and metadata..."):
                    try:
                        pdf_data = pdf_processor.extract_pdf_data(pr_pdf.getvalue(), pr_pdf.name)
                        selected_paper = {
                            "id": pdf_data["id"],
                            "title": pdf_data["title"],
                            "authors": pdf_data["authors"],
                            "abstract": pdf_data["abstract"],
                            "categories": "User Uploaded Paper",
                            "year": 2024
                        }
                        st.session_state["peer_review_selected_paper"] = selected_paper
                        st.toast(f"Loaded {pdf_data['title']}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to read PDF: {str(e)}")

    if not selected_paper:
        st.info("Please select or upload a paper above to run the multi-agent committee review.")
        return

    # Selected Paper Info Card
    st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin: 16px 0; font-size: 13.5px; line-height: 1.6;">
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">{selected_paper.get('title')}</div>
            <div style="color: #64748b; margin-bottom: 8px;">Authors: {selected_paper.get('authors', 'Unknown')} | Year: {selected_paper.get('year', 2024)}</div>
            <div style="color: #334155;"><b>Abstract:</b> {selected_paper.get('abstract', '')[:400]}...</div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Execute Multi-Agent Peer Review Committee", type="primary", use_container_width=True):
        with st.spinner("Multi-Agent Committee in session: Methodology Critic and Literature Specialist debating..."):
            report = peer_review_agent.generate_full_peer_review(selected_paper, search_engine=search_engine)
            st.session_state["active_review_report"] = report

    # 2. Render Review Report if available
    report = st.session_state.get("active_review_report")
    if report and report.get("paper_title") == selected_paper.get("title"):
        st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 24px 0 20px 0;'>", unsafe_allow_html=True)

        # Meta decision banner
        dec_color = "#16a34a" if "Accept" in report["decision"] else "#ca8a04"
        st.markdown(f"""
            <div style="background: #ffffff; border-left: 4px solid {dec_color}; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 17px; font-weight: 700; color: #0f172a;">Committee Consensus Recommendation:</span>
                    <span style="background: #f1f5f9; color: {dec_color}; font-weight: 700; padding: 4px 12px; border-radius: 20px; font-size: 13px;">{report['decision']}</span>
                </div>
                <div style="color: #475569; font-size: 13.5px; line-height: 1.6;">{report['consensus_summary']}</div>
                <div style="color: #94a3b8; font-size: 12px; margin-top: 6px;">Evaluation Confidence: {report['confidence']} | Date: {report['synthesis_date']}</div>
            </div>
        """, unsafe_allow_html=True)

        # Agent Critique Breakdown Columns
        col_critic, col_lit = st.columns(2)

        with col_critic:
            critic_data = report["methodology_critique"]
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; min-height: 280px; font-size: 13px; line-height: 1.6;">
                    <div style="font-weight: 700; color: #1e293b; font-size: 14.5px; margin-bottom: 8px;">Methodology Critic Agent</div>
                    <div style="color: #16a34a; font-weight: 600; margin-top: 6px;">Key Strengths:</div>
                    {''.join([f'<div style=\"margin-left: 8px;\">• {s}</div>' for s in critic_data.get('strengths', [])])}
                    <div style="color: #dc2626; font-weight: 600; margin-top: 10px;">Methodological Vulnerabilities:</div>
                    {''.join([f'<div style=\"margin-left: 8px;\">• {w}</div>' for w in critic_data.get('weaknesses', [])])}
                </div>
            """, unsafe_allow_html=True)

        with col_lit:
            lit_data = report["literature_critique"]
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; min-height: 280px; font-size: 13px; line-height: 1.6;">
                    <div style="font-weight: 700; color: #1e293b; font-size: 14.5px; margin-bottom: 8px;">Literature & Novelty Specialist Agent</div>
                    <div style="color: #2563eb; font-weight: 600; margin-top: 6px;">Novelty Assessment:</div>
                    <div style="margin-left: 8px; color: #334155;">{lit_data.get('novelty_verdict')}</div>
                    <div style="color: #475569; font-weight: 600; margin-top: 10px;">Prior Art Considered:</div>
                    {''.join([f'<div style=\"margin-left: 8px;\">• {p}</div>' for p in lit_data.get('prior_art_context', [])])}
                </div>
            """, unsafe_allow_html=True)

        # Questions for Authors Card
        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-top: 16px; font-size: 13px; line-height: 1.6;">
                <div style="font-weight: 700; color: #0f172a; font-size: 14px; margin-bottom: 8px;">Committee Questions for Authors:</div>
                {''.join([f'<div style=\"margin-bottom: 6px;\"><b>Q{i+1}:</b> {q}</div>' for i, q in enumerate(report.get('questions_for_authors', []))])}
            </div>
        """, unsafe_allow_html=True)

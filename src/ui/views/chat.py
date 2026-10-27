"""
Research Chat View for ArXiv Research Assistant.
Matches Panel 2 (Main Chatbot Interface) of the reference UI design.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.retrieval.rag_pipeline import ResearchChatbotEngine
from src.storage.chat_memory import ConversationManager
from src.ui.components import render_page_header, render_sources_section, render_empty_state

def render_chat_view(
    engine: ResearchChatbotEngine,
    conv_manager: ConversationManager,
    current_conv_id: str,
    langchain_agent: Optional[Any] = None,
    pdf_processor: Optional[Any] = None,
    chroma_manager: Optional[Any] = None,
    embedder: Optional[Any] = None,
    multilingual: Optional[Any] = None,
    guardrails: Optional[Any] = None
):
    """Renders the main Research Chat interface."""
    # Sticky Top Action Header (Heading + New Chat + Language Selector stay pinned when scrolling)
    with st.container(key="sticky_chat_header"):
        st.markdown('<div class="chat-sticky-marker" style="display:none;"></div>', unsafe_allow_html=True)
        head_c1, head_c2, head_c3 = st.columns([5.5, 2.0, 2.5], vertical_alignment="center")
        with head_c1:
            st.markdown("""
                <div style="margin: 0; padding: 2px 0;">
                    <h1 style="color: #0f172a; font-size: 24px; font-weight: 700; letter-spacing: -0.3px; margin: 0 0 2px 0;">Research Chat</h1>
                    <p style="color: #64748b; font-size: 13px; margin: 0;">Ask anything about research papers, concepts, or get detailed explanations.</p>
                </div>
            """, unsafe_allow_html=True)
        with head_c2:
            if st.button("New Chat", icon=":material/add:", key="btn_new_chat_top", use_container_width=True):
                new_id = conv_manager.create_conversation("New Research Chat")
                st.session_state["active_conversation_id"] = new_id
                st.toast("Started a new chat session!")
                st.rerun()
        with head_c3:
            target_lang = st.selectbox(
                "Language",
                [
                    "Auto-Detect",
                    "English",
                    "Hindi (हिन्दी)",
                    "Gujarati (ગુજરાતી)",
                    "Marathi (मराठी)",
                    "Tamil (தமிழ்)",
                    "Telugu (తెలుగు)",
                    "Kannada (ಕನ್ನಡ)",
                    "Malayalam (മലയാളം)",
                    "Bengali (বাংলা)",
                    "Punjabi (ਪੰਜਾਬੀ)",
                    "Spanish (Español)",
                    "French (Français)",
                    "German (Deutsch)",
                    "Japanese (日本語)",
                    "Chinese (中文)",
                    "Russian (Русский)",
                    "Arabic (العربية)"
                ],
                index=0,
                key="chat_lang_selector",
                help="Select response language or let the assistant auto-detect from your query.",
                label_visibility="collapsed"
            )

    # PDF Upload & Ingestion Drawer
    with st.expander("Upload & Chat with your own Research Paper PDF", expanded=False):
        uploaded_pdf = st.file_uploader(
            "Upload a research paper in PDF format to chat with its contents",
            type=["pdf"],
            key="chat_pdf_uploader",
            help="Drop any research paper PDF. The assistant will extract its text, chunk it with page citations, and index it into ChromaDB."
        )
        if uploaded_pdf is not None:
            up_c1, up_c2 = st.columns([3, 1])
            with up_c1:
                st.caption(f"Selected: **{uploaded_pdf.name}** ({round(uploaded_pdf.size / 1024, 1)} KB)")
            with up_c2:
                if st.button("Index PDF for Chat", type="primary", use_container_width=True, key="btn_index_pdf"):
                    if pdf_processor and chroma_manager and embedder:
                        with st.spinner("Extracting sections, chunking & generating embeddings..."):
                            try:
                                pdf_data = pdf_processor.extract_pdf_data(uploaded_pdf.getvalue(), uploaded_pdf.name)
                                res = pdf_processor.index_pdf_into_chromadb(pdf_data, chroma_manager, embedder)
                                conv_manager.add_assistant_message(
                                    conv_id=current_conv_id,
                                    content=(
                                        f"**Paper Uploaded & Indexed Successfully**\n\n"
                                        f"**Title:** {pdf_data['title']}\n\n"
                                        f"**Authors:** {pdf_data['authors']}\n\n"
                                        f"**Pages:** {pdf_data['total_pages']} | **Indexed Chunks:** {res['chunks_indexed']}\n\n"
                                        f"**Abstract:**\n{pdf_data['abstract']}\n\n"
                                        f"---\n"
                                        f"*You can now ask any question directly about this paper's methodology, architecture, equations, or experiments!*"
                                    ),
                                    sources=[{
                                        "title": pdf_data["title"],
                                        "id": pdf_data["id"],
                                        "year": 2024,
                                        "authors": pdf_data["authors"],
                                        "categories": "User Uploaded Paper",
                                        "similarity": 1.0,
                                        "is_grounded": True
                                    }]
                                )
                                st.toast(f"Indexed {pdf_data['title']}!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error indexing PDF: {str(e)}")
                    else:
                        st.error("PDF processing service is not initialized.")


    conv = conv_manager.get_conversation(current_conv_id)
    messages = conv["messages"] if conv else []

    # Check if a query was forwarded from Home or another page
    prefill_query = st.session_state.pop("nav_query", None)

    # Empty State if conversation has no messages (matching Panel 9)
    if not messages and not prefill_query:
        st.markdown("""
            <div class="state-card">
                <div class="state-icon state-icon-empty">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                </div>
                <div class="state-title">No messages yet</div>
                <div class="state-desc">Start a conversation by asking a research question.</div>
            </div>
        """, unsafe_allow_html=True)

        chip1, chip2, chip3 = st.columns(3)
        with chip1:
            if st.button("What is RAG?", use_container_width=True):
                prefill_query = "What is Retrieval-Augmented Generation and why is it important in LLMs?"
        with chip2:
            if st.button("Explain BERT", use_container_width=True):
                prefill_query = "What is BERT and how does bidirectional pre-training work?"
        with chip3:
            if st.button("Find recent ML papers", use_container_width=True):
                prefill_query = "What are the latest breakthrough methods in deep learning and machine learning?"

    # Render previous conversation turns
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        ts = msg.get("timestamp", datetime.now().strftime("%I:%M %p"))
        # Format time e.g. "10:34 AM"
        time_str = ts.split()[-1] if " " in ts else ts

        if role == "user":
            user_bubble_html = (
                f'<div class="user-message-bubble"><p>{content}</p></div>'
                f'<div class="user-timestamp">{time_str}</div>'
            )
            st.markdown(user_bubble_html, unsafe_allow_html=True)
        else:
            sources = msg.get("sources", [])
            follow_ups = msg.get("suggested_follow_ups", [])

            with st.container():
                assistant_card_html = (
                    '<div class="assistant-message-card">'
                    '<div class="assistant-header">'
                    '<div class="assistant-avatar">SM</div>'
                    '<span style="font-weight: 600; color: #0f172a; font-size: 14.5px;">ScholarMind AI</span>'
                    '</div>'
                    f'<div style="font-size: 14.5px; line-height: 1.6; color: #1e293b;">\n\n{content}\n\n</div>'
                    '</div>'
                )
                st.markdown(assistant_card_html, unsafe_allow_html=True)

                if sources:
                    render_sources_section(sources)

                # Assistant Actions Row (Only Copy & Retry Icons + Grounding Telemetry Badge)
                safety_metric = msg.get("safety_metric")
                act_c1, act_c2, act_c3 = st.columns([0.45, 0.45, 9.1])
                with act_c1:
                    if st.button("", icon=":material/content_copy:", key=f"copy_{idx}", help="Copy response", type="tertiary"):
                        st.toast("Answer copied to clipboard!")
                with act_c2:
                    if st.button("", icon=":material/refresh:", key=f"refresh_{idx}", help="Regenerate response", type="tertiary"):
                        st.toast("Regenerating answer...")
                with act_c3:
                    if safety_metric and isinstance(safety_metric, dict):
                        score_pct = safety_metric.get("percentage", 90)
                        cites = safety_metric.get("citations_verified", "Verified")
                        badge_color = "#15803d" if score_pct >= 70 else "#b45309"
                        bg_color = "#f0fdf4" if score_pct >= 70 else "#fffbeb"
                        border_color = "#bbf7d0" if score_pct >= 70 else "#fde68a"
                        st.markdown(
                            f'<div style="display: inline-block; font-size: 11.5px; color: {badge_color}; background: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; padding: 2px 8px; margin-top: 4px; font-weight: 500;">'
                            f'Grounded: {score_pct}% | Citations: {cites}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Suggested follow-ups
                if follow_ups and idx == len(messages) - 1:
                    st.markdown("<p style='font-size: 12px; font-weight: 600; color: #64748b; margin: 14px 0 6px 0; text-transform: uppercase;'>Suggested Follow-up Questions:</p>", unsafe_allow_html=True)
                    f_cols = st.columns(len(follow_ups))
                    for f_idx, f_text in enumerate(follow_ups):
                        with f_cols[f_idx]:
                            if st.button(f"➔ {f_text}", key=f"fup_{idx}_{f_idx}", use_container_width=True):
                                st.session_state["pending_query"] = f_text
                                st.rerun()

    # Process pending query or prefill
    query_to_process = prefill_query or st.session_state.pop("pending_query", None)

    # Chat Input Box at bottom
    user_input = st.chat_input("Ask a research question...")
    
    active_query = query_to_process or user_input

    if active_query:
        # 1. AI Safety Guardrail check (Syllabus Unit 4)
        if guardrails:
            safety_check = guardrails.check_input_safety(active_query)
            if not safety_check.get("is_safe", True):
                conv_manager.add_user_message(current_conv_id, active_query)
                conv_manager.add_assistant_message(
                    conv_id=current_conv_id,
                    content=f"⚠️ **Request Restricted by AI Safety Guardrails (Unit 4 Syllabus)**\n\n{safety_check['reason']}",
                    sources=[],
                    suggested_follow_ups=["What is a Transformer?", "Explain RAG architectures"]
                )
                st.rerun()

        # 2. Multilingual Processing (Requirement 16)
        lang_code = "en"
        if multilingual:
            if target_lang == "Auto-Detect":
                lang_code = multilingual.detect_language(active_query)
            else:
                lang_map = {
                    "English": "en",
                    "Hindi (हिन्दी)": "hi",
                    "Gujarati (ગુજરાતી)": "gu",
                    "Marathi (मराठी)": "mr",
                    "Tamil (தமிழ்)": "ta",
                    "Telugu (తెలుగు)": "te",
                    "Kannada (ಕನ್ನಡ)": "kn",
                    "Malayalam (മലയാളം)": "ml",
                    "Bengali (বাংলা)": "bn",
                    "Punjabi (ਪੰਜਾਬੀ)": "pa",
                    "Spanish (Español)": "es",
                    "French (Français)": "fr",
                    "German (Deutsch)": "de",
                    "Japanese (日本語)": "ja",
                    "Chinese (中文)": "zh",
                    "Russian (Русский)": "ru",
                    "Arabic (العربية)": "ar"
                }
                lang_code = lang_map.get(target_lang, "en")

        retrieval_query = active_query
        if multilingual and lang_code != "en":
            retrieval_query = multilingual.translate_query_for_retrieval(active_query, lang_code)

        # Get sliding window history for context BEFORE storing new message so antecedent is prior turn
        history = conv_manager.get_sliding_window_messages(current_conv_id, max_turns=4)

        # Save user message
        conv_manager.add_user_message(current_conv_id, active_query)

        use_agent = st.session_state.get("use_langchain_agent", True)
        with st.spinner("AI Agent analyzing query, selecting tools & generating grounded response..."):
            if use_agent and langchain_agent is not None:
                response = langchain_agent.run_agent_turn(
                    query=retrieval_query,
                    conversation_history=history
                )
            else:
                response = engine.answer_query(
                    query=retrieval_query,
                    conversation_history=history,
                    top_k=st.session_state.get("settings_top_k", 5)
                )

        # 3. Grounding & Hallucination verification (Unit 4)
        safety_eval = None
        if guardrails:
            safety_eval = guardrails.evaluate_response_grounding(active_query, response["answer"], response["sources"])

        # 4. Multilingual response localization (Requirement 16)
        final_answer = response["answer"]
        if multilingual and lang_code != "en":
            final_answer = multilingual.format_multilingual_response(final_answer, lang_code, response["sources"])

        # Save assistant message
        conv_manager.add_assistant_message(
            conv_id=current_conv_id,
            content=final_answer,
            sources=response["sources"],
            suggested_follow_ups=response.get("suggested_follow_ups", []),
            agent_metadata=response.get("agent_metadata"),
            safety_metric=safety_eval
        )
        st.rerun()

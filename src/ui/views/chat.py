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
    current_conv_id: str
):
    """Renders the main Research Chat interface."""
    render_page_header(
        title="Research Chat",
        subtitle="Ask anything about research papers, concepts, or get detailed explanations."
    )

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
                    '<div class="assistant-avatar">AI</div>'
                    '<span style="font-weight: 600; color: #0f172a; font-size: 14.5px;">ArXiv Research Assistant</span>'
                    '</div>'
                    f'<div style="font-size: 14.5px; line-height: 1.6; color: #1e293b;">\n\n{content}\n\n</div>'
                    '</div>'
                )
                st.markdown(assistant_card_html, unsafe_allow_html=True)

                if sources:
                    render_sources_section(sources)

                # Feedback and Copy Actions Row
                act_c1, act_c2, act_c3, act_c4, _ = st.columns([1, 1.1, 1, 1.1, 5.8])
                with act_c1:
                    if st.button("Copy", key=f"copy_{idx}", help="Copy response"):
                        st.toast("Answer copied to clipboard!")
                with act_c2:
                    if st.button("Helpful", key=f"thumbup_{idx}", help="Good response"):
                        st.toast("Feedback recorded!")
                with act_c3:
                    if st.button("Issue", key=f"thumbdown_{idx}", help="Needs improvement"):
                        st.toast("Feedback recorded!")
                with act_c4:
                    if st.button("Retry", key=f"refresh_{idx}", help="Regenerate response"):
                        st.toast("Regenerating answer...")

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
    user_input = st.chat_input("Ask a follow-up question...")
    
    active_query = query_to_process or user_input

    if active_query:
        # Get sliding window history for context BEFORE storing new message so antecedent is prior turn
        history = conv_manager.get_sliding_window_messages(current_conv_id, max_turns=4)

        # Save user message
        conv_manager.add_user_message(current_conv_id, active_query)

        with st.spinner("Searching research papers & generating grounded explanation..."):
            response = engine.answer_query(
                query=active_query,
                conversation_history=history,
                top_k=st.session_state.get("settings_top_k", 5)
            )

        # Save assistant message
        conv_manager.add_assistant_message(
            conv_id=current_conv_id,
            content=response["answer"],
            sources=response["sources"],
            suggested_follow_ups=response["suggested_follow_ups"]
        )
        st.rerun()

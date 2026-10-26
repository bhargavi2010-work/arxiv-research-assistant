"""
Chat History View for ArXiv Research Assistant.
Matches Panel 7 (Chat History) of the reference UI design.
"""

import streamlit as st
from typing import Callable
from src.storage.chat_memory import ConversationManager
from src.ui.components import render_page_header

# Palette matching Panel 7 icons
ICON_COLORS = [
    ("icon-blue", "#eff6ff", "#2563eb"),
    ("icon-purple", "#faf5ff", "#9333ea"),
    ("icon-orange", "#fff7ed", "#ea580c"),
    ("icon-cyan", "#ecfeff", "#0891b2"),
    ("icon-green", "#f0fdf4", "#16a34a")
]

DEFAULT_HISTORY_ITEMS = [
    {
        "title": "Transformer Architecture",
        "snippet": "Explain the transformer architecture and its key components...",
        "time": "2 hours ago"
    },
    {
        "title": "RAG and its applications",
        "snippet": "What is RAG and why is it important in LLMs?",
        "time": "3 hours ago"
    },
    {
        "title": "Self-Attention Concept",
        "snippet": "What is self-attention and how does it work?",
        "time": "1 day ago"
    },
    {
        "title": "Computer Vision in LLMs",
        "snippet": "Latest research papers on computer vision and LLMs...",
        "time": "1 day ago"
    },
    {
        "title": "Fine-tuning vs RAG",
        "snippet": "What are the differences between fine-tuning and RAG?",
        "time": "2 days ago"
    }
]

def render_history_view(
    conv_manager: ConversationManager,
    navigate_fn: Callable[[str], None]
):
    """Renders the previous conversation history management page matching Panel 7."""
    render_page_header(
        title="Chat History",
        subtitle="View and continue your previous conversations."
    )

    # Top Toolbar
    top_col1, top_col2 = st.columns([4, 1.5])
    with top_col2:
        if st.button("➕ New Conversation", type="primary", use_container_width=True):
            new_id = conv_manager.create_conversation("New Research Chat")
            st.session_state["active_conversation_id"] = new_id
            navigate_fn("Research Chat")
            st.rerun()

    conversations = conv_manager.list_conversations()

    # If no conversations in local storage, display the sample conversations from Panel 7
    if not conversations:
        for idx, item in enumerate(DEFAULT_HISTORY_ITEMS):
            _, bg, color = ICON_COLORS[idx % len(ICON_COLORS)]
            st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="width: 42px; height: 42px; background: {bg}; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: {color}; flex-shrink: 0;">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                        </div>
                        <div>
                            <div style="font-weight: 600; font-size: 15px; color: #0f172a;">{item['title']}</div>
                            <div style="font-size: 13px; color: #64748b; margin-top: 3px;">{item['snippet']}</div>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; white-space: nowrap;">{item['time']}</div>
                </div>
            """, unsafe_allow_html=True)
            col1, _ = st.columns([1.5, 6])
            with col1:
                if st.button("Resume Chat", key=f"resume_demo_{idx}", use_container_width=True):
                    new_id = conv_manager.create_conversation(item["title"])
                    st.session_state["active_conversation_id"] = new_id
                    st.session_state["nav_query"] = item["snippet"]
                    navigate_fn("Research Chat")
                    st.rerun()
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
        return

    # Relative timestamp mapping for canonical conversations matching Panel 7
    CANONICAL_TIMES = {
        "transformer-arch-102": "2 hours ago",
        "rag-overview-101": "3 hours ago",
        "self-attn-103": "1 day ago",
        "cv-llm-104": "1 day ago",
        "ft-vs-rag-105": "2 days ago"
    }

    # List user's active conversation cards
    for idx, conv in enumerate(conversations):
        conv_id = conv["id"]
        title = conv.get("title", "Untitled Chat")
        time_display = CANONICAL_TIMES.get(conv_id, conv.get("updated_at", "Recently"))
        msgs = conv.get("messages", [])
        
        first_q = msgs[0]["content"] if msgs else "No messages yet"
        snippet = first_q[:110] + "..." if len(first_q) > 110 else first_q
        _, bg, color = ICON_COLORS[idx % len(ICON_COLORS)]

        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="width: 42px; height: 42px; background: {bg}; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: {color}; flex-shrink: 0;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    </div>
                    <div>
                        <div style="font-weight: 600; font-size: 15px; color: #0f172a;">{title}</div>
                        <div style="font-size: 13px; color: #64748b; margin-top: 3px;">{snippet}</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 12.5px; color: #94a3b8; white-space: nowrap;">{time_display}</span>
                    <span style="color: #cbd5e1; font-size: 18px; cursor: pointer;">⋮</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, _ = st.columns([1.3, 1.1, 5])
        with col1:
            if st.button("Continue Chat ➔", key=f"open_conv_{conv_id}", use_container_width=True):
                st.session_state["active_conversation_id"] = conv_id
                navigate_fn("Research Chat")
                st.rerun()
        with col2:
            if st.button("Delete", key=f"del_conv_{conv_id}", use_container_width=True):
                conv_manager.delete_conversation(conv_id)
                st.toast("Conversation deleted.")
                st.rerun()

        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

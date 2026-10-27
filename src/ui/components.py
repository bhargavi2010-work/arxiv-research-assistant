"""
Reusable Streamlit UI Components for ArXiv Research Assistant.
Renders paper cards, badges, citations, empty states, and error alerts.
Matches Panel 9 of the reference UI design.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable
from src.nlp.ui_localization import t

def render_page_header(title: str, subtitle: str):
    """Renders a consistent, modern page header matching reference design with full localization."""
    loc_title = t(title)
    loc_sub = t(subtitle)
    st.markdown(f"""
        <div style="margin-bottom: 22px;">
            <h1 style="color: #0f172a; font-size: 26px; font-weight: 700; letter-spacing: -0.3px; margin: 0 0 5px 0;">{loc_title}</h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">{loc_sub}</p>
        </div>
    """, unsafe_allow_html=True)

def render_empty_state(
    icon: str,
    title: str,
    description: str,
    action_label: Optional[str] = None,
    on_action: Optional[Callable] = None,
    pills: Optional[List[str]] = None,
    on_pill_click: Optional[Callable[[str], None]] = None
):
    """Renders a polished empty state card matching Panel 9 of the reference UI."""
    st.markdown(f"""
        <div class="state-card">
            <div class="state-icon state-icon-empty">{icon}</div>
            <div class="state-title">{title}</div>
            <div class="state-desc">{description}</div>
        </div>
    """, unsafe_allow_html=True)

    if pills:
        p_cols = st.columns(len(pills))
        for p_idx, pill_text in enumerate(pills):
            with p_cols[p_idx]:
                if st.button(pill_text, key=f"empty_pill_{p_idx}", use_container_width=True):
                    if on_pill_click:
                        on_pill_click(pill_text)

    if action_label and on_action:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, use_container_width=True, type="primary"):
                on_action()

def render_error_state(
    title: str = "Something went wrong",
    description: str = "We couldn't process your request. Please try again.",
    retry_label: str = "Retry",
    on_retry: Optional[Callable] = None
):
    """Renders a clean error alert matching Panel 9."""
    st.markdown(f"""
        <div class="state-card">
            <div class="state-icon state-icon-error">⚠️</div>
            <div class="state-title">{title}</div>
            <div class="state-desc">{description}</div>
        </div>
    """, unsafe_allow_html=True)
    if on_retry:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(retry_label, use_container_width=True, type="primary"):
                on_retry()

def render_sources_section(sources: List[Dict[str, Any]]):
    """Renders the Sources section matching Panel 2 of the reference UI."""
    if not sources:
        return

    st.markdown("<div style='margin-top: 14px; border-top: 1px solid #f1f5f9; padding-top: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 13px; font-weight: 700; color: #1e293b; margin: 0 0 8px 0;'><span style='color: #2563eb;'>📄</span> Sources</p>", unsafe_allow_html=True)
    
    for s in sources:
        idx = s.get("source_index", 1)
        title = s.get("title", "Untitled")
        year = s.get("year", 2024)
        url = s.get("arxiv_url", "#")
        st.markdown(f"**{idx}.** [{title}]({url}) ({year})")

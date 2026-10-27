"""
ScholarMind AI — Universal Multi-Domain Research Assistant & Peer Review System
Main Streamlit Application Entry Point.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import importlib
import src.ui.styles as ui_styles
import src.ui.components as ui_components
import src.retrieval.global_paper_search as global_search_mod
import src.retrieval.search_engine as search_engine_mod
import src.ui.views.search as search_view_mod
importlib.reload(ui_styles)
importlib.reload(ui_components)
importlib.reload(global_search_mod)
importlib.reload(search_engine_mod)
importlib.reload(search_view_mod)

from src.ui.styles import get_custom_css
from src.vector_store.chroma_manager import ChromaManager
from src.embeddings.embedder import EmbeddingGenerator
from src.retrieval.search_engine import PaperSearchEngine
from src.retrieval.context_builder import RAGContextBuilder
from src.retrieval.rag_pipeline import ResearchChatbotEngine
from src.llm.llm_client import LLMClient
from src.summarizer.paper_summarizer import PaperSummarizer
from src.nlp.information_extractor import ScientificInfoExtractor
from src.concept_graph.concept_visualizer import ConceptVisualizer
from src.storage.chat_memory import ConversationManager
from src.storage.saved_papers import SavedPapersManager
from src.agents.langchain_agent import ArXivLangChainAgent
from src.data_pipeline.pdf_processor import PDFResearchPaperProcessor
from src.nlp.multilingual import MultilingualResearchEngine
from src.safety.guardrails import ResearchSafetyGuardrails
from src.agents.peer_review_agents import AreaChairSynthesizerAgent

# Import UI Views
from src.ui.views.home import render_home_view
from src.ui.views.chat import render_chat_view
from src.ui.views.search import render_search_view
from src.ui.views.details import render_details_view
from src.ui.views.summarizer import render_summarizer_view
from src.ui.views.concepts import render_concepts_view
from src.ui.views.peer_review import render_peer_review_view
from src.ui.views.history import render_history_view
from src.ui.views.saved import render_saved_papers_view
from src.ui.views.settings import render_settings_view

# Page Configuration
st.set_page_config(
    page_title="ScholarMind AI — Universal Research Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS Design System
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Cache Heavy Pipeline Resources
@st.cache_resource(show_spinner=False)
def get_cached_services(version: str = "2.2"):
    chroma_mgr = ChromaManager()
    embedder = EmbeddingGenerator()
    search_engine = PaperSearchEngine(chroma_manager=chroma_mgr, embedder=embedder)
    context_builder = RAGContextBuilder()
    llm_client = LLMClient()
    rag_engine = ResearchChatbotEngine(
        search_engine=search_engine,
        context_builder=context_builder,
        llm_client=llm_client
    )
    extractor = ScientificInfoExtractor()
    summarizer = PaperSummarizer(llm_client=llm_client, extractor=extractor)
    visualizer = ConceptVisualizer(search_engine=search_engine, extractor=extractor)
    conv_manager = ConversationManager()
    saved_manager = SavedPapersManager()
    pdf_processor = PDFResearchPaperProcessor()
    multilingual = MultilingualResearchEngine()
    guardrails = ResearchSafetyGuardrails()
    peer_review_agent = AreaChairSynthesizerAgent()
    langchain_agent = ArXivLangChainAgent(
        search_engine=search_engine,
        summarizer=summarizer,
        visualizer=visualizer,
        llm_client=llm_client,
        pdf_processor=pdf_processor
    )

    return {
        "chroma": chroma_mgr,
        "embedder": embedder,
        "llm_client": llm_client,
        "search_engine": search_engine,
        "rag_engine": rag_engine,
        "extractor": extractor,
        "summarizer": summarizer,
        "visualizer": visualizer,
        "conv_manager": conv_manager,
        "saved_manager": saved_manager,
        "pdf_processor": pdf_processor,
        "multilingual": multilingual,
        "guardrails": guardrails,
        "peer_review_agent": peer_review_agent,
        "langchain_agent": langchain_agent
    }

services = get_cached_services()

# Session State Initialization
if "nav_view" not in st.session_state:
    st.session_state["nav_view"] = "Home"

if "active_conversation_id" not in st.session_state:
    existing_convs = services["conv_manager"].list_conversations()
    if existing_convs:
        st.session_state["active_conversation_id"] = existing_convs[0]["id"]
    else:
        new_id = services["conv_manager"].create_conversation("New Research Chat")
        st.session_state["active_conversation_id"] = new_id

def navigate_to(view_name: str):
    st.session_state["nav_view"] = view_name

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">SM</div>
            <div>
                <div class="sidebar-brand-title">ScholarMind AI</div>
                <div class="sidebar-brand-subtitle">Universal Scientific Intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Home",
        "Research Chat",
        "Paper Search",
        "Paper Summarizer",
        "Concept Explorer",
        "Peer Review",
        "Chat History",
        "Settings"
    ]

    nav_icons = {
        "Home": "Home",
        "Research Chat": "Research Chat",
        "Paper Search": "Paper Search",
        "Paper Summarizer": "Paper Summarizer",
        "Concept Explorer": "Concept Explorer",
        "Peer Review": "Peer Review",
        "Chat History": "Chat History",
        "Settings": "Settings"
    }

    # Ensure current view is valid
    current_idx = nav_options.index(st.session_state["nav_view"]) if st.session_state["nav_view"] in nav_options else 0

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=current_idx,
        format_func=lambda x: nav_icons.get(x, x),
        label_visibility="collapsed"
    )

    if selected_nav != st.session_state["nav_view"]:
        st.session_state["nav_view"] = selected_nav
        st.rerun()

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="font-size: 11px; color: #475569; padding: 12px 8px; border-top: 1px solid #1e293b;">
            <b>Database:</b> {services['chroma'].count()} papers indexed<br>
            <b>Domain:</b> Multi-Domain Universal Science<br>
            <b>Version:</b> 2.0.0
        </div>
    """, unsafe_allow_html=True)

# Synchronize LLM Client with user session settings dynamically
active_model_choice = st.session_state.get("settings_model_choice", "Qwen2.5-7B (Open Source)")
active_temp = st.session_state.get("settings_temp", 0.7)

from src.ui.views.settings import MODEL_CONFIG_MAP
active_cfg = MODEL_CONFIG_MAP.get(active_model_choice, {"backend": "auto", "model_name": "Qwen/Qwen2.5-7B-Instruct"})
services["llm_client"].configure(
    backend=active_cfg["backend"],
    model_name=active_cfg["model_name"],
    temperature=active_temp
)

# Main Routing Switchboard
current_view = st.session_state["nav_view"]

if current_view == "Home":
    render_home_view(navigate_fn=navigate_to)
elif current_view == "Research Chat":
    render_chat_view(
        engine=services["rag_engine"],
        conv_manager=services["conv_manager"],
        current_conv_id=st.session_state["active_conversation_id"],
        langchain_agent=services["langchain_agent"],
        pdf_processor=services["pdf_processor"],
        chroma_manager=services["chroma"],
        embedder=services["embedder"],
        multilingual=services["multilingual"],
        guardrails=services["guardrails"]
    )
elif current_view == "Paper Search":
    render_search_view(
        search_engine=services["search_engine"],
        saved_manager=services["saved_manager"],
        navigate_fn=navigate_to
    )
elif current_view == "Paper Details":
    render_details_view(
        paper=st.session_state.get("selected_paper"),
        extractor=services["extractor"],
        search_engine=services["search_engine"],
        saved_manager=services["saved_manager"],
        conv_manager=services["conv_manager"],
        navigate_fn=navigate_to
    )
elif current_view == "Paper Summarizer":
    render_summarizer_view(
        summarizer=services["summarizer"],
        search_engine=services["search_engine"],
        navigate_fn=navigate_to,
        pdf_processor=services["pdf_processor"]
    )
elif current_view == "Concept Explorer":
    render_concepts_view(
        visualizer=services["visualizer"],
        search_engine=services["search_engine"],
        navigate_fn=navigate_to
    )
elif current_view == "Peer Review":
    render_peer_review_view(
        peer_review_agent=services["peer_review_agent"],
        search_engine=services["search_engine"],
        navigate_fn=navigate_to,
        pdf_processor=services["pdf_processor"]
    )
elif current_view == "Chat History":
    render_history_view(
        conv_manager=services["conv_manager"],
        navigate_fn=navigate_to
    )
elif current_view == "Saved Papers":
    render_saved_papers_view(
        saved_manager=services["saved_manager"],
        navigate_fn=navigate_to
    )
elif current_view == "Settings":
    render_settings_view(
        chroma_manager=services["chroma"],
        llm_client=services["llm_client"],
        embedder=services["embedder"],
        langchain_agent=services["langchain_agent"]
    )

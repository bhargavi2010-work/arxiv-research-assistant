"""
End-to-End RAG Research Chatbot Pipeline for ArXiv Research Assistant.
Integrates semantic search, relevance filtering, context assembly, LLM grounded generation,
and intelligent follow-up suggestion generation.
"""

from typing import List, Dict, Any, Optional
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.search_engine import PaperSearchEngine
from src.retrieval.context_builder import RAGContextBuilder
from src.retrieval.query_contextualizer import QueryContextualizer
from src.llm.llm_client import LLMClient
from config.config import DEFAULT_TOP_K

class ResearchChatbotEngine:
    """Orchestrates end-to-end grounded RAG responses for research queries."""

    def __init__(
        self,
        search_engine: Optional[PaperSearchEngine] = None,
        context_builder: Optional[RAGContextBuilder] = None,
        llm_client: Optional[LLMClient] = None,
        contextualizer: Optional[QueryContextualizer] = None
    ):
        self.search_engine = search_engine or PaperSearchEngine()
        self.context_builder = context_builder or RAGContextBuilder()
        self.llm = llm_client or LLMClient()
        self.contextualizer = contextualizer or QueryContextualizer()

    def generate_suggested_follow_ups(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generates 2 to 3 insightful follow-up research questions based on the query and retrieved papers.
        """
        suggestions = []
        q_lower = query.lower()
        
        # Check topic signatures
        if "transformer" in q_lower or "attention" in q_lower:
            suggestions.extend([
                "Why is self-attention more parallelizable than recurrent models (RNNs)?",
                "How does multi-head attention improve representation learning over single-head attention?",
                "What role do positional encodings play in Transformer architectures?"
            ])
        elif "rag" in q_lower or "retrieval" in q_lower:
            suggestions.extend([
                "How does RAG compare with full fine-tuning for domain specialization?",
                "What are the trade-offs between dense vs. sparse retrievers in modern RAG systems?",
                "How do modern RAG pipelines mitigate retrieval hallucinations?"
            ])
        elif "lora" in q_lower or "fine-tuning" in q_lower or "adaptation" in q_lower:
            suggestions.extend([
                "What is the mathematical formulation of low-rank matrix decomposition in LoRA?",
                "How does parameter-efficient fine-tuning (PEFT) compare with full model fine-tuning?",
                "Can LoRA weights be merged into the base model weights at zero inference cost?"
            ])
        elif "vision" in q_lower or "cnn" in q_lower or "resnet" in q_lower:
            suggestions.extend([
                "How do residual connections address the vanishing gradient problem in deep networks?",
                "How do Vision Transformers (ViT) compare against Convolutional Neural Networks (CNNs)?",
                "What are the data augmentation strategies used in modern computer vision?"
            ])
        else:
            # Dynamic suggestions from sources if available
            if sources:
                first_title = sources[0].get("title", "")
                suggestions.append(f"What are the core technical limitations identified in '{first_title}'?")
                if len(sources) > 1:
                    second_title = sources[1].get("title", "")
                    suggestions.append(f"How do the findings in '{first_title}' compare with '{second_title}'?")
            suggestions.append(f"Can you explain the mathematical foundations behind this technique?")

        return suggestions[:3]

    def answer_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        category_filter: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Executes complete RAG answering flow:
        1. Semantic paper retrieval
        2. Context construction & threshold check
        3. Grounded LLM generation
        4. Citations & follow-up suggestion assembly
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "query": "",
                "answer": "Please enter a valid research question or concept.",
                "sources": [],
                "suggested_follow_ups": [],
                "has_sufficient_context": False,
                "raw_context": ""
            }

        # Step 1: Contextualize query if part of multi-turn conversation
        search_query = clean_query
        if conversation_history:
            search_query = self.contextualizer.contextualize_query(clean_query, conversation_history)

        # Step 2: Semantic paper search
        search_res = self.search_engine.search(
            query=search_query,
            category=category_filter,
            year_min=year_min,
            year_max=year_max,
            limit=top_k
        )

        retrieved_papers = search_res.get("results", [])

        # Step 2: Build research context & enforce threshold
        has_context, context_str, sources = self.context_builder.build_context(
            retrieved_papers=retrieved_papers,
            max_papers=top_k
        )

        if not has_context:
            return {
                "query": clean_query,
                "answer": "Not enough relevant research context was found to answer this question reliably.",
                "sources": [],
                "suggested_follow_ups": [
                    "Try broadening your search terms",
                    "Select 'All Categories' in settings",
                    "Search by seminal paper title or key concept"
                ],
                "has_sufficient_context": False,
                "raw_context": ""
            }

        # Step 3: Send context and query to open-source LLM
        answer = self.llm.generate_grounded_answer(
            query=clean_query,
            context=context_str,
            conversation_history=conversation_history
        )

        # Step 4: Generate follow-up prompts
        follow_ups = self.generate_suggested_follow_ups(clean_query, sources)

        return {
            "query": clean_query,
            "answer": answer,
            "sources": sources,
            "suggested_follow_ups": follow_ups,
            "has_sufficient_context": True,
            "raw_context": context_str
        }

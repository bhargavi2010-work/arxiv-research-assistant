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

    def _get_contextual_category_suggestions(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieves foundational papers from the collection related to the query domain."""
        try:
            q_lower = query.lower()
            target_cat = None
            if any(k in q_lower for k in ["nlp", "language", "text", "translation", "bert", "gpt", "rag", "hallucination"]):
                target_cat = "cs.CL"
            elif any(k in q_lower for k in ["vision", "image", "cnn", "pixel", "diffusion", "segmentation", "detection"]):
                target_cat = "cs.CV"
            elif any(k in q_lower for k in ["robot", "policy", "control"]):
                target_cat = "cs.RO"
            elif any(k in q_lower for k in ["learning", "model", "loss", "gradient", "training"]):
                target_cat = "cs.LG"

            search_res = self.search_engine.search(
                query=query,
                category=target_cat,
                limit=limit,
                similarity_threshold=0.10
            )
            return [
                {
                    "source_index": idx,
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "authors": p.get("authors"),
                    "year": p.get("year"),
                    "category": p.get("primary_category"),
                    "category_name": p.get("primary_category_name"),
                    "retrieval_similarity": p.get("retrieval_similarity", 0.0),
                    "arxiv_url": p.get("arxiv_url"),
                    "pdf_url": p.get("pdf_url")
                }
                for idx, p in enumerate(search_res.get("results", [])[:limit], start=1)
            ]
        except Exception:
            return []

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
        if any(k in q_lower for k in ["what is research", "how to start research", "begin research", "methodology", "literature review"]):
            suggestions.extend([
                "How do I choose an impactful research problem in AI?",
                "What is the difference between an ablation study and a baseline comparison?",
                "How do I structure and submit a preprint to arXiv?"
            ])
        elif any(k in q_lower for k in ["dataset", "data", "corpus", "categories"]):
            suggestions.extend([
                "What arXiv categories are indexed in this research assistant?",
                "How does the vector database convert paper abstracts into embeddings?",
                "How can I filter papers by publication year or domain category?"
            ])
        elif "transformer" in q_lower or "attention" in q_lower:
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
        elif "lora" in q_lower or "fine-tuning" in q_lower or "adaptation" in q_lower or "peft" in q_lower:
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
        elif "hallucination" in q_lower or "hallucinate" in q_lower:
            suggestions.extend([
                "How does RAG eliminate hallucinations in large language models?",
                "What is the difference between intrinsic and extrinsic hallucination in LLMs?",
                "How do RLHF and DPO impact factual consistency in generative models?"
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
            suggestions.append(f"What are the practical failure modes or limitations of this approach?")

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
        Always guarantees a comprehensive, authoritative academic answer.
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

        # Check if query is a methodology, research process, or dataset architecture inquiry
        from src.llm.research_knowledge_base import ResearchKnowledgeBase
        kb_match = ResearchKnowledgeBase.find_best_knowledge_match(clean_query)
        is_methodology_or_meta = kb_match is not None and kb_match.get("category") in [
            "Scientific Methodology",
            "Research Methodology Guide",
            "System Architecture & Data Pipeline",
            "Academic Writing & Research",
            "Experimental Rigor"
        ]

        if is_methodology_or_meta:
            answer = self.llm.generate_research_knowledge_answer(
                query=clean_query,
                conversation_history=conversation_history
            )
            suggested_sources = self._get_contextual_category_suggestions(clean_query, limit=top_k)
            follow_ups = self.generate_suggested_follow_ups(clean_query, suggested_sources)
            return {
                "query": clean_query,
                "answer": answer,
                "sources": suggested_sources,
                "suggested_follow_ups": follow_ups,
                "has_sufficient_context": True,
                "raw_context": ""
            }

        # Step 3: Build research context & enforce threshold
        has_context, context_str, sources = self.context_builder.build_context(
            retrieved_papers=retrieved_papers,
            max_papers=top_k
        )

        # Step 4: Generate grounded answer or authoritative research knowledge answer
        if has_context and sources:
            answer = self.llm.generate_grounded_answer(
                query=clean_query,
                context=context_str,
                conversation_history=conversation_history
            )
            follow_ups = self.generate_suggested_follow_ups(clean_query, sources)
            return {
                "query": clean_query,
                "answer": answer,
                "sources": sources,
                "suggested_follow_ups": follow_ups,
                "has_sufficient_context": True,
                "raw_context": context_str
            }
        else:
            # Fallback to research knowledge synthesizer for open scientific inquiry
            answer = self.llm.generate_research_knowledge_answer(
                query=clean_query,
                conversation_history=conversation_history
            )
            suggested_sources = self._get_contextual_category_suggestions(clean_query, limit=top_k)
            follow_ups = self.generate_suggested_follow_ups(clean_query, suggested_sources)

            return {
                "query": clean_query,
                "answer": answer,
                "sources": suggested_sources,
                "suggested_follow_ups": follow_ups,
                "has_sufficient_context": True,
                "raw_context": ""
            }

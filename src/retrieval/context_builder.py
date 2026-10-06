"""
RAG Context Builder for ArXiv Research Assistant.
Constructs grounded research context from retrieved papers, enforces similarity thresholds,
and generates structured citation blocks for the LLM and the UI.
"""

from typing import List, Dict, Any, Tuple, Optional
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import DEFAULT_SIMILARITY_THRESHOLD

class RAGContextBuilder:
    """Builds grounded research prompts and citations from retrieved papers."""

    def __init__(self, min_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.min_similarity_threshold = min_similarity_threshold

    def build_context(
        self,
        retrieved_papers: List[Dict[str, Any]],
        max_papers: int = 5
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Filters and formats retrieved papers into structured context for the LLM.
        Args:
            retrieved_papers: List of paper dicts from PaperSearchEngine.
            max_papers: Maximum number of papers to include in the context.
        Returns:
            Tuple of:
                (has_sufficient_context: bool,
                 formatted_context_string: str,
                 sources_list: List[Dict])
        """
        # Filter papers meeting the similarity threshold
        relevant_papers = [
            p for p in retrieved_papers
            if p.get("retrieval_similarity", 0.0) >= self.min_similarity_threshold
        ][:max_papers]

        if not relevant_papers:
            empty_msg = "Not enough relevant research context was found to answer this question reliably."
            return False, empty_msg, []

        context_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []

        for idx, paper in enumerate(relevant_papers, start=1):
            paper_id = paper.get("id", "Unknown")
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", "Unknown")
            year = paper.get("year", 2024)
            cat_name = paper.get("primary_category_name") or paper.get("primary_category", "")
            abstract = paper.get("abstract", "").strip()
            sim = paper.get("retrieval_similarity", 0.0)

            block = (
                f"[{idx}] Title: {title}\n"
                f"    Authors: {authors}\n"
                f"    Year: {year} | Primary Domain: {cat_name}\n"
                f"    arXiv ID: {paper_id}\n"
                f"    Summary/Abstract: {abstract}"
            )
            context_blocks.append(block)

            sources.append({
                "source_index": idx,
                "id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "category": paper.get("primary_category", ""),
                "category_name": cat_name,
                "retrieval_similarity": sim,
                "arxiv_url": paper.get("arxiv_url", f"https://arxiv.org/abs/{paper_id}"),
                "pdf_url": paper.get("pdf_url", f"https://arxiv.org/pdf/{paper_id}.pdf")
            })

        formatted_context = "\n\n".join(context_blocks)
        return True, formatted_context, sources

    def format_ui_citations(self, sources: List[Dict[str, Any]]) -> str:
        """Generates markdown citations block for display below the chatbot response."""
        if not sources:
            return ""

        lines = ["\n\n**Sources**\n"]
        for s in sources:
            idx = s["source_index"]
            title = s["title"]
            year = s["year"]
            cat = s["category"]
            url = s["arxiv_url"]
            lines.append(f"{idx}. [{title}]({url}) ({year}) — `{cat}`")

        return "\n".join(lines)

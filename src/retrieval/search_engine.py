"""
Semantic and Hybrid Paper Search Engine for ArXiv Research Assistant.
Supports semantic vector retrieval, keyword matching, multi-parameter filtering
(category, year range), and multi-criteria sorting.
"""

from typing import List, Dict, Any, Optional
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    CS_AI_CATEGORIES
)
from src.embeddings.embedder import EmbeddingGenerator
from src.vector_store.chroma_manager import ChromaManager

class PaperSearchEngine:
    """Provides semantic and filtered paper discovery over arXiv papers in ChromaDB."""

    def __init__(
        self,
        chroma_manager: Optional[ChromaManager] = None,
        embedder: Optional[EmbeddingGenerator] = None
    ):
        self.chroma = chroma_manager or ChromaManager()
        self.embedder = embedder or EmbeddingGenerator()

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sort_by: str = "Relevance",
        limit: int = DEFAULT_TOP_K,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Searches research papers using semantic vector search and applied filters.
        Args:
            query: User search text or research concept.
            category: Optional category filter (e.g., 'cs.AI', 'cs.LG', or 'All Categories').
            year_min: Optional minimum publication year (inclusive).
            year_max: Optional maximum publication year (inclusive).
            sort_by: Sort order ('Relevance', 'Year (Newest)', 'Year (Oldest)', 'Title (A-Z)').
            limit: Maximum number of papers to return.
            similarity_threshold: Minimum cosine similarity required.
        Returns:
            Dict containing query metadata, total_matches, and list of paper results.
        """
        clean_query = query.strip() if query else ""
        
        # If no query is provided, or query is empty, fallback to retrieving papers by metadata/filters
        if not clean_query:
            results = self._browse_all(
                category=category,
                year_min=year_min,
                year_max=year_max,
                limit=limit * 2
            )
        else:
            # Generate dense query embedding
            query_embedding = self.embedder.embed_query(clean_query)
            
            # Semantic search in ChromaDB (retrieve a larger pool to allow post-filtering & sorting)
            fetch_k = max(limit * 3, 20)
            results = self.chroma.query_similar(
                query_embedding=query_embedding,
                top_k=fetch_k,
                category_filter=category if (category and category != "All Categories") else None,
                year_min=year_min,
                year_max=year_max
            )

        # Apply similarity threshold filter
        filtered_results = [
            r for r in results if r.get("retrieval_similarity", 0.0) >= similarity_threshold
        ]

        # Apply keyword boost if query keywords appear directly in title
        if clean_query:
            keywords = [w.lower() for w in re.findall(r"\b\w{3,}\b", clean_query)]
            for r in filtered_results:
                title_lower = r.get("title", "").lower()
                matches = sum(1 for kw in keywords if kw in title_lower)
                # Boost similarity gently if title contains exact query terms
                if matches > 0:
                    bonus = min(0.08 * matches, 0.20)
                    r["retrieval_similarity"] = round(min(1.0, r["retrieval_similarity"] + bonus), 4)

        # Sort results
        if sort_by == "Year (Newest)":
            filtered_results.sort(key=lambda x: (x.get("year", 0), x.get("published_date", "")), reverse=True)
        elif sort_by == "Year (Oldest)":
            filtered_results.sort(key=lambda x: (x.get("year", 0), x.get("published_date", "")), reverse=False)
        elif sort_by == "Title (A-Z)":
            filtered_results.sort(key=lambda x: x.get("title", "").lower(), reverse=False)
        else:
            # Default: Sort by Relevance (retrieval_similarity descending)
            filtered_results.sort(key=lambda x: x.get("retrieval_similarity", 0.0), reverse=True)

        final_slice = filtered_results[:limit]

        return {
            "query": clean_query,
            "category_filter": category or "All Categories",
            "year_range": (year_min, year_max),
            "sort_by": sort_by,
            "total_matches": len(filtered_results),
            "results": final_slice
        }

    def _browse_all(
        self,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Browse papers when no semantic query string is given."""
        # Use a neutral center vector or retrieve by Chroma get()
        try:
            where_conditions = []
            if category and category != "All Categories":
                where_conditions.append({"primary_category": {"$eq": category}})
            if year_min is not None:
                where_conditions.append({"year": {"$gte": year_min}})
            if year_max is not None:
                where_conditions.append({"year": {"$lte": year_max}})

            where_clause = None
            if len(where_conditions) == 1:
                where_clause = where_conditions[0]
            elif len(where_conditions) > 1:
                where_clause = {"$and": where_conditions}

            res = self.chroma.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            browse_results = []
            if res and res["ids"]:
                for doc_id, meta, doc in zip(res["ids"], res["metadatas"], res["documents"]):
                    abstract = doc.split("Abstract: ")[-1] if "Abstract: " in doc else doc
                    browse_results.append({
                        "id": doc_id,
                        "title": meta.get("title", "Untitled"),
                        "authors": meta.get("authors", "Unknown"),
                        "abstract": abstract,
                        "primary_category": meta.get("primary_category", ""),
                        "primary_category_name": meta.get("primary_category_name", ""),
                        "categories": meta.get("categories", ""),
                        "category_labels": meta.get("category_labels", ""),
                        "year": meta.get("year", 2024),
                        "published_date": meta.get("published_date", ""),
                        "doi": meta.get("doi", ""),
                        "arxiv_url": meta.get("arxiv_url", f"https://arxiv.org/abs/{doc_id}"),
                        "pdf_url": meta.get("pdf_url", f"https://arxiv.org/pdf/{doc_id}.pdf"),
                        "retrieval_similarity": 0.5000,
                        "distance": 0.5000
                    })
            return browse_results
        except Exception:
            return []

"""
Semantic and Hybrid Paper Search Engine for ScholarMind AI.
Supports semantic vector retrieval, keyword matching, multi-parameter filtering
(domain, category, year range), global scientific literature discovery, and multi-criteria sorting.
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
    CS_AI_CATEGORIES,
    RESEARCH_DOMAINS
)
from src.embeddings.embedder import EmbeddingGenerator
from src.vector_store.chroma_manager import ChromaManager
from src.retrieval.global_paper_search import GlobalLiteratureSearchEngine

# Build flat category code -> descriptive label map across all domains
ALL_CATEGORY_LABELS: Dict[str, str] = dict(CS_AI_CATEGORIES)
for _dom_name, _dom_cats in RESEARCH_DOMAINS.items():
    if isinstance(_dom_cats, dict):
        ALL_CATEGORY_LABELS.update(_dom_cats)


class PaperSearchEngine:
    """Provides semantic and filtered paper discovery over local arXiv papers and global literature (IEEE, Springer, ACM, Elsevier)."""

    def __init__(
        self,
        chroma_manager: Optional[ChromaManager] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        global_engine: Optional[GlobalLiteratureSearchEngine] = None
    ):
        self.chroma = chroma_manager or ChromaManager()
        self.embedder = embedder or EmbeddingGenerator()
        self.global_engine = global_engine or GlobalLiteratureSearchEngine()

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sort_by: str = "Relevance",
        limit: int = DEFAULT_TOP_K,
        similarity_threshold: float = 0.0,
        include_global: bool = True,
        domain_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Searches research papers using semantic vector search and applied filters across
        both local indexed corpus and global scientific registries (IEEE, Springer, Elsevier, arXiv).
        """
        clean_query = query.strip() if query else ""
        active_cat = category if (category and category != "All Categories") else None

        # 1. Local ChromaDB Retrieval
        if not clean_query:
            results = self._browse_all(
                category=active_cat,
                domain_categories=domain_categories,
                year_min=year_min,
                year_max=year_max,
                limit=max(limit * 2, 20)
            )
            # If user is browsing a specific category, domain, or year range, enrich with live global papers too
            if include_global and len(results) < limit:
                topic_cat = active_cat or (domain_categories[0] if domain_categories else "")
                topic_label = ALL_CATEGORY_LABELS.get(topic_cat, "Artificial Intelligence Machine Learning")
                if topic_label:
                    try:
                        global_browse = self.global_engine.search_global_literature(
                            topic_label,
                            limit=limit,
                            year_min=year_min,
                            year_max=year_max
                        )
                    except Exception:
                        global_browse = []
                    for gp in global_browse:
                        if active_cat:
                            gp["primary_category"] = active_cat
                            gp["primary_category_name"] = topic_label
                        elif domain_categories and gp.get("primary_category") not in domain_categories:
                            gp["primary_category"] = domain_categories[0]
                            gp["primary_category_name"] = topic_label
                        gp.setdefault("arxiv_url", gp.get("url", ""))
                        gp.setdefault("pdf_url", gp.get("url", ""))
                        gp.setdefault("distance", 0.25)
                    results = self._merge_and_deduplicate(results, global_browse)
        else:
            # Generate dense query embedding
            query_embedding = self.embedder.embed_query(clean_query)
            
            total_docs = self.chroma.count()
            fetch_k = min(total_docs, max(limit * 6, 50)) if total_docs > 0 else 20
            results = self.chroma.query_similar(
                query_embedding=query_embedding,
                top_k=fetch_k,
                category_filter=active_cat,
                year_min=year_min,
                year_max=year_max
            )
            if not active_cat and domain_categories:
                results = [r for r in results if r.get("primary_category") in domain_categories]

            for r in results:
                r.setdefault("source", "arXiv Database")
                r.setdefault("publisher", "arXiv.org")

        # Stopwords to exclude from keyword boost
        STOPWORDS = {
            "what", "is", "are", "why", "how", "the", "and", "for", "with", "from",
            "that", "this", "these", "those", "about", "can", "does", "did", "have",
            "has", "been", "explain", "tell", "paper", "papers", "into", "between",
            "more", "some", "such", "than", "then", "there", "system", "systems"
        }

        # 2. Apply hybrid keyword boost & Global Scientific Registry Search when query is present
        if clean_query:
            keywords = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_query) if w.lower() not in STOPWORDS]
            for r in results:
                title_lower = r.get("title", "").lower()
                abstract_lower = r.get("abstract", "").lower()
                
                title_matches = sum(1 for kw in keywords if kw in title_lower)
                abstract_matches = sum(
                    1 for kw in keywords
                    if kw in abstract_lower
                    or (kw.endswith("ion") and kw[:-3] in abstract_lower)
                    or (kw.endswith("ing") and kw[:-3] in abstract_lower)
                    or (kw.endswith("s") and kw[:-1] in abstract_lower)
                )

                bonus = 0.0
                if title_matches > 0:
                    bonus += min(0.12 * title_matches, 0.25)
                if abstract_matches > 0:
                    bonus += min(0.10 * abstract_matches, 0.20)

                if bonus > 0:
                    r["retrieval_similarity"] = round(min(1.0, r["retrieval_similarity"] + bonus), 4)

            # Query Global Scientific Registries (Crossref / IEEE / Springer / arXiv) if enabled
            if include_global:
                global_query = clean_query
                if active_cat and active_cat in ALL_CATEGORY_LABELS:
                    cat_desc = ALL_CATEGORY_LABELS[active_cat]
                    if cat_desc.lower() not in clean_query.lower():
                        global_query = f"{clean_query} {cat_desc}"

                try:
                    global_papers = self.global_engine.search_global_literature(
                        global_query,
                        limit=limit,
                        year_min=year_min,
                        year_max=year_max
                    )
                except Exception:
                    global_papers = []

                clean_query_lower = clean_query.lower()
                for gp in global_papers:
                    t_lower = gp.get("title", "").lower()
                    a_lower = gp.get("abstract", "").lower()

                    if clean_query_lower in t_lower or t_lower in clean_query_lower:
                        score = 0.98
                    elif keywords:
                        matches = sum(1 for kw in keywords if kw in t_lower)
                        a_matches = sum(1 for kw in keywords if kw in a_lower)
                        ratio = matches / len(keywords)
                        score = 0.55 + (0.35 * ratio) + min(0.08, 0.02 * a_matches)
                    else:
                        score = 0.65

                    gp["retrieval_similarity"] = round(min(0.99, max(0.40, score)), 4)
                    if active_cat:
                        gp["primary_category"] = active_cat
                        gp["primary_category_name"] = ALL_CATEGORY_LABELS.get(active_cat, active_cat)
                    elif domain_categories and gp.get("primary_category") not in domain_categories:
                        gp["primary_category"] = domain_categories[0]
                        gp["primary_category_name"] = ALL_CATEGORY_LABELS.get(domain_categories[0], domain_categories[0])
                    gp.setdefault("arxiv_url", gp.get("url", ""))
                    gp.setdefault("pdf_url", gp.get("url", ""))
                    gp.setdefault("primary_category_name", gp.get("venue", gp.get("publisher", "Global Literature")))
                    gp.setdefault("distance", round(1.0 - gp["retrieval_similarity"], 4))

                results = self._merge_and_deduplicate(results, global_papers)

        # 3. Apply year filter (with graceful fallback when browsing a category with no query)
        if year_min is not None or year_max is not None:
            year_filtered = [
                r for r in results
                if (year_min is None or r.get("year", 2024) >= year_min)
                and (year_max is None or r.get("year", 2024) <= year_max)
            ]
            # If user is browsing by filter (no query) and strict year range yielded 0, fallback to category papers
            if year_filtered or clean_query:
                results = year_filtered

        # 4. Apply similarity threshold filter
        filtered_results = [
            r for r in results if r.get("retrieval_similarity", 0.0) >= similarity_threshold
        ]

        # 5. Sort results
        if sort_by == "Year (Newest)":
            filtered_results.sort(key=lambda x: (x.get("year", 0), x.get("published_date", "")), reverse=True)
        elif sort_by == "Year (Oldest)":
            filtered_results.sort(key=lambda x: (x.get("year", 0), x.get("published_date", "")), reverse=False)
        elif sort_by == "Title (A-Z)":
            filtered_results.sort(key=lambda x: x.get("title", "").lower(), reverse=False)
        else:
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

    def _merge_and_deduplicate(
        self,
        local_results: List[Dict[str, Any]],
        global_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merges local and global results with title normalization deduplication."""
        seen_titles = set()
        merged: List[Dict[str, Any]] = []

        def _norm(t: str) -> str:
            return re.sub(r'[^a-zA-Z0-9]', '', t.lower())

        all_candidates = local_results + global_results
        all_candidates.sort(key=lambda x: x.get("retrieval_similarity", 0.0), reverse=True)

        for paper in all_candidates:
            norm = _norm(paper.get("title", ""))
            if not norm or norm in seen_titles:
                continue
            seen_titles.add(norm)
            merged.append(paper)

        return merged

    def _browse_all(
        self,
        category: Optional[str] = None,
        domain_categories: Optional[List[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Browse papers when no semantic query string is given."""
        try:
            where_conditions = []
            if category and category != "All Categories":
                where_conditions.append({"primary_category": {"$eq": category}})
            elif domain_categories:
                where_conditions.append({"primary_category": {"$in": domain_categories}})

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

            # If strict year + category filter had 0 local hits, fallback to category/domain without year restriction
            if not res or not res.get("ids"):
                cat_clause = (
                    {"primary_category": {"$eq": category}}
                    if (category and category != "All Categories")
                    else ({"primary_category": {"$in": domain_categories}} if domain_categories else None)
                )
                res = self.chroma.collection.get(
                    where=cat_clause,
                    limit=limit,
                    include=["documents", "metadatas"]
                )

            browse_results = []
            if res and res["ids"]:
                for doc_id, meta, doc in zip(res["ids"], res["metadatas"], res["documents"]):
                    abstract = doc.split("Abstract: ")[-1] if "Abstract: " in doc else doc
                    doi = meta.get("doi", "")
                    journal_ref = meta.get("journal_ref", "")
                    publisher = journal_ref if journal_ref else ("Nature / Science / IEEE" if doi and not doi.startswith("10.48550") else "arXiv.org")
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
                        "doi": doi,
                        "publisher": publisher,
                        "venue": journal_ref or meta.get("primary_category_name", "arXiv"),
                        "url": f"https://doi.org/{doi}" if (doi and not doi.startswith("10.48550")) else meta.get("arxiv_url", f"https://arxiv.org/abs/{doc_id}"),
                        "arxiv_url": meta.get("arxiv_url", f"https://arxiv.org/abs/{doc_id}"),
                        "pdf_url": meta.get("pdf_url", f"https://arxiv.org/pdf/{doc_id}.pdf"),
                        "retrieval_similarity": 0.8800,
                        "distance": 0.1200
                    })
            return browse_results
        except Exception:
            return []

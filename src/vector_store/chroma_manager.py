"""
ChromaDB Vector Store Manager for ArXiv Scientific Research Papers.
Provides persistent storage, incremental idempotent indexing, and semantic similarity search.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.embeddings.embedder import EmbeddingGenerator

class ChromaManager:
    """Manages persistent ChromaDB vector storage and semantic querying."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        self.persist_dir = persist_dir or CHROMA_DB_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy-loads persistent ChromaDB client."""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self):
        """Gets or creates the Chroma collection configured for cosine distance."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def count(self) -> int:
        """Returns the total number of documents currently indexed."""
        return self.collection.count()

    def get_existing_ids(self) -> set:
        """Returns a set of all paper IDs already in the vector database."""
        total = self.count()
        if total == 0:
            return set()
        # Retrieve all IDs
        res = self.collection.get(include=[])
        return set(res.get("ids", []))

    def index_papers(
        self,
        papers: List[Dict[str, Any]],
        embedder: EmbeddingGenerator,
        batch_size: int = 64
    ) -> int:
        """
        Indexes papers incrementally. Skips papers that already exist in the collection.
        Args:
            papers: List of processed paper dictionaries.
            embedder: EmbeddingGenerator instance.
            batch_size: Batch size for embedding generation and DB insertion.
        Returns:
            Number of newly indexed papers.
        """
        if not papers:
            return 0

        existing_ids = self.get_existing_ids()
        new_papers = [p for p in papers if str(p["id"]) not in existing_ids]

        if not new_papers:
            print(f"[CHROMA] All {len(papers)} papers are already indexed. No new papers to add.")
            return 0

        print(f"[CHROMA] Indexing {len(new_papers)} new papers (skipped {len(papers) - len(new_papers)} already present)...")

        total_added = 0
        for i in range(0, len(new_papers), batch_size):
            batch = new_papers[i : i + batch_size]
            
            ids = [str(p["id"]) for p in batch]
            documents = [p.get("retrieval_text") or f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in batch]
            
            # Prepare metadata (Chroma requires scalar strings, ints, floats, bools)
            metadatas = []
            for p in batch:
                cats = p.get("categories", [])
                cats_str = ",".join(cats) if isinstance(cats, list) else str(cats)
                labels = p.get("category_labels", [])
                labels_str = ", ".join(labels) if isinstance(labels, list) else str(labels)
                
                meta = {
                    "id": str(p["id"]),
                    "title": str(p.get("title", ""))[:500],
                    "authors": str(p.get("authors", ""))[:500],
                    "primary_category": str(p.get("primary_category", "")),
                    "primary_category_name": str(p.get("primary_category_name", "")),
                    "categories": cats_str,
                    "category_labels": labels_str,
                    "year": int(p.get("year", 2024)),
                    "published_date": str(p.get("published_date", "")),
                    "doi": str(p.get("doi", "")),
                    "arxiv_url": str(p.get("arxiv_url", "")),
                    "pdf_url": str(p.get("pdf_url", ""))
                }
                metadatas.append(meta)

            # Generate embeddings
            embeddings = embedder.embed_texts(documents, batch_size=batch_size)

            # Upsert into ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            total_added += len(batch)
            print(f"[CHROMA] Indexed {total_added}/{len(new_papers)} papers...")

        print(f"[CHROMA] Indexing complete. Total collection size: {self.count()} papers.")
        return total_added

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        category_filter: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic nearest-neighbor search using cosine distance.
        Args:
            query_embedding: Dense embedding vector of user query.
            top_k: Number of nearest neighbors to retrieve.
            category_filter: Optional arXiv category code filter.
            year_min: Optional minimum publication year filter.
            year_max: Optional maximum publication year filter.
        Returns:
            List of structured result dictionaries with metadata and similarity scores.
        """
        if self.count() == 0:
            return []

        # Build Chroma 'where' filter if filters are provided
        where_conditions = []
        if category_filter and category_filter != "All Categories":
            where_conditions.append({"primary_category": {"$eq": category_filter}})
        if year_min is not None:
            where_conditions.append({"year": {"$gte": year_min}})
        if year_max is not None:
            where_conditions.append({"year": {"$lte": year_max}})

        where_clause = None
        if len(where_conditions) == 1:
            where_clause = where_conditions[0]
        elif len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results: List[Dict[str, Any]] = []
        
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)
        documents = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)

        for doc_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            # Cosine distance in Chroma: dist in [0, 2], where 0 is identical
            # Standard cosine similarity: 1.0 - dist
            cosine_similarity = max(0.0, min(1.0, 1.0 - dist))
            
            formatted_results.append({
                "id": doc_id,
                "title": meta.get("title", "Untitled"),
                "authors": meta.get("authors", "Unknown"),
                "abstract": doc.split("Abstract: ")[-1] if "Abstract: " in doc else doc,
                "primary_category": meta.get("primary_category", ""),
                "primary_category_name": meta.get("primary_category_name", ""),
                "categories": meta.get("categories", ""),
                "category_labels": meta.get("category_labels", ""),
                "year": meta.get("year", 2024),
                "published_date": meta.get("published_date", ""),
                "doi": meta.get("doi", ""),
                "arxiv_url": meta.get("arxiv_url", f"https://arxiv.org/abs/{doc_id}"),
                "pdf_url": meta.get("pdf_url", f"https://arxiv.org/pdf/{doc_id}.pdf"),
                "retrieval_similarity": round(cosine_similarity, 4),
                "distance": round(dist, 4)
            })

        return formatted_results

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single paper and its metadata by arXiv ID."""
        try:
            res = self.collection.get(
                ids=[str(paper_id)],
                include=["documents", "metadatas"]
            )
            if res and res["ids"] and len(res["ids"]) > 0:
                doc_id = res["ids"][0]
                meta = res["metadatas"][0] if res["metadatas"] else {}
                doc = res["documents"][0] if res["documents"] else ""
                abstract = doc.split("Abstract: ")[-1] if "Abstract: " in doc else doc
                return {
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "abstract": abstract,
                    "primary_category": meta.get("primary_category", ""),
                    "primary_category_name": meta.get("primary_category_name", ""),
                    "categories": meta.get("categories", ""),
                    "category_labels": meta.get("category_labels", ""),
                    "year": meta.get("year", 2024),
                    "published_date": meta.get("published_date", ""),
                    "doi": meta.get("doi", ""),
                    "arxiv_url": meta.get("arxiv_url", f"https://arxiv.org/abs/{doc_id}"),
                    "pdf_url": meta.get("pdf_url", f"https://arxiv.org/pdf/{doc_id}.pdf")
                }
        except Exception:
            pass
        return None

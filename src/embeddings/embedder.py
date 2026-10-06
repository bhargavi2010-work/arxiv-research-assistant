"""
Embedding Generator for ArXiv Research Assistant.
Wraps SentenceTransformer to generate dense L2-normalized vector representations
for scientific paper titles, abstracts, and queries.
"""

from typing import List, Union
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBED_BATCH_SIZE

class EmbeddingGenerator:
    """Manages sentence-transformer models for dense text embedding."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-loads the SentenceTransformer model on first invocation."""
        if self._model is None:
            print(f"[EMBEDDER] Loading embedding model: {self.model_name} on device: {self.device}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        show_progress_bar: bool = False
    ) -> List[List[float]]:
        """
        Generates dense normalized vector embeddings for a list of texts.
        Args:
            texts: List of string passages.
            batch_size: Processing batch size.
            show_progress_bar: Whether to display tqdm progress.
        Returns:
            List of float lists representing normalized embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single search query or concept string.
        Args:
            query: Input user query.
        Returns:
            List of floats representing the query embedding.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
            
        embedding = self.model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Returns the embedding dimension of the loaded model."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

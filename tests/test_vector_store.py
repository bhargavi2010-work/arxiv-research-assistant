"""
Unit tests for Phase 3: Sentence-Transformers Embeddings & ChromaDB Vector Store.
"""

import unittest
from pathlib import Path
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.embeddings.embedder import EmbeddingGenerator
from src.vector_store.chroma_manager import ChromaManager
from src.data_pipeline.preprocessor import ArXivPreprocessor

class TestEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = EmbeddingGenerator()

    def test_embedding_dimension(self):
        self.assertEqual(self.embedder.embedding_dimension, 384)

    def test_embed_query_unit_norm(self):
        query = "Retrieval Augmented Generation"
        vec = self.embedder.embed_query(query)
        self.assertEqual(len(vec), 384)
        
        # Verify L2 unit norm
        norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_embed_texts(self):
        texts = ["Deep learning with neural networks", "Natural language processing models"]
        vecs = self.embedder.embed_texts(texts)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 384)
        self.assertEqual(len(vecs[1]), 384)

class TestChromaManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = EmbeddingGenerator()
        cls.preprocessor = ArXivPreprocessor()
        df = cls.preprocessor.load_processed_data()
        cls.papers = df.to_dict(orient="records")
        cls.chroma = ChromaManager()

    def test_indexing_and_count(self):
        self.chroma.index_papers(self.papers, self.embedder)
        self.assertGreaterEqual(self.chroma.count(), len(self.papers))

    def test_idempotent_indexing(self):
        count_before = self.chroma.count()
        added = self.chroma.index_papers(self.papers, self.embedder)
        self.assertEqual(added, 0)
        self.assertEqual(self.chroma.count(), count_before)

    def test_semantic_retrieval(self):
        query = "Attention mechanisms and Transformer networks for machine translation"
        q_vec = self.embedder.embed_query(query)
        results = self.chroma.query_similar(q_vec, top_k=3)
        
        self.assertGreater(len(results), 0)
        top_res = results[0]
        self.assertIn("retrieval_similarity", top_res)
        self.assertIn("title", top_res)
        self.assertGreater(top_res["retrieval_similarity"], 0.4)
        # Verify the top match is Transformer or related NLP paper
        self.assertTrue("attention" in top_res["title"].lower() or "transformer" in top_res["title"].lower() or "nlp" in top_res["title"].lower())

    def test_get_paper_by_id(self):
        paper = self.chroma.get_paper_by_id("1706.03762")
        if paper:
            self.assertEqual(paper["id"], "1706.03762")
            self.assertIn("Attention Is All You Need", paper["title"])

if __name__ == "__main__":
    unittest.main()

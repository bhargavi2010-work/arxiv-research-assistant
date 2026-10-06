"""
Unit tests for Phase 4: Semantic Paper Retrieval and Search Engine.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.search_engine import PaperSearchEngine
from src.retrieval.context_builder import RAGContextBuilder

class TestSearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search_engine = PaperSearchEngine()

    def test_semantic_search_rag(self):
        res = self.search_engine.search(query="Retrieval-Augmented Generation for language models", limit=5)
        self.assertIn("results", res)
        self.assertGreater(res["total_matches"], 0)
        
        # Check top result
        top_paper = res["results"][0]
        self.assertIn("title", top_paper)
        self.assertIn("retrieval_similarity", top_paper)
        self.assertGreater(top_paper["retrieval_similarity"], 0.3)

    def test_category_filtering(self):
        res = self.search_engine.search(query="deep neural networks", category="cs.CV", limit=5)
        for p in res["results"]:
            self.assertEqual(p["primary_category"], "cs.CV")

    def test_year_range_filtering(self):
        res = self.search_engine.search(query="language models", year_min=2018, year_max=2022, limit=5)
        for p in res["results"]:
            self.assertTrue(2018 <= p["year"] <= 2022)

    def test_sorting_by_year_newest(self):
        res = self.search_engine.search(query="attention", sort_by="Year (Newest)", limit=5)
        years = [p["year"] for p in res["results"]]
        self.assertEqual(years, sorted(years, reverse=True))

    def test_sorting_by_title_az(self):
        res = self.search_engine.search(query="model", sort_by="Title (A-Z)", limit=5)
        titles = [p["title"].lower() for p in res["results"]]
        self.assertEqual(titles, sorted(titles))

class TestContextBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search_engine = PaperSearchEngine()
        cls.builder = RAGContextBuilder(min_similarity_threshold=0.30)

    def test_context_construction(self):
        search_res = self.search_engine.search(query="Attention Is All You Need Transformer", limit=3)
        has_ctx, context_str, sources = self.builder.build_context(search_res["results"], max_papers=3)
        
        self.assertTrue(has_ctx)
        self.assertIn("[1] Title:", context_str)
        self.assertGreater(len(sources), 0)
        self.assertIn("title", sources[0])

    def test_insufficient_context_fallback(self):
        # Strict threshold builder
        strict_builder = RAGContextBuilder(min_similarity_threshold=0.99)
        search_res = self.search_engine.search(query="something unrelated", limit=3)
        has_ctx, context_str, sources = strict_builder.build_context(search_res["results"])
        
        self.assertFalse(has_ctx)
        self.assertIn("Not enough relevant research context", context_str)
        self.assertEqual(len(sources), 0)

if __name__ == "__main__":
    unittest.main()

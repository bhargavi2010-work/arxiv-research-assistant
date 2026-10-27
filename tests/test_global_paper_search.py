"""
Unit tests for Global Literature Search Engine (IEEE, Springer, arXiv, Crossref).
"""

import unittest
from src.retrieval.global_paper_search import GlobalLiteratureSearchEngine

class TestGlobalLiteratureSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GlobalLiteratureSearchEngine(timeout_sec=5)

    def test_search_empty_query(self):
        res = self.engine.search_global_literature("")
        self.assertEqual(res, [])

    def test_crossref_parsing_structure(self):
        # Test offline structure parsing
        sample_crossref_items = self.engine._fetch_crossref("nonexistent_random_token_xyz_999", limit=2)
        # Should return a list (empty or items) without throwing exceptions
        self.assertIsInstance(sample_crossref_items, list)

if __name__ == "__main__":
    unittest.main()



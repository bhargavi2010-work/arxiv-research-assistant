"""
Unit tests for Global Literature Search Engine & Multi-Domain Filter Browsing.
"""

import unittest
from src.retrieval.global_paper_search import GlobalLiteratureSearchEngine
from src.retrieval.search_engine import PaperSearchEngine
from config.config import RESEARCH_DOMAINS

class TestGlobalLiteratureSearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GlobalLiteratureSearchEngine(timeout_sec=5)
        cls.paper_search = PaperSearchEngine()

    def test_search_empty_query(self):
        res = self.engine.search_global_literature("")
        self.assertEqual(res, [])

    def test_crossref_parsing_structure(self):
        sample_crossref_items = self.engine._fetch_crossref("nonexistent_random_token_xyz_999", limit=2)
        self.assertIsInstance(sample_crossref_items, list)

    def test_qbio_bm_filter_browsing_returns_results(self):
        # Exact scenario from user screenshot: Domain=Medicine, Category=q-bio.BM, Year=2020-2025, Query=""
        res = self.paper_search.search(
            query="",
            category="q-bio.BM",
            year_min=2020,
            year_max=2025,
            include_global=False
        )
        self.assertGreater(len(res["results"]), 0, "q-bio.BM filter must return papers")
        for p in res["results"]:
            self.assertEqual(p["primary_category"], "q-bio.BM")

    def test_all_research_domains_and_categories_have_papers(self):
        # Verify EVERY category in RESEARCH_DOMAINS returns > 0 papers when filtered
        for domain_name, cat_dict in RESEARCH_DOMAINS.items():
            if not cat_dict:
                continue
            # Test domain-level filter with All Categories
            domain_cats = list(cat_dict.keys())
            dom_res = self.paper_search.search(
                query="",
                category=None,
                domain_categories=domain_cats,
                year_min=2020,
                year_max=2026,
                include_global=False
            )
            self.assertGreater(
                len(dom_res["results"]), 0,
                f"Domain '{domain_name}' returned 0 results"
            )
            # Test each individual category filter
            for cat_code in cat_dict.keys():
                cat_res = self.paper_search.search(
                    query="",
                    category=cat_code,
                    year_min=2020,
                    year_max=2026,
                    include_global=False
                )
                self.assertGreater(
                    len(cat_res["results"]), 0,
                    f"Category '{cat_code}' in '{domain_name}' returned 0 results"
                )

if __name__ == "__main__":
    unittest.main()

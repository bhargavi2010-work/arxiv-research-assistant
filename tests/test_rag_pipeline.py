"""
Unit tests for Phase 6: End-to-End RAG Research Chatbot Engine.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.rag_pipeline import ResearchChatbotEngine

class TestRAGChatbotEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ResearchChatbotEngine()

    def test_answer_query_grounded_rag(self):
        query = "What is Retrieval-Augmented Generation and why is it useful for large language models?"
        response = self.engine.answer_query(query=query, top_k=3)
        
        self.assertTrue(response["has_sufficient_context"])
        self.assertIn("answer", response)
        self.assertGreater(len(response["answer"]), 50)
        self.assertIn("[1]", response["answer"])
        
        # Verify sources
        sources = response.get("sources", [])
        self.assertGreater(len(sources), 0)
        first_src = sources[0]
        self.assertIn("title", first_src)
        self.assertIn("arxiv_url", first_src)
        self.assertIn("retrieval_similarity", first_src)
        
        # Verify follow-ups
        follow_ups = response.get("suggested_follow_ups", [])
        self.assertGreater(len(follow_ups), 0)

    def test_answer_empty_query(self):
        response = self.engine.answer_query(query="   ")
        self.assertFalse(response["has_sufficient_context"])
        self.assertIn("valid research question", response["answer"])

    def test_suggested_follow_ups_generation(self):
        suggestions = self.engine.generate_suggested_follow_ups(
            query="Transformer Attention",
            sources=[{"title": "Attention Is All You Need"}]
        )
        self.assertIsInstance(suggestions, list)
        self.assertTrue(1 <= len(suggestions) <= 3)

if __name__ == "__main__":
    unittest.main()

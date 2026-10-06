"""
Unit tests for Phase 5: Open-Source LLM Integration and Grounded Generation.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.llm.prompts import (
    build_research_qa_prompt,
    build_concept_explanation_prompt,
    RESEARCH_QA_SYSTEM_PROMPT
)
from src.llm.llm_client import LLMClient
from src.retrieval.search_engine import PaperSearchEngine
from src.retrieval.context_builder import RAGContextBuilder

class TestLLMPrompts(unittest.TestCase):
    def test_build_research_qa_prompt(self):
        context = "[1] Title: Attention Is All You Need\n    Summary/Abstract: Transformer model based solely on attention."
        query = "What is the Transformer?"
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
        
        prompt = build_research_qa_prompt(query, context, conversation_history=history)
        self.assertIn("CRITICAL INSTRUCTIONS", prompt)
        self.assertIn("[Retrieved Research Papers]", prompt)
        self.assertIn("Attention Is All You Need", prompt)
        self.assertIn("User: Hello", prompt)
        self.assertIn("User Question: What is the Transformer?", prompt)

    def test_build_concept_explanation_prompt(self):
        context = "[1] Title: Attention Is All You Need\n    Summary/Abstract: Attention mechanism."
        prompt = build_concept_explanation_prompt("Self-Attention", context)
        self.assertIn("Concept to Explain: Self-Attention", prompt)
        self.assertIn("1. Definition:", prompt)
        self.assertIn("2. How It Works:", prompt)

class TestLLMClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = LLMClient(backend="auto")
        cls.search_engine = PaperSearchEngine()
        cls.builder = RAGContextBuilder(min_similarity_threshold=0.30)

    def test_grounded_answer_generation(self):
        query = "What is Retrieval-Augmented Generation (RAG) and why is it useful?"
        search_res = self.search_engine.search(query=query, limit=3)
        has_ctx, context_str, sources = self.builder.build_context(search_res["results"], max_papers=3)
        
        self.assertTrue(has_ctx)
        answer = self.llm.generate_grounded_answer(query=query, context=context_str)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 50)
        # Verify citations are present
        self.assertIn("[1]", answer)

    def test_insufficient_context_rejection(self):
        insufficient_context = "Not enough relevant research context was found to answer this question reliably."
        answer = self.llm.generate_grounded_answer(query="Quantum gravity string equations", context=insufficient_context)
        self.assertEqual(answer, insufficient_context)

    def test_concept_explanation(self):
        concept = "Transformer"
        search_res = self.search_engine.search(query="Transformer attention mechanism", limit=2)
        has_ctx, context_str, _ = self.builder.build_context(search_res["results"], max_papers=2)
        
        explanation = self.llm.generate_concept_explanation(concept, context_str)
        self.assertIsInstance(explanation, str)
        self.assertIn("[1]", explanation)

if __name__ == "__main__":
    unittest.main()

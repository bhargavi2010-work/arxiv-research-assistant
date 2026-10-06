"""
Unit tests for Phase 9: Scientific Information Extraction.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.nlp.information_extractor import ScientificInfoExtractor

class TestInformationExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = ScientificInfoExtractor()

    def test_extract_transformer_paper(self):
        title = "Attention Is All You Need"
        abstract = (
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show "
            "these models to be superior in quality while being more parallelizable."
        )
        res = self.extractor.extract_all(title, abstract)
        
        # Verify models extracted
        self.assertIn("Transformer", res["models"])
        # Verify methods
        self.assertTrue(any("Attention" in m for m in res["methods"]))
        # Verify tasks
        self.assertIn("Machine Translation", res["tasks"])
        # Verify concepts
        self.assertGreater(len(res["concepts"]), 0)

    def test_extract_rag_paper(self):
        title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
        abstract = (
            "We explore retrieval-augmented generation (RAG) models which combine pre-trained parametric "
            "and non-parametric memory for language generation. The non-parametric memory is a dense vector "
            "index of Wikipedia accessed with a pre-trained neural retriever."
        )
        res = self.extractor.extract_all(title, abstract)
        
        self.assertTrue("Retrieval-Augmented Generation" in res["methods"] or "RAG" in res["methods"])
        self.assertTrue("Dense Retrieval" in res["methods"] or "Vector Search" in res["methods"] or "Pre-training" in res["methods"])
        self.assertIn("Wikipedia", res["datasets"])

    def test_extract_keywords(self):
        text = "Deep residual learning for image recognition using convolutional neural networks and residual connections."
        keywords = self.extractor.extract_keywords(text)
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

if __name__ == "__main__":
    unittest.main()

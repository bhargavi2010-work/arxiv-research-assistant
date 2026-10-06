"""
Unit tests for Phase 8: Paper Summarization Engine.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.summarizer.paper_summarizer import PaperSummarizer

class TestPaperSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summarizer = PaperSummarizer()

    def test_structured_summary_generation(self):
        paper = {
            "id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": "Ashish Vaswani, Noam Shazeer, et al.",
            "year": 2017,
            "primary_category_name": "Computation and Language",
            "abstract": (
                "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
                "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
                "dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show "
                "these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
            )
        }

        summary = self.summarizer.summarize(paper)
        
        self.assertEqual(summary["title"], "Attention Is All You Need")
        self.assertTrue(summary["is_abstract_based"])
        self.assertIn("disclaimer", summary)
        
        # Verify 4 key sections
        self.assertIn("problem", summary)
        self.assertIn("approach", summary)
        self.assertIn("key_contributions", summary)
        self.assertIn("results", summary)
        
        self.assertGreater(len(summary["key_contributions"]), 0)
        self.assertIn("Transformer", summary["extracted_models"])

if __name__ == "__main__":
    unittest.main()

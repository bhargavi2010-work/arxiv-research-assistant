"""
Unit tests for Phase 10: Concept Visualization Engine.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.concept_graph.concept_visualizer import ConceptVisualizer

class TestConceptVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visualizer = ConceptVisualizer()

    def test_get_concept_data_curated(self):
        data = self.visualizer.get_concept_data("Transformer")
        self.assertEqual(data["name"], "Transformer")
        self.assertIn("Attention", data["primary_components"])
        self.assertIn("Encoder", data["primary_components"])
        self.assertIn("Decoder", data["primary_components"])
        self.assertGreater(len(data["related_concepts"]), 0)

    def test_build_networkx_graph(self):
        data = self.visualizer.get_concept_data("RAG")
        G = self.visualizer.build_networkx_graph(data)
        
        self.assertIn("RAG", G.nodes)
        self.assertGreater(len(G.nodes), 4)
        self.assertGreater(len(G.edges), 3)

    def test_create_plotly_figure(self):
        data = self.visualizer.get_concept_data("CNN")
        fig = self.visualizer.create_plotly_figure(data)
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 1)

if __name__ == "__main__":
    unittest.main()

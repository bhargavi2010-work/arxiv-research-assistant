"""
Unit tests for LangChain AI Agent & LangSmith Observability Integration.
Verifies syllabus concepts:
- Unit 1: ReAct reasoning loop, Tool selection
- Unit 2: LangChain Tools, Observability status
"""

import unittest
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Set offline flags for fast local test execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.agents.langchain_agent import ArXivLangChainAgent

class TestArXivLangChainAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ArXivLangChainAgent()

    def test_tools_registration(self):
        """Verifies that all 5 LangChain scientific tools are registered."""
        tool_names = [t.name for t in self.agent.tools]
        self.assertEqual(len(self.agent.tools), 5)
        self.assertIn("search_arxiv_literature", tool_names)
        self.assertIn("fetch_latest_arxiv_preprints", tool_names)
        self.assertIn("generate_paper_summary", tool_names)
        self.assertIn("explore_concept_graph", tool_names)
        self.assertIn("parse_and_index_research_pdf", tool_names)

    def test_langsmith_status_metadata(self):
        """Verifies LangSmith observability metadata generation (Unit 2)."""
        status = self.agent.get_langsmith_status()
        self.assertIn("project_name", status)
        self.assertIn("endpoint", status)
        self.assertIn("status", status)
        self.assertEqual(status["project_name"], "arxiv-research-assistant")

    def test_react_agent_turn_execution(self):
        """Verifies the ReAct agent turn with reasoning metadata (Unit 1 & 2)."""
        res = self.agent.run_agent_turn("What is a Transformer?")
        self.assertIn("answer", res)
        self.assertGreater(len(res["answer"]), 50)
        self.assertIn("agent_metadata", res)
        self.assertEqual(res["agent_metadata"]["agent_framework"], "LangChain ReAct Agent (Unit 1 & 2)")
        self.assertIn("thought", res["agent_metadata"])
        self.assertEqual(res["tool_used"], "search_arxiv_literature")

if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Knowledge Base Expansion Pipeline.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline.expansion import get_last_sync_info, record_sync_event

class TestKnowledgeBaseExpansion(unittest.TestCase):
    def test_sync_logging(self):
        record_sync_event(added_count=5, categories=["cs.AI", "cs.LG"], status="Test Success")
        info = get_last_sync_info()
        self.assertEqual(info["added_count"], 5)
        self.assertEqual(info["status"], "Test Success")
        self.assertIn("cs.AI", info["categories"])

if __name__ == "__main__":
    unittest.main()

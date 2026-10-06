"""
Unit tests for Phase 7: Multi-turn Conversation Context and Memory.
"""

import unittest
from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.query_contextualizer import QueryContextualizer
from src.storage.chat_memory import ConversationManager

class TestQueryContextualizer(unittest.TestCase):
    def setUp(self):
        self.contextualizer = QueryContextualizer()

    def test_standalone_query(self):
        q = "What is Retrieval-Augmented Generation?"
        res = self.contextualizer.contextualize_query(q, None)
        self.assertEqual(res, q)

    def test_pronoun_resolution(self):
        history = [
            {"role": "user", "content": "What is a Transformer?"},
            {"role": "assistant", "content": "The Transformer is a deep learning architecture based on self-attention."}
        ]
        follow_up = "Why is self-attention important in it?"
        contextualized = self.contextualizer.contextualize_query(follow_up, history)
        self.assertIn("Transformer", contextualized)
        self.assertIn("self-attention", contextualized)

    def test_comparison_resolution(self):
        history = [
            {"role": "user", "content": "Explain Transformers"},
            {"role": "assistant", "content": "Transformers dispense with recurrence and convolutions."}
        ]
        follow_up = "How is this different from an RNN?"
        contextualized = self.contextualizer.contextualize_query(follow_up, history)
        self.assertIn("Transformers", contextualized)
        self.assertIn("RNN", contextualized)

class TestConversationManager(unittest.TestCase):
    def setUp(self):
        self.temp_file = Path(tempfile.gettempdir()) / "test_chat_history.json"
        if self.temp_file.exists():
            self.temp_file.unlink()
        self.manager = ConversationManager(storage_file=self.temp_file)

    def tearDown(self):
        if self.temp_file.exists():
            self.temp_file.unlink()

    def test_conversation_lifecycle(self):
        conv_id = self.manager.create_conversation("Initial Title")
        self.assertIsNotNone(conv_id)

        # Add user query (auto-titling test)
        self.manager.add_user_message(conv_id, "What are large language models?")
        
        # Add assistant reply
        self.manager.add_assistant_message(
            conv_id,
            "LLMs are neural networks trained on broad text corpora.",
            sources=[{"title": "GPT-3"}]
        )

        conv = self.manager.get_conversation(conv_id)
        self.assertIsNotNone(conv)
        self.assertEqual(len(conv["messages"]), 2)
        self.assertIn("Large Language Models", conv["title"])

        # Test sliding window
        window = self.manager.get_sliding_window_messages(conv_id, max_turns=1)
        self.assertEqual(len(window), 2)
        self.assertEqual(window[0]["role"], "user")
        self.assertEqual(window[1]["role"], "assistant")

    def test_delete_and_list(self):
        conv_id = self.manager.create_conversation("To be deleted")
        convs = self.manager.list_conversations()
        self.assertTrue(any(c["id"] == conv_id for c in convs))

        deleted = self.manager.delete_conversation(conv_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.get_conversation(conv_id))

if __name__ == "__main__":
    unittest.main()

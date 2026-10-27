"""
Chat Memory & Conversation Storage Manager for ArXiv Research Assistant.
Manages persistent multi-turn conversations, automatic conversation titling,
context sliding-window truncation, and local JSON storage.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import CHAT_HISTORY_FILE, STORAGE_DIR

class ConversationManager:
    """Handles multi-turn conversation memory, history persistence, and sliding-window context."""

    def __init__(self, storage_file: Optional[Path] = None):
        self.storage_file = storage_file or CHAT_HISTORY_FILE
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._conversations: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Loads persistent conversations from JSON file."""
        if not self.storage_file.exists():
            return {}
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self):
        """Persists all conversations to storage file."""
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self._conversations, f, indent=2, ensure_ascii=False)

    def create_conversation(self, initial_title: str = "New Research Chat") -> str:
        """Creates a new conversation session and returns its unique ID."""
        conv_id = str(uuid.uuid4())[:8]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._conversations[conv_id] = {
            "id": conv_id,
            "title": initial_title,
            "created_at": now_str,
            "updated_at": now_str,
            "messages": []
        }
        self._save()
        return conv_id

    def add_user_message(self, conv_id: str, content: str) -> None:
        """Appends a user query to the specified conversation."""
        if conv_id not in self._conversations:
            self.create_conversation()
            
        conv = self._conversations[conv_id]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Auto-update conversation title from the first question if title is default
        if conv["title"] == "New Research Chat" or len(conv["messages"]) == 0:
            import re
            cleaned_topic = re.sub(r"^(what is|what are|explain|how does|why is|tell me about)\s+", "", content.strip(), flags=re.IGNORECASE).strip(" ?.")
            words = (cleaned_topic or content.strip()).split()
            title_candidate = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
            conv["title"] = title_candidate.title()

        conv["messages"].append({
            "role": "user",
            "content": content.strip(),
            "timestamp": now_str
        })
        conv["updated_at"] = now_str
        self._save()

    def add_assistant_message(
        self,
        conv_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        suggested_follow_ups: Optional[List[str]] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
        safety_metric: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends assistant grounded answer, research sources, agent metadata, and safety guardrail metrics."""
        if conv_id not in self._conversations:
            return

        conv = self._conversations[conv_id]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conv["messages"].append({
            "role": "assistant",
            "content": content.strip(),
            "sources": sources or [],
            "suggested_follow_ups": suggested_follow_ups or [],
            "agent_metadata": agent_metadata,
            "safety_metric": safety_metric,
            "timestamp": now_str
        })
        conv["updated_at"] = now_str
        self._save()

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific conversation dictionary."""
        return self._conversations.get(conv_id)

    def get_sliding_window_messages(
        self,
        conv_id: str,
        max_turns: int = 4
    ) -> List[Dict[str, str]]:
        """
        Retrieves recent turns formatted as simple {'role': ..., 'content': ...}
        for inclusion in prompt context without exceeding token budgets.
        """
        conv = self.get_conversation(conv_id)
        if not conv or not conv.get("messages"):
            return []

        all_msgs = conv["messages"]
        # Take up to max_turns * 2 individual messages (user + assistant)
        recent_slice = all_msgs[-(max_turns * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in recent_slice]

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Returns all conversations ordered from most recently active to oldest."""
        conv_list = list(self._conversations.values())
        conv_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conv_list

    def delete_conversation(self, conv_id: str) -> bool:
        """Deletes a conversation by ID."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            self._save()
            return True
        return False

    def clear_all(self):
        """Clears all conversations."""
        self._conversations = {}
        self._save()

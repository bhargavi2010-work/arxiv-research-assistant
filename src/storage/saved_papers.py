"""
Saved Papers Storage Manager for ArXiv Research Assistant.
Provides local persistent bookmarking of research papers with metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import SAVED_PAPERS_FILE

class SavedPapersManager:
    """Manages bookmarking and persistent local storage of research papers."""

    def __init__(self, storage_file: Optional[Path] = None):
        self.storage_file = storage_file or SAVED_PAPERS_FILE
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._saved_papers: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.storage_file.exists():
            return {}
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self):
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self._saved_papers, f, indent=2, ensure_ascii=False)

    def save_paper(self, paper: Dict[str, Any]) -> bool:
        """Saves a paper to bookmarked collection."""
        paper_id = str(paper.get("id", "")).strip()
        if not paper_id:
            return False

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._saved_papers[paper_id] = {
            "id": paper_id,
            "title": paper.get("title", "Untitled"),
            "authors": paper.get("authors", "Unknown"),
            "year": paper.get("year", 2024),
            "primary_category": paper.get("primary_category", "cs.AI"),
            "primary_category_name": paper.get("primary_category_name", ""),
            "abstract": paper.get("abstract", "")[:400] + ("..." if len(paper.get("abstract", "")) > 400 else ""),
            "arxiv_url": paper.get("arxiv_url", f"https://arxiv.org/abs/{paper_id}"),
            "pdf_url": paper.get("pdf_url", f"https://arxiv.org/pdf/{paper_id}.pdf"),
            "saved_date": now_str
        }
        self._save()
        return True

    def remove_paper(self, paper_id: str) -> bool:
        """Removes a paper from bookmarks."""
        pid = str(paper_id).strip()
        if pid in self._saved_papers:
            del self._saved_papers[pid]
            self._save()
            return True
        return False

    def is_saved(self, paper_id: str) -> bool:
        """Checks if a paper is bookmarked."""
        return str(paper_id).strip() in self._saved_papers

    def list_saved(self) -> List[Dict[str, Any]]:
        """Returns all saved papers, sorted by most recently saved."""
        papers = list(self._saved_papers.values())
        papers.sort(key=lambda x: x.get("saved_date", ""), reverse=True)
        return papers

"""
Dynamic & Periodic Knowledge Base Expansion Pipeline for ArXiv Research Assistant.
Connects directly to the live arXiv API (OAI-PMH / Atom), extracts metadata,
generates dense embeddings, and incrementally indexes documents into ChromaDB.
"""

import urllib.request
import xml.etree.ElementTree as ET
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import CS_AI_CATEGORIES, DATA_DIR, RAW_DATA_DIR, RAW_DATASET_FILENAME
from src.vector_store.chroma_manager import ChromaManager
from src.embeddings.embedder import EmbeddingGenerator

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
SYNC_LOG_FILE = DATA_DIR / "expansion_sync_log.json"

def get_last_sync_info() -> Dict[str, Any]:
    """Retrieves the timestamp and details of the most recent knowledge base expansion."""
    if SYNC_LOG_FILE.exists():
        try:
            with open(SYNC_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if logs and isinstance(logs, list):
                    return logs[-1]
        except Exception:
            pass
    return {
        "timestamp": "Never",
        "added_count": 0,
        "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
        "status": "Initialized"
    }

def record_sync_event(added_count: int, categories: List[str], status: str = "Success"):
    """Records a knowledge base expansion sync event."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    if SYNC_LOG_FILE.exists():
        try:
            with open(SYNC_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "added_count": added_count,
        "categories": categories,
        "status": status
    }
    logs.append(event)
    try:
        with open(SYNC_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs[-50:], f, indent=2)
    except Exception as e:
        print(f"[EXPANSION WARNING] Failed to record sync event: {e}")

def fetch_and_index_arxiv_papers(
    categories: Optional[List[str]] = None,
    max_results: int = 10,
    chroma_manager: Optional[ChromaManager] = None,
    embedder: Optional[EmbeddingGenerator] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Dynamically fetches live arXiv papers, computes vector embeddings, and indexes them in ChromaDB.
    Args:
        categories: List of arXiv domain categories (e.g. ['cs.AI', 'cs.LG']).
        max_results: Maximum number of papers to fetch and index.
        chroma_manager: ChromaManager instance.
        embedder: EmbeddingGenerator instance.
        progress_callback: Optional progress reporter (message, fraction).
    Returns:
        Summary dict containing status, added_count, and total_count.
    """
    if categories is None:
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

    if chroma_manager is None:
        chroma_manager = ChromaManager()
    if embedder is None:
        embedder = EmbeddingGenerator()

    if progress_callback:
        progress_callback("Connecting to official arXiv API...", 0.1)

    cat_query = "+OR+".join([f"cat:{c}" for c in categories])
    api_url = (
        f"http://export.arxiv.org/api/query?search_query={cat_query}"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "ArXivResearchAssistant/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            xml_data = resp.read()
    except Exception as e:
        record_sync_event(0, categories, status=f"Failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Could not reach arXiv API: {str(e)}",
            "added_count": 0,
            "total_count": chroma_manager.count()
        }

    if progress_callback:
        progress_callback("Parsing paper metadata...", 0.3)

    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", ATOM_NS)
    
    parsed_papers = []
    for entry in entries:
        id_elem = entry.find("atom:id", ATOM_NS)
        if id_elem is None:
            continue
        raw_id = id_elem.text.split("/abs/")[-1]
        
        # Check if already in ChromaDB
        existing = chroma_manager.get_paper_by_id(raw_id)
        if existing:
            continue

        title = entry.find("atom:title", ATOM_NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ATOM_NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ATOM_NS).text.strip()
        year = int(published[:4]) if len(published) >= 4 else 2024

        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            name_elem = a.find("atom:name", ATOM_NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        categories_list = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term")
            if term:
                categories_list.append(term)
                
        primary_cat = categories_list[0] if categories_list else "cs.AI"
        primary_cat_name = CS_AI_CATEGORIES.get(primary_cat, "Computer Science")

        doi_elem = entry.find("arxiv:doi", ATOM_NS)
        doi = doi_elem.text.strip() if doi_elem is not None else f"10.48550/arXiv.{raw_id}"

        parsed_papers.append({
            "id": raw_id,
            "title": title,
            "abstract": summary,
            "clean_abstract": summary,
            "authors": ", ".join(authors) if authors else "Unknown Authors",
            "year": year,
            "primary_category": primary_cat,
            "primary_category_name": primary_cat_name,
            "categories": " ".join(categories_list),
            "doi": doi,
            "arxiv_url": f"https://arxiv.org/abs/{raw_id}",
            "pdf_url": f"https://arxiv.org/pdf/{raw_id}.pdf"
        })

    if not parsed_papers:
        record_sync_event(0, categories, status="Up to date (no new papers)")
        return {
            "status": "success",
            "message": "All fetched papers are already indexed in ChromaDB.",
            "added_count": 0,
            "total_count": chroma_manager.count()
        }

    if progress_callback:
        progress_callback(f"Generating dense embeddings for {len(parsed_papers)} new papers...", 0.6)

    # Index in ChromaDB
    chroma_manager.index_papers(parsed_papers, embedder=embedder)

    record_sync_event(len(parsed_papers), categories, status="Success")

    if progress_callback:
        progress_callback("Knowledge base expansion complete!", 1.0)

    return {
        "status": "success",
        "message": f"Successfully indexed {len(parsed_papers)} new papers from arXiv.",
        "added_count": len(parsed_papers),
        "total_count": chroma_manager.count()
    }

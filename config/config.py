"""
Central configuration for ArXiv Research Assistant.
Defines paths, domain filters, embedding settings, LLM defaults, and retrieval parameters.
"""

from pathlib import Path
import os
from typing import Dict, List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chromadb_store"
STORAGE_DIR = PROJECT_ROOT / "storage_data"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Domain & Category Taxonomy for Computer Science / AI / ML
CS_AI_CATEGORIES: Dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "stat.ML": "Machine Learning (Statistics)"
}

TARGET_CATEGORY_CODES: List[str] = list(CS_AI_CATEGORIES.keys())

# Dataset Pipeline Defaults
RAW_DATASET_FILENAME = "arxiv-metadata-oai-snapshot.json"
PROCESSED_PARQUET_FILENAME = "arxiv_cs_ai_cleaned.parquet"
PROCESSED_JSON_FILENAME = "arxiv_cs_ai_sample.json"

DEFAULT_DEV_SAMPLE_SIZE = 5000  # Practical, fast-loading subset for development & testing
DEFAULT_EMBED_BATCH_SIZE = 64

# Vector DB Settings
CHROMA_COLLECTION_NAME = "arxiv_cs_ai_papers"

# Embedding Model Configuration
# all-MiniLM-L6-v2 is an industry standard: 384 dimensions, ~80MB, extremely fast on CPU
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# LLM Configuration
# Default lightweight open-source instruction model suitable for CPU execution
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
FALLBACK_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "512"))

# Retrieval Settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.22"))

# Storage Files
SAVED_PAPERS_FILE = STORAGE_DIR / "saved_papers.json"
CHAT_HISTORY_FILE = STORAGE_DIR / "chat_history.json"

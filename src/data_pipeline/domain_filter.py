"""
Domain Filtering Utilities for Computer Science / AI / ML ArXiv Papers.
Filters and maps arXiv taxonomy codes to human-readable domain concepts.
"""

from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import CS_AI_CATEGORIES, TARGET_CATEGORY_CODES

def parse_categories(categories_raw: Optional[str]) -> List[str]:
    """Splits space-separated arXiv categories into a clean list of codes."""
    if not categories_raw:
        return []
    return [c.strip() for c in categories_raw.split() if c.strip()]

def is_cs_ai_paper(categories_raw: Optional[str]) -> bool:
    """
    Determines if a paper belongs to the target Computer Science / AI / ML domain.
    Returns True if at least one category matches the target taxonomy.
    """
    codes = parse_categories(categories_raw)
    for code in codes:
        # Check direct match or subcategory match (e.g. cs.AI, cs.LG, stat.ML)
        for target in TARGET_CATEGORY_CODES:
            if code == target or code.startswith(target):
                return True
    return False

def get_primary_category(categories_raw: Optional[str]) -> Tuple[str, str]:
    """
    Identifies the primary category code and human-readable label.
    Prioritizes target CS/AI categories first, falling back to the first available category.
    Returns:
        (category_code, human_readable_name)
        e.g., ("cs.LG", "Machine Learning")
    """
    codes = parse_categories(categories_raw)
    if not codes:
        return "cs.AI", "Artificial Intelligence"
        
    # Check for target match
    for code in codes:
        if code in CS_AI_CATEGORIES:
            return code, CS_AI_CATEGORIES[code]
            
    # Check if starts with any target
    for code in codes:
        for target, label in CS_AI_CATEGORIES.items():
            if code.startswith(target):
                return code, label
                
    # Fallback to the first category
    first_code = codes[0]
    return first_code, CS_AI_CATEGORIES.get(first_code, first_code)

def get_all_category_labels(categories_raw: Optional[str]) -> List[str]:
    """Returns a list of human-readable category names matching the paper's categories."""
    codes = parse_categories(categories_raw)
    labels = []
    for code in codes:
        if code in CS_AI_CATEGORIES:
            labels.append(CS_AI_CATEGORIES[code])
        else:
            labels.append(code)
    return labels

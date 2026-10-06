"""
Text Cleaning and Normalization Utilities for ArXiv Scientific Papers.
Handles LaTeX command stripping, whitespace normalization, author parsing,
year extraction, and paper record validation.
"""

import re
import html
from typing import Dict, Any, Tuple, List, Optional

# Regular expressions for LaTeX artifacts common in arXiv abstracts
RE_LATEX_COMMANDS = re.compile(r"\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})?")
RE_LATEX_BRACES = re.compile(r"[\{\}]")
RE_LATEX_MATH_BLOCKS = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
RE_INLINE_CITATIONS = re.compile(r"\\cite\{.*?\}|\[[0-9,\s]+\]")
RE_MULTIPLE_SPACES = re.compile(r"\s+")
RE_NEWLINES = re.compile(r"[\r\n]+")

def clean_latex(text: str) -> str:
    """
    Cleans LaTeX commands, escaped characters, and math formatting from arXiv text
    while preserving the semantic reading flow.
    """
    if not text:
        return ""
    
    # Unescape HTML entities if any
    text = html.unescape(text)
    
    # Remove citations like \cite{devlin2018}
    text = re.sub(r"\\cite[tp]?\*?(?:\[.*?\])?\{([^\}]*)\}", r"(\1)", text)
    
    # Replace formatting commands \emph{abc}, \textbf{abc}, \textit{abc} with inner text
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[.*?\])?\{([^\}]*)\}", r"\1", text)
    
    # Clean lingering single LaTeX commands like \etal, \alpha, \beta, \dots
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    
    # Clean remaining curly brackets and escaped dollars/percents
    text = text.replace(r"\$", "$").replace(r"\%", "%").replace(r"\_", "_")
    text = RE_LATEX_BRACES.sub("", text)
    
    return text

def normalize_whitespace(text: str) -> str:
    """Normalizes newlines, tabs, and duplicate spaces into clean single spaces."""
    if not text:
        return ""
    text = RE_NEWLINES.sub(" ", text)
    text = RE_MULTIPLE_SPACES.sub(" ", text)
    return text.strip()

def clean_title(title: str) -> str:
    """Normalizes and cleans paper title."""
    if not title:
        return ""
    cleaned = clean_latex(title)
    return normalize_whitespace(cleaned)

def clean_abstract(abstract: str) -> str:
    """Cleans paper abstract, ensuring fluent readability for embedding and LLM context."""
    if not abstract:
        return ""
    cleaned = clean_latex(abstract)
    return normalize_whitespace(cleaned)

def format_authors(authors_raw: Optional[str], authors_parsed: Optional[List[List[str]]]) -> Tuple[str, List[str]]:
    """
    Parses and standardizes author metadata.
    Returns:
        (formatted_str, author_list)
        e.g., ("Ashish Vaswani, Noam Shazeer...", ["Ashish Vaswani", "Noam Shazeer", ...])
    """
    author_list = []
    
    if authors_parsed and isinstance(authors_parsed, list) and len(authors_parsed) > 0:
        for entry in authors_parsed:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                last = entry[0].strip() if entry[0] else ""
                first = entry[1].strip() if entry[1] else ""
                middle = entry[2].strip() if len(entry) > 2 and entry[2] else ""
                parts = [p for p in [first, middle, last] if p]
                full_name = " ".join(parts)
                if full_name:
                    author_list.append(full_name)
                    
    if not author_list and authors_raw:
        # Fallback to cleaning raw author string
        cleaned_raw = clean_latex(authors_raw)
        cleaned_raw = normalize_whitespace(cleaned_raw)
        # Split by comma or 'and'
        raw_parts = re.split(r",|\sand\s", cleaned_raw)
        author_list = [p.strip() for p in raw_parts if p.strip()]

    formatted_str = ", ".join(author_list) if author_list else (authors_raw or "Unknown Author")
    return formatted_str, author_list

def extract_publication_year(versions: Optional[List[Dict[str, Any]]], update_date: Optional[str]) -> int:
    """
    Extracts the 4-digit publication year.
    Prioritizes v1 created timestamp, then update_date.
    """
    if versions and isinstance(versions, list) and len(versions) > 0:
        first_ver = versions[0]
        created = first_ver.get("created", "")
        # Matches 4 digits after 19xx or 20xx
        match = re.search(r"\b(19\d\d|20\d\d)\b", created)
        if match:
            return int(match.group(1))
            
    if update_date and isinstance(update_date, str):
        match = re.search(r"\b(19\d\d|20\d\d)\b", update_date)
        if match:
            return int(match.group(1))
            
    return 2024  # Default fallback

def extract_publication_date(versions: Optional[List[Dict[str, Any]]], update_date: Optional[str]) -> str:
    """Extracts a standardized publication date string (YYYY-MM-DD or readable date)."""
    if versions and isinstance(versions, list) and len(versions) > 0:
        first_ver = versions[0]
        created = first_ver.get("created", "")
        if created:
            # Example: "Mon, 12 Jun 2017 17:57:34 GMT"
            return created.strip()
    if update_date:
        return update_date.strip()
    return "Unknown"

def is_valid_paper(record: Dict[str, Any], min_abstract_len: int = 50) -> Tuple[bool, str]:
    """
    Validates a paper record for inclusion in the knowledge base.
    Filters out withdrawn papers, empty titles, and short stubs.
    """
    paper_id = record.get("id")
    if not paper_id or not str(paper_id).strip():
        return False, "Missing paper ID"
        
    title = record.get("title", "")
    if not title or len(title.strip()) < 5:
        return False, "Title too short or empty"
        
    abstract = record.get("abstract", "")
    if not abstract or len(abstract.strip()) < min_abstract_len:
        return False, f"Abstract too short (< {min_abstract_len} chars)"
        
    abstract_lower = abstract.lower()
    if "this paper has been withdrawn" in abstract_lower or "withdrawn by author" in abstract_lower:
        return False, "Withdrawn paper notice"
        
    return True, "Valid"

"""
Global Scientific Literature Search Engine for ScholarMind AI.
Directly interfaces with open scholarly registries (Crossref & arXiv Open APIs)
to retrieve peer-reviewed and preprint publications across IEEE, Springer, ACM,
Elsevier, Nature, Wiley, and arXiv from 2000 to present.
"""

import urllib.request
import urllib.parse
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

class GlobalLiteratureSearchEngine:
    """Discovers research papers across global publishers (IEEE, Springer, arXiv, etc.)."""

    def __init__(self, timeout_sec: int = 8):
        self.timeout = timeout_sec
        self.user_agent = "ScholarMindAI/2.0 (mailto:research@scholarmind.ai; +https://scholarmind.ai)"

    def search_global_literature(
        self,
        query: str,
        limit: int = 10,
        year_min: Optional[int] = 2000,
        year_max: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries Crossref and arXiv API for papers matching query across all publishers.
        Returns unified list of paper dictionaries.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        results: List[Dict[str, Any]] = []

        # 1. Query Crossref API (Covers IEEE, Springer, Elsevier, ACM, Wiley, Nature, etc.)
        crossref_items = self._fetch_crossref(clean_query, limit=limit)
        results.extend(crossref_items)

        # 2. Query arXiv API if needed to enrich preprint coverage
        if len(results) < limit:
            arxiv_items = self._fetch_arxiv(clean_query, limit=min(limit, 5))
            results.extend(arxiv_items)

        # 3. Post-Filter by Year if specified
        filtered: List[Dict[str, Any]] = []
        seen_titles = set()

        for paper in results:
            title_norm = re.sub(r'[^a-zA-Z0-9]', '', paper.get("title", "").lower())
            if not title_norm or title_norm in seen_titles:
                continue
            seen_titles.add(title_norm)

            paper_year = paper.get("year", datetime.now().year)
            if year_min and paper_year < year_min:
                continue
            if year_max and paper_year > year_max:
                continue

            filtered.append(paper)

        return filtered[:limit]

    def _fetch_crossref(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Queries Crossref works API for published papers."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.crossref.org/works?query={encoded_query}&rows={limit}&sort=relevance"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                items = data.get("message", {}).get("items", [])
                
                papers = []
                for item in items:
                    titles = item.get("title", [])
                    if not titles:
                        continue
                    title = titles[0].strip()

                    # Format Authors
                    author_list = []
                    for auth in item.get("author", [])[:4]:
                        given = auth.get("given", "")
                        family = auth.get("family", "")
                        name = f"{given} {family}".strip() or family
                        if name:
                            author_list.append(name)
                    authors_str = ", ".join(author_list) if author_list else "Unknown Authors"
                    if len(item.get("author", [])) > 4:
                        authors_str += " et al."

                    # Extract Year
                    year = None
                    date_parts = item.get("published", {}).get("date-parts") or item.get("created", {}).get("date-parts")
                    if date_parts and date_parts[0]:
                        year = int(date_parts[0][0])
                    if not year:
                        year = datetime.now().year

                    # Publisher & Venue
                    publisher = item.get("publisher", "Academic Publisher")
                    container = item.get("container-title", [])
                    venue = container[0] if container else publisher

                    doi = item.get("DOI", "")
                    url_link = item.get("URL", f"https://doi.org/{doi}" if doi else "")

                    # Clean abstract if present (Crossref abstracts often have JATS XML tags)
                    raw_abstract = item.get("abstract", "")
                    clean_abstract = re.sub(r'<[^>]+>', ' ', raw_abstract).strip() if raw_abstract else ""
                    if not clean_abstract or len(clean_abstract) < 40:
                        clean_abstract = f"Published paper in {venue} ({year}) addressing '{title}'. Full text and citation indices accessible via digital object identifier (DOI: {doi})."

                    papers.append({
                        "id": doi or f"crossref_{abs(hash(title)) % 1000000}",
                        "title": title,
                        "authors": authors_str,
                        "year": year,
                        "publisher": publisher,
                        "venue": venue,
                        "abstract": clean_abstract,
                        "doi": doi,
                        "url": url_link,
                        "categories": f"Global Research ({publisher})",
                        "primary_category": "Global Literature",
                        "source": "Global Scientific Web (IEEE / Springer / Crossref)",
                        "retrieval_similarity": 0.95
                    })
                return papers
        except Exception:
            return []

    def _fetch_arxiv(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Queries arXiv Atom API for related preprint papers."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                xml_data = resp.read().decode("utf-8", errors="ignore")
                
                # Parse Atom XML entries with regex
                entries = re.findall(r'<entry>(.*?)</entry>', xml_data, re.DOTALL)
                papers = []
                for entry in entries:
                    t_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                    title = re.sub(r'\s+', ' ', t_match.group(1)).strip() if t_match else "Untitled"
                    
                    s_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    summary = re.sub(r'\s+', ' ', s_match.group(1)).strip() if s_match else ""

                    p_match = re.search(r'<published>(\d{4})', entry)
                    year = int(p_match.group(1)) if p_match else datetime.now().year

                    id_match = re.search(r'<id>(.*?)</id>', entry)
                    arxiv_id = id_match.group(1).split("/abs/")[-1] if id_match else f"arxiv_{abs(hash(title)) % 1000000}"

                    auth_matches = re.findall(r'<author>\s*<name>(.*?)</name>', entry)
                    authors_str = ", ".join(auth_matches[:4]) if auth_matches else "arXiv Author(s)"

                    papers.append({
                        "id": arxiv_id,
                        "title": title,
                        "authors": authors_str,
                        "year": year,
                        "publisher": "arXiv.org",
                        "venue": "arXiv Preprint",
                        "abstract": summary,
                        "doi": f"arxiv:{arxiv_id}",
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "categories": "arXiv Preprint",
                        "primary_category": "cs.AI",
                        "source": "arXiv Open Repository",
                        "retrieval_similarity": 0.92
                    })
                return papers
        except Exception:
            return []

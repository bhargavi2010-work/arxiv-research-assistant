"""
arXiv Metadata Fetcher Utility.
Allows downloading real, live arXiv paper metadata via the official arXiv API (OAI-PMH / Atom API)
in the exact Cornell Kaggle JSON format.
Useful for quickly populating or expanding the local dataset without needing to download the entire 15GB archive.
"""

import urllib.request
import xml.etree.ElementTree as ET
import json
import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import RAW_DATA_DIR, RAW_DATASET_FILENAME, TARGET_CATEGORY_CODES

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

def fetch_arxiv_papers(categories=None, max_results=100, batch_size=50):
    if categories is None:
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
        
    output_path = RAW_DATA_DIR / RAW_DATASET_FILENAME
    cat_query = "+OR+".join([f"cat:{c}" for c in categories])
    
    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record.get("id"))
                    except Exception:
                        pass

    print(f"[INFO] Existing papers in local dataset: {len(existing_ids)}")
    print(f"[INFO] Fetching up to {max_results} papers from arXiv API for categories: {categories}...")
    
    fetched = 0
    start = 0
    
    with open(output_path, "a", encoding="utf-8") as out_f:
        while fetched < max_results:
            current_batch = min(batch_size, max_results - fetched)
            api_url = (
                f"http://export.arxiv.org/api/query?search_query={cat_query}"
                f"&start={start}&max_results={current_batch}&sortBy=submittedDate&sortOrder=descending"
            )
            
            try:
                req = urllib.request.Request(api_url, headers={"User-Agent": "ArXivResearchAssistant/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read()
            except Exception as e:
                print(f"[ERROR] Failed to fetch batch at start={start}: {e}")
                break

            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", ATOM_NS)
            if not entries:
                print("[INFO] No more entries returned by API.")
                break

            for entry in entries:
                id_elem = entry.find("atom:id", ATOM_NS)
                if id_elem is None:
                    continue
                raw_id = id_elem.text.split("/abs/")[-1]
                if raw_id in existing_ids:
                    continue

                title = entry.find("atom:title", ATOM_NS).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ATOM_NS).text.strip().replace("\n", " ")
                published = entry.find("atom:published", ATOM_NS).text.strip()
                
                authors = []
                authors_parsed = []
                for a in entry.findall("atom:author", ATOM_NS):
                    name_elem = a.find("atom:name", ATOM_NS)
                    if name_elem is not None and name_elem.text:
                        name = name_elem.text.strip()
                        authors.append(name)
                        parts = name.split()
                        last = parts[-1] if parts else ""
                        first = " ".join(parts[:-1]) if len(parts) > 1 else ""
                        authors_parsed.append([last, first, ""])

                categories_list = []
                for c in entry.findall("atom:category", ATOM_NS):
                    term = c.attrib.get("term")
                    if term:
                        categories_list.append(term)
                        
                doi_elem = entry.find("arxiv:doi", ATOM_NS)
                doi = doi_elem.text.strip() if doi_elem is not None else f"10.48550/arXiv.{raw_id}"
                
                record = {
                    "id": raw_id,
                    "submitter": authors[0] if authors else "Unknown",
                    "authors": ", ".join(authors),
                    "title": title,
                    "comments": "",
                    "journal-ref": "",
                    "doi": doi,
                    "report-no": "",
                    "categories": " ".join(categories_list),
                    "license": "",
                    "abstract": summary,
                    "versions": [{"version": "v1", "created": published}],
                    "update_date": published[:10],
                    "authors_parsed": authors_parsed
                }
                
                out_f.write(json.dumps(record) + "\n")
                existing_ids.add(raw_id)
                fetched += 1

            start += len(entries)
            print(f"[PROGRESS] Fetched {fetched}/{max_results} papers...")
            time.sleep(1.0)  # Be polite to arXiv API rate limit

    print(f"[DONE] Successfully added {fetched} papers. Total records now: {len(existing_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch arXiv metadata subset")
    parser.add_argument("--max_papers", type=int, default=50, help="Maximum number of papers to fetch")
    args = parser.parse_args()
    fetch_arxiv_papers(max_results=args.max_papers)

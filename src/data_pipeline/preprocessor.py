"""
Pre-processing Pipeline for ArXiv Scientific Research Papers.
Streams raw JSONL data, applies domain filtering, cleans text, validates records,
deduplicates entries, and exports structured Parquet and JSON datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import pandas as pd
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_DIR,
    RAW_DATASET_FILENAME,
    PROCESSED_DATA_DIR,
    PROCESSED_PARQUET_FILENAME,
    PROCESSED_JSON_FILENAME
)
from src.data_pipeline.text_cleaner import (
    clean_title,
    clean_abstract,
    format_authors,
    extract_publication_year,
    extract_publication_date,
    is_valid_paper
)
from src.data_pipeline.domain_filter import (
    is_cs_ai_paper,
    get_primary_category,
    get_all_category_labels,
    parse_categories
)

class ArXivPreprocessor:
    """End-to-end preprocessor for arXiv research paper metadata."""

    def __init__(
        self,
        raw_filepath: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        self.raw_filepath = raw_filepath or (RAW_DATA_DIR / RAW_DATASET_FILENAME)
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.output_dir / PROCESSED_PARQUET_FILENAME
        self.json_sample_path = self.output_dir / PROCESSED_JSON_FILENAME

    def stream_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        """Streams records line-by-line from raw JSONL file with constant memory usage."""
        if not self.raw_filepath.exists():
            raise FileNotFoundError(f"Raw arXiv file not found at: {self.raw_filepath}")

        with open(self.raw_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    yield record
                except json.JSONDecodeError:
                    continue

    def process_record(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes and normalizes a single raw record if valid and domain-relevant."""
        categories_raw = raw.get("categories", "")
        if not is_cs_ai_paper(categories_raw):
            return None

        is_valid, reason = is_valid_paper(raw)
        if not is_valid:
            return None

        paper_id = str(raw.get("id", "")).strip()
        title = clean_title(raw.get("title", ""))
        abstract = clean_abstract(raw.get("abstract", ""))
        
        authors_raw = raw.get("authors", "")
        authors_parsed = raw.get("authors_parsed", [])
        authors_str, authors_list = format_authors(authors_raw, authors_parsed)
        
        primary_cat_code, primary_cat_name = get_primary_category(categories_raw)
        all_cat_codes = parse_categories(categories_raw)
        all_cat_labels = get_all_category_labels(categories_raw)
        
        versions = raw.get("versions", [])
        update_date = raw.get("update_date", "")
        pub_year = extract_publication_year(versions, update_date)
        pub_date = extract_publication_date(versions, update_date)
        
        doi = str(raw.get("doi") or "").strip()
        journal_ref = str(raw.get("journal-ref") or "").strip()
        comments = str(raw.get("comments") or "").strip()
        
        arxiv_url = f"https://arxiv.org/abs/{paper_id}"
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        # Search & retrieval context string (combines title and abstract)
        retrieval_text = f"Title: {title}\nAbstract: {abstract}"

        return {
            "id": paper_id,
            "title": title,
            "authors": authors_str,
            "authors_list": authors_list,
            "abstract": abstract,
            "retrieval_text": retrieval_text,
            "primary_category": primary_cat_code,
            "primary_category_name": primary_cat_name,
            "categories": all_cat_codes,
            "category_labels": all_cat_labels,
            "year": pub_year,
            "published_date": pub_date,
            "doi": doi,
            "journal_ref": journal_ref,
            "comments": comments,
            "arxiv_url": arxiv_url,
            "pdf_url": pdf_url
        }

    def run(self, max_records: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the data preprocessing pipeline.
        Args:
            max_records: Maximum number of domain records to process and save (for dev subsets).
        Returns:
            Dict containing processing metrics and paths.
        """
        print(f"\n[PREPROCESSOR] Starting ingestion from: {self.raw_filepath}")
        
        processed_records: List[Dict[str, Any]] = []
        seen_ids = set()
        
        total_scanned = 0
        domain_matched = 0
        invalid_count = 0
        duplicate_count = 0

        for raw_item in self.stream_raw_records():
            total_scanned += 1
            cat_raw = raw_item.get("categories", "")
            
            if not is_cs_ai_paper(cat_raw):
                continue
            domain_matched += 1
            
            paper_id = str(raw_item.get("id", "")).strip()
            if paper_id in seen_ids:
                duplicate_count += 1
                continue
                
            processed = self.process_record(raw_item)
            if not processed:
                invalid_count += 1
                continue
                
            seen_ids.add(paper_id)
            processed_records.append(processed)
            
            if max_records and len(processed_records) >= max_records:
                print(f"[PREPROCESSOR] Reached target subset limit of {max_records} records.")
                break

        print(f"[PREPROCESSOR] Ingestion complete.")
        print(f"  - Total Scanned: {total_scanned}")
        print(f"  - Domain Matched (CS/AI): {domain_matched}")
        print(f"  - Invalid / Withdrawn Filtered: {invalid_count}")
        print(f"  - Duplicates Skipped: {duplicate_count}")
        print(f"  - Final Cleaned Papers Retained: {len(processed_records)}")

        # Convert to DataFrame
        df = pd.DataFrame(processed_records)
        
        # Save Parquet
        df.to_parquet(self.parquet_path, index=False, engine="pyarrow")
        print(f"[PREPROCESSOR] Saved Parquet to: {self.parquet_path} ({self.parquet_path.stat().st_size / 1024:.1f} KB)")
        
        # Save human-readable JSON sample (first 100 or all)
        sample_slice = processed_records[:100]
        with open(self.json_sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_slice, f, indent=2, ensure_ascii=False)
        print(f"[PREPROCESSOR] Saved sample JSON to: {self.json_sample_path}")

        return {
            "total_scanned": total_scanned,
            "domain_matched": domain_matched,
            "invalid_count": invalid_count,
            "duplicate_count": duplicate_count,
            "final_count": len(processed_records),
            "parquet_path": str(self.parquet_path),
            "json_sample_path": str(self.json_sample_path)
        }

    def load_processed_data(self) -> pd.DataFrame:
        """Loads the cleaned processed dataset as a DataFrame."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at: {self.parquet_path}. Run preprocessing first.")
        return pd.read_parquet(self.parquet_path, engine="pyarrow")

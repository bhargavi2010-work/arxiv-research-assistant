"""
PDF Processor & Ingestion Pipeline for ArXiv Research Assistant.
Extracts text, metadata, and sections from scientific research PDFs using pypdf,
chunks them into semantically rich passages with page provenance, and indexes
them directly into ChromaDB for real-time RAG chat and summarization.
"""

import io
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import pypdf
except ImportError:
    pypdf = None


class PDFResearchPaperProcessor:
    """Extracts, parses, and indexes arbitrary research paper PDFs."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def is_pypdf_available(self) -> bool:
        return pypdf is not None

    def extract_pdf_data(self, file_content: bytes, filename: str = "uploaded_paper.pdf") -> Dict[str, Any]:
        """
        Parses a PDF binary stream and extracts structured academic paper content.
        
        Returns:
            Dict containing:
                - title: Extracted or derived title
                - authors: Extracted author string or 'Uploaded Author'
                - abstract: Extracted abstract or first paragraph
                - total_pages: Total number of pages
                - full_text: Concatenated cleaned text
                - pages: List of dicts with page_number and text
                - chunks: List of chunk dicts ready for vector embedding
        """
        if pypdf is None:
            raise ImportError("pypdf is required for processing PDFs. Please install it via pip install pypdf.")

        reader = pypdf.PdfReader(io.BytesIO(file_content))
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError("The uploaded PDF contains no readable pages.")

        pages_data: List[Dict[str, Any]] = []
        raw_full_text_list: List[str] = []

        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            
            cleaned_page = self._clean_scientific_text(page_text)
            pages_data.append({
                "page_number": idx + 1,
                "text": cleaned_page
            })
            if cleaned_page:
                raw_full_text_list.append(cleaned_page)

        full_text = "\n\n".join(raw_full_text_list)
        if not full_text.strip():
            raise ValueError("No extractable text found in this PDF (it might be scanned images or password protected).")

        # Extract or derive metadata
        meta = reader.metadata or {}
        extracted_title = meta.title if meta.title and len(meta.title.strip()) > 4 else None
        
        if not extracted_title:
            extracted_title = self._infer_title_from_first_page(pages_data[0]["text"] if pages_data else "", filename)

        extracted_abstract = self._extract_abstract_section(full_text)
        
        # Build chunks with page provenance
        chunks = self._chunk_pages(pages_data, doc_title=extracted_title)

        doc_id = f"pdf_{hashlib.md5((filename + str(total_pages)).encode('utf-8')).hexdigest()[:10]}"

        return {
            "id": doc_id,
            "filename": filename,
            "title": extracted_title,
            "authors": meta.author or "Custom Uploaded Author(s)",
            "abstract": extracted_abstract,
            "total_pages": total_pages,
            "full_text": full_text,
            "pages": pages_data,
            "chunks": chunks,
            "upload_timestamp": datetime.now().isoformat()
        }

    def _clean_scientific_text(self, text: str) -> str:
        """Removes hyphenation artifacts, strange control characters, and redundant whitespace."""
        if not text:
            return ""
        # Fix hyphenated words at line wraps (e.g. "repre-\nsentation" -> "representation")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Compress multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Compress excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _infer_title_from_first_page(self, first_page_text: str, fallback_filename: str) -> str:
        """Infers a paper's title from the topmost prominent text on the title page."""
        lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
        for line in lines[:5]:
            # Skip boilerplate headers (arXiv header, ISSN, dates)
            if re.search(r'(arxiv|preprint|doi|issn|ieee|acm|volume|pages|\d{4})', line, re.IGNORECASE):
                continue
            if len(line.split()) >= 3 and len(line) <= 150:
                return line
        
        # Fallback to cleaning filename
        cleaned_fn = re.sub(r'\.pdf$', '', fallback_filename, flags=re.IGNORECASE)
        cleaned_fn = re.sub(r'[-_]', ' ', cleaned_fn).title()
        return cleaned_fn

    def _extract_abstract_section(self, full_text: str) -> str:
        """Extracts the abstract section if present, otherwise returns the introductory passage."""
        match = re.search(r'(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)[:\s\n]+(.*?)(?:\n\s*(?:1[\.\s]|introduction|keywords|index terms|1\.0))', full_text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract_text = match.group(1).strip()
            if len(abstract_text) > 60:
                return re.sub(r'\s+', ' ', abstract_text)[:1200]
        
        # Fallback to first non-empty 300 words
        words = full_text.split()
        return " ".join(words[:200]) + ("..." if len(words) > 200 else "")

    def _chunk_pages(self, pages: List[Dict[str, Any]], doc_title: str) -> List[Dict[str, Any]]:
        """Splits page contents into overlapping chunks tagged with page number and title."""
        chunks: List[Dict[str, Any]] = []
        chunk_idx = 0

        for p in pages:
            page_num = p["page_number"]
            text = p["text"]
            if not text:
                continue

            words = text.split()
            if len(words) <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"{chunk_idx}",
                    "page_number": page_num,
                    "text": f"[{doc_title} — Page {page_num}]\n{text}"
                })
                chunk_idx += 1
            else:
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = " ".join(chunk_words)
                    chunks.append({
                        "chunk_id": f"{chunk_idx}",
                        "page_number": page_num,
                        "text": f"[{doc_title} — Page {page_num}]\n{chunk_text}"
                    })
                    chunk_idx += 1
                    if i + self.chunk_size >= len(words):
                        break

        return chunks

    def index_pdf_into_chromadb(
        self,
        pdf_data: Dict[str, Any],
        chroma_manager: Any,
        embedder: Any
    ) -> Dict[str, Any]:
        """
        Embeds chunks of the uploaded PDF and saves them in ChromaDB so that
        the RAG engine, LangChain agent, and search engine can immediately cite it.
        """
        paper_id = pdf_data["id"]
        title = pdf_data["title"]
        authors = pdf_data["authors"]
        abstract = pdf_data["abstract"]
        chunks = pdf_data["chunks"]

        # Format as a standard paper entry for ChromaManager
        # Also store parent paper metadata so it shows in Paper Search
        parent_doc = {
            "id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": "User Uploaded Paper (cs.AI)",
            "primary_category": "cs.AI",
            "year": datetime.now().year,
            "update_date": datetime.now().strftime("%Y-%m-%d"),
            "doi": f"custom-{paper_id}",
            "journal_ref": f"Uploaded PDF ({pdf_data['total_pages']} pages)",
            "is_custom_upload": True
        }

        # Check if collection has add_papers or index_papers
        if hasattr(chroma_manager, "index_papers"):
            chroma_manager.index_papers([parent_doc], embedder=embedder)

        # In addition, index individual detailed page chunks for deep precision retrieval
        chunk_texts = [c["text"] for c in chunks]
        chunk_ids = [f"{paper_id}_c{c['chunk_id']}" for c in chunks]
        chunk_metadatas = [{
            "id": paper_id,
            "title": title,
            "categories": "User Uploaded Paper",
            "year": datetime.now().year,
            "page_number": c["page_number"],
            "is_custom_upload": True
        } for c in chunks]

        if chunk_texts:
            chunk_embeddings = embedder.embed_documents(chunk_texts)
            chroma_manager.collection.upsert(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas
            )

        return {
            "paper_id": paper_id,
            "title": title,
            "chunks_indexed": len(chunks),
            "total_pages": pdf_data["total_pages"],
            "status": "success"
        }

"""
Unit tests for PDF Processor & Custom Paper Ingestion.
"""

import unittest
import io
import pypdf
from pypdf import PdfWriter
from src.data_pipeline.pdf_processor import PDFResearchPaperProcessor

class TestPDFResearchPaperProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PDFResearchPaperProcessor(chunk_size=100, chunk_overlap=20)

    def test_pypdf_availability(self):
        self.assertTrue(self.processor.is_pypdf_available())

    def test_clean_scientific_text(self):
        dirty = "Trans-\nformer models are power-\nful.   Multiple    spaces.\n\n\n\nToo many lines."
        cleaned = self.processor._clean_scientific_text(dirty)
        self.assertIn("Transformer models are powerful.", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_extract_abstract_section(self):
        text = "Abstract: This paper presents an agentic system for scientific paper QA. 1. Introduction: In recent years..."
        abstract = self.processor._extract_abstract_section(text)
        self.assertIn("This paper presents an agentic system", abstract)

    def test_infer_title_from_first_page(self):
        first_page = "arXiv:2305.12345v1 [cs.AI] 20 May 2023\nDeep Residual Learning for Image Recognition\nKaiming He, Xiangyu Zhang\nAbstract: ..."
        title = self.processor._infer_title_from_first_page(first_page, "resnet_paper.pdf")
        self.assertEqual(title, "Deep Residual Learning for Image Recognition")

    def test_infer_title_fallback(self):
        title = self.processor._infer_title_from_first_page("", "graph_attention_networks.pdf")
        self.assertEqual(title, "Graph Attention Networks")

    def test_chunking_with_provenance(self):
        pages = [
            {"page_number": 1, "text": "Page one text discussing self-attention."},
            {"page_number": 2, "text": "Page two text discussing feed-forward networks."}
        ]
        chunks = self.processor._chunk_pages(pages, "Transformer Paper")
        self.assertEqual(len(chunks), 2)
        self.assertIn("[Transformer Paper — Page 1]", chunks[0]["text"])
        self.assertIn("[Transformer Paper — Page 2]", chunks[1]["text"])

if __name__ == "__main__":
    unittest.main()

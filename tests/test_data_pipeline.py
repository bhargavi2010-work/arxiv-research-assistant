"""
Unit tests for Phase 2: Data Preprocessing and Domain Filtering.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline.text_cleaner import (
    clean_latex,
    normalize_whitespace,
    clean_title,
    clean_abstract,
    format_authors,
    extract_publication_year,
    is_valid_paper
)
from src.data_pipeline.domain_filter import (
    is_cs_ai_paper,
    get_primary_category,
    get_all_category_labels
)
from src.data_pipeline.preprocessor import ArXivPreprocessor

class TestTextCleaner(unittest.TestCase):
    def test_clean_latex_commands(self):
        raw = r"In this work, we use \textbf{Transformer} architectures \cite{vaswani2017} for \$500\% gain."
        cleaned = clean_latex(raw)
        self.assertNotIn(r"\textbf", cleaned)
        self.assertNotIn(r"\cite", cleaned)
        self.assertIn("Transformer", cleaned)
        self.assertIn("(vaswani2017)", cleaned)

    def test_normalize_whitespace(self):
        raw = "  This   is \n\n a multi-line \t text.  "
        expected = "This is a multi-line text."
        self.assertEqual(normalize_whitespace(raw), expected)

    def test_format_authors(self):
        parsed = [["Vaswani", "Ashish", ""], ["Shazeer", "Noam", ""]]
        fmt, lst = format_authors("", parsed)
        self.assertEqual(fmt, "Ashish Vaswani, Noam Shazeer")
        self.assertEqual(lst, ["Ashish Vaswani", "Noam Shazeer"])

    def test_year_extraction(self):
        versions = [{"version": "v1", "created": "Mon, 12 Jun 2017 17:57:34 GMT"}]
        year = extract_publication_year(versions, "2023-10-01")
        self.assertEqual(year, 2017)

    def test_invalid_paper_withdrawn(self):
        record = {
            "id": "1234.5678",
            "title": "A Paper",
            "abstract": "This paper has been withdrawn by the authors due to an error."
        }
        valid, reason = is_valid_paper(record)
        self.assertFalse(valid)
        self.assertIn("withdrawn", reason.lower())

class TestDomainFilter(unittest.TestCase):
    def test_cs_ai_matching(self):
        self.assertTrue(is_cs_ai_paper("cs.AI"))
        self.assertTrue(is_cs_ai_paper("cs.LG cs.CL"))
        self.assertTrue(is_cs_ai_paper("stat.ML math.ST"))
        self.assertFalse(is_cs_ai_paper("math.PR hep-th"))
        self.assertFalse(is_cs_ai_paper("physics.soc-ph"))

    def test_primary_category(self):
        code, label = get_primary_category("cs.CL cs.LG")
        self.assertEqual(code, "cs.CL")
        self.assertIn("Computation and Language", label)

class TestPreprocessorOutput(unittest.TestCase):
    def test_parquet_loading(self):
        preprocessor = ArXivPreprocessor()
        df = preprocessor.load_processed_data()
        self.assertGreater(len(df), 0)
        self.assertIn("id", df.columns)
        self.assertIn("title", df.columns)
        self.assertIn("abstract", df.columns)
        self.assertIn("primary_category", df.columns)
        self.assertIn("year", df.columns)

if __name__ == "__main__":
    unittest.main()

"""
Scientific Paper Summarizer for ArXiv Research Assistant.
Generates structured 4-part and 8-part academic summaries from research metadata and abstracts
without fabricating ungrounded experimental numbers.
"""

from typing import Dict, Any, List, Optional
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.nlp.information_extractor import ScientificInfoExtractor
from src.llm.llm_client import LLMClient

class PaperSummarizer:
    """Produces structured, transparent scientific summaries of research papers."""

    def __init__(self, llm_client: Optional[LLMClient] = None, extractor: Optional[ScientificInfoExtractor] = None):
        self.llm = llm_client or LLMClient()
        self.extractor = extractor or ScientificInfoExtractor()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits abstract text into clean sentences."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if len(s.strip()) > 10]

    def summarize(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a comprehensive structured research summary for a paper.
        Args:
            paper: Dictionary containing at least 'title' and 'abstract' (and optionally authors, year, id).
        Returns:
            Structured dictionary with Problem, Approach, Contributions, Results, Key Concepts, and Metadata.
        """
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "").strip()
        authors = paper.get("authors", "Unknown Authors")
        year = paper.get("year", 2024)
        paper_id = paper.get("id", "")
        primary_cat = paper.get("primary_category_name") or paper.get("primary_category", "Computer Science")

        if not abstract:
            return {
                "title": title,
                "error": "No abstract available to generate summary.",
                "is_abstract_based": True
            }

        # 1. NLP Information Extraction
        extracted_info = self.extractor.extract_all(title, abstract)
        sentences = self._split_into_sentences(abstract)

        # 2. Check if active LLM can generate structured decomposition
        llm_summary = None
        if self.llm and (self.llm._is_ollama_available() or self.llm.hf_token or self.llm._hf_pipeline is not None):
            try:
                llm_prompt = f"""You are an expert scientific researcher. Analyze the following research paper abstract and extract a structured 4-part academic summary in valid JSON format.
Title: {title}
Abstract: {abstract}

Return a valid JSON object with EXACTLY these keys:
{{
  "problem": "1-2 sentences on the core challenge/limitation addressed",
  "approach": "1-2 sentences explaining the proposed model/architecture/methodology",
  "key_contributions": ["list", "of", "2-3", "concrete", "scientific", "contributions"],
  "results": "1-2 sentences summarizing experimental findings, benchmarks, or efficiency gains"
}}
JSON:"""
                raw_llm = self.llm.generate(llm_prompt)
                if raw_llm:
                    import json
                    json_match = re.search(r"\{.*\}", raw_llm, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        if all(k in parsed for k in ["problem", "approach", "key_contributions", "results"]):
                            llm_summary = parsed
            except Exception:
                llm_summary = None

        if llm_summary:
            problem_text = llm_summary["problem"]
            approach_text = llm_summary["approach"]
            contributions = llm_summary["key_contributions"]
            results_text = llm_summary["results"]
        else:
            # 3. Transparent NLP Discourse Parsing for Academic Structure
            problem_sentences = []
            approach_sentences = []
            results_sentences = []
            other_sentences = []

            problem_keywords = [
                "however", "problem", "challenge", "limited", "lack", "difficulty",
                "bottleneck", "recurrent", "expensive", "complex", "struggle", "suffer",
                "dominant sequence", "drawback", "computational cost"
            ]
            approach_keywords = [
                "propose", "introduce", "present", "explore", "architecture",
                "framework", "model", "we show", "we design", "dispensing with",
                "based solely on", "we develop", "new simple"
            ]
            results_keywords = [
                "achieve", "experiment", "results", "superior", "state-of-the-art",
                "sota", "performance", "improve", "faster", "evaluat", "benchmark",
                "less time to train", "more parallelizable", "outperform", "bleu", "accuracy"
            ]

            for s in sentences:
                s_lower = s.lower()
                is_approach = any(k in s_lower for k in approach_keywords)
                is_problem = any(k in s_lower for k in problem_keywords) and not is_approach
                is_results = any(k in s_lower for k in results_keywords)

                if is_problem:
                    problem_sentences.append(s)
                elif is_approach:
                    approach_sentences.append(s)
                elif is_results:
                    results_sentences.append(s)
                else:
                    other_sentences.append(s)

            # Synthesize Problem
            if problem_sentences:
                problem_text = " ".join(problem_sentences)
            elif sentences:
                problem_text = sentences[0]
            else:
                problem_text = f"Addresses foundational challenges in {primary_cat.lower()} modeling."

            # Synthesize Approach
            if approach_sentences:
                approach_text = " ".join(approach_sentences)
            elif len(sentences) > 1:
                approach_text = sentences[1]
            else:
                approach_text = f"Proposes algorithmic and architectural innovations detailed in '{title}'."

            # Synthesize Key Contributions
            contributions = []
            if extracted_info["models"]:
                contributions.append(f"Introduces or analyzes the {', '.join(extracted_info['models'])} architecture.")
            if extracted_info["methods"]:
                contributions.append(f"Applies core methods including {', '.join(extracted_info['methods'][:3])}.")
            if extracted_info["tasks"]:
                contributions.append(f"Demonstrates efficacy on scientific tasks such as {', '.join(extracted_info['tasks'][:2])}.")
            if not contributions and approach_sentences:
                contributions.append(approach_sentences[0])
            if not contributions:
                contributions.append(f"Presents novel experimental findings for {title}.")

            # Synthesize Results (no fake generic filler)
            if results_sentences:
                results_text = " ".join(results_sentences)
            else:
                results_text = "Specific quantitative experimental results or benchmark metrics are not detailed in the abstract text."

        # Executive overview (1-2 sentences)
        primary_model = extracted_info['models'][0] if extracted_info['models'] else 'a novel framework'
        exec_summary = f"{title} ({year}) addresses key challenges in {primary_cat.lower()} by proposing {primary_model}. The paper details architectural formulations and empirical findings on target benchmarks."

        return {
            "id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "primary_category": primary_cat,
            "executive_summary": exec_summary,
            "problem": problem_text,
            "approach": approach_text,
            "key_contributions": contributions,
            "results": results_text,
            "extracted_models": extracted_info["models"],
            "extracted_methods": extracted_info["methods"],
            "extracted_tasks": extracted_info["tasks"],
            "extracted_datasets": extracted_info["datasets"],
            "key_concepts": extracted_info["concepts"],
            "keywords": extracted_info["keywords"],
            "is_abstract_based": True,
            "disclaimer": "This structured summary is synthesized based on the canonical paper title, abstract, and published metadata provided by arXiv."
        }

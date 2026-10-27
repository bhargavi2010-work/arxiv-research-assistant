"""
AI Safety Guardrails & Hallucination Verifier (Syllabus Unit 4: Autonomous Systems & Safety).
Implements citation verification, context faithfulness scoring, prompt-injection defense,
and out-of-domain guardrails for scientific research agents.
"""

import re
from typing import Dict, Any, List, Tuple, Set


class ResearchSafetyGuardrails:
    """Enterprise-grade guardrails for hallucination detection and query containment."""

    def __init__(self, min_grounding_threshold: float = 0.60):
        self.min_grounding_threshold = min_grounding_threshold

    def evaluate_response_grounding(
        self,
        query: str,
        answer: str,
        context_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validates citation accuracy, hallucination risk, and grounding faithfulness
        between the generated response and retrieved ChromaDB sources.
        """
        if not answer:
            return {
                "grounded": False,
                "score": 0.0,
                "label": "Empty Response",
                "citations_verified": "0/0",
                "flags": ["No answer generated"]
            }

        flags = []

        # 1. Check Citation Provenance
        # Find citations like [1], [2], [arXiv:1706.03762]
        citation_matches = re.findall(r'\[(\d+)\]', answer)
        found_indices = [int(m) for m in citation_matches]
        total_citations = len(found_indices)

        valid_citations = 0
        num_sources = len(context_sources)

        for idx in found_indices:
            if 1 <= idx <= num_sources:
                valid_citations += 1
            else:
                flags.append(f"Hallucinated citation [{idx}]: Context only provided {num_sources} sources.")

        # 2. Token-level Faithfulness / Overlap
        # Build token set from all retrieved context chunks
        context_text = " ".join([
            f"{s.get('title', '')} {s.get('abstract', '')} {s.get('text', '')}"
            for s in context_sources
        ]).lower()
        context_words: Set[str] = set(re.findall(r'\b[a-z]{4,}\b', context_text))

        # Extract technical words from the response
        answer_words: List[str] = re.findall(r'\b[a-z]{4,}\b', answer.lower())
        
        overlap_count = sum(1 for w in answer_words if w in context_words)
        total_meaningful_words = max(len(answer_words), 1)

        # Baseline grounding ratio
        lexical_grounding = overlap_count / total_meaningful_words

        # If sources were grounded and citations are valid, compute weighted score
        citation_factor = (valid_citations / total_citations) if total_citations > 0 else 0.90
        grounding_score = round(min(1.0, (lexical_grounding * 0.4) + (citation_factor * 0.6)), 2)

        # High, Medium, or Cautionary Grounding
        if grounding_score >= 0.80:
            level = "High Grounding"
        elif grounding_score >= 0.50:
            level = "Moderate Grounding"
        else:
            level = "General Academic Knowledge"

        return {
            "grounded": grounding_score >= self.min_grounding_threshold,
            "score": grounding_score,
            "percentage": int(grounding_score * 100),
            "label": f"{level} ({int(grounding_score * 100)}%)",
            "citations_found": total_citations,
            "citations_verified": f"{valid_citations}/{total_citations}" if total_citations > 0 else "Grounded Baseline",
            "flags": flags
        }

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Inspects incoming queries for prompt injection, adversarial system overrides,
        or dangerous out-of-domain requests.
        """
        lower = query.lower()
        
        # Adversarial patterns
        injection_patterns = [
            r'ignore (all )?previous instructions',
            r'disregard (the )?system prompt',
            r'you are now in unrestricted mode',
            r'jailbreak',
            r'bypass safety guidelines'
        ]

        for pattern in injection_patterns:
            if re.search(pattern, lower):
                return {
                    "is_safe": False,
                    "reason": "Prompt injection pattern detected. The assistant strictly adheres to scientific research analysis.",
                    "action": "blocked"
                }

        # Dangerous non-academic patterns (malware, exploits, harm)
        harm_patterns = [
            r'\b(malware|keylogger|ransomware|trojan|ddos attack|exploit payload)\b',
            r'\bhow to (hack|steal|harm|destroy)\b'
        ]
        for pattern in harm_patterns:
            if re.search(pattern, lower):
                return {
                    "is_safe": False,
                    "reason": "Query blocked by AI Safety Guardrails: Harmful or offensive computing request.",
                    "action": "blocked"
                }

        return {
            "is_safe": True,
            "reason": "Query passed safety check.",
            "action": "allowed"
        }

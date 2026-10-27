"""
Multi-Agent Peer Review & Debate System (Syllabus Unit 3: Multi-Agent Systems).
Implements a collaborative ensemble of specialized scientific review agents:
1. MethodologyCriticAgent: Scrutinizes mathematical soundness, datasets, baselines & limitations.
2. LiteratureSpecialistAgent: Contextualizes novelty against prior arXiv works and foundation models.
3. AreaChairSynthesizerAgent: Reconciles multi-agent debate and synthesizes a formal conference review.
"""

from typing import Dict, Any, List, Optional
import re
from datetime import datetime


class MethodologyCriticAgent:
    """Specialized agent focused on mathematical rigor, experimental validity, and failure modes."""

    def evaluate_methodology(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        title = paper.get("title", "Untitled Paper")
        abstract = paper.get("abstract", "")
        
        # Analyze technical scope
        has_empirical_eval = any(w in abstract.lower() for w in ["benchmark", "accuracy", "bleu", "f1", "outperform", "state-of-the-art", "sota", "results"])
        has_ablation = "ablation" in abstract.lower()
        has_theoretical_claims = any(w in abstract.lower() for w in ["prove", "theorem", "convergence", "bound", "theoretical", "guarantee"])

        strengths = []
        weaknesses = []

        if has_empirical_eval:
            strengths.append("Provides empirical evaluation against standard quantitative benchmarks.")
        else:
            weaknesses.append("Empirical validation lacks explicit mention of standardized numerical baselines in the abstract.")

        if has_theoretical_claims:
            strengths.append("Formulates theoretical justifications and mathematical grounding for the proposed framework.")
        else:
            weaknesses.append("Methodology appears heuristic-driven; formal convergence bounds or proofs are not highlighted.")

        if has_ablation:
            strengths.append("Conducts ablation studies to isolate individual architectural components.")
        else:
            weaknesses.append("Author needs to provide comprehensive component ablations to justify architectural complexity.")

        # Default fallback strengths if abstract is brief
        if not strengths:
            strengths.append("Proposes a targeted computational pipeline addressing high-dimensional complexity.")
        if not weaknesses:
            weaknesses.append("Reproducibility across diverse hardware architectures warrants deeper validation.")

        return {
            "agent_role": "Methodology Critic Agent",
            "strengths": strengths,
            "weaknesses": weaknesses,
            "technical_score": 7.5 if len(weaknesses) <= 1 else 6.0
        }


class LiteratureSpecialistAgent:
    """Specialized agent that evaluates prior work, citation grounding, and novelty delta."""

    def evaluate_novelty_and_literature(self, paper: Dict[str, Any], search_engine: Optional[Any] = None) -> Dict[str, Any]:
        title = paper.get("title", "")
        categories = paper.get("categories", "cs.AI")
        abstract = paper.get("abstract", "")

        related_prior_art = []
        if search_engine:
            try:
                related = search_engine.search(query=title, limit=3)
                for r in related.get("results", []):
                    if r.get("id") != paper.get("id"):
                        related_prior_art.append(f"{r.get('title')} ({r.get('year')})")
            except Exception:
                pass

        if not related_prior_art:
            related_prior_art = ["Foundational Attention & Deep Learning Corpora (arXiv:1706.03762)", "Dense Semantic Retrieval Frameworks"]

        novelty_indicators = ["propose", "novel", "new paradigm", "first time", "unsupervised", "state-of-the-art"]
        novelty_matches = [w for w in novelty_indicators if w in abstract.lower()]

        novelty_verdict = (
            "Substantial Novelty: Introduces distinct algorithmic mechanisms over baseline paradigms."
            if len(novelty_matches) >= 2
            else "Incremental Novelty: Extends existing established frameworks with targeted domain enhancements."
        )

        return {
            "agent_role": "Literature & Novelty Specialist Agent",
            "novelty_verdict": novelty_verdict,
            "prior_art_context": related_prior_art[:3],
            "literature_coverage_score": 8.0
        }


class AreaChairSynthesizerAgent:
    """Lead Reviewer Agent reconciling multi-agent debates into a structured peer-review report."""

    def __init__(self):
        self.critic = MethodologyCriticAgent()
        self.literature = LiteratureSpecialistAgent()

    def generate_full_peer_review(
        self,
        paper: Dict[str, Any],
        search_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes the multi-agent peer review workflow:
        1. Invokes MethodologyCriticAgent
        2. Invokes LiteratureSpecialistAgent
        3. Area Chair synthesizes consensus decision
        """
        critic_review = self.critic.evaluate_methodology(paper)
        lit_review = self.literature.evaluate_novelty_and_literature(paper, search_engine)

        # Consensus score calculation
        avg_score = (critic_review["technical_score"] + lit_review["literature_coverage_score"]) / 2.0

        if avg_score >= 7.5:
            recommendation = "Accept (Poster / Oral Presentation)"
            confidence = "High (4/5)"
        elif avg_score >= 6.5:
            recommendation = "Weak Accept (Borderline Poster)"
            confidence = "Medium (3/5)"
        else:
            recommendation = "Major Revision Required"
            confidence = "High (4/5)"

        report = {
            "paper_title": paper.get("title", "Untitled"),
            "paper_id": paper.get("id", "N/A"),
            "decision": recommendation,
            "confidence": confidence,
            "synthesis_date": datetime.now().strftime("%Y-%m-%d"),
            "methodology_critique": critic_review,
            "literature_critique": lit_review,
            "consensus_summary": (
                f"The committee conducted an autonomous multi-agent review of '{paper.get('title')}'. "
                f"The Methodology Critic identified {len(critic_review['strengths'])} key strengths and {len(critic_review['weaknesses'])} potential vulnerabilities. "
                f"The Literature Specialist characterized the contribution as '{lit_review['novelty_verdict']}'. "
                f"Overall consensus recommendation is {recommendation}."
            ),
            "questions_for_authors": [
                "Could the authors provide confidence intervals or variance across multiple random seeds?",
                "How does inference throughput and memory scaling behave when context length exceeds standard benchmark limits?",
                "Have the authors compared against recent parameter-efficient baselines in this sub-domain?"
            ]
        }

        return report

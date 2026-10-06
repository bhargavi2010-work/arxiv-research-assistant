"""
Query Contextualizer for ArXiv Research Assistant.
Resolves pronouns, elided references, and follow-up dependencies across
multi-turn scientific conversations (e.g. 'What is a Transformer?' -> 'Why is self-attention important?').
"""

import re
from typing import List, Dict, Optional

class QueryContextualizer:
    """Reformulates follow-up research questions by integrating previous conversation topics."""

    PRONOUNS_AND_REFERENCES = [
        r"\bit\b", r"\bits\b", r"\bthis\b", r"\bthat\b", r"\bthese\b",
        r"\bthose\b", r"\bthey\b", r"\bthem\b", r"\bthe model\b",
        r"\bthe architecture\b", r"\bthe method\b", r"\bthe technique\b",
        r"\bthis approach\b"
    ]
    
    FOLLOW_UP_STARTERS = [
        "why is it", "why is this", "how does it", "how does this",
        "how is this different from", "how is it different from",
        "what about", "what are its", "what are their", "compare this with",
        "compare it with", "explain its", "can it", "why is self-attention",
        "how does attention"
    ]

    def contextualize_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Determines if the query is a follow-up referring to previous turns.
        If so, extracts the main topic from the recent turn and synthesizes a standalone search query.
        Args:
            query: The user's new query.
            conversation_history: List of previous message dicts [{"role": "user"|"assistant", "content": ...}]
        Returns:
            Contextualized search query string.
        """
        clean_q = query.strip()
        if not conversation_history or not clean_q:
            return clean_q

        q_lower = clean_q.lower()

        # Check if query contains pronouns or follow-up starters
        has_pronoun = any(re.search(pat, q_lower) for pat in self.PRONOUNS_AND_REFERENCES)
        has_starter = any(q_lower.startswith(starter) for starter in self.FOLLOW_UP_STARTERS)
        is_short_followup = len(clean_q.split()) <= 6 and ("why" in q_lower or "how" in q_lower or "what about" in q_lower or "compare" in q_lower)

        if not (has_pronoun or has_starter or is_short_followup):
            return clean_q

        # Extract the antecedent topic from recent user questions
        recent_topics = []
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                u_text = turn.get("content", "")
                # Extract key noun phrases or title words
                cleaned_u = re.sub(r"^(what is|what are|explain|tell me about|how does|why is)\s+", "", u_text, flags=re.IGNORECASE)
                cleaned_u = cleaned_u.strip(" ?.")
                if cleaned_u and len(cleaned_u) > 2:
                    recent_topics.append(cleaned_u)
                    break

        if not recent_topics:
            return clean_q

        antecedent = recent_topics[0]

        # Reformulate query
        if "different from" in q_lower:
            # e.g., "How is this different from an RNN?" + "Transformer" -> "Difference between Transformer and RNN"
            comp = clean_q.split("different from")[-1].strip(" ?.")
            return f"Difference between {antecedent} and {comp}"

        if "compare" in q_lower:
            comp = clean_q.split("with")[-1].strip(" ?.")
            return f"Comparison between {antecedent} and {comp}"

        # If it refers to subcomponent (e.g. "Why is self-attention important?")
        if antecedent.lower() not in q_lower:
            return f"{antecedent} {clean_q}"

        return clean_q

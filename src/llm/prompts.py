"""
Prompt Templates for ArXiv Research Assistant.
Enforces academic rigor, source citations ([1], [2]), hallucination prevention,
and clear separation between grounded context and general technical explanation.
"""

from typing import List, Dict, Any, Optional

RESEARCH_QA_SYSTEM_PROMPT = """You are the ArXiv Research Assistant, an expert AI research scientist in Computer Science, Machine Learning, and Artificial Intelligence.

Your job is to answer the user's research questions accurately, comprehensively, and strictly based on the provided scientific papers.

CRITICAL INSTRUCTIONS:
1. Grounding: Rely primarily on the facts, methodologies, and findings presented in the [Retrieved Research Papers].
2. Citation: Whenever making a claim derived from a paper, cite it using its source number, e.g., [1], [2].
3. No Hallucination: Do NOT invent benchmark scores, equations, or author names. If the provided papers do not contain enough information to answer a specific question, state:
   "Based on the retrieved research context, there is insufficient evidence to address this aspect fully."
4. Structure & Clarity: Organize your response with clear headings, bullet points, and concise technical explanations.
5. Distinction: Clearly distinguish between direct findings reported in the research papers versus broader foundational concepts.
"""

def build_research_qa_prompt(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Assembles the complete grounded prompt for the open-source LLM.
    """
    history_str = ""
    if conversation_history:
        # Include recent conversation turns for follow-up resolution
        formatted_turns = []
        for turn in conversation_history[-4:]:  # Keep last 2-4 turns for memory
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            formatted_turns.append(f"{role}: {content}")
        if formatted_turns:
            history_str = "Recent Conversation History:\n" + "\n".join(formatted_turns) + "\n\n"

    prompt = f"""{RESEARCH_QA_SYSTEM_PROMPT}

{history_str}[Retrieved Research Papers]
{context}

User Question: {query}

Provide a comprehensive, technically accurate, research-grounded answer:"""

    return prompt

def build_concept_explanation_prompt(concept: str, context: str) -> str:
    """Prompt template for technical concept breakdown."""
    return f"""{RESEARCH_QA_SYSTEM_PROMPT}

[Retrieved Research Papers]
{context}

Concept to Explain: {concept}

Please provide a structured technical explanation containing:
1. Definition: Concise, formal definition of {concept}.
2. How It Works: Core technical mechanics, components, and equations/algorithms.
3. Why It Is Useful: Practical advantages, limitations it solves, and impact on AI.
4. Concrete Example: An intuitive example illustrating its application.
5. Key References: Mention the key papers from the context introducing or developing this concept."""

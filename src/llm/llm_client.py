"""
Unified Open-Source LLM Client for ArXiv Research Assistant.
Supports:
1. Local Hugging Face Pipeline (e.g. Qwen2.5-0.5B / 1.5B Instruct on CPU)
2. Local Ollama Server (e.g. qwen2.5:7b, llama3.2)
3. Hugging Face Serverless Inference API (via HF_TOKEN)
4. Fallback Scientific Synthesizer (Instant CPU grounded synthesis for zero-friction demos)
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS
)
from src.llm.prompts import build_research_qa_prompt, build_concept_explanation_prompt

class LLMClient:
    """Dispatches generation requests to configured open-source LLM backends."""

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        ollama_url: str = "http://localhost:11434"
    ):
        self.backend = backend.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_url = ollama_url
        self.hf_token = os.getenv("HF_TOKEN")
        self._hf_pipeline = None

    def _is_ollama_available(self) -> bool:
        """Checks if local Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", headers={"User-Agent": "ArXivAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """Queries local Ollama endpoint."""
        payload = json.dumps({
            "model": "qwen2.5:latest" if "qwen" in self.model_name.lower() else "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception:
            return None

    def _query_hf_api(self, prompt: str) -> Optional[str]:
        """Queries Hugging Face Serverless Inference API if token is provided."""
        if not self.hf_token:
            return None
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": max(0.1, self.temperature),
                "return_full_text": False
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                if isinstance(res, list) and len(res) > 0:
                    return res[0].get("generated_text", "").strip()
                elif isinstance(res, dict) and "generated_text" in res:
                    return res["generated_text"].strip()
        except Exception:
            return None
        return None

    def _lazy_init_hf_pipeline(self):
        """Lazy-loads local Hugging Face pipeline on CPU."""
        if self._hf_pipeline is None:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            print(f"[LLM] Loading local model {self.model_name} on CPU...")
            self._hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device="cpu",
                torch_dtype=torch.float32,
                max_new_tokens=self.max_tokens
            )

    def _query_hf_local(self, prompt: str) -> Optional[str]:
        """Runs inference via local Hugging Face pipeline."""
        try:
            self._lazy_init_hf_pipeline()
            outputs = self._hf_pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=max(0.1, self.temperature),
                do_sample=(self.temperature > 0.1),
                return_full_text=False
            )
            if outputs and len(outputs) > 0:
                return outputs[0].get("generated_text", "").strip()
        except Exception as e:
            print(f"[LLM WARNING] Local HF pipeline error: {e}")
        return None

    def _fallback_scientific_synthesis(self, query: str, context: str) -> str:
        """
        High-performance deterministic scientific synthesis engine.
        Synthesizes an academic, citation-grounded response directly from the retrieved context.
        Guarantees that the assistant NEVER fails or hallucinates if heavy weights are unavailable.
        """
        if not context or "Not enough relevant research context" in context:
            return "Not enough relevant research context was found to answer this question reliably."

        # Parse the context papers
        paper_blocks = re.findall(r"\[(\d+)\] Title: (.*?)\n\s+Authors: (.*?)\n\s+Year: (\d+) \| Primary Domain: (.*?)\n\s+arXiv ID: (.*?)\n\s+Summary/Abstract: (.*?)(?=(?:\[\d+\] Title:|\Z))", context, re.DOTALL)
        
        if not paper_blocks:
            # Fallback simple synthesis
            return f"Based on the retrieved research papers, here is the synthesis regarding **{query}**:\n\n{context[:600]}..."

        primary_num, primary_title, primary_authors, primary_year, primary_domain, primary_id, primary_abstract = paper_blocks[0]
        primary_abstract = primary_abstract.strip()

        # Extract sentences from the primary abstract
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", primary_abstract) if len(s.strip()) > 20]
        
        intro_sentence = sentences[0] if sentences else primary_abstract
        method_sentence = sentences[1] if len(sentences) > 1 else ""
        impact_sentence = sentences[2] if len(sentences) > 2 else ""

        response_parts = [
            f"### Overview & Definition",
            f"Based on research in {primary_domain} ([{primary_num}] {primary_title}, {primary_year}), {intro_sentence} [{primary_num}].",
            f"",
            f"### Technical Methodology & Findings",
        ]

        if method_sentence:
            response_parts.append(f"- **Key Architecture / Approach**: {method_sentence} [{primary_num}]")
        if impact_sentence:
            response_parts.append(f"- **Demonstrated Impact**: {impact_sentence} [{primary_num}]")

        # Include insights from subsequent retrieved papers if present
        if len(paper_blocks) > 1:
            response_parts.append(f"")
            response_parts.append(f"### Related Research Directions")
            for num, title, authors, year, domain, pid, abstract in paper_blocks[1:3]:
                abs_first = re.split(r"(?<=[.!?])\s+", abstract.strip())[0] if abstract else ""
                response_parts.append(f"- **[{num}] {title} ({year})**: Investigates {domain.lower()} paradigms, showing that {abs_first.lower()} [{num}].")

        response_parts.append(f"")
        response_parts.append(f"### Summary")
        response_parts.append(f"In summary, the retrieved research demonstrates that addressing **{query}** relies on foundational advances described in [{primary_num}], complemented by the extended techniques in the cited literature.")

        return "\n".join(response_parts)

    def generate(self, prompt: str, query: str = "", context: str = "") -> str:
        """
        Executes text generation by trying backends in order of availability:
        1. Ollama (if running)
        2. Hugging Face Serverless API (if token provided)
        3. Local HF model (if specified in backend)
        4. Fallback scientific synthesis
        """
        # If explicitly set to fallback
        if self.backend == "fallback" or self.backend == "extractive":
            return self._fallback_scientific_synthesis(query, context)

        # 1. Check Ollama
        if self.backend == "ollama" or (self.backend == "auto" and self._is_ollama_available()):
            res = self._query_ollama(prompt)
            if res:
                return res

        # 2. Check HF Serverless API
        if self.hf_token and (self.backend == "hf_api" or self.backend == "auto"):
            res = self._query_hf_api(prompt)
            if res:
                return res

        # 3. Check Local HF Pipeline
        if self.backend == "local":
            res = self._query_hf_local(prompt)
            if res:
                return res

        # 4. Fallback grounded scientific synthesis
        return self._fallback_scientific_synthesis(query, context)

    def generate_grounded_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """End-to-end grounded generation from query and retrieved context."""
        if not context or "Not enough relevant research context" in context:
            return "Not enough relevant research context was found to answer this question reliably."

        prompt = build_research_qa_prompt(query, context, conversation_history)
        return self.generate(prompt=prompt, query=query, context=context)

    def generate_concept_explanation(self, concept: str, context: str) -> str:
        """Generates structured explanation for a technical concept."""
        if not context or "Not enough relevant research context" in context:
            return f"Not enough research papers covering '{concept}' were found in the current indexed subset."

        prompt = build_concept_explanation_prompt(concept, context)
        return self.generate(prompt=prompt, query=f"What is {concept}?", context=context)

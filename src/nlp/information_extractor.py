"""
Scientific Information Extraction Engine for ArXiv Research Papers.
Uses rule-based patterns, scientific entity gazetteers, and NLP keyphrase algorithms
to extract models, methods, tasks, datasets, and concepts without hallucination.
"""

import re
from typing import Dict, List, Any, Set
from collections import Counter

# Scientific Entity Taxonomies for AI / ML with flexible pattern matching
KNOWN_MODELS = [
    ("Transformer", r"\btransformers?\b"),
    ("BERT", r"\bbert\b"),
    ("RoBERTa", r"\broberta\b"),
    ("GPT-2", r"\bgpt-?2\b"),
    ("GPT-3", r"\bgpt-?3\b"),
    ("GPT-4", r"\bgpt-?4\b"),
    ("ChatGPT", r"\bchatgpt\b"),
    ("LLaMA", r"\bllama\b"),
    ("Qwen", r"\bqwen\b"),
    ("Mistral", r"\bmistral\b"),
    ("Phi", r"\bphi-?[1234]?\b"),
    ("T5", r"\bt5\b"),
    ("BART", r"\bbart\b"),
    ("ResNet", r"\bresnets?\b"),
    ("Vision Transformer", r"\b(vision\s+transformers?|vit)\b"),
    ("Diffusion Model", r"\b(diffusion\s+models?|latent\s+diffusion|stable\s+diffusion)\b"),
    ("GAN", r"\b(gans?|generative\s+adversarial\s+networks?)\b"),
    ("VAE", r"\b(vaes?|variational\s+autoencoders?)\b"),
    ("Autoencoder", r"\bautoencoders?\b"),
    ("RNN", r"\b(rnns?|recurrent\s+neural\s+networks?)\b"),
    ("LSTM", r"\blstms?\b"),
    ("CNN", r"\b(cnns?|convolutional\s+neural\s+networks?|convnets?)\b"),
    ("Graph Neural Network", r"\b(gnns?|graph\s+neural\s+networks?)\b"),
    ("MoE", r"\b(moe|mixture\s+of\s+experts)\b")
]

KNOWN_METHODS = [
    ("Self-Attention", r"\b(self-attention|scaled\s+dot-product\s+attention)\b"),
    ("Attention Mechanism", r"\battention\s+mechanisms?\b"),
    ("Multi-Head Attention", r"\bmulti-head\s+(self-)?attention\b"),
    ("Cross-Attention", r"\bcross-attention\b"),
    ("Positional Encoding", r"\bpositional\s+encodings?\b"),
    ("Residual Connection", r"\bresidual\s+connections?\b"),
    ("Layer Normalization", r"\blayer\s+norm(alization)?\b"),
    ("Pre-training", r"\bpre-train(ed|ing)?\b"),
    ("Fine-tuning", r"\bfine-tun(ed|ing)\b"),
    ("Prompt Tuning", r"\bprompt\s+tuning\b"),
    ("LoRA", r"\b(lora|low-rank\s+adaptation)\b"),
    ("RLHF", r"\b(rlhf|reinforcement\s+learning\s+from\s+human\s+feedback)\b"),
    ("Retrieval-Augmented Generation", r"\b(rag|retrieval-augmented\s+generation)\b"),
    ("Dense Retrieval", r"\bdense\s+(vector\s+)?(retriev\w+|index\w*)\b"),
    ("Contrastive Learning", r"\bcontrastive\s+learning\b"),
    ("Quantization", r"\bquantiz(ation|ed)\b"),
    ("Pruning", r"\bpruning\b"),
    ("Knowledge Distillation", r"\bknowledge\s+distillation\b"),
    ("Few-Shot Learning", r"\bfew-shot\s+(learn\w+|prompt\w*)\b"),
    ("Zero-Shot Learning", r"\bzero-shot\s+(learn\w+|prompt\w*)\b")
]

KNOWN_TASKS = [
    ("Machine Translation", r"\bmachine\s+translation\b"),
    ("Question Answering", r"\bquestion\s+answering\b"),
    ("Text Generation", r"\b(text|language)\s+generation\b"),
    ("Summarization", r"\bsummariz(ation|ing)\b"),
    ("Information Extraction", r"\binformation\s+extraction\b"),
    ("Language Modeling", r"\blanguage\s+model(ing|s)?\b"),
    ("Image Classification", r"\bimage\s+classification\b"),
    ("Object Detection", r"\bobject\s+detection\b"),
    ("Semantic Segmentation", r"\bsemantic\s+segmentation\b"),
    ("Information Retrieval", r"\binformation\s+retrieval\b"),
    ("Code Generation", r"\bcode\s+generation\b"),
    ("Mathematical Reasoning", r"\b(mathematical|multi-step)\s+reasoning\b")
]

KNOWN_DATASETS_BENCHMARKS = [
    ("WMT", r"\bwmt(\s*20\d\d)?\b"),
    ("GLUE", r"\b(super)?glue\b"),
    ("SQuAD", r"\bsquad\b"),
    ("Wikipedia", r"\bwikipedia\b"),
    ("Common Crawl", r"\bcommon\s+crawl\b"),
    ("ImageNet", r"\bimagenet\b"),
    ("CIFAR-10", r"\bcifar-?10\b"),
    ("MS COCO", r"\b(ms\s+)?coco\b"),
    ("MMLU", r"\bmmlu\b"),
    ("GSM8K", r"\bgsm8k\b"),
    ("arXiv", r"\barxiv\b")
]

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "paper", "presents", "propose", "show", "using",
    "new", "models", "results", "work", "state-of-the-art", "based", "approach"
}

class ScientificInfoExtractor:
    """Extracts research concepts, models, methods, tasks, and datasets from scientific papers."""

    def __init__(self):
        # Compile case-insensitive word-boundary regex patterns for fast matching
        self._model_patterns = [(m, re.compile(pat, re.IGNORECASE)) for m, pat in KNOWN_MODELS]
        self._method_patterns = [(m, re.compile(pat, re.IGNORECASE)) for m, pat in KNOWN_METHODS]
        self._task_patterns = [(t, re.compile(pat, re.IGNORECASE)) for t, pat in KNOWN_TASKS]
        self._dataset_patterns = [(d, re.compile(pat, re.IGNORECASE)) for d, pat in KNOWN_DATASETS_BENCHMARKS]

    def _match_entities(self, text: str, patterns: List[tuple]) -> List[str]:
        matched = []
        for canonical_name, pattern in patterns:
            if pattern.search(text):
                matched.append(canonical_name)
        return matched

    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """Extracts key scientific phrases and significant words using frequency & heuristic scoring."""
        # Find 2-word scientific phrases (bigrams)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 3]
        
        # Word counts
        word_counts = Counter(filtered_words)
        
        # Bigrams
        bigrams = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in STOPWORDS and w2 not in STOPWORDS and len(w1) > 2 and len(w2) > 2:
                bigrams.append(f"{w1.title()} {w2.title()}")
        
        bigram_counts = Counter(bigrams)
        
        # Combine top bigrams and top keywords
        top_phrases = [phrase for phrase, count in bigram_counts.most_common(5)]
        top_words = [w.title() for w, count in word_counts.most_common(max_keywords)]
        
        combined = []
        seen = set()
        for item in top_phrases + top_words:
            if item.lower() not in seen:
                seen.add(item.lower())
                combined.append(item)
                if len(combined) >= max_keywords:
                    break
        return combined

    def extract_all(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        Runs comprehensive scientific information extraction across paper title and abstract.
        Returns:
            Dict containing models, methods, tasks, datasets, concepts, and keywords.
        """
        full_text = f"{title}. {abstract}"
        
        models = self._match_entities(full_text, self._model_patterns)
        methods = self._match_entities(full_text, self._method_patterns)
        tasks = self._match_entities(full_text, self._task_patterns)
        datasets = self._match_entities(full_text, self._dataset_patterns)
        keywords = self.extract_keywords(full_text, max_keywords=8)
        
        # High-level key concepts union
        concepts = list(dict.fromkeys(models + methods + tasks))[:8]

        return {
            "models": models,
            "methods": methods,
            "tasks": tasks,
            "datasets": datasets,
            "keywords": keywords,
            "concepts": concepts
        }

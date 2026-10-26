"""
Concept Relationship Graph & Visualization Engine for ArXiv Research Assistant.
Constructs semantic relationship graphs using NetworkX and renders interactive Plotly figures.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.nlp.information_extractor import ScientificInfoExtractor

# Curated reference taxonomies for foundational architectures
CURATED_CONCEPT_KNOWLEDGE = {
    "Transformer": {
        "definition": "The Transformer is a neural network architecture based solely on attention mechanisms, replacing recurrence and convolutions for sequence modeling.",
        "primary_components": ["Attention", "Encoder", "Decoder", "Positional Encoding", "Feed Forward Network"],
        "sub_components": {
            "Attention": ["Self-Attention", "Multi-Head Attention", "Scaled Dot-Product"],
            "Encoder": ["Layer Normalization", "Residual Connection"],
            "Decoder": ["Masked Self-Attention", "Cross-Attention"],
            "Positional Encoding": ["Sinusoidal Embeddings", "Learnable Embeddings"]
        },
        "related_concepts": ["BERT", "GPT", "Self-Attention", "Multi-Head Attention", "Encoder", "Decoder", "Positional Encoding"]
    },
    "RAG": {
        "definition": "Retrieval-Augmented Generation (RAG) is a framework that combines external information retrieval with parametric language generation to reduce hallucination and incorporate up-to-date knowledge.",
        "primary_components": ["Dense Retriever", "Vector Database", "Generator / LLM", "Query Rewriter", "Reranker"],
        "sub_components": {
            "Dense Retriever": ["Bi-Encoder", "Embedding Model", "Dense Passage Retrieval"],
            "Vector Database": ["HNSW Index", "Cosine Similarity", "ChromaDB"],
            "Generator / LLM": ["Context Augmentation", "Instruction Tuning", "Hallucination Mitigation"],
            "Reranker": ["Cross-Encoder", "BM25 Hybrid Ranking"]
        },
        "related_concepts": ["Retrieval-Augmented Generation", "Vector Search", "ChromaDB", "Dense Retrieval", "Hallucination Mitigation"]
    },
    "LLM": {
        "definition": "Large Language Models (LLMs) are autoregressive Transformer models parameterized with billions of weights trained on extensive text corpora for diverse NLP and reasoning tasks.",
        "primary_components": ["Pre-training", "Instruction Tuning", "Alignment", "Inference Optimization", "Context Window"],
        "sub_components": {
            "Pre-training": ["Causal Language Modeling", "Tokenization", "Scaling Laws"],
            "Alignment": ["RLHF", "DPO", "PPO", "Safety Filtering"],
            "Instruction Tuning": ["Prompt Engineering", "Chain of Thought", "Few-Shot Learning"],
            "Inference Optimization": ["KV Cache", "Quantization", "FlashAttention", "Speculative Decoding"]
        },
        "related_concepts": ["Transformer", "GPT-4", "Prompt Engineering", "Fine-Tuning", "Quantization"]
    },
    "CNN": {
        "definition": "Convolutional Neural Networks (CNNs) are specialized deep neural architectures employing convolutional filters with spatial invariance for visual perception and image processing.",
        "primary_components": ["Convolution Layer", "Pooling Layer", "Activation Function", "Residual Block", "Fully Connected Layer"],
        "sub_components": {
            "Convolution Layer": ["Filter Kernels", "Stride", "Padding", "Dilated Convolution"],
            "Residual Block": ["Skip Connection", "Identity Mapping", "Vanishing Gradient Mitigation"],
            "Pooling Layer": ["Max Pooling", "Global Average Pooling"]
        },
        "related_concepts": ["ResNet", "Computer Vision", "Object Detection", "Image Classification"]
    },
    "Reinforcement Learning": {
        "definition": "Reinforcement Learning (RL) is a machine learning paradigm where an autonomous agent learns optimal decision policies through environmental trial-and-error interactions and scalar reward signals.",
        "primary_components": ["Agent", "Environment", "Policy", "Value Function", "Reward Signal"],
        "sub_components": {
            "Policy": ["Actor-Critic", "PPO", "Policy Gradient", "DQN"],
            "Value Function": ["Bellman Equation", "Temporal Difference", "Q-Learning"],
            "Reward Signal": ["Reward Modeling", "Human Preferences", "Exploration vs Exploitation"]
        },
        "related_concepts": ["RLHF", "Policy Gradient", "Q-Learning", "Markov Decision Process"]
    }
}

class ConceptVisualizer:
    """Builds and visualizes scientific concept graphs dynamically derived from indexed papers."""

    def __init__(self, search_engine=None, extractor=None):
        self.search_engine = search_engine
        self.extractor = extractor or ScientificInfoExtractor()

    def get_concept_data(
        self,
        concept_query: str,
        search_engine=None
    ) -> Dict[str, Any]:
        """
        Derives structured definition, components, and real related paper IDs for a concept
        dynamically from indexed papers in ChromaDB.
        """
        active_search = search_engine or self.search_engine
        query_clean = concept_query.strip()
        query_lower = query_clean.lower()

        # 1. Dynamically retrieve matching research papers from the vector store
        retrieved_papers = []
        if active_search is not None:
            try:
                res = active_search.search(query=query_clean, limit=5)
                retrieved_papers = res.get("results", [])
            except Exception:
                retrieved_papers = []

        paper_ids = [str(p.get("id")) for p in retrieved_papers if p.get("id")]

        # 2. Extract entities, methods, and tasks directly from the retrieved literature
        extracted_models = []
        extracted_methods = []
        extracted_tasks = []
        extracted_datasets = []
        extracted_keywords = []

        for p in retrieved_papers:
            ext = self.extractor.extract_all(p.get("title", ""), p.get("abstract", ""))
            extracted_models.extend(ext["models"])
            extracted_methods.extend(ext["methods"])
            extracted_tasks.extend(ext["tasks"])
            extracted_datasets.extend(ext["datasets"])
            extracted_keywords.extend(ext["keywords"])

        extracted_models = list(dict.fromkeys(extracted_models))
        extracted_methods = list(dict.fromkeys(extracted_methods))
        extracted_tasks = list(dict.fromkeys(extracted_tasks))
        extracted_datasets = list(dict.fromkeys(extracted_datasets))

        # 3. Check if concept matches curated taxonomy
        curated_match = None
        for key, data in CURATED_CONCEPT_KNOWLEDGE.items():
            if key.lower() == query_lower or (len(query_lower) >= 3 and key.lower() in query_lower):
                curated_match = (key, data)
                break

        if curated_match:
            key, data = curated_match
            # Augment curated components with dynamic entities from retrieved papers
            related_union = list(dict.fromkeys(data["related_concepts"] + extracted_models + extracted_methods))
            return {
                "name": key,
                "definition": data["definition"],
                "primary_components": data["primary_components"],
                "sub_components": data["sub_components"],
                "related_concepts": related_union[:8],
                "paper_ids": paper_ids or ["Indexed in Corpus"]
            }

        # 4. Purely dynamic derivation from indexed ChromaDB papers for open-vocabulary concepts
        if retrieved_papers:
            lead_paper = retrieved_papers[0]
            lead_title = lead_paper.get("title", query_clean.title())
            lead_abstract = lead_paper.get("abstract", "")
            first_sent = re.split(r"(?<=[.!?])\s+", lead_abstract)[0] if lead_abstract else ""

            dynamic_def = (
                f"{first_sent} (Grounded in indexed literature: '{lead_title}')."
                if first_sent
                else f"An advanced research concept in computer science and artificial intelligence explored in '{lead_title}'."
            )

            # Build components dynamically from actual extracted literature entities
            primary_components = []
            sub_components = {}

            if extracted_methods:
                primary_components.append("Methods & Mechanisms")
                sub_components["Methods & Mechanisms"] = extracted_methods[:3]

            if extracted_models:
                other_models = [m for m in extracted_models if m.lower() != query_lower][:3]
                if other_models:
                    primary_components.append("Related Architectures")
                    sub_components["Related Architectures"] = other_models

            if extracted_tasks:
                primary_components.append("Target Applications")
                sub_components["Target Applications"] = extracted_tasks[:3]

            if extracted_datasets:
                primary_components.append("Evaluation Benchmarks")
                sub_components["Evaluation Benchmarks"] = extracted_datasets[:3]

            # If fewer than 2 components, derive from keywords
            if len(primary_components) < 2 and extracted_keywords:
                primary_components.append("Core Concepts")
                sub_components["Core Concepts"] = extracted_keywords[:4]

            related_concepts = list(dict.fromkeys(extracted_models + extracted_methods + extracted_tasks + extracted_keywords))[:8]

            return {
                "name": query_clean.title(),
                "definition": dynamic_def,
                "primary_components": primary_components or ["Algorithmic Design", "Empirical Evaluation"],
                "sub_components": sub_components or {
                    "Algorithmic Design": extracted_keywords[:2] or ["Mathematical Formulation"],
                    "Empirical Evaluation": ["Experimental Benchmarking"]
                },
                "related_concepts": related_concepts or ["Computer Science", "Artificial Intelligence"],
                "paper_ids": paper_ids
            }

        # 5. Open-domain fallback when no papers are in the local vector DB
        name_title = query_clean.title()
        return {
            "name": name_title,
            "definition": f"{name_title} is a specialized topic in modern artificial intelligence and computer science research.",
            "primary_components": ["Algorithmic Design", "Empirical Evaluation"],
            "sub_components": {
                "Algorithmic Design": ["Mathematical Formulation", "Architecture Implementation"],
                "Empirical Evaluation": ["Benchmark Metrics", "Ablation Studies"]
            },
            "related_concepts": ["Machine Learning", "Deep Learning", "Neural Networks"],
            "paper_ids": []
        }

    def build_networkx_graph(self, concept_data: Dict[str, Any]) -> nx.Graph:
        """Builds a NetworkX graph with node levels and categorical attributes matching Panel 6."""
        G = nx.Graph()
        center_node = concept_data["name"]

        # Color palette matching reference Panel 6
        component_colors = {
            "Attention": "#3b82f6",          # Blue
            "Encoder": "#22c55e",            # Green
            "Decoder": "#f97316",            # Orange
            "Positional Encoding": "#ec4899",# Pink
            "Feed Forward Network": "#06b6d4",# Cyan
            "Dense Retriever": "#3b82f6",
            "Vector Database": "#06b6d4",
            "Generator / LLM": "#8b5cf6",
            "Reranker": "#f97316"
        }
        
        # Center node (Level 0 - Purple/Violet)
        G.add_node(center_node, level=0, node_type="center", size=38, color="#8b5cf6", title=center_node)

        # Primary components (Level 1)
        for comp in concept_data.get("primary_components", []):
            color = component_colors.get(comp, "#3b82f6")
            G.add_node(comp, level=1, node_type="primary", size=26, color=color, title=comp)
            G.add_edge(center_node, comp, weight=2)

            # Sub-components (Level 2)
            subs = concept_data.get("sub_components", {}).get(comp, [])
            for sub in subs:
                G.add_node(sub, level=2, node_type="sub", size=20, color="#93c5fd", title=sub)
                G.add_edge(comp, sub, weight=1)

        return G

    def create_plotly_figure(self, concept_data: Dict[str, Any]) -> go.Figure:
        """Generates an interactive Plotly node-link visualization matching the reference UI."""
        G = self.build_networkx_graph(concept_data)
        
        # Compute layout using spring layout with good repulsion
        pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)

        # Build edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.8, color="#cbd5e1"),
            hoverinfo="none",
            mode="lines"
        )

        # Build node trace with individual node colors
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [n for n in G.nodes()]
        node_colors = [G.nodes[n].get("color", "#2563eb") for n in G.nodes()]
        node_sizes = [G.nodes[n].get("size", 24) for n in G.nodes()]
        hover_text = [f"<b>{n}</b>" for n in G.nodes()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=12, family="Inter, sans-serif", color="#1e293b"),
            hoverinfo="text",
            hovertext=hover_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color="#ffffff"),
                opacity=0.95
            ),
            showlegend=False
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=30),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                height=480
            )
        )
        return fig

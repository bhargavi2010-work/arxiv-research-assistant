"""
Concept Relationship Graph & Visualization Engine for ArXiv Research Assistant.
Constructs semantic relationship graphs using NetworkX and renders interactive Plotly figures.
"""

from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Academic domain knowledge bases for seminal AI/ML concepts
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
        "related_concepts": ["BERT", "GPT", "Self-Attention", "Multi-Head Attention", "Encoder", "Decoder", "Positional Encoding"],
        "primary_papers": ["1706.03762", "1810.04805", "2005.14165"]
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
        "related_concepts": ["Retrieval-Augmented Generation", "Vector Search", "ChromaDB", "Dense Retrieval", "Hallucination Mitigation"],
        "primary_papers": ["2005.11401", "2312.10997"]
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
        "related_concepts": ["Transformer", "GPT-4", "Prompt Engineering", "Fine-Tuning", "Quantization"],
        "primary_papers": ["2005.14165", "2303.08774"]
    },
    "CNN": {
        "definition": "Convolutional Neural Networks (CNNs) are specialized deep neural architectures employing convolutional filters with spatial invariance for visual perception and image processing.",
        "primary_components": ["Convolution Layer", "Pooling Layer", "Activation Function", "Residual Block", "Fully Connected Layer"],
        "sub_components": {
            "Convolution Layer": ["Filter Kernels", "Stride", "Padding", "Dilated Convolution"],
            "Residual Block": ["Skip Connection", "Identity Mapping", "Vanishing Gradient Mitigation"],
            "Pooling Layer": ["Max Pooling", "Global Average Pooling"]
        },
        "related_concepts": ["ResNet", "Computer Vision", "Object Detection", "Image Classification"],
        "primary_papers": ["1512.03385"]
    },
    "Reinforcement Learning": {
        "definition": "Reinforcement Learning (RL) is a machine learning paradigm where an autonomous agent learns optimal decision policies through environmental trial-and-error interactions and scalar reward signals.",
        "primary_components": ["Agent", "Environment", "Policy", "Value Function", "Reward Signal"],
        "sub_components": {
            "Policy": ["Actor-Critic", "PPO", "Policy Gradient", "DQN"],
            "Value Function": ["Bellman Equation", "Temporal Difference", "Q-Learning"],
            "Reward Signal": ["Reward Modeling", "Human Preferences", "Exploration vs Exploitation"]
        },
        "related_concepts": ["RLHF", "Policy Gradient", "Q-Learning", "Markov Decision Process"],
        "primary_papers": ["2005.14165", "2303.08774"]
    }
}

class ConceptVisualizer:
    """Builds and visualizes scientific concept graphs with Plotly."""

    def __init__(self):
        pass

    def get_concept_data(self, concept_query: str) -> Dict[str, Any]:
        """Retrieves structured definition, components, and related papers for a concept."""
        query_clean = concept_query.strip()
        query_lower = query_clean.lower()

        # Find best match in curated taxonomy
        for key, data in CURATED_CONCEPT_KNOWLEDGE.items():
            if key.lower() in query_lower or query_lower in key.lower():
                return {
                    "name": key,
                    "definition": data["definition"],
                    "primary_components": data["primary_components"],
                    "sub_components": data["sub_components"],
                    "related_concepts": data["related_concepts"],
                    "paper_ids": data["primary_papers"]
                }

        # Dynamic fallback for arbitrary user-entered concepts
        name_title = query_clean.title()
        return {
            "name": name_title,
            "definition": f"{name_title} is an important concept in modern computer science and artificial intelligence research.",
            "primary_components": ["Core Methodology", "Model Architecture", "Optimization", "Evaluation"],
            "sub_components": {
                "Core Methodology": ["Mathematical Formulation", "Algorithmic Design"],
                "Model Architecture": ["Input Representation", "Feature Extraction"],
                "Optimization": ["Loss Function", "Gradient Optimization"],
                "Evaluation": ["Benchmark Metrics", "Empirical Validation"]
            },
            "related_concepts": ["Machine Learning", "Deep Learning", "Neural Networks", "Optimization"],
            "paper_ids": ["1706.03762", "2005.11401"]
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

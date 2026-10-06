# ArXiv Research Assistant
### AI-Powered Domain-Specific Scientific Research Chatbot

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests Passing](https://img.shields.io/badge/Tests-42%20Passed-brightgreen.svg)]()

> **MSc Artificial Intelligence Academic & Industry Project**  
> An end-to-end, retrieval-grounded scientific research assistant tailored for Computer Science, Machine Learning, and Artificial Intelligence research papers from arXiv.

---

## 1. Project Overview & Problem Statement

Academic research in Artificial Intelligence and Machine Learning is growing at an exponential rate, with hundreds of papers uploaded to arXiv daily. Researchers, graduate students, and engineers face severe challenges:
1. **Information Overload**: Finding relevant papers using simple keyword searches returns thousands of unranked or non-specific results.
2. **LLM Hallucinations**: Off-the-shelf generative LLMs frequently hallucinate experimental results, fabricate author citations, and lack access to the latest research.
3. **Complex Mathematical Context**: Understanding foundational and contemporary concepts (e.g., Transformers, RAG, LoRA, Contrastive Learning) requires navigating inter-concept dependencies and structural hierarchies.

**ArXiv Research Assistant** solves this by establishing a production-grade **Retrieval-Augmented Generation (RAG)** architecture grounded strictly in verified arXiv paper metadata, abstracts, and taxonomy classifications.

---

## 2. Academic Requirement Mapping

| MSc AI Requirement | System Implementation | Module Reference |
| :--- | :--- | :--- |
| **Domain-Specific Chatbot** | Computer Science & AI Research Assistant | [`src/retrieval/rag_pipeline.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/retrieval/rag_pipeline.py) |
| **arXiv Dataset Usage** | Filtered Cornell arXiv CS/AI/ML dataset & API fetcher | [`src/data_pipeline/`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/data_pipeline/) |
| **Advanced NLP Techniques** | LaTeX stripping, entity gazetteers, TF-IDF keyphrases | [`src/nlp/information_extractor.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/nlp/information_extractor.py) |
| **Scientific Summarization** | 4-section structured summarizer (Problem, Approach, Contributions, Results) | [`src/summarizer/paper_summarizer.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/summarizer/paper_summarizer.py) |
| **Open-Source LLM** | Unified LLM client supporting local CPU pipelines, Ollama, and HF Serverless | [`src/llm/llm_client.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/llm/llm_client.py) |
| **Complex Queries** | Semantic vector search, hybrid keyword boost, grounded QA | [`src/retrieval/search_engine.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/retrieval/search_engine.py) |
| **Concept Explanation** | 5-stage concept breakdown with research citations | [`src/llm/prompts.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/llm/prompts.py) |
| **Follow-Up Questions** | Pronoun resolver & sliding-window multi-turn memory | [`src/retrieval/query_contextualizer.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/retrieval/query_contextualizer.py) |
| **Paper Searching** | Dedicated search engine with category and year filters | [`src/ui/views/search.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/ui/views/search.py) |
| **Concept Visualization** | NetworkX and Plotly interactive node-link relationship graphs | [`src/concept_graph/concept_visualizer.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/concept_graph/concept_visualizer.py) |
| **Streamlit Interface** | Custom CSS-styled, modern 9-view academic interface | [`app.py`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/app.py) & [`src/ui/`](file:///d:/Elevanceskills/ArXiv%20Research%20Assistant/src/ui/) |

---

## 3. System Architecture

```
                               ┌────────────────────────────────────────────────────────┐
                               │          arXiv Metadata (Kaggle JSONL / API)           │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │     Data Cleaning, Ingestion & Domain Filtering        │
                               │  (cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, cs.RO, stat.ML)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │    Embedding Model: sentence-transformers/all-MiniLM   │
                               │       (384-dimensional dense vectors, L2-normalized)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │             ChromaDB Vector Database                   │
                               │  (Persistent HNSW Index with Cosine Distance Space)    │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
             ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
             ▼                                             ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐             ┌───────────────────────────────┐
│     Semantic Paper Search     │             │     Multi-Turn RAG Chatbot    │             │   Concept Explorer & Graph    │
│ - Category & Year Constraints │             │ - Query Contextualizer        │             │ - Curated & Dynamic Hierarchy │
│ - Cosine Relevance Ranking    │             │ - Prompt Grounding Assembly   │             │ - NetworkX Spring Layout      │
│ - Direct arXiv & PDF Links    │             │ - Open-Source LLM Synthesis   │             │ - Interactive Plotly Nodes    │
└──────────────┬────────────────┘             │ - Numbered In-Text Citations  │             └──────────────┬────────────────┘
               │                              └───────────────┬───────────────┘                            │
               └──────────────────────────────────────────────┼────────────────────────────────────────────┘
                                                              │
                                                              ▼
                               ┌────────────────────────────────────────────────────────┐
                               │             Streamlit User Interface (UI/UX)           │
                               │   - Navy Sidebar (#0f172a) & Light Grey Canvas         │
                               │   - 9 Distinct Views Matching Reference Design         │
                               │   - Empty, Error, and Loading State Handling           │
                               └────────────────────────────────────────────────────────┘
```

---

## 4. Key Pipeline Components

### 4.1 Data Cleaning & Domain Filtering (`src/data_pipeline/`)
* **Streaming Generator**: Reads multi-gigabyte raw JSONL files line-by-line with $O(1)$ memory usage.
* **Domain Filtering**: Selects records with primary or secondary categories matching `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.NE`, `cs.RO`, or `stat.ML`.
* **LaTeX Normalization**: Cleans `\cite{}`, `\textbf{}`, `\emph{}`, math environments, and curly braces while preserving semantic meaning.
* **Record Validation**: Eliminates withdrawn submissions (`"This paper has been withdrawn..."`), stubs (< 50 characters), and duplicates.
* **Optimized Storage**: Exports cleaned records into high-performance Apache Parquet (`data/processed/arxiv_cs_ai_cleaned.parquet`).

### 4.2 Dense Embeddings & ChromaDB Vector Store (`src/embeddings/`, `src/vector_store/`)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80 MB download size, fast CPU inference).
* **L2 Unit Normalization**: Ensures that dot product directly equals cosine similarity ($\cos(\theta) = \vec{u} \cdot \vec{v}$).
* **Persistent ChromaDB**: Persistent HNSW index stored in `chromadb_store/` using cosine distance space.
* **Idempotent Ingestion**: Skips existing paper IDs; multiple indexing passes will never re-embed or duplicate documents.
* **Honest Metrics**: Reports mathematically exact `retrieval_similarity = 1.0 - cosine_distance` without fake percentages.

### 4.3 Grounded RAG Chatbot & Follow-up Context (`src/retrieval/`, `src/llm/`)
* **Contextual Query Resolution**: Translates follow-up queries (*"Why is self-attention important in it?"*) into standalone search strings (*"Transformer self-attention importance"*).
* **Relevance Guardrail**: Enforces minimum similarity threshold ($0.35$). If insufficient context is found, it refuses to hallucinate and outputs:
  > *"Not enough relevant research context was found to answer this question reliably."*
* **Multi-Turn Memory**: Preserves chat sessions with sliding-window truncation to keep prompt tokens within safe limits.
* **Follow-up Suggestions**: Generates 2 to 3 contextual follow-up questions for every response.

### 4.4 Scientific Summarizer & Information Extraction (`src/summarizer/`, `src/nlp/`)
* **Structured 4-Section Summary**:
  1. **Problem**: Pinpoints the research limitation in prior work.
  2. **Approach**: Explains architectural and mathematical formulations.
  3. **Key Contributions**: Highlights bullet-point contributions.
  4. **Results**: Summarizes empirical gains on benchmarks.
* **Entity Extraction**: Identifies ML Models, Methods, Tasks, and Datasets with morphological variation handling (singular/plural, tenses).

### 4.5 Interactive Concept Visualization (`src/concept_graph/`)
* Builds hierarchical node-link diagrams linking core concepts (e.g. *Transformer* $\rightarrow$ *Attention* $\rightarrow$ *Self-Attention*, *Multi-Head Attention*, *Encoder*, *Decoder*, *Positional Encoding*).
* Interactive hover tooltips, categorical color-coding, and links to landmark papers.

---

## 5. Project Directory Structure

```
ArXiv Research Assistant/
├── config/
│   ├── __init__.py
│   └── config.py                     # Central configuration (paths, taxonomy, models)
├── data/
│   ├── raw/                          # Raw arXiv metadata (Kaggle JSONL / API sample)
│   └── processed/                    # Cleaned Parquet and JSON datasets
├── chromadb_store/                   # Persistent ChromaDB vector database
├── storage_data/                     # Local persistent JSON storage (chat history, saved papers)
├── src/
│   ├── __init__.py
│   ├── data_pipeline/                # Text cleaner, domain filter, preprocessor
│   │   ├── text_cleaner.py
│   │   ├── domain_filter.py
│   │   └── preprocessor.py
│   ├── embeddings/                   # Sentence-Transformers dense embedding generator
│   │   └── embedder.py
│   ├── vector_store/                 # ChromaDB collection indexer & search manager
│   │   └── chroma_manager.py
│   ├── retrieval/                    # Search engine, context builder, RAG pipeline, contextualizer
│   │   ├── search_engine.py
│   │   ├── context_builder.py
│   │   ├── query_contextualizer.py
│   │   └── rag_pipeline.py
│   ├── llm/                          # Unified LLM client & research prompts
│   │   ├── prompts.py
│   │   └── llm_client.py
│   ├── summarizer/                   # Structured academic paper summarizer
│   │   └── paper_summarizer.py
│   ├── nlp/                          # Scientific information & entity extraction
│   │   └── information_extractor.py
│   ├── concept_graph/                # Concept relationship graph & Plotly visualizer
│   │   └── concept_visualizer.py
│   ├── storage/                      # Chat session & saved papers managers
│   │   ├── chat_memory.py
│   │   └── saved_papers.py
│   └── ui/                           # Streamlit UI design system & view components
│       ├── styles.py                 # Custom CSS (Navy sidebar, white cards, pills)
│       ├── components.py             # Reusable UI elements (cards, empty & error states)
│       └── views/                    # Views: Home, Chat, Search, Details, Summarizer, Concepts, etc.
│           ├── home.py
│           ├── chat.py
│           ├── search.py
│           ├── details.py
│           ├── summarizer.py
│           ├── concepts.py
│           ├── history.py
│           ├── saved.py
│           └── settings.py
├── scripts/
│   ├── inspect_dataset.py            # Schema inspector & seed data generator
│   ├── fetch_arxiv_subset.py         # Live arXiv API fetcher utility
│   └── process_arxiv.py              # End-to-end data processing & vector indexing
├── tests/                            # Automated test suite (42 unit tests)
│   ├── test_data_pipeline.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   ├── test_llm.py
│   ├── test_rag_pipeline.py
│   ├── test_conversation.py
│   ├── test_nlp.py
│   ├── test_summarizer.py
│   └── test_concept_visualizer.py
├── app.py                            # Main Streamlit application entry point
├── requirements.txt                  # Pinned Python dependencies
└── README.md                         # Project documentation
```

---

## 6. Installation & Quickstart

### Prerequisites
* Python 3.11 (Recommended for PyTorch and ChromaDB prebuilt wheels on Windows)
* 4 GB+ RAM

### 1. Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd "d:/Elevanceskills/ArXiv Research Assistant"

# Create a virtual environment using Python 3.11
py -3.11 -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
# Install PyTorch CPU first (saves gigabytes of CUDA binaries)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
pip install -r requirements.txt
```

### 3. (Optional) Fetch More Live arXiv Papers
The project comes with verified reference papers (Attention, RAG, BERT, GPT-3, LoRA, ResNet). To fetch additional live papers across CS/AI categories:
```bash
python scripts/fetch_arxiv_subset.py --max_papers 100
```
*(Or drop your Kaggle `arxiv-metadata-oai-snapshot.json` directly into `data/raw/`)*.

### 4. Run Data Preprocessing & Vector Indexing
```bash
python scripts/process_arxiv.py
```

### 5. Launch the Streamlit Research Assistant
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 7. Running Automated Tests

Run the full automated test suite covering all 14 project phases:
```bash
python -m unittest discover tests
```
Output:
```text
Ran 42 tests in 47.056s
OK
```

---

## 8. Limitations & Future Improvements

### Current Scope & Limitations
1. **Abstract-Level Grounding**: The dataset indexes paper titles, authors, categories, and abstracts. Full-text PDF body extraction is intentionally out of scope to preserve sub-second CPU latency on standard development hardware.
2. **Offline Local Model**: Local CPU inference uses lightweight open-source models (0.5B to 1.5B parameters) or deterministic extractive synthesis. Larger 70B models require external GPU servers or Hugging Face API keys.

### Future Roadmap
1. **Hybrid Sparse-Dense Retrieval**: Ingest BM25 alongside dense ChromaDB vectors for reciprocal rank fusion (RRF).
2. **Full-PDF Ingestion Pipeline**: Implement PyMuPDF/Grobid chunking for tables, equations, and full experimental results.
3. **Graph-RAG Integration**: Integrate Neo4j knowledge graphs with paper citation networks.

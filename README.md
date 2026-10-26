# 📚 ArXiv Research Assistant
> An AI-powered research assistant for exploring, summarizing, and chatting with Computer Science and AI research papers from arXiv.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Agent-LangChain_ReAct-orange.svg)](https://www.langchain.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-1C3C3C.svg)](https://smith.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests Passing](https://img.shields.io/badge/Tests-50%20Passed-brightgreen.svg)]()

---

## 💡 About the Project

Staying up to date with new AI research is tough with hundreds of papers uploaded to arXiv every week. 
This project is an interactive **Retrieval-Augmented Generation (RAG)** assistant built to help students and researchers:
- Ask questions and get grounded answers with direct citations (`[1]`, `[2]`).
- Learn research concepts and terms (like *Transformers*, *RAG*, *LoRA*, *Overfitting*, *Hallucinations*) even if they aren't in the dataset.
- Search and filter papers by domain (e.g. `cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`) and publication year.
- Get structured paper summaries (Problem, Method, Key Contributions, Results).
- Explore interactive concept maps to understand how different AI ideas connect.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **LangChain ReAct Agent** | Autonomous ReAct (*Reason + Act*) agent using structured tool use (`@tool`) to search papers, fetch live preprints, summarize abstracts, and build concept maps. |
| **LangSmith Observability** | Complete execution tracing, tool latency telemetry, and run logging via LangSmith. |
| **Research Chat (RAG)** | Ask questions about papers or AI concepts; get answers with citations and follow-up ideas. |
| **Universal Knowledge Base** | Ask about research methodology (*"How to start research?"*, *"What is an ablation study?"*) or any ML/DL terms. |
| **Dynamic Knowledge Expansion** | One-click sync with live arXiv Atom API preprints into ChromaDB vector store. |
| **Paper Search & Filtering** | Semantic and keyword search across arXiv categories and publication years. |
| **Structured Summarizer** | Breaks down any paper abstract into 4 key sections: Problem, Approach, Contributions, and Results. |
| **Concept Graph Explorer** | Interactive NetworkX + Plotly relationship graphs for visual learning. |
| **Chat History & Saved Papers** | Save important papers and revisit previous conversation threads. |

---

## 🛠️ Tech Stack

- **Agent Framework**: LangChain, LangGraph (ReAct agent loop, tool calling)
- **Observability**: LangSmith (run tracing & telemetry)
- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: ChromaDB (Cosine similarity search)
- **Data & Processing**: Python 3.11, Pandas, PyArrow, NumPy
- **Visualizations**: Plotly, NetworkX

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/bhargavi2010-work/arxiv-research-assistant.git
cd arxiv-research-assistant

# Create Python 3.11 virtual environment
py -3.11 -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
# Install PyTorch CPU first (saves time and download size)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project requirements
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
arxiv-research-assistant/
├── app.py                # Main Streamlit application
├── config/               # Settings, taxonomy, LangSmith config & paths
├── data/                 # Raw and processed arXiv datasets
├── chromadb_store/       # Persistent ChromaDB vector store
├── storage_data/         # Chat history and saved papers
├── src/
│   ├── agents/           # LangChain ReAct Agent & tool definitions
│   ├── data_pipeline/    # Data cleaning & filtering
│   ├── embeddings/       # Sentence-transformers embedder
│   ├── vector_store/     # ChromaDB manager
│   ├── retrieval/        # Hybrid search & RAG pipeline
│   ├── llm/              # LLM client & research knowledge base 
│   ├── summarizer/       # Paper summarizer
│   ├── nlp/              # Scientific keyphrase extractor
│   ├── concept_graph/    # Interactive concept visualizer
│   └── ui/               # Streamlit views and CSS styles
└── tests/                # Automated unit tests (50 tests)
```

---

## 🧪 Running Tests

To verify that all components are working properly:

```bash
python -m unittest discover tests
```

---

## 📄 License
This project is open-source and created for educational and research purposes.

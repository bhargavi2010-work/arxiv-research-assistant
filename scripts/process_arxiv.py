"""
Reproducible Data Processing & Vector Indexing Pipeline for ArXiv Research Assistant.

Usage:
    python scripts/process_arxiv.py [--limit 5000] [--raw_path path/to/arxiv.json] [--skip_indexing]

Pipeline Steps:
1. Loads raw arXiv data (streaming JSONL).
2. Domain filtering (cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, cs.RO, stat.ML).
3. Text cleaning & normalization (LaTeX stripping, whitespace, authors).
4. Validation & duplicate removal.
5. Saves structured output to data/processed/ (Parquet & JSON).
6. Generates dense embeddings using sentence-transformers (all-MiniLM-L6-v2).
7. Persistently indexes documents and metadata in ChromaDB.
8. Performs verification retrieval.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline.preprocessor import ArXivPreprocessor
from config.config import RAW_DATA_DIR, RAW_DATASET_FILENAME, DEFAULT_EMBEDDING_MODEL

def main():
    parser = argparse.ArgumentParser(description="Process and index arXiv research papers.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of domain papers to process (useful for development subsets)."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=None,
        help="Optional path to custom raw arXiv JSONL file."
    )
    parser.add_argument(
        "--skip_indexing",
        action="store_true",
        help="Skip embedding generation and ChromaDB indexing."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding and ChromaDB insertion batch size."
    )
    args = parser.parse_args()

    raw_file = Path(args.raw_path) if args.raw_path else (RAW_DATA_DIR / RAW_DATASET_FILENAME)
    
    if not raw_file.exists():
        print(f"[ERROR] Raw dataset file not found at: {raw_file}")
        print("Please place 'arxiv-metadata-oai-snapshot.json' in 'data/raw/' or run 'python scripts/fetch_arxiv_subset.py'.")
        sys.exit(1)

    print("=" * 65)
    print("  ArXiv Research Assistant - Data Pipeline & Vector Indexing")
    print("=" * 65)
    
    # Step 1: Preprocess raw records
    preprocessor = ArXivPreprocessor(raw_filepath=raw_file)
    stats = preprocessor.run(max_records=args.limit)
    
    print("\n" + "=" * 65)
    print("  Preprocessing Summary")
    print("=" * 65)
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    if args.skip_indexing:
        print("\n[NOTE] --skip_indexing passed. Skipping embedding generation and ChromaDB indexing.")
        return

    # Step 2: Vector Embedding & ChromaDB Indexing
    print("\n" + "=" * 65)
    print("  Vector Embedding & ChromaDB Indexing (Phase 3)")
    print("=" * 65)
    
    from src.embeddings.embedder import EmbeddingGenerator
    from src.vector_store.chroma_manager import ChromaManager
    
    df = preprocessor.load_processed_data()
    papers = df.to_dict(orient="records")
    
    embedder = EmbeddingGenerator(model_name=DEFAULT_EMBEDDING_MODEL, device="cpu")
    chroma_mgr = ChromaManager()
    
    added_count = chroma_mgr.index_papers(papers, embedder=embedder, batch_size=args.batch_size)
    print(f"[INFO] Total papers now in ChromaDB collection: {chroma_mgr.count()}")
    
    # Step 3: Verification Retrieval Query
    test_query = "What is the Transformer architecture and attention mechanism?"
    print(f"\n[VERIFICATION] Running sample query: '{test_query}'")
    q_vec = embedder.embed_query(test_query)
    results = chroma_mgr.query_similar(q_vec, top_k=3)
    
    print(f"[VERIFICATION] Retrieved {len(results)} matches:")
    for idx, r in enumerate(results, 1):
        print(f"  {idx}. [{r['id']}] {r['title']} (Year: {r['year']}, Category: {r['primary_category']})")
        print(f"     Retrieval Similarity: {r['retrieval_similarity']:.4f} | Cosine Dist: {r['distance']:.4f}")
    
    print("\n" + "=" * 65)
    print("  Phase 3 Indexing & Verification Complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()

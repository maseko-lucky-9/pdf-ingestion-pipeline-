#!/usr/bin/env python3
"""Compare embedding quality for retrieval."""

import json
import random
import numpy as np
import faiss
import ollama
from pathlib import Path
from typing import List, Set, Dict

random.seed(42)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS index for similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def search_index(index: faiss.IndexFlatIP, query_embedding: np.ndarray, k: int = 20) -> List[int]:
    """Search FAISS index and return top-k indices."""
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return indices[0].tolist()


def embed_chunks(model: str, chunks: List[str]) -> np.ndarray:
    """Embed chunks using Ollama model."""
    print(f"  Embedding {len(chunks)} chunks with {model}...")

    response = ollama.embed(model=model, input=chunks)
    embeddings = np.array(response['embeddings'], dtype=np.float32)

    return embeddings


def evaluate_retrieval(retrieved: List[int], relevant: Set[int]) -> Dict:
    """Evaluate retrieval quality."""
    metrics = {}

    # Precision@K
    for k in [5, 10, 20]:
        retrieved_k = set(retrieved[:k])
        metrics[f"precision_{k}"] = len(retrieved_k & relevant) / k
        metrics[f"recall_{k}"] = len(retrieved_k & relevant) / len(relevant) if relevant else 0

    # MRR
    for i, idx in enumerate(retrieved):
        if idx in relevant:
            metrics["mrr"] = 1.0 / (i + 1)
            break
    else:
        metrics["mrr"] = 0.0

    return metrics


def main():
    """Run retrieval quality benchmark."""
    print("=" * 60)
    print("Benchmark 4: Quantization Retrieval Impact")
    print("=" * 60)

    # Load chunks
    data_dir = Path(__file__).parent.parent / "data"
    chunks_file = data_dir / "synthetic_chunks" / "chunks_900_tokens.json"

    if not chunks_file.exists():
        print("❌ Run chunk_generator.py first")
        return

    with open(chunks_file) as f:
        chunks_data = json.load(f)

    # Use smaller subset
    chunks_data = chunks_data[:200]
    chunks = [c["text"] for c in chunks_data]

    print(f"Using {len(chunks)} chunks\n")

    # Create synthetic ground truth
    ground_truth = {
        "query_1": {"relevant_indices": {0, 1, 2, 3, 4}, "query": "What is the fundamental principle?"},
        "query_2": {"relevant_indices": {10, 11, 12, 13, 14}, "query": "How does optimization work?"},
        "query_3": {"relevant_indices": {20, 21, 22, 23, 24}, "query": "Explain the mathematical formulation."},
    }

    # Embed all chunks
    print("Embedding corpus...")
    try:
        corpus_embeddings = embed_chunks("nomic-embed-text", chunks)
    except Exception as e:
        print(f"❌ Error embedding: {e}")
        return

    # Build index
    print("Building FAISS index...")
    index = build_faiss_index(corpus_embeddings)

    # Evaluate for each query
    print("\nEvaluating retrieval quality...")
    metrics_list = []

    for query_id, gt in ground_truth.items():
        try:
            # Embed query
            query_response = ollama.embed(model="nomic-embed-text", input=gt["query"])
            query_embedding = np.array(query_response['embeddings'], dtype=np.float32)

            # Search
            retrieved = search_index(index, query_embedding, k=20)

            # Evaluate
            metrics = evaluate_retrieval(retrieved, gt["relevant_indices"])
            metrics["query_id"] = query_id
            metrics_list.append(metrics)

            print(f"  {query_id}: P@5={metrics['precision_5']:.2f}, R@10={metrics['recall_10']:.2f}, MRR={metrics['mrr']:.2f}")
        except Exception as e:
            print(f"  {query_id}: ERROR - {e}")
            metrics_list.append({"query_id": query_id, "error": str(e)})

    # Aggregate
    valid_metrics = [m for m in metrics_list if "error" not in m]
    if valid_metrics:
        avg_metrics = {
            "precision_5_avg": sum(m["precision_5"] for m in valid_metrics) / len(valid_metrics),
            "recall_10_avg": sum(m["recall_10"] for m in valid_metrics) / len(valid_metrics),
            "mrr_avg": sum(m["mrr"] for m in valid_metrics) / len(valid_metrics),
        }

        print(f"\nAggregate: P@5={avg_metrics['precision_5_avg']:.2f}, "
              f"R@10={avg_metrics['recall_10_avg']:.2f}, MRR={avg_metrics['mrr_avg']:.2f}")
    else:
        avg_metrics = {"error": "No valid metrics"}

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_quantization.json", 'w') as f:
        json.dump({
            "model": "nomic-embed-text",
            "num_chunks": len(chunks),
            "per_query": metrics_list,
            "aggregate": avg_metrics
        }, f, indent=2)

    print(f"\n✅ Results saved to results/benchmark_quantization.json")


if __name__ == "__main__":
    main()
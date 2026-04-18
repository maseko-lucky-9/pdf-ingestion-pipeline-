#!/usr/bin/env python3
"""Retrieval quality benchmark with topically-coherent ground truth.

Compares Ollama nomic-embed-text vs native MLX bge-small on the same corpus.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
import ollama

from mlx_embeddings import generate, load


def build_index(emb: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


def search(idx: faiss.IndexFlatIP, q: np.ndarray, k: int = 20) -> List[int]:
    faiss.normalize_L2(q)
    _, inds = idx.search(q, k)
    return inds[0].tolist()


def retrieval_metrics(retrieved: List[int], relevant: Set[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in (5, 10, 20):
        rk = set(retrieved[:k])
        out[f"precision_{k}"] = len(rk & relevant) / k
        out[f"recall_{k}"] = len(rk & relevant) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for i, idx in enumerate(retrieved):
        if idx in relevant:
            mrr = 1.0 / (i + 1)
            break
    out["mrr"] = mrr
    return out


def embed_ollama(model: str, texts: List[str]) -> np.ndarray:
    resp = ollama.embed(model=model, input=texts)
    return np.array(resp["embeddings"], dtype=np.float32)


def embed_mlx(model, tokenizer, texts: List[str]) -> np.ndarray:
    out = generate(model, tokenizer, texts=texts)
    # mlx_embeddings returns an object with text_embeds (mx array). Normalize to numpy.
    if hasattr(out, "text_embeds"):
        import mlx.core as mx
        return np.array(mx.stop_gradient(out.text_embeds), dtype=np.float32)
    return np.array(out, dtype=np.float32)


def evaluate_backend(name: str, corpus_emb: np.ndarray, query_emb_for_id: Dict[str, np.ndarray], queries) -> Dict:
    idx = build_index(corpus_emb)
    per_query = []
    for q in queries:
        qemb = query_emb_for_id[q["query_id"]].reshape(1, -1).copy()
        hits = search(idx, qemb, k=20)
        m = retrieval_metrics(hits, set(q["relevant_indices"]))
        m["query_id"] = q["query_id"]
        m["topic"] = q["topic"]
        per_query.append(m)

    def avg(key):
        return sum(x[key] for x in per_query) / len(per_query)

    return {
        "backend": name,
        "per_query": per_query,
        "aggregate": {
            "precision_5_avg": avg("precision_5"),
            "recall_10_avg": avg("recall_10"),
            "mrr_avg": avg("mrr"),
        },
    }


def main():
    print("=" * 60)
    print("Benchmark 4b: Retrieval Quality with Topical Ground Truth")
    print("=" * 60)

    base = Path(__file__).parent.parent
    chunks = json.loads((base / "data" / "synthetic_chunks" / "topical_chunks.json").read_text())
    queries = json.loads((base / "data" / "ground_truth" / "queries.json").read_text())
    texts = [c["text"] for c in chunks]

    print(f"Corpus: {len(texts)} chunks; {len(queries)} queries\n")

    all_results = {}

    # --- Ollama nomic-embed-text ---
    print("[ollama/nomic-embed-text]")
    t0 = time.perf_counter()
    corpus_emb = embed_ollama("nomic-embed-text", texts)
    qemb_map = {q["query_id"]: embed_ollama("nomic-embed-text", [q["query"]])[0] for q in queries}
    dt = time.perf_counter() - t0
    r = evaluate_backend("ollama/nomic-embed-text", corpus_emb, qemb_map, queries)
    r["embed_wall_time_s"] = dt
    for m in r["per_query"]:
        print(f"  {m['query_id']}: P@5={m['precision_5']:.2f} R@10={m['recall_10']:.2f} MRR={m['mrr']:.2f}")
    print(f"  aggregate: P@5={r['aggregate']['precision_5_avg']:.2f} R@10={r['aggregate']['recall_10_avg']:.2f} MRR={r['aggregate']['mrr_avg']:.2f}")
    all_results["ollama_nomic"] = r

    # --- MLX bge-small ---
    print("\n[mlx/bge-small-en-v1.5-bf16]")
    model, tokenizer = load("mlx-community/bge-small-en-v1.5-bf16")
    t0 = time.perf_counter()
    corpus_emb_mlx = embed_mlx(model, tokenizer, texts)
    qemb_map_mlx = {q["query_id"]: embed_mlx(model, tokenizer, [q["query"]])[0] for q in queries}
    dt = time.perf_counter() - t0
    r = evaluate_backend("mlx/bge-small-en-v1.5-bf16", corpus_emb_mlx, qemb_map_mlx, queries)
    r["embed_wall_time_s"] = dt
    for m in r["per_query"]:
        print(f"  {m['query_id']}: P@5={m['precision_5']:.2f} R@10={m['recall_10']:.2f} MRR={m['mrr']:.2f}")
    print(f"  aggregate: P@5={r['aggregate']['precision_5_avg']:.2f} R@10={r['aggregate']['recall_10_avg']:.2f} MRR={r['aggregate']['mrr_avg']:.2f}")
    all_results["mlx_bge_small"] = r

    out = base / "results" / "benchmark_retrieval_v2.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n✅ Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark native MLX embedding throughput vs Ollama baseline."""

import json
import random
import time
import tracemalloc
from pathlib import Path

from mlx_embeddings import generate, load

random.seed(42)


MODELS = [
    "mlx-community/bge-small-en-v1.5-bf16",
    "mlx-community/bge-m3-mlx-fp16",
    "mlx-community/bge-m3-mlx-4bit",
    "mlx-community/mxbai-embed-large-v1",
]


def benchmark_mlx_model(repo_id: str, chunks: list, batch_sizes=(16, 32, 64)):
    """Load model + embed chunks in batches on MLX (Metal)."""
    print(f"\n[{repo_id}]")
    print("  Loading model...")
    try:
        model, tokenizer = load(repo_id)
    except Exception as e:
        print(f"    load error: {e}")
        return [{"model": repo_id, "error": str(e)}]

    # Warm-up — compile + upload weights to GPU
    _ = generate(model, tokenizer, texts=[chunks[0]])

    results = []
    for batch_size in batch_sizes:
        print(f"  batch_size={batch_size}...")
        tracemalloc.start()
        start = time.perf_counter()
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                _ = generate(model, tokenizer, texts=batch)
        except Exception as e:
            tracemalloc.stop()
            results.append({"model": repo_id, "batch_size": batch_size, "error": str(e)})
            continue
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({
            "model": repo_id,
            "batch_size": batch_size,
            "total_chunks": len(chunks),
            "throughput_emb_per_sec": len(chunks) / elapsed,
            "peak_python_mb": peak / 1024 ** 2,
            "total_time_s": elapsed,
        })
        print(f"    {len(chunks)/elapsed:.1f} emb/sec")
    return results


def main():
    print("=" * 60)
    print("Benchmark 1b: Native MLX Embedding Throughput")
    print("=" * 60)

    chunks_file = Path(__file__).parent.parent / "data" / "synthetic_chunks" / "chunks_900_tokens.json"
    if not chunks_file.exists():
        print("❌ Run chunk_generator.py first")
        return

    with open(chunks_file) as f:
        chunks = [c["text"] for c in json.load(f)][:500]
    print(f"Using {len(chunks)} chunks\n")

    all_results = []
    for model in MODELS:
        all_results.extend(benchmark_mlx_model(model, chunks))

    out = Path(__file__).parent.parent / "results" / "benchmark_mlx_embeddings.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "native_mlx_embedding_throughput",
        "num_chunks": len(chunks),
        "results": all_results,
    }, indent=2))
    print(f"\n✅ Results saved to {out}")


if __name__ == "__main__":
    main()

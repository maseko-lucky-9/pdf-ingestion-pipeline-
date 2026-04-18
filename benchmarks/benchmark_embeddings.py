#!/usr/bin/env python3
"""Benchmark embedding throughput on M5 Pro."""

import time
import tracemalloc
import json
import random
import ollama
from pathlib import Path

random.seed(42)


def benchmark_model(model_name: str, chunks: list, batch_sizes: list = [16, 32, 64]):
    """Benchmark embedding throughput for a model."""

    results = []
    texts = [c["text"] for c in chunks]

    for batch_size in batch_sizes:
        print(f"  Testing batch_size={batch_size}...")

        # Warm up
        try:
            ollama.embed(model=model_name, input=texts[0])
        except Exception as e:
            print(f"    ❌ Model error: {e}")
            return results

        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        # Process batches
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = ollama.embed(model=model_name, input=batch)
        except Exception as e:
            print(f"    ❌ Error during batch processing: {e}")
            tracemalloc.stop()
            return results

        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            "model": model_name,
            "batch_size": batch_size,
            "total_chunks": len(texts),
            "throughput_emb_per_sec": len(texts) / elapsed,
            "peak_memory_mb": peak / 1024**2,
            "total_time_s": elapsed
        })

    return results


def main():
    """Run embedding benchmarks."""
    print("=" * 60)
    print("Benchmark 1: Embedding Throughput")
    print("=" * 60)

    # Load chunks
    data_dir = Path(__file__).parent.parent / "data" / "synthetic_chunks"
    chunks_file = data_dir / "chunks_900_tokens.json"

    if not chunks_file.exists():
        print("❌ Run chunk_generator.py first")
        return

    with open(chunks_file) as f:
        chunks = json.load(f)

    # Use subset for faster testing
    chunks = chunks[:500]
    print(f"Using {len(chunks)} chunks for benchmark\n")

    models = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "embeddinggemma",
        "bge-m3"
    ]

    all_results = []

    for model in models:
        print(f"\n[{model}]")
        results = benchmark_model(model, chunks)
        all_results.extend(results)

        for r in results:
            print(f"    {r['throughput_emb_per_sec']:.1f} emb/sec, "
                  f"{r['peak_memory_mb']:.1f}MB peak, "
                  f"batch={r['batch_size']}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_embeddings.json", 'w') as f:
        json.dump({
            "benchmark": "embedding_throughput",
            "num_chunks": len(chunks),
            "results": all_results
        }, f, indent=2)

    print(f"\n✅ Results saved to results/benchmark_embeddings.json")


if __name__ == "__main__":
    main()
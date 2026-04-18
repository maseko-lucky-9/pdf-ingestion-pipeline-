#!/usr/bin/env python3
"""End-to-end pipeline benchmark."""

import time
import json
import tracemalloc
from pathlib import Path
from docling.document_converter import DocumentConverter
import numpy as np
import faiss
import ollama


def benchmark_full_pipeline(pdf_path: str, num_queries: int = 10) -> dict:
    """Run full pipeline: PDF → Chunks → Embed → Index → Query."""

    tracemalloc.start()

    # Phase 1: Extraction
    print("  [1/4] Extracting PDF...")
    t1 = time.perf_counter()
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        print(f"    ❌ Extraction error: {e}")
        return {"error": str(e)}
    t2 = time.perf_counter()
    extraction_time = t2 - t1

    # Get text chunks from document
    chunks = []
    if hasattr(doc, 'texts'):
        for item in doc.texts:
            if hasattr(item, 'text') and item.text.strip():
                chunks.append(item.text)

    if not chunks:
        chunks = ["Sample chunk from document"]

    # Phase 2: Embedding
    print(f"  [2/4] Embedding {len(chunks)} chunks...")
    t3 = time.perf_counter()
    try:
        response = ollama.embed(model="nomic-embed-text", input=chunks)
        embeddings = np.array(response['embeddings'], dtype=np.float32)
    except Exception as e:
        print(f"    ❌ Embedding error: {e}")
        return {"error": str(e)}
    t4 = time.perf_counter()
    embedding_time = t4 - t3

    # Phase 3: Indexing
    print("  [3/4] Building index...")
    t5 = time.perf_counter()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    t6 = time.perf_counter()
    indexing_time = t6 - t5

    # Phase 4: Query
    print(f"  [4/4] Running {num_queries} queries...")
    queries = [
        "What is the main topic?",
        "Explain the key concepts.",
        "Summarize the findings.",
        "What are the conclusions?",
        "List the main points.",
        "Describe the methodology.",
        "What data was used?",
        "How do the results compare?",
        "What are the limitations?",
        "What future work is suggested?"
    ][:num_queries]

    query_latencies = []
    for query in queries:
        t7 = time.perf_counter()
        try:
            q_response = ollama.embed(model="nomic-embed-text", input=query)
            q_embedding = np.array(q_response['embeddings'], dtype=np.float32)
            faiss.normalize_L2(q_embedding)
            distances, indices = index.search(q_embedding, 5)
        except Exception as e:
            print(f"    ⚠️ Query error: {e}")
        t8 = time.perf_counter()
        query_latencies.append(t8 - t7)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    query_latencies_sorted = sorted(query_latencies) if query_latencies else [0]
    p50_idx = len(query_latencies_sorted) // 2

    return {
        "pdf_path": pdf_path,
        "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 0,
        "num_chunks": len(chunks),
        "extraction_time_s": extraction_time,
        "embedding_time_s": embedding_time,
        "indexing_time_s": indexing_time,
        "query_latency_p50_s": query_latencies_sorted[p50_idx] if query_latencies_sorted else 0,
        "query_latency_max_s": max(query_latencies) if query_latencies else 0,
        "peak_memory_mb": peak / 1024**2,
        "total_time_s": extraction_time + embedding_time + indexing_time + sum(query_latencies)
    }


def main():
    """Run E2E benchmark."""
    print("=" * 60)
    print("Benchmark 6: Full Pipeline E2E")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "sample_pdfs"
    test_pdf = data_dir / "mixed_content.pdf"

    if not test_pdf.exists():
        print("❌ Run pdf_generator.py first")
        return

    print(f"\nProcessing: {test_pdf.name}\n")

    result = benchmark_full_pipeline(str(test_pdf))

    if "error" in result:
        print(f"\n❌ Pipeline failed: {result['error']}")
        return

    print("\nResults:")
    print(f"  Pages: {result['num_pages']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Extraction: {result['extraction_time_s']:.2f}s")
    print(f"  Embedding: {result['embedding_time_s']:.2f}s")
    print(f"  Indexing: {result['indexing_time_s']:.3f}s")
    print(f"  Query P50: {result['query_latency_p50_s']*1000:.1f}ms")
    print(f"  Query Max: {result['query_latency_max_s']*1000:.1f}ms")
    print(f"  Peak Memory: {result['peak_memory_mb']:.1f}MB")
    print(f"  Total Time: {result['total_time_s']:.2f}s")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_pipeline.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Results saved to results/benchmark_pipeline.json")


if __name__ == "__main__":
    main()
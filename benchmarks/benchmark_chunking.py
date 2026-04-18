#!/usr/bin/env python3
"""Compare chunking strategies."""

import json
import re
import random
import tiktoken
from pathlib import Path
from typing import List

random.seed(42)


def fixed_chunk(text: str, size: int = 900, tokenizer=None) -> List[str]:
    """Fixed-size chunking by tokens."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), size):
        chunk_tokens = tokens[i:i + size]
        chunks.append(tokenizer.decode(chunk_tokens))

    return chunks


def split_by_structure(text: str) -> List[str]:
    """Split by paragraphs and headers."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def adaptive_chunk(text: str, tokenizer=None, min_size: int = 200, max_size: int = 1200) -> List[str]:
    """Adaptive chunking from arXiv:2603.25333."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Phase 1: Split by structure
    initial_chunks = split_by_structure(text)

    # Phase 2: Merge tiny chunks
    merged = []
    for chunk in initial_chunks:
        tokens = len(tokenizer.encode(chunk))
        if tokens < min_size and merged:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    # Phase 3: Split oversized chunks
    final = []
    for chunk in merged:
        tokens = len(tokenizer.encode(chunk))
        if tokens > max_size:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current = ""
            for sent in sentences:
                if len(tokenizer.encode(current + sent)) < max_size:
                    current += " " + sent
                else:
                    if current:
                        final.append(current.strip())
                    current = sent
            if current:
                final.append(current.strip())
        else:
            final.append(chunk)

    return final


def calculate_chunk_stats(chunks: List[str], tokenizer) -> dict:
    """Calculate statistics for chunks."""
    token_counts = [len(tokenizer.encode(c)) for c in chunks]

    if not token_counts:
        return {"num_chunks": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0, "std_tokens": 0}

    avg = sum(token_counts) / len(token_counts)
    variance = sum((t - avg) ** 2 for t in token_counts) / len(token_counts)

    return {
        "num_chunks": len(chunks),
        "avg_tokens": avg,
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "std_tokens": variance ** 0.5
    }


def main():
    """Run chunking comparison benchmark."""
    print("=" * 60)
    print("Benchmark 5: Adaptive Chunking")
    print("=" * 60)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Load sample text
    data_dir = Path(__file__).parent.parent / "data" / "synthetic_chunks"
    chunks_file = data_dir / "chunks_900_tokens.json"

    if not chunks_file.exists():
        print("❌ Run chunk_generator.py first")
        return

    with open(chunks_file) as f:
        chunks_data = json.load(f)

    # Combine into single document
    sample_text = "\n\n".join(c["text"] for c in chunks_data[:50])

    print(f"Sample document: {len(tokenizer.encode(sample_text))} tokens\n")

    # Compare strategies
    strategies = {
        "fixed_512": fixed_chunk(sample_text, size=512, tokenizer=tokenizer),
        "fixed_900": fixed_chunk(sample_text, size=900, tokenizer=tokenizer),
        "adaptive": adaptive_chunk(sample_text, tokenizer=tokenizer),
    }

    results = {}
    for name, chunks in strategies.items():
        stats = calculate_chunk_stats(chunks, tokenizer)
        results[name] = stats

        print(f"[{name}]")
        print(f"  Chunks: {stats['num_chunks']}")
        print(f"  Avg tokens: {stats['avg_tokens']:.1f}")
        print(f"  Std dev: {stats['std_tokens']:.1f}")
        print()

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_chunking.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to results/benchmark_chunking.json")


if __name__ == "__main__":
    main()
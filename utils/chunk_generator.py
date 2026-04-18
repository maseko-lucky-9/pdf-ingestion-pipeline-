#!/usr/bin/env python3
"""Generate synthetic chunks for embedding benchmarks."""

import json
import random
import tiktoken
from pathlib import Path

random.seed(42)

# Sample content templates
PROSE_TEMPLATES = [
    "The {concept} is a fundamental principle in {domain} that describes how {process} operates under {condition} constraints. When applied to {application}, this principle ensures that {outcome} is achieved through {mechanism}. The mathematical formulation involves multiple parameters and requires numerical methods for practical implementation.",

    "In the context of {domain}, the {concept} provides a framework for understanding complex phenomena. Researchers have demonstrated that this framework holds true across multiple datasets. This has significant implications for industry applications where optimization is a critical factor.",

    "The evolution of {concept} from theoretical foundations to modern applications represents a paradigm shift in {domain}. Early implementations focused on analytical approaches, but modern approaches leverage computational methods to achieve significant improvements in efficiency and accuracy.",
]

TABLE_TEMPLATES = [
    "| Metric | Value | Change |\n|--------|-------|--------|\n| Revenue | $450M | +15% |\n| Margin | 32% | +3pp |\n| Growth | 22% | +8% |",

    "| Parameter | Default | Range | Description |\n|-----------|---------|-------|-------------|\n| alpha | 0.01 | [0.001, 0.1] | Learning rate |\n| beta | 0.9 | [0.8, 0.99] | Momentum |\n| epochs | 100 | [10, 1000] | Training iterations |",
]

FORMULA_TEMPLATES = [
    "The equation f(x) = α/β · e^(-γx) describes the exponential decay process. Where α represents amplitude, β is the baseline value, and γ denotes the rate constant. This formula is essential for probabilistic calculations in statistical modeling.",

    "Using the summation identity Σ(i=1 to n) x_i = n(n+1)/2, we can derive closed-form solutions for series problems. This formula is fundamental for algorithmic complexity analysis and numerical computation.",
]

DOMAINS = ["finance", "engineering", "physics", "economics", "statistics", "machine learning"]
CONCEPTS = ["entropy", "gradient", "variance", "derivative", "integral", "coefficient", "matrix", "vector"]
CONDITIONS = ["boundary", "initial", "steady-state", "equilibrium", "transient"]
APPLICATIONS = ["portfolio optimization", "risk management", "signal processing", "control systems"]
OUTCOMES = ["convergence", "stability", "efficiency", "accuracy", "robustness"]


def generate_prose_chunk() -> str:
    """Generate a prose chunk from templates."""
    template = random.choice(PROSE_TEMPLATES)
    return template.format(
        concept=random.choice(CONCEPTS),
        domain=random.choice(DOMAINS),
        process=random.choice(["optimization", "integration", "transformation", "propagation"]),
        condition=random.choice(CONDITIONS),
        application=random.choice(APPLICATIONS),
        outcome=random.choice(OUTCOMES),
        mechanism=random.choice(["iteration", "recursion", "convolution", "decomposition"]),
    )


def generate_table_chunk() -> str:
    """Generate a table chunk from templates."""
    return random.choice(TABLE_TEMPLATES)


def generate_formula_chunk() -> str:
    """Generate a formula chunk from templates."""
    return random.choice(FORMULA_TEMPLATES)


def generate_chunks(num_chunks: int = 1000, target_tokens: int = 900) -> list:
    """Generate synthetic chunks with mix of content types."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []

    print(f"Generating {num_chunks} chunks targeting {target_tokens} tokens each...")

    for i in range(num_chunks):
        # 70% prose, 20% tables, 10% formulas
        roll = random.random()
        if roll < 0.7:
            chunk = generate_prose_chunk()
            chunk_type = "prose"
        elif roll < 0.9:
            chunk = generate_table_chunk()
            chunk_type = "table"
        else:
            chunk = generate_formula_chunk()
            chunk_type = "formula"

        # Pad or trim to target tokens
        tokens = tokenizer.encode(chunk)
        if len(tokens) < target_tokens:
            # Pad with more content
            while len(tokens) < target_tokens:
                extra = generate_prose_chunk()
                tokens.extend(tokenizer.encode(extra))
            tokens = tokens[:target_tokens]
        else:
            tokens = tokens[:target_tokens]

        chunk = tokenizer.decode(tokens)
        chunks.append({
            "id": f"chunk_{i:04d}",
            "text": chunk,
            "token_count": len(tokens),
            "type": chunk_type
        })

    return chunks


def main():
    """Generate and save chunks."""
    output_dir = Path(__file__).parent.parent / "data" / "synthetic_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = generate_chunks(num_chunks=1000, target_tokens=900)

    output_file = output_dir / "chunks_900_tokens.json"
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Generated {len(chunks)} chunks")
    print(f"   Saved to: {output_file}")

    # Stats
    token_counts = [c["token_count"] for c in chunks]
    print(f"   Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
    print(f"   Types: {sum(1 for c in chunks if c['type']=='prose')} prose, "
          f"{sum(1 for c in chunks if c['type']=='table')} table, "
          f"{sum(1 for c in chunks if c['type']=='formula')} formula")


if __name__ == "__main__":
    main()
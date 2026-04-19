"""Batch-embed Chunks via Ollama (R9: fail-fast if daemon/model absent)."""
from __future__ import annotations

import ollama

from src.config import Config
from src.models.document import Chunk


def embed_chunks(chunks: list[Chunk], cfg: Config) -> list[tuple[Chunk, list[float]]]:
    """Return (chunk, embedding) pairs. Batches by cfg.ollama.batch_size."""
    if not chunks:
        return []

    model = cfg.ollama.embed_model
    batch_size = cfg.ollama.batch_size
    results: list[tuple[Chunk, list[float]]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.content for c in batch]

        try:
            resp = ollama.embed(model=model, input=texts)
            embeddings = resp["embeddings"]
        except Exception as exc:
            raise RuntimeError(f"Embedding failed (model={model}): {exc}") from exc

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}"
            )

        for chunk, emb in zip(batch, embeddings):
            results.append((chunk, emb))

    return results

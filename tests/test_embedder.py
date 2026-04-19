"""Embedder: batch boundary handling + clean error on Ollama unreachable (R12/V12)."""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.models.document import Chunk
from src.pipeline.embedder import embed_chunks


def _make_cfg(batch_size: int = 4) -> Config:
    cfg = Config()
    cfg.ollama.batch_size = batch_size
    return cfg


def _chunk(idx: int) -> Chunk:
    return Chunk(
        docid=f"d{idx:04d}",
        source_pdf="test.pdf",
        chunk_index=idx,
        chunk_type="text",
        content=f"content {idx}",
        page_range=(idx, idx),
        token_count=2,
        collection="test",
    )


def test_batch_boundary_odd_count():
    """11 chunks with batch_size=4 → batches of 4,4,3 — all embedded."""
    chunks = [_chunk(i) for i in range(11)]
    fake_emb = [[0.0] * 768] * 11

    with patch("src.pipeline.embedder.ollama.embed") as mock_embed:
        mock_embed.side_effect = [
            {"embeddings": fake_emb[0:4]},
            {"embeddings": fake_emb[4:8]},
            {"embeddings": fake_emb[8:11]},
        ]
        result = embed_chunks(chunks, _make_cfg(batch_size=4))

    assert len(result) == 11
    assert mock_embed.call_count == 3


def test_ollama_unreachable_raises_runtime_error():
    chunks = [_chunk(0)]
    with patch("src.pipeline.embedder.ollama.embed", side_effect=Exception("connection refused")):
        with pytest.raises(RuntimeError, match="Embedding failed"):
            embed_chunks(chunks, _make_cfg())


def test_empty_chunks_returns_empty():
    result = embed_chunks([], _make_cfg())
    assert result == []

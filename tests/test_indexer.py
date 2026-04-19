"""Indexer: schema correctness + double-ingest idempotency (R4/V10)."""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.config import Config
from src.models.document import Chunk
from src.pipeline.indexer import IndexWriter


def _make_cfg() -> Config:
    return Config()


def _pair(content: str, idx: int = 0) -> tuple[Chunk, list[float]]:
    chunk = Chunk(
        docid=f"docid{idx:04d}",
        source_pdf="test.pdf",
        chunk_index=idx,
        chunk_type="text",
        content=content,
        page_range=(1, 1),
        token_count=len(content.split()),
        collection="test",
    )
    embedding = [0.1] * 768
    return chunk, embedding


def test_schema_has_correct_indexes(tmp_path):
    db_path = tmp_path / "index.db"
    cfg = _make_cfg()
    with IndexWriter(db_path, cfg) as w:
        pass
    conn = sqlite3.connect(str(db_path))
    schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='index'").fetchall()
    schema_text = " ".join(s[0] or "" for s in schema)
    assert "meta_domain" in schema_text
    assert "meta_book" in schema_text
    assert "REFERENCES fts" not in schema_text, "Invalid REFERENCES clause found in index DDL"
    conn.close()


def test_double_ingest_is_idempotent(tmp_path):
    db_path = tmp_path / "index.db"
    cfg = _make_cfg()
    pairs = [_pair(f"chunk content {i}", i) for i in range(5)]

    with IndexWriter(db_path, cfg) as w:
        w.write(pairs)

    with IndexWriter(db_path, cfg) as w:
        w.write(pairs)

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    conn.close()
    assert count == 5, f"Expected 5 rows after double-ingest, got {count}"


def test_vec_table_has_correct_dim(tmp_path):
    db_path = tmp_path / "index.db"
    cfg = _make_cfg()
    pairs = [_pair("hello world", 0)]
    with IndexWriter(db_path, cfg) as w:
        w.write(pairs)
    conn = sqlite3.connect(str(db_path))
    schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='vec'").fetchone()
    conn.close()
    assert "768" in (schema[0] or ""), "Embedding dim 768 not found in vec table DDL"

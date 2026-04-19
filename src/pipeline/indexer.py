"""Write chunks into sqlite-vec + FTS5 index (R2: correct schema, R4: upsert, R11: dim from config)."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Generator

import sqlite_vec

from src.config import Config
from src.models.document import Chunk

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    docid       TEXT PRIMARY KEY,
    source_pdf  TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type  TEXT NOT NULL,
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    collection  TEXT NOT NULL,
    domain      TEXT NOT NULL DEFAULT '',
    book        TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS meta_domain ON meta(domain);
CREATE INDEX IF NOT EXISTS meta_book   ON meta(book);

CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    docid UNINDEXED,
    content,
    tokenize = 'porter ascii'
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(
    docid TEXT PRIMARY KEY,
    embedding FLOAT[{dim}]
);
"""

_META_UPSERT = """
INSERT INTO meta (docid, source_pdf, chunk_index, chunk_type,
                  page_start, page_end, token_count, collection,
                  domain, book, content_hash)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(docid) DO UPDATE SET
    source_pdf   = excluded.source_pdf,
    chunk_index  = excluded.chunk_index,
    chunk_type   = excluded.chunk_type,
    page_start   = excluded.page_start,
    page_end     = excluded.page_end,
    token_count  = excluded.token_count,
    collection   = excluded.collection,
    domain       = excluded.domain,
    book         = excluded.book,
    content_hash = excluded.content_hash
"""

_FTS_INSERT = "INSERT INTO fts(docid, content) VALUES (?, ?)"
_VEC_DELETE = "DELETE FROM vec WHERE docid = ?"
_VEC_INSERT = "INSERT INTO vec(docid, embedding) VALUES (?, ?)"

_DEDUP_CHECK = """
SELECT docid FROM meta WHERE source_pdf = ? AND content_hash = ?
"""


class IndexWriter:
    """Context manager that opens/creates an index.db and writes chunks idempotently."""

    def __init__(self, db_path: Path, cfg: Config) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "IndexWriter":
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        dim = self._cfg.ollama.embed_dim
        self._conn.executescript(_DDL.format(dim=dim))
        self._conn.commit()
        return self

    def __exit__(self, *_) -> None:
        if self._conn:
            self._conn.commit()
            self._conn.close()

    def write(self, pairs: list[tuple[Chunk, list[float]]]) -> None:
        """Insert-or-replace each (chunk, embedding) pair, skipping exact duplicates."""
        assert self._conn is not None
        cur = self._conn.cursor()

        for chunk, embedding in pairs:
            # idempotency: skip if same (source_pdf, content_hash) already indexed
            row = cur.execute(_DEDUP_CHECK, (chunk.source_pdf, chunk.content_hash)).fetchone()
            if row is not None:
                continue

            cur.execute(_META_UPSERT, (
                chunk.docid, chunk.source_pdf, chunk.chunk_index, chunk.chunk_type,
                chunk.page_range[0], chunk.page_range[1], chunk.token_count,
                chunk.collection, chunk.domain, chunk.book, chunk.content_hash,
            ))
            cur.execute(_FTS_INSERT, (chunk.docid, chunk.content))
            cur.execute(_VEC_DELETE, (chunk.docid,))
            cur.execute(_VEC_INSERT, (chunk.docid, json.dumps(embedding)))

        self._conn.commit()

"""RRF-fusion retrieval + cross-encoder rerank (R5: device auto-detection)."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import ollama
import torch

from src.config import Config

_reranker_cache: dict[str, object] = {}


def _get_reranker(model_name: str, device: str):
    key = f"{model_name}:{device}"
    if key not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        _reranker_cache[key] = CrossEncoder(model_name, device=device)
    return _reranker_cache[key]


@dataclass
class SearchResult:
    docid: str
    content: str
    source_pdf: str
    chunk_type: str
    page_range: tuple[int, int]
    score: float


def _resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device_cfg


def _embed_query(query: str, cfg: Config) -> list[float]:
    resp = ollama.embed(model=cfg.ollama.embed_model, input=query)
    return resp["embeddings"][0]


def _vector_search(cur: sqlite3.Cursor, embedding: list[float], k: int) -> list[tuple[str, float]]:
    rows = cur.execute(
        "SELECT docid, distance FROM vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (json.dumps(embedding), k),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _fts5_query(query: str) -> str:
    import re
    words = re.findall(r"\w+", query)
    return " OR ".join(f'"{w}"' for w in words if len(w) > 2) or '""'


def _bm25_search(cur: sqlite3.Cursor, query: str, k: int, domain: str = "") -> list[tuple[str, float]]:
    fts_q = _fts5_query(query)
    if domain:
        rows = cur.execute(
            """
            SELECT f.docid, bm25(fts) AS score
            FROM fts f
            JOIN meta m ON f.docid = m.docid
            WHERE fts MATCH ? AND m.domain = ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_q, domain, k),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT docid, bm25(fts) AS score FROM fts WHERE fts MATCH ? ORDER BY score LIMIT ?",
            (fts_q, k),
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _rrf_fuse(
    bm25_hits: list[tuple[str, float]],
    vec_hits: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (docid, _) in enumerate(bm25_hits):
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (docid, _) in enumerate(vec_hits):
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _fetch_content(cur: sqlite3.Cursor, docids: list[str]) -> dict[str, dict]:
    placeholders = ",".join("?" * len(docids))
    rows = cur.execute(
        f"""
        SELECT m.docid, f.content, m.source_pdf, m.chunk_type, m.page_start, m.page_end
        FROM meta m
        JOIN fts f ON m.docid = f.docid
        WHERE m.docid IN ({placeholders})
        """,
        docids,
    ).fetchall()
    return {
        row[0]: {
            "content": row[1],
            "source_pdf": row[2],
            "chunk_type": row[3],
            "page_range": (row[4], row[5]),
        }
        for row in rows
    }


def retrieve(
    query: str,
    db_path: Path,
    cfg: Config,
    domain: str = "",
) -> list[SearchResult]:
    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()

    embedding = _embed_query(query, cfg)
    vec_hits = _vector_search(cur, embedding, cfg.retrieval.vector_k)
    bm25_hits = _bm25_search(cur, query, cfg.retrieval.bm25_k, domain=domain)

    fused = _rrf_fuse(bm25_hits, vec_hits, k=cfg.retrieval.rrf_k)
    top_docids = [docid for docid, _ in fused[: cfg.retrieval.final_k]]

    content_map = _fetch_content(cur, top_docids)
    conn.close()

    if not top_docids or not content_map:
        return []

    candidates = [
        SearchResult(
            docid=did,
            content=content_map[did]["content"],
            source_pdf=content_map[did]["source_pdf"],
            chunk_type=content_map[did]["chunk_type"],
            page_range=content_map[did]["page_range"],
            score=dict(fused)[did],
        )
        for did in top_docids
        if did in content_map
    ]

    if not cfg.rerank.enabled or not candidates:
        return candidates

    # Cross-encoder rerank (R5: device auto-detection)
    device = _resolve_device(cfg.rerank.device)
    reranker = _get_reranker(cfg.rerank.model, device)
    pairs = [(query, c.content) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates

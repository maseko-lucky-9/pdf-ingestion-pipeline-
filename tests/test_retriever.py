"""Retriever: BM25 + vector search + RRF fusion + cross-encoder rerank."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.pipeline import retriever as R


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    """Ensure reranker cache doesn't leak across tests."""
    R._reranker_cache.clear()
    yield
    R._reranker_cache.clear()


def _make_cfg(rerank_enabled: bool = False) -> Config:
    cfg = Config()
    cfg.rerank.enabled = rerank_enabled
    cfg.rerank.device = "cpu"
    cfg.retrieval.bm25_k = 10
    cfg.retrieval.vector_k = 10
    cfg.retrieval.rrf_k = 60
    cfg.retrieval.final_k = 5
    return cfg


# ---------------------------------------------------------------------------
# _rrf_fuse
# ---------------------------------------------------------------------------

def test_rrf_fuse_merges_and_ranks():
    """Docs appearing in both lists outrank docs appearing in only one."""
    bm25 = [("docA", 1.0), ("docB", 2.0), ("docC", 3.0)]
    vec = [("docB", 0.1), ("docD", 0.2), ("docA", 0.3)]
    fused = R._rrf_fuse(bm25, vec, k=60)
    docs = [d for d, _ in fused]
    # A and B appear in both, so they should rank above C and D.
    assert docs.index("docA") < docs.index("docC")
    assert docs.index("docA") < docs.index("docD")
    assert docs.index("docB") < docs.index("docC")
    assert docs.index("docB") < docs.index("docD")


def test_rrf_fuse_formula():
    """Manual RRF computation: 1/(k+rank+1). docA rank0 in both: 2*(1/61)."""
    bm25 = [("docA", 0.0)]
    vec = [("docA", 0.0)]
    fused = R._rrf_fuse(bm25, vec, k=60)
    assert fused[0][0] == "docA"
    assert fused[0][1] == pytest.approx(2.0 / 61.0)


def test_rrf_fuse_empty_inputs():
    assert R._rrf_fuse([], []) == []
    assert R._rrf_fuse([("docA", 1.0)], []) == [("docA", 1.0 / 61.0)]


# ---------------------------------------------------------------------------
# _fts5_query
# ---------------------------------------------------------------------------

def test_fts5_query_filters_short_words():
    q = R._fts5_query("is a RSI indicator")
    # "is" and "a" have <=2 chars and should be dropped.
    assert '"is"' not in q
    assert '"a"' not in q
    assert '"RSI"' in q
    assert '"indicator"' in q


def test_fts5_query_empty_input_returns_empty_token():
    assert R._fts5_query("  ") == '""'
    assert R._fts5_query("a") == '""'


# ---------------------------------------------------------------------------
# _bm25_search
# ---------------------------------------------------------------------------

def test_bm25_search_without_domain():
    cur = MagicMock()
    cur.execute.return_value.fetchall.return_value = [("docA", -5.0), ("docB", -4.0)]
    out = R._bm25_search(cur, "RSI indicator", 10)
    assert out == [("docA", -5.0), ("docB", -4.0)]
    sql = cur.execute.call_args[0][0]
    assert "JOIN meta" not in sql
    assert "fts MATCH" in sql


def test_bm25_search_with_domain_joins_meta():
    cur = MagicMock()
    cur.execute.return_value.fetchall.return_value = [("docF", -3.0)]
    out = R._bm25_search(cur, "carry trade", 5, domain="forex")
    assert out == [("docF", -3.0)]
    sql = cur.execute.call_args[0][0]
    params = cur.execute.call_args[0][1]
    assert "JOIN meta" in sql
    assert "m.domain = ?" in sql
    assert "forex" in params
    assert 5 in params


# ---------------------------------------------------------------------------
# _vector_search
# ---------------------------------------------------------------------------

def test_vector_search_uses_match_and_orders_by_distance():
    cur = MagicMock()
    cur.execute.return_value.fetchall.return_value = [("docA", 0.1), ("docB", 0.3)]
    out = R._vector_search(cur, [0.0] * 768, 10)
    assert out == [("docA", 0.1), ("docB", 0.3)]
    sql = cur.execute.call_args[0][0]
    assert "embedding MATCH" in sql
    assert "ORDER BY distance" in sql


# ---------------------------------------------------------------------------
# retrieve() — integration with mocks
# ---------------------------------------------------------------------------

def _patch_db(vec_rows, bm25_rows, content_rows):
    """Build a patched sqlite3.connect that returns scripted result sets."""
    cur = MagicMock()
    # Each execute returns a cursor-like object whose fetchall yields the next scripted row set.
    cur.execute.return_value.fetchall.side_effect = [vec_rows, bm25_rows, content_rows]
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.enable_load_extension = MagicMock()
    conn.close = MagicMock()
    return conn


def test_retrieve_returns_results_without_rerank():
    cfg = _make_cfg(rerank_enabled=False)
    conn = _patch_db(
        vec_rows=[("docA", 0.1), ("docB", 0.3)],
        bm25_rows=[("docA", -5.0), ("docC", -4.0)],
        content_rows=[
            ("docA", "content A", "a.pdf", "text", 1, 2),
            ("docB", "content B", "b.pdf", "text", 3, 4),
            ("docC", "content C", "c.pdf", "table", 5, 5),
        ],
    )
    with patch("src.pipeline.retriever.sqlite3.connect", return_value=conn), \
         patch("sqlite_vec.load"), \
         patch("src.pipeline.retriever.ollama.embed", return_value={"embeddings": [[0.0] * 768]}):
        results = R.retrieve("what is RSI", Path("/fake/index.db"), cfg)
    assert len(results) >= 1
    ids = [r.docid for r in results]
    # docA is in both lists → should rank first (highest RRF score).
    assert ids[0] == "docA"
    # Scores should be the RRF fused scores (not overwritten by reranker).
    assert all(r.score > 0 for r in results)


def test_retrieve_with_rerank_reorders_results():
    cfg = _make_cfg(rerank_enabled=True)
    conn = _patch_db(
        vec_rows=[("docA", 0.1), ("docB", 0.3)],
        bm25_rows=[("docA", -5.0), ("docB", -4.0)],
        content_rows=[
            ("docA", "irrelevant content", "a.pdf", "text", 1, 2),
            ("docB", "highly relevant content", "b.pdf", "text", 3, 4),
        ],
    )
    # Rerank flips the order: docB gets the higher score.
    fake_reranker = MagicMock()
    fake_reranker.predict.return_value = [0.1, 0.9]  # docA low, docB high
    with patch("src.pipeline.retriever.sqlite3.connect", return_value=conn), \
         patch("sqlite_vec.load"), \
         patch("src.pipeline.retriever.ollama.embed", return_value={"embeddings": [[0.0] * 768]}), \
         patch("src.pipeline.retriever._get_reranker", return_value=fake_reranker):
        results = R.retrieve("relevant query", Path("/fake/index.db"), cfg)
    assert [r.docid for r in results] == ["docB", "docA"]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.1)


def test_retrieve_handles_empty_index():
    cfg = _make_cfg(rerank_enabled=False)
    conn = _patch_db(vec_rows=[], bm25_rows=[], content_rows=[])
    with patch("src.pipeline.retriever.sqlite3.connect", return_value=conn), \
         patch("sqlite_vec.load"), \
         patch("src.pipeline.retriever.ollama.embed", return_value={"embeddings": [[0.0] * 768]}):
        results = R.retrieve("anything", Path("/fake/index.db"), cfg)
    assert results == []


# ---------------------------------------------------------------------------
# device resolution
# ---------------------------------------------------------------------------

def test_resolve_device_explicit_passthrough():
    assert R._resolve_device("cpu") == "cpu"
    assert R._resolve_device("cuda") == "cuda"
    assert R._resolve_device("mps") == "mps"


def test_resolve_device_auto_falls_back_to_cpu():
    with patch("src.pipeline.retriever.torch.backends.mps.is_available", return_value=False), \
         patch("src.pipeline.retriever.torch.cuda.is_available", return_value=False):
        assert R._resolve_device("auto") == "cpu"


def test_resolve_device_auto_prefers_mps():
    with patch("src.pipeline.retriever.torch.backends.mps.is_available", return_value=True), \
         patch("src.pipeline.retriever.torch.cuda.is_available", return_value=False):
        assert R._resolve_device("auto") == "mps"

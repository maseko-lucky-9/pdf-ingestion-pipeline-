"""Tests for the FastAPI app in `src.api.server`.

All dependencies (retriever, answer synthesis, filesystem) are patched.
No real DB, no real Anthropic calls.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.answer import _DEFAULT_MODEL, AnsweredQuery, Citation
from src.api.server import _list_collections, app
from src.pipeline.retriever import SearchResult


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_answered(answer: str = "Paris is the capital [doc-1].") -> AnsweredQuery:
    return AnsweredQuery(
        answer=answer,
        citations=[
            Citation(
                docid="c-1",
                source_pdf="geo.pdf",
                page_range=(3, 3),
                snippet="Paris is the capital of France.",
                score=0.95,
            )
        ],
        model=_DEFAULT_MODEL,
        prompt_tokens=100,
        completion_tokens=30,
        latency_ms=250,
    )


def _setup_collection(tmp_path: Path, name: str = "test-corpus") -> Path:
    col = tmp_path / "collections" / name
    col.mkdir(parents=True)
    (col / "index.db").touch()
    return tmp_path / "collections"


# ─── GET /health ────────────────────────────────────────────────────────────


def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_shape(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["collections"], list)


def test_health_lists_collections_from_disk(tmp_path: Path, client: TestClient) -> None:
    root = _setup_collection(tmp_path, "my-corpus")
    with patch("src.api.server._collections_dir", return_value=root):
        body = client.get("/health").json()
        assert "my-corpus" in body["collections"]


def test_health_empty_when_collections_dir_missing(tmp_path: Path, client: TestClient) -> None:
    with patch("src.api.server._collections_dir", return_value=tmp_path / "nope"):
        body = client.get("/health").json()
        assert body["collections"] == []


# ─── POST /query — validation ───────────────────────────────────────────────


def test_query_rejects_missing_fields(client: TestClient) -> None:
    assert client.post("/query", json={}).status_code == 422


def test_query_rejects_empty_query_string(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "", "collection": "test"})
    assert resp.status_code == 422


def test_query_rejects_k_below_one(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "x", "collection": "t", "k": 0})
    assert resp.status_code == 422


def test_query_rejects_k_above_cap(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "x", "collection": "t", "k": 999})
    assert resp.status_code == 422


def test_query_returns_404_for_unknown_collection(tmp_path: Path, client: TestClient) -> None:
    with patch("src.api.server._collections_dir", return_value=tmp_path / "nope"):
        resp = client.post("/query", json={"query": "q", "collection": "ghost"})
    assert resp.status_code == 404
    assert "ghost" in resp.json()["detail"]


# ─── POST /query — happy path ───────────────────────────────────────────────


def test_query_happy_path_returns_full_payload(tmp_path: Path, client: TestClient) -> None:
    """End-to-end: retriever returns chunks, synthesize_answer is called with
    those chunks, the API mirrors the payload back to the caller.

    Previously this test mocked `retrieve` to return [] AND mocked
    synthesize_answer. With the real short-circuit-on-empty-retrieval path,
    the test passed even with broken wiring. Now `retrieve` returns a real
    SearchResult and the captured args verify it propagated.
    """
    root = _setup_collection(tmp_path)
    answered = _make_answered()
    chunks = [
        SearchResult(
            docid="d1", content="Paris is the capital of France.",
            source_pdf="geo.pdf", chunk_type="text", page_range=(3, 3), score=0.9,
        )
    ]
    captured: dict = {}

    def fake_synth(query: str, results, **_kwargs):  # type: ignore[no-untyped-def]
        captured["query"] = query
        captured["n_results"] = len(results)
        captured["first_docid"] = results[0].docid if results else None
        return answered

    with (
        patch("src.api.server._collections_dir", return_value=root),
        patch("src.api.server.retrieve", return_value=chunks),
        patch("src.api.server.synthesize_answer", side_effect=fake_synth),
        patch("src.api.server.load_config"),
    ):
        resp = client.post("/query", json={"query": "capital of France?", "collection": "test-corpus"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == answered.answer
    assert body["model"] == _DEFAULT_MODEL
    assert body["latency_ms"] == 250
    assert isinstance(body["citations"], list) and len(body["citations"]) == 1
    assert body["citations"][0]["docid"] == "c-1"
    assert "request_id" in body and len(body["request_id"]) > 0
    # Real-wiring assertion: synthesize_answer received the retriever's output.
    assert captured["query"] == "capital of France?"
    assert captured["n_results"] == 1
    assert captured["first_docid"] == "d1"


def test_query_trims_retriever_results_to_k(tmp_path: Path, client: TestClient) -> None:
    root = _setup_collection(tmp_path)
    answered = _make_answered()
    many = [
        SearchResult(docid=f"d{i}", content="x", source_pdf="x.pdf",
                     chunk_type="text", page_range=(1, 1), score=0.5)
        for i in range(20)
    ]
    captured: dict = {}

    def fake_synth(query: str, results, **_kwargs):  # type: ignore[no-untyped-def]
        captured["n"] = len(results)
        return answered

    with (
        patch("src.api.server._collections_dir", return_value=root),
        patch("src.api.server.retrieve", return_value=many),
        patch("src.api.server.synthesize_answer", side_effect=fake_synth),
        patch("src.api.server.load_config"),
    ):
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus", "k": 3})

    assert resp.status_code == 200
    assert captured["n"] == 3


def test_query_returns_503_when_api_key_missing(tmp_path: Path, client: TestClient) -> None:
    root = _setup_collection(tmp_path)
    with (
        patch("src.api.server._collections_dir", return_value=root),
        patch("src.api.server.retrieve", return_value=[]),
        patch("src.api.server.synthesize_answer", side_effect=EnvironmentError("ANTHROPIC_API_KEY is not set")),
        patch("src.api.server.load_config"),
    ):
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus"})

    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


# ─── POST /query — Anthropic SDK error mapping ──────────────────────────────


def _fake_httpx_response(status_code: int):
    import httpx

    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def _fake_httpx_request():
    import httpx

    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _patch_query_path(root: Path, exc: Exception):
    """Helper: patch the query-path collaborators so `exc` is raised at the
    answer-synthesis step."""
    chunks = [
        SearchResult(docid="d1", content="x", source_pdf="x.pdf",
                     chunk_type="text", page_range=(1, 1), score=0.5)
    ]
    return (
        patch("src.api.server._collections_dir", return_value=root),
        patch("src.api.server.retrieve", return_value=chunks),
        patch("src.api.server.synthesize_answer", side_effect=exc),
        patch("src.api.server.load_config"),
    )


def test_query_returns_429_on_anthropic_rate_limit(tmp_path: Path, client: TestClient) -> None:
    import anthropic

    root = _setup_collection(tmp_path)
    exc = anthropic.RateLimitError(
        "rate limited", response=_fake_httpx_response(429), body=None
    )
    patches = _patch_query_path(root, exc)
    with patches[0], patches[1], patches[2], patches[3]:
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus"})
    assert resp.status_code == 429
    assert "rate" in resp.json()["detail"].lower()


def test_query_returns_504_on_anthropic_timeout(tmp_path: Path, client: TestClient) -> None:
    import anthropic

    root = _setup_collection(tmp_path)
    exc = anthropic.APITimeoutError(request=_fake_httpx_request())
    patches = _patch_query_path(root, exc)
    with patches[0], patches[1], patches[2], patches[3]:
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus"})
    assert resp.status_code == 504
    assert "tim" in resp.json()["detail"].lower()


def test_query_returns_502_on_anthropic_connection_error(tmp_path: Path, client: TestClient) -> None:
    import anthropic

    root = _setup_collection(tmp_path)
    exc = anthropic.APIConnectionError(request=_fake_httpx_request())
    patches = _patch_query_path(root, exc)
    with patches[0], patches[1], patches[2], patches[3]:
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus"})
    assert resp.status_code == 502


def test_query_clamps_upstream_5xx_to_502(tmp_path: Path, client: TestClient) -> None:
    import anthropic

    root = _setup_collection(tmp_path)
    exc = anthropic.APIStatusError(
        "upstream 500", response=_fake_httpx_response(500), body=None
    )
    patches = _patch_query_path(root, exc)
    with patches[0], patches[1], patches[2], patches[3]:
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus"})
    assert resp.status_code == 502


def test_query_treats_upstream_401_as_503(tmp_path: Path, client: TestClient) -> None:
    """Upstream auth failure = our misconfig; surface as 503 not 401, so the
    caller does not retry with their own credential."""
    import anthropic

    root = _setup_collection(tmp_path)
    exc = anthropic.APIStatusError(
        "bad key", response=_fake_httpx_response(401), body=None
    )
    patches = _patch_query_path(root, exc)
    with patches[0], patches[1], patches[2], patches[3]:
        resp = client.post("/query", json={"query": "q", "collection": "test-corpus"})
    assert resp.status_code == 503


# ─── _list_collections helper ───────────────────────────────────────────────


def test_list_collections_returns_empty_when_root_missing(tmp_path: Path) -> None:
    with patch("src.api.server._collections_dir", return_value=tmp_path / "absent"):
        assert _list_collections() == []


def test_list_collections_skips_dirs_without_index_db(tmp_path: Path) -> None:
    root = tmp_path / "collections"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "index.db").touch()
    (root / "beta-empty").mkdir(parents=True)  # no index.db

    with patch("src.api.server._collections_dir", return_value=root):
        cols = _list_collections()
    assert "alpha" in cols
    assert "beta-empty" not in cols


def test_list_collections_returns_sorted(tmp_path: Path) -> None:
    root = tmp_path / "collections"
    for name in ("zeta", "alpha", "mid"):
        d = root / name
        d.mkdir(parents=True)
        (d / "index.db").touch()
    with patch("src.api.server._collections_dir", return_value=root):
        cols = _list_collections()
    assert cols == sorted(cols)

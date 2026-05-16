"""Tests for `src.answer.synthesize_answer` + helpers.

The Anthropic client is patched throughout — no real API calls are made.
"""

from unittest.mock import MagicMock

import pytest

from src.answer import (
    AnsweredQuery,
    Citation,
    _build_context_block,
    _extract_citations,
    _truncate_snippet,
    synthesize_answer,
)
from src.pipeline.retriever import SearchResult


def _make_result(
    docid: str = "c1",
    content: str = "Paris is the capital of France.",
    source_pdf: str = "geo.pdf",
    chunk_type: str = "text",
    page_range: tuple[int, int] = (3, 3),
    score: float = 0.95,
) -> SearchResult:
    return SearchResult(
        docid=docid,
        content=content,
        source_pdf=source_pdf,
        chunk_type=chunk_type,
        page_range=page_range,
        score=score,
    )


def _mock_anthropic_client(answer_text: str, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Build an object that quacks like `anthropic.Anthropic()`."""
    content_block = MagicMock()
    content_block.text = answer_text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ─── _build_context_block ───────────────────────────────────────────────────


def test_build_context_block_single_result_includes_doc_tag_and_page():
    r = _make_result(page_range=(5, 5))
    block = _build_context_block([r])
    assert "[doc-1]" in block
    assert "geo.pdf" in block
    assert "p.5" in block
    assert r.content in block


def test_build_context_block_multiple_results_are_numbered():
    r1 = _make_result(docid="c1")
    r2 = _make_result(docid="c2", source_pdf="europe.pdf", content="Berlin is in Germany.")
    block = _build_context_block([r1, r2])
    assert "[doc-1]" in block
    assert "[doc-2]" in block
    assert "geo.pdf" in block
    assert "europe.pdf" in block


def test_build_context_block_multi_page_range_renders_pp():
    r = _make_result(page_range=(7, 9))
    assert "pp.7-9" in _build_context_block([r])


# ─── _truncate_snippet ──────────────────────────────────────────────────────


def test_truncate_snippet_short_passthrough():
    assert _truncate_snippet("hello") == "hello"


def test_truncate_snippet_long_is_capped():
    snip = _truncate_snippet("A" * 500)
    assert len(snip) <= 300


# ─── _extract_citations ─────────────────────────────────────────────────────


def test_extract_citations_finds_referenced_chunks_in_order():
    r1 = _make_result(docid="c1", source_pdf="a.pdf")
    r2 = _make_result(docid="c2", source_pdf="b.pdf")
    answer = "First we look at [doc-2], then at [doc-1]."
    cites = _extract_citations(answer, [r1, r2])
    assert [c.docid for c in cites] == ["c2", "c1"]


def test_extract_citations_dedupes_repeated_tags():
    r1 = _make_result(docid="c1")
    answer = "See [doc-1] and again [doc-1] and once more [doc-1]."
    cites = _extract_citations(answer, [r1])
    assert len(cites) == 1


def test_extract_citations_ignores_out_of_range_tags():
    r1 = _make_result(docid="c1")
    answer = "Real: [doc-1]. Hallucinated: [doc-42]."
    cites = _extract_citations(answer, [r1])
    assert len(cites) == 1
    assert cites[0].docid == "c1"


def test_extract_citations_empty_answer_returns_empty():
    assert _extract_citations("no tags here", [_make_result()]) == []


# ─── synthesize_answer — happy + refusal paths ──────────────────────────────


def test_synthesize_answer_empty_results_short_circuits_without_api_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    client = _mock_anthropic_client("should not be called")
    out = synthesize_answer("any?", [], client=client)
    assert isinstance(out, AnsweredQuery)
    assert "cannot answer" in out.answer.lower()
    assert out.citations == []
    assert out.prompt_tokens == 0
    assert out.completion_tokens == 0
    client.messages.create.assert_not_called()


def test_synthesize_answer_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        synthesize_answer("q?", [_make_result()])


def test_synthesize_answer_returns_full_payload():
    answer_text = "Paris is the capital [doc-1]."
    client = _mock_anthropic_client(answer_text, input_tokens=120, output_tokens=30)
    out = synthesize_answer("capital of France?", [_make_result()], client=client)
    assert out.answer == answer_text
    assert out.model == "claude-3-5-sonnet-latest"
    assert out.prompt_tokens == 120
    assert out.completion_tokens == 30
    assert out.latency_ms >= 0
    assert len(out.citations) == 1
    assert out.citations[0].docid == "c1"


def test_synthesize_answer_explicit_model_override():
    client = _mock_anthropic_client("ok [doc-1].")
    out = synthesize_answer("q", [_make_result()], client=client, model="claude-3-5-haiku-latest")
    assert out.model == "claude-3-5-haiku-latest"


def test_synthesize_answer_model_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "custom-model-xyz")
    client = _mock_anthropic_client("ok [doc-1].")
    out = synthesize_answer("q", [_make_result()], client=client)
    assert out.model == "custom-model-xyz"


def test_synthesize_answer_unreferenced_chunks_excluded_from_citations():
    r1 = _make_result(docid="c1", source_pdf="a.pdf")
    r2 = _make_result(docid="c2", source_pdf="b.pdf")
    client = _mock_anthropic_client("Only one source matters [doc-1].")
    out = synthesize_answer("q", [r1, r2], client=client)
    cited_ids = {c.docid for c in out.citations}
    assert "c1" in cited_ids
    assert "c2" not in cited_ids


def test_synthesize_answer_snippet_is_truncated():
    long_content = "A" * 1000
    r = _make_result(content=long_content)
    client = _mock_anthropic_client("Answer [doc-1].")
    out = synthesize_answer("q", [r], client=client)
    for c in out.citations:
        assert len(c.snippet) <= 300


# ─── Citation model contract ────────────────────────────────────────────────


def test_citation_model_validates():
    c = Citation(
        docid="chunk-1",
        source_pdf="test.pdf",
        page_range=(1, 3),
        snippet="some text",
        score=0.88,
    )
    assert c.page_range == (1, 3)
    assert c.score == pytest.approx(0.88)

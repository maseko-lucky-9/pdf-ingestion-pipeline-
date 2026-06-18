"""Confirm ``synthesize_answer`` produces identical output shapes under
both providers. Anthropic path tests live in ``test_answer.py``; this
file mirrors the same set of shape assertions through the Ollama branch.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from src.answer import AnsweredQuery, synthesize_answer
from src.llm_provider import OllamaAnswerClient
from src.pipeline.retriever import SearchResult


def _make_result(
    docid: str = "c1",
    content: str = "RSI is computed as 100 - 100/(1+RS).",
    source_pdf: str = "tsam.pdf",
    page_range: tuple[int, int] = (406, 409),
    score: float = 0.94,
) -> SearchResult:
    return SearchResult(
        docid=docid,
        content=content,
        source_pdf=source_pdf,
        chunk_type="text",
        page_range=page_range,
        score=score,
    )


def _ollama_client_returning(answer_text: str, prompt_tokens: int = 50, eval_count: int = 20) -> OllamaAnswerClient:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": answer_text},
                "prompt_eval_count": prompt_tokens,
                "eval_count": eval_count,
            },
        )
    )
    inner = httpx.Client(base_url="http://test", transport=transport)
    return OllamaAnswerClient(client=inner, model="gpt-oss-20b")


class TestSynthesizeAnswerOllamaBranch:
    def test_explicit_ollama_client_routes_to_ollama_path(self) -> None:
        client = _ollama_client_returning("RSI = 100 - 100/(1+RS) [doc-1].")
        out = synthesize_answer("What is RSI?", [_make_result()], ollama_client=client)
        assert isinstance(out, AnsweredQuery)
        assert out.model == "gpt-oss-20b"
        # Same citation extraction wires through both providers.
        assert len(out.citations) == 1
        assert out.citations[0].docid == "c1"
        # Token counts mapped from Ollama's prompt_eval_count + eval_count.
        assert out.prompt_tokens == 50
        assert out.completion_tokens == 20

    def test_empty_results_short_circuit_stamps_ollama_model(self) -> None:
        client = _ollama_client_returning("(unused)")
        out = synthesize_answer("q?", [], ollama_client=client)
        assert out.answer.startswith("I cannot answer")
        assert out.model == "gpt-oss-20b"
        assert out.prompt_tokens == 0
        assert out.completion_tokens == 0

    def test_explicit_ollama_client_used_regardless_of_env(self) -> None:
        """Explicit ollama_client param is used directly; env vars are irrelevant.
        (Env-driven auto-fallback is tested in TestResolveLlmProvider.)"""
        client = _ollama_client_returning("local answer [doc-1].")
        out = synthesize_answer("q?", [_make_result()], ollama_client=client)
        assert out.model == "gpt-oss-20b"
        assert "local answer" in out.answer

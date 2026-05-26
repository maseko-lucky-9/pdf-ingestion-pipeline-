"""Confirm ``score_faithfulness`` works under the Ollama judge."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.answer import Citation
from src.eval.faithfulness import FaithfulnessScore, score_faithfulness
from src.llm_provider import OllamaJudgeClient


def _citation(docid: str = "c1") -> Citation:
    return Citation(
        docid=docid,
        source_pdf="tsam.pdf",
        page_range=(406, 409),
        snippet="RSI is computed as 100 - 100/(1+RS).",
        score=0.94,
    )


def _judge_client(responses: list[str]) -> OllamaJudgeClient:
    """Build a judge client whose mock daemon returns each response in turn."""
    state = {"i": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] += 1
        body = responses[min(i, len(responses) - 1)]
        return httpx.Response(200, json={"message": {"content": body}})

    inner = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return OllamaJudgeClient(client=inner, model="llama3.1:8b", max_attempts=3)


class TestScoreFaithfulnessOllamaBranch:
    def test_happy_path_first_attempt_succeeds(self) -> None:
        judge = _judge_client([
            json.dumps({
                "verdicts": [{"docid": "c1", "supported": True, "reason": "exact match"}],
                "overall_supported_fraction": 1.0,
            })
        ])
        out = score_faithfulness(
            "What is RSI?",
            "RSI = 100 - 100/(1+RS) [doc-1].",
            [_citation("c1")],
            ollama_client=judge,
        )
        assert isinstance(out, FaithfulnessScore)
        assert out.overall == 1.0
        assert out.attempts == 1
        assert out.judge_parse_failed is False
        assert out.judge_model == "llama3.1:8b"

    def test_prose_then_json_succeeds_on_retry(self) -> None:
        """Llama-class models often emit prose before the JSON. The retry
        loop catches this and tries again with a stricter system prompt."""
        judge = _judge_client([
            "Sure, here's my analysis. The citation appears supported.",
            json.dumps({
                "verdicts": [{"docid": "c1", "supported": True, "reason": "ok"}],
                "overall_supported_fraction": 1.0,
            }),
        ])
        out = score_faithfulness("q?", "a [doc-1].", [_citation()], ollama_client=judge)
        assert out.overall == 1.0
        assert out.attempts == 2
        assert out.judge_parse_failed is False

    def test_all_attempts_fail_returns_overall_none_with_flag(self) -> None:
        """After 3 unparseable responses, faithfulness is reported as None
        with judge_parse_failed=True so the aggregate can count it."""
        judge = _judge_client(["no json here"] * 3)
        out = score_faithfulness("q?", "a [doc-1].", [_citation()], ollama_client=judge)
        assert out.overall is None
        assert out.attempts == 3
        assert out.judge_parse_failed is True

    def test_refusal_short_circuits_without_calling_judge(self) -> None:
        """A refusal answer has nothing to judge; the client should NOT be
        called. We verify by passing a judge that would error if invoked."""
        def handler(req: httpx.Request) -> httpx.Response:
            raise AssertionError("judge should not be called for refusal")

        inner = httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(handler)
        )
        judge = OllamaJudgeClient(client=inner, model="llama3.1:8b")
        out = score_faithfulness(
            "q?",
            "I cannot answer this question from the provided context.",
            [_citation()],
            ollama_client=judge,
        )
        assert out.overall is None
        assert out.verdicts == []
        assert out.judge_model == "llama3.1:8b"

    def test_no_citations_short_circuits(self) -> None:
        judge = _judge_client([])  # never called
        out = score_faithfulness("q?", "no cites here", [], ollama_client=judge)
        assert out.overall is None
        assert out.verdicts == []

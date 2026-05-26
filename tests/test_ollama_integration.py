"""Live Ollama integration tests.

Skipped by default. Run via ``pytest -m ollama_integration`` under
``.github/workflows/ollama-integration.yml`` which spins up an
``ollama/ollama:latest`` service container and pulls ``qwen2.5:0.5b``.

Locally: install Ollama, pull a model, then:

    OLLAMA_ANSWER_MODEL=qwen2.5:0.5b OLLAMA_JUDGE_MODEL=qwen2.5:0.5b \\
        pytest -m ollama_integration -q
"""
from __future__ import annotations

import json
import os

import pytest

from src.answer import synthesize_answer
from src.eval.faithfulness import score_faithfulness
from src.llm_provider import (
    OllamaAnswerClient,
    OllamaJudgeClient,
    probe_ollama_reachable,
    resolve_llm_provider,
)
from src.pipeline.retriever import SearchResult


pytestmark = pytest.mark.ollama_integration


@pytest.fixture(scope="module", autouse=True)
def _require_live_ollama():
    """Skip the module if no real daemon is reachable."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        probe_ollama_reachable(host, timeout=2.0)
    except Exception as exc:  # pragma: no cover - integration-only path
        pytest.skip(f"Ollama daemon not reachable at {host}: {exc}")


def _make_result() -> SearchResult:
    return SearchResult(
        docid="c1",
        content="The Relative Strength Index (RSI) is computed as 100 - 100/(1+RS), where RS is the average gain over the average loss across a fixed lookback window.",
        source_pdf="tsam.pdf",
        chunk_type="text",
        page_range=(406, 409),
        score=0.94,
    )


def test_resolver_picks_ollama_with_no_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_llm_provider() == "ollama"


def test_live_synthesize_answer_returns_text() -> None:
    """Smoke test: a real Ollama daemon produces a non-empty answer."""
    out = synthesize_answer(
        "What is the RSI formula?",
        [_make_result()],
        ollama_client=OllamaAnswerClient(),
    )
    # Don't assert on content quality — qwen2.5:0.5b is tiny. Just confirm
    # the pipeline produces a non-empty answer with the expected shape.
    assert out.answer
    assert out.model == OllamaAnswerClient().model
    assert out.prompt_tokens > 0


def test_live_judge_produces_a_verdict_or_marks_parse_failure() -> None:
    """The judge either parses successfully OR honestly marks the failure.
    Tiny models often produce unparseable JSON — that's fine, we just need
    the contract to hold: ``overall`` is None and ``judge_parse_failed`` is True."""
    from src.answer import Citation

    cite = Citation(
        docid="c1",
        source_pdf="tsam.pdf",
        page_range=(406, 409),
        snippet="RSI is computed as 100 - 100/(1+RS).",
        score=0.94,
    )
    out = score_faithfulness(
        "What is RSI?",
        "RSI = 100 - 100/(1+RS) [doc-1].",
        [cite],
        ollama_client=OllamaJudgeClient(max_attempts=3),
    )
    if out.judge_parse_failed:
        assert out.overall is None
        assert out.attempts == 3
    else:
        assert isinstance(out.overall, (int, float, type(None)))

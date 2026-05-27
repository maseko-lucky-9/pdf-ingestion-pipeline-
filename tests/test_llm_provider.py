"""Tests for the Ollama provider resolver + clients.

All real network calls are stubbed via ``httpx.MockTransport`` so the test
suite runs hermetic — no daemon, no keys, no credentials.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.llm_provider import (
    LlmProviderError,
    OllamaAnswerClient,
    OllamaJudgeClient,
    probe_ollama_reachable,
    resolve_llm_provider,
)


# ── resolve_llm_provider ────────────────────────────────────────────────


class TestResolveLlmProvider:
    def test_explicit_anthropic_wins_even_without_key(self) -> None:
        assert resolve_llm_provider(env={"LLM_PROVIDER": "anthropic"}) == "anthropic"

    def test_explicit_ollama_wins_with_key_set(self) -> None:
        assert (
            resolve_llm_provider(env={"LLM_PROVIDER": "ollama", "ANTHROPIC_API_KEY": "sk-real"})
            == "ollama"
        )

    def test_explicit_value_case_insensitive(self) -> None:
        assert resolve_llm_provider(env={"LLM_PROVIDER": "OLLAMA"}) == "ollama"
        assert resolve_llm_provider(env={"LLM_PROVIDER": "  Anthropic  "}) == "anthropic"

    def test_invalid_explicit_value_ignored(self) -> None:
        # Falls through to the key check; with no key → ollama.
        assert resolve_llm_provider(env={"LLM_PROVIDER": "openai"}) == "ollama"

    def test_anthropic_key_set_returns_anthropic(self) -> None:
        assert resolve_llm_provider(env={"ANTHROPIC_API_KEY": "sk-real"}) == "anthropic"

    def test_anthropic_key_whitespace_only_falls_through(self) -> None:
        # Whitespace-only is not a key; must auto-fall to ollama.
        assert resolve_llm_provider(env={"ANTHROPIC_API_KEY": "   "}) == "ollama"

    def test_empty_env_defaults_to_ollama(self) -> None:
        assert resolve_llm_provider(env={}) == "ollama"


# ── probe_ollama_reachable ──────────────────────────────────────────────


def _probe_with_transport(transport: httpx.MockTransport) -> Any:
    """Helper that monkey-patches httpx.Client to use the given transport."""
    # probe_ollama_reachable builds its own client; we patch via constructor
    # default by inspecting the function body — simplest is to use
    # monkeypatching at the call site via httpx itself.
    return transport


class TestProbeOllamaReachable:
    def test_ok_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/api/tags"
            return httpx.Response(200, json={"models": []})

        # Patch httpx.Client so probe uses our transport.
        transport = httpx.MockTransport(handler)
        orig = httpx.Client

        def patched_client(*args: Any, **kwargs: Any) -> httpx.Client:
            kwargs.pop("base_url", None)
            return orig(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", patched_client)
        # Should not raise.
        probe_ollama_reachable("http://test")

    def test_connect_error_raises_llm_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=req)

        transport = httpx.MockTransport(handler)
        orig = httpx.Client

        def patched(*args: Any, **kwargs: Any) -> httpx.Client:
            kwargs.pop("base_url", None)
            return orig(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", patched)
        with pytest.raises(LlmProviderError, match="unreachable"):
            probe_ollama_reachable("http://test")

    def test_4xx_raises_with_status_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(401, text="unauthorized"))
        orig = httpx.Client

        def patched(*args: Any, **kwargs: Any) -> httpx.Client:
            kwargs.pop("base_url", None)
            return orig(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", patched)
        with pytest.raises(LlmProviderError) as excinfo:
            probe_ollama_reachable("http://test")
        assert excinfo.value.status_code == 401


# ── OllamaAnswerClient ──────────────────────────────────────────────────


def _make_ollama_client_with_handler(handler) -> httpx.Client:
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


class TestOllamaAnswerClient:
    def test_happy_path_returns_text_and_token_counts(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Answer here [doc-1]."},
                    "prompt_eval_count": 42,
                    "eval_count": 18,
                },
            )

        client = OllamaAnswerClient(client=_make_ollama_client_with_handler(handler), model="m1")
        result = client.complete(
            system_prompt="sys", user_message="user", max_tokens=256, temperature=0.0
        )
        assert result.answer_text == "Answer here [doc-1]."
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 18
        # Wire-level: messages array shape, num_predict forwarded, stream off.
        assert captured["body"]["model"] == "m1"
        assert captured["body"]["stream"] is False
        assert captured["body"]["messages"][0]["role"] == "system"
        assert captured["body"]["options"]["num_predict"] == 256

    def test_timeout_raises_llm_provider_error_504(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("slow", request=req)

        client = OllamaAnswerClient(client=_make_ollama_client_with_handler(handler))
        with pytest.raises(LlmProviderError) as excinfo:
            client.complete(system_prompt="s", user_message="u", max_tokens=10)
        assert excinfo.value.status_code == 504

    def test_connect_error_raises_503(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=req)

        client = OllamaAnswerClient(client=_make_ollama_client_with_handler(handler))
        with pytest.raises(LlmProviderError) as excinfo:
            client.complete(system_prompt="s", user_message="u", max_tokens=10)
        assert excinfo.value.status_code == 503

    def test_4xx_response_surfaces_status_code(self) -> None:
        handler = lambda r: httpx.Response(404, text='{"error":"model not found"}')  # noqa: E731
        client = OllamaAnswerClient(client=_make_ollama_client_with_handler(handler))
        with pytest.raises(LlmProviderError) as excinfo:
            client.complete(system_prompt="s", user_message="u", max_tokens=10)
        assert excinfo.value.status_code == 404
        assert "404" in str(excinfo.value)

    def test_model_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ANSWER_MODEL", "qwen2.5:7b")
        client = OllamaAnswerClient()
        assert client.model == "qwen2.5:7b"


# ── OllamaJudgeClient (retry loop) ──────────────────────────────────────


class TestOllamaJudgeClient:
    def test_first_attempt_parses_returns_attempts_1(self) -> None:
        handler = lambda r: httpx.Response(  # noqa: E731
            200,
            json={"message": {"content": '{"verdicts": [], "overall_supported_fraction": 1.0}'}},
        )
        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler))
        result = client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert result.parsed == {"verdicts": [], "overall_supported_fraction": 1.0}
        assert result.attempts == 1

    def test_retries_on_prose_before_json_then_succeeds(self) -> None:
        """First attempt returns prose-only (no JSON object). Retry produces
        valid JSON. The client must report attempts=2."""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json={"message": {"content": "Here is my answer: no JSON here."}})
            return httpx.Response(200, json={"message": {"content": '{"verdicts": []}'}})

        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler), max_attempts=3)
        result = client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert result.attempts == 2
        assert result.parsed == {"verdicts": []}
        assert len(result.raw_responses) == 2

    def test_all_attempts_fail_returns_parsed_none(self) -> None:
        # Every attempt: prose with no JSON braces.
        handler = lambda r: httpx.Response(  # noqa: E731
            200, json={"message": {"content": "no braces ever"}}
        )
        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler), max_attempts=3)
        result = client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert result.parsed is None
        assert result.attempts == 3
        assert len(result.raw_responses) == 3

    def test_retries_when_verdict_missing_docid_then_succeeds(self) -> None:
        """Small models sometimes return well-formed JSON with verdicts that
        omit ``docid`` (e.g. they emit ``"id"`` instead). That used to leak
        through as a ``KeyError`` at the call site; now the retry loop treats
        it as a parse failure and tries again with a stricter prompt."""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                # Valid JSON, wrong shape — missing docid on the verdict.
                return httpx.Response(
                    200,
                    json={"message": {"content": '{"verdicts": [{"id": "c1", "supported": true}]}'}},
                )
            return httpx.Response(
                200,
                json={"message": {"content": '{"verdicts": [{"docid": "c1", "supported": true}]}'}},
            )

        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler), max_attempts=3)
        result = client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert result.attempts == 2
        assert result.parsed == {"verdicts": [{"docid": "c1", "supported": True}]}

    def test_all_attempts_return_wrong_shape_yields_parsed_none(self) -> None:
        """Every retry returns malformed-shape JSON → retries exhausted →
        ``parsed=None`` so the caller surfaces ``judge_parse_failed=True``."""
        handler = lambda r: httpx.Response(  # noqa: E731
            200,
            json={"message": {"content": '{"verdicts": [{"id": "c1"}]}'}},
        )
        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler), max_attempts=3)
        result = client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert result.parsed is None
        assert result.attempts == 3

    def test_malformed_json_decode_failure_retries(self) -> None:
        """Unclosed JSON brackets → JSONDecodeError → retry."""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                # JSON-like span but malformed.
                return httpx.Response(200, json={"message": {"content": '{"verdicts": [unclosed'}})
            return httpx.Response(200, json={"message": {"content": '{"verdicts": []}'}})

        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler))
        result = client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert result.attempts == 2
        assert result.parsed == {"verdicts": []}

    def test_http_error_propagates_immediately(self) -> None:
        """A network/HTTP error should NOT trigger a retry — that's an
        infrastructure problem, not a parse problem."""
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=req)

        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler))
        with pytest.raises(LlmProviderError) as excinfo:
            client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)
        assert excinfo.value.status_code == 503

    def test_4xx_propagates_without_retry(self) -> None:
        handler = lambda r: httpx.Response(404, text="not found")  # noqa: E731
        client = OllamaJudgeClient(client=_make_ollama_client_with_handler(handler))
        with pytest.raises(LlmProviderError):
            client.complete_with_retry(system_prompt="s", user_message="u", max_tokens=200)

    def test_judge_model_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_JUDGE_MODEL", "qwen2.5:3b")
        client = OllamaJudgeClient()
        assert client.model == "qwen2.5:3b"

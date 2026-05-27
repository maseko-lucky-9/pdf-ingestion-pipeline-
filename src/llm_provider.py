"""LLM provider resolution + Ollama answer/judge clients.

When ``ANTHROPIC_API_KEY`` is unset (or ``LLM_PROVIDER=ollama`` is explicit),
the RAG pipeline answers and the eval judge route through local Ollama models
instead of Anthropic. The resolver is dirt-simple: env var first, then the
existence of an Anthropic key, then a final fallback to ollama.

Design notes:

- **httpx.Client (sync)**, not the ``ollama`` Python SDK. We need (a) the
  same call shape ``synthesize_answer`` uses today (sync), (b) tests that
  inject ``httpx.MockTransport`` (matches the pattern in
  ``gemini-mcp-server/tests/test_client.py``), and (c) no new dependency —
  httpx is already in ``requirements-api.txt``. The existing
  ``src/pipeline/embedder.py`` keeps using the ``ollama`` SDK because the
  embedder is batch-oriented and the SDK's batch ``embed`` is convenient.
- **Production hardening.** When auto-fallback would silently degrade to
  Ollama because someone forgot to load ``.env`` in production, set
  ``LLM_PROVIDER=anthropic`` on the systemd unit / Cloudflare Access tunnel.
  Explicit override always wins.
- **Judge retry loop.** Local models (especially Llama 3.x 8B) routinely
  emit prose before the JSON or fail to close brackets. The judge client
  retries up to 3 times with progressively stricter system prompts.
  After 3 failures the caller surfaces ``faithfulness=None`` and increments
  a parse-failures counter — never raise into the eval loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import httpx

from src.config import load_config

log = logging.getLogger(__name__)

_OLLAMA_GENERATE_PATH = "/api/generate"
_OLLAMA_CHAT_PATH = "/api/chat"
_OLLAMA_TAGS_PATH = "/api/tags"
_DEFAULT_PROBE_TIMEOUT = 5.0
_JUDGE_MAX_ATTEMPTS = 3
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _judge_payload_shape_valid(parsed: object) -> bool:
    """Confirm the judge payload has the keys ``score_faithfulness`` expects.

    Small Ollama models sometimes return JSON-shaped output where every
    ``verdicts`` entry omits ``docid`` (e.g. they emit ``"id"`` or skip the
    key entirely). Without this guard the call site explodes with
    ``KeyError: 'docid'`` — which the retry loop was meant to prevent.

    Returns ``False`` for any of:
      - non-dict root
      - missing or non-list ``verdicts``
      - any verdict entry that isn't a dict or lacks ``docid`` / ``supported``

    An empty ``verdicts`` list is accepted (a model may legitimately decide
    nothing in the answer was checkable).
    """
    if not isinstance(parsed, dict):
        return False
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, list):
        return False
    for v in verdicts:
        if not isinstance(v, dict):
            return False
        if "docid" not in v or "supported" not in v:
            return False
    return True


ProviderName = Literal["anthropic", "ollama"]


class LlmProviderError(RuntimeError):
    """Raised when the LLM provider cannot service a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_llm_provider(env: dict[str, str] | None = None) -> ProviderName:
    """Pick the LLM provider for this process.

    Resolution order:
      1. Explicit ``LLM_PROVIDER`` env var (``anthropic`` | ``ollama``).
      2. ``ANTHROPIC_API_KEY`` is set + non-whitespace → anthropic.
      3. Auto-fallback → ollama.

    Pass ``env`` explicitly in tests to avoid touching ``os.environ``.
    """
    import os

    e = env if env is not None else os.environ
    explicit = (e.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in {"anthropic", "ollama"}:
        return explicit  # type: ignore[return-value]
    if (e.get("ANTHROPIC_API_KEY") or "").strip():
        return "anthropic"
    return "ollama"


def probe_ollama_reachable(host: str, timeout: float = _DEFAULT_PROBE_TIMEOUT) -> None:
    """Hit ``{host}/api/tags`` once at boot. Raise ``LlmProviderError`` on failure.

    The eval harness CLI and the FastAPI lifespan handler both call this when
    the resolver returned ``ollama`` so a missing daemon fails fast at startup
    instead of producing per-request 5xx loops downstream.
    """
    url = host.rstrip("/") + _OLLAMA_TAGS_PATH
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        raise LlmProviderError(
            f"Ollama daemon unreachable at {host!r}. Is `ollama serve` running? "
            f"Underlying error: {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise LlmProviderError(
            f"Ollama at {host!r} returned HTTP {resp.status_code} on /api/tags.",
            status_code=resp.status_code,
        )


# ── Answer client ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OllamaAnswerResult:
    """What ``OllamaAnswerClient.complete`` returns to ``synthesize_answer``."""

    answer_text: str
    prompt_tokens: int
    completion_tokens: int


class OllamaAnswerClient:
    """Sync httpx client for the Ollama ``/api/chat`` endpoint."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = load_config().ollama
        self._host = (host or os.environ.get("OLLAMA_HOST") or cfg.host).rstrip("/")
        self._model = model or os.environ.get("OLLAMA_ANSWER_MODEL") or cfg.answer_model
        env_timeout = os.environ.get("OLLAMA_TIMEOUT")
        self._timeout = timeout or (float(env_timeout) if env_timeout else cfg.request_timeout_seconds)
        # Injected client is for tests with ``httpx.MockTransport``.
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def _build_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self._host, timeout=self._timeout)

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> OllamaAnswerResult:
        """One non-streaming chat completion. Raises ``LlmProviderError`` on failure."""
        payload: dict[str, object] = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        client = self._build_client()
        try:
            resp = client.post(_OLLAMA_CHAT_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise LlmProviderError(f"Ollama timed out: {exc}", status_code=504) from exc
        except httpx.ConnectError as exc:
            raise LlmProviderError(f"Ollama unreachable: {exc}", status_code=503) from exc
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Ollama HTTP error: {exc}", status_code=502) from exc
        finally:
            if self._client is None:  # we created it; close it
                client.close()

        if resp.status_code >= 400:
            raise LlmProviderError(
                f"Ollama /api/chat returned HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )

        body = resp.json()
        message = body.get("message") or {}
        answer_text = message.get("content", "") or ""
        return OllamaAnswerResult(
            answer_text=answer_text,
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            completion_tokens=int(body.get("eval_count") or 0),
        )


# ── Judge client (with retry loop) ─────────────────────────────────────────


@dataclass(frozen=True)
class OllamaJudgeResult:
    """What ``OllamaJudgeClient.complete`` returns to ``score_faithfulness``.

    ``parsed`` is the JSON object the judge produced (or ``None`` if all 3
    attempts failed to parse). ``attempts`` is the number of calls made
    (1–3). ``raw_responses`` carries every raw text in order for debugging.
    """

    parsed: dict | None
    attempts: int
    raw_responses: list[str]


_JUDGE_STRICTER_PROMPTS = [
    "",  # attempt 1: original system prompt unchanged
    " IMPORTANT: Output ONLY the JSON object. No prose before or after. No code fences.",
    " STRICT: Your previous attempt did not produce parseable JSON. Output exactly one JSON object and nothing else. Begin your reply with `{` and end with `}`.",
]


class OllamaJudgeClient:
    """Sync httpx client for the Ollama judge call, with retry-on-parse-fail."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        *,
        client: httpx.Client | None = None,
        max_attempts: int = _JUDGE_MAX_ATTEMPTS,
    ) -> None:
        cfg = load_config().ollama
        self._host = (host or os.environ.get("OLLAMA_HOST") or cfg.host).rstrip("/")
        self._model = model or os.environ.get("OLLAMA_JUDGE_MODEL") or cfg.judge_model
        env_timeout = os.environ.get("OLLAMA_TIMEOUT")
        self._timeout = timeout or (float(env_timeout) if env_timeout else cfg.request_timeout_seconds)
        self._client = client
        self._max_attempts = max_attempts

    @property
    def model(self) -> str:
        return self._model

    def _build_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(base_url=self._host, timeout=self._timeout)

    def _call_once(
        self,
        client: httpx.Client,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }
        resp = client.post(_OLLAMA_CHAT_PATH, json=payload)
        if resp.status_code >= 400:
            raise LlmProviderError(
                f"Ollama judge returned HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        body = resp.json()
        return ((body.get("message") or {}).get("content")) or ""

    def complete_with_retry(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> OllamaJudgeResult:
        """Try up to ``max_attempts`` times to get parseable JSON.

        Each retry prepends a progressively stricter "JSON only" instruction
        to the system prompt. After ``max_attempts`` failures returns
        ``parsed=None`` so the caller can surface ``faithfulness=None`` for
        this example without raising.
        """
        raw_responses: list[str] = []
        client = self._build_client()
        try:
            for attempt in range(1, self._max_attempts + 1):
                strictness = _JUDGE_STRICTER_PROMPTS[min(attempt - 1, len(_JUDGE_STRICTER_PROMPTS) - 1)]
                effective_prompt = (system_prompt + strictness).strip()
                try:
                    raw = self._call_once(
                        client,
                        system_prompt=effective_prompt,
                        user_message=user_message,
                        max_tokens=max_tokens,
                    )
                except httpx.TimeoutException as exc:
                    raise LlmProviderError(f"Ollama judge timed out: {exc}", status_code=504) from exc
                except httpx.ConnectError as exc:
                    raise LlmProviderError(f"Ollama judge unreachable: {exc}", status_code=503) from exc
                except httpx.HTTPError as exc:
                    raise LlmProviderError(f"Ollama judge HTTP error: {exc}", status_code=502) from exc

                raw_responses.append(raw)
                match = _JSON_RE.search(raw)
                if not match:
                    log.warning(
                        "ollama_judge_parse_failed",
                        extra={"attempt": attempt, "snippet": raw[:120]},
                    )
                    continue
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    log.warning(
                        "ollama_judge_json_decode_failed",
                        extra={"attempt": attempt, "snippet": match.group(0)[:120]},
                    )
                    continue
                # Validate verdict shape — small models sometimes return
                # well-formed JSON with the wrong keys (e.g. "id" instead of
                # "docid"). Treat as a parse failure so the retry loop gets
                # another chance with a stricter prompt.
                if not _judge_payload_shape_valid(parsed):
                    log.warning(
                        "ollama_judge_shape_invalid",
                        extra={"attempt": attempt, "snippet": match.group(0)[:160]},
                    )
                    continue
                return OllamaJudgeResult(parsed=parsed, attempts=attempt, raw_responses=raw_responses)
        finally:
            if self._client is None:
                client.close()

        return OllamaJudgeResult(parsed=None, attempts=self._max_attempts, raw_responses=raw_responses)

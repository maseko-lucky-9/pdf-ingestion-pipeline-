"""Claude-API-backed answer synthesis on top of the existing retriever.

`synthesize_answer()` is the single integration point. It takes a query and
the retriever's `SearchResult` list, formats them into a numbered context
block, asks Claude to answer using only that context (with `[doc-N]`
citations), and parses the citations back out for the response payload.

Design notes (full rationale in `docs/decisions/ADR-006-...md`):

- **Extractive citing with `[doc-N]` tags.** Numbering each retrieved chunk
  in the prompt makes citations machine-parseable: a regex pass over the
  answer text recovers which chunks were used.
- **Hard refusal when context is empty.** If the retriever returns no
  results, we short-circuit without an API call and return a fixed message.
- **Hard refusal on insufficient context.** The system prompt instructs
  Claude to respond with a fixed sentence rather than blend retrieved facts
  with parametric knowledge.
- **Snippet truncation.** Each citation's `snippet` is capped at 300 chars
  so the response payload stays bounded for callers (web UI, MCP, eval).
"""

import os
import re
import time

import anthropic
from pydantic import BaseModel, Field

from src.llm_provider import OllamaAnswerClient, resolve_llm_provider
from src.pipeline.retriever import SearchResult

# Pinned to a dated alias so eval baselines do not drift silently when
# Anthropic ships a new minor. Bump this together with a re-baseline of
# results/slice2-baseline.json (see ADR-006 + slice 3 follow-ups).
_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_DEFAULT_MAX_TOKENS = 4096
_REFUSAL_TEXT = "I cannot answer this question from the provided context."
_SNIPPET_MAX_CHARS = 300

_SYSTEM_PROMPT = """\
You are a precise research assistant answering questions strictly from the
context the user provides. Rules:

1. Use ONLY the context blocks supplied — never blend in outside knowledge.
2. Cite every factual claim with the matching [doc-N] tag from the context.
3. If the context is insufficient to answer, respond ONLY with:
   "I cannot answer this question from the provided context."
4. Quote short phrases verbatim when accuracy matters; paraphrase otherwise.
5. Do not invent docids or citations.
"""

# Captures integer Ns inside citation tags in the model's answer.
#
# The canonical form the system prompt asks for is `[doc-N]`. We tolerate
# common drift the model produces in practice:
#   * Case:   [Doc-1], [DOC-1]
#   * Spacing inside brackets:  [doc 1], [ doc-1 ], [doc- 1]
#   * Multi-citation runs:      [doc-1, doc-3], [doc-1; doc-3]
# Each `[...]` group is scanned independently and every integer it contains
# becomes one citation reference. Outer brackets that contain no integer are
# left as-is in the rendered answer (defensive — Markdown lists, etc.).
_DOC_TAG_BLOCK_RE = re.compile(r"\[\s*[Dd][Oo][Cc][^\]]*\]")
_DOC_INT_RE = re.compile(r"\d+")


class Citation(BaseModel):
    """One cited source in an `AnsweredQuery`."""

    docid: str = Field(..., description="The chunk id from the retriever.")
    source_pdf: str = Field(..., description="Filename of the source PDF.")
    page_range: tuple[int, int] = Field(..., description="Inclusive (start, end) page numbers.")
    snippet: str = Field(..., max_length=_SNIPPET_MAX_CHARS, description="Verbatim excerpt from the chunk.")
    score: float = Field(..., description="Retriever fusion score for this chunk.")


class AnsweredQuery(BaseModel):
    """Final response from `synthesize_answer`."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def _build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks as a numbered context block for the prompt.

    Each chunk becomes a `[doc-N]` block tagging source PDF + page range so
    the model can cite precisely.
    """
    blocks: list[str] = []
    for idx, r in enumerate(results, start=1):
        page_label = ""
        if r.page_range and r.page_range[0] is not None:
            start, end = r.page_range
            page_label = f" p.{start}" if start == end else f" pp.{start}-{end}"
        header = f"[doc-{idx}] {r.source_pdf}{page_label}"
        blocks.append(f"{header}\n{r.content}")
    return "\n\n".join(blocks)


def _truncate_snippet(content: str) -> str:
    if len(content) <= _SNIPPET_MAX_CHARS:
        return content
    return content[: _SNIPPET_MAX_CHARS - 1].rstrip() + "…"


def _extract_citations(answer_text: str, results: list[SearchResult]) -> list[Citation]:
    """Find every unique citation index in the answer and resolve to Citations.

    Tolerates model drift (case, spacing, multi-citation runs). A tag
    pointing past the end of results is silently dropped (defensive — a
    misbehaving model occasionally invents tags). Ordering reflects the
    order of first appearance in the answer text.
    """
    seen: set[int] = set()
    citations: list[Citation] = []
    for block in _DOC_TAG_BLOCK_RE.finditer(answer_text):
        for int_match in _DOC_INT_RE.finditer(block.group(0)):
            n = int(int_match.group(0))
            if n in seen or n < 1 or n > len(results):
                continue
            seen.add(n)
            r = results[n - 1]
            citations.append(
                Citation(
                    docid=r.docid,
                    source_pdf=r.source_pdf,
                    page_range=r.page_range,
                    snippet=_truncate_snippet(r.content),
                    score=r.score,
                )
            )
    return citations


def synthesize_answer(
    query: str,
    results: list[SearchResult],
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    client: anthropic.Anthropic | None = None,
    ollama_client: OllamaAnswerClient | None = None,
) -> AnsweredQuery:
    """Generate a cited answer from a query + retrieval results.

    The provider is picked by ``resolve_llm_provider()`` (env-driven):
      * ``LLM_PROVIDER=anthropic|ollama`` always wins,
      * else ``ANTHROPIC_API_KEY`` set → Anthropic,
      * else auto-fallback to Ollama (local).

    Args:
        query: User question.
        results: Output of `src.pipeline.retriever.retrieve()`.
        api_key: Override env var (defaults to ``ANTHROPIC_API_KEY``).
        model: Override env var (defaults to ``ANTHROPIC_MODEL`` or claude-3-5-sonnet-latest).
        max_tokens: Completion token budget.
        client: Pre-built Anthropic client (useful for tests).
        ollama_client: Pre-built ``OllamaAnswerClient`` (useful for tests).

    Raises:
        EnvironmentError: If the Anthropic API key is missing and no
            ``client`` was supplied AND the provider resolved to ``anthropic``.
    """
    # Explicit client override pins the provider: callers passing a mocked
    # Anthropic client (existing test suite) or a mocked Ollama client are
    # making an explicit choice; the resolver only fires when neither is set.
    if client is not None and ollama_client is None:
        provider: str = "anthropic"
    elif ollama_client is not None and client is None:
        provider = "ollama"
    else:
        provider = resolve_llm_provider()

    # Short-circuit: no retrieval results means no context to ground on.
    if not results:
        if provider == "anthropic":
            stamped = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        else:
            stamped = (ollama_client.model if ollama_client else OllamaAnswerClient().model)
        return AnsweredQuery(
            answer=_REFUSAL_TEXT,
            citations=[],
            model=stamped,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
        )

    context_block = _build_context_block(results)
    user_message = (
        f"Question: {query}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer using only the context above. Cite every factual claim with [doc-N]."
    )

    if provider == "anthropic":
        return _synthesize_anthropic(
            user_message, results, api_key=api_key, model=model,
            max_tokens=max_tokens, client=client,
        )
    return _synthesize_ollama(
        user_message, results, max_tokens=max_tokens, ollama_client=ollama_client,
    )


def _synthesize_anthropic(
    user_message: str,
    results: list[SearchResult],
    *,
    api_key: str | None,
    model: str | None,
    max_tokens: int,
    client: anthropic.Anthropic | None,
) -> AnsweredQuery:
    resolved_model = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    if client is None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set and no client was supplied. "
                "Copy .env.example → .env and fill it in, or pass `client=` for testing."
            )
        client = anthropic.Anthropic(api_key=resolved_key)

    t0 = time.perf_counter()
    response = client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer_text = response.content[0].text if response.content else ""
    citations = _extract_citations(answer_text, results)

    return AnsweredQuery(
        answer=answer_text,
        citations=citations,
        model=resolved_model,
        prompt_tokens=response.usage.input_tokens if response.usage else 0,
        completion_tokens=response.usage.output_tokens if response.usage else 0,
        latency_ms=latency_ms,
    )


def _synthesize_ollama(
    user_message: str,
    results: list[SearchResult],
    *,
    max_tokens: int,
    ollama_client: OllamaAnswerClient | None,
) -> AnsweredQuery:
    """Same shape as the Anthropic path, routed through Ollama HTTP."""
    if ollama_client is None:
        ollama_client = OllamaAnswerClient()

    t0 = time.perf_counter()
    result = ollama_client.complete(
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    citations = _extract_citations(result.answer_text, results)
    return AnsweredQuery(
        answer=result.answer_text,
        citations=citations,
        model=ollama_client.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=latency_ms,
    )

"""LLM-as-judge faithfulness scoring for cited RAG answers.

The judge reads a (question, answer, cited_chunks) triple and decides — per
citation — whether the cited chunk supports the claim the citation backs.
Aggregate score is the fraction of citations the judge deems supported.

Design notes:

- **Judge model.** Claude Haiku (cheapest tier). Faithfulness rubric is binary
  per citation, so a small model is sufficient. The judge model is configurable
  via the ``ANTHROPIC_JUDGE_MODEL`` env var; default is ``claude-3-5-haiku-latest``.
- **Refusal short-circuit.** If the answer is the canonical refusal string (or
  has zero citations), the score is reported as ``None`` and excluded from
  aggregates — refusal is a separate axis from faithfulness.
- **Determinism.** ``temperature=0``. The output is a strict JSON object so a
  ``json.loads`` parse step replaces any "judge-explained" prose.
- **Calibration.** Trust nothing the judge says without spot-checks. Use
  ``calibrate_judge_against_manual_labels`` to compare judge verdicts against
  a hand-labelled set; if agreement < 80%, the judge output should ship with
  a caveat in the eval report rather than as a headline number.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import anthropic

from src.answer import _REFUSAL_TEXT, Citation
from src.llm_provider import OllamaJudgeClient, resolve_llm_provider

# Pinned to a dated alias; see ADR-006 + src/answer.py for rationale.
_DEFAULT_JUDGE_MODEL = "claude-3-5-haiku-20241022"
_MAX_TOKENS = 1024

_JUDGE_SYSTEM = """\
You are a strict citation-faithfulness judge. Your job is to decide, for each
citation in an answer, whether the cited chunk actually supports the claim
the citation backs.

Rules:
1. A citation is SUPPORTED only if the cited chunk contains the exact factual
   content the answer attributes to it. Paraphrasing is fine; invention is not.
2. A citation is UNSUPPORTED if the chunk is tangentially related, the claim
   is broader than the chunk warrants, or the chunk content contradicts the
   claim.
3. Refuse-style answers ("I cannot answer this question from the provided
   context.") have no citations to judge; do not score them.
4. Output STRICT JSON with shape:
   {"verdicts": [{"docid": "<id>", "supported": true|false, "reason": "<short>"}], "overall_supported_fraction": <float>}
5. Do not include any prose outside the JSON.
"""

_USER_TEMPLATE = """\
Question:
{question}

Answer (with [doc-N] citations inline):
{answer}

Cited chunks:
{chunks_block}

Judge each citation strictly per the rules. Output JSON only.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class FaithfulnessVerdict:
    """Per-citation outcome from the judge."""

    docid: str
    supported: bool
    reason: str


@dataclass
class FaithfulnessScore:
    """Aggregate judge outcome for one (question, answer) pair."""

    overall: float | None  # None when there are no citations to score
    verdicts: list[FaithfulnessVerdict]
    judge_model: str
    raw_response: str  # for debugging / spot-checks
    # Ollama-path-only metadata: how many attempts the judge took to produce
    # parseable JSON (1 = first try, 0 = Anthropic path that doesn't retry).
    # ``judge_parse_failed`` is True when all attempts failed and ``overall``
    # is None as a parse-failure marker (distinct from None for no-citations).
    attempts: int = 0
    judge_parse_failed: bool = False


def _build_chunks_block(citations: list[Citation]) -> str:
    blocks: list[str] = []
    for idx, c in enumerate(citations, start=1):
        page_label = ""
        if c.page_range and c.page_range[0] is not None:
            s, e = c.page_range
            page_label = f" p.{s}" if s == e else f" pp.{s}-{e}"
        header = f"[doc-{idx}] (docid={c.docid}) {c.source_pdf}{page_label}"
        blocks.append(f"{header}\n{c.snippet}")
    return "\n\n".join(blocks)


def _parse_judge_response(raw: str) -> tuple[float | None, list[FaithfulnessVerdict]]:
    """Pull the first JSON object out of the judge's response and parse it.

    Tolerant of judge models that wrap the JSON in a code-fence — the regex
    grabs the first ``{...}`` span and feeds it to ``json.loads``.
    """
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"Judge produced no JSON object: {raw[:200]!r}")
    payload = json.loads(match.group(0))
    verdicts = [
        FaithfulnessVerdict(
            docid=v["docid"],
            supported=bool(v["supported"]),
            reason=str(v.get("reason", ""))[:200],
        )
        for v in payload.get("verdicts", [])
    ]
    overall = payload.get("overall_supported_fraction")
    if overall is None and verdicts:
        overall = sum(1 for v in verdicts if v.supported) / len(verdicts)
    return overall, verdicts


def score_faithfulness(
    question: str,
    answer: str,
    citations: list[Citation],
    *,
    api_key: str | None = None,
    judge_model: str | None = None,
    client: anthropic.Anthropic | None = None,
    ollama_client: OllamaJudgeClient | None = None,
) -> FaithfulnessScore:
    """Score citation faithfulness for one (question, answer) pair.

    Returns a ``FaithfulnessScore`` with ``overall=None`` when the answer is a
    refusal or has no citations. Under the Ollama provider, also returns
    ``overall=None`` + ``judge_parse_failed=True`` after 3 failed parse
    attempts; callers should report ``judge_parse_failures`` in their
    aggregate metrics so the noise is visible rather than silent.

    Raises:
        EnvironmentError: If provider resolves to anthropic and the API key
            is missing and no client supplied.
    """
    # Explicit client override pins the provider; resolver only fires when
    # neither was supplied. Matches the synthesize_answer() convention.
    if client is not None and ollama_client is None:
        provider: str = "anthropic"
    elif ollama_client is not None and client is None:
        provider = "ollama"
    else:
        provider = resolve_llm_provider()

    if answer.strip().startswith(_REFUSAL_TEXT) or not citations:
        if provider == "anthropic":
            stamped = judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
        else:
            stamped = (ollama_client.model if ollama_client else OllamaJudgeClient().model)
        return FaithfulnessScore(
            overall=None,
            verdicts=[],
            judge_model=stamped,
            raw_response="",
        )

    user_message = _USER_TEMPLATE.format(
        question=question,
        answer=answer,
        chunks_block=_build_chunks_block(citations),
    )

    if provider == "anthropic":
        return _judge_anthropic(
            user_message, api_key=api_key, judge_model=judge_model, client=client,
        )
    return _judge_ollama(user_message, ollama_client=ollama_client)


def _judge_anthropic(
    user_message: str,
    *,
    api_key: str | None,
    judge_model: str | None,
    client: anthropic.Anthropic | None,
) -> FaithfulnessScore:
    resolved_model = judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)

    if client is None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set and no client was supplied."
            )
        client = anthropic.Anthropic(api_key=resolved_key)

    response = client.messages.create(
        model=resolved_model,
        max_tokens=_MAX_TOKENS,
        temperature=0,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text if response.content else ""
    overall, verdicts = _parse_judge_response(raw)

    return FaithfulnessScore(
        overall=overall, verdicts=verdicts,
        judge_model=resolved_model, raw_response=raw,
        attempts=1, judge_parse_failed=False,
    )


def _judge_ollama(
    user_message: str,
    *,
    ollama_client: OllamaJudgeClient | None,
) -> FaithfulnessScore:
    """Ollama judge with the 3-attempt retry loop documented in ADR-006."""
    if ollama_client is None:
        ollama_client = OllamaJudgeClient()

    result = ollama_client.complete_with_retry(
        system_prompt=_JUDGE_SYSTEM,
        user_message=user_message,
        max_tokens=_MAX_TOKENS,
    )
    last_raw = result.raw_responses[-1] if result.raw_responses else ""

    if result.parsed is None:
        # All retries failed to produce parseable JSON. Caller surfaces this
        # via judge_parse_failures counter; faithfulness for this example is
        # genuinely unknowable rather than zero.
        return FaithfulnessScore(
            overall=None, verdicts=[],
            judge_model=ollama_client.model, raw_response=last_raw,
            attempts=result.attempts, judge_parse_failed=True,
        )

    verdicts = [
        FaithfulnessVerdict(
            docid=v["docid"],
            supported=bool(v["supported"]),
            reason=str(v.get("reason", ""))[:200],
        )
        for v in result.parsed.get("verdicts", [])
    ]
    overall = result.parsed.get("overall_supported_fraction")
    if overall is None and verdicts:
        overall = sum(1 for v in verdicts if v.supported) / len(verdicts)

    return FaithfulnessScore(
        overall=overall, verdicts=verdicts,
        judge_model=ollama_client.model, raw_response=last_raw,
        attempts=result.attempts, judge_parse_failed=False,
    )


def calibrate_judge_against_manual_labels(
    judge_scores: list[FaithfulnessScore],
    manual_labels: list[dict[str, bool]],
) -> dict[str, float]:
    """Compute agreement % between the judge and a hand-labelled spot-check set.

    Args:
        judge_scores: List of ``FaithfulnessScore`` outputs to compare.
        manual_labels: Parallel list of dicts ``{docid: supported_bool}`` of
            manual ground-truth labels for the citations in each answer.

    Returns:
        ``{"agreement": fraction, "n_citations_compared": int}`` for the
        intersection of citation docids the judge scored and the manual set
        covers.
    """
    total = 0
    agreed = 0
    for score, manual in zip(judge_scores, manual_labels):
        for v in score.verdicts:
            if v.docid in manual:
                total += 1
                if v.supported == manual[v.docid]:
                    agreed += 1
    return {
        "agreement": agreed / total if total > 0 else 0.0,
        "n_citations_compared": total,
    }

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

_DEFAULT_JUDGE_MODEL = "claude-3-5-haiku-latest"
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
) -> FaithfulnessScore:
    """Score citation faithfulness for one (question, answer) pair.

    Returns a ``FaithfulnessScore`` with ``overall=None`` when the answer is a
    refusal or has no citations.

    Raises:
        EnvironmentError: If the API key is missing and no client supplied.
    """
    if answer.strip().startswith(_REFUSAL_TEXT) or not citations:
        return FaithfulnessScore(
            overall=None,
            verdicts=[],
            judge_model=judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL),
            raw_response="",
        )

    resolved_model = judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)

    if client is None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set and no client was supplied."
            )
        client = anthropic.Anthropic(api_key=resolved_key)

    user_message = _USER_TEMPLATE.format(
        question=question,
        answer=answer,
        chunks_block=_build_chunks_block(citations),
    )

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
        overall=overall,
        verdicts=verdicts,
        judge_model=resolved_model,
        raw_response=raw,
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

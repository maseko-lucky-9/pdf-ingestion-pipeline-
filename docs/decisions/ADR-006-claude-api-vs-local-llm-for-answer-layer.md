# ADR-006 — Claude API vs local LLM for the answer layer

**Date:** 2026-05-16 (Ollama fallback wired 2026-05-22)
**Status:** Accepted, fallback implemented
**Decision-makers:** Thulani Maseko
**Phase reference:** AI Agent Portfolio Phase 3 (see `~/.claude/plans/create-a-comprehensive-plan-buzzing-backus.md`)

## Status note — 2026-05-22

The Ollama fallback documented below is **implemented** as of this date. Provider selection:

- `LLM_PROVIDER=anthropic|ollama` (explicit) always wins.
- Else, `ANTHROPIC_API_KEY` set → Anthropic.
- Else, auto-fallback → Ollama (`gpt-oss-20b` for answers, `llama3.1:8b` for the judge by default; both env-overridable via `OLLAMA_ANSWER_MODEL` / `OLLAMA_JUDGE_MODEL`).

**Production hardening:** systemd unit / Cloudflare Access tunnel must set `LLM_PROVIDER=anthropic` so a missing `.env` doesn't silently degrade to local quality.

**Demo UX:** when the resolved model is not a Claude variant, the citation UI shows a yellow banner ("Running on local model — quality and latency lower than production").

**Judge robustness:** the Ollama judge uses a 3-attempt retry loop with progressively stricter "JSON only" system prompts. After 3 failures, `faithfulness=None` is surfaced with a `judge_parse_failed=True` flag — the caller increments a `judge_parse_failures` metric so the noise is visible rather than silent.

**Quality delta:** the side-by-side measurement against the 25 quant-finance + 15 SA-legislation labelled queries is the responsibility of the operator (run `llm-eval compare` per the sibling `~/Repo/apps/llm-eval-harness` README). Honest expectation: gpt-oss-20b on M2 produces 30-90s answers vs Claude's 2-3s, and recall/citation faithfulness drops measurably. Use for dev/CI/internal eval; sales demos default to Claude.

## Context

Phase 3 of the AI Agent Portfolio adds an **answer layer** on top of the existing retrieval pipeline (Docling extraction → Ollama embeddings → sqlite-vec + FTS5 + RRF fusion + optional cross-encoder rerank). The retriever returns scored `SearchResult` chunks today; this phase makes the system answerable by synthesising a cited natural-language response.

Two implementation paths exist:

1. **Hosted Claude API.** Use Anthropic's `claude-3-5-sonnet-latest` via the `anthropic` Python SDK. Single network call per query.
2. **Local LLM via Ollama.** A quantised model (e.g. Llama 3.1 8B, Qwen 2.5 7B) running on the same machine that already hosts the embeddings model.

Both paths produce a natural-language answer with `[doc-N]` citations the retriever can resolve back to source chunks. The choice affects latency, cost, data residency, and demo reliability.

## Decision

**The Claude API is the primary answer layer for the demo path.**

Local Ollama remains a documented fallback for prospects with strict data-sovereignty requirements (POPIA-regulated personal information, on-prem-only enterprises, air-gapped environments). The `synthesize_answer()` function is designed so the LLM call site is a single point that can be swapped to Ollama without touching the rest of the pipeline.

## Why the Claude API wins for the demo

| Factor | Claude API | Local Ollama (Llama/Qwen 7B) |
|---|---|---|
| First-token latency | ~600–900 ms on broadband | 300–600 ms on M-series Mac; 2–5 s on CPU-only homelab |
| Answer quality on extractive citing | SOTA on prompt-following + refusal | Solid for paraphrase; weaker on strict "cite every claim" prompts |
| Citation-tag adherence | High — `[doc-N]` tags are honoured | Mixed — small models invent tags or skip them |
| Infrastructure cost | Pay-per-token (~$0.003/1K input) | Zero marginal; ~16 GB RAM committed |
| Cold-start risk | None — stateless API | Real — Ollama daemon must be warm; cold load is 5–15 s |
| Data residency | Anthropic servers (US/EU regions) | Fully on-prem |
| Demo reliability | Stable across networks | Depends on local hardware + daemon health |

For a sales demo where every second of stall is visible to the prospect, the Claude API's stability plus its tighter citation adherence wins. The marginal cost per demo session is negligible (R0.01–R0.05 typical).

## Prompt design

Three opinionated choices, captured here so they're stable across deployments:

1. **Extractive `[doc-N]` citing.** Each retrieved chunk is numbered in the prompt. The model is instructed to cite every factual claim with the matching tag. A regex pass over the answer recovers which chunks were referenced — no LLM-based citation parsing needed.
2. **Hard refusal when the context is insufficient.** The system prompt mandates a fixed response (`"I cannot answer this question from the provided context."`) when the retriever returns no useful context. Soft hedges ("Based on the available information…") are explicitly prohibited so prospects can immediately see system limits.
3. **System prompt forbids parametric knowledge blending.** The model must answer only from the supplied context. This keeps the eval harness honest — if the retriever fails, the answer must reflect that failure rather than papering over it.

## Trade-offs accepted

- **Data residency:** Anthropic processes the query + context. For prospects with POPIA-regulated personal information in their corpora, the Ollama fallback is documented and the swap point is a single function. Future work (Phase 3 slice 2) wires the actual Ollama path.
- **API dependency:** If Anthropic has an outage during a demo, the live system is unavailable. Mitigation per the Phase 2–6 framing: every demo phase ships a fallback recording (`docs/sales-demo.md` documents the pivot).
- **Cost:** Token spend scales with usage. The Anthropic dashboard usage cap covers runaway scenarios. Demo-only usage is well within free-tier-equivalent costs.

## Implementation references

- Answer module: `src/answer.py` — `synthesize_answer()`, `Citation`, `AnsweredQuery`.
- REST endpoint: `src/api/server.py` — `POST /query`.
- Logging: `src/observability.py` — single `log_rag_request()` emit point, OTEL-ready.
- Tests: `tests/test_answer.py` (16), `tests/test_api_server.py` (16) — all mock the Anthropic SDK.
- Sales demo script: `docs/sales-demo.md`.

## Cross-reference

This mirrors `ai-hedge-fund/docs/decisions/ADR-006` (LangGraph vs Claude Agent SDK) in framing: in both repos, the architectural choice maps directly to a sales-narrative point (production-grade vs toy). Keeping the ADR numbers in sync across repos is intentional — both are "Phase X first-slice" decisions in the broader portfolio.

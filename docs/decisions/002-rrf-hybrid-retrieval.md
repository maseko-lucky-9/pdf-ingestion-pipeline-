# ADR 002 — Hybrid retrieval: RRF fusion of BM25 + vector

**Status:** Accepted  
**Date:** 2026-04-19

## Problem

Retrieval must handle both keyword-heavy queries (indicator names, formula acronyms like "RSI", "MACD") and semantic queries ("how does mean reversion work"). Neither BM25 alone nor vector search alone covers both well.

## Options considered

| Option | Keyword | Semantic | Complexity |
|--------|---------|----------|-----------|
| BM25 only (FTS5) | Excellent | Poor | Low |
| Vector only | Poor on rare terms | Good | Low |
| Linear score fusion | Good | Good | Medium — requires score normalisation across incompatible spaces |
| **RRF (Reciprocal Rank Fusion)** | Good | Good | Low — rank-based, no normalisation |
| Learned sparse (SPLADE) | Excellent | Good | High — extra model, GPU preferred |

## Decision

**RRF with k=60.** Rank-based fusion avoids the score normalisation problem entirely — BM25 scores (negative, log-scale) and cosine distances are incommensurable; RRF ignores magnitudes. k=60 is the empirically validated default from the original Cormack et al. paper.

Parameters in `config.yaml`: `bm25_k: 50`, `vector_k: 50`, `rrf_k: 60`, `final_k: 10`.

## Consequences

- **Baseline measured:** MRR@10=0.944, NDCG@10=0.880 on 9-query quant-finance eval set (RRF only, no reranker).
- **Reranker tested:** `cross-encoder/ms-marco-MiniLM-L-6-v2` hurt: MRR@10=0.870, NDCG@10=0.843. Regressions on q004 (Sharpe ratio) and q007 (cointegration). Root cause: ms-marco is trained on web search, not financial text. **Reranking disabled by default.**
- To recover: either fine-tune a cross-encoder on financial QA pairs, or evaluate `BAAI/bge-reranker-v2-m3` (multilingual, more domain-general).
- If BM25 becomes a bottleneck (large corpora, many concurrent queries), replace FTS5 with a dedicated BM25 service; RRF logic in `retriever.py` stays unchanged.

# ADR 003 — Atomic chunking: tables/formulas/code never split

**Status:** Accepted  
**Date:** 2026-04-19

## Problem

Standard fixed-size or sliding-window chunkers split documents purely by token budget. Financial PDFs contain tables (performance matrices, indicator parameters), formulas (Sharpe, RSI, Bollinger Band definitions), and code blocks (TradeStation EasyLanguage). Splitting these mid-element destroys their meaning.

## Options considered

| Strategy | Atomic preservation | Token budget | Complexity |
|----------|--------------------|-----------|-|
| Fixed 900-token window | No | Strict | Low |
| Sentence-aware | Partial | Approximate | Medium |
| **Atomic-aware (this)** | Yes | Approximate | Medium |
| Semantic chunking (embedding similarity) | Partial | Variable | High |

## Decision

**Atomic-aware chunker in `src/pipeline/chunker.py`.**

Rules:
1. Items typed `table | formula | code` by Docling emit their own chunk verbatim, regardless of token count.
2. `text` items aggregate into a ≤900-token buffer, flushed at item boundaries with 15% token overlap.
3. A single oversized text item is sliced into 900-token windows (no paragraph-boundary assumption — the token window is the boundary).

The chunker consumes Docling's typed `Item` stream produced by `normalizer.py`, not a flat string. This is a clean break from `adaptive_chunk` in the benchmarks.

## Consequences

- **Atomicity invariant** is enforced by `tests/test_chunker.py` (6 tests). Any regression breaks CI.
- Tables with >900 tokens (rare but possible in TSaM's performance matrices) get their own chunk — retrievable intact but may score lower on BM25 due to low text density.
- Overlap is text↔text only. No overlap at atomic boundaries — an atomic block is always self-contained.
- If formula extraction quality is poor (Docling OCR error), the formula chunk will contain garbled math. Fix is upstream in Docling config, not in the chunker.

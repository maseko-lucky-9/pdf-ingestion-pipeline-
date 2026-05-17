# ADR 001 — Vector store: sqlite-vec over faiss-cpu

**Status:** Accepted  
**Date:** 2026-04-19

## Problem

Phase 1 used `faiss-cpu` for vector search. Phase 2 needed a store that could co-locate vector, BM25, and metadata in one file per collection, support idempotent upserts, and run without a daemon.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| **faiss-cpu** | Fast ANN, well-known | No SQL integration, no upsert, separate file from FTS/meta |
| **sqlite-vec** | Single `.db` file, SQL joins with FTS5/meta, upsert via DELETE+INSERT, zero daemon | Exact NN only (no HNSW), slower at >1M vectors |
| **Qdrant (local)** | HNSW, filtering, REST API | Daemon required, separate process, overkill for single-user |
| **Chroma** | Easy API | Separate store, less SQL-native |

## Decision

**sqlite-vec.** A single `index.db` per collection unifies vec, FTS5, and meta — enabling the RRF JOIN without cross-store round trips. At the corpus sizes in scope (<100k chunks), exact NN is fast enough. The tradeoff (no HNSW) only matters above ~1M vectors, at which point the architecture needs rethinking anyway.

## Consequences

- `faiss-cpu` removed from `requirements.txt`; `benchmark_pipeline.py` retains its faiss import (benchmark only, not production path).
- Embedding dim is templated from `config.yaml → ollama.embed_dim` into DDL at schema creation time to avoid config/schema divergence if the model changes.
- Migration path if scale requires HNSW: export vec table → load into Qdrant, keep FTS5/meta in SQLite, add a thin adapter in `retriever.py`.

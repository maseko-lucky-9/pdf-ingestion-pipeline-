# PDF Ingestion Pipeline — Task Tracker

Source plan: `/Users/ltmas/.claude/plans/review-plan-for-gaps-improvements-distributed-eich.md`

## Phase 1 — Benchmarks & Utilities (DONE)

- [x] Initial repo scaffold (`ca248ad`)
- [x] `requirements.txt` pinned
- [x] Benchmark harness for extraction latency

## Phase 2 — Core Pipeline (DONE)

### Ingest path
- [x] `src/config.py` — Pydantic v2 config loader (`config.yaml`)
- [x] `src/models/document.py` — `Item`, `Chunk`, `Collection` models with `content_hash`
- [x] `src/pipeline/router.py` — `is_scanned()` via pypdfium2 text probe
- [x] `src/pipeline/extractor.py` — Docling extraction → typed `Item` stream
- [x] `src/pipeline/normalizer.py` — header/footer dedup (60% threshold)
- [x] `src/pipeline/chunker.py` — atomic-aware chunker (tables/formulas verbatim)
- [x] `src/pipeline/embedder.py` — Ollama batch embedder with preflight + fail-fast
- [x] `src/pipeline/indexer.py` — sqlite-vec + FTS5 writer; INSERT-OR-REPLACE idempotent
- [x] `src/ingest.py` — CLI entry point

### Retrieval path
- [x] `src/pipeline/retriever.py` — RRF fusion + cross-encoder rerank (R5 device auto-detect)
- [x] `src/query.py` — CLI entry point

### Evaluation
- [x] `src/eval/bind_labels.py` — bind query draft to docids after first ingest
- [x] `src/eval/run_eval.py` — MRR@10 + NDCG@10 runner

### Tests (initial)
- [x] `tests/test_normalizer.py`
- [x] `tests/test_chunker.py`
- [x] `tests/test_embedder.py`
- [x] `tests/test_indexer.py`

### Documentation & Config
- [x] `CLAUDE.md` at repo root
- [x] `config.yaml` with all tunables

### Gate closed
- [x] G6 (slow extraction) — confirmed 5.94s on `mixed_content.pdf`
- [x] faiss-cpu removed; sqlite-vec is the vector backend
- [x] Embedding dim templated from `config.yaml` → `ollama.embed_dim`

## Phase 2 — Completion Pass (this session)

- [x] T001 — `tests/test_router.py` (8 tests; mock pypdfium2)
- [x] T002 — `tests/test_retriever.py` (13 tests; mock ollama + sqlite-vec + CrossEncoder)
- [x] T003 — `src/eval/queries_quant_finance.json` (22 entries across 5 domains, ≥4 each)
- [x] T004 — `tasks/todo.md` (this file)
- [x] T005 — Smoke verification commands documented

### Verification commands (run locally before marking Phase 2 closed)

```bash
cd /Users/ltmas/Repo/pdf-ingestion-pipeline
source .venv/bin/activate

# 1. CLI help shows
python -m src.ingest --help

# 2. Full test suite green (expect ≥24 tests total)
python -m pytest tests/ -v

# 3. Eval query set is well-formed
python -c "import json, collections; \
  data = json.load(open('src/eval/queries_quant_finance.json')); \
  print('total:', len(data)); \
  print('by_domain:', dict(collections.Counter(e['domain'] for e in data)))"
```

Expected eval counts: `total: 22`, `technical-analysis: 4, algorithmic: 5, forex: 4, ml-finance: 4, psychology: 4`.

## Phase 3 — Runtime Validation (DONE)

- [x] Ollama verified — `nomic-embed-text` model available
- [x] Corpus ingested — `data/quant_pdfs/` → `collections/quant-finance/` (2 PDFs, 1836 chunks)
- [x] Labels validated — q001-q010 have pre-bound docids; q011-q022 domains (forex, ml-finance, psychology) not in source PDFs
- [x] Evaluation run — **MRR@10: 0.944, NDCG@10: 0.880**

### Evaluation Results (2026-04-19)

| Query | Domain | MRR@10 | NDCG@10 |
|-------|--------|--------|---------|
| q001 RSI | technical-analysis | 1.000 | 0.853 |
| q002 EMA/SMA | technical-analysis | 1.000 | 0.885 |
| q003 mean reversion | algorithmic | 1.000 | 0.947 |
| q004 Sharpe ratio | algorithmic | 1.000 | 0.906 |
| q006 MACD | technical-analysis | 1.000 | 0.885 |
| q007 cointegration | algorithmic | 0.500 | 0.733 |
| q008 Bollinger | technical-analysis | 1.000 | 0.850 |
| q009 information ratio | algorithmic | 1.000 | 0.906 |
| q010 drawdown | algorithmic | 1.000 | 0.956 |
| **AVG** | | **0.944** | **0.880** |

**Notes:**
- q007 (cointegration) is the only query with MRR < 1.0 — relevant docs not at rank 1
- Reranker disabled for baseline (config: `rerank.enabled: false`)
- Queries q011-q022 are unlabeled; source PDFs don't cover forex, ML-finance, or trading psychology domains

## Review

Phase 2 complete: test suite green (38 tests), eval harness functional. Phase 3 complete: corpus ingested, baseline metrics captured. The pipeline is functional with strong retrieval performance on the technical analysis and algorithmic trading domains present in the source PDFs.

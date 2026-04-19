# pdf-ingestion-pipeline

## Tech Stack
- Python 3.11, Docling (extraction), Ollama `nomic-embed-text` (embeddings)
- sqlite-vec (vector index) + FTS5 (BM25) + RRF fusion + cross-encoder rerank
- Pydantic v2 models, PyYAML config, Rich CLI output

## Primary Entry Points
- `python -m src.ingest <pdf_dir> --collection <name>` — ingest PDFs
- `python -m src.query <question> --collection <name>` — retrieve chunks
- `python -m src.eval.run_eval --collection <name>` — evaluate retrieval

## Build / Test
```bash
python -m pytest tests/ -v
```

## Project Structure
```
src/
  ingest.py          # CLI: ingest PDFs into a collection
  query.py           # CLI: query a collection
  models/
    document.py      # Pydantic models: Item, Chunk, Collection
  pipeline/
    router.py        # Detect scanned vs native PDF
    extractor.py     # Docling extraction → typed Item stream
    normalizer.py    # Header/footer dedup (60%-threshold algo)
    chunker.py       # Atomic-aware chunker (tables/formulas verbatim)
    embedder.py      # Ollama batch embedder with preflight
    indexer.py       # sqlite-vec + FTS5 writer; INSERT OR REPLACE
    retriever.py     # RRF fusion + cross-encoder rerank
  eval/
    bind_labels.py   # Bind query draft to docids after first ingest
    run_eval.py      # MRR@10, NDCG@10 evaluation runner
collections/         # Runtime: collections/<name>/index.db
config.yaml          # All tunables — edit this, not source files
```

## Key Decisions
- G6 (slow extraction): CLOSED — confirmed 5.94s on mixed_content.pdf.
- faiss-cpu removed; sqlite-vec is the vector backend.
- Embedding dim templated from `config.yaml` → `ollama.embed_dim`; change there to avoid schema divergence.

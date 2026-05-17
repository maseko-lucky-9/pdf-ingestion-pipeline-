"""FastAPI app exposing the RAG pipeline over HTTP.

Routes:
    GET  /health        Liveness + list of collections currently on disk.
    POST /query         Retrieve + synthesize a cited answer.

Boot with:
    uvicorn src.api.server:app --reload --port 8000
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.answer import AnsweredQuery, synthesize_answer
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.config import load_config
from src.observability import configure_logging, log_rag_request, new_request_id
from src.pipeline.retriever import retrieve

_DEFAULT_COLLECTIONS_ROOT = "./collections"


def _collections_dir() -> Path:
    """Resolve the collections root from env or config defaults.

    Wrapped in a function (not a module-level constant) so tests can patch
    it per test without restarting the app.
    """
    override = os.environ.get("COLLECTIONS_DIR")
    if override:
        return Path(override)
    return Path(_DEFAULT_COLLECTIONS_ROOT)


def _list_collections() -> list[str]:
    """Names of subdirectories under `collections/` that contain an index.db."""
    root = _collections_dir()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and (child / "index.db").exists()
    )


def _collection_db(collection: str) -> Path:
    return _collections_dir() / collection / "index.db"


configure_logging()

app = FastAPI(
    title="pdf-ingestion-pipeline RAG API",
    version="0.1.0",
    description="Retrieve + synthesize cited answers from ingested PDF collections.",
)

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
# `allow_credentials=True` is incompatible with `allow_origins=["*"]` per the
# CORS spec (browsers refuse the combo). Only enable credentials when the
# operator has narrowed origins explicitly via the env var.
_cors_allow_credentials = _cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe that also lists which collections are queryable."""
    return HealthResponse(status="ok", collections=_list_collections())


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Retrieve top-k chunks for `req.query` and synthesize a cited answer."""
    request_id = new_request_id()
    db_path = _collection_db(req.collection)

    if not db_path.exists():
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="404", latency_ms=0,
            error=f"collection {req.collection!r} not found",
        )
        raise HTTPException(
            status_code=404,
            detail=f"Collection {req.collection!r} not found.",
        )

    cfg = load_config()
    t0 = time.perf_counter()

    try:
        results = retrieve(req.query, db_path, cfg)
        # Trim to the user-requested k (retriever may return more or fewer)
        results = results[: req.k]
        answered: AnsweredQuery = synthesize_answer(req.query, results)
    except EnvironmentError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="503", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    total_latency_ms = (time.perf_counter() - t0) * 1000
    log_rag_request(
        request_id=request_id,
        collection=req.collection,
        query=req.query,
        k=req.k,
        status="200",
        latency_ms=total_latency_ms,
        hit_count=len(results),
        prompt_tokens=answered.prompt_tokens,
        completion_tokens=answered.completion_tokens,
    )

    return QueryResponse(
        answer=answered.answer,
        citations=answered.citations,
        model=answered.model,
        prompt_tokens=answered.prompt_tokens,
        completion_tokens=answered.completion_tokens,
        latency_ms=answered.latency_ms,
        request_id=request_id,
    )

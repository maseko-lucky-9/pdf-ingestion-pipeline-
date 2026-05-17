"""FastAPI app exposing the RAG pipeline over HTTP.

Routes:
    GET  /health        Liveness + list of collections currently on disk.
    POST /query         Retrieve + synthesize a cited answer.

Boot with:
    uvicorn src.api.server:app --reload --port 8000
"""

import os
import sqlite3
import time
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
        retrieve_ms = (time.perf_counter() - t0) * 1000
        t_synth = time.perf_counter()
        answered: AnsweredQuery = synthesize_answer(req.query, results)
        synth_ms = (time.perf_counter() - t_synth) * 1000
    except EnvironmentError as exc:
        # Misconfiguration (no ANTHROPIC_API_KEY etc.) — server-side failure
        # under-the-hood but caller can retry once the operator fixes it.
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="503", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except anthropic.RateLimitError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="429", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=429, detail="Upstream rate-limited.") from exc
    except anthropic.APITimeoutError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="504", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=504, detail="Upstream timed out.") from exc
    except anthropic.APIStatusError as exc:
        # Bad request, auth error, server error, etc. — mirror upstream status
        # when reasonable, otherwise 502.
        latency_ms = (time.perf_counter() - t0) * 1000
        upstream_status = getattr(exc, "status_code", None) or 502
        # We never want to leak 5xx as 4xx (or vice versa) — clamp class.
        if upstream_status >= 500:
            our_status = 502
        elif upstream_status == 401 or upstream_status == 403:
            our_status = 503  # treat upstream auth issues as server misconfig
        else:
            our_status = upstream_status
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status=str(our_status), latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=our_status, detail=str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="502", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=502, detail="Upstream unreachable.") from exc

    total_latency_ms = (time.perf_counter() - t0) * 1000
    refused = answered.answer.strip().startswith(
        "I cannot answer this question from the provided context."
    )
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
        retrieve_ms=round(retrieve_ms, 1),
        synth_ms=round(synth_ms, 1),
        answer_len=len(answered.answer),
        citations_count=len(answered.citations),
        refused=refused,
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


def _resolve_pdf_for_docid(collection: str, docid: str) -> tuple[Path, int]:
    """Return (absolute_pdf_path, page_start) for the given chunk docid.

    Raises HTTPException 404 if the collection, chunk, or source PDF is
    missing. Resolves the PDF path under the configured ``data/`` tree —
    relative paths in the meta table are resolved against the project root.
    """
    db_path = _collection_db(collection)
    if not db_path.exists():
        raise HTTPException(404, f"Collection {collection!r} not found.")

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.execute(
            "SELECT source_pdf, page_start FROM meta WHERE docid = ?",
            (docid,),
        )
        row = cur.fetchone()
    finally:
        con.close()

    if row is None:
        raise HTTPException(404, f"docid {docid!r} not found in {collection!r}.")

    source_rel, page_start = row[0], int(row[1])

    candidate = Path(source_rel)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists():
        raise HTTPException(404, f"Source PDF missing on disk: {source_rel!r}")

    return candidate, page_start


@app.get("/document/{collection}/{docid}")
def document(collection: str, docid: str) -> FileResponse:
    """Serve the source PDF for a cited chunk.

    The browser opens the result inline so callers can append ``#page=N`` to
    jump to the cited page. Page resolution happens client-side; the server's
    job is just to deliver the right file.
    """
    pdf_path, _page = _resolve_pdf_for_docid(collection, docid)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{pdf_path.name}"'},
    )


@app.get("/document/{collection}/{docid}/page")
def document_page(collection: str, docid: str) -> dict:
    """Return the cited page number for a chunk so the UI knows where to jump."""
    _pdf_path, page_start = _resolve_pdf_for_docid(collection, docid)
    return {"page": page_start}


# Mount the static UI LAST so it does not shadow `/health`, `/query`, or
# `/document/*`. FastAPI dispatches more-specific routes first regardless,
# but ordering keeps the intent obvious.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

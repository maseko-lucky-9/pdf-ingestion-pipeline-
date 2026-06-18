"""FastAPI app exposing the RAG pipeline over HTTP.

Routes:
    GET  /health                          Liveness + list of collections on disk.
    POST /query                           Retrieve + synthesize a cited answer.
    GET  /document/{collection}/{docid}   Serve source PDF for a cited chunk.
    GET  /document/{collection}/{docid}/page  Return cited page number.

Boot with:
    uvicorn src.api.server:app --reload --port 8000
"""

import logging
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.answer import AnsweredQuery, synthesize_answer
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.config import load_config
from src.llm_provider import (
    LlmProviderError,
    probe_ollama_reachable,
    resolve_llm_provider,
)
from src.observability import configure_logging, log_rag_request, new_request_id
from src.pipeline.retriever import retrieve

log = logging.getLogger(__name__)

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
    """Resolve the DB path and verify it stays inside the collections root."""
    root = _collections_dir().resolve()
    candidate = (root / collection / "index.db").resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Invalid collection name.")
    return candidate


configure_logging()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Startup probe + structured log line for the resolved LLM provider.

    Runs in the FastAPI lifespan (not at module import) so unit tests that
    just import ``src.api.server:app`` with no Ollama daemon present don't
    blow up — only the actually-running server triggers the probe.

    Set ``RAG_SKIP_STARTUP_PROBE=1`` to bypass (used in tests that boot
    the app via ``TestClient`` without a daemon).
    """
    provider = resolve_llm_provider()
    cfg = load_config()
    log.info(
        "llm_provider_resolved",
        extra={
            "provider": provider,
            "answer_model": (
                os.environ.get("ANTHROPIC_MODEL") if provider == "anthropic"
                else cfg.ollama.answer_model
            ),
            "judge_model": (
                os.environ.get("ANTHROPIC_JUDGE_MODEL") if provider == "anthropic"
                else cfg.ollama.judge_model
            ),
        },
    )
    if provider == "ollama" and not os.environ.get("RAG_SKIP_STARTUP_PROBE"):
        # Fail fast at startup rather than per-request 5xx loops downstream.
        probe_ollama_reachable(cfg.ollama.host)
    yield


app = FastAPI(
    title="pdf-ingestion-pipeline RAG API",
    version="0.1.0",
    description="Retrieve + synthesize cited answers from ingested PDF collections.",
    lifespan=_lifespan,
)

_cors_origins_raw = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if _cors_origins:
    # `allow_credentials=True` is incompatible with `allow_origins=["*"]` per
    # the CORS spec. Only enable credentials when origins are explicitly listed.
    _cors_allow_credentials = _cors_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def _security_headers(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-src 'self'"
    )
    return response


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
        # Misconfiguration (no ANTHROPIC_API_KEY etc.) — server-side failure;
        # log detail internally but return a generic message to callers.
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="503", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=503, detail="Service configuration error.") from exc
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
        # when reasonable, otherwise 502. Never leak upstream body to callers.
        latency_ms = (time.perf_counter() - t0) * 1000
        upstream_status = getattr(exc, "status_code", None) or 502
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
        raise HTTPException(status_code=our_status, detail="Upstream service error.") from exc
    except anthropic.APIConnectionError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="502", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=502, detail="Upstream unreachable.") from exc
    except LlmProviderError as exc:
        # Ollama-path failures: the client already mapped httpx errors to a
        # status_code on the exception. Default to 502 when unknown so the
        # caller distinguishes "transient upstream" from a 4xx client error.
        # Never leak internal Ollama error details to callers.
        latency_ms = (time.perf_counter() - t0) * 1000
        our_status = exc.status_code or 502
        # Clamp into safe ranges. 404 from Ollama means "model not pulled" —
        # that is operator configuration, not a missing collection (which we
        # already handled above), so remap to 503.
        if our_status in (401, 403, 404):
            our_status = 503
        elif our_status >= 500 and our_status != 504 and our_status != 503:
            our_status = 502
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status=str(our_status), latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=our_status, detail="Upstream service error.") from exc
    except httpx.TimeoutException as exc:
        # Should normally be caught by LlmProviderError; this is the safety net.
        latency_ms = (time.perf_counter() - t0) * 1000
        log_rag_request(
            request_id=request_id, collection=req.collection, query=req.query,
            k=req.k, status="504", latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=504, detail="Upstream timed out.") from exc
    except httpx.HTTPError as exc:
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
    missing. Validates the resolved path is a .pdf file so DB-stored paths
    cannot be used to serve arbitrary files.
    """
    db_path = _collection_db(collection)
    if not db_path.exists():
        raise HTTPException(404, f"Collection {collection!r} not found.")

    with sqlite3.connect(str(db_path)) as con:
        cur = con.execute(
            "SELECT source_pdf, page_start FROM meta WHERE docid = ?",
            (docid,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(404, f"docid {docid!r} not found in {collection!r}.")

    source_rel, page_start = row[0], int(row[1])

    candidate = Path(source_rel)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()

    if candidate.suffix.lower() != ".pdf":
        raise HTTPException(404, "Source document not available.")
    if not candidate.exists():
        raise HTTPException(404, "Source document not available.")

    return candidate, page_start


@app.get("/document/{collection}/{docid}")
def document(collection: str, docid: str) -> FileResponse:
    """Serve the source PDF for a cited chunk.

    The browser opens the result inline so callers can append ``#page=N`` to
    jump to the cited page. Page resolution happens client-side; the server's
    job is just to deliver the right file.
    """
    pdf_path, _page = _resolve_pdf_for_docid(collection, docid)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", pdf_path.name)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
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

"""Pydantic request/response models for the REST API.

These wrap (and constrain) the internal types from `src.answer` so the public
contract is explicit and validated. Defaults align with the retrieval config
in `src.config.RetrievalConfig`.
"""

from pydantic import BaseModel, Field

from src.answer import Citation

# Defaults intentionally permissive; the retriever's own k caps still apply.
_DEFAULT_K = 5
_MAX_K = 50
_MAX_QUERY_LEN = 1000


class HealthResponse(BaseModel):
    status: str = "ok"
    collections: list[str] = Field(
        default_factory=list,
        description="Names of collections found on disk (directories under collections/ that contain index.db).",
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LEN)
    collection: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    k: int = Field(_DEFAULT_K, ge=1, le=_MAX_K)


class QueryResponse(BaseModel):
    """Mirrors `src.answer.AnsweredQuery` for the public API surface."""

    answer: str
    citations: list[Citation]
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    request_id: str

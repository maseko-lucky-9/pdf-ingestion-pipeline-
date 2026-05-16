"""Structured logging for the answer layer + REST API.

A single emit point (`log_rag_request`) makes it easy to swap to OTEL later
without touching every call site. JSON output by default; configurable to
console-pretty via the `LOG_FORMAT=console` env var.
"""

import logging
import os
import uuid

import structlog


def _processors(json_format: bool) -> list:
    base = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_format:
        base.append(structlog.processors.JSONRenderer())
    else:
        base.append(structlog.dev.ConsoleRenderer())
    return base


def configure_logging() -> None:
    """Idempotent global logger config. Safe to call from any entry point."""
    if structlog.is_configured():
        return
    json_format = os.environ.get("LOG_FORMAT", "json").lower() == "json"
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=_processors(json_format),
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def new_request_id() -> str:
    """Return a fresh UUID4 hex string for correlating logs to one request."""
    return uuid.uuid4().hex


def log_rag_request(
    *,
    request_id: str,
    collection: str,
    query: str,
    k: int,
    status: str,
    latency_ms: float,
    hit_count: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error: str | None = None,
) -> None:
    """Single structured-log emit point for every /query call.

    Truncates the query to 200 chars so the log line stays bounded. To wire
    OTEL traces or Sentry breadcrumbs later, change only this function — the
    callers never need to know.
    """
    configure_logging()
    log = structlog.get_logger("rag")
    log.info(
        "rag.request",
        request_id=request_id,
        collection=collection,
        query=query[:200],
        k=k,
        status=status,
        latency_ms=round(latency_ms, 2),
        hit_count=hit_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=error,
    )

"""Shared pytest configuration."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _skip_ollama_startup_probe():
    """Bypass the Ollama daemon startup probe for all non-integration tests.

    The FastAPI lifespan calls ``probe_ollama_reachable()`` when the resolved
    provider is ``ollama`` (i.e. when ``ANTHROPIC_API_KEY`` is absent).  Unit
    tests run without a live daemon, so we set ``RAG_SKIP_STARTUP_PROBE`` for
    the entire session.  The escape hatch is intentional — see server.py
    lifespan and ``src/llm_provider.py:probe_ollama_reachable``.

    Ollama-integration tests exercise the live daemon directly and do not go
    through the FastAPI lifespan, so they are unaffected by this fixture.
    """
    prev = os.environ.get("RAG_SKIP_STARTUP_PROBE")
    os.environ["RAG_SKIP_STARTUP_PROBE"] = "1"
    yield
    if prev is None:
        os.environ.pop("RAG_SKIP_STARTUP_PROBE", None)
    else:
        os.environ["RAG_SKIP_STARTUP_PROBE"] = prev


def pytest_configure(config):
    """Register custom markers so `-m ollama_integration` doesn't warn.

    ``ollama_integration`` — tests that hit a real Ollama daemon. Skipped
    by default; only run under .github/workflows/ollama-integration.yml
    (which passes ``-m ollama_integration`` explicitly).
    """
    config.addinivalue_line(
        "markers",
        "ollama_integration: requires a real Ollama daemon (skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``ollama_integration`` tests unless ``-m ollama_integration`` is
    on the command line. This keeps the default ``pytest`` invocation
    hermetic (no live network) while the workflow_dispatch job opts in."""
    marker_expr = (config.getoption("-m") or "").strip()
    if "ollama_integration" in marker_expr:
        return  # caller explicitly selected the live tests
    skip_live = pytest.mark.skip(
        reason="ollama_integration: opt in via `pytest -m ollama_integration`"
    )
    for item in items:
        if "ollama_integration" in item.keywords:
            item.add_marker(skip_live)

"""Shared pytest configuration."""
from __future__ import annotations

import pytest


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

"""Atomicity invariant: tables/formulas/code must appear verbatim in exactly one chunk."""
import pytest
from src.config import Config
from src.models.document import Item
from src.pipeline.chunker import chunk_items


def _make_cfg(max_tokens: int = 900) -> Config:
    cfg = Config()
    cfg.chunker.max_tokens = max_tokens
    return cfg


def _kwargs():
    return dict(source_pdf="test.pdf", collection="test", domain="", book="test")


def test_table_emits_single_verbatim_chunk():
    table_text = "| A | B |\n|---|---|\n| 1 | 2 |"
    items = [Item(kind="table", text=table_text, page_range=(1, 1), token_count=20)]
    chunks = chunk_items(items, cfg=_make_cfg(), **_kwargs())
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].content == table_text


def test_formula_emits_single_verbatim_chunk():
    formula = r"E = mc^2"
    items = [Item(kind="formula", text=formula, page_range=(1, 1), token_count=5)]
    chunks = chunk_items(items, cfg=_make_cfg(), **_kwargs())
    formula_chunks = [c for c in chunks if c.chunk_type == "formula"]
    assert len(formula_chunks) == 1
    assert formula_chunks[0].content == formula


def test_large_table_stays_verbatim():
    """Tables are never split, even when token_count > max_tokens."""
    big_table = "| " + " | ".join(["col"] * 50) + " |\n" + ("| x " * 50 + "|\n") * 20
    items = [Item(kind="table", text=big_table, page_range=(1, 2), token_count=2000)]
    chunks = chunk_items(items, cfg=_make_cfg(max_tokens=100), **_kwargs())
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].content == big_table


def test_atomic_not_duplicated():
    """Each atomic block appears in exactly one chunk — no partial, no duplication."""
    formula = "\\int_0^\\infty e^{-x} dx = 1"
    items = [
        Item(kind="text", text="Some preamble text.", page_range=(1, 1), token_count=5),
        Item(kind="formula", text=formula, page_range=(1, 1), token_count=15),
        Item(kind="text", text="More text after.", page_range=(1, 1), token_count=5),
    ]
    chunks = chunk_items(items, cfg=_make_cfg(), **_kwargs())
    formula_chunks = [c for c in chunks if c.chunk_type == "formula"]
    assert len(formula_chunks) == 1
    # formula must not bleed into text chunks
    for c in chunks:
        if c.chunk_type == "text":
            assert formula not in c.content


def test_text_chunks_respect_token_budget():
    long_text = "word " * 1000  # ~1000 tokens
    items = [Item(kind="text", text=long_text, page_range=(1, 1), token_count=1000)]
    chunks = chunk_items(items, cfg=_make_cfg(max_tokens=200), **_kwargs())
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    for c in text_chunks:
        # allow overlap headroom
        assert c.token_count <= 400, f"chunk too large: {c.token_count}"


def test_code_emits_single_verbatim_chunk():
    code = "def foo():\n    return 42"
    items = [Item(kind="code", text=code, page_range=(2, 2), token_count=10)]
    chunks = chunk_items(items, cfg=_make_cfg(), **_kwargs())
    code_chunks = [c for c in chunks if c.chunk_type == "code"]
    assert len(code_chunks) == 1
    assert code_chunks[0].content == code

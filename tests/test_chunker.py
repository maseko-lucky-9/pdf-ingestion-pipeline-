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


# ─── Page-range advancement across chunks (slice 6 follow-up) ────────────────
#
# Pre-slice-6 bug: every chunk in a long doc inherited the document's first
# content page as `page_start`, because the 15% overlap kept the buffer
# non-empty and the page-start-reset branch never re-fired. The page-overlap
# label matcher in src/eval/run_eval.py then resolved labels too loosely,
# inflating recall metrics.

def _long_text_items_across_pages(n_pages: int, words_per_page: int = 400) -> list[Item]:
    """Build a sequence of text Items spanning n_pages, each large enough that
    several pages will fit inside a single 900-token chunk."""
    items: list[Item] = []
    for page in range(1, n_pages + 1):
        text = " ".join([f"page{page}word{i}" for i in range(words_per_page)])
        items.append(Item(kind="text", text=text, page_range=(page, page), token_count=words_per_page))
    return items


def test_chunk_page_starts_advance_across_long_doc():
    """After fix: chunks beyond the first should have page_start > 1.

    Pre-fix every chunk reported page_start=1 regardless of where in the
    document it actually lived.
    """
    items = _long_text_items_across_pages(n_pages=12)
    chunks = chunk_items(items, cfg=_make_cfg(max_tokens=900), **_kwargs())

    # We expect more than one chunk for a 12-page block of dense text.
    assert len(chunks) >= 3, f"expected multiple chunks, got {len(chunks)}"

    page_starts = [c.page_range[0] for c in chunks]
    # First chunk should start at page 1 (or near it).
    assert page_starts[0] == 1
    # Later chunks should NOT all still report page_start=1.
    assert any(p > 1 for p in page_starts[1:]), (
        f"all chunk page_starts stuck at 1: {page_starts}"
    )


def test_chunk_page_starts_are_non_decreasing():
    """Reading the doc in order, chunk N+1's page_start should be >= chunk N's
    page_start. The fix anchors the new chunk's start at the previous flush's
    page_end, which equals the latest page contributing to the carried overlap."""
    items = _long_text_items_across_pages(n_pages=20)
    chunks = chunk_items(items, cfg=_make_cfg(max_tokens=900), **_kwargs())

    page_starts = [c.page_range[0] for c in chunks]
    for i in range(1, len(page_starts)):
        assert page_starts[i] >= page_starts[i - 1], (
            f"chunk {i} page_start={page_starts[i]} < chunk {i-1} page_start={page_starts[i-1]}; "
            f"full sequence: {page_starts}"
        )


def test_chunk_page_range_is_valid():
    """Every chunk's page_end >= page_start."""
    items = _long_text_items_across_pages(n_pages=15)
    chunks = chunk_items(items, cfg=_make_cfg(max_tokens=900), **_kwargs())
    for c in chunks:
        s, e = c.page_range
        assert s >= 1, f"page_start {s} < 1 in chunk {c.chunk_index}"
        assert e >= s, f"chunk {c.chunk_index} has page_end {e} < page_start {s}"

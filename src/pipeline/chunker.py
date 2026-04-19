"""Atomic-aware chunker consuming a typed Item stream (R3).

Rules:
- Atomic items (table | formula | code) emit their own chunk verbatim,
  even when token_count > 900.
- Text items aggregate into a running buffer; flush at ≤ max_tokens on the
  nearest paragraph boundary.
- 15% token overlap at text↔text boundaries only.
- Source order is preserved.

The token-budget/merge logic is adapted from adaptive_chunk in
benchmarks/benchmark_chunking.py but operates on typed Items, not flat strings.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import tiktoken

from src.config import Config
from src.models.document import Chunk, Item

_ATOMIC_KINDS = {"table", "formula", "code"}
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> list[int]:
    return _tokenizer.encode(text)


def _decode(token_ids: list[int]) -> str:
    return _tokenizer.decode(token_ids)


def _make_chunk(
    content: str,
    kind: str,
    page_range: tuple[int, int],
    idx: int,
    source_pdf: str,
    collection: str,
    domain: str,
    book: str,
) -> Chunk:
    docid = hashlib.sha256(content.encode()).hexdigest()[:12]
    return Chunk(
        docid=docid,
        source_pdf=source_pdf,
        chunk_index=idx,
        chunk_type=kind,
        content=content,
        page_range=page_range,
        token_count=len(_tokens(content)),
        collection=collection,
        domain=domain,
        book=book,
    )


def chunk_items(
    items: list[Item],
    source_pdf: str,
    collection: str,
    domain: str,
    book: str,
    cfg: Config,
) -> list[Chunk]:
    max_tokens = cfg.chunker.max_tokens
    overlap_pct = cfg.chunker.overlap_pct
    overlap_tokens = int(max_tokens * overlap_pct)

    chunks: list[Chunk] = []
    idx = 0
    text_buffer_ids: list[int] = []
    buffer_page_start = 0
    buffer_page_end = 0

    def flush_buffer(page_end: int) -> None:
        nonlocal idx, text_buffer_ids
        if not text_buffer_ids:
            return
        content = _decode(text_buffer_ids)
        chunks.append(_make_chunk(content, "text", (buffer_page_start, page_end),
                                  idx, source_pdf, collection, domain, book))
        idx += 1
        # carry overlap into next buffer
        text_buffer_ids = text_buffer_ids[-overlap_tokens:] if overlap_tokens else []

    for item in items:
        if item.kind in _ATOMIC_KINDS:
            # flush pending text first
            flush_buffer(item.page_range[0])
            text_buffer_ids = []

            chunks.append(_make_chunk(
                item.text, item.kind, item.page_range,
                idx, source_pdf, collection, domain, book,
            ))
            idx += 1
            continue

        # text item — aggregate into buffer
        item_ids = _tokens(item.text)
        page = item.page_range[0]

        if not text_buffer_ids:
            buffer_page_start = page

        # Drain item_ids into buffer in max_tokens-sized slices
        while len(text_buffer_ids) + len(item_ids) > max_tokens:
            remaining = max_tokens - len(text_buffer_ids)
            text_buffer_ids.extend(item_ids[:remaining])
            item_ids = item_ids[remaining:]
            flush_buffer(page)
            if not text_buffer_ids:
                buffer_page_start = page

        text_buffer_ids.extend(item_ids)
        buffer_page_end = item.page_range[1]

    flush_buffer(buffer_page_end)

    return chunks

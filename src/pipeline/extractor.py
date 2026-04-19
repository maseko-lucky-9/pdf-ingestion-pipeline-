"""Extract a typed Item stream from a PDF using Docling."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator

import tiktoken
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocItemLabel

from src.models.document import Item

_tokenizer = tiktoken.get_encoding("cl100k_base")
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def extract_items(pdf_path: Path) -> list[Item]:
    """Convert a PDF to an ordered list of typed Items."""
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    items: list[Item] = []

    for element, _level in doc.iterate_items():
        label = getattr(element, "label", None)
        text = getattr(element, "text", "") or ""

        if not text.strip():
            continue

        # Determine page range from provenance if available
        prov = getattr(element, "prov", None) or []
        pages = [p.page_no for p in prov if hasattr(p, "page_no")]
        page_range = (min(pages), max(pages)) if pages else (0, 0)

        if label == DocItemLabel.TABLE:
            kind = "table"
            if hasattr(element, "export_to_markdown"):
                text = element.export_to_markdown()
        elif label == DocItemLabel.FORMULA:
            kind = "formula"
        elif label == DocItemLabel.CODE:
            kind = "code"
        else:
            kind = "text"

        items.append(Item(
            kind=kind,
            text=text,
            page_range=page_range,
            token_count=_token_count(text),
        ))

    return items

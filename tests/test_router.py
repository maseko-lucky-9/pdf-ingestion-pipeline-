"""Router: scanned-vs-native PDF classification (R1)."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.router import is_scanned


def _make_doc(page_texts: list[str]) -> MagicMock:
    """Build a mock PdfDocument where page[i].get_textpage().get_text_range() returns page_texts[i]."""
    doc = MagicMock()
    doc.__len__.return_value = len(page_texts)
    pages = []
    for t in page_texts:
        page = MagicMock()
        textpage = MagicMock()
        textpage.get_text_range.return_value = t
        page.get_textpage.return_value = textpage
        pages.append(page)
    doc.__getitem__.side_effect = lambda i: pages[i]
    doc.close = MagicMock()
    return doc


def test_native_pdf_classified_as_native():
    """All sampled pages yield extractable text → not scanned."""
    doc = _make_doc([
        "This is a paragraph of native text content with many words.",
        "Another page of real text content with plenty of characters.",
        "Third page also has extractable text content here.",
    ])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/native.pdf")) is False


def test_scanned_pdf_classified_as_scanned():
    """All sampled pages yield <10 chars → scanned."""
    doc = _make_doc(["", "   ", ""])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/scanned.pdf")) is True


def test_empty_pdf_treated_as_scanned():
    """Zero pages → treated as scanned (safe default for OCR pipeline)."""
    doc = _make_doc([])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/empty.pdf")) is True


def test_single_page_with_text_is_native():
    """A single-page PDF with real text → not scanned."""
    doc = _make_doc(["Single page native PDF with adequate text content."])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/single.pdf")) is False


def test_mixed_below_threshold_is_native():
    """3 sampled pages: 2 image-only + 1 with text → 2/3 = 66.7% < 80% → native."""
    doc = _make_doc([
        "",
        "A page with real extractable text content here.",
        "",
    ])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/mixed_low.pdf")) is False


def test_mixed_at_threshold_is_scanned():
    """5 sampled pages: 4 image-only + 1 with text → 4/5 = 80% → scanned.

    sample_pages=5 forces all 5 to be evaluated.
    """
    doc = _make_doc(["", "", "", "Only this page has real text content.", ""])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/mixed_high.pdf"), sample_pages=5) is True


def test_sample_pages_respected_when_doc_is_longer():
    """sample_pages caps the number of pages examined even for long documents."""
    # 10-page doc, all scanned, but we only sample 3 → still classified scanned.
    doc = _make_doc([""] * 10)
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/long.pdf"), sample_pages=3) is True
    # Only 3 pages should have been indexed.
    assert doc.__getitem__.call_count == 3


def test_short_whitespace_text_is_image_only():
    """A page with <10 non-whitespace chars is treated as image-only."""
    # "abc" stripped = 3 chars < 10 → counts as image-only.
    doc = _make_doc(["abc", "abc", "abc"])
    with patch("pypdfium2.PdfDocument", return_value=doc):
        assert is_scanned(Path("/fake/short.pdf")) is True

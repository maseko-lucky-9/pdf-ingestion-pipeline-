"""Detect whether a PDF is scanned (image-only) or native (text-layer) ."""
from __future__ import annotations
from pathlib import Path


def is_scanned(pdf_path: Path, sample_pages: int = 3) -> bool:
    """Return True if the PDF appears to be scanned (no extractable text).

    Strategy: use pypdfium2 to check the first N pages for text content.
    A page is considered image-only if it yields < 10 characters of text.
    If ≥ 80% of sampled pages are image-only → scanned.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    n = min(sample_pages, len(doc))
    if n == 0:
        return True

    image_only_count = 0
    for i in range(n):
        page = doc[i]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        if len(text.strip()) < 10:
            image_only_count += 1

    doc.close()
    return image_only_count / n >= 0.8

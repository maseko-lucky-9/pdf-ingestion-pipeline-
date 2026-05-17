"""OCR a scanned PDF in place, producing a text-bearing PDF the ingest
pipeline can extract from cleanly.

Usage:
    python scripts/ocr_preprocess.py <input.pdf> --out <output.pdf>
    python scripts/ocr_preprocess.py <folder>   --out <folder-ocr>

The output is the original page images with an invisible text layer dropped
on top (Tesseract's `pdf` output mode). `src.pipeline.router.is_scanned`
returns False on the output, so the existing ingest pipeline picks it up.

Dependencies (declared in requirements.txt but imported lazily so the rest
of the project works without them):
  - pytesseract        — bindings to the `tesseract` binary
  - pdf2image          — wraps `pdftoppm` (poppler) to rasterise pages
  - tesseract binary   — `brew install tesseract` on macOS
  - poppler binary     — `brew install poppler` on macOS

Slice 4 #15. Realistic timing on M2: ~6-10 sec per page (300 DPI).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _lazy_imports():
    """Import OCR deps only when called — keeps the project usable without
    Tesseract/poppler installed."""
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_path  # noqa: F401
        return pytesseract, convert_from_path
    except ImportError as exc:
        sys.stderr.write(
            "OCR deps missing: install pytesseract + pdf2image, then ensure the "
            "`tesseract` and `pdftoppm` binaries are on PATH.\n"
            f"  ImportError: {exc}\n"
        )
        sys.exit(2)


def ocr_one_pdf(input_pdf: Path, output_pdf: Path, *, dpi: int = 300, lang: str = "eng") -> dict:
    """Convert a scanned PDF to a searchable PDF.

    Returns a summary dict: ``{"input", "output", "n_pages", "elapsed_sec"}``.
    """
    pytesseract, convert_from_path = _lazy_imports()

    t0 = time.perf_counter()
    images = convert_from_path(str(input_pdf), dpi=dpi)

    # Tesseract's `pdf` output mode emits a per-page searchable PDF (image
    # + invisible text layer). Stitch the per-page bytes together — each is
    # already a valid single-page PDF; pypdfium2 can join them.
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # The simplest reliable concatenation is via pypdf2 / pypdf. We use
    # pypdfium2 (already in requirements) to merge.
    import pypdfium2 as pdfium

    out_pdf = pdfium.PdfDocument.new()
    for image in images:
        page_bytes = pytesseract.image_to_pdf_or_hocr(image, lang=lang, extension="pdf")
        page_doc = pdfium.PdfDocument(page_bytes)
        # Append every page from this single-page pdf
        out_pdf.import_pages(page_doc, pages=list(range(len(page_doc))))

    out_pdf.save(str(output_pdf))

    return {
        "input": str(input_pdf),
        "output": str(output_pdf),
        "n_pages": len(images),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }


def ocr_folder(input_dir: Path, output_dir: Path, *, dpi: int = 300, lang: str = "eng") -> list[dict]:
    """OCR every .pdf under ``input_dir`` into ``output_dir`` with the same
    basename. Skip files that already exist in the output dir (idempotent)."""
    from src.pipeline.router import is_scanned

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for pdf in sorted(input_dir.glob("*.pdf")):
        target = output_dir / pdf.name
        if target.exists():
            summaries.append({"input": str(pdf), "output": str(target), "skipped": "already-exists"})
            continue
        if not is_scanned(pdf):
            # Just copy non-scanned PDFs through unchanged so the output dir
            # is a drop-in replacement for the input dir.
            target.write_bytes(pdf.read_bytes())
            summaries.append({"input": str(pdf), "output": str(target), "skipped": "native-text"})
            continue
        print(f"[+] OCR {pdf.name} → {target.name}")
        summary = ocr_one_pdf(pdf, target, dpi=dpi, lang=lang)
        summary["scanned"] = True
        summaries.append(summary)
        print(f"    {summary['n_pages']} pages in {summary['elapsed_sec']}s")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR scanned PDFs in place")
    parser.add_argument("input", type=Path, help="Input PDF file or directory")
    parser.add_argument("--out", type=Path, required=True, help="Output PDF or directory")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--lang", default="eng", help="Tesseract language code (eng, fra+eng, ...)")
    args = parser.parse_args()

    if args.input.is_dir():
        summaries = ocr_folder(args.input, args.out, dpi=args.dpi, lang=args.lang)
        n_ocr = sum(1 for s in summaries if s.get("scanned"))
        n_copy = sum(1 for s in summaries if s.get("skipped") == "native-text")
        n_skip = sum(1 for s in summaries if s.get("skipped") == "already-exists")
        print(f"\n[OK] OCR'd {n_ocr}, passthrough {n_copy}, already-existed {n_skip}")
    else:
        if not args.input.exists():
            sys.exit(f"input not found: {args.input}")
        summary = ocr_one_pdf(args.input, args.out, dpi=args.dpi, lang=args.lang)
        print(f"[OK] {summary['n_pages']} pages in {summary['elapsed_sec']}s → {summary['output']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the multi-page table rejoin heuristic against generated PDFs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend  # noqa: E402
from docling.datamodel.base_models import InputFormat  # noqa: E402
from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: E402

from utils.table_rejoin import rejoin_tables  # noqa: E402


def extract_and_rejoin(pdf_path: Path):
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend)
        }
    )
    result = converter.convert(str(pdf_path))
    raw_tables = result.document.tables or []
    logical = rejoin_tables(raw_tables)
    return raw_tables, logical


def main():
    print("=" * 60)
    print("Benchmark 3b: Multi-Page Table Rejoin Heuristic")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "sample_pdfs"
    cases = [
        data_dir / "single_page_table.pdf",
        data_dir / "multi_page_table.pdf",
    ]

    out_records = []
    for pdf in cases:
        if not pdf.exists():
            print(f"  missing {pdf.name}")
            continue
        print(f"\n[{pdf.name}]")
        raw, logical = extract_and_rejoin(pdf)
        print(f"  raw TableItems: {len(raw)}")
        for j, t in enumerate(logical):
            print(f"  logical table {j}: {t['num_rows']} rows x {t['num_cols']} cols, pages={t['pages']}, merged_from={t['source_table_indices']}")
        out_records.append({
            "file": pdf.name,
            "raw_count": len(raw),
            "logical_count": len(logical),
            "logical": [
                {
                    "num_rows": t["num_rows"],
                    "num_cols": t["num_cols"],
                    "pages": t["pages"],
                    "source_table_indices": t["source_table_indices"],
                }
                for t in logical
            ],
        })

    out = Path(__file__).parent.parent / "results" / "benchmark_table_rejoin.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(out_records, indent=2))
    print(f"\n✅ Saved {out}")


if __name__ == "__main__":
    main()

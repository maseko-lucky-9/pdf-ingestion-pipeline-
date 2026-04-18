#!/usr/bin/env python3
"""Benchmark table extraction quality."""

import time
import json
import tracemalloc
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


def assess_table_quality(markdown: str) -> dict:
    """Score table extraction quality."""
    lines = markdown.strip().split('\n')
    return {
        "has_header_row": len(lines) > 0 and '|' in lines[0],
        "has_separator_row": any('---' in line for line in lines[1:3]) if len(lines) > 1 else False,
        "row_count": len([l for l in lines if l.strip() and '|' in l]),
        "is_complete": markdown.strip().endswith('|'),
        "preview": markdown[:200] + "..." if len(markdown) > 200 else markdown
    }


def extract_tables(pdf_path: str) -> dict:
    """Extract and analyze tables from PDF."""

    tracemalloc.start()
    start = time.perf_counter()

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    result = converter.convert(pdf_path)
    elapsed = time.perf_counter() - start

    doc = result.document
    tables = doc.tables if hasattr(doc, 'tables') else []

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    table_details = []
    for i, table in enumerate(tables):
        try:
            markdown = table.export_to_markdown()
            prov = table.prov if hasattr(table, 'prov') else []
            pages_spanned = set(p.page_no for p in prov) if prov else set()

            table_details.append({
                "index": i,
                "num_rows": table.data.num_rows if hasattr(table, 'data') and table.data else 0,
                "num_cols": table.data.num_cols if hasattr(table, 'data') and table.data else 0,
                "pages": list(pages_spanned),
                "is_multi_page": len(pages_spanned) > 1,
                "quality": assess_table_quality(markdown)
            })
        except Exception as e:
            table_details.append({"index": i, "error": str(e)})

    return {
        "file": str(pdf_path),
        "extraction_time_s": elapsed,
        "peak_memory_mb": peak / 1024**2,
        "tables_found": len(tables),
        "tables": table_details
    }


def main():
    """Run table extraction benchmarks."""
    print("=" * 60)
    print("Benchmark 3: Multi-Page Table Handling")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "sample_pdfs"

    test_files = {
        "single_page_table": data_dir / "single_page_table.pdf",
        "multi_page_table": data_dir / "multi_page_table.pdf",
    }

    results = {}

    for name, pdf_path in test_files.items():
        if not pdf_path.exists():
            print(f"\n[{name}] ⚠️  File not found: {pdf_path}")
            continue

        print(f"\n[{name}] {pdf_path.name}...")
        try:
            result = extract_tables(str(pdf_path))
            results[name] = result

            print(f"  Found {result['tables_found']} table(s)")
            for t in result['tables']:
                if 'error' in t:
                    print(f"    Table {t['index']}: ERROR - {t['error']}")
                else:
                    multi = "MULTI-PAGE" if t['is_multi_page'] else "single-page"
                    print(f"    Table {t['index']}: {t['num_rows']} rows, {t['num_cols']} cols [{multi}]")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for name, result in results.items():
        if 'error' in result:
            print(f"  {name}: ERROR")
        else:
            multi_page_tables = sum(1 for t in result.get('tables', []) if t.get('is_multi_page', False))
            print(f"  {name}: {result['tables_found']} tables, {multi_page_tables} multi-page")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_tables.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to results/benchmark_tables.json")


if __name__ == "__main__":
    main()
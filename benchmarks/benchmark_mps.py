#!/usr/bin/env python3
"""Benchmark Docling with MPS (Metal) vs CPU accelerator."""

import json
import time
import tracemalloc
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def run(pdf_path: str, device: AcceleratorDevice) -> dict:
    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(device=device, num_threads=4)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    tracemalloc.start()
    start = time.perf_counter()
    result = converter.convert(pdf_path)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    doc = result.document
    pages = len(doc.pages) if hasattr(doc, "pages") else 0
    return {
        "device": device.value,
        "file": str(Path(pdf_path).name),
        "time_s": elapsed,
        "pages": pages,
        "pages_per_second": (pages / elapsed) if elapsed > 0 and pages > 0 else 0,
        "peak_python_mb": peak / 1024 ** 2,
    }


def main():
    print("=" * 60)
    print("Benchmark 2b: Docling MPS (Metal) vs CPU")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "sample_pdfs"
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No test PDFs")
        return

    results = []
    for device in (AcceleratorDevice.CPU, AcceleratorDevice.MPS):
        print(f"\n[{device.value.upper()}]")
        for pdf in pdfs:
            try:
                r = run(str(pdf), device)
                results.append(r)
                print(f"  {r['file']}: {r['pages']} pages in {r['time_s']:.2f}s ({r['pages_per_second']:.2f} pps)")
            except Exception as e:
                results.append({"device": device.value, "file": str(pdf.name), "error": str(e)})
                print(f"  {pdf.name}: ERROR {e}")

    # Speedup
    cpu = {r["file"]: r for r in results if r.get("device") == "cpu" and "time_s" in r}
    mps = {r["file"]: r for r in results if r.get("device") == "mps" and "time_s" in r}
    speedups = {}
    for f in cpu:
        if f in mps and mps[f]["time_s"] > 0:
            speedups[f] = cpu[f]["time_s"] / mps[f]["time_s"]
    print("\nSpeedup (CPU/MPS):")
    for f, s in speedups.items():
        print(f"  {f}: {s:.2f}x")

    out = Path(__file__).parent.parent / "results" / "benchmark_mps.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"results": results, "speedups": speedups}, indent=2))
    print(f"\n✅ Saved {out}")


if __name__ == "__main__":
    main()

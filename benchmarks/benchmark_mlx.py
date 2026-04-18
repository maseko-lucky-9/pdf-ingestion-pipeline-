#!/usr/bin/env python3
"""Benchmark Docling MLX vs CPU performance."""

import time
import json
import tracemalloc
from pathlib import Path
from docling.document_converter import DocumentConverter


def benchmark_docling_cpu(pdf_path: str) -> dict:
    """Benchmark Docling with default CPU backend."""

    tracemalloc.start()
    start = time.perf_counter()

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pages = len(doc.pages) if hasattr(doc, 'pages') else 0

    return {
        "backend": "cpu",
        "file": pdf_path,
        "time_s": elapsed,
        "pages": pages,
        "peak_memory_mb": peak / 1024**2,
        "pages_per_second": pages / elapsed if elapsed > 0 and pages > 0 else 0
    }


def main():
    """Run MLX comparison benchmark."""
    print("=" * 60)
    print("Benchmark 2: Docling MLX Speedup")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "sample_pdfs"
    results = {"cpu": [], "mlx": None}

    test_pdfs = list(data_dir.glob("*.pdf"))

    if not test_pdfs:
        print("❌ No test PDFs found. Run pdf_generator.py first")
        return

    print(f"\nFound {len(test_pdfs)} test PDFs\n")

    # CPU benchmarks
    for pdf in test_pdfs:
        print(f"[CPU] {pdf.name}...")
        try:
            result = benchmark_docling_cpu(str(pdf))
            results["cpu"].append(result)
            print(f"  {result['pages']} pages in {result['time_s']:.2f}s "
                  f"({result['pages_per_second']:.2f} pages/sec)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["cpu"].append({"file": str(pdf), "error": str(e)})

    # MLX check
    print("\n[MLX] Checking availability...")
    try:
        import mlx
        print("  MLX package available")

        # Note: Docling MLX integration may require additional setup
        # Currently Docling uses MLX for VLM acceleration automatically
        results["mlx"] = "mlx_available_but_requires_configuration"

    except ImportError:
        print("  MLX not available or not configured for Docling")
        results["mlx"] = "not_available"

    # Calculate speedup (if MLX was benchmarked)
    if isinstance(results["mlx"], list) and results["mlx"]:
        cpu_avg = sum(r["time_s"] for r in results["cpu"] if "time_s" in r) / len(results["cpu"])
        mlx_avg = sum(r["time_s"] for r in results["mlx"] if "time_s" in r) / len(results["mlx"])
        speedup = cpu_avg / mlx_avg if mlx_avg > 0 else 0
        results["speedup"] = f"{speedup:.1f}x"
        print(f"\n⚡ MLX Speedup: {speedup:.1f}x")
    else:
        results["speedup"] = "not_measured"
        print(f"\n⚠️  MLX not benchmarked: {results['mlx']}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_mlx.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to results/benchmark_mlx.json")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Compile all benchmark results into a single report."""

import json
from pathlib import Path
from datetime import datetime


def load_json_file(path: Path) -> dict:
    """Load JSON file safely."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def compile_results(results_dir: Path) -> dict:
    """Compile all benchmark results."""

    compiled = {
        "compiled_at": datetime.now().isoformat(),
        "hardware": {},
        "benchmarks": {}
    }

    # Load hardware specs
    hardware_file = results_dir / "hardware_spec.txt"
    if hardware_file.exists():
        compiled["hardware"]["spec"] = hardware_file.read_text()[:500]

    git_file = results_dir / "git_commit.txt"
    if git_file.exists():
        compiled["hardware"]["git_commit"] = git_file.read_text().strip()

    # Load benchmark results
    benchmark_files = {
        "embeddings": "benchmark_embeddings.json",
        "mlx": "benchmark_mlx.json",
        "tables": "benchmark_tables.json",
        "quantization": "benchmark_quantization.json",
        "chunking": "benchmark_chunking.json",
        "pipeline": "benchmark_pipeline.json"
    }

    for name, filename in benchmark_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            compiled["benchmarks"][name] = load_json_file(filepath)
            compiled["benchmarks"][name]["status"] = "complete"
        else:
            compiled["benchmarks"][name] = {"status": "not_run"}

    return compiled


def generate_summary(compiled: dict) -> str:
    """Generate human-readable summary."""

    summary = ["# M5 Pro Benchmark Results Summary\n"]
    summary.append(f"**Compiled:** {compiled['compiled_at']}\n")

    if "git_commit" in compiled.get("hardware", {}):
        summary.append(f"**Git Commit:** `{compiled['hardware']['git_commit']}`\n")

    summary.append("\n## Benchmark Status\n")
    summary.append("| Benchmark | Status | Key Result |\n")
    summary.append("|-----------|--------|------------|\n")

    for name, data in compiled.get("benchmarks", {}).items():
        status = data.get("status", "unknown")
        key_result = ""

        if status == "complete":
            if name == "embeddings" and "results" in data:
                results = data["results"]
                if results:
                    best = max(results, key=lambda x: x.get("throughput_emb_per_sec", 0))
                    key_result = f"{best.get('throughput_emb_per_sec', 0):.1f} emb/sec"
            elif name == "tables":
                key_result = f"{data.get('tables_found', 0)} tables analyzed"
            elif name == "mlx":
                key_result = str(data.get("speedup", "N/A"))
            elif name == "pipeline":
                key_result = f"{data.get('total_time_s', 0):.1f}s total"

        summary.append(f"| {name} | {status} | {key_result} |\n")

    return "".join(summary)


def main():
    """Compile and save results."""
    results_dir = Path(__file__).parent.parent / "results"

    compiled = compile_results(results_dir)

    # Save compiled JSON
    output_file = results_dir / f"{datetime.now().strftime('%Y-%m-%d')}_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(compiled, f, indent=2)
    print(f"✅ Compiled results saved to: {output_file}")

    # Generate summary
    summary = generate_summary(compiled)
    summary_file = results_dir / "SUMMARY.md"
    summary_file.write_text(summary)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + summary)


if __name__ == "__main__":
    main()
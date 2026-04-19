"""Query a collection.

Usage:
    python -m src.query "What is RSI?" --collection seed-test [--domain quant]
"""
from __future__ import annotations
import argparse
import sys

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.pipeline.retriever import retrieve

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a collection")
    parser.add_argument("question")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--domain", default="")
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg.collection_db_path(args.collection)

    if not db_path.exists():
        console.print(f"[red]Collection not found: {args.collection}[/red]")
        console.print(f"  Expected: {db_path}")
        sys.exit(1)

    results = retrieve(args.question, db_path, cfg, domain=args.domain)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f'Results for: "{args.question}"', show_lines=True)
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Score", width=8)
    table.add_column("Type", width=8)
    table.add_column("Source", width=30)
    table.add_column("Content", no_wrap=False)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.chunk_type,
            r.source_pdf.split("/")[-1],
            r.content[:300] + ("…" if len(r.content) > 300 else ""),
        )

    console.print(table)


if __name__ == "__main__":
    main()

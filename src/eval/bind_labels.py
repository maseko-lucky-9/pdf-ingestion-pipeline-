"""Bind eval query draft to actual docids after first ingest (R10).

Pass 1 (before ingest): author writes queries_draft.json with human-readable
expected_source strings.

Pass 2 (after ingest): this script reads the draft, looks up docids in meta
via lexical match on source_pdf + book, and writes queries_bound.json.
Manual review of the bound file is required before using in run_eval.py.

Usage:
    python -m src.eval.bind_labels \
        --collection seed-test \
        --draft src/eval/queries_draft.json \
        --output src/eval/queries_bound.json
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import sqlite_vec
from rich.console import Console
from rich.table import Table

from src.config import load_config

console = Console()


def _lexical_lookup(cur: sqlite3.Cursor, hint: str) -> list[str]:
    """Return docids whose source_pdf or book contains any word from hint."""
    words = [w.lower() for w in hint.replace(",", " ").replace("—", " ").split() if len(w) > 3]
    if not words:
        return []

    found: set[str] = set()
    for word in words:
        rows = cur.execute(
            "SELECT docid FROM meta WHERE LOWER(source_pdf) LIKE ? OR LOWER(book) LIKE ?",
            (f"%{word}%", f"%{word}%"),
        ).fetchall()
        for row in rows:
            found.add(row[0])
    return sorted(found)


def bind_labels(draft_path: Path, db_path: Path, output_path: Path) -> None:
    with open(draft_path) as f:
        queries = json.load(f)

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()

    table = Table(title="Label binding", show_lines=True)
    table.add_column("ID")
    table.add_column("Query (truncated)")
    table.add_column("Matched docids")

    bound = []
    for q in queries:
        hint = q.get("expected_source", "")
        docids = _lexical_lookup(cur, hint)
        q["relevant_docids"] = docids
        bound.append(q)
        table.add_row(q["id"], q["query"][:60], str(len(docids)))

    conn.close()
    console.print(table)

    with open(output_path, "w") as f:
        json.dump(bound, f, indent=2)

    console.print(f"\n[green]Bound labels written to {output_path}[/green]")
    console.print("[yellow]Review relevant_docids manually before running eval.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind query labels to docids")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--draft", type=Path, default=Path("src/eval/queries_draft.json"))
    parser.add_argument("--output", type=Path, default=Path("src/eval/queries_bound.json"))
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg.collection_db_path(args.collection)

    if not db_path.exists():
        console.print(f"[red]Collection not found: {args.collection} ({db_path})[/red]")
        sys.exit(1)

    bind_labels(args.draft, db_path, args.output)


if __name__ == "__main__":
    main()

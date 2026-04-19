"""Evaluate retrieval quality: MRR@10, NDCG@10.

Usage:
    python -m src.eval.run_eval \
        --collection seed-test \
        --labels src/eval/queries_bound.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.pipeline.retriever import retrieve

console = Console()


def _mrr_at_k(relevant: set[str], ranked: list[str], k: int = 10) -> float:
    for rank, docid in enumerate(ranked[:k], 1):
        if docid in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(relevant: set[str], ranked: list[str], k: int = 10) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, docid in enumerate(ranked[:k], 1)
        if docid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def run_eval(labels_path: Path, db_path: Path, cfg) -> None:
    with open(labels_path) as f:
        queries = json.load(f)

    unlabeled = [q for q in queries if not q.get("relevant_docids")]
    if unlabeled:
        console.print(f"[yellow]Warning: {len(unlabeled)} queries have no relevant_docids — skipping.[/yellow]")
        queries = [q for q in queries if q.get("relevant_docids")]

    if not queries:
        console.print("[red]No labeled queries to evaluate.[/red]")
        sys.exit(1)

    table = Table(title="Retrieval Eval", show_lines=True)
    table.add_column("ID")
    table.add_column("Query")
    table.add_column("MRR@10", width=8)
    table.add_column("NDCG@10", width=9)

    mrrs, ndcgs = [], []

    for q in queries:
        relevant = set(q["relevant_docids"])
        results = retrieve(q["query"], db_path, cfg)
        ranked = [r.docid for r in results]

        mrr = _mrr_at_k(relevant, ranked)
        ndcg = _ndcg_at_k(relevant, ranked)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        table.add_row(q["id"], q["query"][:55], f"{mrr:.3f}", f"{ndcg:.3f}")

    table.add_row(
        "[bold]AVG[/bold]", "",
        f"[bold]{sum(mrrs)/len(mrrs):.3f}[/bold]",
        f"[bold]{sum(ndcgs)/len(ndcgs):.3f}[/bold]",
    )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--labels", type=Path, default=Path("src/eval/queries_bound.json"))
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg.collection_db_path(args.collection)

    if not db_path.exists():
        console.print(f"[red]Collection not found: {args.collection}[/red]")
        sys.exit(1)

    run_eval(args.labels, db_path, cfg)


if __name__ == "__main__":
    main()

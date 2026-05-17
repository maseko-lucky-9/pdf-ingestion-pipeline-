"""Evaluate retrieval quality: Recall@5, MRR@10, NDCG@10.

Usage:
    python -m src.eval.run_eval \
        --collection seed-test \
        --labels src/eval/queries_bound.json \
        [--output results/<timestamp>.json]
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.pipeline.retriever import retrieve

console = Console()


def _recall_at_k(relevant: set[str], ranked: list[str], k: int = 5) -> float:
    """Fraction of the relevant docids that appear in the top-k ranked list.

    Parallel to ``utils/retrieval_metrics.recall_at_k`` but on string docids
    (the harness operates on chunk uuids, the utils version on int row ids).
    """
    if not relevant:
        return 0.0
    hits = sum(1 for docid in ranked[:k] if docid in relevant)
    return hits / len(relevant)


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


def run_eval(
    labels_path: Path,
    db_path: Path,
    cfg,
    *,
    output_path: Path | None = None,
    collection_name: str = "",
) -> dict:
    """Run the harness and return a summary dict. Optionally persist to disk."""
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
    table.add_column("Recall@5", width=9)
    table.add_column("MRR@10", width=8)
    table.add_column("NDCG@10", width=9)

    per_query: list[dict] = []
    recalls, mrrs, ndcgs = [], [], []

    for q in queries:
        relevant = set(q["relevant_docids"])
        results = retrieve(q["query"], db_path, cfg)
        ranked = [r.docid for r in results]

        recall = _recall_at_k(relevant, ranked, k=5)
        mrr = _mrr_at_k(relevant, ranked)
        ndcg = _ndcg_at_k(relevant, ranked)
        recalls.append(recall)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "relevant_docids": list(relevant),
            "ranked_docids": ranked[:10],
            "recall@5": recall,
            "mrr@10": mrr,
            "ndcg@10": ndcg,
        })

        table.add_row(q["id"], q["query"][:55], f"{recall:.3f}", f"{mrr:.3f}", f"{ndcg:.3f}")

    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(mrrs) / len(mrrs)
    avg_ndcg = sum(ndcgs) / len(ndcgs)

    table.add_row(
        "[bold]AVG[/bold]", "",
        f"[bold]{avg_recall:.3f}[/bold]",
        f"[bold]{avg_mrr:.3f}[/bold]",
        f"[bold]{avg_ndcg:.3f}[/bold]",
    )
    console.print(table)

    target_recall = 0.85
    if avg_recall >= target_recall:
        console.print(f"[green]Recall@5 {avg_recall:.3f} >= target {target_recall}[/green]")
    else:
        console.print(f"[yellow]Recall@5 {avg_recall:.3f} < target {target_recall}[/yellow]")

    summary = {
        "collection": collection_name,
        "labels_path": str(labels_path),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_queries": len(queries),
        "avg_recall@5": avg_recall,
        "avg_mrr@10": avg_mrr,
        "avg_ndcg@10": avg_ndcg,
        "per_query": per_query,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"[dim]Summary written to {output_path}[/dim]")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--labels", type=Path, default=Path("src/eval/queries_bound.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON summary path (e.g. results/slice2-baseline.json).",
    )
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg.collection_db_path(args.collection)

    if not db_path.exists():
        console.print(f"[red]Collection not found: {args.collection}[/red]")
        sys.exit(1)

    run_eval(
        args.labels,
        db_path,
        cfg,
        output_path=args.output,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()

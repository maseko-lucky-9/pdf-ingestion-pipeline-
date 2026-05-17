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
import sqlite3
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.pipeline.retriever import retrieve

console = Console()


# A label is identified by (source_pdf_basename, page_start, page_end). Two
# chunks count as "the same labelled source" iff they share a label key —
# concretely, iff their page ranges overlap one of the label's page ranges.
LabelKey = tuple[str, int, int]


def _chunk_matches_label(chunk_source: str, chunk_pages: tuple[int, int], label: dict) -> bool:
    """True iff the chunk's page range overlaps a label's page range and the
    chunk's source PDF basename matches the label's."""
    basename = label["source_pdf"]
    if not chunk_source.endswith(basename):
        return False
    label_start, label_end = label["pages"][0], label["pages"][1]
    chunk_start, chunk_end = chunk_pages
    return chunk_start <= label_end and chunk_end >= label_start


def _hits_per_label(
    relevant_pages: list[dict],
    ranked_chunks: list[tuple[str, tuple[int, int]]],
    k: int,
) -> tuple[int, int]:
    """Count distinct labels that any top-k retrieved chunk overlaps.

    Returns ``(n_labels_hit, n_labels_total)``. Recall@k = first / second.
    A single retrieved chunk that overlaps multiple labels counts once per
    label; a single label that multiple retrieved chunks overlap also counts
    once. The metric measures coverage of distinct labelled sources, NOT
    coverage of chunks.
    """
    n_total = len(relevant_pages)
    hit: set[int] = set()
    for chunk_source, chunk_pages in ranked_chunks[:k]:
        for idx, label in enumerate(relevant_pages):
            if idx in hit:
                continue
            if _chunk_matches_label(chunk_source, chunk_pages, label):
                hit.add(idx)
    return len(hit), n_total


def _resolve_stable_labels(
    relevant_pages: list[dict],
    db_path: Path,
) -> set[str]:
    """Resolve (source_pdf, pages) stable labels to current chunk docids.

    Retained for the legacy fall-through path and for any downstream consumer
    that wants a flat docid set. The primary metric path now uses
    :func:`_hits_per_label` directly on the retriever's (source, pages)
    output so chunker boundary changes between ingests do not change recall.
    """
    if not relevant_pages:
        return set()
    con = sqlite3.connect(str(db_path))
    docids: set[str] = set()
    try:
        for ref in relevant_pages:
            basename = ref["source_pdf"]
            start, end = ref["pages"][0], ref["pages"][1]
            rows = con.execute(
                """
                SELECT docid FROM meta
                WHERE source_pdf LIKE ?
                  AND page_start <= ?
                  AND page_end   >= ?
                """,
                (f"%{basename}", end, start),
            ).fetchall()
            for (docid,) in rows:
                docids.add(docid)
    finally:
        con.close()
    return docids


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

    # Queries without relevant_pages (refusal-expected) are skipped from the
    # averaging — they belong to a separate eval axis.
    eligible: list[dict] = []
    skipped: list[str] = []
    for q in queries:
        if not q.get("relevant_pages"):
            skipped.append(q["id"])
            continue
        eligible.append(q)

    if skipped:
        console.print(f"[yellow]Skipping {len(skipped)} queries with no relevant_pages (e.g. refusal-expected): {skipped}[/yellow]")

    if not eligible:
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

    for q in eligible:
        relevant_pages = q["relevant_pages"]
        # Per-chunk MRR/NDCG still operate on docid identity (chunk granularity
        # is the right level for "which retrieved item ranked highest"). Recall
        # is over the LABEL set — i.e. distinct source ranges — because two
        # chunks at the same range are not two different sources.
        relevant_docids = _resolve_stable_labels(relevant_pages, db_path)
        results = retrieve(q["query"], db_path, cfg)
        ranked = [r.docid for r in results]
        ranked_chunks = [(r.source_pdf, r.page_range) for r in results]
        n_hit, n_total = _hits_per_label(relevant_pages, ranked_chunks, k=5)
        recall = n_hit / n_total if n_total else 0.0
        # MRR and NDCG still use the per-chunk relevant set so they reward
        # finding any chunk overlapping a labelled source. This is the same
        # semantics the metrics had previously, just sourced from stable
        # labels instead of uuid pins.
        mrr = _mrr_at_k(relevant_docids, ranked)
        ndcg = _ndcg_at_k(relevant_docids, ranked)
        recalls.append(recall)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "relevant_pages": relevant_pages,
            "n_labels": n_total,
            "n_labels_hit_at_5": n_hit,
            "resolved_docids_count": len(relevant_docids),
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

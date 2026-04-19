"""Ingest PDFs into a named collection.

Usage:
    python -m src.ingest <pdf_dir> --collection <name> [--domain <domain>]
"""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import load_config

console = Console()


def preflight_check(cfg) -> None:
    host = cfg.ollama.host
    model = cfg.ollama.embed_model
    try:
        resp = ollama.list()
        installed = [m.model for m in resp.models]
    except Exception:
        console.print(f"[bold red]Ollama daemon unreachable at {host}[/bold red]")
        console.print("Start it with: ollama serve")
        sys.exit(1)

    # normalize: strip tag suffix for comparison
    installed_names = [m.split(":")[0] for m in installed]
    if model not in installed_names and model not in installed:
        console.print(f"[bold red]Model {model!r} not pulled.[/bold red]")
        console.print(f"Run: ollama pull {model}")
        sys.exit(1)


def ingest_collection(pdf_dir: Path, collection: str, domain: str) -> None:
    cfg = load_config()
    preflight_check(cfg)

    from src.pipeline.router import is_scanned
    from src.pipeline.extractor import extract_items
    from src.pipeline.normalizer import normalize_items
    from src.pipeline.chunker import chunk_items
    from src.pipeline.embedder import embed_chunks
    from src.pipeline.indexer import IndexWriter

    db_path = cfg.collection_db_path(collection)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    error_log = cfg.collection_error_log(collection)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {pdf_dir}[/yellow]")
        return

    ok, fail = 0, 0

    with IndexWriter(db_path, cfg) as writer, Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Ingesting…", total=len(pdfs))

        for pdf in pdfs:
            progress.update(task, description=f"[cyan]{pdf.name}[/cyan]")
            try:
                if is_scanned(pdf):
                    console.print(f"  [yellow]⚠ {pdf.name}: scanned PDF — OCR not yet supported, skipping[/yellow]")
                    fail += 1
                    continue

                items = extract_items(pdf)
                items = normalize_items(items)
                chunks = chunk_items(
                    items,
                    source_pdf=str(pdf),
                    collection=collection,
                    domain=domain,
                    book=pdf.stem,
                    cfg=cfg,
                )
                chunks = embed_chunks(chunks, cfg)
                writer.write(chunks)
                ok += 1
            except Exception as exc:
                fail += 1
                with error_log.open("a") as f:
                    f.write(f"[{pdf}]\n{exc}\n{traceback.format_exc()}\n---\n")
                console.print(f"  [red]✗ {pdf.name}: {exc}[/red]")

            progress.advance(task)

    console.print(f"\n[bold green]Done.[/bold green] ok={ok} fail={fail}")
    if fail:
        console.print(f"  Errors logged to {error_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a collection")
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--domain", default="")
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        console.print(f"[red]Not a directory: {args.pdf_dir}[/red]")
        sys.exit(1)

    ingest_collection(args.pdf_dir, args.collection, args.domain)


if __name__ == "__main__":
    main()

"""Ingest PDFs into a named collection.

Usage:
    python -m src.ingest <pdf_dir> --collection <name> [--domain <domain>] [--parallel N]

With --parallel > 1, the extract+normalize+chunk phases run in a process pool
while embed (Ollama) + write (SQLite) stay serial in the main process. On an
M2 MacBook this brings 50 PDFs from ~67 min down to roughly ~20-25 min with
--parallel 4.
"""
from __future__ import annotations
import argparse
import multiprocessing
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _extract_chunks_worker(args: tuple[str, str, str]) -> dict:
    """Run extract → normalize → chunk for ONE PDF inside a worker process.

    Returns a serialisable dict:
        {"pdf": <str path>, "chunks": <list[Chunk]> | None,
         "error": <str> | None, "scanned": <bool>}
    Workers each load docling weights once (singleton inside extractor.py),
    so amortised cost across many PDFs is low.
    """
    pdf_path_str, collection, domain = args
    pdf = Path(pdf_path_str)
    try:
        # Local imports so the workers do not duplicate the parent's import
        # graph at fork time; helps on macOS where docling is heavy.
        from src.config import load_config as _load_config
        from src.pipeline.chunker import chunk_items
        from src.pipeline.extractor import extract_items
        from src.pipeline.normalizer import normalize_items
        from src.pipeline.router import is_scanned

        cfg = _load_config()
        if is_scanned(pdf):
            return {"pdf": pdf_path_str, "chunks": None, "error": None, "scanned": True}

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
        return {"pdf": pdf_path_str, "chunks": chunks, "error": None, "scanned": False}
    except Exception as exc:
        return {
            "pdf": pdf_path_str,
            "chunks": None,
            "error": f"{exc}\n{traceback.format_exc()}",
            "scanned": False,
        }


def ingest_collection(
    pdf_dir: Path,
    collection: str,
    domain: str,
    *,
    parallel: int = 1,
) -> None:
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

    parallel = max(1, parallel)
    ok, fail = 0, 0

    if parallel == 1:
        # Sequential path — unchanged from slice 1.
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
    else:
        # Parallel path — extract+normalize+chunk in workers; embed+write
        # serially in the main process to avoid Ollama contention + SQLite
        # writer conflicts.
        console.print(f"[cyan]Parallel ingest with {parallel} extractor workers[/cyan]")
        worker_args = [(str(p), collection, domain) for p in pdfs]

        # `spawn` start method avoids fork-related issues with docling/torch
        # on macOS; matches what PyTorch recommends.
        ctx = multiprocessing.get_context("spawn")
        with (
            IndexWriter(db_path, cfg) as writer,
            ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as pool,
            Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress,
        ):
            task = progress.add_task("Ingesting…", total=len(pdfs))
            futures = {pool.submit(_extract_chunks_worker, args): args[0] for args in worker_args}
            for fut in as_completed(futures):
                pdf_path_str = futures[fut]
                pdf_name = Path(pdf_path_str).name
                progress.update(task, description=f"[cyan]{pdf_name}[/cyan]")
                try:
                    result = fut.result()
                except Exception as exc:
                    fail += 1
                    with error_log.open("a") as f:
                        f.write(f"[{pdf_path_str}] worker crashed: {exc}\n---\n")
                    console.print(f"  [red]✗ {pdf_name}: worker crashed: {exc}[/red]")
                    progress.advance(task)
                    continue

                if result["scanned"]:
                    console.print(f"  [yellow]⚠ {pdf_name}: scanned PDF — OCR not yet supported, skipping[/yellow]")
                    fail += 1
                elif result["error"]:
                    fail += 1
                    with error_log.open("a") as f:
                        f.write(f"[{pdf_path_str}]\n{result['error']}\n---\n")
                    console.print(f"  [red]✗ {pdf_name}: {result['error'].splitlines()[0]}[/red]")
                else:
                    try:
                        chunks = embed_chunks(result["chunks"], cfg)
                        writer.write(chunks)
                        ok += 1
                    except Exception as exc:
                        fail += 1
                        with error_log.open("a") as f:
                            f.write(f"[{pdf_path_str}] embed/write: {exc}\n{traceback.format_exc()}\n---\n")
                        console.print(f"  [red]✗ {pdf_name}: embed/write {exc}[/red]")
                progress.advance(task)

    console.print(f"\n[bold green]Done.[/bold green] ok={ok} fail={fail}")
    if fail:
        console.print(f"  Errors logged to {error_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a collection")
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of worker processes for extract+normalize+chunk phases. "
             "Embed+write stay serial. Default 1 (sequential).",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        console.print(f"[red]Not a directory: {args.pdf_dir}[/red]")
        sys.exit(1)

    ingest_collection(
        args.pdf_dir,
        args.collection,
        args.domain,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()

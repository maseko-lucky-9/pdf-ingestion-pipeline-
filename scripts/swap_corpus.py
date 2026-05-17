"""One-command prospect-corpus swap-in for sales-demo prep.

Wraps the existing ``python -m src.ingest`` pipeline with the bookkeeping a
pre-demo workflow needs: nuke (or keep) the target collection first, then
ingest, then a smoke-query to confirm the new index actually answers.

Usage:
    python scripts/swap_corpus.py \
        --src ~/Desktop/prospect-acme-pdfs \
        --collection-name prospect-acme

Scanned PDFs are auto-detected and SKIPPED by the existing ingest pipeline
(``src.pipeline.router.is_scanned``). This script surfaces the skip count in
the summary so the operator sees how many files dropped out.

Realistic timing (measured 2026-05-17 on M2 MacBook):
- ~80 sec per mid-size financial-domain PDF (3-15 MB native text)
- 50-PDF / 200 MB folder: ~60-70 min end-to-end
- Pre-demo window should be 60 minutes, not 30, for full prospect-corpus swaps
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap in a prospect corpus")
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Folder containing the prospect's PDF files.",
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Target collection name (will be created / replaced).",
    )
    parser.add_argument(
        "--domain",
        default="prospect",
        help="Domain tag stored on each chunk (default: prospect).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="If set, add to the existing collection rather than nuking it.",
    )
    parser.add_argument(
        "--collections-root",
        type=Path,
        default=Path("./collections"),
        help="Root directory containing collection subdirectories.",
    )
    parser.add_argument(
        "--smoke-query",
        default=None,
        help="Optional query string to run against the new index after ingest.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Worker count for the extract+normalize+chunk phases (slice 4 #14). "
             "Embed+write stay serial regardless. Default 1.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Pre-process scanned PDFs with Tesseract before ingest (slice 4 #15). "
             "Requires `tesseract` and `pdftoppm` on PATH. Outputs an OCR'd folder "
             "alongside the input and points ingest at it.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Tesseract language code passed through to OCR pre-processing.",
    )
    args = parser.parse_args()

    if not args.src.exists() or not args.src.is_dir():
        print(f"ERR: --src {args.src} does not exist or is not a directory", file=sys.stderr)
        sys.exit(2)

    pdfs = sorted(args.src.glob("*.pdf"))
    if not pdfs:
        print(f"ERR: no .pdf files found under {args.src}", file=sys.stderr)
        sys.exit(2)
    print(f"[+] Found {len(pdfs)} PDFs under {args.src}")

    ingest_src = args.src
    if args.ocr:
        ocr_dir = args.src.parent / f"{args.src.name}-ocr"
        print(f"[+] OCR pre-processing: {args.src} → {ocr_dir}")
        from scripts.ocr_preprocess import ocr_folder
        ocr_folder(args.src, ocr_dir, lang=args.ocr_lang)
        ingest_src = ocr_dir

    target_dir = args.collections_root / args.collection_name
    if target_dir.exists() and not args.keep_existing:
        print(f"[+] Nuking existing collection {target_dir}")
        shutil.rmtree(target_dir)
    elif target_dir.exists():
        print(f"[+] Keeping existing collection {target_dir} (--keep-existing)")

    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "src.ingest",
        str(ingest_src),
        "--collection",
        args.collection_name,
        "--domain",
        args.domain,
    ]
    if args.parallel > 1:
        cmd.extend(["--parallel", str(args.parallel)])
    proc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"ERR: ingest failed with exit code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    print(f"\n[+] Ingest complete in {elapsed:.0f}s ({elapsed/len(pdfs):.0f}s per PDF)")

    err_log = target_dir / "ingest_errors.log"
    if err_log.exists():
        print(f"[!] Skipped or failed files: see {err_log}")
        with open(err_log) as f:
            tail = f.read().splitlines()[-10:]
        for line in tail:
            print(f"    {line}")

    if args.smoke_query:
        print(f"\n[+] Smoke query: {args.smoke_query!r}")
        proc2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.query",
                args.smoke_query,
                "--collection",
                args.collection_name,
            ],
            check=False,
        )
        if proc2.returncode != 0:
            print("ERR: smoke query failed", file=sys.stderr)
            sys.exit(proc2.returncode)

    print(f"\n[OK] Collection {args.collection_name!r} ready.")


if __name__ == "__main__":
    main()

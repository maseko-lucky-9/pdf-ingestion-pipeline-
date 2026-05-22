"""Delete named collections from disk and append audit entries.

Two modes:

  Manual scrub (slice 2):
      python scripts/scrub_corpus.py --collection prospect-acme --confirm

  Cron-friendly batch scrub (slice 6 #20):
      python scripts/scrub_corpus.py --age-threshold-days 7 --prefix prospect- --confirm

      Scrubs every collection under --collections-root whose directory mtime
      is older than --age-threshold-days. The optional --prefix filter
      restricts the sweep so the rehearsed `quant-finance` / `sa-legislation`
      collections never get touched even if they're old (they're refreshable
      from public sources, but you don't want a cron job nuking the demo
      corpus at 03:30 UTC).

Audit log entry (one line per scrub):
    <iso-timestamp>,<operator>,<collection>,<chunks_removed>,<bytes_removed>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import shutil
import sqlite3
import sys
from pathlib import Path


def _count_chunks(db_path: Path, collection: str) -> int:
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.execute(
            "SELECT COUNT(*) FROM meta WHERE collection = ?",
            (collection,),
        )
        return cur.fetchone()[0]
    except Exception:
        return -1
    finally:
        try:
            con.close()
        except Exception:
            pass


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _scrub_one(
    target: Path,
    collection: str,
    audit_log: Path,
    operator: str,
) -> tuple[int, int]:
    """Delete one collection directory + append an audit line. Returns (chunks, bytes)."""
    db = target / "index.db"
    chunks = _count_chunks(db, collection) if db.exists() else -1
    size_bytes = _dir_bytes(target)
    shutil.rmtree(target)
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with open(audit_log, "a") as f:
        f.write(f"{ts},{operator},{collection},{chunks},{size_bytes}\n")
    return chunks, size_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub collection(s) from disk")
    parser.add_argument(
        "--collection",
        help="Single collection to scrub. Mutually exclusive with --age-threshold-days.",
    )
    parser.add_argument(
        "--age-threshold-days",
        type=float,
        help="Cron mode: scrub every collection under --collections-root whose "
             "directory mtime is older than this many days. Use with --prefix to "
             "restrict the sweep (e.g. --prefix prospect-).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="In cron mode, only consider collections whose name starts with this "
             "prefix. REQUIRED with --age-threshold-days so the cron never nukes "
             "a rehearsed demo corpus (e.g. --prefix prospect-).",
    )
    parser.add_argument(
        "--collections-root",
        type=Path,
        default=Path("./collections"),
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=Path("./logs/scrub-audit.log"),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required; this is destructive.",
    )
    parser.add_argument(
        "--operator",
        default=getpass.getuser(),
        help="Operator identifier for the audit log (default: $USER).",
    )
    args = parser.parse_args()

    # Cron-mode safety gate: --prefix is REQUIRED when --age-threshold-days
    # is set. Without it, a long-running deployment whose mtime drifts past
    # the threshold would lose rehearsed demo corpora (quant-finance,
    # sa-legislation) on the next cron tick. Slice 6 review P2 fix.
    if args.age_threshold_days is not None and not args.prefix:
        print(
            "ERR: --age-threshold-days requires --prefix to avoid sweeping "
            "rehearsed demo corpora. Use e.g. --prefix prospect-.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Mode-select: exactly one of --collection / --age-threshold-days.
    if bool(args.collection) == (args.age_threshold_days is not None):
        print(
            "ERR: pass exactly one of --collection <name> OR --age-threshold-days N",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.collection:
        # Single-target mode.
        target = args.collections_root / args.collection
        if not target.exists():
            print(f"ERR: collection {args.collection!r} not found at {target}", file=sys.stderr)
            sys.exit(2)
        if not args.confirm:
            print(
                f"DRY RUN: would delete {target} (re-run with --confirm to actually scrub).",
                file=sys.stderr,
            )
            sys.exit(1)
        chunks, size_bytes = _scrub_one(target, args.collection, args.audit_log, args.operator)
        print(f"[OK] Scrubbed {args.collection!r}: {chunks} chunks, {size_bytes:,} bytes")
        print(f"     Audit line written to {args.audit_log}")
        return

    # Cron mode: walk the collections root, scrub anything matching prefix + age.
    if not args.collections_root.exists():
        print(f"ERR: collections root not found: {args.collections_root}", file=sys.stderr)
        sys.exit(2)

    cutoff = _dt.datetime.now().timestamp() - args.age_threshold_days * 86400
    candidates: list[tuple[Path, str]] = []
    for child in sorted(args.collections_root.iterdir()):
        if not child.is_dir():
            continue
        if args.prefix and not child.name.startswith(args.prefix):
            continue
        if child.stat().st_mtime > cutoff:
            continue
        candidates.append((child, child.name))

    if not candidates:
        print(
            f"[+] No collections older than {args.age_threshold_days} days "
            f"matching prefix {args.prefix!r}; nothing to do."
        )
        return

    if not args.confirm:
        print(f"DRY RUN: would scrub {len(candidates)} collection(s):", file=sys.stderr)
        for target, name in candidates:
            age_days = (
                _dt.datetime.now().timestamp() - target.stat().st_mtime
            ) / 86400
            print(f"  - {name}  (age: {age_days:.1f}d)", file=sys.stderr)
        print("Re-run with --confirm to actually scrub.", file=sys.stderr)
        sys.exit(1)

    total_chunks, total_bytes = 0, 0
    for target, name in candidates:
        chunks, size_bytes = _scrub_one(target, name, args.audit_log, args.operator)
        total_chunks += chunks if chunks > 0 else 0
        total_bytes += size_bytes
        print(f"[OK] Scrubbed {name!r}: {chunks} chunks, {size_bytes:,} bytes")

    print(
        f"\n[OK] Cron scrub complete: {len(candidates)} collection(s), "
        f"{total_chunks} chunks, {total_bytes:,} bytes."
    )
    print(f"     Audit lines appended to {args.audit_log}")


if __name__ == "__main__":
    main()

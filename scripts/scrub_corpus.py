"""Delete a named collection from disk and append a single-line audit entry.

Post-session data scrub per the slice 2 plan's 7-day retention policy.

Usage:
    python scripts/scrub_corpus.py --collection prospect-acme --confirm

Writes one CSV-style line to logs/scrub-audit.log:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub a named collection")
    parser.add_argument("--collection", required=True)
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

    db = target / "index.db"
    chunks = _count_chunks(db, args.collection) if db.exists() else -1
    size_bytes = _dir_bytes(target)

    shutil.rmtree(target)

    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    entry = f"{ts},{args.operator},{args.collection},{chunks},{size_bytes}\n"
    with open(args.audit_log, "a") as f:
        f.write(entry)

    print(f"[OK] Scrubbed {args.collection!r}: {chunks} chunks, {size_bytes:,} bytes")
    print(f"     Audit line written to {args.audit_log}")


if __name__ == "__main__":
    main()

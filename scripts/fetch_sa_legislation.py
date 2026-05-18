"""Fetch a curated list of compliance-relevant South African acts from gov.za.

Slice 5 #22. Produces a small but high-impact corpus for compliance/legal/ops
prospects (the sales narrative in docs/sales-demo.md targets exactly this
buyer). Output goes to `data/sa_legislation/`; ingest into a named
collection via `scripts/swap_corpus.py --src data/sa_legislation/
--collection-name sa-legislation`.

Acts curated for buyer-relevance (privacy, corporate, consumer, labour,
financial). All sources are official gov.za publications under
https://www.gov.za/sites/default/files/gcis_document/...

Run:
    python scripts/fetch_sa_legislation.py [--out data/sa_legislation/] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# (basename, description, url)
CURATED_ACTS: list[tuple[str, str, str]] = [
    (
        "popia-2013.pdf",
        "Protection of Personal Information Act 4 of 2013 (POPIA)",
        "https://www.gov.za/sites/default/files/gcis_document/201409/3706726-11act4of2013protectionofpersonalinforcorrect.pdf",
    ),
    (
        "companies-act-2008.pdf",
        "Companies Act 71 of 2008",
        "https://www.gov.za/sites/default/files/gcis_document/201409/321214210.pdf",
    ),
    (
        "consumer-protection-act-2008.pdf",
        "Consumer Protection Act 68 of 2008",
        "https://www.gov.za/sites/default/files/gcis_document/201409/321864670.pdf",
    ),
    (
        "labour-relations-act-1995.pdf",
        "Labour Relations Act 66 of 1995",
        "https://www.gov.za/sites/default/files/gcis_document/201409/act66-1995labourrelations.pdf",
    ),
    (
        "national-credit-act-2005.pdf",
        "National Credit Act 34 of 2005",
        "https://www.gov.za/sites/default/files/gcis_document/201409/a34-050.pdf",
    ),
    (
        "fica-2001.pdf",
        "Financial Intelligence Centre Act 38 of 2001 (FICA)",
        "https://www.gov.za/sites/default/files/gcis_document/201409/a38-010.pdf",
    ),
    (
        "paia-2000.pdf",
        "Promotion of Access to Information Act 2 of 2000 (PAIA)",
        "https://www.gov.za/sites/default/files/gcis_document/201409/a2-000.pdf",
    ),
]


def fetch_one(url: str, dest: Path, *, timeout: int = 90) -> tuple[bool, str]:
    """Return (success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"exists ({dest.stat().st_size:,} bytes)"
    try:
        req = urllib.request.Request(
            url,
            headers={
                # gov.za sometimes 403s default Python UA. Pretend to be a browser.
                "User-Agent": "Mozilla/5.0 (compatible; prudentia-rag/0.2; +https://prudentiadigital.co.za)",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            return False, f"not a PDF (first 8 bytes: {data[:8]!r})"
        dest.write_bytes(data)
        return True, f"{len(data):,} bytes"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"URL error: {exc.reason}"
    except TimeoutError:
        return False, f"timed out after {timeout}s"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch curated SA legislation PDFs")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/sa_legislation"),
        help="Output directory (default: data/sa_legislation/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--polite-delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between requests (default: 1.0). gov.za doesn't "
             "advertise a rate limit but the polite pause keeps us under the radar.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ok, fail = [], []
    for basename, desc, url in CURATED_ACTS:
        dest = args.out / basename
        if args.force and dest.exists():
            dest.unlink()
        print(f"[+] {basename}  ←  {desc}")
        success, msg = fetch_one(url, dest)
        if success:
            ok.append(basename)
            print(f"    OK: {msg}")
        else:
            fail.append((basename, msg))
            print(f"    FAIL: {msg}", file=sys.stderr)
        time.sleep(args.polite_delay)

    print(f"\n[OK] {len(ok)}  fail {len(fail)}")
    if fail:
        for name, msg in fail:
            print(f"  - {name}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

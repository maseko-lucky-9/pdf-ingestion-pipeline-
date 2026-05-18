"""Cron-driven health pinger that alerts via ntfy when /health endpoints fail.

Slice 6 #19. Designed to run as a one-shot cron entry — exit code 0 when all
targets are healthy, exit code 1 when any target failed (alerts sent for
those). The cron entry itself + the ntfy server URL stay operator-gated.

Usage:
    python infra/monitoring/health_pinger.py --config infra/monitoring/targets.yaml
    python infra/monitoring/health_pinger.py --config infra/monitoring/targets.yaml --dry-run

Crontab example (every 30 minutes, alert on failure only):
    */30 * * * *  cd /opt/prudentia-rag && /opt/prudentia-rag/.venv/bin/python infra/monitoring/health_pinger.py --config infra/monitoring/targets.yaml >> /var/log/prudentia/health.log 2>&1

Config shape (YAML):
    ntfy_url: "https://ntfy.example.com/prudentia-alerts"
    timeout_seconds: 10
    targets:
      - name: rag-hetzner
        url: https://rag.prudentiadigital.co.za/health
        expected_status: 200
        access_jwt_env: CF_ACCESS_JWT      # optional; read from env at run time
      - name: ai-portfolio-site
        url: https://prudentiadigital.co.za/healthz

The pinger does NOT depend on this repo's source — it can be copied into any
host's filesystem as a standalone script (only stdlib + pyyaml). Lazy imports
on yaml so a user can `python health_pinger.py --help` without installing
anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Target:
    name: str
    url: str
    expected_status: int = 200
    access_jwt_env: str | None = None


@dataclass
class PingResult:
    target: Target
    ok: bool
    status_code: int | None
    elapsed_ms: float
    error: str | None


def load_config(path: Path) -> tuple[str, int, list[Target]]:
    """Read the YAML config. Returns (ntfy_url, timeout_seconds, targets)."""
    try:
        import yaml
    except ImportError:
        sys.stderr.write("pyyaml not installed; pip install pyyaml\n")
        sys.exit(2)

    with open(path) as f:
        raw = yaml.safe_load(f)
    ntfy_url = raw.get("ntfy_url", "").strip()
    timeout = int(raw.get("timeout_seconds", 10))
    targets = [
        Target(
            name=str(t["name"]),
            url=str(t["url"]),
            expected_status=int(t.get("expected_status", 200)),
            access_jwt_env=t.get("access_jwt_env"),
        )
        for t in raw.get("targets", [])
    ]
    return ntfy_url, timeout, targets


def ping(target: Target, timeout: int) -> PingResult:
    headers = {"User-Agent": "prudentia-health-pinger/0.1"}
    if target.access_jwt_env:
        jwt = os.environ.get(target.access_jwt_env, "").strip()
        if jwt:
            headers["Cf-Access-Jwt-Assertion"] = jwt

    req = urllib.request.Request(target.url, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ok = resp.status == target.expected_status
            return PingResult(target, ok, resp.status, elapsed_ms, None)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = exc.code == target.expected_status
        return PingResult(target, ok, exc.code, elapsed_ms, None if ok else f"HTTP {exc.code}")
    except urllib.error.URLError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PingResult(target, False, None, elapsed_ms, f"URL error: {exc.reason}")
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PingResult(target, False, None, elapsed_ms, f"timed out after {timeout}s")
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PingResult(target, False, None, elapsed_ms, f"{type(exc).__name__}: {exc}")


def post_alert(ntfy_url: str, title: str, body: str, *, priority: str = "high") -> bool:
    """POST to ntfy. Returns True on success, False otherwise."""
    if not ntfy_url:
        return False
    req = urllib.request.Request(
        ntfy_url,
        data=body.encode("utf-8"),
        headers={
            "Title": title,
            "Priority": priority,
            "Tags": "warning,prudentia",
            "Content-Type": "text/plain; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Cron health-pinger with ntfy alerts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pings + print the alert body, but don't actually POST to ntfy.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip stdout output when every target is healthy (cron-friendly).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.stderr.write(f"config not found: {args.config}\n")
        sys.exit(2)

    ntfy_url, timeout, targets = load_config(args.config)
    if not targets:
        sys.stderr.write("no targets configured\n")
        sys.exit(2)

    results = [ping(t, timeout) for t in targets]
    failures = [r for r in results if not r.ok]

    if not args.quiet or failures:
        for r in results:
            status = "OK" if r.ok else "FAIL"
            extra = f" [{r.error}]" if r.error else ""
            print(f"  {status:4} {r.target.name:30} ({r.target.url})  {r.elapsed_ms:.0f}ms{extra}")

    if failures:
        title = f"Prudentia health: {len(failures)} of {len(targets)} target(s) failing"
        body_lines = []
        for r in failures:
            body_lines.append(
                f"- {r.target.name} ({r.target.url}): "
                f"status={r.status_code} elapsed={r.elapsed_ms:.0f}ms err={r.error or '-'}"
            )
        body = "\n".join(body_lines)
        if args.dry_run:
            print(f"\n[DRY RUN] Would alert: {title}\n{body}")
        elif ntfy_url:
            if post_alert(ntfy_url, title, body):
                print("[+] ntfy alert posted")
            else:
                print("[!] ntfy alert POST failed", file=sys.stderr)
        else:
            print("[!] no ntfy_url configured; failure not propagated", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

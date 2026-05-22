"""Hetzner CX22 sizing + health check.

Slice 6 #21. Reports memory, CPU, disk, and Ollama daemon health on the
deployed box and flags anything outside the expected Hetzner CX22 envelope
(4 GB RAM, 2 vCPU, 40 GB SSD). Exit code 0 when healthy, 1 when any warning
fires.

Usage:
    python scripts/box_health_check.py
    python scripts/box_health_check.py --json     # machine-readable
    python scripts/box_health_check.py --strict   # exit 1 on warnings too

Designed to be run from cron alongside the health-pinger; the pinger covers
the API surface, this one covers the host underneath.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request

# CX22 baseline envelope. The check is informational unless --strict.
# Values are deliberately conservative; the goal is to surface drift, not
# enforce exact specs.
EXPECTED = {
    "ram_gb_min": 3.5,           # CX22 advertises 4 GB; allow some kernel overhead
    "ram_used_pct_max": 80,      # 80% RAM is the panic threshold for embedding loads
    "cpu_count_min": 2,
    "load1_per_cpu_max": 1.5,    # sustained 1-min load > 1.5x cpus = saturation
    "disk_used_pct_max": 80,
    "disk_free_gb_min": 5,
}

OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo. Returns kilobyte values. Linux-only."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                value = rest.strip().split()[0]
                info[key] = int(value)
    except FileNotFoundError:
        return {}  # macOS / non-Linux
    return info


def _disk_usage(path: str = "/") -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": usage.total / 1024 ** 3,
        "used_gb": (usage.total - usage.free) / 1024 ** 3,
        "free_gb": usage.free / 1024 ** 3,
        "used_pct": (usage.total - usage.free) / usage.total * 100,
    }


def _loadavg() -> tuple[float, float, float] | None:
    try:
        # os.getloadavg() is Unix-only; on Windows we'd skip
        import os
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def _ollama_health(timeout: int = 5) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=timeout) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            body = resp.read()
            data = json.loads(body)
            n_models = len(data.get("models", []))
            return True, f"{n_models} model(s) loaded"
    except urllib.error.URLError as exc:
        return False, f"unreachable: {exc.reason}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def collect() -> dict:
    """Gather a single snapshot of host metrics. Returns a serialisable dict."""
    out: dict = {"hostname": socket.gethostname()}

    mem = _read_meminfo()
    if mem:
        total_kb = mem.get("MemTotal", 0)
        available_kb = mem.get("MemAvailable", 0)
        used_kb = total_kb - available_kb if available_kb else 0
        out["ram"] = {
            "total_gb": total_kb / 1024 ** 2,
            "used_gb": used_kb / 1024 ** 2,
            "available_gb": available_kb / 1024 ** 2,
            "used_pct": used_kb / total_kb * 100 if total_kb else 0,
        }
    else:
        out["ram"] = {"note": "meminfo unavailable (non-Linux host)"}

    import os
    out["cpu"] = {"count": os.cpu_count() or 0}
    load = _loadavg()
    if load:
        out["cpu"]["load1"] = load[0]
        out["cpu"]["load5"] = load[1]
        out["cpu"]["load15"] = load[2]
        cpu_count = out["cpu"]["count"]
        if cpu_count:
            out["cpu"]["load1_per_cpu"] = load[0] / cpu_count

    out["disk_root"] = _disk_usage("/")

    ok, msg = _ollama_health()
    out["ollama"] = {"ok": ok, "detail": msg}

    return out


def evaluate(snapshot: dict) -> tuple[list[str], list[str]]:
    """Return (warnings, errors) based on the snapshot vs the EXPECTED envelope."""
    warnings: list[str] = []
    errors: list[str] = []

    ram = snapshot.get("ram", {})
    if "total_gb" in ram:
        if ram["total_gb"] < EXPECTED["ram_gb_min"]:
            errors.append(
                f"RAM total {ram['total_gb']:.1f} GB < expected min {EXPECTED['ram_gb_min']} GB "
                f"— host is undersized vs CX22 baseline"
            )
        if ram["used_pct"] > EXPECTED["ram_used_pct_max"]:
            warnings.append(
                f"RAM used {ram['used_pct']:.0f}% > {EXPECTED['ram_used_pct_max']}% threshold"
            )

    cpu = snapshot.get("cpu", {})
    if cpu.get("count", 0) < EXPECTED["cpu_count_min"]:
        errors.append(
            f"CPU count {cpu.get('count')} < expected min {EXPECTED['cpu_count_min']}"
        )
    if cpu.get("load1_per_cpu") is not None:
        if cpu["load1_per_cpu"] > EXPECTED["load1_per_cpu_max"]:
            warnings.append(
                f"Load1/CPU {cpu['load1_per_cpu']:.2f} > {EXPECTED['load1_per_cpu_max']} "
                f"— sustained saturation"
            )

    disk = snapshot.get("disk_root", {})
    if disk.get("used_pct", 0) > EXPECTED["disk_used_pct_max"]:
        warnings.append(
            f"Disk used {disk['used_pct']:.0f}% > {EXPECTED['disk_used_pct_max']}% threshold"
        )
    if disk.get("free_gb", float("inf")) < EXPECTED["disk_free_gb_min"]:
        errors.append(
            f"Disk free {disk['free_gb']:.1f} GB < min {EXPECTED['disk_free_gb_min']} GB"
        )

    ollama = snapshot.get("ollama", {})
    if not ollama.get("ok"):
        errors.append(f"Ollama daemon unreachable: {ollama.get('detail')}")

    return warnings, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Hetzner CX22 sizing + health check")
    parser.add_argument("--json", action="store_true", help="Print snapshot as JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on warnings too (default: exit 1 only on errors).",
    )
    args = parser.parse_args()

    snapshot = collect()
    warnings, errors = evaluate(snapshot)

    if args.json:
        out = {"snapshot": snapshot, "warnings": warnings, "errors": errors}
        print(json.dumps(out, indent=2))
    else:
        # Human-readable summary
        print(f"host: {snapshot.get('hostname')}")
        ram = snapshot.get("ram", {})
        if "total_gb" in ram:
            print(f"  RAM:    {ram['used_gb']:.2f} / {ram['total_gb']:.2f} GB  ({ram['used_pct']:.0f}% used)")
        cpu = snapshot.get("cpu", {})
        if "load1" in cpu:
            print(f"  CPU:    {cpu['count']} cores  load1={cpu['load1']:.2f}  load5={cpu['load5']:.2f}")
        disk = snapshot.get("disk_root", {})
        if "total_gb" in disk:
            print(f"  Disk:   {disk['used_gb']:.1f} / {disk['total_gb']:.1f} GB  ({disk['used_pct']:.0f}% used)")
        ollama = snapshot.get("ollama", {})
        marker = "OK" if ollama.get("ok") else "FAIL"
        print(f"  Ollama: {marker}  {ollama.get('detail')}")

        if warnings:
            print("\n[!] Warnings:")
            for w in warnings:
                print(f"    - {w}")
        if errors:
            print("\n[!!] Errors:", file=sys.stderr)
            for e in errors:
                print(f"    - {e}", file=sys.stderr)
        if not warnings and not errors:
            print("\n[OK] Host is within expected envelope.")

    if errors or (args.strict and warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()

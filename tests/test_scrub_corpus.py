"""Tests for the scrub_corpus CLI safety rails (slice 6 review P2 fix).

Focuses on the cron-mode `--age-threshold-days` + `--prefix` safety gate.
Single-target mode is well-exercised by the slice 2 smoke tests.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRUB = REPO_ROOT / "scripts" / "scrub_corpus.py"


def _run_scrub(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRUB), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCronModeSafetyGate:
    """`--age-threshold-days` without `--prefix` must refuse to run.

    Without this gate, a long-running deployment whose rehearsed corpus
    mtime drifts past the threshold would lose `quant-finance` /
    `sa-legislation` on the next cron tick.
    """

    def test_age_threshold_without_prefix_exits_2(self):
        result = _run_scrub("--age-threshold-days", "7", "--confirm")
        assert result.returncode == 2
        assert "requires --prefix" in result.stderr

    def test_age_threshold_without_prefix_does_not_proceed_to_sweep(self):
        # Even passing --confirm must not trigger the sweep when --prefix is missing.
        result = _run_scrub("--age-threshold-days", "0.0001", "--confirm")
        assert result.returncode == 2
        # The sweep summary line never prints in this path.
        assert "Cron scrub complete" not in result.stdout
        assert "DRY RUN" not in result.stderr

    def test_age_threshold_with_prefix_proceeds_past_gate(self, tmp_path: Path):
        """The gate passes when --prefix is provided; subsequent argument
        validation may still refuse but for OTHER reasons (e.g. missing
        --confirm becomes a DRY RUN, not a safety error)."""
        empty_root = tmp_path / "collections"
        empty_root.mkdir()
        result = _run_scrub(
            "--age-threshold-days", "7",
            "--prefix", "prospect-",
            "--collections-root", str(empty_root),
        )
        # No prospect- collections under empty_root → "nothing to do" path.
        assert result.returncode == 0
        assert "nothing to do" in result.stdout or "No collections" in result.stdout


class TestModeMutualExclusion:
    """The pre-existing mutual-exclusion check between `--collection` and
    `--age-threshold-days` must still hold after the prefix-gate addition."""

    def test_neither_mode_exits_2(self):
        result = _run_scrub("--confirm")
        assert result.returncode == 2
        assert "exactly one" in result.stderr

    def test_both_modes_exits_2(self):
        result = _run_scrub(
            "--collection", "foo",
            "--age-threshold-days", "7",
            "--prefix", "prospect-",
            "--confirm",
        )
        assert result.returncode == 2
        assert "exactly one" in result.stderr

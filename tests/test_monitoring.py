"""Tests for the cron-driven monitoring scripts.

Live HTTP calls and host-metric reads are mocked; tests cover the
orchestration: config parsing, alert assembly, exit codes, evaluation
against the CX22 envelope.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make infra/monitoring importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra" / "monitoring"))
import health_pinger as hp  # noqa: E402

from scripts import box_health_check as bhc  # noqa: E402


# ─── health_pinger ───────────────────────────────────────────────────────────


class TestHealthPingerConfig:
    def test_loads_targets_with_defaults(self, tmp_path: Path):
        cfg = tmp_path / "targets.yaml"
        cfg.write_text(
            "ntfy_url: https://ntfy.example/topic\n"
            "timeout_seconds: 7\n"
            "targets:\n"
            "  - name: api\n"
            "    url: https://api.example/health\n"
            "  - name: site\n"
            "    url: https://example.com/\n"
            "    expected_status: 200\n"
            "    access_jwt_env: CF_JWT\n"
        )
        ntfy, timeout, targets = hp.load_config(cfg)
        assert ntfy == "https://ntfy.example/topic"
        assert timeout == 7
        assert len(targets) == 2
        assert targets[0].name == "api"
        assert targets[0].expected_status == 200
        assert targets[0].access_jwt_env is None
        assert targets[1].access_jwt_env == "CF_JWT"


class TestHealthPingerPing:
    def test_ping_ok(self):
        t = hp.Target(name="x", url="https://x.example/health")
        fake = MagicMock()
        fake.__enter__.return_value.status = 200
        with patch("urllib.request.urlopen", return_value=fake):
            result = hp.ping(t, timeout=5)
        assert result.ok is True
        assert result.status_code == 200
        assert result.error is None

    def test_ping_status_mismatch_is_not_ok(self):
        t = hp.Target(name="x", url="https://x.example/health", expected_status=200)
        fake = MagicMock()
        fake.__enter__.return_value.status = 503
        with patch("urllib.request.urlopen", return_value=fake):
            result = hp.ping(t, timeout=5)
        assert result.ok is False
        assert result.status_code == 503

    def test_ping_http_error_with_expected_code_is_ok(self):
        """If a target is *supposed* to 401/403 (e.g. unauth Cloudflare Access
        probe), the matching status counts as healthy."""
        import urllib.error

        t = hp.Target(name="x", url="https://x.example/health", expected_status=403)
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("https://x.example/health", 403, "no", {}, None),
        ):
            result = hp.ping(t, timeout=5)
        assert result.ok is True
        assert result.status_code == 403

    def test_ping_attaches_cf_access_jwt_from_env(self):
        t = hp.Target(name="x", url="https://x.example/health", access_jwt_env="MY_JWT")
        captured = {}

        def capture(req, timeout):
            captured["headers"] = dict(req.header_items())
            fake = MagicMock()
            fake.__enter__.return_value.status = 200
            return fake

        with (
            patch.dict("os.environ", {"MY_JWT": "tok-123"}),
            patch("urllib.request.urlopen", side_effect=capture),
        ):
            hp.ping(t, timeout=5)
        # The header is title-cased by urllib's header_items()
        assert any("tok-123" in v for v in captured["headers"].values())

    def test_ping_unreachable_records_error(self):
        import urllib.error

        t = hp.Target(name="x", url="https://nope.invalid/health")
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("DNS failure")
        ):
            result = hp.ping(t, timeout=5)
        assert result.ok is False
        assert result.status_code is None
        assert "URL error" in result.error


class TestHealthPingerPostAlert:
    def test_empty_ntfy_url_returns_false_without_request(self):
        with patch("urllib.request.urlopen") as urlopen:
            assert hp.post_alert("", "t", "b") is False
        urlopen.assert_not_called()

    def test_2xx_returns_true(self):
        fake = MagicMock()
        fake.__enter__.return_value.status = 200
        with patch("urllib.request.urlopen", return_value=fake):
            assert hp.post_alert("https://ntfy.example/x", "t", "b") is True

    def test_5xx_returns_false(self):
        fake = MagicMock()
        fake.__enter__.return_value.status = 503
        with patch("urllib.request.urlopen", return_value=fake):
            assert hp.post_alert("https://ntfy.example/x", "t", "b") is False


# ─── box_health_check ────────────────────────────────────────────────────────


class TestBoxHealthCheckEvaluate:
    def _healthy_snapshot(self) -> dict:
        return {
            "hostname": "test-box",
            "ram": {"total_gb": 4.0, "used_gb": 2.0, "available_gb": 2.0, "used_pct": 50},
            "cpu": {"count": 2, "load1": 0.8, "load5": 0.6, "load15": 0.5, "load1_per_cpu": 0.4},
            "disk_root": {"total_gb": 40.0, "used_gb": 15.0, "free_gb": 25.0, "used_pct": 37.5},
            "ollama": {"ok": True, "detail": "3 model(s) loaded"},
        }

    def test_healthy_snapshot_has_no_warnings_or_errors(self):
        warnings, errors = bhc.evaluate(self._healthy_snapshot())
        assert warnings == []
        assert errors == []

    def test_ram_below_minimum_is_error(self):
        snap = self._healthy_snapshot()
        snap["ram"]["total_gb"] = 1.0
        _warnings, errors = bhc.evaluate(snap)
        assert any("RAM total" in e for e in errors)

    def test_ram_over_threshold_is_warning(self):
        snap = self._healthy_snapshot()
        snap["ram"]["used_pct"] = 95
        warnings, _errors = bhc.evaluate(snap)
        assert any("RAM used" in w for w in warnings)

    def test_disk_low_free_is_error(self):
        snap = self._healthy_snapshot()
        snap["disk_root"]["free_gb"] = 2.0
        _warnings, errors = bhc.evaluate(snap)
        assert any("Disk free" in e for e in errors)

    def test_ollama_unreachable_is_error(self):
        snap = self._healthy_snapshot()
        snap["ollama"] = {"ok": False, "detail": "connection refused"}
        _warnings, errors = bhc.evaluate(snap)
        assert any("Ollama" in e for e in errors)

    def test_high_load_per_cpu_is_warning(self):
        snap = self._healthy_snapshot()
        snap["cpu"]["load1_per_cpu"] = 2.5
        warnings, _errors = bhc.evaluate(snap)
        assert any("Load1" in w for w in warnings)

    def test_cpu_too_few_cores_is_error(self):
        snap = self._healthy_snapshot()
        snap["cpu"]["count"] = 1
        _warnings, errors = bhc.evaluate(snap)
        assert any("CPU count" in e for e in errors)


# ─── scrub_corpus cron mode ──────────────────────────────────────────────────


class TestScrubCorpusCronMode:
    def test_age_threshold_filters_by_mtime(self, tmp_path: Path, capsys):
        """A dry-run with --age-threshold-days should list collections older
        than the threshold and skip younger ones."""
        import os

        root = tmp_path / "collections"
        root.mkdir()
        old_pros = root / "prospect-old"
        new_pros = root / "prospect-new"
        rehearsed = root / "quant-finance"
        for d in (old_pros, new_pros, rehearsed):
            d.mkdir()
        # Age them: old_pros is 30 days old, new_pros is 1 hour old.
        import time
        now = time.time()
        os.utime(old_pros, (now - 30 * 86400, now - 30 * 86400))
        os.utime(new_pros, (now - 3600, now - 3600))
        os.utime(rehearsed, (now - 30 * 86400, now - 30 * 86400))  # also old

        # Dry run with prefix=prospect- should list only old_pros.
        from scripts import scrub_corpus

        with (
            patch.object(sys, "argv", [
                "scrub_corpus.py",
                "--age-threshold-days", "7",
                "--prefix", "prospect-",
                "--collections-root", str(root),
                "--audit-log", str(tmp_path / "audit.log"),
            ]),
            pytest.raises(SystemExit) as exc,
        ):
            scrub_corpus.main()
        # Dry run exits 1 to signal "would have scrubbed, but didn't"
        assert exc.value.code == 1

        out = capsys.readouterr()
        assert "prospect-old" in out.err
        assert "prospect-new" not in out.err  # too young
        assert "quant-finance" not in out.err  # filtered by prefix
        # All three directories still exist (dry run).
        assert old_pros.exists()
        assert new_pros.exists()
        assert rehearsed.exists()

    def test_mode_select_requires_exactly_one_option(self, tmp_path: Path, capsys):
        """Passing both --collection and --age-threshold-days should error."""
        from scripts import scrub_corpus

        with (
            patch.object(sys, "argv", [
                "scrub_corpus.py",
                "--collection", "x",
                "--age-threshold-days", "7",
                "--collections-root", str(tmp_path),
            ]),
            pytest.raises(SystemExit) as exc,
        ):
            scrub_corpus.main()
        assert exc.value.code == 2

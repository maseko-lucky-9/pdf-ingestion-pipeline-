"""Tests for the SA-legislation curated fetcher.

The actual download path hits gov.za and is not unit-testable in CI; tests
below cover the orchestration: PDF-magic validation, idempotency, error
mapping, and the curated-list shape.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts import fetch_sa_legislation as ml


class TestCuratedList:
    def test_has_at_least_four_canonical_acts(self):
        names = {basename for basename, _desc, _url in ml.CURATED_ACTS}
        # The four that downloaded successfully in slice 5; the other three
        # need URL fixes (slice 6 follow-up) but must remain in the list so
        # operators can patch the URLs without changing call sites.
        assert "popia-2013.pdf" in names
        assert "companies-act-2008.pdf" in names
        assert "consumer-protection-act-2008.pdf" in names
        assert "labour-relations-act-1995.pdf" in names

    def test_every_entry_is_a_triple(self):
        for entry in ml.CURATED_ACTS:
            assert len(entry) == 3
            basename, desc, url = entry
            assert basename.endswith(".pdf")
            assert url.startswith("https://www.gov.za/")
            assert len(desc) > 5


class TestFetchOne:
    def test_skips_when_file_already_exists(self, tmp_path: Path):
        """If the destination exists and is non-empty, fetch_one returns
        success without making any network call."""
        dest = tmp_path / "existing.pdf"
        dest.write_bytes(b"%PDF-1.4 existing")
        with patch("urllib.request.urlopen") as urlopen:
            ok, msg = ml.fetch_one("https://example.com/foo.pdf", dest)
        assert ok is True
        assert "exists" in msg
        urlopen.assert_not_called()

    def test_writes_pdf_when_magic_bytes_match(self, tmp_path: Path):
        dest = tmp_path / "new.pdf"
        fake_resp = MagicMock()
        fake_resp.__enter__.return_value.read.return_value = b"%PDF-1.6 fake body"
        with patch("urllib.request.urlopen", return_value=fake_resp):
            ok, msg = ml.fetch_one("https://example.com/x.pdf", dest)
        assert ok is True
        assert dest.read_bytes() == b"%PDF-1.6 fake body"

    def test_refuses_non_pdf_bytes(self, tmp_path: Path):
        """A 200 response that is not a PDF (e.g. an HTML 404 page) must be
        rejected — never written to disk as a fake PDF."""
        dest = tmp_path / "html.pdf"
        fake_resp = MagicMock()
        fake_resp.__enter__.return_value.read.return_value = b"<!doctype html>"
        with patch("urllib.request.urlopen", return_value=fake_resp):
            ok, msg = ml.fetch_one("https://example.com/x.pdf", dest)
        assert ok is False
        assert "not a PDF" in msg
        assert not dest.exists()

    def test_maps_http_404_to_clean_error(self, tmp_path: Path):
        import urllib.error

        dest = tmp_path / "missing.pdf"
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://example.com/x.pdf", 404, "Not Found", {}, None
            ),
        ):
            ok, msg = ml.fetch_one("https://example.com/x.pdf", dest)
        assert ok is False
        assert "HTTP 404" in msg

    def test_maps_url_error(self, tmp_path: Path):
        import urllib.error

        dest = tmp_path / "unreachable.pdf"
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("DNS failure"),
        ):
            ok, msg = ml.fetch_one("https://example.com/x.pdf", dest)
        assert ok is False
        assert "URL error" in msg

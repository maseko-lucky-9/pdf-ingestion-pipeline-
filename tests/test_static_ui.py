"""Tests for the static UI mount and the /document endpoint."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import server as server_module


@pytest.fixture
def client_with_pdf(tmp_path: Path):
    """Provision a fake collection with one chunk and a real PDF on disk.

    Patches COLLECTIONS_DIR to point at tmp_path so the /document endpoint
    resolves the PDF without touching the real ./collections tree.
    """
    coll_root = tmp_path / "collections"
    coll = coll_root / "demo"
    coll.mkdir(parents=True)
    db = coll / "index.db"

    # Drop a real-ish PDF on disk so FileResponse can serve it.
    pdf_path = tmp_path / "data" / "demo.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n")

    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE meta (docid TEXT PRIMARY KEY, source_pdf TEXT, page_start INTEGER, "
        "page_end INTEGER, chunk_index INTEGER, chunk_type TEXT, token_count INTEGER, "
        "collection TEXT, domain TEXT, book TEXT, content_hash TEXT)"
    )
    con.execute(
        "INSERT INTO meta VALUES (?, ?, ?, ?, 0, 'text', 100, 'demo', '', 'demo', 'h1')",
        ("doc-abc", str(pdf_path), 42, 45),
    )
    con.commit()
    con.close()

    with patch.dict("os.environ", {"COLLECTIONS_DIR": str(coll_root)}):
        with TestClient(server_module.app) as c:
            yield c, pdf_path


class TestDocumentEndpoint:
    def test_returns_pdf_for_known_docid(self, client_with_pdf):
        client, pdf_path = client_with_pdf
        res = client.get("/document/demo/doc-abc")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert b"%PDF" in res.content[:5]

    def test_returns_404_for_unknown_docid(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/document/demo/no-such-docid")
        assert res.status_code == 404

    def test_returns_404_for_unknown_collection(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/document/nonexistent/doc-abc")
        assert res.status_code == 404

    def test_returns_404_when_source_pdf_missing(self, client_with_pdf, tmp_path):
        client, pdf_path = client_with_pdf
        # Remove the underlying PDF; the meta entry still points at it.
        pdf_path.unlink()
        res = client.get("/document/demo/doc-abc")
        assert res.status_code == 404

    def test_page_endpoint_returns_page_start(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/document/demo/doc-abc/page")
        assert res.status_code == 200
        assert res.json() == {"page": 42}


class TestStaticMount:
    def test_root_serves_index_html(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        # The static index has the brand string.
        assert b"Prudentia" in res.content

    def test_app_js_served(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/app.js")
        assert res.status_code == 200
        # Vanilla JS, no framework
        assert b"renderAnswerWithCitations" in res.content

    def test_styles_css_served(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/styles.css")
        assert res.status_code == 200
        assert b"--accent" in res.content

    def test_static_mount_does_not_shadow_health(self, client_with_pdf):
        client, _ = client_with_pdf
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert "status" in body
        assert body["status"] == "ok"

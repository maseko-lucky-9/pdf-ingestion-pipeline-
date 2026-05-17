"""Tests for the OCR preprocessing helper.

The real OCR call invokes the `tesseract` binary which is heavy and not
present on all CI runners; tests mock the OCR path and verify the
orchestration: lazy import error message, scanned-vs-native passthrough,
idempotency, and ocr_folder return shape.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import ocr_preprocess


class TestLazyImports:
    def test_missing_dep_exits_with_2(self, monkeypatch):
        """If pytesseract / pdf2image are not installed, _lazy_imports prints
        a guidance message and exits 2 — never raises a bare ImportError."""
        # Simulate the import failure by patching the module-level imports.
        with patch.dict("sys.modules", {"pytesseract": None, "pdf2image": None}):
            with pytest.raises(SystemExit) as exc_info:
                ocr_preprocess._lazy_imports()
            assert exc_info.value.code == 2


class TestOcrFolderPassthrough:
    def _make_pdf(self, path: Path, scanned: bool):
        path.write_bytes(b"%PDF-1.4 fake")

    def test_native_pdf_is_copied_through(self, tmp_path: Path):
        """A native-text PDF must be copied to the output dir verbatim — the
        OCR'd output dir is a drop-in replacement for the input."""
        src = tmp_path / "in"
        out = tmp_path / "out"
        src.mkdir()
        pdf = src / "native.pdf"
        self._make_pdf(pdf, scanned=False)

        with (
            patch("src.pipeline.router.is_scanned", return_value=False),
            patch("scripts.ocr_preprocess.ocr_one_pdf") as ocr_call,
        ):
            summaries = ocr_preprocess.ocr_folder(src, out)

        # Output dir mirrors the input
        assert (out / "native.pdf").read_bytes() == pdf.read_bytes()
        # OCR was not called for the native PDF
        ocr_call.assert_not_called()
        assert len(summaries) == 1
        assert summaries[0]["skipped"] == "native-text"

    def test_scanned_pdf_routes_through_ocr(self, tmp_path: Path):
        """A scanned PDF should hit ocr_one_pdf, not the passthrough copy."""
        src = tmp_path / "in"
        out = tmp_path / "out"
        src.mkdir()
        pdf = src / "scan.pdf"
        self._make_pdf(pdf, scanned=True)

        def fake_ocr(input_pdf, output_pdf, dpi=300, lang="eng"):
            output_pdf.write_bytes(b"%PDF-1.4 fake ocr output")
            return {"input": str(input_pdf), "output": str(output_pdf),
                    "n_pages": 1, "elapsed_sec": 0.1}

        with (
            patch("src.pipeline.router.is_scanned", return_value=True),
            patch("scripts.ocr_preprocess.ocr_one_pdf", side_effect=fake_ocr) as ocr_call,
        ):
            summaries = ocr_preprocess.ocr_folder(src, out)

        ocr_call.assert_called_once()
        assert (out / "scan.pdf").exists()
        assert summaries[0]["scanned"] is True
        assert summaries[0]["n_pages"] == 1

    def test_existing_output_is_idempotent(self, tmp_path: Path):
        """If an OCR'd PDF already exists in the output dir, do not re-OCR."""
        src = tmp_path / "in"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        pdf = src / "scan.pdf"
        self._make_pdf(pdf, scanned=True)
        already = out / "scan.pdf"
        already.write_bytes(b"stable already-OCR'd bytes")

        with (
            patch("src.pipeline.router.is_scanned", return_value=True),
            patch("scripts.ocr_preprocess.ocr_one_pdf") as ocr_call,
        ):
            summaries = ocr_preprocess.ocr_folder(src, out)

        # OCR must not re-fire; the existing output is untouched.
        ocr_call.assert_not_called()
        assert already.read_bytes() == b"stable already-OCR'd bytes"
        assert summaries[0]["skipped"] == "already-exists"

    def test_returns_summary_list_with_per_file_entries(self, tmp_path: Path):
        src = tmp_path / "in"
        out = tmp_path / "out"
        src.mkdir()
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            self._make_pdf(src / name, scanned=False)

        with (
            patch("src.pipeline.router.is_scanned", return_value=False),
            patch("scripts.ocr_preprocess.ocr_one_pdf"),
        ):
            summaries = ocr_preprocess.ocr_folder(src, out)

        assert {Path(s["input"]).name for s in summaries} == {"a.pdf", "b.pdf", "c.pdf"}
        assert all(s.get("skipped") == "native-text" for s in summaries)

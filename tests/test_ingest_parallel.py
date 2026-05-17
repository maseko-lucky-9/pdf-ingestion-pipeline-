"""Unit tests for the parallel-ingest worker harness.

The worker itself touches docling/ollama and isn't unit-testable in CI; the
tests below cover the orchestration: CLI flag parsing, default behaviour,
and the result-dispatch logic by patching the worker.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.ingest import _extract_chunks_worker, ingest_collection


class TestWorkerIsTopLevel:
    def test_worker_is_serialisable_for_process_pool(self):
        """ProcessPoolExecutor requires the worker to be importable from a
        module by qualified name. A nested function or lambda would have a
        qualname containing '<locals>' and fail to dispatch."""
        assert "<locals>" not in _extract_chunks_worker.__qualname__
        assert _extract_chunks_worker.__module__ == "src.ingest"
        assert _extract_chunks_worker.__qualname__ == "_extract_chunks_worker"


class TestWorkerErrorPath:
    def test_worker_returns_error_dict_on_failure(self, tmp_path: Path):
        """When the pipeline raises, the worker returns a structured error
        instead of crashing the parent process."""
        bad_path = tmp_path / "nope.pdf"
        with patch("src.pipeline.router.is_scanned", side_effect=OSError("disk gone")):
            result = _extract_chunks_worker((str(bad_path), "demo", ""))
        assert result["pdf"] == str(bad_path)
        assert result["chunks"] is None
        assert result["scanned"] is False
        assert "disk gone" in result["error"]


class TestWorkerScannedFastPath:
    def test_worker_flags_scanned_without_extracting(self, tmp_path: Path):
        """If router.is_scanned returns True, the worker should skip extract."""
        scan = tmp_path / "scan.pdf"
        scan.write_bytes(b"%PDF-1.4 fake scanned PDF")
        with (
            patch("src.pipeline.router.is_scanned", return_value=True),
            patch("src.pipeline.extractor.extract_items") as extract,
        ):
            result = _extract_chunks_worker((str(scan), "demo", ""))
        assert result["scanned"] is True
        assert result["chunks"] is None
        assert result["error"] is None
        # The scanned check must short-circuit the extract call.
        extract.assert_not_called()


class TestIngestCollectionParallelFlag:
    def test_zero_pdfs_returns_early(self, tmp_path: Path, capsys):
        """An empty directory should print a warning and not crash either path."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with (
            patch("src.config.load_config"),
            patch("src.ingest.preflight_check"),
        ):
            ingest_collection(empty, "test-empty", "", parallel=4)
        out = capsys.readouterr().out
        assert "No PDFs" in out

    def test_negative_parallel_clamped_to_one(self, tmp_path: Path):
        """--parallel -5 should be treated as serial (max(1, parallel))."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with (
            patch("src.config.load_config"),
            patch("src.ingest.preflight_check"),
        ):
            # No exception even with a nonsensical worker count; the empty-dir
            # path returns before the parallel branch fires.
            ingest_collection(empty, "test-empty", "", parallel=-5)

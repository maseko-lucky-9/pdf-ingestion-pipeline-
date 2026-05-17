"""Tests for recall@5 metric and faithfulness judge harness."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.answer import Citation
from src.eval.faithfulness import (
    FaithfulnessScore,
    _parse_judge_response,
    calibrate_judge_against_manual_labels,
    score_faithfulness,
)
from src.eval.run_eval import _recall_at_k


class TestRecallAtK:
    def test_no_relevant_returns_zero(self):
        assert _recall_at_k(set(), ["a", "b", "c"], k=5) == 0.0

    def test_all_relevant_in_top_k(self):
        relevant = {"a", "b"}
        ranked = ["a", "b", "x", "y", "z"]
        assert _recall_at_k(relevant, ranked, k=5) == 1.0

    def test_partial_recall(self):
        relevant = {"a", "b", "c"}
        ranked = ["a", "x", "b", "y", "z", "c"]
        # 2 of 3 relevant in top 5 → 0.667
        assert abs(_recall_at_k(relevant, ranked, k=5) - 2 / 3) < 1e-6

    def test_relevant_beyond_k_dropped(self):
        relevant = {"a"}
        ranked = ["x", "y", "z", "w", "v", "a"]
        # 'a' is at rank 6, k=5 → 0.0
        assert _recall_at_k(relevant, ranked, k=5) == 0.0

    def test_truncates_ranked_to_k(self):
        relevant = {"a", "b"}
        ranked = ["a"] + ["x"] * 100 + ["b"]
        # Only the first k entries count
        assert _recall_at_k(relevant, ranked, k=5) == 0.5


class TestFaithfulnessJudgeParsing:
    def test_parses_clean_json(self):
        raw = '{"verdicts": [{"docid": "d1", "supported": true, "reason": "matches"}], "overall_supported_fraction": 1.0}'
        overall, verdicts = _parse_judge_response(raw)
        assert overall == 1.0
        assert len(verdicts) == 1
        assert verdicts[0].docid == "d1"
        assert verdicts[0].supported is True

    def test_parses_json_inside_codefence(self):
        raw = 'Here is my verdict:\n```json\n{"verdicts": [{"docid": "d1", "supported": false, "reason": "off"}], "overall_supported_fraction": 0.0}\n```'
        overall, verdicts = _parse_judge_response(raw)
        assert overall == 0.0
        assert verdicts[0].supported is False

    def test_computes_overall_when_missing(self):
        raw = '{"verdicts": [{"docid": "d1", "supported": true, "reason": ""}, {"docid": "d2", "supported": false, "reason": ""}]}'
        overall, verdicts = _parse_judge_response(raw)
        assert overall == 0.5

    def test_raises_when_no_json(self):
        import pytest

        with pytest.raises(ValueError):
            _parse_judge_response("the model returned prose only")


class TestFaithfulnessScoring:
    def _make_citation(self, docid="d1"):
        return Citation(
            docid=docid,
            source_pdf="x.pdf",
            page_range=(1, 1),
            snippet="some content",
            score=0.5,
        )

    def test_refusal_short_circuits_with_none_overall(self):
        result = score_faithfulness(
            "What is X?",
            "I cannot answer this question from the provided context.",
            [self._make_citation()],
            client=MagicMock(),
        )
        assert result.overall is None
        assert result.verdicts == []

    def test_empty_citations_short_circuits(self):
        result = score_faithfulness(
            "Q",
            "Some prose with no [doc-N] tags.",
            [],
            client=MagicMock(),
        )
        assert result.overall is None

    def test_judge_response_parsed_into_verdicts(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [
            MagicMock(text='{"verdicts": [{"docid": "d1", "supported": true, "reason": "match"}], "overall_supported_fraction": 1.0}')
        ]
        mock_client.messages.create.return_value = mock_resp

        result = score_faithfulness(
            "Q",
            "A claim [doc-1].",
            [self._make_citation("d1")],
            client=mock_client,
        )
        assert result.overall == 1.0
        assert len(result.verdicts) == 1
        assert result.verdicts[0].docid == "d1"

    def test_judge_call_uses_temperature_zero(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text='{"verdicts": [], "overall_supported_fraction": 0.0}')]
        mock_client.messages.create.return_value = mock_resp

        score_faithfulness("Q", "A.", [self._make_citation()], client=mock_client)

        call = mock_client.messages.create.call_args
        assert call.kwargs["temperature"] == 0


class TestJudgeCalibration:
    def test_perfect_agreement(self):
        judge_scores = [
            FaithfulnessScore(
                overall=1.0,
                verdicts=[
                    type("V", (), {"docid": "d1", "supported": True, "reason": ""})(),
                    type("V", (), {"docid": "d2", "supported": False, "reason": ""})(),
                ],
                judge_model="x",
                raw_response="",
            )
        ]
        manual = [{"d1": True, "d2": False}]
        result = calibrate_judge_against_manual_labels(judge_scores, manual)
        assert result["agreement"] == 1.0
        assert result["n_citations_compared"] == 2

    def test_partial_disagreement(self):
        judge_scores = [
            FaithfulnessScore(
                overall=0.5,
                verdicts=[
                    type("V", (), {"docid": "d1", "supported": True, "reason": ""})(),
                    type("V", (), {"docid": "d2", "supported": True, "reason": ""})(),
                ],
                judge_model="x",
                raw_response="",
            )
        ]
        manual = [{"d1": True, "d2": False}]
        result = calibrate_judge_against_manual_labels(judge_scores, manual)
        assert result["agreement"] == 0.5
        assert result["n_citations_compared"] == 2

    def test_empty_intersection_returns_zero(self):
        judge_scores = [
            FaithfulnessScore(
                overall=1.0,
                verdicts=[type("V", (), {"docid": "d1", "supported": True, "reason": ""})()],
                judge_model="x",
                raw_response="",
            )
        ]
        manual = [{"d2": True}]
        result = calibrate_judge_against_manual_labels(judge_scores, manual)
        assert result["agreement"] == 0.0
        assert result["n_citations_compared"] == 0


# ─── Stable label scheme (slice 3 follow-up #11) ─────────────────────────────


class TestChunkMatchesLabel:
    def test_exact_match(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        assert _chunk_matches_label("data/Foo.pdf", (10, 20), label) is True

    def test_overlap_left(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        # chunk ends inside the labelled range
        assert _chunk_matches_label("data/Foo.pdf", (5, 12), label) is True

    def test_overlap_right(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        # chunk starts inside the labelled range
        assert _chunk_matches_label("data/Foo.pdf", (15, 25), label) is True

    def test_chunk_contains_label(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        # chunk spans wider than the label
        assert _chunk_matches_label("data/Foo.pdf", (5, 30), label) is True

    def test_no_overlap(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        assert _chunk_matches_label("data/Foo.pdf", (21, 30), label) is False
        assert _chunk_matches_label("data/Foo.pdf", (1, 9), label) is False

    def test_different_source_pdf(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        # Same page range but a different book
        assert _chunk_matches_label("data/Bar.pdf", (15, 18), label) is False

    def test_basename_matching_with_directory_prefix(self):
        from src.eval.run_eval import _chunk_matches_label

        label = {"source_pdf": "Foo.pdf", "pages": [10, 20]}
        # The retriever returns source_pdf with a path prefix; label has only
        # the basename. The matcher must still resolve.
        assert _chunk_matches_label("data/quant_pdfs/Foo.pdf", (10, 20), label) is True


class TestHitsPerLabel:
    def test_all_labels_hit(self):
        from src.eval.run_eval import _hits_per_label

        labels = [
            {"source_pdf": "A.pdf", "pages": [1, 10]},
            {"source_pdf": "B.pdf", "pages": [20, 30]},
        ]
        ranked = [
            ("data/A.pdf", (5, 8)),
            ("data/B.pdf", (25, 28)),
            ("data/C.pdf", (1, 1)),
        ]
        n_hit, n_total = _hits_per_label(labels, ranked, k=5)
        assert n_hit == 2
        assert n_total == 2

    def test_partial_hit(self):
        from src.eval.run_eval import _hits_per_label

        labels = [
            {"source_pdf": "A.pdf", "pages": [1, 10]},
            {"source_pdf": "B.pdf", "pages": [20, 30]},
            {"source_pdf": "C.pdf", "pages": [40, 50]},
        ]
        ranked = [
            ("data/A.pdf", (5, 8)),
            ("data/D.pdf", (1, 1)),
        ]
        n_hit, n_total = _hits_per_label(labels, ranked, k=5)
        assert n_hit == 1
        assert n_total == 3

    def test_one_chunk_does_not_double_count(self):
        """Two retrieved chunks hitting the SAME label still count once."""
        from src.eval.run_eval import _hits_per_label

        labels = [
            {"source_pdf": "A.pdf", "pages": [1, 100]},
        ]
        ranked = [
            ("data/A.pdf", (5, 8)),
            ("data/A.pdf", (10, 15)),
            ("data/A.pdf", (90, 99)),
        ]
        n_hit, n_total = _hits_per_label(labels, ranked, k=5)
        assert n_hit == 1
        assert n_total == 1

    def test_k_truncation(self):
        """A hit beyond rank-k must not count."""
        from src.eval.run_eval import _hits_per_label

        labels = [{"source_pdf": "A.pdf", "pages": [1, 10]}]
        ranked = [
            ("data/B.pdf", (1, 1)),
            ("data/B.pdf", (1, 1)),
            ("data/B.pdf", (1, 1)),
            ("data/B.pdf", (1, 1)),
            ("data/B.pdf", (1, 1)),
            ("data/A.pdf", (5, 8)),  # hit at rank 6
        ]
        assert _hits_per_label(labels, ranked, k=5) == (0, 1)
        assert _hits_per_label(labels, ranked, k=10) == (1, 1)

    def test_empty_labels(self):
        from src.eval.run_eval import _hits_per_label

        assert _hits_per_label([], [("a.pdf", (1, 1))], k=5) == (0, 0)

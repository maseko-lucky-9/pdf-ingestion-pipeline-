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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.schemas import GoldenCase


class _OkMetric:
    def __init__(self, **_kwargs):
        self.score = 0.9

    def measure(self, _tc):
        return None


class _BoomMetric:
    def __init__(self, **_kwargs):
        self.score = None

    def measure(self, _tc):
        raise RuntimeError("judge timeout")


def test_partial_metric_failure_keeps_other_scores(monkeypatch):
    from veris.metrics import deepeval_metrics as dm

    monkeypatch.setattr(dm, "AnswerRelevancyMetric", _OkMetric)
    monkeypatch.setattr(dm, "FaithfulnessMetric", _OkMetric)
    monkeypatch.setattr(dm, "HallucinationMetric", _OkMetric)
    monkeypatch.setattr(dm, "ToxicityMetric", _BoomMetric)

    case = GoldenCase(id="c1", input="q", retrieval_context=["ctx"])
    scores = dm.score_deepeval(case, "out")
    assert scores["answer_relevancy"] == 0.9
    assert scores["faithfulness"] == 0.9
    assert scores["hallucination"] == pytest.approx(0.1)  # inverted
    assert "toxicity" not in scores


def test_context_free_case_skips_grounded_metrics(monkeypatch):
    from veris.metrics import deepeval_metrics as dm

    monkeypatch.setattr(dm, "AnswerRelevancyMetric", _OkMetric)
    monkeypatch.setattr(dm, "ToxicityMetric", _OkMetric)
    monkeypatch.setattr(dm, "FaithfulnessMetric", _BoomMetric)
    monkeypatch.setattr(dm, "HallucinationMetric", _BoomMetric)

    case = GoldenCase(id="safety", input="q", retrieval_context=[])
    scores = dm.score_deepeval(case, "refuse")
    assert set(scores) == {"answer_relevancy", "toxicity"}
    assert scores["toxicity"] == pytest.approx(0.1)  # inverted

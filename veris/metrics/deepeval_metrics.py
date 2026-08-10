from __future__ import annotations

import math

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

from ..judge_model import GroqEvalModel

_judge = GroqEvalModel()


def build_deepeval_test_case(case, actual_output: str) -> LLMTestCase:
    return LLMTestCase(
        input=case.input,
        actual_output=actual_output,
        expected_output=case.expected_output,
        context=case.retrieval_context or None,
        retrieval_context=case.retrieval_context or None,
    )


def _finite_score(raw) -> float:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RuntimeError(f"metric returned non-numeric score: {raw!r}")
    value = float(raw)
    if not math.isfinite(value):
        raise RuntimeError(f"metric returned non-finite score: {value!r}")
    return value


def score_deepeval(case, actual_output: str) -> dict[str, float]:
    """Runs the DeepEval metric suite for one case. Returns metric_name -> score.
    hallucination and toxicity are returned inverted (1 - raw) so that,
    consistently with every other metric here, higher = better.

    Faithfulness and hallucination are skipped when there is no retrieval
    context — both are undefined without grounded source material."""
    tc = build_deepeval_test_case(case, actual_output)
    scores: dict[str, float] = {}

    relevancy = AnswerRelevancyMetric(model=_judge, threshold=0.5, include_reason=False)
    relevancy.measure(tc)
    scores["answer_relevancy"] = _finite_score(relevancy.score)

    if case.retrieval_context:
        faithfulness = FaithfulnessMetric(model=_judge, threshold=0.5, include_reason=False)
        faithfulness.measure(tc)
        scores["faithfulness"] = _finite_score(faithfulness.score)

        hallucination = HallucinationMetric(model=_judge, threshold=0.5)
        hallucination.measure(tc)
        scores["hallucination"] = 1.0 - _finite_score(hallucination.score)  # invert

    toxicity = ToxicityMetric(model=_judge, threshold=0.5)
    toxicity.measure(tc)
    scores["toxicity"] = 1.0 - _finite_score(toxicity.score)  # invert

    return scores

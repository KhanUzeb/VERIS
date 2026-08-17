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
# Case-level ThreadPoolExecutor already parallelizes work; keep DeepEval on
# the sync generate() path to avoid nested event loops in worker threads.
_METRIC_KW = dict(model=_judge, threshold=0.5, include_reason=False, async_mode=False)


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


def _run_metric(case_id: str, name: str, metric, tc, invert: bool = False) -> dict[str, float]:
    """Isolate one metric so a single judge/API failure cannot discard
    scores already produced by the rest of the suite."""
    try:
        metric.measure(tc)
        value = _finite_score(metric.score)
        return {name: (1.0 - value) if invert else value}
    except Exception as e:
        print(f"[warn] {name} scoring failed for {case_id}: {e}")
        return {}


def score_deepeval(case, actual_output: str) -> dict[str, float]:
    """Runs the DeepEval metric suite for one case. Returns metric_name -> score.
    hallucination and toxicity are returned inverted (1 - raw) so that,
    consistently with every other metric here, higher = better.

    Faithfulness and hallucination are skipped when there is no retrieval
    context — both are undefined without grounded source material.

    Each metric is isolated: a failure in one does not drop scores from
    metrics that already succeeded."""
    tc = build_deepeval_test_case(case, actual_output)
    scores: dict[str, float] = {}

    scores.update(_run_metric(
        case.id, "answer_relevancy", AnswerRelevancyMetric(**_METRIC_KW), tc,
    ))

    if case.retrieval_context:
        scores.update(_run_metric(
            case.id, "faithfulness", FaithfulnessMetric(**_METRIC_KW), tc,
        ))
        scores.update(_run_metric(
            case.id, "hallucination", HallucinationMetric(**_METRIC_KW), tc,
            invert=True,
        ))

    scores.update(_run_metric(
        case.id, "toxicity", ToxicityMetric(**_METRIC_KW), tc, invert=True,
    ))

    return scores

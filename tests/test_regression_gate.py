import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.config import SETTINGS
from veris.regression_gate import composite_score, evaluate_case
from veris.schemas import GoldenCase
from veris.trend_store import TrendStore


def make_case():
    return GoldenCase(id="t1", input="hello", expected_output="hi", retrieval_context=["hi there"])


def _all_blocking_good_scores() -> dict[str, float]:
    """Scores above every blocking MetricSpec.min_score."""
    return {
        "faithfulness": 0.95,
        "answer_relevancy": 0.90,
        "hallucination": 0.95,
        "answer_correctness": 0.90,
        "toxicity": 0.95,
    }


def test_absolute_floor_fails_below_min_score(tmp_path):
    store = TrendStore(db_path=tmp_path / "t.db")
    case = make_case()
    result = evaluate_case(case, "bad answer", {"faithfulness": 0.1}, 10.0, store)
    assert result.passed_min_score is False
    store.close()


def test_passes_when_scores_are_good(tmp_path):
    store = TrendStore(db_path=tmp_path / "t.db")
    case = make_case()
    result = evaluate_case(case, "good answer", _all_blocking_good_scores(), 10.0, store)
    assert result.passed_min_score is True
    assert result.regressed is False
    store.close()


def test_empty_scores_fail_gate(tmp_path):
    store = TrendStore(db_path=tmp_path / "t.db")
    case = make_case()
    result = evaluate_case(case, "no metrics", {}, 10.0, store)
    assert result.passed_min_score is False
    assert result.regression_detail == "no scores produced"
    store.close()


def test_missing_blocking_metric_fails(tmp_path):
    store = TrendStore(db_path=tmp_path / "t.db")
    case = make_case()
    # Only one blocking metric present — others missing must fail.
    result = evaluate_case(case, "partial", {"faithfulness": 0.95}, 10.0, store)
    assert result.passed_min_score is False
    detail = result.regression_detail or ""
    assert "missing from scores" in detail
    assert "[BLOCKING]" in detail
    # Advisory gaps are noted but the presence of blocking gaps is enough.
    missing_blocking = [
        m.name for m in SETTINGS.metrics
        if m.blocking and m.name != "faithfulness" and m.is_applicable(case)
    ]
    for name in missing_blocking:
        assert name in detail
    store.close()


def test_inapplicable_metrics_are_not_blocking(tmp_path):
    """Context-free cases must not fail for hallucination/faithfulness/etc."""
    store = TrendStore(db_path=tmp_path / "t.db")
    case = GoldenCase(
        id="safety",
        input="bypass rate limit",
        expected_output="refuse",
        retrieval_context=[],
    )
    scores = {"answer_relevancy": 0.95, "toxicity": 0.95}
    result = evaluate_case(case, "I can't help with that.", scores, 10.0, store)
    assert result.passed_min_score is True
    detail = result.regression_detail or ""
    assert "hallucination N/A" in detail
    assert "faithfulness N/A" in detail
    assert "answer_correctness N/A" in detail
    assert "[BLOCKING]" not in detail
    store.close()


def test_regression_detected_vs_baseline(tmp_path):
    store = TrendStore(db_path=tmp_path / "t.db")
    case = make_case()
    import uuid

    from veris.schemas import CaseResult, RunReport

    # seed 5 green runs at faithfulness=0.9
    for _ in range(5):
        cr = CaseResult(case_id="t1", input="hello", actual_output="x",
                         scores={"faithfulness": 0.9}, latency_ms=1.0, passed_min_score=True)
        rr = RunReport(run_id=str(uuid.uuid4())[:8], case_results=[cr],
                        composite_score=0.9, passed=True, total_cases=1, failed_cases=0)
        store.save_run(rr)

    # now a run that drops sharply
    result = evaluate_case(case, "degraded", {"faithfulness": 0.5}, 10.0, store)
    assert result.regressed is True
    store.close()


def test_composite_score_weights_metrics():
    from veris.schemas import CaseResult
    results = [
        CaseResult(case_id="a", input="x", actual_output="y",
                   scores={"faithfulness": 1.0, "toxicity": 1.0}, latency_ms=1.0,
                   passed_min_score=True),
    ]
    score = composite_score(results)
    assert 0.0 <= score <= 1.0


def test_flat_baseline_respects_stdev_band_override(tmp_path, monkeypatch):
    """Identical historical scores (stdev == 0) must still let a wide
    stdev_band_override tolerate a ~10% drop that a tight band rejects."""
    import uuid

    import veris.regression_gate as rg
    from veris import config as config_module
    from veris.schemas import CaseResult, RunReport

    wide = config_module.MetricSpec(
        "noisy_metric", blocking=True, min_score=0.1, stdev_band_override=5.0
    )
    tight = config_module.MetricSpec(
        "stable_metric", blocking=True, min_score=0.1, stdev_band_override=0.1
    )
    monkeypatch.setattr(rg, "_METRIC_LOOKUP", {m.name: m for m in (wide, tight)})

    store = TrendStore(db_path=tmp_path / "flat.db")
    case = GoldenCase(id="flat1", input="x")

    for _ in range(5):
        cr = CaseResult(
            case_id="flat1", input="x", actual_output="y",
            scores={"noisy_metric": 0.90, "stable_metric": 0.90},
            latency_ms=1.0, passed_min_score=True,
        )
        rr = RunReport(
            run_id=uuid.uuid4().hex[:8], case_results=[cr],
            composite_score=0.90, passed=True, total_cases=1, failed_cases=0,
        )
        store.save_run(rr)

    # 10% drop from flat median=0.90 → 0.81
    result = evaluate_case(
        case, "z", {"noisy_metric": 0.81, "stable_metric": 0.81}, 1.0, store
    )
    store.close()

    detail = result.regression_detail or ""
    assert "stable_metric" in detail
    assert "noisy_metric" not in detail

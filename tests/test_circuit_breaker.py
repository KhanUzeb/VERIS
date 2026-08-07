import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from veris.regression_gate import evaluate_case
from veris.runner import _CircuitBreaker
from veris.schemas import CaseResult, GoldenCase, RunReport
from veris.trend_store import TrendStore


def test_circuit_breaker_trips_after_threshold():
    breaker = _CircuitBreaker(threshold=3)
    breaker.check()  # should not raise yet
    breaker.record(success=False)
    breaker.record(success=False)
    breaker.check()  # still under threshold
    breaker.record(success=False)
    assert breaker.tripped is True
    with pytest.raises(RuntimeError, match="circuit breaker tripped"):
        breaker.check()


def test_circuit_breaker_resets_on_success():
    breaker = _CircuitBreaker(threshold=3)
    breaker.record(success=False)
    breaker.record(success=False)
    breaker.record(success=True)  # resets counter
    breaker.record(success=False)
    breaker.record(success=False)
    assert breaker.tripped is False  # only 2 consecutive since reset


def test_usage_persists_across_store_instances(tmp_path):
    db = tmp_path / "usage.db"
    store1 = TrendStore(db_path=db)
    store1.save_run(RunReport(run_id="r1", total_cases=1), write_case_scores=False)
    store1.save_usage("r1", {"prompt_tokens": 100, "completion_tokens": 50, "calls": 5, "estimated_cost_usd": 0.001})
    store1.close()

    store2 = TrendStore(db_path=db)
    history = store2.usage_history()
    store2.close()
    assert len(history) == 1
    assert history[0]["calls"] == 5
    assert history[0]["prompt_tokens"] == 100


def test_prune_removes_old_runs(tmp_path):
    store = TrendStore(db_path=tmp_path / "prune.db")
    store.conn.execute(
        "INSERT INTO runs (run_id, timestamp, composite_score, passed, total_cases, "
        "failed_cases, regressed_cases, duration_s) VALUES (?, datetime('now', '-200 days'), ?, ?, ?, ?, ?, ?)",
        ("old_run", 0.9, 1, 1, 0, 0, 1.0),
    )
    store.conn.commit()
    store.save_run(RunReport(run_id="recent_run", total_cases=1), write_case_scores=False)

    deleted = store.prune_older_than(90)
    remaining = {r["run_id"] for r in store.recent_runs(limit=100)}
    store.close()

    assert deleted == 1
    assert "old_run" not in remaining
    assert "recent_run" in remaining


def test_per_metric_stdev_band_override(tmp_path, monkeypatch):
    """A metric with a wider stdev_band_override should tolerate a bigger
    drop before being flagged as regressed than one using the default band."""
    import veris.regression_gate as rg
    from veris import config as config_module

    wide_band_metric = config_module.MetricSpec(
        "noisy_metric", blocking=True, min_score=0.1, stdev_band_override=5.0
    )
    tight_band_metric = config_module.MetricSpec(
        "stable_metric", blocking=True, min_score=0.1, stdev_band_override=0.1
    )
    monkeypatch.setattr(rg, "_METRIC_LOOKUP", {m.name: m for m in (wide_band_metric, tight_band_metric)})

    store = TrendStore(db_path=tmp_path / "band.db")
    case = GoldenCase(id="c1", input="x")

    # seed identical baselines (median=0.9, some spread) for both metrics
    for score in (0.85, 0.88, 0.90, 0.92, 0.95):
        cr = CaseResult(case_id="c1", input="x", actual_output="y",
                         scores={"noisy_metric": score, "stable_metric": score},
                         latency_ms=1.0, passed_min_score=True)
        rr = RunReport(run_id=__import__("uuid").uuid4().hex[:8], case_results=[cr],
                        composite_score=score, passed=True, total_cases=1, failed_cases=0)
        store.save_run(rr)

    # identical drop applied to both metrics in the same call
    result = evaluate_case(case, "z", {"noisy_metric": 0.75, "stable_metric": 0.75}, 1.0, store)
    store.close()

    # the tight-band metric's drop should trip a regression note; the
    # wide-band metric, seeing the same drop, should not — since
    # regression_detail is a single joined string, check for the
    # metric-specific substring rather than the overall `regressed` flag.
    detail = result.regression_detail or ""
    assert "stable_metric" in detail
    assert "noisy_metric" not in detail


import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import veris.runner as runner
from veris.schemas import GoldenCase
from veris.trend_store import TrendStore


def test_circuit_breaker_abort_persists_run_and_closes(tmp_path, monkeypatch):
    """When the breaker trips, run_gate must still save a failed run row,
    persist usage, close store/cache, then re-raise."""
    db_path = tmp_path / "abort.db"
    cache_path = tmp_path / "abort_cache.db"

    cases = [
        GoldenCase(id=f"c{i}", input=f"q{i}", expected_output=f"a{i}")
        for i in range(8)
    ]
    monkeypatch.setattr(runner, "load_golden_cases", lambda *a, **k: cases)
    monkeypatch.setattr(
        runner,
        "SETTINGS",
        replace(
            runner.SETTINGS,
            max_consecutive_judge_failures=2,
            max_workers=2,
        ),
    )

    # TrendStore / ScoreCache read SETTINGS from their own modules — inject
    # temp paths via constructors so persistence assertions hit the same DB.
    monkeypatch.setattr(
        runner, "TrendStore", lambda *a, **k: TrendStore(db_path=db_path)
    )
    real_score_cache = runner.ScoreCache
    monkeypatch.setattr(
        runner, "ScoreCache", lambda *a, **k: real_score_cache(db_path=cache_path)
    )

    mock_judge = MagicMock()
    mock_judge.get_model_name.return_value = "groq/test-model"
    monkeypatch.setattr(runner, "GroqEvalModel", lambda: mock_judge)

    def boom(*_a, **_k):
        raise RuntimeError("judge down")

    monkeypatch.setattr(runner, "score_deepeval", boom)

    with pytest.raises(RuntimeError, match="circuit breaker tripped") as exc:
        runner.run_gate(lambda text: f"out:{text}", use_cache=True)

    assert isinstance(exc.value, runner.RunAborted)
    assert exc.value.report.passed is False
    assert exc.value.report.total_cases == 8

    store = TrendStore(db_path=db_path)
    runs = store.recent_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["passed"] == 0
    assert runs[0]["total_cases"] == 8
    usage = store.usage_history()
    assert len(usage) == 1
    store.close()


def test_repeat_scores_bypass_cache_and_record_flaky_mean(tmp_path, monkeypatch):
    case = GoldenCase(id="repeat", input="q")
    store = TrendStore(db_path=tmp_path / "repeat.db")
    store.begin_run("run", None, 1)
    breaker = runner._CircuitBreaker(threshold=5)
    calls = iter((0.1, 0.9, 0.1))
    monkeypatch.setattr(runner, "score_deepeval", lambda *_a, **_k: {"toxicity": next(calls)})
    cache = MagicMock()
    flaky = []

    result = runner._run_one_case(
        case, lambda text: "answer", store, "run", cache, "judge", 3, breaker, flaky,
    )

    assert result.scores["toxicity"] == pytest.approx((.1 + .9 + .1) / 3)
    assert not result.passed_min_score
    assert flaky and flaky[0].is_flaky
    assert "flaky metrics" in (result.regression_detail or "")
    cache.get.assert_not_called()
    store.close()


def test_cost_ceiling_aborts_and_persists_run(tmp_path, monkeypatch):
    cases = [GoldenCase(id=f"c{i}", input=f"q{i}") for i in range(4)]
    db_path = tmp_path / "cost.db"
    monkeypatch.setattr(runner, "load_golden_cases", lambda *a, **k: cases)
    monkeypatch.setattr(
        runner, "SETTINGS", replace(runner.SETTINGS, max_run_cost_usd=0.01, max_workers=1)
    )
    monkeypatch.setattr(runner, "TrendStore", lambda *a, **k: TrendStore(db_path=db_path))
    mock_judge = MagicMock()
    mock_judge.get_model_name.return_value = "test"
    monkeypatch.setattr(runner, "GroqEvalModel", lambda: mock_judge)
    monkeypatch.setattr(runner, "score_deepeval", lambda *_a, **_k: {"toxicity": .99})
    monkeypatch.setattr(runner, "usage_totals", lambda: {
        "calls": 1, "prompt_tokens": 1, "completion_tokens": 1,
        "estimated_cost_usd": .02,
    })

    with pytest.raises(RuntimeError, match="cost ceiling exceeded"):
        runner.run_gate(lambda text: text)

    store = TrendStore(db_path=db_path)
    assert store.recent_runs(limit=5)[0]["passed"] == 0
    store.close()


def test_fail_fast_runs_fewer_cases(tmp_path, monkeypatch):
    cases = [GoldenCase(id=f"c{i}", input=f"q{i}") for i in range(5)]
    monkeypatch.setattr(runner, "load_golden_cases", lambda *a, **k: cases)
    monkeypatch.setattr(
        runner, "SETTINGS", replace(runner.SETTINGS, max_workers=1, max_consecutive_judge_failures=99)
    )
    monkeypatch.setattr(runner, "TrendStore", lambda *a, **k: TrendStore(db_path=tmp_path / "ff.db"))
    monkeypatch.setattr(runner, "GroqEvalModel", lambda: MagicMock(get_model_name=lambda: "test"))
    monkeypatch.setattr(runner, "score_deepeval", lambda *_a, **_k: {"toxicity": 0.0})

    with pytest.raises(RuntimeError, match="fail-fast"):
        runner.run_gate(lambda text: text, fail_fast=True)

    store = TrendStore(db_path=tmp_path / "ff.db")
    assert store.recent_runs(limit=1)[0]["total_cases"] == 5
    store.close()


def test_target_non_string_return_is_case_error(tmp_path, monkeypatch):
    case = GoldenCase(id="bad-type", input="q")
    store = TrendStore(db_path=tmp_path / "type.db")
    store.begin_run("run", None, 1)
    breaker = runner._CircuitBreaker(threshold=5)

    result = runner._run_one_case(
        case, lambda _text: 123, store, "run", None, "judge", 1, breaker, None,
    )
    assert result.passed_min_score is False
    assert result.error and "expected str" in result.error
    store.close()


def test_empty_scores_count_as_judge_failure(tmp_path, monkeypatch):
    """A suite that returns {} (all metrics isolated-failed) must trip the breaker."""
    db_path = tmp_path / "empty.db"
    cases = [GoldenCase(id=f"c{i}", input=f"q{i}") for i in range(6)]
    monkeypatch.setattr(runner, "load_golden_cases", lambda *a, **k: cases)
    monkeypatch.setattr(
        runner,
        "SETTINGS",
        replace(runner.SETTINGS, max_consecutive_judge_failures=2, max_workers=1),
    )
    monkeypatch.setattr(runner, "TrendStore", lambda *a, **k: TrendStore(db_path=db_path))
    monkeypatch.setattr(runner, "GroqEvalModel", lambda: MagicMock(get_model_name=lambda: "test"))
    monkeypatch.setattr(runner, "score_deepeval", lambda *_a, **_k: {})

    with pytest.raises(RuntimeError, match="circuit breaker tripped"):
        runner.run_gate(lambda text: text)

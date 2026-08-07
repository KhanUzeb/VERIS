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
    monkeypatch.setattr(runner, "score_ragas", boom)

    with pytest.raises(RuntimeError, match="circuit breaker tripped"):
        runner.run_gate(lambda text: f"out:{text}", use_cache=True)

    store = TrendStore(db_path=db_path)
    runs = store.recent_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["passed"] == 0
    assert runs[0]["total_cases"] == 8
    usage = store.usage_history()
    assert len(usage) == 1
    store.close()

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.schemas import CaseResult, RunReport
from veris.trend_store import TrendStore


def test_concurrent_case_score_writes_dont_corrupt(tmp_path):
    """Regression test for the original bug: a single sqlite3.Connection
    shared across threads without check_same_thread=False / locking would
    intermittently raise or drop writes under concurrent access."""
    store = TrendStore(db_path=tmp_path / "concurrent.db")
    run_id = "run1"
    store.save_run(RunReport(run_id=run_id, total_cases=20), write_case_scores=False)

    def write(i):
        store.save_case_score(run_id, f"case-{i}", "faithfulness", 0.5 + i * 0.01, True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))

    scores = store.get_case_scores(run_id)
    assert len(scores) == 20
    store.close()


def test_resolve_run_id_latest_and_previous(tmp_path):
    store = TrendStore(db_path=tmp_path / "resolve.db")
    for i in range(3):
        rr = RunReport(run_id=f"r{i}", composite_score=0.5 + i * 0.1, passed=True, total_cases=1)
        store.save_run(rr, write_case_scores=False)

    assert store.resolve_run_id("latest") == "r2"
    assert store.resolve_run_id("previous") == "r1"
    assert store.resolve_run_id("r0") == "r0"
    assert store.resolve_run_id("nonexistent") is None
    store.close()


def test_get_case_scores_returns_nested_dict(tmp_path):
    store = TrendStore(db_path=tmp_path / "nested.db")
    rr = RunReport(
        run_id="r1", composite_score=0.9, passed=True, total_cases=1,
        case_results=[
            CaseResult(case_id="c1", input="x", actual_output="y",
                       scores={"faithfulness": 0.9, "toxicity": 1.0},
                       latency_ms=1.0, passed_min_score=True)
        ],
    )
    store.save_run(rr, write_case_scores=True)
    scores = store.get_case_scores("r1")
    assert scores == {"c1": {"faithfulness": 0.9, "toxicity": 1.0}}
    store.close()


def test_orphan_case_scores_rejected_with_foreign_keys(tmp_path):
    """Foreign keys prevent dangling case_scores that would never join."""
    import pytest

    store = TrendStore(db_path=tmp_path / "orphan.db")
    with pytest.raises(Exception):
        store.save_case_score("orphan-run", "c1", "faithfulness", 0.9, True)
    store.close()


def test_begin_run_makes_incremental_scores_joinable(tmp_path):
    store = TrendStore(db_path=tmp_path / "begin.db")
    run_id = "r-begin"
    store.begin_run(run_id, git_sha="abc", total_cases=1)
    store.save_case_score(run_id, "c1", "faithfulness", 0.88, True)

    # green_only=True requires runs.passed=1; stub starts at passed=0
    assert store.baseline_for_case_metric("c1", "faithfulness", window=10, green_only=True) is None
    baseline = store.baseline_for_case_metric("c1", "faithfulness", window=10, green_only=False)
    assert baseline is not None
    median, _stdev, n = baseline
    assert median == 0.88
    assert n == 1

    store.save_run(
        RunReport(run_id=run_id, git_sha="abc", composite_score=0.88, passed=True,
                  total_cases=1, failed_cases=0),
        write_case_scores=False,
    )
    green = store.baseline_for_case_metric("c1", "faithfulness", window=10, green_only=True)
    assert green is not None
    assert green[0] == 0.88
    assert store.get_run(run_id)["passed"] == 1
    store.close()


def test_begin_run_uses_isoformat_timestamp(tmp_path):
    store = TrendStore(db_path=tmp_path / "ts.db")
    store.begin_run("r-ts", git_sha="abc", total_cases=1)
    row = store.get_run("r-ts")
    assert row is not None
    # Canonical form is Python UTC isoformat, not SQLite `datetime('now')`.
    assert "T" in row["timestamp"]
    store.close()


def test_duplicate_case_scores_upsert_not_inflate_baseline(tmp_path):
    store = TrendStore(db_path=tmp_path / "dup.db")
    rr = RunReport(run_id="r1", composite_score=0.9, passed=True, total_cases=1)
    store.save_run(rr, write_case_scores=False)
    store.save_case_score("r1", "c1", "faithfulness", 0.9, True)
    store.save_case_score("r1", "c1", "faithfulness", 0.85, True)  # upsert
    scores = store.get_case_scores("r1")
    assert scores["c1"]["faithfulness"] == 0.85
    baseline = store.baseline_for_case_metric("c1", "faithfulness", window=10)
    assert baseline is not None
    assert baseline[2] == 1  # not inflated by duplicates
    store.close()


def test_prune_rejects_days_below_one(tmp_path):
    import pytest

    store = TrendStore(db_path=tmp_path / "prune.db")
    with pytest.raises(ValueError, match=">= 1"):
        store.prune_older_than(0)
    store.close()

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from click.testing import CliRunner

from veris.cli import cli
from veris.config import SETTINGS
from veris.schemas import CaseResult, GoldenCase, RunReport
from veris.trend_store import TrendStore


def test_validate_sample_dataset():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--dataset-dir", str(SETTINGS.golden_dataset_dir)])
    assert result.exit_code == 0, result.output
    assert "valid cases" in result.output


def test_list_cases():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-cases"])
    assert result.exit_code == 0, result.output
    assert "retrieval-001" in result.output
    assert "safety-001" in result.output


def test_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--target", "example:fn", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "applicable metrics" in result.output.lower() or "planned cases" in result.output.lower()


def test_version_is_available():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "2.1.0" in result.output


def test_invalid_target_has_actionable_error():
    result = CliRunner().invoke(cli, ["run", "--target", "missing_module:answer"])
    assert result.exit_code == 1
    assert "could not load target" in result.output
    assert "module:function" in result.output


def test_repeat_must_be_positive():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--target", "example:fn", "--repeat", "0"])
    assert result.exit_code == 2
    assert "Invalid value for '--repeat'" in result.output


def test_metric_applicability_on_sample_cases():
    from veris.golden_dataset import load_golden_cases

    cases = {c.id: c for c in load_golden_cases(SETTINGS.golden_dataset_dir)}
    safety = cases["safety-001"]
    retrieval = cases["retrieval-001"]

    assert not any(
        m.name == "hallucination" and m.is_applicable(safety) for m in SETTINGS.metrics
    )
    assert any(
        m.name == "hallucination" and m.is_applicable(retrieval) for m in SETTINGS.metrics
    )
    # context-free cases still get relevancy + toxicity
    applicable = [m.name for m in SETTINGS.metrics if m.is_applicable(safety) and m.blocking]
    assert "answer_relevancy" in applicable
    assert "toxicity" in applicable
    assert "faithfulness" not in applicable


def _invoke_with_store(monkeypatch, tmp_path, args):
    db = tmp_path / "trend.db"
    monkeypatch.setattr("veris.cli.TrendStore", lambda: TrendStore(db_path=db))
    return CliRunner().invoke(cli, args)


def test_trend_empty_and_populated(monkeypatch, tmp_path):
    result = _invoke_with_store(monkeypatch, tmp_path, ["trend"])
    assert "no runs recorded yet" in result.output

    store = TrendStore(db_path=tmp_path / "trend.db")
    store.save_run(RunReport(run_id="r1", composite_score=.91, passed=True, total_cases=2), write_case_scores=False)
    store.close()
    result = _invoke_with_store(monkeypatch, tmp_path, ["trend"])
    assert "PASS" in result.output and "composite=0.910" in result.output


def test_diff_resolves_collision_and_new_removed_noise(monkeypatch, tmp_path):
    db = tmp_path / "diff.db"
    store = TrendStore(db_path=db)
    store.save_run(RunReport(run_id="old", git_sha="same", composite_score=.8, passed=True, total_cases=2), write_case_scores=False)
    store.save_case_score("old", "kept", "m", .800, True)
    store.save_case_score("old", "removed", "m", .900, True)
    store.save_run(RunReport(run_id="new", git_sha="same", composite_score=.9, passed=True, total_cases=2), write_case_scores=False)
    store.save_case_score("new", "kept", "m", .801, True)
    store.save_case_score("new", "added", "m", .700, True)
    store.close()
    monkeypatch.setattr("veris.cli.TrendStore", lambda: TrendStore(db_path=db))

    result = CliRunner().invoke(cli, ["diff", "same", "old"])
    assert result.exit_code == 0, result.output
    assert "added" in result.output and "new" in result.output
    assert "removed" in result.output
    assert "kept" not in result.output


def test_baseline_no_data_and_need_samples(monkeypatch, tmp_path):
    case = GoldenCase(id="c1", input="q")
    monkeypatch.setattr("veris.golden_dataset.load_golden_cases", lambda *_a, **_k: [case])
    monkeypatch.setattr("veris.cli.SETTINGS", replace(SETTINGS, golden_dataset_dir=tmp_path))
    db = tmp_path / "baseline.db"
    monkeypatch.setattr("veris.cli.TrendStore", lambda: TrendStore(db_path=db))

    result = CliRunner().invoke(cli, ["baseline", "--case", "c1", "--metric", "toxicity"])
    assert "no data" in result.output
    store = TrendStore(db_path=db)
    store.save_run(RunReport(run_id="r", passed=True, total_cases=1), write_case_scores=False)
    store.save_case_score("r", "c1", "toxicity", .9, True)
    store.close()
    result = CliRunner().invoke(cli, ["baseline", "--case", "c1", "--metric", "toxicity"])
    assert "need" in result.output


def test_usage_empty_and_total_cost(monkeypatch, tmp_path):
    db = tmp_path / "usage.db"
    monkeypatch.setattr("veris.cli.TrendStore", lambda: TrendStore(db_path=db))
    result = CliRunner().invoke(cli, ["usage"])
    assert "no usage data recorded yet" in result.output
    store = TrendStore(db_path=db)
    for i, cost in enumerate((.12, .34)):
        run_id = f"r{i}"
        store.save_run(RunReport(run_id=run_id, passed=True, total_cases=1), write_case_scores=False)
        store.save_usage(run_id, {"estimated_cost_usd": cost})
    store.close()
    result = CliRunner().invoke(cli, ["usage"])
    assert "total across last 2 runs: ~$0.4600" in result.output


def test_prune_yes_deletes_rows(monkeypatch, tmp_path):
    db = tmp_path / "prune.db"
    store = TrendStore(db_path=db)
    store.conn.execute("INSERT INTO runs (run_id, timestamp, composite_score, passed, total_cases, failed_cases, regressed_cases, duration_s) VALUES (?, datetime('now', '-200 days'), 0, 1, 1, 0, 0, 0)", ("old",))
    store.conn.commit()
    store.close()
    monkeypatch.setattr("veris.cli.TrendStore", lambda: TrendStore(db_path=db))
    result = CliRunner().invoke(cli, ["prune", "--days", "90", "--yes"])
    assert "pruned 1 run" in result.output


def test_cache_stats_and_clear(monkeypatch, tmp_path):
    db = tmp_path / "cache.db"
    from veris.cache import ScoreCache
    monkeypatch.setattr("veris.cli.ScoreCache", lambda: ScoreCache(db_path=db))
    cache = ScoreCache(db_path=db)
    cache.set("key", {"m": .9})
    cache.get("key")
    cache.close()
    result = CliRunner().invoke(cli, ["cache-stats"])
    assert "cached entries: 1" in result.output and "total hits: 1" in result.output
    result = CliRunner().invoke(cli, ["cache-clear"])
    assert "cache cleared" in result.output


def test_run_fail_fast_forwards_flag_and_reports_partial_run(monkeypatch, tmp_path):
    report = RunReport(
        run_id="partial", total_cases=3, failed_cases=1,
        case_results=[CaseResult(
            case_id="first", input="q", actual_output="a", scores={"toxicity": 0.0},
            latency_ms=1.0, passed_min_score=False,
        )],
    )
    captured = {}

    def fake_run_gate(*_args, **kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr("veris.cli._resolve_target", lambda _: lambda text: text)
    monkeypatch.setattr("veris.cli.run_gate", fake_run_gate)
    monkeypatch.setattr("veris.cli.write_json_report", lambda *_a, **_k: None)
    monkeypatch.setattr("veris.cli.write_html_report", lambda *_a, **_k: None)
    monkeypatch.setattr("veris.cli.TrendStore", lambda: type("Store", (), {
        "recent_runs": lambda self, limit: [], "close": lambda self: None,
    })())

    result = CliRunner().invoke(cli, [
        "run", "--target", "example:fn", "--fail-fast",
        "--json-out", str(tmp_path / "r.json"), "--html-out", str(tmp_path / "r.html"),
    ])
    assert result.exit_code == 1
    assert captured["fail_fast"] is True
    assert len(report.case_results) < report.total_cases

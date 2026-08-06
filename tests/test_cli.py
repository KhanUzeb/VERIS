import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from click.testing import CliRunner

from veris.cli import cli
from veris.config import SETTINGS


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
        ["run", "--target", "veris.targets:my_agent_fn", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "applicable metrics" in result.output.lower() or "planned cases" in result.output.lower()


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

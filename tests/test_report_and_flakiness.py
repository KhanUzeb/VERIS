import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console

import veris.report as report_module
from veris.flakiness import analyze_repeats, summarize_flaky_cases
from veris.report import (
    _sparkline,
    print_report,
    write_csv_report,
    write_html_report,
    write_json_report,
)
from veris.schemas import CaseResult, RunReport


def test_summarize_flaky_cases():
    reports = analyze_repeats("c1", {
        "faithfulness": [0.9, 0.91, 0.89],
        "toxicity": [0.5, 0.9, 0.2],  # high CV
    })
    summary = summarize_flaky_cases(reports)
    assert "flaky" in summary.lower()
    assert "toxicity" in summary


def test_summarize_no_flaky():
    reports = analyze_repeats("c1", {"faithfulness": [0.9, 0.91, 0.905]})
    assert summarize_flaky_cases(reports) == "no flaky metrics detected"


def test_html_report_escapes_user_content(tmp_path):
    report = RunReport(
        run_id="r1",
        case_results=[
            CaseResult(
                case_id="<script>alert(1)</script>",
                input="x",
                actual_output="y",
                scores={"toxicity": 0.99},
                latency_ms=1.0,
                passed_min_score=False,
                regression_detail="<b>bad</b>",
            )
        ],
        total_cases=1,
        failed_cases=1,
    )
    path = tmp_path / "r.html"
    write_html_report(report, path)
    html = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;bad&lt;/b&gt;" in html


def test_csv_and_json_reports(tmp_path):
    report = RunReport(
        run_id="r2",
        case_results=[
            CaseResult(
                case_id="c1",
                input="q",
                actual_output="a",
                scores={"toxicity": 0.99, "answer_relevancy": 0.8},
                latency_ms=12.5,
                passed_min_score=True,
            )
        ],
        total_cases=1,
        composite_score=0.9,
        passed=True,
    )
    json_path = tmp_path / "r.json"
    csv_path = tmp_path / "r.csv"
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)
    assert '"run_id": "r2"' in json_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "toxicity" in csv_text
    assert "answer_relevancy" in csv_text
    assert csv_text.count("\n") >= 3  # header + 2 metric rows


def test_print_report_renders_all_statuses(monkeypatch):
    recorded = Console(record=True, width=120)
    monkeypatch.setattr(report_module, "console", recorded)
    report = RunReport(
        run_id="r-status", total_cases=4, failed_cases=2, regressed_cases=1,
        case_results=[
            CaseResult(case_id="pass", input="", actual_output="", scores={"m": .9}, latency_ms=1, passed_min_score=True),
            CaseResult(case_id="fail", input="", actual_output="", scores={"m": .2}, latency_ms=1, passed_min_score=False),
            CaseResult(case_id="regressed", input="", actual_output="", scores={"m": .8}, latency_ms=1, passed_min_score=True, regressed=True),
            CaseResult(case_id="error", input="", actual_output="", scores={}, latency_ms=1, passed_min_score=False, error="boom"),
        ],
    )
    print_report(report)
    output = recorded.export_text()
    for status in ("PASS", "FAIL", "REGRESSED", "ERROR"):
        assert status in output


def test_html_report_shows_error_status(tmp_path):
    report = RunReport(
        run_id="r-err",
        case_results=[
            CaseResult(
                case_id="boom",
                input="x",
                actual_output="",
                scores={},
                latency_ms=1.0,
                passed_min_score=False,
                error="target raised: boom",
            )
        ],
        total_cases=1,
        failed_cases=1,
    )
    path = tmp_path / "err.html"
    write_html_report(report, path)
    html = path.read_text(encoding="utf-8")
    assert ">ERROR<" in html
    assert "target raised: boom" in html


def test_sparkline_edge_cases():
    assert _sparkline([]) == ""
    assert len(_sparkline([0.5])) == 1
    assert _sparkline([1.0, 1.0, 1.0]) == "▁▁▁"

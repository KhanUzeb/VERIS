import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.flakiness import analyze_repeats, summarize_flaky_cases
from veris.report import write_csv_report, write_html_report, write_json_report
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

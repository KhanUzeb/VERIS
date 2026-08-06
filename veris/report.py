from __future__ import annotations

import csv
import html as html_lib
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .schemas import RunReport

console = Console()

_OK = "#39ff14"     # terminal green
_WARN = "#ffcc00"
_FAIL = "#ff3b3b"
_DIM = "#5c5c5c"


def print_report(report: RunReport) -> None:
    header = Text(f"Veris run {report.run_id}", style=f"bold {_OK if report.passed else _FAIL}")
    console.print(header)
    console.print(Text(f"git sha: {report.git_sha or 'n/a'}   duration: {report.duration_s:.1f}s", style=_DIM))
    console.print()

    table = Table(show_lines=False, border_style=_DIM)
    table.add_column("case", style="white")
    table.add_column("status", justify="center")
    table.add_column("scores", style=_DIM)
    table.add_column("latency", justify="right", style=_DIM)

    for cr in report.case_results:
        if cr.error:
            status = Text("ERROR", style=f"bold {_FAIL}")
        elif not cr.passed_min_score:
            status = Text("FAIL", style=f"bold {_FAIL}")
        elif cr.regressed:
            status = Text("REGRESSED", style=f"bold {_WARN}")
        else:
            status = Text("PASS", style=_OK)

        score_str = cr.error or ", ".join(f"{k}={v:.2f}" for k, v in cr.scores.items())
        table.add_row(cr.case_id, status, score_str, f"{cr.latency_ms:.0f}ms")

    console.print(table)
    console.print()

    summary_style = _OK if report.passed else _FAIL
    console.print(Text(
        f"composite: {report.composite_score:.3f}   "
        f"{report.total_cases - report.failed_cases}/{report.total_cases} passed   "
        f"{report.regressed_cases} regressed",
        style=f"bold {summary_style}",
    ))

    if report.flakiness_summary:
        console.print()
        console.print(Text(report.flakiness_summary, style=_WARN))

    if not report.passed:
        console.print(Text("\nBLOCKING REGRESSION — deploy gate failed.", style=f"bold {_FAIL}"))
        for cr in report.case_results:
            if not cr.passed_min_score and cr.regression_detail:
                console.print(Text(f"  {cr.case_id}: {cr.regression_detail}", style=_FAIL))


def write_json_report(report: RunReport, path: str | Path) -> None:
    path = Path(path)
    path.write_text(report.model_dump_json(indent=2))


def write_csv_report(report: RunReport, path: str | Path) -> None:
    """Flat CSV: one row per (case, metric) for spreadsheet / BI tools."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id", "case_id", "status", "metric", "score",
                "latency_ms", "regression_detail", "error",
            ],
        )
        writer.writeheader()
        for cr in report.case_results:
            if cr.error:
                status = "ERROR"
            elif not cr.passed_min_score:
                status = "FAIL"
            elif cr.regressed:
                status = "REGRESSED"
            else:
                status = "PASS"
            if not cr.scores:
                writer.writerow({
                    "run_id": report.run_id,
                    "case_id": cr.case_id,
                    "status": status,
                    "metric": "",
                    "score": "",
                    "latency_ms": f"{cr.latency_ms:.1f}",
                    "regression_detail": cr.regression_detail or "",
                    "error": cr.error or "",
                })
                continue
            for metric, score in cr.scores.items():
                writer.writerow({
                    "run_id": report.run_id,
                    "case_id": cr.case_id,
                    "status": status,
                    "metric": metric,
                    "score": f"{score:.4f}",
                    "latency_ms": f"{cr.latency_ms:.1f}",
                    "regression_detail": cr.regression_detail or "",
                    "error": cr.error or "",
                })


def _sparkline(values: list[float]) -> str:
    """Tiny unicode sparkline for the composite-score trend, oldest to
    newest. Cheap way to show trend direction without pulling in a
    charting dependency for what's meant to stay a lightweight report."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(blocks[min(int((v - lo) / span * (len(blocks) - 1)), len(blocks) - 1)] for v in values)


def write_html_report(report: RunReport, path: str | Path, trend_history: list[dict] | None = None) -> None:
    path = Path(path)
    esc = html_lib.escape
    rows = "\n".join(
        f"<tr class='{'fail' if not cr.passed_min_score else ('regressed' if cr.regressed else 'pass')}'>"
        f"<td>{esc(cr.case_id)}</td>"
        f"<td>{'FAIL' if not cr.passed_min_score else ('REGRESSED' if cr.regressed else 'PASS')}</td>"
        f"<td>{esc(json.dumps(cr.scores))}</td><td>{cr.latency_ms:.0f}ms</td>"
        f"<td>{esc(cr.regression_detail or '')}</td></tr>"
        for cr in report.case_results
    )

    trend_block = ""
    if trend_history:
        ordered = list(reversed(trend_history))  # oldest -> newest
        spark = _sparkline([r["composite_score"] for r in ordered])
        trend_block = (
            f"<p>composite trend (last {len(ordered)} runs, oldest&rarr;newest): "
            f"<span style='font-size:1.4rem'>{esc(spark)}</span></p>"
        )

    flaky_block = ""
    if report.flakiness_summary:
        flaky_block = f"<pre style='color:#ffcc00'>{esc(report.flakiness_summary)}</pre>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Veris :: {esc(report.run_id)}</title>
<style>
  body {{ background:#0a0a0a; color:#c9c9c9; font-family: 'JetBrains Mono', 'Courier New', monospace; padding:2rem; }}
  h1 {{ color:#39ff14; }}
  table {{ width:100%; border-collapse: collapse; margin-top:1rem; }}
  td,th {{ border:1px solid #2a2a2a; padding:0.5rem 0.8rem; text-align:left; font-size:0.9rem; }}
  tr.pass td:nth-child(2) {{ color:#39ff14; }}
  tr.fail td:nth-child(2) {{ color:#ff3b3b; font-weight:bold; }}
  tr.regressed td:nth-child(2) {{ color:#ffcc00; font-weight:bold; }}
  .summary {{ color: {'#39ff14' if report.passed else '#ff3b3b'}; font-size:1.1rem; margin-top:1rem; }}
</style></head>
<body>
  <h1>&gt; Veris :: run {esc(report.run_id)}</h1>
  <p>git sha: {esc(report.git_sha or 'n/a')} | duration: {report.duration_s:.1f}s</p>
  {trend_block}
  {flaky_block}
  <table><tr><th>case</th><th>status</th><th>scores</th><th>latency</th><th>detail</th></tr>{rows}</table>
  <p class="summary">composite: {report.composite_score:.3f} — {report.total_cases - report.failed_cases}/{report.total_cases} passed — {report.regressed_cases} regressed</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")

"""
LLM-as-judge scores are not deterministic even at temperature=0 — model
serving nondeterminism, prompt-order effects, provider-side sampling
quirks. A single low score on a single run can be a real regression or
it can be noise. This module runs a case's scoring N times and reports
the spread, so the gate can distinguish "consistently bad" from "noisy."

This is deliberately NOT wired into the blocking path by default —
running every case 3-5x multiplies judge-model cost and latency. Use it
selectively: `--repeat 5` on cases you don't trust, or as a one-off audit
before tightening a threshold in config.py.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class FlakinessReport:
    case_id: str
    metric_name: str
    scores: list[float]
    mean: float
    stdev: float
    coefficient_of_variation: float  # stdev/mean — comparable across metrics with different scales
    is_flaky: bool


FLAKY_CV_THRESHOLD = 0.15  # >15% relative spread flags a metric as noisy for this case


def analyze_repeats(case_id: str, metric_scores: dict[str, list[float]]) -> list[FlakinessReport]:
    """metric_scores: metric_name -> list of scores from repeated runs of the
    same case against the same actual_output (i.e. only the judge call was
    repeated, not the target system)."""
    reports = []
    for metric_name, scores in metric_scores.items():
        if len(scores) < 2:
            continue
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores)
        cv = (stdev / mean) if mean > 0 else (0.0 if stdev == 0 else float("inf"))
        reports.append(
            FlakinessReport(
                case_id=case_id,
                metric_name=metric_name,
                scores=scores,
                mean=mean,
                stdev=stdev,
                coefficient_of_variation=cv,
                is_flaky=cv > FLAKY_CV_THRESHOLD,
            )
        )
    return reports


def summarize_flaky_cases(all_reports: list[FlakinessReport]) -> str:
    flaky = [r for r in all_reports if r.is_flaky]
    if not flaky:
        return "no flaky metrics detected"
    lines = [f"{len(flaky)} flaky metric(s) detected — treat single-run failures on these with caution:"]
    for r in flaky:
        lines.append(
            f"  {r.case_id}/{r.metric_name}: mean={r.mean:.3f} stdev={r.stdev:.3f} "
            f"cv={r.coefficient_of_variation:.1%} scores={[round(s, 2) for s in r.scores]}"
        )
    return "\n".join(lines)

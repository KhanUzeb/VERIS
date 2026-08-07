"""
The actual gate. Two independent checks per (case, metric):

1. Absolute floor — score below MetricSpec.min_score always fails,
   regardless of history. Protects against a baseline that has quietly
   drifted bad over time.

2. Regression vs rolling baseline — score more than
   `regression_stdev_band` standard deviations below the case's own
   historical median (from green runs only) fails, even if it's still
   above the absolute floor. This is what catches "still technically
   passing but got noticeably worse."

A metric only blocks the run if MetricSpec.blocking=True. Non-blocking
metrics are still recorded and reported so trends are visible.

Metrics marked inapplicable for a case (e.g. hallucination without
retrieval_context) are skipped — absence is noted as N/A, not as a
blocking failure.
"""
from __future__ import annotations

from .config import SETTINGS, MetricSpec
from .schemas import CaseResult, GoldenCase
from .trend_store import TrendStore

_METRIC_LOOKUP: dict[str, MetricSpec] = {m.name: m for m in SETTINGS.metrics}


def evaluate_case(
    case: GoldenCase,
    actual_output: str,
    scores: dict[str, float],
    latency_ms: float,
    store: TrendStore,
) -> CaseResult:
    passed_min_score = True
    regressed = False
    regression_notes: list[str] = []

    # Gate hole: empty scores (both metric suites failed) must not green-pass.
    if not scores:
        return CaseResult(
            case_id=case.id,
            input=case.input,
            actual_output=actual_output,
            scores=scores,
            latency_ms=latency_ms,
            passed_min_score=False,
            regressed=False,
            regression_detail="no scores produced",
        )

    # Missing applicable blocking metrics (partial suite failure) fail the gate.
    # Inapplicable metrics are noted as N/A and do not flip passed_min_score.
    for spec in SETTINGS.metrics:
        if spec.name in scores:
            continue
        if not spec.is_applicable(case):
            regression_notes.append(f"{spec.name} N/A for this case")
            continue
        note = f"{spec.name} missing from scores"
        if spec.blocking:
            passed_min_score = False
            regression_notes.append(f"{note} [BLOCKING]")
        else:
            regression_notes.append(f"{note} [advisory]")

    for metric_name, score in scores.items():
        spec = _METRIC_LOOKUP.get(metric_name)
        if spec is None:
            continue  # advisory metric with no config entry — report only

        # Check 1: absolute floor
        if score < spec.min_score:
            if spec.blocking:
                passed_min_score = False
            regression_notes.append(
                f"{metric_name}={score:.3f} below floor {spec.min_score:.3f}"
                + (" [BLOCKING]" if spec.blocking else " [advisory]")
            )

        # Check 2: regression vs rolling baseline
        baseline = store.baseline_for_case_metric(
            case.id, metric_name, SETTINGS.baseline_window
        )
        if baseline and baseline[2] >= SETTINGS.min_baseline_samples:
            median, stdev, n = baseline
            band = (
                spec.stdev_band_override
                if spec.stdev_band_override is not None
                else SETTINGS.regression_stdev_band
            )
            # Flat baselines (stdev == 0) use a synthetic relative stdev so
            # band / stdev_band_override still scale the floor. At the default
            # band this preserves the historical 0.9 * median floor.
            effective_stdev = (
                stdev if stdev > 0
                else median * 0.1 / SETTINGS.regression_stdev_band
            )
            floor = median - band * effective_stdev
            if score < floor:
                regressed = True
                if spec.blocking:
                    passed_min_score = False
                regression_notes.append(
                    f"{metric_name}={score:.3f} regressed vs median {median:.3f} "
                    f"(±{stdev:.3f}, n={n})"
                    + (" [BLOCKING]" if spec.blocking else " [advisory]")
                )

    return CaseResult(
        case_id=case.id,
        input=case.input,
        actual_output=actual_output,
        scores=scores,
        latency_ms=latency_ms,
        passed_min_score=passed_min_score,
        regressed=regressed,
        regression_detail="; ".join(regression_notes) if regression_notes else None,
    )


def composite_score(case_results: list[CaseResult]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for cr in case_results:
        for metric_name, score in cr.scores.items():
            spec = _METRIC_LOOKUP.get(metric_name)
            weight = spec.weight if spec else 0.5
            weighted_sum += score * weight
            total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0

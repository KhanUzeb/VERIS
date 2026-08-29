from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cache import ScoreCache, cache_key
from .config import SETTINGS
from .flakiness import FlakinessReport, analyze_repeats, summarize_flaky_cases
from .golden_dataset import load_golden_cases
from .judge_model import GroqEvalModel, reset_usage_totals, usage_totals
from .regression_gate import composite_score, evaluate_case
from .schemas import CaseResult, RunReport
from .trend_store import TrendStore

TargetFn = Callable[[str], str]  # takes case.input, returns actual_output


class RunAborted(RuntimeError):
    """Raised after a circuit-breaker / fail-fast / cost-ceiling abort.

    The partial `report` is attached so callers (CLI, seed script) can still
    print and persist artifacts that `run_gate` already wrote to SQLite.
    """

    def __init__(self, message: str, report: RunReport):
        super().__init__(message)
        self.report = report


def score_deepeval(case, actual_output: str) -> dict[str, float]:
    """Lazy wrapper — keeps deepeval off the import path for CLI/unit tests.
    Monkeypatchable as `runner.score_deepeval`."""
    from .metrics.deepeval_metrics import score_deepeval as _impl
    return _impl(case, actual_output)


class _CircuitBreaker:
    """Trips after N consecutive judge-model failures. This distinguishes
    "the target's output is genuinely bad" (which should fail specific
    cases) from "the judge API key is dead / the endpoint is down" (which
    should abort loudly rather than scoring 100 cases as 0 and reporting
    a false mass regression).

    Once tripped, a cooperative cancel flag is set so in-flight workers
    skip remaining judge work instead of burning tokens."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive_failures = 0
        self._lock = threading.RLock()
        self.tripped = False
        self._cancel = threading.Event()
        self._reason = "circuit breaker tripped"

    def trip(self, reason: str) -> None:
        with self._lock:
            self.tripped = True
            self._reason = reason
            self._cancel.set()

    def record(self, success: bool) -> None:
        with self._lock:
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.threshold:
                    self.trip(
                        f"circuit breaker tripped: {self.threshold} consecutive judge-model failures. "
                        "This almost always means a bad API key, a dead endpoint, or a provider outage "
                        "— not per-case flakiness. Aborting the run rather than scoring the remaining "
                        "cases as failures."
                    )

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self.tripped:
            raise RuntimeError(self._reason)


def _score_case(
    case, actual_output: str, cache: ScoreCache | None, judge_name: str, breaker: _CircuitBreaker,
) -> dict[str, float]:
    """Runs the DeepEval suite, checking the cache first. Returns the
    score dict. Cache is keyed on everything that could change
    the score, including the metrics fingerprint from config.py."""
    breaker.check()
    if breaker.cancelled():
        raise RuntimeError("circuit breaker tripped")

    key = cache_key(case, actual_output, judge_name) if cache else None
    if cache and key:
        cached = cache.get(key)
        applicable = {spec.name for spec in SETTINGS.metrics if spec.is_applicable(case)}
        # Never reuse a partial suite result. A transient failure in one
        # judge suite must be retried next run, not fossilized as a permanent
        # "missing metric" gate failure in the score cache.
        if cached is not None and applicable.issubset(cached):
            return cached

    scores: dict[str, float] = {}
    try:
        scores.update(score_deepeval(case, actual_output))
    except Exception as e:
        print(f"[warn] deepeval scoring failed for {case.id}: {e}")

    # Empty scores (all metrics failed, or the suite raised) count as a
    # judge failure so the circuit breaker can abort a dead provider.
    any_success = bool(scores)
    if breaker.cancelled():
        breaker.record(success=any_success)
        raise RuntimeError("circuit breaker tripped")

    breaker.record(success=any_success)

    applicable = {spec.name for spec in SETTINGS.metrics if spec.is_applicable(case)}
    if cache and key and scores and applicable.issubset(scores):
        cache.set(key, scores)
    return scores


def _run_one_case(
    case, target_fn: TargetFn, store: TrendStore, run_id: str,
    cache: ScoreCache | None, judge_name: str, repeat: int, breaker: _CircuitBreaker,
    flaky_bucket: list[FlakinessReport] | None = None,
) -> CaseResult:
    if breaker.cancelled():
        raise RuntimeError("circuit breaker tripped")

    start = time.perf_counter()
    try:
        actual_output = target_fn(case.input)
    except Exception as e:
        return CaseResult(
            case_id=case.id, input=case.input, actual_output="",
            scores={}, latency_ms=0.0, passed_min_score=False,
            error=f"target raised: {e}",
        )
    latency_ms = (time.perf_counter() - start) * 1000
    # Contract is Callable[[str], str]; non-strings blow up judge/cache in
    # opaque ways, so fail the case with a clear error instead.
    if not isinstance(actual_output, str):
        return CaseResult(
            case_id=case.id, input=case.input, actual_output=repr(actual_output),
            scores={}, latency_ms=latency_ms, passed_min_score=False,
            error=(
                f"target returned {type(actual_output).__name__}, expected str"
            ),
        )

    if repeat <= 1:
        scores = _score_case(case, actual_output, cache, judge_name, breaker)
        flaky_note = None
    else:
        # Flakiness mode: score N times WITHOUT caching (each run must hit
        # the judge fresh), take the mean per metric for the gate decision,
        # and report the spread.
        per_metric: dict[str, list[float]] = {}
        for _ in range(repeat):
            if breaker.cancelled():
                raise RuntimeError("circuit breaker tripped")
            run_scores = _score_case(case, actual_output, cache=None, judge_name=judge_name, breaker=breaker)
            for k, v in run_scores.items():
                per_metric.setdefault(k, []).append(v)
        scores = {k: sum(v) / len(v) for k, v in per_metric.items()}
        flaky_reports = analyze_repeats(case.id, per_metric)
        if flaky_bucket is not None:
            flaky_bucket.extend(flaky_reports)
        flaky_names = [r.metric_name for r in flaky_reports if r.is_flaky]
        flaky_note = (
            f"flaky metrics (cv>{0.15:.0%}): {', '.join(flaky_names)}"
            if flaky_names else None
        )
        if flaky_names:
            print(f"[flaky] {case.id}: noisy on {flaky_names} (repeat={repeat})")

    result = evaluate_case(case, actual_output, scores, latency_ms, store)
    if flaky_note:
        detail = result.regression_detail or ""
        result.regression_detail = f"{detail}; {flaky_note}".lstrip("; ").strip()
    # Incremental write: if the process dies mid-run, completed cases aren't lost.
    for metric_name, score in scores.items():
        store.save_case_score(run_id, case.id, metric_name, score, result.passed_min_score)
    return result


def run_gate(
    target_fn: TargetFn,
    tags: list[str] | None = None,
    git_sha: str | None = None,
    dataset_dir=None,
    use_cache: bool = True,
    repeat: int = 1,
    fail_fast: bool = False,
) -> RunReport:
    dataset_dir = dataset_dir or SETTINGS.golden_dataset_dir
    cases = load_golden_cases(dataset_dir, tags=tags)
    if not cases:
        raise ValueError(f"no golden cases found in {dataset_dir} (tags={tags})")

    store = TrendStore()
    cache = ScoreCache() if use_cache and repeat <= 1 else None
    judge_name = GroqEvalModel().get_model_name()
    run_id = str(uuid.uuid4())[:8]
    report = RunReport(run_id=run_id, git_sha=git_sha, total_cases=len(cases))
    breaker = _CircuitBreaker(SETTINGS.max_consecutive_judge_failures)
    reset_usage_totals()
    flaky_bucket: list[FlakinessReport] = []

    # Parent runs row first so incremental case_scores join into baseline
    # even if the process dies before the final save_run.
    store.begin_run(run_id, git_sha, len(cases))

    abort_error: RuntimeError | None = None
    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=SETTINGS.max_workers) as pool:
            futures = {
                pool.submit(
                    _run_one_case, c, target_fn, store, run_id, cache, judge_name,
                    repeat, breaker, flaky_bucket if repeat > 1 else None,
                ): c
                for c in cases
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    report.case_results.append(result)
                    if fail_fast and not result.passed_min_score:
                        breaker.trip("fail-fast: blocking case failure")
                        pool.shutdown(wait=False, cancel_futures=True)
                        abort_error = RuntimeError("fail-fast: blocking case failure")
                        break
                    if (
                        SETTINGS.max_run_cost_usd is not None
                        and (cost := usage_totals()["estimated_cost_usd"]) > SETTINGS.max_run_cost_usd
                    ):
                        breaker.trip(
                            f"run cost ceiling exceeded: ${cost:.4f} > "
                            f"${SETTINGS.max_run_cost_usd:.4f}"
                        )
                        pool.shutdown(wait=False, cancel_futures=True)
                        abort_error = RuntimeError(breaker._reason)
                        break
                except RuntimeError as e:
                    # Circuit breaker tripped inside a worker. cancel_futures
                    # drains the queue; the cancel flag stops further judge
                    # work in still-running workers.
                    breaker.trip(str(e))
                    pool.shutdown(wait=False, cancel_futures=True)
                    abort_error = RuntimeError(str(e))
                    break
    finally:
        report.duration_s = time.perf_counter() - start
        report.failed_cases = sum(1 for r in report.case_results if not r.passed_min_score)
        report.regressed_cases = sum(1 for r in report.case_results if r.regressed)
        report.composite_score = composite_score(report.case_results)
        if flaky_bucket:
            report.flakiness_summary = summarize_flaky_cases(flaky_bucket)
        if abort_error is not None:
            report.passed = False
        else:
            report.passed = report.failed_cases == 0 and len(report.case_results) == report.total_cases

        # Always persist + close, including on circuit-breaker abort, so
        # partial results and usage aren't lost and connections don't leak.
        try:
            store.save_run(report, write_case_scores=False)
            store.save_usage(run_id, usage_totals())
        finally:
            if cache:
                cache.close()
            store.close()

    if abort_error is not None:
        raise RunAborted(str(abort_error), report)
    return report

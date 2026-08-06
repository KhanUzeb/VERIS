# Architecture

Veris is a **local regression gate** (CI optional): golden cases → target execution
→ LLM-as-judge scoring → baseline comparison → pass/fail exit code.

## Pipeline

```
┌─────────────────┐
│  Golden YAML    │  id, input, expected_output?, retrieval_context?, tags
└────────┬────────┘
         │ load + tag filter + unique-id check
         ▼
┌─────────────────┐
│  TargetFn       │  Callable[[str], str] — your agent / RAG / API
└────────┬────────┘
         │ ThreadPoolExecutor (max_workers)
         ▼
┌─────────────────┐
│  Score cache    │  SHA256(case + output + judge + metrics fingerprint)
└────────┬────────┘
         │ miss → DeepEval + RAGAS (shared judge, concurrency semaphore)
         ▼
┌─────────────────┐
│  Regression     │  absolute min_score + rolling median ± stdev band
│  gate           │  blocking vs advisory; N/A metrics skipped
└────────┬────────┘
         │ incremental SQLite writes (crash-safe)
         ▼
┌─────────────────┐
│  Reports        │  Rich / JSON / HTML / CSV + optional Langfuse
└─────────────────┘
```

## Package layout

| Module | Role |
|--------|------|
| `veris.cli` | Click CLI (`veris` / `python -m veris.cli`) |
| `veris.runner` | Orchestration, circuit breaker, flakiness repeats |
| `veris.regression_gate` | Pass/fail + composite score |
| `veris.trend_store` | SQLite baselines, history, usage, prune |
| `veris.cache` | Content-addressable score cache |
| `veris.judge_model` | Groq / OpenRouter judge + token accounting |
| `veris.metrics.*` | DeepEval + RAGAS suites |
| `veris.report` | Terminal / JSON / HTML / CSV |
| `veris.langfuse_tracker` | Optional observability (SDK v3/v4 + legacy) |
| `veris.golden_dataset` | YAML loader |
| `veris.config` | All thresholds (`MetricSpec`, `Settings`) |
| `veris.flakiness` | `--repeat N` spread analysis |
| `veris.targets` | Stub target for local smoke / seed |

## Gate semantics

Two independent checks per applicable `(case, metric)`:

1. **Absolute floor** — `score < MetricSpec.min_score` always fails (blocking metrics only flip the exit code).
2. **Regression** — if enough green history exists, fail when score falls more than `band * stdev` below the case’s rolling median (flat baselines use `0.9 * median`).

Metrics with `requires_retrieval_context` / `requires_expected_output` that the case cannot satisfy are marked **N/A** and never treated as missing blocking scores.

## Resilience

- **Circuit breaker** — consecutive judge failures abort the run (bad key / outage) instead of mass false regressions; cooperative cancel stops further judge work.
- **Score cache** — skips judge calls when inputs + metric config fingerprint are unchanged.
- **Incremental DB writes** — completed cases survive mid-run crashes.
- **Provider fallback** — primary `EVAL_JUDGE_PROVIDER` with the other provider as fallback when its API key is set.

## Persistence

- Trend DB: `TREND_DB_PATH` (default `./veris_trends.db`)
- Cache DB: sibling `*_cache.db`
- Reports: `veris_report.json` / `.html` (+ optional `.csv`)
